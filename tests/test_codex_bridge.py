"""Loopback transport checks use a fake model, no paid calls or credentials."""
import importlib.util
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from fixtures import allow_loopback

from robot_tasks.policies.codex_bridge import CodexBridgeClient
from robot_tasks.policies.model_config import GPTSettings
from robot_tasks.policies.responses import ModelReply, ModelError

spec=importlib.util.spec_from_file_location('bridge_server',Path(__file__).resolve().parents[1]/'skills/r5-remote-deploy/scripts/codex_bridge_server.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class BridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.settings=GPTSettings(base_url='https://api.openai.com/v1',model='test-model',transport='codex_cli',timeout_s=5)
        self.cancelled=threading.Event();self.started=threading.Event();self.calls=[]
        owner=self
        class Fake:
            last_diagnostic=None
            def __init__(self,settings):self.settings=settings
            def require_chatgpt_login(self):pass
            def cancel_pending(self):owner.cancelled.set()
            def generate(self,**kwargs):
                owner.calls.append(kwargs);owner.started.set()
                if kwargs['instructions']=='wait':
                    owner.cancelled.wait(6)
                    raise ModelError('cancelled')
                kwargs['on_reasoning']('public summary')
                return ModelReply('{"ok":true}',self.settings.model,'test-thread',{},.01)
        self.token='a'*48
        self.server=module.Server(('127.0.0.1',0),self.token,self.settings,Fake)
        self.enterContext(allow_loopback(self.server.server_address[1]))
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        def close():self.server.shutdown();self.server.server_close();self.thread.join()
        self.addCleanup(close)
        path=Path(self.temp.name)/'token';path.write_text(self.token)
        self.env={'GPT_BENCHMARK_CODEX_BRIDGE_URL':f'http://127.0.0.1:{self.server.server_address[1]}','GPT_BENCHMARK_CODEX_BRIDGE_TOKEN_FILE':str(path)}
        self.enterContext(patch.dict(os.environ,self.env));self.client=CodexBridgeClient(self.settings)

    def test_streaming_health_and_exact_payload(self):
        self.client.require_chatgpt_login();summaries=[]
        messages=[{'role':'user','content':[{'type':'input_text','text':'evidence'}]}]
        result=self.client.generate(instructions='robot only',messages=messages,schema={'type':'object'},on_reasoning=summaries.append)
        self.assertEqual(result.text,'{"ok":true}');self.assertEqual(summaries,['public summary'])
        self.assertEqual(self.calls[0]['messages'],messages)

    def test_unauthorized_never_calls_model(self):
        self.client.token='b'*48
        with self.assertRaisesRegex(ModelError,'401'):self.client.require_chatgpt_login()
        self.assertEqual(self.calls,[])

    def test_zero_call_cap_is_accepted_by_local_bridge(self):
        from dataclasses import replace
        client = CodexBridgeClient(replace(self.settings, max_calls=0))
        result = client.generate(instructions='robot only', messages=[], schema={})
        self.assertEqual(result.text, '{"ok":true}')

    def test_disconnect_cancels_owned_local_request(self):
        errors=[]
        def request():
            try:self.client.generate(instructions='wait',messages=[],schema={})
            except BaseException as exc:errors.append(exc)
        thread=threading.Thread(target=request);thread.start()
        self.assertTrue(self.started.wait(3));self.client.cancel_pending();thread.join(3)
        self.assertFalse(thread.is_alive());self.assertTrue(self.cancelled.wait(4))
        self.assertTrue(errors)

    def test_reject_non_loopback_endpoint(self):
        with patch.dict(os.environ,{'GPT_BENCHMARK_CODEX_BRIDGE_URL':'http://0.0.0.0:18465'}):
            with self.assertRaises(ValueError):CodexBridgeClient(self.settings)

if __name__=='__main__':unittest.main()
