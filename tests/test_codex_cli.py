"""Robot/Codex boundary fixtures; no account, network, model or Isaac required."""

import base64
from dataclasses import replace
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from robot_tasks import batch
from robot_tasks.execution import _gpt_policy
from robot_tasks.policies.codex_cli import (CodexCLIClient, codex_environment, parse_events,
                                          serialize_observations, failure_diagnostic, codex_failure)
from robot_tasks.policies.model_config import GPTSettings, load_gpt_settings
from robot_tasks.policies.responses import ModelError, RetryableModelError
from robot_tasks.registry import resolve_task


SETTINGS = GPTSettings(base_url="https://api.openai.com/v1", model="gpt-6-astra", transport="codex_cli",
                       reasoning_effort="medium", timeout_s=5)


def events(*items, complete=True, text='{"commands": []}'):
    rows = [{"type": "thread.started", "thread_id": "fixture-thread"}, {"type": "turn.started"}, *items]
    if text is not None:
        rows.append({"type": "item.completed", "item": {"type": "agent_message", "text": text}})
    if complete:
        rows.append({"type": "turn.completed", "usage": {"input_tokens": 12, "cached_input_tokens": 4,
                                                         "output_tokens": 3, "reasoning_output_tokens": 2}})
    return "\n".join(map(json.dumps, rows))


class CodexBoundaryTests(unittest.TestCase):
    def setUp(self):
        settings = patch("robot_tasks.policies.codex_process._local_settings", return_value={})
        settings.start()
        self.addCleanup(settings.stop)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def png_url(self):
        output = BytesIO()
        Image.new("RGB", (4, 4), (255, 0, 0)).save(output, format="PNG")
        return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()

    def test_images_stay_ordered_and_labelled_across_history(self):
        messages = [{"role": "user", "content": [
            {"type": "input_text", "text": "before: RGB left wrist"},
            {"type": "input_image", "image_url": self.png_url()}]},
            {"role": "assistant", "content": '{"commands":[]}'},
            {"role": "user", "content": [
                {"type": "input_text", "text": "current: depth overview"},
                {"type": "input_image", "image_url": self.png_url()}]}]
        prompt, images = serialize_observations(messages, self.root)
        self.assertEqual(len(images), 2)
        self.assertLess(prompt.index("before:"), prompt.index("Attached image #1"))
        self.assertLess(prompt.index("current:"), prompt.index("Attached image #2"))
        self.assertIn("Record 3, role=user", prompt)
        with Image.open(images[1]) as image:
            self.assertEqual(image.size, (4, 4))

    def test_remote_images_and_malformed_inline_data_are_rejected(self):
        for url in ("https://untrusted/image.png", "data:image/png;base64,@@@",
                    "data:image/png;base64," + base64.b64encode(b"not PNG").decode()):
            with self.subTest(url=url), self.assertRaises(ModelError):
                serialize_observations([{"role": "user", "content": [{"type": "input_image", "image_url": url}]}], self.root)

    def test_only_completed_json_decisions_are_accepted_and_usage_preserved(self):
        result = parse_events(events(), model=SETTINGS.model, latency_s=1.5)
        self.assertEqual(result.response_id, "fixture-thread")
        self.assertEqual(result.usage, {"input_tokens": 12, "output_tokens": 3, "cached_tokens": 4,
                                       "reasoning_tokens": 2, "total_tokens": 15})
        for stream in (events(complete=False), events(text="```json\n{}\n```"), events(text="[]"), "malformed"):
            with self.subTest(stream=stream), self.assertRaises(ModelError):
                parse_events(stream, model=SETTINGS.model, latency_s=1)

    def test_completed_turn_with_tool_activity_is_rejected(self):
        for kind in ("command_execution", "mcp_tool_call", "file_change", "web_search"):
            with self.subTest(kind=kind), self.assertRaisesRegex(ModelError, "tool"):
                parse_events(events({"type": "item.started", "item": {"type": kind}}), model=SETTINGS.model, latency_s=1)

    def test_failure_redacts_bearer_api_key_and_jwt(self):
        stream = events({"type": "turn.failed", "error": {
            "message": "Bearer secret sk-private eyJtest.payload.signature quota exceeded"}}, complete=False, text=None)
        with self.assertRaises(ModelError) as caught:
            parse_events(stream, model=SETTINGS.model, latency_s=1)
        self.assertIn("quota exceeded", str(caught.exception))
        for private in ("secret", "sk-private", "eyJtest"):
            self.assertNotIn(private, str(caught.exception))

    def test_error_event_and_stderr_survive_unknown_cli_exit_without_private_output(self):
        stream = events({"type": "error", "message": "stream disconnected"}, complete=False,
                        text="private model output")
        diagnostic = failure_diagnostic(stream, "TLS failure Bearer secret sk-private eyJabc.xyz.sig", 1)
        failure = codex_failure(diagnostic)
        self.assertIsInstance(failure, RetryableModelError)
        self.assertIn("stream disconnected", str(failure))
        self.assertIn("TLS failure", str(failure))
        for secret in ("private model output", "secret", "sk-private", "eyJabc"):
            self.assertNotIn(secret, json.dumps(diagnostic))

    def test_quota_login_and_context_errors_are_not_retried(self):
        for message in ("You've hit your usage limit", "quota exceeded", "not logged in",
                        "input exceeds the context window", "authentication failed", "input_too_large",
                        "Input exceeds the maximum length of 1048576 characters"):
            with self.subTest(message=message):
                failure = codex_failure(failure_diagnostic("", message, 1))
                self.assertIsInstance(failure, ModelError)
                self.assertNotIsInstance(failure, RetryableModelError)

    def test_recovered_error_event_does_not_discard_completed_decision(self):
        self.assertEqual(parse_events(events({"type": "error", "message": "reconnecting"}),
                                      model="fixture", latency_s=0).text, '{"commands": []}')

    def test_compaction_preserves_history_and_distinct_task_instructions(self):
        messages = [{"role": "user", "content": [{"type": "input_text", "text": json.dumps({
            "instruction": instruction, "step": step, "robot": {"position": [1., 2., 3.]}})}]}
                    for step, instruction in enumerate(("same long task", "same long task", "new task"), 1)]
        before = json.dumps(messages)
        prompt, images = serialize_observations(messages, self.root)
        self.assertEqual(prompt.count("same long task"), 1)
        self.assertEqual(prompt.count('"task_instruction_ref":"task_1"'), 2)
        self.assertIn('"task_instruction_ref":"task_2"', prompt)
        for step in range(1, 4):
            self.assertIn(f'"step":{step}', prompt)
        self.assertEqual(json.dumps(messages), before)
        self.assertFalse(images)

    def test_oversized_history_is_rejected_before_starting_codex(self):
        with patch("robot_tasks.policies.codex_cli.CodexProcess") as host:
            host.return_value.path.side_effect = lambda path: path
            client = CodexCLIClient(SETTINGS)
            with self.assertRaisesRegex(ModelError, "input_too_large"):
                client.generate(instructions="contract", messages=[{"role": "user", "content": "x" * 1_048_576}], schema={})
            host.return_value.run.assert_not_called()

    def test_environment_drops_api_keys_and_application_secrets(self):
        result = codex_environment({"HOME": "/identity", "PATH": "/bin", "CODEX_HOME": "/auth",
                                    "HTTPS_PROXY": "http://localhost:7897", "OPENAI_API_KEY": "private",
                                    "CODEX_API_KEY": "private", "OPENAI_BASE_URL": "https://relay",
                                    "OTHER_SECRET": "private"})
        self.assertEqual(set(result), {"HOME", "PATH", "CODEX_HOME", "HTTPS_PROXY"})

    def test_login_rejects_api_auth_even_if_cli_exits_zero(self):
        with patch("robot_tasks.policies.codex_process.shutil.which", return_value="fixture"):
            client = CodexCLIClient(SETTINGS)
        for message in ("Logged in using an API key", "Not logged in"):
            with patch.object(client.host, "run", return_value=subprocess.CompletedProcess([], 0, message.encode(), b"")):
                with self.assertRaisesRegex(ModelError, "ChatGPT login required"):
                    client.require_chatgpt_login()

    def test_codex_task_and_batch_do_not_require_or_load_api_key(self):
        model = self.root / "model.toml"
        model.write_text('[gpt]\nbase_url="https://api.openai.com/v1"\nmodel="gpt-6-astra"\ntransport="codex_cli"\n')
        _, config = resolve_task("L1-01")
        with patch("robot_tasks.policies.codex_process.shutil.which", return_value="fixture"), \
                patch.object(CodexCLIClient, "require_chatgpt_login") as login, \
                patch("robot_tasks.execution.read_api_key", side_effect=AssertionError("API credentials must not be used")), \
                patch.object(batch, "load_env", side_effect=AssertionError(".env must not be loaded")):
            policy = _gpt_policy(config, model)
            self.assertIsInstance(policy.client, CodexCLIClient)
            environment = batch.worker_environment(self.root / ".env", {"model_config": str(model), "options": {"dry_run": False}})
            self.assertIn("PYTHONPATH", environment)
            self.assertEqual(login.call_count, 2)

    def test_optional_service_tier_reaches_each_codex_decision_separately_from_effort(self):
        for tier in (None, "fast", "ultrafast"):
            with self.subTest(tier=tier), patch("robot_tasks.policies.codex_cli.CodexProcess") as host:
                host.return_value.path.side_effect = lambda path: path
                host.return_value.run.return_value = subprocess.CompletedProcess([], 0, events().encode(), b"")
                client = CodexCLIClient(replace(SETTINGS, service_tier=tier))
                for _ in range(2):
                    client.generate(instructions="Robot contract", messages=[], schema={"type": "object"})
                for call in host.return_value.run.call_args_list:
                    command = call.args[0]
                    overrides = dict(command[i + 1].split("=", 1) for i, arg in enumerate(command) if arg == "-c")
                    self.assertEqual(json.loads(overrides["model_reasoning_effort"]), "medium")
                    if tier is None:
                        self.assertNotIn("service_tier", overrides)
                        self.assertNotIn("features.fast_mode", overrides)
                    else:
                        self.assertEqual(json.loads(overrides["service_tier"]), tier)
                        self.assertEqual(overrides["features.fast_mode"], "true")

    def test_ultrafast_profile_survives_task_policy_overrides(self):
        root = Path(__file__).resolve().parents[1]
        model = self.root / "ultrafast.toml"
        model.write_text((root / "configs/models/gpt.toml").read_text(encoding="utf-8")
                         + '\nservice_tier = "ultrafast"\n', encoding="utf-8")
        _, config = resolve_task("L1-02")
        with patch("robot_tasks.policies.codex_cli.CodexProcess"), \
                patch.object(CodexCLIClient, "require_chatgpt_login"), \
                patch("robot_tasks.execution.read_api_key", side_effect=AssertionError("No API key")):
            policy = _gpt_policy(config, model)
        self.assertEqual(policy.settings.transport, "codex_cli")
        self.assertEqual(policy.settings.service_tier, "ultrafast")
        from robot_tasks.policies.model_config import load_gpt_settings
        self.assertEqual(policy.settings.reasoning_effort, load_gpt_settings(model).reasoning_effort)
        self.assertEqual(policy.settings.max_calls, 150)
        self.assertEqual(policy.run_metadata["provider"]["service_tier"], "ultrafast")

    @unittest.skipIf(os.name == "nt", "POSIX mock executable")
    def test_real_cli_boundary_gets_images_schema_model_and_clean_env(self):
        executable = self.root / "codex-fixture"
        executable.write_text(f"#!{sys.executable}\n" +
            "import json,os,pathlib,sys\n"
            "a=sys.argv[1:]\n"
            "assert a[a.index('--model')+1]=='gpt-6-astra'\n"
            "assert 'model_reasoning_effort=\"medium\"' in a\n"
            "assert 'model_reasoning_summary=\"auto\"' in a\n"
            "assert 'forced_login_method=\"chatgpt\"' in a\n"
            "assert 'OPENAI_API_KEY' not in os.environ\n"
            "assert json.loads(pathlib.Path(a[a.index('--output-schema')+1]).read_text())['type']=='object'\n"
            "assert pathlib.Path(a[a.index('--image')+1]).read_bytes().startswith(b'\\x89PNG')\n"
            "assert 'Attached image #1' in sys.stdin.read()\n"
            f"print({events({'type': 'item.completed', 'item': {'id': 'r1', 'type': 'reasoning', 'text': 'Inspect handle.'}})!r})\n")
        executable.chmod(0o700)
        summaries = []
        with patch.dict(os.environ, {"OPENAI_API_KEY": "must-not-forward"}):
            client = CodexCLIClient(replace(SETTINGS, codex_bin=str(executable)))
            result = client.generate(instructions="Robot contract", schema={"type": "object"}, messages=[{
                "role": "user", "content": [{"type": "input_text", "text": "RGB label"},
                                             {"type": "input_image", "image_url": self.png_url()}]}],
                                     on_reasoning=summaries.append)
        self.assertEqual(result.usage["total_tokens"], 15)
        self.assertEqual(summaries, ["Inspect handle."])

    @unittest.skipIf(os.name == "nt", "POSIX mock executable")
    def test_timeout_discards_partial_reply_and_stops_process(self):
        executable = self.root / "codex-slow"
        pidfile = self.root / "pid"
        executable.write_text(f"#!{sys.executable}\nimport os,time,pathlib\n"
                              f"pathlib.Path({str(pidfile)!r}).write_text(str(os.getpid()))\n"
                              f"print({events(complete=False)!r},flush=True)\ntime.sleep(30)\n")
        executable.chmod(0o700)
        # Allow the fixture interpreter to start on a busy host. The test checks
        # termination of a running child, not Python's cold-start latency.
        client = CodexCLIClient(replace(SETTINGS, codex_bin=str(executable), timeout_s=2))
        with self.assertRaisesRegex(RetryableModelError, "timed out"):
            client.generate(instructions="contract", messages=[], schema={})
        with self.assertRaises(ProcessLookupError):
            os.kill(int(pidfile.read_text()), 0)

    def test_model_config_stays_explicit_and_rejects_unsupported_transport(self):
        with self.assertRaises(ValueError):
            replace(SETTINGS, transport="implicit_fallback")
        with self.assertRaises(ValueError):
            replace(SETTINGS, max_output_tokens=100)


if __name__ == "__main__":
    unittest.main()
