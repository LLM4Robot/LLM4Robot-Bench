"""Authenticated loopback RPC to a local Codex login, forwarded over SSH."""

import http.client
import json
import os
from pathlib import Path
import threading
from urllib.parse import urlsplit

from .responses import ModelError, ModelReply, RetryableModelError


class CodexBridgeClient:
    provider_metadata = {"authentication": "chatgpt_local_ssh_bridge",
                         "model_verification": "explicit_cli_configuration",
                         "response_id_kind": "codex_thread_id", "codex_host": "local_bridge",
                         "input_serialization": "compact_json_shared_task_instruction_v1"}

    def __init__(self, settings):
        self.settings = settings
        self.last_diagnostic = None
        self.cancelled = threading.Event()
        self.connection = None
        self.socket = None
        url = urlsplit(os.environ['GPT_BENCHMARK_CODEX_BRIDGE_URL'])
        if url.scheme != 'http' or url.hostname != '127.0.0.1' or url.path not in ('', '/') or url.query or url.fragment or url.username or url.password:
            raise ValueError('Codex bridge must use an SSH-forwarded http://127.0.0.1:port URL')
        self.port = url.port or 18465
        self.token = Path(os.environ['GPT_BENCHMARK_CODEX_BRIDGE_TOKEN_FILE']).read_text(encoding='utf-8').strip()
        if len(self.token) < 32 or any(c.isspace() for c in self.token):
            raise ValueError('Invalid bridge token file')

    def cancel_pending(self):
        self.cancelled.set()
        connection, pending_socket = self.connection, self.socket
        if connection is not None:
            if pending_socket:
                import socket
                try:
                    pending_socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            connection.close()

    def reset_cancellation(self):
        self.cancelled.clear()

    def _request(self, path, payload):
        if self.cancelled.is_set():
            raise KeyboardInterrupt('Bridge request cancelled')
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=self.settings.timeout_s + 20)
        self.connection = connection
        try:
            connection.request('POST', path, json.dumps(payload, allow_nan=False).encode(),
                               {'Authorization': 'Bearer ' + self.token, 'Content-Type': 'application/json'})
            self.socket = connection.sock
            response = connection.getresponse()
            if response.status != 200:
                raise ModelError(f'Local Codex bridge returned HTTP {response.status}')
            while True:
                line = response.readline(4 * 1024 * 1024 + 1)
                if not line:
                    raise RetryableModelError('Codex bridge disconnected before a complete result')
                if len(line) > 4 * 1024 * 1024:
                    raise ModelError('Oversized bridge event')
                event = json.loads(line)
                if event['type'] == 'error':
                    self.last_diagnostic = event.get('diagnostic')
                    cls = RetryableModelError if event.get('retryable') else ModelError
                    raise cls(event.get('message', 'Local Codex call failed'))
                yield event
                if event['type'] in ('reply', 'health'):
                    return
        except (OSError, http.client.HTTPException) as exc:
            if self.cancelled.is_set():
                raise KeyboardInterrupt('Bridge request cancelled') from None
            raise RetryableModelError('Cannot reach local Codex bridge; check the SSH tunnel') from exc
        finally:
            connection.close()
            self.connection = None
            self.socket = None

    def require_chatgpt_login(self):
        for event in self._request('/health', {}):
            if event['type'] == 'health' and not event.get('logged_in'):
                raise ModelError('Local Codex is not logged in')

    def generate(self, *, instructions, messages, schema, on_reasoning=None):
        payload = {'settings': self.settings.public_dict(), 'instructions': instructions,
                   'messages': messages, 'schema': schema}
        for event in self._request('/generate', payload):
            if event['type'] == 'reasoning' and on_reasoning:
                on_reasoning(event['text'])
            elif event['type'] == 'reply':
                return ModelReply(**event['reply'])
        raise ModelError('Missing Codex bridge result')


def create_codex_client(settings):
    if os.environ.get('GPT_BENCHMARK_CODEX_BRIDGE_URL'):
        return CodexBridgeClient(settings)
    from .codex_cli import CodexCLIClient
    return CodexCLIClient(settings)
