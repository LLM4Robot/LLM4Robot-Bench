#!/usr/bin/env python3
"""Run on the user's WSL computer. Never copies Codex credentials to workers."""
import argparse
from dataclasses import asdict, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
from pathlib import Path
import queue
import sys
import threading

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from robot_tasks.policies.codex_cli import CodexCLIClient
from robot_tasks.policies.model_config import GPTSettings, load_gpt_settings
from robot_tasks.policies.responses import ModelError, RetryableModelError


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address, token, settings, factory=CodexCLIClient, *, max_concurrent=5):
        if max_concurrent < 1:
            raise ValueError('max_concurrent must be positive')
        super().__init__(address, Handler)
        self.token, self.settings, self.factory = token, settings, factory
        self.slots = threading.BoundedSemaphore(max_concurrent)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # No headers, credentials, images or prompts in access logs.

    def do_POST(self):
        expected = 'Bearer ' + self.server.token
        if not hmac.compare_digest(self.headers.get('Authorization', '').encode(), expected.encode()):
            self.send_error(401); return
        if self.path not in ('/health', '/generate'):
            self.send_error(404); return
        try:
            size = int(self.headers.get('Content-Length', '0'))
            if not 0 <= size <= 32 * 1024 * 1024:
                raise ValueError('Invalid request size')
            self.connection.settimeout(15)
            payload = json.loads(self.rfile.read(size))
            if self.path == '/generate':
                settings = GPTSettings(**payload['settings'])
                # Executable and login always come from the trusted local config.
                if settings.transport != 'codex_cli' or settings.model != self.server.settings.model:
                    raise ValueError('Unsupported transport/model')
                settings = replace(settings, codex_bin=self.server.settings.codex_bin,
                                   timeout_s=min(settings.timeout_s, 9000))
                if not isinstance(payload['instructions'], str) or not isinstance(payload['messages'], list) or not isinstance(payload['schema'], dict):
                    raise ValueError('Invalid generation payload')
            else:
                settings = self.server.settings
        except (ValueError, TypeError, KeyError, OSError):
            self.send_error(400); return
        if not self.server.slots.acquire(blocking=False):
            self.send_error(429); return
        client = None
        done = threading.Event()
        try:
            client = self.server.factory(settings)
            events = queue.Queue()
            def work():
                try:
                    if self.path == '/health':
                        client.require_chatgpt_login()
                        events.put({'type': 'health', 'logged_in': True})
                    else:
                        reply = client.generate(instructions=payload['instructions'], messages=payload['messages'],
                                                schema=payload['schema'], on_reasoning=lambda text: events.put({'type': 'reasoning', 'text': text}))
                        events.put({'type': 'reply', 'reply': asdict(reply)})
                except BaseException as exc:
                    message = str(exc) if isinstance(exc, ModelError) else 'Local Codex call interrupted or failed'
                    events.put({'type': 'error', 'message': message, 'retryable': isinstance(exc, RetryableModelError),
                                'diagnostic': getattr(client, 'last_diagnostic', None)})
                finally:
                    done.set()
                    self.server.slots.release()
            threading.Thread(target=work, daemon=True).start()
            self.send_response(200)
            self.send_header('Content-Type', 'application/x-ndjson')
            self.send_header('Connection', 'close')
            self.end_headers()
            self.close_connection = True
            self.connection.settimeout(10)
            while True:
                try:
                    event = events.get(timeout=1)
                except queue.Empty:
                    event = {'type': 'heartbeat'}
                self.wfile.write(json.dumps(event, allow_nan=False).encode() + b'\n')
                self.wfile.flush()
                if event['type'] in ('reply', 'health', 'error'):
                    break
        except (OSError, ValueError):
            if client:
                client.cancel_pending()
        except BaseException:
            if client:
                client.cancel_pending()
            else:
                self.server.slots.release()
            raise
        finally:
            if client and not done.is_set():
                client.cancel_pending()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=18465)
    p.add_argument('--token-file', type=Path, required=True)
    p.add_argument('--max-concurrent', type=int, default=5)
    a = p.parse_args()
    token = a.token_file.read_text().strip()
    if len(token) < 32:
        raise ValueError('Use a random private token with at least 32 characters')
    settings = load_gpt_settings(ROOT / 'configs/models/gpt.toml')
    CodexCLIClient(settings).require_chatgpt_login()
    server = Server(('127.0.0.1', a.port), token, settings, max_concurrent=a.max_concurrent)
    print(f'CODEX_BRIDGE_READY 127.0.0.1:{a.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

if __name__ == '__main__':
    main()
