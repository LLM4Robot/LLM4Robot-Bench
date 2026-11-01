"""Public summary streaming fixtures; no model, account or network calls."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from robot_tasks.policies.codex_process import CodexProcess
from robot_tasks.policies.reasoning import DecisionProgress, ReasoningSummaries
from robot_tasks.policies.responses import ModelError, read_response
from test_gpt_policy import SETTINGS, Stream, event, response_document


class ReasoningProgressTests(unittest.TestCase):
    def test_api_summary_is_visible_before_action_and_final_snapshots_do_not_repeat(self):
        lines = []
        item = {"id": "r1", "type": "reasoning", "summary": [
            {"type": "summary_text", "text": "Inspect the handle.\nPlan a short pull."}]}
        events = [
            {"type": "response.reasoning_summary_text.delta", "item_id": "r1", "summary_index": 0,
             "delta": "Inspect the "},
            {"type": "response.reasoning_summary_text.delta", "item_id": "r1", "summary_index": 0,
             "delta": "handle.\nPlan a short pull."},
            {"type": "response.reasoning_summary_text.done", "item_id": "r1", "summary_index": 0,
             "text": item["summary"][0]["text"]},
            {"type": "response.output_item.done", "item": item},
        ]
        document = response_document()
        document["output"].insert(0, item)

        class GuardedStream(Stream):
            def readline(inner, *args):
                line = super().readline(*args)
                if b'"type": "response.completed"' in line:
                    self.assertEqual(lines, ["Inspect the handle.", "Plan a short pull."])
                return line

        data = b"".join(map(event, events)) + event({"type": "response.completed", "response": document})
        reply = read_response(GuardedStream(data), expected_model=SETTINGS.model,
                              start=time.monotonic(), timeout_s=10, on_reasoning=lines.append)
        self.assertEqual(len(lines), 2)
        self.assertEqual(reply.text, document["output"][1]["content"][0]["text"])

    def test_json_fallback_reads_summary_only_not_raw_or_encrypted_reasoning(self):
        lines = []
        document = response_document()
        document["output"].insert(0, {"type": "reasoning", "id": "r1",
                                      "content": [{"type": "reasoning_text", "text": "private raw"}],
                                      "encrypted_content": "ciphertext",
                                      "summary": [{"type": "summary_text", "text": "Public summary."}]})
        read_response(Stream(json.dumps(document).encode(), "application/json"),
                      expected_model=SETTINGS.model, start=time.monotonic(), timeout_s=10,
                      on_reasoning=lines.append)
        self.assertEqual(lines, ["Public summary."])

    def test_split_secret_redaction_and_failed_generation_never_accept_partial_actions(self):
        lines = []
        data = event({"type": "response.reasoning_text.delta", "delta": "private raw"})
        for text in ("Inspect ", "test-private", "-token Bearer hidden sk-secret eyJfake.payload.\n"):
            data += event({"type": "response.reasoning_summary_text.delta", "item_id": "r1", "delta": text})
        data += event({"type": "response.output_text.delta", "delta": '{"commands": []}'})
        with self.assertRaisesRegex(ModelError, "without a completed"):
            read_response(Stream(data), expected_model=SETTINGS.model, start=time.monotonic(), timeout_s=10,
                          secret="test-private-token", on_reasoning=lines.append)
        self.assertEqual(len(lines), 1)
        for secret in ("private raw", "test-private", "hidden", "sk-secret", "eyJfake", "commands"):
            self.assertNotIn(secret, lines[0])

    def test_codex_reasoning_updates_are_deduplicated_without_printing_action(self):
        lines = []
        summaries = ReasoningSummaries(lines.append)
        for kind, text in (("item.started", ""), ("item.updated", "Inspect.\n"),
                           ("item.updated", "Inspect.\nPull."), ("item.completed", "Inspect.\nPull.")):
            summaries.codex_line(json.dumps({"type": kind, "item": {"id": "r1", "type": "reasoning", "text": text}}))
        summaries.codex_line(json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "action"}}))
        summaries.codex_line(b"malformed")
        self.assertEqual(lines, ["Inspect.", "Pull."])

    def test_request_labels_and_no_summary_are_explicit_even_on_failure(self):
        output, exchange = StringIO(), {}
        with redirect_stdout(output):
            with DecisionProgress(7, exchange) as progress:
                progress.summary("Inspect first.")
            with self.assertRaises(TimeoutError):
                with DecisionProgress(8, {}):
                    raise TimeoutError()
        self.assertEqual(exchange["reasoning_summary"], ["Inspect first."])
        self.assertIn("[request 7] thinking (summary): Inspect first.", output.getvalue())
        self.assertIn("[request 8] request interrupted or failed; no reasoning summary returned", output.getvalue())

    def test_native_process_streams_split_utf8_before_exit_and_preserves_full_output(self):
        # The child refuses to complete until the parent receives its event.
        # This catches buffering until process exit, including on Windows.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ack = root / "ack"
            text = "\u6293\u53d6\u628a\u624b"
            first = json.dumps({"type": "item.completed", "item": {
                "id": "r1", "type": "reasoning", "text": text}}, ensure_ascii=False).encode() + b"\n"
            split = first.index(text.encode()) + 1
            script = ("import os, pathlib, time\n"
                      f"os.write(1, {first[:split]!r})\ntime.sleep(.15)\n"
                      f"os.write(1, {first[split:]!r})\n"
                      f"ack = pathlib.Path({str(ack)!r})\n"
                      "deadline = time.monotonic() + 4\n"
                      "while not ack.exists() and time.monotonic() < deadline: time.sleep(.01)\n"
                      "assert ack.exists(), 'event was buffered until exit'\n"
                      "os.write(1, b'last line without newline')\n")
            received = []

            def receive(line):
                received.append(line)
                if line.startswith(b'{'):
                    self.assertEqual(json.loads(line)["item"]["text"], text)
                    ack.touch()

            with patch("robot_tasks.policies.codex_process._local_settings", return_value={"host": "native"}):
                host = CodexProcess(sys.executable)
                result = host.run(["-c", script], directory=root, timeout_s=6, on_stdout_line=receive)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(received, [first.rstrip(b"\n"), b"last line without newline"])
            self.assertEqual(result.stdout, first + b"last line without newline")


if __name__ == "__main__":
    unittest.main()
