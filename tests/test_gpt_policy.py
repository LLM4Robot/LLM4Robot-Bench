import base64
from dataclasses import replace
from contextlib import redirect_stdout
from io import BytesIO
import json
import os
from pathlib import Path
import socket
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

from PIL import Image

from robot_tasks.backends.mock import MockBackend, MockRobot, MockTask
from fixtures import robot_config, rgbd, gripper_command, decision_document, RGBDBackend
from robot_tasks.policies.action_codec import decode_action
from robot_tasks.policies.gpt import GPTPolicy
from robot_tasks.policies.model_config import GPTSettings, load_gpt_settings, read_api_key
from robot_tasks.policies.responses import ModelError, ModelReply, ResponsesClient, RetryableModelError, completed_response, read_response
from robot_tasks.runner import run_episode
from robot_tasks.types import ArmState, Feedback, Observation, PolicyInput, RGBFrame


ROOT = Path(__file__).resolve().parents[1]
SETTINGS = GPTSettings("https://provider.example/openai", "gpt-6-astra")


def action_document(**updates):
    result = {"commands": [{"arm": "left", "kind": "gripper", "values": [0.06], "frame": "world"}],
              "request_stop": False}
    result.update(updates)
    return result


def response_document(text=None, **updates):
    result = {"id": "response_test", "model": SETTINGS.model, "status": "completed",
              "output": [{"type": "message", "role": "assistant", "content": [
                  {"type": "output_text", "text": text if text is not None else json.dumps(action_document())}]}],
              "usage": {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30}}
    result.update(updates)
    return result


class Stream(BytesIO):
    def __init__(self, data, content_type="text/event-stream"):
        super().__init__(data)
        self.headers = {"Content-Type": content_type}


def event(data):
    return b"data: " + json.dumps(data).encode() + b"\r\n\r\n"


class ActionCodecTests(unittest.TestCase):
    def setUp(self):
        self.config = robot_config()

    def test_valid_gripper_and_fixed_duration(self):
        action = decode_action(json.dumps(action_document()), self.config)
        self.assertEqual(action.commands[0].values, (0.06,))
        self.assertEqual(action.duration_s, self.config.control_dt_s)

    def test_invalid_robot_commands_rejected(self):
        commands = [
            {"arm": "franka"}, {"kind": "python"}, {"values": [True]},
            {"values": [float("nan")]}, {"values": [float("inf")]}, {"values": [0.10]},
            {"values": [0.06, 0.06]}, {"frame": "camera"},
            {"kind": "joint_position", "values": [3.0] * 6},
            {"kind": "cartesian_pose", "values": [0.3, 0.4, 0.2, 2.0, 0, 0, 0]},
        ]
        for changes in commands:
            with self.subTest(changes=changes):
                document = action_document()
                document["commands"][0].update(changes)
                with self.assertRaises(ValueError):
                    decode_action(json.dumps(document), self.config)

    def test_rejects_extra_fields_duplicate_keys_fences_and_stop_with_motion(self):
        rows = [json.dumps(action_document(code="do_something()")),
                '{"commands":[],"request_stop":true,"request_stop":false}',
                "```json\n" + json.dumps(action_document()) + "\n```",
                json.dumps(action_document(request_stop=True)),
                json.dumps(action_document(request_stop=1))]
        for text in rows:
            with self.subTest(text=text), self.assertRaises(ValueError):
                decode_action(text, self.config)

    def test_rejects_conflicting_channels_and_inactive_arm(self):
        document = action_document()
        document["commands"] *= 2
        with self.assertRaises(ValueError):
            decode_action(json.dumps(document), self.config)
        with self.assertRaises(ValueError):
            decode_action(json.dumps(action_document()), replace(self.config, arms=("right",)))

    def test_wait_and_stop_are_distinct_and_have_no_commands(self):
        for stopped in (True, False):
            action = decode_action(json.dumps(action_document(commands=[], request_stop=stopped)), self.config)
            self.assertEqual(action.commands, ())
            self.assertIs(action.request_stop, stopped)


class ResponsesTests(unittest.TestCase):
    def read(self, data, content_type="text/event-stream"):
        return read_response(Stream(data, content_type), expected_model=SETTINGS.model,
                             start=time.monotonic(), timeout_s=10)

    def test_sse_uses_only_completed_output_not_partial_delta(self):
        data = b": heartbeat\n\n" + event({"type": "response.output_text.delta", "delta": "bad partial"})
        data += event({"type": "response.completed", "response": response_document()})
        reply = self.read(data)
        self.assertEqual(reply.text, json.dumps(action_document()))
        self.assertEqual(reply.usage["total_tokens"], 30)

    def test_json_response_supported(self):
        reply = self.read(json.dumps(response_document()).encode(), "application/json")
        self.assertEqual(reply.model, SETTINGS.model)

    def test_failed_incomplete_and_error_events_preserve_sanitized_details(self):
        cases = [
            {"type": "response.failed", "response": {"id": "resp_fixture", "error": {
                "code": "server_error", "message": "test-private-token Bearer hidden-secret", "input": "private request"}}},
            {"type": "response.incomplete", "response": {"incomplete_details": {"reason": "max_output_tokens"}}},
            {"type": "error", "code": "server_error", "message": "upstream generation failed"},
        ]
        for value in cases:
            with self.subTest(value=value), self.assertRaises(RetryableModelError) as error:
                read_response(Stream(event(value)), expected_model=SETTINGS.model,
                              start=time.monotonic(), timeout_s=10, secret="test-private-token")
            message = str(error.exception)
            self.assertIn(value['type'], message)
            for secret in ('test-private-token', 'hidden-secret', 'private request'):
                self.assertNotIn(secret, message)

    def test_explicit_quota_auth_and_refusal_events_are_permanent(self):
        for value in (
            {'type': 'response.failed', 'response': {'error': {'code': 'invalid_api_key'}}},
            {'type': 'error', 'code': 'insufficient_quota'},
            {'type': 'response.incomplete', 'response': {'incomplete_details': {'reason': 'content_filter'}}},
            {'type': 'error', 'code': 'rate_limit_exceeded', 'message': 'apikey 7天限额已用完'},
            {'type': 'error', 'error': 'apikey 7天限额已用完'},
        ):
            with self.subTest(value=value), self.assertRaises(ModelError) as error:
                self.read(event(value))
            self.assertNotIsInstance(error.exception, RetryableModelError)

    def test_json_failed_response_uses_same_retry_contract(self):
        with self.assertRaises(RetryableModelError):
            self.read(json.dumps(response_document(status='failed', error={'code': 'server_error'})).encode(),
                      'application/json')

    def test_truncated_failed_and_incomplete_stream_never_produces_action(self):
        for data in (event({"type": "response.output_text.delta", "delta": json.dumps(action_document())}),
                     b"data: [DONE]\n\n", event({"type": "error"}),
                     event({"type": "response.incomplete"}), event({"type": "response.failed"})):
            with self.subTest(data=data), self.assertRaises(ModelError):
                self.read(data)

    def test_refusal_unexpected_model_and_tool_calls_rejected(self):
        for document in (
            response_document(model="gpt-5.5"), response_document(status="incomplete"),
            response_document(output=[{"type": "function_call"}]),
            response_document(output=[{"type": "message", "role": "assistant",
                                       "content": [{"type": "refusal", "refusal": "No"}]}]),
        ):
            with self.subTest(document=document), self.assertRaises(ModelError):
                completed_response(document, SETTINGS.model, 1)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-private-token"})
    def test_request_preserves_developer_contract_and_never_puts_key_in_body(self):
        stream = Stream(event({"type": "response.completed", "response": response_document()}))
        opener = Mock()
        opener.open.return_value = stream
        with patch("robot_tasks.policies.responses.build_opener", return_value=opener):
            ResponsesClient(SETTINGS).generate(instructions="ACTION CONTRACT", messages=[], schema={})
        request = opener.open.call_args.args[0]
        self.assertEqual(request.full_url, "https://provider.example/openai/responses")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-private-token")
        payload = json.loads(request.data)
        self.assertFalse(payload["store"])
        self.assertTrue(payload["stream"])
        self.assertEqual(payload["reasoning"]["summary"], "auto")
        self.assertEqual(payload["input"][0]["role"], "developer")
        self.assertEqual(payload["input"][0]["content"][0]["text"], "ACTION CONTRACT")
        self.assertNotIn(b"test-private-token", request.data)
        self.assertNotIn("max_output_tokens", payload)
        self.assertNotIn("service_tier", payload)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-private-token"})
    def test_speed_tier_is_forwarded_independently_of_reasoning_effort(self):
        opener = Mock()
        opener.open.return_value = Stream(event({"type": "response.completed", "response": response_document()}))
        settings = replace(SETTINGS, service_tier="ultrafast", reasoning_effort="medium")
        with patch("robot_tasks.policies.responses.build_opener", return_value=opener):
            ResponsesClient(settings).generate(instructions="contract", messages=[], schema={})
        payload = json.loads(opener.open.call_args.args[0].data)
        self.assertEqual(payload["service_tier"], "ultrafast")
        self.assertEqual(payload["reasoning"]["effort"], "medium")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-private-token"})
    def test_transport_timeouts_are_retryable_without_hidden_client_retries(self):
        for failure in (TimeoutError("test-private-token"), URLError(socket.timeout("test-private-token"))):
            with self.subTest(failure=type(failure).__name__):
                opener = Mock()
                opener.open.side_effect = failure
                with patch("robot_tasks.policies.responses.build_opener", return_value=opener):
                    with self.assertRaises(RetryableModelError) as error:
                        ResponsesClient(SETTINGS).generate(instructions="", messages=[], schema={})
                opener.open.assert_called_once()
                self.assertNotIn("test-private-token", str(error.exception))

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-private-token"})
    def test_non_timeout_transport_error_remains_non_retryable(self):
        opener = Mock()
        opener.open.side_effect = URLError("connection refused")
        with patch("robot_tasks.policies.responses.build_opener", return_value=opener):
            with self.assertRaises(ModelError) as error:
                ResponsesClient(SETTINGS).generate(instructions="", messages=[], schema={})
        self.assertNotIsInstance(error.exception, RetryableModelError)
        opener.open.assert_called_once()

    def test_stream_time_budget_is_retryable_and_partial_output_is_discarded(self):
        stream = Stream(event({"type": "response.output_text.delta", "delta": "partial action"}))
        with patch("robot_tasks.policies.responses.time.monotonic", side_effect=[0, 0, 11]):
            with self.assertRaisesRegex(RetryableModelError, "stream exceeded"):
                read_response(stream, expected_model=SETTINGS.model, start=0, timeout_s=10)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-private-token"})
    def test_read_timeout_after_partial_stream_is_retryable(self):
        stream = Mock()
        stream.headers = {"Content-Type": "text/event-stream"}
        stream.__enter__ = Mock(return_value=stream)
        stream.__exit__ = Mock(return_value=False)
        stream.readline.side_effect = [b'data: {"type": "response.output_text.delta", "delta": "partial"}\n',
                                       b'\n', socket.timeout("test-private-token")]
        opener = Mock()
        opener.open.return_value = stream
        with patch("robot_tasks.policies.responses.build_opener", return_value=opener):
            with self.assertRaises(RetryableModelError) as error:
                ResponsesClient(SETTINGS).generate(instructions="", messages=[], schema={})
        self.assertNotIn("test-private-token", str(error.exception))
        stream.__exit__.assert_called_once()
        opener.open.assert_called_once()

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-private-token"})
    def test_completed_response_after_deadline_is_discarded_and_retryable(self):
        opener = Mock()
        opener.open.return_value = Stream(json.dumps(response_document()).encode(), "application/json")
        with patch("robot_tasks.policies.responses.build_opener", return_value=opener), \
                patch("robot_tasks.policies.responses.time.monotonic", side_effect=[0, 11, 11]):
            with self.assertRaisesRegex(RetryableModelError, "time budget; output discarded"):
                ResponsesClient(replace(SETTINGS, timeout_s=10)).generate(instructions="", messages=[], schema={})
        opener.open.assert_called_once()

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-private-token"})
    def test_http_failure_is_sanitized_and_never_retried(self):
        opener = Mock()
        opener.open.side_effect = HTTPError("https://example.org", 401, "test-private-token", {}, BytesIO(b"secret"))
        with patch("robot_tasks.policies.responses.build_opener", return_value=opener):
            with self.assertRaises(ModelError) as error:
                ResponsesClient(SETTINGS).generate(instructions="", messages=[], schema={})
        self.assertNotIn("test-private-token", str(error.exception))
        self.assertIn("401", str(error.exception))
        self.assertEqual(opener.open.call_count, 1)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-private-token"})
    def test_429_preserves_provider_reason_and_retry_header_without_credentials(self):
        from email.message import Message
        headers = Message()
        headers["retry-after"] = "30"
        headers["x-request-id"] = "req-fixture"
        body = BytesIO(json.dumps({"error": {
            "type": "rate_limit_error", "code": "insufficient_quota",
            "message": "Account quota exceeded; key=test-private-token Authorization: Bearer secret-other",
            "request": {"api_key": "nested-secret-must-not-be-logged"}}}).encode())
        opener = Mock()
        opener.open.side_effect = HTTPError("https://example.org", 429, "ignored reason", headers, body)
        with patch("robot_tasks.policies.responses.build_opener", return_value=opener), self.assertRaises(ModelError) as error:
            ResponsesClient(SETTINGS).generate(instructions="", messages=[], schema={})
        message = str(error.exception)
        for expected in ("429", "insufficient_quota", "Account quota exceeded", "req-fixture", '"retry_after": "30"'):
            self.assertIn(expected, message)
        for secret in ("test-private-token", "secret-other", "nested-secret-must-not-be-logged"):
            self.assertNotIn(secret, message)
        opener.open.assert_called_once()
        self.assertTrue(body.closed)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-private-token"})
    def test_http_error_does_not_dump_html_large_payloads_or_control_characters(self):
        cases = ((b"<html>test-private-token</html>", "non_json_or_empty"),
                 (b"x" * 16385, "too_large"),
                 (json.dumps({"error": {"message": "Denied\n\x1b[31m " + "x"*1000}}).encode(), "Denied"),
                 (b'{"message":"Channel busy","code":429}', "Channel busy"))
        for raw, expected in cases:
            with self.subTest(expected=expected):
                opener = Mock()
                opener.open.side_effect = HTTPError("https://example.org", 429, "ignored", {}, BytesIO(raw))
                with patch("robot_tasks.policies.responses.build_opener", return_value=opener), self.assertRaises(ModelError) as error:
                    ResponsesClient(SETTINGS).generate(instructions="", messages=[], schema={})
                message = str(error.exception)
                self.assertIn(expected, message)
                self.assertNotIn("test-private-token", message)
                self.assertNotIn("<html>", message)
                self.assertNotIn("\x1b", message)
                self.assertLess(len(message), 800)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-private-token"})
    def test_provider_echoing_token_is_not_recordable(self):
        opener = Mock()
        opener.open.return_value = Stream(json.dumps(response_document("test-private-token")).encode(), "application/json")
        with patch("robot_tasks.policies.responses.build_opener", return_value=opener), self.assertRaises(ModelError):
            ResponsesClient(SETTINGS).generate(instructions="", messages=[], schema={})


class GPTPolicyTests(unittest.TestCase):
    def setUp(self):
        self.config = robot_config()
        self.client = Mock()
        self.client.generate.return_value = ModelReply(json.dumps(decision_document(commands=[gripper_command()])), SETTINGS.model, "id", {}, 1.0)
        frames = tuple(rgbd(camera, 1.) for camera in self.config.cameras)
        self.context = PolicyInput("Open the left gripper", Observation(1, frames, {"left": ArmState()}, Feedback("completed")), 1)

    def test_image_contract_and_input_boundary(self):
        policy = GPTPolicy(self.config, SETTINGS, client=self.client)
        policy.act(self.context)
        messages = self.client.generate.call_args.kwargs["messages"]
        content = messages[-1]["content"]
        text = json.loads(content[0]["text"])
        self.assertEqual(set(text), {"instruction", "step", "simulation_time_s", "robot", "feedback", "tcp_orientation_reference_deg"})
        data = content[2]["image_url"].split(",", 1)[1]
        image = Image.open(BytesIO(base64.b64decode(data)))
        self.assertEqual(image.size, (32, 24))
        self.assertEqual(sum(v["type"] == "input_image" for v in content), 6)
        self.assertNotIn("data:image", json.dumps(policy.last_exchange))

    def test_public_summary_is_recorded_but_not_added_to_model_history(self):
        def generate(**kwargs):
            kwargs["on_reasoning"]("Inspect the handle before pulling.")
            return self.client.generate.return_value

        self.client.generate.side_effect = generate
        policy = GPTPolicy(self.config, SETTINGS, client=self.client)
        policy.act(self.context)
        self.assertEqual(policy.last_exchange["reasoning_summary"], ["Inspect the handle before pulling."])
        self.assertNotIn("Inspect the handle before pulling.", json.dumps(list(policy.history)))

    def test_zero_call_cap_keeps_generating_after_100_calls(self):
        policy = GPTPolicy(self.config, replace(SETTINGS, max_calls=0), client=self.client)
        policy.calls = 100
        policy.act(self.context)
        self.assertEqual(policy.calls, 101)
        self.assertEqual(self.client.generate.call_count, 1)

    def test_history_bounded_and_reset_clears_budget(self):
        policy = GPTPolicy(self.config, replace(SETTINGS, max_calls=4), client=self.client)
        for _ in range(4):
            policy.act(self.context)
        self.assertEqual(len(policy.history), 4)
        self.assertEqual(len(self.client.generate.call_args.kwargs["messages"]), 7)
        self.assertFalse(any(v["type"] == "input_image" for v in policy.history[0][0]["content"]))
        with self.assertRaises(ModelError):
            policy.act(self.context)
        self.assertEqual(self.client.generate.call_count, 4)
        policy.reset(0)
        self.assertEqual(policy.calls, 0)
        self.assertFalse(policy.history)

    def test_invalid_output_retained_as_rejection_without_action_or_history(self):
        self.client.generate.return_value = ModelReply('{"wrong":true}', SETTINGS.model, "id", {}, 1)
        policy = GPTPolicy(self.config, SETTINGS, client=self.client)
        with self.assertRaises(ValueError):
            policy.act(self.context)
        self.assertEqual(policy.last_exchange["status"], "rejected")
        self.assertNotIn("action", policy.last_exchange)
        self.assertFalse(policy.history)

    def test_missing_image_fails_before_paid_request(self):
        policy = GPTPolicy(self.config, SETTINGS, client=self.client)
        context = replace(self.context, observation=replace(self.context.observation, rgb=()))
        with self.assertRaises(ValueError):
            policy.act(context)
        self.client.generate.assert_not_called()

    def test_runner_retains_rejected_call_and_never_moves_or_claims_success(self):
        self.client.generate.return_value = ModelReply('{"wrong":true}', SETTINGS.model, "id", {}, 1)
        config = replace(self.config, raw={**self.config.raw, "execution": {"settle_steps": 0}})
        policy = GPTPolicy(config, SETTINGS, client=self.client)
        backend, robot = RGBDBackend(), MockRobot()
        with tempfile.TemporaryDirectory() as directory, patch.object(robot, "apply") as apply:
            path, result = run_episode(config, backend, robot, MockTask(), policy, Path(directory))
            events = [json.loads(line) for line in (path / "episode.jsonl").read_text().splitlines()]
        apply.assert_not_called()
        self.assertEqual(backend.time_s(), 0)
        self.assertIsNone(result["success"])
        self.assertEqual(result["status"], "finished")
        self.assertEqual(result["termination_reason"], "model_call_limit")
        self.assertEqual(result["model_calls"], SETTINGS.max_calls)
        self.assertEqual(result["invalid_model_actions"], SETTINGS.max_calls)
        self.assertEqual(events[0]["event"], "policy_configuration")
        exchanges = [e["exchange"] for e in events if e["event"] == "model_exchange"]
        self.assertEqual(len(exchanges), SETTINGS.max_calls)
        self.assertTrue(all(e["status"] == "rejected" for e in exchanges))



class SettingsTests(unittest.TestCase):
    def test_service_tier_validation_does_not_conflate_speed_with_reasoning(self):
        for tier in (None, "fast", "ultrafast"):
            self.assertEqual(replace(SETTINGS, service_tier=tier).service_tier, tier)
        for tier in ("ultra fast", "medium", "", True, []):
            with self.subTest(tier=tier), self.assertRaisesRegex(ValueError, "service_tier"):
                replace(SETTINGS, service_tier=tier)
        with self.assertRaisesRegex(ValueError, "reasoning_effort"):
            replace(SETTINGS, reasoning_effort="ultrafast")

    def test_committed_config_has_no_credentials(self):
        settings = load_gpt_settings(ROOT / "configs/models/gpt.toml")
        self.assertEqual(settings.api_key_env, "OPENAI_API_KEY")
        self.assertNotIn("api_key", settings.public_dict())

    def test_invalid_endpoint_and_limits_rejected(self):
        for changes in ({"base_url": "http://example.org"}, {"base_url": "https://user:secret@example.org"},
                        {"base_url": "https://example.org/?key=secret"}, {"base_url": "https://example.org/responses"},
                        {"timeout_s": float("inf")}, {"max_calls": True}, {"history_steps": -1}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(SETTINGS, **changes)

    def test_timeout_is_positive_finite_and_can_exceed_ten_minutes(self):
        for value in (0, -1, True, "90", None, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite positive"):
                replace(SETTINGS, timeout_s=value)
        self.assertEqual(replace(SETTINGS, timeout_s=1800).timeout_s, 1800)

    def test_unit_process_blocks_dns_and_connections_before_network_io(self):
        with self.assertRaisesRegex(RuntimeError, "Unit tests cannot access the network"):
            socket.getaddrinfo("example.invalid", 443)
        with socket.socket() as connection:
            with self.assertRaisesRegex(RuntimeError, "Unit tests cannot access the network"):
                connection.connect(("127.0.0.1", 9))

    @patch.dict(os.environ, {"ROBOT_TEST_API_KEY": "private-token"})
    def test_env_key_read_without_modification(self):
        self.assertEqual(read_api_key("ROBOT_TEST_API_KEY"), "private-token")


if __name__ == "__main__":
    unittest.main()
