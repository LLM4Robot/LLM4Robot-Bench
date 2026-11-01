"""Uncapped generation retries reuse observations; no paid/network calls."""

import json
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

from fixtures import RGBDBackend, decision_document, gripper_command, robot_config

from robot_tasks.backends.mock import MockRobot, MockTask
from robot_tasks.policies.codex_cli import CodexCLIClient
from robot_tasks.policies.gpt import GPTPolicy
from robot_tasks.policies.model_config import GPTSettings
from robot_tasks.policies.responses import ModelError, ModelReply, RetryableModelError
from robot_tasks.runner import run_episode
from robot_tasks.types import Assessment, Feedback


def failed():
    return RetryableModelError("Provider reported failed/incomplete generation")


def reply(*, move=False):
    doc = decision_document(commands=[gripper_command()] if move else [])
    return ModelReply(json.dumps(doc), "fixture", "offline", {}, 0.0)


class ModelRetryTests(unittest.TestCase):
    def run_fixture(self, replies=None, calls=10, *, interrupt=False, client=None,
                    finish_after_action=False, wall_time_s=0., sleep_effect=None):
        config = robot_config()
        config = replace(
            config,
            max_steps=0,
            max_wall_time_s=wall_time_s,
            raw={
                **config.raw,
                "execution": {
                    "max_decisions": calls,
                    "max_action_steps": 1,
                    "completion_samples": 1,
                },
            },
        )
        settings = GPTSettings("https://example.invalid", "fixture", max_calls=calls)
        if client is None:
            client = Mock()
            client.generate.side_effect = replies
        else:
            settings = replace(client.settings, max_calls=calls)
        policy = GPTPolicy(
            config,
            settings,
            client=client,
        )
        robot, task = MockRobot(), MockTask()
        robot.apply = Mock()
        robot.feedback = Mock(return_value=Feedback("completed"))
        task.evaluate = Mock(return_value=Assessment("running", "Offline fixture"))
        if finish_after_action:
            task.evaluate.side_effect = lambda *_: Assessment(
                "succeeded" if robot.apply.called else "running", "Offline fixture")
        with (
            tempfile.TemporaryDirectory() as directory,
            redirect_stdout(StringIO()),
            patch(
                "robot_tasks.runner.time.sleep",
                side_effect=KeyboardInterrupt if interrupt else sleep_effect,
            ) as sleep,
        ):
            output, result = run_episode(
                config, RGBDBackend(), robot, task, policy, Path(directory)
            )
            events = [
                json.loads(l)
                for l in (output / "episode.jsonl").read_text().splitlines()
            ]
        return result, events, policy, robot, client, sleep

    def test_unlimited_retries_pass_100_failures_then_execute_once(self):
        result, events, policy, robot, client, sleep = self.run_fixture(
            [failed()] * 105 + [reply(move=True)], calls=0, finish_after_action=True)
        self.assertEqual(client.generate.call_count, 106)
        self.assertEqual(result["status"], "finished")
        self.assertTrue(result["success"])
        self.assertEqual(result["model_retry_attempts"], 105)
        self.assertIsNone(result["generation_retry_limit"])
        robot.apply.assert_called_once()
        self.assertEqual(len(policy.history), 1)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [2., 4.] + [8.] * 103)
        retries = [e for e in events if e["event"].startswith("model_retry")]
        self.assertTrue(all(e["remaining_decisions"] is None for e in retries))
        self.assertTrue(all(e["retry_limit"] is None for e in retries))
        self.assertTrue(all(e["retry_exhaustion"] is None for e in retries))
        self.assertTrue(all(e["simulation_time_s"] == 0. for e in retries))
        exchanges = [e["exchange"] for e in events if e["event"] == "model_exchange"]
        self.assertTrue(all(e["frames"] == exchanges[0]["frames"] for e in exchanges))

    def test_five_hour_deadline_interrupts_retry_backoff(self):
        clock = [0.]
        client = Mock()
        client.settings = GPTSettings("https://example.invalid", "fixture", max_calls=0)
        def generate(**_):
            if client.generate.call_count == 5:
                clock[0] = 17999.
            raise failed()
        client.generate.side_effect = generate
        def sleep_for(delay):
            clock[0] += delay
        with patch("robot_tasks.runner.time.monotonic", side_effect=lambda: clock[0]):
            result, events, _policy, robot, _client, sleep = self.run_fixture(
                calls=0, client=client, wall_time_s=18000., sleep_effect=sleep_for)
        self.assertEqual(result["termination_reason"], "wall_time_limit")
        self.assertEqual(result["wall_time_s"], 18000.)
        self.assertEqual(result["model_calls"], 5)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [2., 4., 8., 8., 1.])
        robot.apply.assert_not_called()

    def codex_fixture(self):
        settings = GPTSettings("https://example.invalid", "fixture", transport="codex_cli", timeout_s=2400)
        with patch("robot_tasks.policies.codex_cli.CodexProcess") as host_type:
            host_type.return_value.mode = "native"
            host_type.return_value.path.side_effect = lambda path: path
            client = CodexCLIClient(settings)
        rows = [
            {"type": "thread.started", "thread_id": "offline"},
            {"type": "item.completed", "item": {"type": "agent_message", "text": reply(move=True).text}},
            {"type": "turn.completed", "usage": {}},
        ]
        completed = subprocess.CompletedProcess([], 0, "\n".join(map(json.dumps, rows)).encode(), b"")
        return client, completed

    def test_codex_timeouts_retry_same_context_then_resume_after_three_retries(self):
        client, completed = self.codex_fixture()
        client.host.run.side_effect = [completed] + [subprocess.TimeoutExpired("codex", 2400)] * 3 + [completed]
        result, events, policy, robot, _, sleep = self.run_fixture(calls=5, client=client)
        self.assertEqual(result["status"], "finished")
        self.assertEqual(result["model_calls"], 5)
        self.assertEqual(result["model_retry_attempts"], 3)
        self.assertEqual(robot.apply.call_count, 2)
        self.assertEqual(len(policy.history), 2)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [2.0, 4.0, 8.0])
        calls = client.host.run.call_args_list
        self.assertEqual(len(calls), 5)
        self.assertTrue(all(c.kwargs["timeout_s"] == 2400 for c in calls))
        # The request counter advances; sensor data and accepted history do not.
        for step, call in enumerate(calls[1:], 2):
            normalized = call.kwargs["data"].replace(f'"step":{step},'.encode(), b'"step":2,')
            self.assertEqual(normalized, calls[1].kwargs["data"])
        exchanges = [e["exchange"] for e in events if e["event"] == "model_exchange"]
        self.assertEqual([e["history_steps"] for e in exchanges], [0, 1, 1, 1, 1])
        self.assertEqual([e["retryable"] for e in exchanges[1:4]], [True] * 3)
        self.assertTrue(all(e["frames"] == exchanges[1]["frames"] for e in exchanges[1:]))

    def test_codex_timeouts_continue_until_configured_call_budget(self):
        client, _completed = self.codex_fixture()
        client.host.run.side_effect = subprocess.TimeoutExpired("codex", 2400, output=b"partial action")
        result, events, policy, robot, _, _sleep = self.run_fixture(client=client)
        self.assertEqual(result["status"], "error")
        self.assertEqual(client.host.run.call_count, 10)
        self.assertEqual(result["model_calls"], 10)
        self.assertEqual(result["model_retry_attempts"], 9)
        self.assertEqual(result["completed_steps"], 0)
        self.assertFalse(policy.history)
        robot.apply.assert_not_called()
        exhausted = [e for e in events if e["event"] == "model_retry_exhausted"]
        self.assertEqual(exhausted[0]["retry_exhaustion"], "model_call_limit")

    def test_unknown_codex_exit_retries_beyond_three_and_preserves_diagnostics(self):
        client, completed = self.codex_fixture()
        failure = subprocess.CompletedProcess([], 1, b'{"type":"error","message":"connection reset"}',
                                              b"upstream stream closed")
        client.host.run.side_effect = [failure] * 6 + [completed]
        result, events, policy, robot, _, sleep = self.run_fixture(calls=7, client=client)
        self.assertEqual(result["status"], "finished")
        self.assertEqual(result["model_retry_attempts"], 6)
        self.assertEqual(robot.apply.call_count, 1)
        self.assertEqual(len(policy.history), 1)
        exchanges = [e["exchange"] for e in events if e["event"] == "model_exchange"]
        self.assertEqual(exchanges[0]["provider_diagnostic"]["stderr_excerpt"], "upstream stream closed")
        self.assertNotIn("provider_diagnostic", exchanges[-1])

    def test_codex_quota_exit_and_user_cancellation_do_not_retry(self):
        for code, stderr, status in ((1, b"You've hit your usage limit", "error"), (130, b"", "cancelled")):
            with self.subTest(code=code):
                client, _ = self.codex_fixture()
                client.host.run.return_value = subprocess.CompletedProcess([], code, b"", stderr)
                result, events, policy, robot, _, sleep = self.run_fixture(client=client)
                self.assertEqual(result["status"], status)
                self.assertEqual(result["model_calls"], 1)
                robot.apply.assert_not_called()
                sleep.assert_not_called()

    def test_three_retries_then_success_execute_once_and_count_four_calls(self):
        result, events, policy, robot, client, sleep = self.run_fixture(
            [failed(), failed(), failed(), reply(move=True)], calls=4
        )
        self.assertEqual(result["status"], "finished")
        self.assertEqual(result["model_calls"], 4)
        self.assertEqual(client.generate.call_count, 4)
        self.assertEqual(result["model_retry_attempts"], 3)
        self.assertEqual(result["generation_failures"], 3)
        robot.apply.assert_called_once()
        self.assertEqual(len(policy.history), 1)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [2.0, 4.0, 8.0])
        retries = [e for e in events if e["event"] == "model_retry"]
        self.assertEqual([e["failure_number"] for e in retries], [1, 2, 3])
        self.assertTrue(all(e["simulation_time_s"] == 0.0 for e in retries))
        exchanges = [e["exchange"] for e in events if e["event"] == "model_exchange"]
        self.assertTrue(all(e["frames"] == exchanges[0]["frames"] for e in exchanges))
        self.assertEqual([e["retryable"] for e in exchanges[:3]], [True] * 3)

    def test_configured_call_budget_stops_retries_without_partial_actions(self):
        result, events, policy, robot, client, _sleep = self.run_fixture([failed()] * 7, calls=7)
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["model_calls"], 7)
        self.assertEqual(result["model_retry_attempts"], 6)
        self.assertEqual(client.generate.call_count, 7)
        self.assertEqual(result["completed_steps"], 0)
        self.assertFalse(result["task_evaluated"])
        self.assertIsNone(result["success"])
        robot.apply.assert_not_called()
        self.assertFalse(policy.history)
        exhausted = [e for e in events if e["event"] == "model_retry_exhausted"]
        self.assertEqual(exhausted[0]["retry_exhaustion"], "model_call_limit")

    def test_success_resets_retry_streak_and_retains_accepted_history(self):
        result, events, policy, robot, client, sleep = self.run_fixture(
            [failed(), failed(), reply(), failed(), failed(), reply()], calls=6
        )
        self.assertEqual(result["status"], "finished")
        self.assertEqual(result["model_retry_attempts"], 4)
        self.assertEqual(client.generate.call_count, 6)
        self.assertEqual(robot.apply.call_count, 2)
        self.assertEqual(len(policy.history), 2)
        exchanges = [e["exchange"] for e in events if e["event"] == "model_exchange"]
        self.assertEqual([e["history_steps"] for e in exchanges], [0, 0, 0, 1, 1, 1])
        self.assertEqual(
            [c.args[0] for c in sleep.call_args_list], [2.0, 4.0, 2.0, 4.0]
        )

    def test_retry_cannot_exceed_100_total_requests(self):
        result, events, _policy, _robot, client, _sleep = self.run_fixture(
            [reply()] * 98 + [failed(), failed()], calls=100
        )
        self.assertEqual(result["model_calls"], 100)
        self.assertEqual(client.generate.call_count, 100)
        self.assertEqual(result["model_retry_attempts"], 1)
        self.assertEqual(result["status"], "error")
        self.assertFalse(result["task_evaluated"])
        exhausted = [e for e in events if e["event"] == "model_retry_exhausted"]
        self.assertEqual(exhausted[0]["retry_exhaustion"], "model_call_limit")

    def test_permanent_error_is_not_retried(self):
        result, _events, _policy, robot, _client, sleep = self.run_fixture(
            [ModelError("Responses HTTP 401")]
        )
        self.assertEqual(result["model_calls"], 1)
        self.assertEqual(result["model_retry_attempts"], 0)
        sleep.assert_not_called()
        robot.apply.assert_not_called()

    def test_interrupt_during_backoff_does_not_start_another_call(self):
        result, _events, _policy, robot, client, _sleep = self.run_fixture(
            [failed()], interrupt=True
        )
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(client.generate.call_count, 1)
        self.assertEqual(result["model_calls"], 1)
        self.assertEqual(result["model_retry_attempts"], 0)
        robot.apply.assert_not_called()

    def test_manual_stop_after_long_retry_streak(self):
        result, _events, _policy, robot, client, _sleep = self.run_fixture(
            [failed()] * 6, calls=0, sleep_effect=[None] * 5 + [KeyboardInterrupt])
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(result["termination_reason"], "user_interrupt")
        self.assertEqual(client.generate.call_count, 6)
        robot.apply.assert_not_called()


if __name__ == "__main__":
    unittest.main()
