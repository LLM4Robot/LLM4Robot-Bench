"""Offline episode termination regressions; no simulator or model requests."""

from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fixtures import RGBDBackend
from robot_tasks.backends.mock import MockRobot, MockTask
from robot_tasks.config import ExecutionSettings
from robot_tasks.registry import TASKS, resolve_task
from robot_tasks.runner import run_episode
from robot_tasks.types import Action, Assessment, RetryablePolicyError


class TerminalTask(MockTask):
    def __init__(self, *, initial_failure=False, retryable=False):
        self.resets = 0
        self.initial_failure = initial_failure
        self.retryable = retryable

    def reset(self, backend, seed):
        self.resets += 1

    def evaluate(self, state, feedback):
        if self.initial_failure or state.values["simulation_time_s"] > 0:
            return Assessment("failed", "Object landed on the table", retryable=self.retryable)
        return Assessment("running", "Initial state")


class TrialPolicy:
    def __init__(self, *, stop=False, generation_failures=0):
        self.calls = self.resets = 0
        self.stop = stop
        self.generation_failures = generation_failures
        self.observations = []

    def reset(self, seed):
        self.resets += 1

    def act(self, context):
        self.calls += 1
        self.observations.append(context.observation)
        if self.calls <= self.generation_failures:
            raise RetryablePolicyError("Offline simulated timeout")
        return Action((), 1 / 12, request_stop=self.stop)


class SingleEpisodeTests(unittest.TestCase):
    def setUp(self):
        _, base = resolve_task("GPT-PICK-PLACE")
        self.config = replace(base, raw={**base.raw, "execution": {
            "max_action_steps": 4, "max_decisions": 100, "stop_steps": 1}})
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def run_case(self, task, policy):
        backend = RGBDBackend()
        with redirect_stdout(io.StringIO()), patch("robot_tasks.runner.time.sleep"):
            path, result = run_episode(self.config, backend, MockRobot(), task, policy, Path(self.temp.name))
        events = [json.loads(s) for s in (path / "episode.jsonl").read_text().splitlines()]
        self.assertEqual(task.resets, 1)
        self.assertEqual(policy.resets, 1)
        self.assertFalse(any(e["event"].startswith("attempt_reset") for e in events))
        self.assertEqual(result["evaluation_protocol"], "single_episode")
        self.assertEqual(result["simulation_time_s"], backend.time_s())
        return result

    def test_terminal_failure_never_resets_even_with_legacy_retryable_flag(self):
        for retryable in (False, True):
            with self.subTest(retryable=retryable):
                result = self.run_case(TerminalTask(retryable=retryable), TrialPolicy())
                self.assertFalse(result["success"])
                self.assertEqual(result["model_calls"], 1)
                self.assertEqual(result["termination_reason"], "task_terminal")

    def test_failure_during_stop_observation_ends_without_another_call(self):
        result = self.run_case(TerminalTask(), TrialPolicy(stop=True))
        self.assertFalse(result["success"])
        self.assertEqual(result["model_calls"], 1)
        self.assertEqual(result["rejected_stop_requests"], 0)
        self.assertEqual(result["termination_reason"], "task_terminal")

    def test_failed_initial_state_does_not_invoke_model(self):
        result = self.run_case(TerminalTask(initial_failure=True), TrialPolicy())
        self.assertFalse(result["success"])
        self.assertEqual(result["model_calls"], 0)

    def test_three_generation_retries_keep_the_same_scene_and_observation(self):
        policy = TrialPolicy(generation_failures=3)
        result = self.run_case(TerminalTask(), policy)
        self.assertEqual(result["model_calls"], 4)
        self.assertEqual(result["model_retry_attempts"], 3)
        self.assertEqual(result["generation_failures"], 3)
        self.assertTrue(all(o is policy.observations[0] for o in policy.observations))
        self.assertEqual(result["termination_reason"], "task_terminal")

    def test_generation_failures_beyond_three_resume_without_scene_reset(self):
        result = self.run_case(TerminalTask(), TrialPolicy(generation_failures=4))
        self.assertEqual(result["model_calls"], 5)
        self.assertEqual(result["model_retry_attempts"], 4)
        self.assertEqual(result["status"], "finished")
        self.assertEqual(result["termination_reason"], "task_terminal")
        self.assertFalse(result["success"])

    def test_all_launchable_tasks_have_no_physical_retry_switch(self):
        for task in TASKS.values():
            if task.implemented:
                with self.subTest(task=task.task_id):
                    _, config = resolve_task(task.task_id)
                    ExecutionSettings.from_config(config)
                    self.assertNotIn("retry_failed_attempts", config.raw.get("execution", {}))
        old = replace(self.config, raw={**self.config.raw, "execution": {"retry_failed_attempts": True}})
        with self.assertRaisesRegex(ValueError, "Unknown execution settings"):
            ExecutionSettings.from_config(old)
