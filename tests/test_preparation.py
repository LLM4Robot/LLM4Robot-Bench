"""Startup settling preserves joint home and is never scored as model work."""

from dataclasses import replace
import json
import tempfile
import unittest
from pathlib import Path


from robot_tasks.backends.mock import MockRobot, MockTask
from robot_tasks.config import ExecutionSettings
from robot_tasks.workflow import workflow_metadata
from robot_tasks.registry import resolve_task
from robot_tasks.robots.preparation import preparation_action
from robot_tasks.runner import run_episode
from robot_tasks.types import Action, Assessment, Feedback
from fixtures import RGBDBackend


class PreparingRobot(MockRobot):
    def initialize(self, backend, config):
        super().initialize(backend, config)
        self.backend = backend
        self.applied = []
        self.finish_time = 0.0

    def apply(self, action):
        self.applied.append(action)
        self.finish_time = self.backend.time_s() + 0.25

    def feedback(self):
        return Feedback("completed" if self.backend.time_s() >= self.finish_time else "executing")


class StopPolicy:
    def __init__(self):
        self.seen = []

    def reset(self, seed):
        self.seen.clear()

    def act(self, context):
        self.seen.append(context)
        return Action((), 1 / 12, True)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        _, self.config = resolve_task("GPT-PICK-PLACE")

    def test_startup_holds_home_without_moving_or_aiming_any_arm(self):
        from robot_tasks.registry import TASKS

        for task in TASKS.values():
            if not task.implemented:
                continue
            _, config = resolve_task(task.task_id)
            with self.subTest(task=task.task_id):
                action = preparation_action(config)
                self.assertEqual(action.commands, ())
                self.assertFalse(action.request_stop)
                for mount in config.robot.mounts:
                    self.assertEqual(mount.home_positions_rad, (0.,) * 6)

    def test_task_and_camera_changes_do_not_inject_pre_model_motion(self):
        changed = replace(self.config, seed=99, instruction="Different task",
                          arms=("left", "right"), observation_arms=())
        self.assertEqual(preparation_action(changed), preparation_action(self.config))
        self.assertEqual(changed.command_arms, ("left", "right"))

    def test_policy_knows_observation_camera_is_not_pre_aimed(self):
        from robot_tasks.policies.operating_notes import operating_notes

        _, config = resolve_task("L1-01")
        notes = operating_notes(config)
        self.assertIn("No scripted downward movement", notes["startup_preparation"])
        self.assertIn("move an observation arm", notes["startup_preparation"])
        self.assertEqual(config.arm_roles["left"], "observation")

    def test_preparation_uses_shared_executor_without_task_switches(self):
        self.assertEqual(workflow_metadata()["startup_preparation"], "hold_joint_home_v1")
        for field, value in (("startup_pose", "none"), ("motion_mode", "direct_ik")):
            raw = {**self.config.raw, "execution": {**self.config.raw["execution"], field: value}}
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Unknown execution"):
                ExecutionSettings.from_config(replace(self.config, raw=raw))

    def run_case(self, robot):
        policy = StopPolicy()
        with tempfile.TemporaryDirectory() as directory:
            config = replace(self.config, raw={**self.config.raw, "execution": {**self.config.raw["execution"], "max_decisions": 1}})
            path, result = run_episode(config, RGBDBackend(), robot, MockTask(), policy, Path(directory),
                                       preparation=preparation_action(self.config))
            events = [json.loads(l) for l in (path / "episode.jsonl").read_text().splitlines()]
        return result, events, policy

    def test_preparation_finishes_before_first_policy_observation(self):
        robot = PreparingRobot()
        result, events, policy = self.run_case(robot)
        self.assertEqual(len(robot.applied), 1)
        self.assertEqual(robot.applied[0], preparation_action(self.config))
        self.assertEqual(len(policy.seen), 1)
        complete = next(e for e in events if e["event"] == "preparation_complete")
        self.assertAlmostEqual(policy.seen[0].observation.time_s, complete["time_s"])
        self.assertEqual(policy.seen[0].observation.feedback.status, "completed")
        self.assertEqual(result["executed_actions"], 0)
        self.assertEqual(result["decisions"], 1)
        for event in events:
            if event.get("phase") == "preparation":
                self.assertEqual(event["evaluation"]["status"], "not_evaluated")

    def test_preparation_failure_never_calls_policy(self):
        class BlockedRobot(PreparingRobot):
            def feedback(self):
                return Feedback("executing")

        result, events, policy = self.run_case(BlockedRobot())
        self.assertEqual(policy.seen, [])
        self.assertEqual(result["status"], "error")
        self.assertFalse(result["task_evaluated"])
        self.assertIsNone(result["success"])
        self.assertIn("preparation", result["error"])

    def test_terminal_failure_does_not_repeat_startup_preparation(self):
        class RetryPolicy(StopPolicy):
            def act(self, context):
                self.seen.append(context)
                return Action((), 1 / 12, len(self.seen) > 1)

        policy = RetryPolicy()

        class RetryTask(MockTask):
            resets = 0

            def reset(self, backend, seed):
                self.resets += 1

            def evaluate(self, state, feedback):
                if self.resets == 1 and policy.seen:
                    return Assessment("failed", "Synthetic retry trigger", retryable=True)
                return Assessment("running", "Fresh attempt")

        config = replace(self.config, raw={**self.config.raw, "execution": {
            **self.config.raw["execution"], "max_decisions": 2}})
        robot = PreparingRobot()
        with tempfile.TemporaryDirectory() as directory:
            path, result = run_episode(config, RGBDBackend(), robot, RetryTask(), policy, Path(directory),
                                       preparation=preparation_action(config))
            events = [json.loads(line) for line in (path / "episode.jsonl").read_text().splitlines()]
        self.assertFalse(result["success"])
        self.assertEqual(result["termination_reason"], "task_terminal")
        self.assertEqual(len(policy.seen), 1)
        # The policy's first wait has the same value as startup settling; the
        # event phase, not Action equality, distinguishes these two executions.
        self.assertEqual(len(robot.applied), 2)
        self.assertEqual(robot.applied[0], preparation_action(config))
        self.assertEqual(sum(e["event"] == "preparation_complete" for e in events), 1)
        self.assertFalse(any(e["event"].startswith("attempt_reset") for e in events))



if __name__ == "__main__":
    unittest.main()
