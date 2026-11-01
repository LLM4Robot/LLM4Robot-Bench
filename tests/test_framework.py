import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from robot_tasks.backends.mock import MockBackend, MockRobot, MockTask, NoopPolicy
from robot_tasks.config import load_config
from robot_tasks.registry import get_spec
from robot_tasks.runner import run_episode
from robot_tasks.types import Action, ArmCommand, Assessment


ROOT = Path(__file__).resolve().parents[1]


class FrameworkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.config = replace(load_config(ROOT / "configs/checks/gpt_single_arm_pick_place.toml"),
                              control_dt_s=0.1, include_depth=False, save_depth=False)

    def run_mock(self, *, config=None, backend=None, task=None, policy=None, dry_run=True):
        config = config or self.config
        return run_episode(
            config, backend or MockBackend(), MockRobot(), task or MockTask(),
            policy or NoopPolicy(config.control_dt_s), self.output,
            dry_run=dry_run, step_limit=3,
        )

    def test_dry_run_persists_evidence_but_never_task_success(self):
        for name in ("l1_01_cylinder_insertion.toml", "l2_01_marker_uncap.toml"):
            with self.subTest(name=name):
                config = load_config(ROOT / "configs/tasks" / name)
                backend = MockBackend()
                path, result = self.run_mock(config=config, backend=backend)
                self.assertEqual(result["status"], "dry_run_complete")
                self.assertFalse(result["task_evaluated"])
                self.assertIsNone(result["success"])
                self.assertTrue(backend.closed)
                self.assertEqual(json.loads((path / "result.json").read_text()), result)
                report = (path / "report.md").read_text(encoding="utf-8")
                self.assertIn("未评估", report)
                self.assertIn("(<episode.jsonl>)", report)
                self.assertIn("(<frames>)", report)
                events = [json.loads(s) for s in (path / "episode.jsonl").read_text().splitlines()]
                self.assertEqual(len(events), 4)  # reset + 3 transitions
                self.assertEqual(set(events[0]["observation"]["robot"]), set(config.command_arms))
                self.assertEqual(len(events[-1]["observation"]["rgb"]), len(config.cameras))
                self.assertAlmostEqual(events[-1]["observation"]["time_s"], 3 * config.control_dt_s)
                self.assertEqual(len(list(path.rglob("*.ppm"))), 4 * len(config.cameras))
                self.assertEqual((path / "robot_profile.toml").read_bytes(), config.robot.source.read_bytes())
                self.assertEqual((path / "asset_source.lock.json").read_bytes(), config.robot.asset_lock.read_bytes())
                manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["available_arms"], ["left", "right"])
                self.assertEqual(manifest["active_arms"], list(config.arms))

    def test_unimplemented_task_fails_explicitly(self):
        class PendingTask(MockTask):
            def build_scene(self, backend, config):
                raise NotImplementedError("Task scene is not implemented")

        _, result = self.run_mock(task=PendingTask(), dry_run=False)
        self.assertEqual(result["status"], "implementation_missing")
        self.assertIsNone(result["success"])
        self.assertEqual(result["completed_steps"], 0)

    def test_exception_closes_backend_and_writes_failure(self):
        class BrokenBackend(MockBackend):
            def step(self, duration_s):
                raise RuntimeError("physics failed")

        backend = BrokenBackend()
        path, result = self.run_mock(backend=backend)
        self.assertTrue(backend.closed)
        self.assertEqual(result["status"], "error")
        self.assertIn("physics failed", result["error"])
        self.assertTrue((path / "result.json").exists())

    def test_policy_stop_does_not_mean_success(self):
        class StopPolicy(NoopPolicy):
            def act(self, context):
                return Action((), self.duration_s, request_stop=True)

        _, result = self.run_mock(policy=StopPolicy(0.1))
        self.assertEqual(result["termination_reason"], "policy_requested_stop")
        self.assertEqual(result["completed_steps"], 0)
        self.assertIsNone(result["success"])

    def test_invalid_arm_is_rejected_before_physics(self):
        class BadPolicy(NoopPolicy):
            def act(self, context):
                return Action((ArmCommand("missing_arm", "gripper", (0.02,)),), 0.1)

        backend = MockBackend()
        _, result = self.run_mock(backend=backend, policy=BadPolicy(0.1))
        self.assertEqual(result["status"], "error")
        self.assertEqual(backend.time_s(), 0)
        self.assertIsNone(result["success"])

    def test_hidden_evaluation_state_not_in_policy_input(self):
        test = self

        class InspectPolicy(NoopPolicy):
            def act(self, context):
                test.assertEqual(set(vars(context)), {"instruction", "observation", "step"})
                test.assertFalse(hasattr(context.observation, "evaluation_state"))
                test.assertEqual(context.observation.robot, {})
                test.assertEqual(context.observation.feedback.status, "not_provided")
                return super().act(context)

        config = replace(self.config, include_robot_state=False, include_feedback=False)
        _, result = self.run_mock(config=config, policy=InspectPolicy(0.1))
        self.assertEqual(result["status"], "dry_run_complete")

    def test_dry_run_rejects_a_fake_success_evaluator(self):
        class FakeSuccess(MockTask):
            def evaluate(self, state, feedback):
                return Assessment("succeeded", "should not be accepted in dry run")

        _, result = self.run_mock(task=FakeSuccess())
        self.assertEqual(result["status"], "error")
        self.assertFalse(result["task_evaluated"])
        self.assertIsNone(result["success"])
        self.assertEqual(result["assessment"]["status"], "not_evaluated")

    def test_wall_limit_does_not_execute_policy_or_claim_success(self):
        config = replace(self.config, max_wall_time_s=1.0)
        policy = NoopPolicy(0.1)
        # Windows monotonic clocks can return the same value across fast calls.
        with patch("robot_tasks.runner.time.monotonic", side_effect=(10.0, 12.0, 12.0)), \
                patch.object(policy, "act") as act:
            _, result = self.run_mock(config=config, policy=policy)
            act.assert_not_called()
        self.assertEqual(result["termination_reason"], "wall_time_limit")
        self.assertEqual(result["completed_steps"], 0)
        self.assertIsNone(result["success"])

    def test_wall_limit_config_accepts_zero_and_rejects_invalid_values(self):
        for value in (0, 0.0, 60.0, -1, True, "0", float("inf"), float("nan")):
            raw = {**self.config.raw, "runtime": {**self.config.raw["runtime"], "max_wall_time_s": value}}
            with self.subTest(value=value), patch("robot_tasks.workflow.load_workflow", return_value=(raw, None, None)):
                if type(value) in (int, float) and value in (0, 60.0):
                    self.assertEqual(load_config(self.config.source).max_wall_time_s, value)
                else:
                    with self.assertRaisesRegex(ValueError, "max_wall_time_s"):
                        load_config(self.config.source)

    def test_task_arm_count_must_match(self):
        bad = replace(self.config, task_id="L2-01", arms=("left",))
        with self.assertRaisesRegex(ValueError, "active arm counts"):
            get_spec(bad)


if __name__ == "__main__":
    unittest.main()
