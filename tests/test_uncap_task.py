"""Workflow and lifecycle checks for the current native marker task."""

import unittest
from dataclasses import replace

from robot_tasks.policies.operating_notes import operating_notes
from robot_tasks.registry import resolve_task
from robot_tasks.robots.preparation import preparation_action
from robot_tasks.tasks.bench.task import BenchTask


class UncapWorkflowTests(unittest.TestCase):
    def test_configuration_is_two_manipulation_arms_with_shared_workflow(self):
        spec, cfg = resolve_task("L2-01")
        cfg = spec.factory.prepare_config(cfg)
        self.assertEqual(cfg.arms, ("left", "right"))
        self.assertEqual(cfg.observation_arms, ())
        self.assertEqual(cfg.grasp_check_mode, "contact_inspection")
        self.assertEqual(cfg.raw["execution"]["max_decisions"], 150)
        self.assertNotIn("reasoning_effort", cfg.raw["policy"])
        self.assertIn(
            "Both hands start empty", operating_notes(cfg)["startup_preparation"]
        )
        self.assertNotIn(
            "during a short lift", operating_notes(cfg)["manipulation_guidance"]
        )
        self.assertEqual(preparation_action(cfg).commands, ())
        with self.assertRaises(ValueError):
            BenchTask.prepare_config(replace(cfg, arms=("left",)))

    def test_policy_guidance_does_not_require_free_vertical_lift(self):
        from robot_tasks.policies.gpt import GPTPolicy
        from robot_tasks.policies.model_config import GPTSettings

        _, cfg = resolve_task("L2-01")
        policy = GPTPolicy(
            BenchTask.prepare_config(cfg),
            GPTSettings("https://example.org", "fixture-model"),
        )
        self.assertIn(
            "free-object vertical test-lift guard is NOT applicable",
            policy.instructions,
        )
        self.assertNotIn("Until status=verified", policy.instructions)
        self.assertNotIn("Observation arms (", policy.instructions)

    def test_optional_task_initialization_hook_precedes_policy_observation(self):
        import tempfile
        from pathlib import Path

        from fixtures import rgbd

        from robot_tasks.backends.mock import MockBackend, MockRobot, MockTask
        from robot_tasks.runner import run_episode
        from robot_tasks.types import Action

        order = []

        class Backend(MockBackend):
            def capture_rgb(self):
                return tuple(rgbd(name, self.t) for name in self.config.cameras)

        class Task(MockTask):
            def initialize_episode(self, backend, robot, tick):
                order.append("initialize")
                tick(phase="task_initialization", score=False)

            def evaluate(self, state, feedback):
                assert order and order[0] == "initialize"
                return super().evaluate(state, feedback)

        class Policy:
            def reset(self, seed):
                pass

            def act(self, context):
                order.append("policy")
                return Action((), cfg.control_dt_s)

        _, cfg = resolve_task("L2-01")
        cfg = BenchTask.prepare_config(cfg)
        cfg = replace(cfg, raw={**cfg.raw, "execution": {"max_decisions": 1}})
        with tempfile.TemporaryDirectory() as directory:
            _, result = run_episode(
                cfg, Backend(), MockRobot(), Task(), Policy(), Path(directory)
            )
        self.assertEqual(order, ["initialize", "policy"])
        self.assertIsNone(result["error"])
        self.assertEqual(result["grasp_check_protocol"], "contact_inspection")
