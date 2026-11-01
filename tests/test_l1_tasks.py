"""Shared workflow regressions for the maintained L1 task pool."""

import unittest

from robot_tasks.config import ExecutionSettings
from robot_tasks.registry import resolve_task


class L1ProtocolTests(unittest.TestCase):
    def test_all_l1_tasks_use_shared_workflow_and_declared_roles(self):
        _, reference = resolve_task("GPT-PICK-PLACE")
        from robot_tasks.tasks.bench.task import BenchTask
        for name in ("L1-01", "L1-02", "L1-03", "L1-04", "L1-05"):
            spec, config = resolve_task(name)
            self.assertTrue(spec.implemented)
            self.assertEqual(config.robot.source.name, "r5_video_20261015.toml")
            self.assertEqual(config.cameras, reference.cameras)
            arms = ("left", "right") if name == "L1-04" else ("right",)
            observers = tuple(a for a in ("left", "right") if a not in arms)
            self.assertEqual(config.arm_roles, {a: "manipulation" if a in arms else "observation"
                                               for a in ("left", "right")})
            for section in ("observation", "policy", "recording"):
                self.assertEqual(config.raw[section], reference.raw[section], (name, section))
            execution = dict(config.raw["execution"])
            self.assertEqual(execution, reference.raw["execution"])
            self.assertNotIn("policy_options", config.raw)
            self.assertEqual(config.observation_arms, observers)
            self.assertFalse(config.include_depth)
            self.assertEqual(ExecutionSettings.from_config(config).max_decisions, 150)
            self.assertNotIn("reasoning_effort", config.raw["policy"])
            self.assertEqual(config.control_dt_s, reference.control_dt_s)
            self.assertEqual(config.physics_dt_s, reference.physics_dt_s)
            prompt = BenchTask.prepare_config(config).instruction
            self.assertNotIn("metadata", prompt)
            self.assertNotIn("sha256", prompt)
            from robot_tasks.policies.gpt import GPTPolicy
            from robot_tasks.policies.model_config import GPTSettings
            policy = GPTPolicy(BenchTask.prepare_config(config), GPTSettings("https://example.invalid", "offline"), client=object())
            self.assertEqual(policy.run_metadata['arm_roles'], config.arm_roles)
            if observers:
                self.assertIn("Observation arms", policy.instructions)
            self.assertNotIn("keep the other arm still", prompt)
