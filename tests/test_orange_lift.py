"""Goal regressions for the green-box lift and return; no model calls."""

import unittest
from copy import deepcopy

from test_native_l2_scoring import soft_row
from robot_tasks.registry import resolve_task
from robot_tasks.tasks.bench.orange_evaluation import OrangeEvaluator
from robot_tasks.tasks.bench.spec import build_spec
from robot_tasks.types import InvalidSimulationState


class OrangeLiftTests(unittest.TestCase):
    def setUp(self):
        self.spec = build_spec("L2-04")
        self.evaluator = OrangeEvaluator(self.spec)
        self.initial = soft_row(self.spec)
        self.first = self.evaluator.update(0.0, self.initial)

    def sample(self, rise, touching=True):
        rows = deepcopy(self.initial)
        row = rows["orange"]
        for k in (
            "bounds_min",
            "bounds_max",
            "centroid",
            "collision_bounds_min",
            "collision_bounds_max",
        ):
            row[k][2] += rise
        if touching:
            row["finger_distances_m"][2:4] = [0.001, 0.001]
        return rows

    def test_initial_pedestal_height_does_not_count_as_lift(self):
        self.assertEqual(self.first.status, "running")
        self.assertGreater(self.first.metrics["criteria"]["initial_centroid_z_m"], self.spec.parameters["support_z"])
        self.assertEqual(self.first.metrics["criteria"]["centroid_rise_m"], 0)
        self.assertEqual(self.evaluator.update(0.1, self.sample(0)).status, "running")

    def test_observer_fingers_cannot_substitute_for_right_operator(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from robot_tasks.tasks.bench.orange_scene import OrangeScene

        scene = OrangeScene.__new__(OrangeScene)
        scene.spec = self.spec
        scene.config = SimpleNamespace(arms=("right",))
        observer_touch = self.sample(.07, touching=False)
        observer_touch["orange"]["finger_distances_m"] = [.001, .001, .1, .1]
        self.evaluator.update(.1, observer_touch)
        self.assertFalse(self.evaluator.lifted)
        scene.measure = Mock(return_value=observer_touch)
        self.assertEqual(scene.planning_payloads(), {"right": []})

        operator_touch = self.sample(.07)
        self.evaluator.update(.2, operator_touch)
        self.assertTrue(self.evaluator.lifted)
        scene.measure.return_value = operator_touch
        self.assertEqual(set(scene.planning_payloads()), {"right"})
        self.assertEqual(len(scene.planning_payloads()["right"]), 1)

    def test_lift_requires_return_to_box_and_release(self):
        self.assertEqual(self.evaluator.update(.1, self.sample(.049)).status, "running")
        result = self.evaluator.update(.2, self.sample(.05))
        self.assertEqual(result.status, "running")
        self.assertTrue(result.metrics["criteria"]["release_required"])
        self.assertTrue(self.evaluator.lifted)
        self.assertEqual(self.evaluator.update(.3, self.sample(0)).status, "running")
        self.assertEqual(self.evaluator.update(.4, self.sample(0, False)).status, "succeeded")

    def test_flying_or_one_finger_touch_is_not_a_held_lift(self):
        for distances in ([0.1, 0.1], [0.001, 0.1], [0.1, 0.001]):
            row = self.sample(0.07)
            row["orange"]["finger_distances_m"][2:4] = distances
            self.assertEqual(
                self.evaluator.update(self.evaluator.last_time + 0.1, row).status,
                "running",
            )

    def test_release_elsewhere_cannot_replace_return_to_box(self):
        self.assertEqual(self.evaluator.update(.1, self.sample(.06)).status, "running")
        rows = self.sample(0, False)
        rows["orange"]["centroid"][0] += .2
        self.assertEqual(self.evaluator.update(.2, rows).status, "running")
        self.assertEqual(self.evaluator.update(.3, self.sample(0, False)).status, "succeeded")

    def test_speed_deformation_and_local_inversion_are_diagnostic_only(self):
        rows = self.sample(.06)
        rows["orange"].update(speed_m_s=2., centroid_speed_m_s=.2,
            shape_speed_p95_m_s=.5, strain_p95=.8, strain_max=2., minimum_volume_ratio=-.1)
        result = self.evaluator.update(.1, rows)
        self.assertEqual(result.status, "running")
        self.assertTrue(result.metrics["quality_warnings"])
        self.assertEqual(self.evaluator.update(.2, self.sample(0, False)).status, "succeeded")

    def test_remaining_on_support_does_not_count_as_lift(self):
        rows = self.sample(0.06)
        rows["orange"]["collision_bounds_min"][2] = self.spec.parameters["support_z"]
        self.assertEqual(self.evaluator.update(0.1, rows).status, "running")

    def test_fallen_onto_table_can_retry_but_below_table_fails(self):
        self.assertEqual(
            self.evaluator.update(0.1, self.sample(-0.13, False)).status, "running"
        )
        self.assertEqual(
            self.evaluator.update(0.2, self.sample(-0.25, False)).status, "failed"
        )

    def test_pause_does_not_reset_reference_or_create_progress(self):
        self.assertIs(self.evaluator.update(0.0, self.sample(.06)), self.first)
        self.assertFalse(self.evaluator.lifted)
        self.evaluator.update(.1, self.sample(.05))
        self.assertTrue(self.evaluator.lifted)
        self.assertEqual(self.evaluator.update(.2, self.sample(0, False)).status, "succeeded")

    def test_invalid_clock_and_measurements_remain_framework_errors(self):
        for t in (-1.0, float("nan"), float("inf")):
            with self.assertRaises(InvalidSimulationState):
                self.evaluator.update(t, self.initial)
        rows = self.sample(0.06)
        rows["orange"]["centroid"][2] = float("nan")
        with self.assertRaises(InvalidSimulationState):
            self.evaluator.update(0.1, rows)

    def test_task_prompt_physics_and_goal_agree(self):
        entry, cfg = resolve_task("L2-04")
        cfg = entry.factory.prepare_config(cfg)
        self.assertNotIn("orange_physics", cfg.raw)  # Retain latest delivered FEM material.
        from robot_tasks.tasks.bench.arm_selection import resolve_bench_spec
        current = resolve_bench_spec(cfg.raw['task'], cfg.robot)
        self.assertEqual(
            cfg.raw["task"]["scoring_protocol"],
            current.parameters["scoring_protocol"],
        )
        self.assertEqual(cfg.arms, ("right",))
        self.assertEqual(cfg.observation_arms, ("left",))
        self.assertTrue(self.spec.design.release)
        self.assertFalse(self.spec.design.fixtures)
        self.assertIn("5–20 cm", cfg.instruction)
        self.assertIn("without damaging it", cfg.instruction)
        self.assertNotIn("Firm squeezing is allowed", cfg.instruction)
        self.assertAlmostEqual(self.spec.design.articles[0].initial[2], self.spec.parameters["support_z"] + .002)
