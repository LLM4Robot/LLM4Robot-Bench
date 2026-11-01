"""Fixed bottle completion accepts cap removal by the correctly mapped arm."""

import unittest
import numpy as np
from robot_tasks.registry import resolve_task
from robot_tasks.robots.transforms import pose_transform
from robot_tasks.tasks.bench.spec import build_spec
from robot_tasks.tasks.bench.evaluation import BenchEvaluator
from robot_tasks.tasks.l1.specs import pose, local_corners
from test_household_tasks import measurement


class BottleTaskTests(unittest.TestCase):
    def setUp(self):
        self.spec = build_spec("L3-01")
        self.rows = measurement(self.spec)
        self.e = BenchEvaluator(
            self.spec, {a.name: a.initial for a in self.spec.design.articles}
        )
        self.cap = self.spec.design.articles[1]

    def place(self, p):
        r = self.rows["cap"]
        r["pose"] = list(p)
        t = pose_transform(p)
        r["corners"] = (local_corners(self.cap) @ t[:3, :3].T + t[:3, 3]).tolist()
        return r

    def test_legacy_fixed_geometry_and_current_bimanual_permissions(self):
        _, c = resolve_task("L3-01")
        self.assertLess(
            np.ptp(local_corners(self.cap)[:, 0]), c.robot.gripper.width_limits_m[1]
        )
        self.assertEqual(c.arms, ("left", "right"))
        self.assertEqual(c.observation_arms, ())
        item = self.spec.parameters["native_instances"][0]
        self.assertIn("Body", item.get("static", []))
        self.assertAlmostEqual(item["initial_body_transforms"]["Cap"][2], 0.0075)
        self.assertEqual(self.e.update(1, self.rows).status, "running")

    def test_detach_completes_while_held_without_placement(self):
        r = self.place(
            pose(self.cap.initial[0], self.cap.initial[1], self.cap.initial[2] + 0.05)
        )
        r["fingers"]["right"] = [1, 1]
        result = self.e.update(1, self.rows)
        self.assertEqual(result.status, "succeeded")
        self.assertFalse(result.metrics["criteria"]["release_required"])
        r = self.place(pose(0.10, 1.25, 0.78 - 0.1742))
        r["fingers"]["right"] = [0, 0]
        r["bench_force_n"] = 0.0035 * 9.81
        r["angular_velocity"] = [0, 0, 2.0]
        self.assertEqual(self.e.update(2, self.rows).status, "succeeded")
        r["bench_force_n"] = 0
        self.assertEqual(self.e.update(3, self.rows).status, "succeeded")

    def test_air_contact_or_no_manipulation_do_not_pass(self):
        r = self.place(pose(0.10, 1.25, 0.78 - 0.1742))
        r["bench_force_n"] = 0.0343
        self.assertEqual(self.e.update(1, self.rows).status, "running")
        self.e.touched.add("cap")
        r["pair_forces"]["bottle"] = 0.02
        self.assertEqual(self.e.update(2, self.rows).status, "running")
        r["pair_forces"]["bottle"] = 0
        r["fingers"]["right"] = [1, 1]
        self.assertEqual(self.e.update(3, self.rows).status, "succeeded")

    def test_fixed_bottle_requires_detachment_but_no_extra_three_cm_lift(self):
        r = self.rows["cap"]
        r["fingers"]["right"] = [1, 1]
        self.assertEqual(self.e.update(1, self.rows).status, "running")
        self.assertNotIn("cap_detached", self.e.flags)
        self.place(pose(self.cap.initial[0], self.cap.initial[1], self.cap.initial[2] + .006))
        result = self.e.update(2, self.rows)
        self.assertEqual(result.status, "succeeded")
        self.assertLess(result.metrics["criteria"]["neck_clearance_m"], .03)
        self.assertIn("cap_detached", self.e.flags)
        r = self.place(pose(.10, 1.25, .78 - .1742))
        r["fingers"]["right"] = [0, 0]
        r["bench_force_n"] = .0343
        self.assertEqual(self.e.update(3, self.rows).status, "succeeded")

    def test_observation_arm_cannot_supply_cap_manipulation_witness(self):
        row = self.place(pose(self.cap.initial[0], self.cap.initial[1], self.cap.initial[2] + .05))
        row["fingers"]["left"] = [1, 1]
        self.assertNotEqual(self.e.update(1, self.rows).status, "succeeded")
        row["fingers"]["left"] = [0, 0]
        row["fingers"]["right"] = [1, 1]
        self.assertEqual(self.e.update(2, self.rows).status, "succeeded")
