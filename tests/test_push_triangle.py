"""Exact triangle containment and push witnesses; no model calls."""

import math
import unittest
import numpy as np
from test_household_tasks import measurement
from robot_tasks.registry import resolve_task
from robot_tasks.robots.transforms import pose_transform
from robot_tasks.tasks.bench.evaluation import BenchEvaluator
from robot_tasks.tasks.bench.push_triangle import build, containment_margin
from robot_tasks.tasks.l1.specs import local_corners, pose


class PushTriangleTests(unittest.TestCase):
    def setUp(self):
        self.s = build()
        self.a = self.s.design.articles[0]
        self.rows = measurement(self.s)
        self.r = self.rows["eraser"]
        self.e = BenchEvaluator(self.s, {self.a.name: self.a.initial})

    def place(self, p):
        self.r["pose"] = list(p)
        t = pose_transform(p)
        self.r["corners"] = (local_corners(self.a) @ t[:3, :3].T + t[:3, 3]).tolist()

    def push(self):
        self.r["fingers"]["right"] = [0.015, 0]
        self.e.update(1, self.rows)
        self.place(pose(self.a.initial[0] + 0.01, *self.a.initial[1:3]))
        self.e.update(2, self.rows)

    def test_small_native_mass_and_three_visual_lines(self):
        self.assertAlmostEqual(self.a.mass * 9.81, .0154 * 9.81)
        self.assertEqual(len(self.s.design.fixtures), 3)
        self.assertTrue(all(not x.collision for x in self.s.design.fixtures))
        b, c = resolve_task("L3-03")
        self.assertEqual(
            b.factory.prepare_config(c).grasp_check_mode, "contact_inspection"
        )

    def test_any_yaw_and_moving_contact_can_complete_without_wait(self):
        self.push()
        self.place(pose(*self.a.goal[:3], yaw=math.pi / 2))
        self.r["linear_velocity"] = [0.04, 0, 0]
        self.r["fingers"]["right"] = [0.1, 0]
        self.assertEqual(self.e.update(3, self.rows).status, "running")
        self.r["fingers"]["right"] = [0, 0]
        self.assertEqual(self.e.update(4, self.rows).status, "succeeded")

    def test_center_inside_but_tip_outside_is_not_success(self):
        self.push()
        self.place(pose(self.a.goal[0], self.a.goal[1] + 0.090, 0.78))
        self.assertLess(containment_margin(self.r["pose"], self.a.goal), 0)
        self.assertEqual(self.e.update(3, self.rows).status, "running")

    def test_no_push_or_floating_block_does_not_complete(self):
        self.place(self.a.goal)
        self.assertEqual(self.e.update(1, self.rows).status, "running")
        self.e.flags.add("gripper_pushed_block")
        self.r["bench_force_n"] = 0
        self.assertEqual(self.e.update(2, self.rows).status, "running")

    def test_grasped_carrying_fails(self):
        self.place(pose(self.a.initial[0], self.a.initial[1], 0.84))
        self.r["fingers"]["right"] = [1, 1]
        self.r["bench_force_n"] = 0
        self.e.update(1, self.rows)
        self.assertEqual(self.e.update(1.3, self.rows).status, "failed")

    def test_rectangle_footprint_fits_centre_but_not_triangle_tip(self):
        self.assertGreater(containment_margin(self.a.goal, self.a.goal), 0)
        p = pose(self.a.goal[0], self.a.goal[1] + .12, .78)
        self.assertLess(containment_margin(p, self.a.goal), 0)
