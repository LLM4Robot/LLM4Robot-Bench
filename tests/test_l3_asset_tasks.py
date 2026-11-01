"""Contact mechanism and final-state regressions; no simulator or paid models."""

import math
import unittest
from copy import deepcopy
from dataclasses import replace

import numpy as np

from robot_tasks.registry import resolve_task
from robot_tasks.robots.transforms import pose_transform
from robot_tasks.tasks.bench.evaluation import BenchEvaluator
from robot_tasks.tasks.bench.spec import build_spec
from robot_tasks.tasks.bench.wiping import WipeLedger
from robot_tasks.tasks.l1.specs import pose
from test_household_tasks import measurement


class CleaningTests(unittest.TestCase):
    def setUp(self):
        self.points = np.array([[0, 0, 0.0102], [0.008, 0, 0.0102]])
        self.ledger = WipeLedger(pose(0, 0, 0), self.points)

    def sweep(self, *, height=0.028, roll=math.pi, contact=0.2, held=True):
        for i in range(10):
            self.ledger.update(
                i / 12,
                pose(-0.004 + i * 0.001, 0, height, roll=roll),
                pad_contact_n=contact,
                held=held,
            )

    def test_actual_pad_sliding_removes_visible_cells(self):
        self.sweep()
        self.assertEqual(self.ledger.metrics()["remaining_cells"], 0)

    def test_hover_wrong_face_unheld_and_stationary_do_not_clean(self):
        for kwargs in ({"height": 0.030}, {"roll": 0}, {"contact": 0}, {"held": False}):
            with self.subTest(kwargs=kwargs):
                self.ledger = WipeLedger(pose(0, 0, 0), self.points)
                self.sweep(**kwargs)
                self.assertEqual(self.ledger.metrics()["remaining_cells"], 2)
        self.ledger = WipeLedger(pose(0, 0, 0), self.points)
        for i in range(10):
            self.ledger.update(
                i, pose(0, 0, 0.028, roll=math.pi), pad_contact_n=0.2, held=True
            )
        self.assertEqual(self.ledger.metrics()["remaining_cells"], 2)

    def test_repeated_frame_cannot_manufacture_wiping(self):
        self.ledger.update(
            1, pose(0, 0, 0.028, roll=math.pi), pad_contact_n=0.2, held=True
        )
        self.ledger.update(
            1, pose(0.01, 0, 0.028, roll=math.pi), pad_contact_n=0.2, held=True
        )
        self.assertEqual(self.ledger.metrics()["remaining_cells"], 2)

    def test_one_removed_cell_succeeds_while_eraser_is_still_held(self):
        spec = build_spec("L2-07")
        rows = measurement(spec)
        e = BenchEvaluator(spec, {a.name: a.initial for a in spec.design.articles})
        rows["eraser"]["fingers"]["right"] = [1, 1]
        ledger = WipeLedger(pose(0, 0, 0), self.points, half_extents=(.003, .003))
        rows["eraser"]["wiping"] = ledger.metrics()
        self.assertEqual(e.update(1, rows).status, "running")
        for i in range(2):
            ledger.update(i, pose(i * .001, 0, .028, roll=math.pi),
                          pad_contact_n=.2, held=True)
        self.assertEqual(ledger.metrics()["remaining_cells"], 1)
        rows["eraser"]["wiping"] = ledger.metrics()
        rows["eraser"]["bench_force_n"] = 0
        result = e.update(2, rows)
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.metrics["criteria"]["removed_cells"], 1)
        self.assertFalse(result.metrics["criteria"]["release_required"])


class SweepingTests(unittest.TestCase):
    def setUp(self):
        self.spec = build_spec("L3-04")
        self.rows = measurement(self.spec)
        self.e = BenchEvaluator(
            self.spec, {a.name: a.initial for a in self.spec.design.articles}
        )
        self.e.flags.update(("swept_block_0",))

    def collect(self, i, center):
        pan = pose_transform(self.rows["dustpan"]["pose"])
        local = np.array(
            [[x, y, z] for x in (-0.01, 0.01) for y in (-0.01, 0.01) for z in (0, 0.01)]
        )
        local += np.asarray(center)
        row = self.rows[f"block_{i}"]
        row["corners"] = (local @ pan[:3, :3].T + pan[:3, 3]).tolist()
        row["pose"][:3] = (np.asarray(center) @ pan[:3, :3].T + pan[:3, 3]).tolist()
        row["pair_forces"]["dustpan"] = 0.02943
        row["bench_force_n"] = 0

    def all_inside(self):
        for i, c in enumerate(((-0.025, 0, 0.0012), (0.025, 0, 0.0012))):
            self.collect(i, c)

    def test_both_supported_finish_without_tool_holding_or_wait(self):
        self.all_inside()
        self.rows["dustpan"]["angular_velocity"] = [0, 0, 1]
        result = self.e.update(1, self.rows)
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.metrics["criteria"]["required"], 2)
        self.assertEqual(result.metrics["criteria"]["blocks_in_pan"], 2)

    def test_handle_air_outer_lip_and_missing_support_do_not_count(self):
        for c in (
            (0, -0.107, 0.0012),
            (0, 0, 0.05),
            (0.07, 0, 0.0012),
            (0, 0.041, 0.0012),
        ):
            with self.subTest(center=c):
                self.all_inside()
                self.collect(1, c)
                self.assertEqual(
                    self.e.update(
                        self.e.last_time + 1 if self.e.last_time else 1, self.rows
                    ).status,
                    "running",
                )
        self.all_inside()
        self.rows["block_1"]["pair_forces"]["dustpan"] = 0
        self.assertEqual(self.e.update(10, self.rows).status, "running")

    def test_prefilled_positions_without_tool_use_do_not_pass(self):
        self.e.flags.clear()
        self.all_inside()
        self.assertEqual(self.e.update(1, self.rows).status, "running")

    def test_only_manipulation_hand_brush_use_qualifies(self):
        self.e.flags.clear()
        broom = self.rows["broom"]
        broom["pair_forces"]["block_0"] = .02
        broom["fingers"]["left"] = [1, 1]
        self.e.update(1, self.rows)
        self.assertNotIn("swept_block_0", self.e.flags)
        broom["fingers"]["left"] = [0, 0]
        broom["fingers"]["right"] = [1, 1]
        self.e.update(2, self.rows)
        self.assertIn("swept_block_0", self.e.flags)


class PhotoLayoutTests(unittest.TestCase):
    def screen(self, cfg, point):
        from robot_tasks.backends.settings import SimulationSettings

        settings = SimulationSettings.from_config(cfg)
        eye = np.asarray(settings.camera_position)
        forward = np.asarray(settings.camera_target) - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0, 0, 1])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        p = np.asarray(point) - eye
        return np.array([p @ right, p @ up]) / (p @ forward)

    def test_current_l314_has_two_stacked_books_and_two_active_arms(self):
        from robot_tasks.tasks.bench.arm_selection import resolve_bench_spec

        entry, cfg = resolve_task("L3-03")
        entry.factory.prepare_config(cfg)
        spec = resolve_bench_spec(cfg.raw["task"], cfg.robot)
        articles = {a.name: a for a in spec.design.articles}
        self.assertEqual(set(articles), {"lower_book", "upper_book"})
        self.assertGreater(articles["upper_book"].initial[2], articles["lower_book"].initial[2])
        self.assertEqual(cfg.arms, ("right", "left"))
        self.assertEqual(cfg.observation_arms, ())
        self.assertEqual(spec.parameters["boundary_reference"], "both_books")

    def test_pan_mouth_faces_two_blocks_and_brush_on_image_right(self):
        entry, cfg = resolve_task("L3-04")
        entry.factory.prepare_config(cfg)
        spec = build_spec(cfg.task_id)
        articles = {a.name: a for a in spec.design.articles}
        self.assertEqual(set(articles), {"dustpan", "broom", "block_0", "block_1"})
        pan, brush = articles["dustpan"], articles["broom"]
        pan_x, brush_x = (
            self.screen(cfg, pan.initial[:3])[0],
            self.screen(cfg, brush.initial[:3])[0],
        )
        for i in range(2):
            x = self.screen(cfg, articles[f"block_{i}"].initial[:3])[0]
            self.assertLess(pan_x, x)
            self.assertLess(x, brush_x)
        self.assertGreater(pose_transform(pan.initial)[0, 1], 0.99)
        self.assertLess(pose_transform(brush.initial)[2, 1], -.9)  # Bristles down, handle up.
        self.assertEqual(cfg.arms, ("right",))
        self.assertEqual(cfg.observation_arms, ("left",))

    def test_ink_rotation_keeps_face_up_and_contact_map_on_visible_board(self):
        from robot_tasks.tasks.bench.wiping import INK_TEXT, INK_COLOR, ink_centers

        entry, cfg = resolve_task("L2-07")
        entry.factory.prepare_config(cfg)
        spec = build_spec(cfg.task_id)
        self.assertEqual(cfg.arms, ("right",))
        self.assertEqual(cfg.observation_arms, ("left",))
        self.assertEqual(INK_TEXT, "GPT6")
        self.assertLess(max(INK_COLOR), 0.05)
        board = spec.parameters["board_pose"]
        t = pose_transform(board)
        np.testing.assert_allclose(t[:3, 2], [0, 0, 1], atol=1e-12)
        self.assertLess(
            self.screen(cfg, t[:3, 3] + t[:3, 1] * 0.02)[1],
            self.screen(cfg, board[:3])[1],
        )
        self.assertGreater(
            self.screen(cfg, spec.design.articles[0].initial[:3])[0],
            self.screen(cfg, board[:3])[0],
        )
        # Erase a rotated board-local ink cell using the matching world pose.
        ledger = WipeLedger(board)
        target = ledger.world[0]
        for i in range(2):
            eraser_pose = pose(
                target[0] + i * 0.001, target[1], board[2] + 0.028, roll=math.pi
            )
            ledger.update(i, eraser_pose, pad_contact_n=0.2, held=True)
        self.assertFalse(ledger.dirty[0])
