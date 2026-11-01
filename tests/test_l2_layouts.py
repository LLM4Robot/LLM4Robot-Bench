"""Native L2 placement, contact history and round-trip goals; no model calls."""

import unittest
import numpy as np
from test_household_tasks import measurement
from robot_tasks.registry import resolve_task
from robot_tasks.robots.transforms import pose_transform
from robot_tasks.tasks.bench.evaluation import BenchEvaluator
from robot_tasks.tasks.bench.spec import build_spec


class L2LayoutTests(unittest.TestCase):
    def test_real_roles_map_to_overview_ids_in_config_and_evaluator(self):
        from robot_tasks.policies.operating_notes import operating_notes
        from robot_tasks.tasks.screw.nut import nut_spec
        from robot_tasks.tasks.bench.arm_selection import resolve_bench_spec
        from robot_tasks.tasks.bench.english_reference import apply_reference

        operators = {
            "L2-01": ("left", "right"), "L2-02": ("left", "right"),
            "L2-03": ("right",), "L2-04": ("right",), "L2-05": ("right",),
            "L3-01": ("left", "right"), "L3-02": ("left", "right"),
            "L2-07": ("right",), "L3-03": ("right", "left"), "L3-04": ("right",),
        }
        for task_id, expected in operators.items():
            with self.subTest(task=task_id):
                entry, cfg = resolve_task(task_id)
                prepared = entry.factory.prepare_config(cfg)
                self.assertEqual(prepared.arms, expected)
                self.assertEqual(prepared.observation_arms,
                                 tuple(a for a in ("left", "right") if a not in expected))
                spec = (apply_reference(nut_spec(), cfg.raw['task']) if task_id == "L3-02"
                        else resolve_bench_spec(cfg.raw['task'], cfg.robot))
                self.assertEqual(spec.arms, expected)
                names = operating_notes(prepared)["arm_naming"]
                self.assertIn("real setup's RIGHT", names["simulation_left"])
                self.assertIn("real setup's LEFT", names["simulation_right"])

    def test_marker_cap_and_nut_face_image_right_with_left_body_grasps(self):
        from robot_tasks.tasks.l1.specs import local_corners
        from robot_tasks.tasks.screw.nut import nut_spec

        for spec, key in ((build_spec("L2-01"), "cap_arm"), (nut_spec(), "nut_arm")):
            with self.subTest(task=spec.design.task_id):
                body, end = spec.design.articles
                centers = []
                for article in (body, end):
                    t = pose_transform(article.initial)
                    points = local_corners(article) @ t[:3, :3].T + t[:3, 3]
                    centers.append(points.mean(axis=0))
                self.assertGreater(centers[1][0], centers[0][0])
                np.testing.assert_allclose(pose_transform(body.initial)[:3, 2], [1, 0, 0], atol=1e-12)
                self.assertEqual(spec.parameters["body_arm"], "left")
                self.assertEqual(spec.parameters[key], "right")

    def test_roles_and_native_directions(self):
        for task, roles in (("L2-02", ("left", "right")), ("L2-05", ("right",))):
            with self.subTest(task=task):
                entry, cfg = resolve_task(task)
                entry.factory.prepare_config(cfg)
                self.assertEqual(cfg.arms, roles)
                self.assertEqual(cfg.observation_arms, () if len(roles) == 2 else ("left",))
                self.assertEqual(cfg.raw["simulation"]["environment"], "tuojing_lab_v4")
                self.assertFalse(cfg.raw["completion"]["return_home"])
        book = build_spec("L2-02")
        np.testing.assert_allclose(book.parameters["outward_axis_world"], [1, 0, 0])
        self.assertEqual(book.parameters["manipulation_arm"], "right")
        self.assertEqual(book.parameters["support_arm"], "left")

    def test_native_books_are_dynamic_and_exposed_without_extra_gap(self):
        spec = build_spec("L2-02")
        lower, middle, upper = spec.design.articles
        self.assertAlmostEqual(middle.mass, .3789)
        np.testing.assert_allclose(np.subtract(*middle.asset_bounds[::-1]), [.14, .22, .01958])
        self.assertAlmostEqual(middle.initial[0] - lower.initial[0], .04)
        self.assertLess(upper.initial[2] - middle.initial[2] - .01958, .001)
        items = spec.parameters["native_instances"]
        np.testing.assert_allclose(items[0]["pose"][:2], [.19565, 1.20])
        np.testing.assert_allclose(np.subtract(lower.initial[:2], items[0]["pose"][:2]), [.035, .020])
        self.assertEqual(items[0]["static"], [""])
        self.assertTrue(all(not item.get("static") for item in items[1:]))

    def book_sample(self):
        spec = build_spec("L2-02")
        rows = measurement(spec)
        rows["lower_book"]["fixture_force_n"] = 3.7
        rows["middle_book"]["pair_forces"]["lower_book"] = 3.7
        rows["upper_book"]["pair_forces"]["middle_book"] = 3.7
        return spec, rows, BenchEvaluator(spec, {a.name: a.initial for a in spec.design.articles})

    @staticmethod
    def shift(row, delta):
        row["pose"][:3] = (np.asarray(row["pose"][:3]) + delta).tolist()
        row["corners"] = (np.asarray(row["corners"]) + delta).tolist()

    def test_sideways_or_insufficient_pull_does_not_finish(self):
        for delta in ([0, .15, 0], [.01, 0, 0], [-.05, 0, 0]):
            with self.subTest(delta=delta):
                spec, rows, evaluator = self.book_sample()
                self.shift(rows["middle_book"], delta)
                rows["middle_book"]["fingers"]["right"] = [1, 1]
                self.assertEqual(evaluator.update(.1, rows).status, "running")
                self.assertNotIn("middle_extracted_while_held", evaluator.flags)

    def test_fifty_mm_pull_then_release_is_enough_no_full_extraction(self):
        spec, rows, evaluator = self.book_sample()
        row = rows["middle_book"]
        row["fingers"]["right"] = [1, 1]
        self.shift(row, [.05, 0, 0])
        result = evaluator.update(.1, rows)
        self.assertEqual(result.status, "running")
        self.assertTrue(result.metrics["criteria"]["middle_extracted"])
        row["fingers"]["right"] = [0, 0]
        self.assertEqual(evaluator.update(.2, rows).status, "succeeded")
        self.assertFalse(result.metrics["criteria"]["complete_extraction_required"])
        rows["upper_book"]["pair_forces"]["middle_book"] = 0
        self.assertEqual(evaluator.update(.3, rows).status, "running")

    def test_book_final_pose_without_manipulation_history_cannot_pass(self):
        spec, rows, evaluator = self.book_sample()
        self.shift(rows["middle_book"], [.05, 0, 0])
        self.assertEqual(evaluator.update(.1, rows).status, "running")
        self.shift(rows["middle_book"], [0, 0, -.4])
        self.assertEqual(evaluator.update(.2, rows).status, "failed")

    def test_drawer_native_rail_and_contents_follow_initial_offset(self):
        spec = build_spec("L2-05")
        p = spec.parameters
        item = p["native_instances"][0]
        np.testing.assert_allclose(item["pose"][:2], [.55, 1.12])
        axis = np.asarray(p["outward_axis_world"])
        # Opening must head toward the robot side, not toward the overview camera.
        self.assertGreater(axis[1], 0.0)
        _, cfg = resolve_task("L2-05")
        arm = next(m for m in cfg.robot.mounts if m.name == p["manipulation_arm"])
        towards_arm = np.asarray(arm.base_pose_wxyz[:2]) - np.asarray(p["closed_pose"][:2])
        towards_arm /= np.linalg.norm(towards_arm)
        # Preserve the photographed diagonal layout. Changing the operator must
        # not rotate the drawer to aim exactly at that robot's base.
        np.testing.assert_allclose(axis, [-np.sqrt(3) / 2, .5, 0], atol=1e-12)
        self.assertGreater(float(axis[:2] @ towards_arm), 0.0)
        self.assertAlmostEqual(p["max_travel_m"], .09)
        self.assertLess(p["required_open_m"], p["max_travel_m"])
        np.testing.assert_allclose(-pose_transform(item["pose"])[:3, 1], axis, atol=1e-12)
        np.testing.assert_allclose(np.subtract(spec.design.articles[0].initial[:3], p["closed_pose"][:3]), .03 * axis)
        for part in ["Drawer1"] + [f"Contents/Cube{i:02}" for i in range(1, 6)]:
            np.testing.assert_allclose(item["initial_body_offsets"][part], [0, -.03, 0])
        self.assertFalse(item.get("adapters"))
        self.assertFalse(item.get("mass_additions"))
        self.assertEqual(item["static"], ["Body", "Drawer0"])
        self.assertAlmostEqual(spec.design.articles[0].mass, .080)
