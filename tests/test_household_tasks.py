"""Offline geometry/process regressions; no Isaac startup or live GPT calls."""

import json
import unittest
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np

from robot_tasks.registry import TASKS, resolve_task
from robot_tasks.robots.transforms import pose_transform
from robot_tasks.tasks.bench import l1, l2, l3
from robot_tasks.types import InvalidSimulationState
from robot_tasks.tasks.bench.evaluation import BenchEvaluator
from robot_tasks.tasks.bench.spec import build_spec
from robot_tasks.tasks.l1.specs import local_corners

IDS = tuple(l1.BUILDERS | l2.BUILDERS | l3.BUILDERS)
ROOT = Path(__file__).resolve().parents[1]


def measurement(spec):
    result = {}
    for a in spec.design.articles:
        t = pose_transform(a.initial)
        corners = local_corners(a) @ t[:3, :3].T + t[:3, 3]
        result[a.name] = {
            "pose": list(a.initial),
            "corners": corners.tolist(),
            "linear_velocity": [0.0, 0.0, 0.0],
            "angular_velocity": [0.0, 0.0, 0.0],
            "finger_forces": [0.0, 0.0],
            "fingers": {"left": [0.0, 0.0], "right": [0.0, 0.0]},
            "fixture_force_n": 0.0,
            "bench_force_n": a.mass * 9.81 if corners[:, 2].min() < spec.parameters.get("table_z", 0.0) + 0.006 else 0.0,
            "pair_forces": {
                b.name: 0.0 for b in spec.design.articles if b.name != a.name and (not a.name.startswith("load_") or b.name in spec.parameters.get("load_container_names", ()))
            },
        }
    if spec.design.task_id == "L2-07":
        from robot_tasks.tasks.bench.wiping import WipeLedger
        result["eraser"]["wiping"] = WipeLedger(spec.parameters["board_pose"]).metrics()
    return result


class HouseholdTests(unittest.TestCase):
    def test_all_maintained_tasks_have_launchable_bindings_and_roles(self):
        tasks = [
            s for s in TASKS.values() if s.task_id.startswith("L") and s.implemented
        ]
        self.assertEqual(len(tasks), 32)
        for spec in tasks:
            with self.subTest(task=spec.task_id):
                _, cfg = resolve_task(spec.task_id)
                prepared = spec.factory.prepare_config(cfg)
                self.assertEqual(
                    cfg.raw["task"]["launch_command"],
                    f"python tools/launch.py run {spec.task_id} --gui",
                )
                self.assertEqual(cfg.raw["execution"]["max_decisions"], 150)
                self.assertNotIn("reasoning_effort", cfg.raw["policy"])
                self.assertEqual(
                    cfg.observation_arms,
                    tuple(a for a in ("left", "right") if a not in cfg.arms),
                )
                self.assertFalse(prepared.include_depth)
                self.assertTrue(prepared.raw["task"]["description"])
        self.assertTrue(resolve_task("L2-07")[0].implemented)
        self.assertNotIn("L3-16", TASKS)
        with self.assertRaises(ValueError):
            resolve_task("L3-16")
        # The old key task was removed. Its former number now belongs to the
        # flower task in the sequential catalogue; task identity is the check.
        self.assertIn("rose", {a.name for a in build_spec("L1-05").design.articles})
        self.assertFalse(any(a.name == "key" for i in IDS
                             for a in build_spec(i).design.articles))
        for removed in ("L1-99", "L1-99_v3"):
            self.assertNotIn(removed, TASKS)
            with self.assertRaisesRegex(ValueError, "Unknown task ID"):
                resolve_task(removed)
            with self.assertRaisesRegex(ValueError, "No local scene implementation"):
                build_spec(removed)

    def test_every_initial_state_is_incomplete_without_robot_evidence(self):
        # FEM tasks use OrangeEvaluator and have dedicated initial-state tests.
        for task_id in (i for i in IDS if i not in {"L2-04", "L2-06"}):
            with self.subTest(task=task_id):
                spec = build_spec(task_id)
                json.dumps(asdict(spec), allow_nan=False)
                rows = measurement(spec)
                evaluator = BenchEvaluator(
                    spec, {a.name: a.initial for a in spec.design.articles}
                )
                for i in range(1, 30):
                    assessment = evaluator.update(i * 0.1, rows)
                    self.assertEqual(assessment.status, "running")
                    json.dumps(asdict(assessment), allow_nan=False)

    def test_marker_entry_needs_no_release_support_history_or_wait(self):
        spec = build_spec("L1-01")
        rows = measurement(spec)
        row = rows["marker"]
        # A shallow entry while held and moving; no cup contact or bottom seating.
        row["pose"][:3] = [*spec.parameters["holder_xy"], spec.parameters["rim"] - .005]
        row["bench_force_n"] = row["fixture_force_n"] = 0.0
        row["fingers"]["left"] = [1, 1]
        row["linear_velocity"], row["angular_velocity"] = [.04, 0, 0], [.5, 0, 10]
        evaluator = BenchEvaluator(spec, {a.name: a.initial for a in spec.design.articles})
        result = evaluator.update(.1, rows)
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.metrics["grasp_lift_confirmed"], [])
        self.assertEqual(result.metrics["stable_duration_s"], 0)
        self.assertEqual(result.metrics["scoring_protocol"], "l1_entry_only_v1")
        self.assertFalse(result.metrics["criteria"]["released"])
        self.assertFalse(result.metrics["criteria"]["fixture_supported"])
        row["fingers"]["left"] = [0, 0]
        self.assertEqual(evaluator.update(.2, rows).status, "succeeded")

    def test_marker_released_drop_succeeds_only_when_it_reaches_the_cavity(self):
        spec = build_spec("L1-01")
        for offset, expected in ((0.0, "succeeded"), (.06, "running")):
            with self.subTest(horizontal_offset=offset):
                rows = measurement(spec)
                row = rows["marker"]
                row["pose"][:3] = [spec.parameters["holder_xy"][0] + offset,
                                   spec.parameters["holder_xy"][1], spec.parameters["rim"] + .02]
                row["bench_force_n"] = row["fixture_force_n"] = 0.0
                row["fingers"]["left"] = [1, 1]
                e = BenchEvaluator(spec, {a.name: a.initial for a in spec.design.articles})
                self.assertEqual(e.update(.1, rows).status, "running")
                row["fingers"]["left"] = [0, 0]
                row["linear_velocity"] = [0, 0, -.3]
                self.assertEqual(e.update(.2, rows).status, "running")
                row["pose"][2] = spec.parameters["rim"] - .005
                result = e.update(.3, rows)
                self.assertEqual(result.status, expected)
                self.assertTrue(result.metrics["criteria"]["released"])
                self.assertFalse(result.metrics["criteria"]["fixture_supported"])

    def test_marker_entry_rejects_rim_hover_handle_and_wall_or_floor_penetration(self):
        spec = build_spec("L1-01")
        p = spec.parameters
        cx, cy = p["holder_xy"]
        for xyz in ((cx, cy, p["rim"] + .005), (cx, cy, p["rim"]),
                    (cx, cy - .06, p["floor"]),
                    (cx + .05, cy, p["rim"] - .01),
                    (cx, cy, p["floor"] - .005)):
            with self.subTest(xyz=xyz):
                rows = measurement(spec)
                rows["marker"]["pose"][:3] = xyz
                rows["marker"]["bench_force_n"] = 0
                rows["marker"]["fixture_force_n"] = .22
                e = BenchEvaluator(spec, {a.name: a.initial for a in spec.design.articles})
                self.assertEqual(e.update(.1, rows).status, "running")

    def test_nested_cups_require_support_and_lift_only_for_upper_cup(self):
        spec = build_spec("L1-04")
        rows = measurement(spec)
        evaluator = BenchEvaluator(
            spec, {a.name: a.initial for a in spec.design.articles}
        )
        top, base = rows["upper_cup"], rows["lower_cup"]
        top["pose"][2] += 0.04
        top["fingers"]["left"] = [1, 1]
        for i in range(1, 5):
            evaluator.update(i * 0.1, rows)
        top["pose"][:3] = list(spec.design.articles[-1].goal[:3])
        top["fingers"]["left"] = [0, 0]
        top["pair_forces"]["lower_cup"] = 0.45
        base["bench_force_n"] = 1.5
        for i in range(5, 20):
            result = evaluator.update(i * 0.1, rows)
        self.assertEqual(result.status, "succeeded")
        top["pair_forces"]["lower_cup"] = 0
        self.assertEqual(evaluator.update(2.0, rows).status, "running")

    def test_native_marker_cup_rejects_penetration_but_has_no_40mm_depth_gate(self):
        from robot_tasks.tasks.bench.scoring_l1 import marker

        spec = build_spec("L1-01")
        article = spec.design.articles[0]
        corners = local_corners(article)
        self.assertAlmostEqual(corners[:, 2].min(), 0.)
        self.assertAlmostEqual(corners[:, 2].max(), .1512)
        evaluator = BenchEvaluator(spec, {"marker": article.initial})
        evaluator.grasped.add("marker")  # Scorer fixture, not robot execution evidence.
        rows = measurement(spec)
        row = rows["marker"]
        row["pose"] = list(article.goal)
        row["bench_force_n"] = 0.
        row["fixture_force_n"] = article.mass * 9.81
        self.assertTrue(marker(evaluator, rows)[0])
        # An offset that fits the broad mouth still penetrates the narrower floor wall.
        row["pose"][0] += .025
        self.assertFalse(marker(evaluator, rows)[1]["inside_holder"])
        row["pose"] = list(article.goal)
        row["pose"][2] = spec.parameters["rim"] - .02
        self.assertTrue(marker(evaluator, rows)[1]["inside_holder"])
        row["pose"][2] = spec.parameters["rim"]
        self.assertFalse(marker(evaluator, rows)[1]["inside_holder"])
        row["pose"] = list(article.goal)
        row["pose"][2] -= .005
        self.assertFalse(marker(evaluator, rows)[1]["inside_holder"])

    def test_marker_motion_and_finger_contact_do_not_override_valid_entry(self):
        spec = build_spec("L1-01")
        article = spec.design.articles[0]
        evaluator = BenchEvaluator(spec, {"marker": article.initial})
        rows = measurement(spec); row = rows["marker"]
        row["pose"] = list(article.goal)
        row["bench_force_n"] = 0.
        row["fixture_force_n"] = article.mass * 9.81
        row["angular_velocity"] = [10.,0.,0.]
        for now in (.1,1.2):
            self.assertEqual(evaluator.update(now,rows).status,"succeeded")
        row["angular_velocity"] = [0.,0.,10.]
        row["fingers"]["left"] = [1.,1.]
        self.assertEqual(evaluator.update(1.3,rows).status,"succeeded")
        row["fingers"]["left"] = [0.,0.]
        self.assertEqual(evaluator.update(1.4,rows).status,"succeeded")
        self.assertEqual(evaluator.update(2.5,rows).status,"succeeded")

    def test_lab_floor_failure_uses_raised_table_not_world_zero(self):
        spec = build_spec("L1-01")
        evaluator = BenchEvaluator(spec, {a.name: a.initial for a in spec.design.articles})
        rows = measurement(spec)
        rows["marker"]["pose"][2] -= .1
        rows["marker"]["corners"] = (np.asarray(rows["marker"]["corners"]) - [0, 0, .1]).tolist()
        result = evaluator.update(.1, rows)
        self.assertEqual(result.status, "failed")
        self.assertIn("fell below the table", result.reason)

    def test_no_l3_shortcut_by_final_positions_without_process(self):
        for task_id in l3.BUILDERS:
            with self.subTest(task=task_id):
                spec = build_spec(task_id)
                rows = measurement(spec)
                evaluator = BenchEvaluator(
                    spec, {a.name: a.initial for a in spec.design.articles}
                )
                for a in spec.design.articles:
                    rows[a.name]["pose"] = list(a.goal)
                if task_id == "L3-01":
                    rows["cap"]["pose"] = [0.50, 0, 0.03, 1, 0, 0, 0]
                    rows["cap"]["bench_force_n"] = 0.25
                for i in range(1, 30):
                    self.assertNotEqual(
                        evaluator.update(i * 0.1, rows).status, "succeeded"
                    )


    def test_drawer_speed_warning_does_not_replace_rail_and_opening_checks(self):
        spec = build_spec("L2-05")
        rows = measurement(spec)
        row = rows["drawer"]
        e = BenchEvaluator(spec, {a.name: a.initial for a in spec.design.articles})
        row["pose"] = list(spec.parameters["closed_pose"])
        row["linear_velocity"] = [.5, 0, 0]
        result = e.update(.1, rows)
        self.assertEqual(result.status, "running")
        self.assertTrue(result.metrics["quality_warnings"])
        row["fingers"]["right"] = [1, 1]
        e.update(.15, rows)
        row["pose"][:3] = (np.asarray(row["pose"][:3])
                              + .08 * np.asarray(spec.parameters["outward_axis_world"])).tolist()
        row["linear_velocity"] = [0, 0, 0]
        row["fingers"]["right"] = [1, 1]
        e.update(.2, rows)
        row["fingers"]["right"] = [0, 0]
        row["pose"] = list(spec.design.articles[0].initial)
        e.update(.5, rows)
        result = e.update(1.4, rows)
        self.assertEqual(result.status, "succeeded")
        self.assertTrue(result.metrics["quality_warnings"])
        # An unbreakable rail leaving its axis is invalid physics, not modeled damage.
        rows = measurement(spec)
        e = BenchEvaluator(spec, {a.name: a.initial for a in spec.design.articles})
        rows["drawer"]["pose"][0] += .02
        self.assertEqual(e.update(.1, rows).status, "running")
        self.assertEqual(e.update(.4, rows).status, "invalid")

    def test_sweeping_allows_incidental_touch_but_not_direct_block_manipulation(self):
        spec = build_spec("L3-04")
        rows = measurement(spec)
        e = BenchEvaluator(spec, {a.name: a.initial for a in spec.design.articles})
        block = rows["block_0"]
        block["fingers"]["right"] = [1, 0]
        result = e.update(.1, rows)
        self.assertEqual(result.status, "running")
        self.assertTrue(result.metrics["quality_warnings"])
        block["fingers"]["right"] = [0, 0]
        self.assertEqual(e.update(.2, rows).status, "running")
        block["fingers"]["right"] = [1, 0]
        e.update(.3, rows)
        block["pose"][0] += .01
        result = e.update(.5, rows)
        self.assertEqual(result.status, "failed")
        self.assertIn("manipulated directly", result.reason)

    def test_geometry_and_passive_joints_are_finite_and_self_contained(self):
        for task_id in IDS:
            with self.subTest(task=task_id):
                spec = build_spec(task_id)
                names = {a.name for a in spec.design.articles}
                self.assertEqual(len(names), len(spec.design.articles))
                for a in spec.design.articles:
                    self.assertGreater(a.mass, 0)
                    self.assertTrue(np.isfinite(local_corners(a)).all())
                for joint in spec.joints:
                    self.assertIn(joint.body1, names)
                    self.assertTrue(not joint.body0 or joint.body0 in names)
                    for lower, upper, k, d in joint.axes.values():
                        self.assertLessEqual(lower, upper)
                        self.assertGreaterEqual(k, 0)
                        self.assertGreaterEqual(d, 0)

    def test_scoring_rejects_nonfinite_data_and_stale_time(self):
        spec = build_spec("L1-01")
        rows = measurement(spec)
        evaluator = BenchEvaluator(
            spec, {a.name: a.initial for a in spec.design.articles}
        )
        evaluator.update(0.1, rows)
        with self.assertRaises(InvalidSimulationState):
            evaluator.update(0.1, rows)
        bad = deepcopy(rows)
        bad["marker"]["pose"][0] = float("nan")
        with self.assertRaises(InvalidSimulationState):
            evaluator.update(0.2, bad)

    def test_drawer_requires_grasped_opening_then_return_and_release(self):
        spec = build_spec("L2-05")
        rows = measurement(spec)
        row = rows["drawer"]
        initial = {a.name: a.initial for a in spec.design.articles}
        e = BenchEvaluator(spec, initial)
        self.assertEqual(e.update(.1, rows).status, "running")
        row["fingers"]["right"] = [1, 1]
        e.update(.2, rows)
        row["pose"][:3] = (np.asarray(row["pose"][:3]) + .05 * np.asarray(spec.parameters["outward_axis_world"])).tolist()
        self.assertEqual(e.update(.3, rows).status, "running")
        row["fingers"]["right"] = [0, 0]
        self.assertEqual(e.update(.4, rows).status, "running")
        row["pose"] = list(spec.design.articles[0].initial)
        self.assertEqual(e.update(.5, rows).status, "succeeded")
        self.assertEqual(BenchEvaluator(spec, initial).update(.1, rows).status, "running")

    def test_books_retain_dynamic_neighbours_and_require_released_support(self):
        spec = build_spec("L2-02")
        self.assertEqual([a.name for a in spec.design.articles], ["lower_book", "middle_book", "upper_book"])
        rows = measurement(spec)
        e = BenchEvaluator(spec, {a.name: a.initial for a in spec.design.articles})
        row = rows["middle_book"]
        row["pose"][0] += .05
        row["corners"] = (np.asarray(row["corners"]) + [.05, 0, 0]).tolist()
        self.assertEqual(e.update(.1, rows).status, "running")
        row["fingers"]["right"] = [1, 1]
        self.assertEqual(e.update(.2, rows).status, "running")
        for r in rows.values():
            r["fixture_force_n"] = 3.7
        row["fingers"]["right"] = [0, 0]
        result = e.update(.3, rows)
        self.assertEqual(result.status, "succeeded")
        self.assertTrue(result.metrics["criteria"]["release_required"])

    def test_books_require_box_anchored_support_not_airborne_contact_or_table_drop(self):
        spec = build_spec("L2-02")
        rows = measurement(spec)
        e = BenchEvaluator(spec, {a.name: a.initial for a in spec.design.articles})
        middle = rows["middle_book"]
        middle["pose"][0] += .02
        middle["corners"] = (np.asarray(middle["corners"]) + [.02, 0, 0]).tolist()
        middle["fingers"]["right"] = [1, 1]
        e.update(.1, rows)
        middle["fingers"]["right"] = [0, 0]
        middle["pair_forces"]["lower_book"] = 7.4
        rows["upper_book"]["pair_forces"]["middle_book"] = 3.7
        self.assertEqual(e.update(.2, rows).status, "running")
        rows["lower_book"]["fixture_force_n"] = 11.1
        self.assertEqual(e.update(.3, rows).status, "succeeded")
        # A later drop of the neighbours to the table loses the support goal.
        for name in ("lower_book", "upper_book"):
            rows[name]["fixture_force_n"] = 0
            rows[name]["bench_force_n"] = 3.7
            rows[name]["corners"] = (np.asarray(rows[name]["corners"]) - [0, 0, .1]).tolist()
        self.assertEqual(e.update(.4, rows).status, "running")


    def test_marker_airborne_tilt_never_fails_by_angle_alone(self):
        spec = build_spec("L1-01")
        for angle in (0, 30, 75, 90, 180):
            with self.subTest(angle=angle):
                rows = measurement(spec)
                row = rows["marker"]
                row["pose"][2] += .3
                half_angle = np.deg2rad(angle) / 2
                row["pose"][3:] = [np.cos(half_angle), 0, np.sin(half_angle), 0]
                row["bench_force_n"] = 0
                e = BenchEvaluator(spec, {a.name: a.initial for a in spec.design.articles})
                for now in (.1, .5, 1.):
                    self.assertEqual(e.update(now, rows).status, "running")

    def test_marker_table_landing_is_recoverable_and_does_not_block_later_entry(self):
        spec = build_spec("L1-01")
        rows = measurement(spec)
        e = BenchEvaluator(spec, {a.name: a.initial for a in spec.design.articles})
        self.assertEqual(e.update(.1, rows).status, "running")
        row = rows["marker"]
        row["pose"][2] += .05
        row["bench_force_n"] = 0
        self.assertEqual(e.update(.2, rows).status, "running")
        row["pose"] = list(spec.design.articles[0].initial)
        row["bench_force_n"] = .22
        result = e.update(.3, rows)
        self.assertEqual(result.status, "running")
        self.assertTrue(result.metrics["criteria"]["dropped_on_table"])
        self.assertTrue(result.metrics["quality_warnings"])
        row["pose"] = list(spec.design.articles[0].goal)
        row["bench_force_n"] = 0
        self.assertEqual(e.update(.4, rows).status, "succeeded")

    def test_marker_knocked_over_on_table_is_recoverable_before_any_lift(self):
        spec = build_spec("L1-01")
        rows = measurement(spec)
        e = BenchEvaluator(spec, {a.name: a.initial for a in spec.design.articles})
        self.assertEqual(e.update(.1, rows).status, "running")
        row = rows["marker"]
        row["pose"][2] = spec.parameters["table_z"] + spec.parameters["marker_visual_radius"]
        row["pose"][3:] = [2**-.5, 0, 2**-.5, 0]
        result = e.update(.2, rows)
        self.assertEqual(result.status, "running")
        self.assertTrue(result.metrics["criteria"]["dropped_on_table"])

    def test_marker_can_settle_leaning_or_cap_first_inside_cup(self):
        spec = build_spec("L1-01")
        p = spec.parameters
        for angle in (20, 180):
            with self.subTest(angle=angle):
                rows = measurement(spec)
                row = rows["marker"]
                row["pose"] = list(spec.design.articles[0].goal)
                half_angle = np.deg2rad(angle) / 2
                row["pose"][3:] = [np.cos(half_angle), 0, np.sin(half_angle), 0]
                if angle == 20:
                    row["pose"][0] -= .012
                    row["pose"][2] += p["marker_visual_radius"] * np.sin(np.deg2rad(angle)) + .001
                else:
                    row["pose"][2] += p["marker_height"]
                row["bench_force_n"] = 0
                row["fixture_force_n"] = .22
                e = BenchEvaluator(spec, {a.name: a.initial for a in spec.design.articles})
                for now in (.1, .5, 1.2):
                    result = e.update(now, rows)
                self.assertEqual(result.status, "succeeded", result.metrics["criteria"])

    def test_native_hat_succeeds_on_selected_peg_release_without_sway_wait(self):
        spec=build_spec("L1-03"); article=spec.design.articles[0]
        rows=measurement(spec); e=BenchEvaluator(spec,{article.name:article.initial})
        self.assertEqual(e.update(.1,rows).status,"running")
        r=rows["hat"]; r["pose"]=list(article.goal)
        t=pose_transform(r["pose"])
        r["corners"]=(local_corners(article) @ t[:3,:3].T+t[:3,3]).tolist()
        r["fixture_force_n"],r["bench_force_n"]=.63765,0.
        r["target_hook_vertical_force_n"] = .63765
        r["linear_velocity"],r["angular_velocity"]=[.03,0,0],[0,0,1.]
        r["fingers"]["left"]=[1,1]
        self.assertEqual(e.update(.2,rows).status,"running")
        r["fingers"]["left"]=[0,0]
        result=e.update(.3,rows)
        self.assertEqual(result.status,"succeeded",result.metrics["criteria"])
        self.assertEqual(result.metrics["stable_duration_s"],0.)
        # Aggregate rack contact, source-pad contact or another peg cannot pass.
        r["target_hook_vertical_force_n"]=0.
        self.assertEqual(e.update(.4,rows).status,"running")



if __name__ == "__main__":
    unittest.main()
