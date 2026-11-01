"""Regressions for native masses, axial spin and soft FEM scoring."""

import unittest

import numpy as np
from test_household_tasks import measurement

from robot_tasks.robots.transforms import (
    origin_transform,
    pose_transform,
    transform_pose,
)
from robot_tasks.tasks.bench.evaluation import BenchEvaluator
from robot_tasks.tasks.bench.orange_evaluation import (
    OrangeEvaluator,
    deformation_metrics,
    motion_metrics,
)
from robot_tasks.tasks.bench.spec import build_spec
from robot_tasks.types import InvalidSimulationState


def soft_row(spec, *, at_goal=False, lifted=False, strain=0.0, touching=False):
    a = spec.design.articles[0]
    p = np.asarray((a.goal if at_goal else a.initial)[:3]).copy()
    z = spec.parameters["support_z"] + (0.06 if lifted else 0.0)
    return {
        "orange": {
            "bounds_min": [p[0] - 0.025, p[1] - 0.013, z],
            "bounds_max": [p[0] + 0.025, p[1] + 0.013, z + 0.017],
            "centroid": [p[0], p[1], z + 0.0085],
            "collision_bounds_min": [p[0] - 0.025, p[1] - 0.013, z],
            "collision_bounds_max": [p[0] + 0.025, p[1] + 0.013, z + 0.017],
            "finger_distances_m": [0.10, 0.10, 0.001, 0.001] if touching else [0.1] * 4,
            "speed_m_s": 0.0,
            "centroid_speed_m_s": 0.0,
            "shape_speed_p95_m_s": 0.0,
            "centroid": [p[0], p[1], z + 0.0085],
            "strain_p95": strain,
            "strain_max": strain,
            "minimum_volume_ratio": 1.0,
        }
    }


class NativeL2ScoringTests(unittest.TestCase):



    def test_drawer_passive_final_millimetres_do_not_erase_pull_but_need_return(self):
        spec = build_spec("L2-05")
        rows = measurement(spec)
        e = BenchEvaluator(spec, {a.name: a.initial for a in spec.design.articles})
        row = rows["drawer"]
        axis = np.asarray(spec.parameters["outward_axis_world"])
        row["fingers"]["right"] = [1, 1]
        e.update(.1, rows)
        row["pose"][:3] = (np.asarray(row["pose"][:3]) + .044 * axis).tolist()
        e.update(.2, rows)
        row["fingers"]["right"] = [0, 0]
        row["pose"][:3] = (np.asarray(row["pose"][:3]) + .006 * axis).tolist()
        self.assertEqual(e.update(.3, rows).status, "running")
        self.assertIn("opening_reached", e.flags)
        row["pose"] = list(spec.design.articles[0].initial)
        self.assertEqual(e.update(.4, rows).status, "succeeded")

    def test_drawer_touch_or_small_oscillations_do_not_fake_a_grasped_pull(self):
        spec = build_spec("L2-05")
        axis = np.asarray(spec.parameters["outward_axis_world"])
        for oscillate in (False, True):
            rows = measurement(spec)
            row = rows["drawer"]
            start = np.asarray(row["pose"][:3]) + .05 * axis
            e = BenchEvaluator(spec, {a.name: a.initial for a in spec.design.articles})
            row["fingers"]["right"] = [1, 1]
            for i in range(1, 15):
                row["pose"][:3] = (start + (.001 * (i % 2) if oscillate else 0) * axis).tolist()
                e.update(i * .1, rows)
            row["fingers"]["right"] = [0, 0]
            row["pose"] = list(spec.design.articles[0].initial)
            self.assertEqual(e.update(1.5, rows).status, "running")
            self.assertNotIn("pulled_while_held", e.flags)

    def test_drawer_preopened_layout_scores_absolute_opening_return_and_native_stop(self):
        spec = build_spec("L2-05")
        rows = measurement(spec)
        row = rows["drawer"]
        axis = np.asarray(spec.parameters["outward_axis_world"])
        e = BenchEvaluator(spec, {a.name: a.initial for a in spec.design.articles})
        self.assertAlmostEqual(e.update(.1, rows).metrics["criteria"]["opening_m"], .03)
        row["fingers"]["right"] = [1, 1]
        e.update(.2, rows)
        row["pose"][:3] = (np.asarray(row["pose"][:3]) + .05 * axis).tolist()
        result = e.update(.3, rows)
        self.assertEqual(result.status, "running")
        self.assertAlmostEqual(result.metrics["criteria"]["opening_m"], .08)
        row["fingers"]["right"] = [0, 0]
        row["pose"] = list(spec.design.articles[0].initial)
        self.assertEqual(e.update(.4, rows).status, "succeeded")
        row["pose"] = list(spec.parameters["closed_pose"])
        self.assertEqual(e.update(.5, rows).status, "running")
        row["pose"][:3] = (np.asarray(row["pose"][:3]) + .1 * axis).tolist()
        e.update(.6, rows)
        self.assertEqual(e.update(1., rows).status, "invalid")


    def test_initial_resting_objects_do_not_count_as_completed_placement_stages(self):
        spec = build_spec("L2-01")
        evaluator = BenchEvaluator(
            spec, {a.name: a.initial for a in spec.design.articles}
        )
        rows = measurement(spec)
        for a in spec.design.articles:
            rows[a.name]["fixture_force_n"] = a.mass * 9.81
        criteria = evaluator.update(0.1, rows).metrics["criteria"]
        self.assertTrue(all(criteria["placed"].values()))
        self.assertFalse(criteria["stages"]["barrel_placed"])
        self.assertFalse(criteria["stages"]["cap_placed"])
        spec = build_spec("L2-04")
        result = OrangeEvaluator(spec).update(0.1, soft_row(spec))
        self.assertFalse(result.metrics["stages"]["lifted_50mm"])

    def test_tilted_native_cap_root_does_not_erase_valid_capped_pickup(self):
        spec = build_spec("L2-01")
        rows = measurement(spec)
        for row in rows.values():
            row["pose"][2] += 0.08
            row["corners"] = (np.asarray(row["corners"]) + [0, 0, 0.08]).tolist()
            row["bench_force_n"] = 0.0
        rows["body"]["fingers"]["left"] = [1, 1]
        body = pose_transform(rows["body"]["pose"])
        pivot = origin_transform((0, 0, spec.parameters["body_tip_z"]), (0, 0, 0))
        cap = (
            body
            @ pivot
            @ origin_transform((0, 0, 0), (0, 0.08, 0))
            @ np.linalg.inv(pivot)
        )
        self.assertGreater(np.linalg.norm(cap[:3, 3] - body[:3, 3]), 0.006)
        rows["cap"]["pose"] = list(transform_pose(cap))
        evaluator = BenchEvaluator(
            spec, {a.name: a.initial for a in spec.design.articles}
        )
        result = evaluator.update(0.1, rows)
        self.assertTrue(result.metrics["criteria"]["assembly_lifted"])
        self.assertEqual(result.status, "running")  # Pickup alone is not uncapping.
        # A cap alongside the nib must not count as an assembled marker.
        cap[:3, 3] += body[:3, 0] * 0.03
        rows["cap"]["pose"] = list(transform_pose(cap))
        evaluator = BenchEvaluator(
            spec, {a.name: a.initial for a in spec.design.articles}
        )
        self.assertFalse(
            evaluator.update(0.1, rows).metrics["criteria"]["assembly_lifted"]
        )

    def test_held_cap_separation_finishes_before_placement_or_release(self):
        s = build_spec("L2-01")
        rows = measurement(s)
        e = BenchEvaluator(s, {a.name: a.initial for a in s.design.articles})
        self.assertEqual(e.update(0.1, rows).status, "running")
        for r in rows.values():
            r["pose"][2] += 0.08
            r["corners"] = (np.asarray(r["corners"]) + [0, 0, 0.08]).tolist()
            r["bench_force_n"] = 0.0
        rows["body"]["fingers"]["left"] = [1, 1]
        e.update(0.2, rows)
        rows["cap"]["pose"][0] += 0.05
        rows["cap"]["fingers"]["right"] = [1, 1]
        separated = e.update(0.3, rows)
        self.assertEqual(separated.status, "succeeded")
        self.assertFalse(separated.metrics["criteria"]["placement_required"])
        self.assertFalse(separated.metrics["criteria"]["release_required"])
        self.assertIn("separated_while_held", e.flags)
        for name, r in rows.items():
            r["pose"][2] -= 0.08
            r["corners"] = (np.asarray(r["corners"]) - [0, 0, 0.08]).tolist()
            r["fingers"] = {"left": [0, 0], "right": [0, 0]}
            r["bench_force_n"] = s.parameters[name + "_mass_kg"] * 9.81
            r["angular_velocity"] = [
                9.0,
                0.0,
                0.0,
            ]  # Round native parts lie along world X.
        e.update(0.4, rows)
        result = e.update(1.0, rows)
        self.assertEqual(result.status, "succeeded")
        # Later support readings do not add a placement gate to uncapping.
        self.assertLess(rows["cap"]["bench_force_n"], 0.05)
        rows["cap"]["bench_force_n"] = 0.0
        self.assertEqual(e.update(1.1, rows).status, "succeeded")

    def test_element_strain_ignores_rigid_motion_but_detects_compression(self):
        rest = np.array(
            [[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01]], dtype=float
        )
        tet = np.array([[0, 1, 2, 3]])
        inverse = np.array([np.eye(3) * 100])
        t = origin_transform((0.4, 1.2, 0.8), (0.4, -0.2, 0.9))
        transformed = rest @ t[:3, :3].T + t[:3, 3]
        self.assertLess(
            deformation_metrics(inverse, transformed, tet)["strain_p95"], 1e-12
        )
        compressed = rest * [1, 1, 0.6]
        m = deformation_metrics(inverse, compressed, tet)
        self.assertAlmostEqual(m["strain_p95"], 0.4)
        self.assertAlmostEqual(m["minimum_volume_ratio"], 0.6)









if __name__ == "__main__":
    unittest.main()
