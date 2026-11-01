"""Physical layout and counting boundaries; no Isaac or provider calls."""

import unittest

import numpy as np
from test_household_tasks import measurement

from robot_tasks.tasks.bench.evaluation import BenchEvaluator
from robot_tasks.tasks.bench.native_geometry import cup_cavity
from robot_tasks.tasks.bench.pearl_pouring import cup_contains, packing, pouring


class PearlPouringTests(unittest.TestCase):
    def test_l2_03_is_pouring_and_l2_06_is_a_distinct_jelly_task(self):
        from robot_tasks.registry import TASKS, resolve_task

        entry, config = resolve_task("L2-03")
        prepared = entry.factory.prepare_config(config)
        self.assertEqual(prepared.arms, ("right",))
        self.assertEqual(prepared.observation_arms, ("left",))
        self.assertEqual(config.raw["task"]["name"], "纸杯间倒珍珠")
        self.assertEqual(pouring().design.task_id, "L2-03")
        _, jelly = resolve_task("L2-06")
        from robot_tasks.tasks.bench.arm_selection import resolve_bench_spec
        self.assertEqual(resolve_bench_spec(jelly.raw["task"], jelly.robot).mechanism, "native_fem_jelly_lift")

    def setUp(self):
        self.spec = pouring()
        self.rows = measurement(self.spec)
        # Fixed bodies need not produce gravity/table support reactions.
        self.rows["receiver_cup"]["bench_force_n"] = 0.0
        self.e = BenchEvaluator(self.spec, {a.name: a.initial for a in self.spec.design.articles})

    def put_received(self, count):
        for i, p in enumerate(packing(count)):
            self.rows[f"load_{i}"]["pose"][:3] = (
                np.asarray(self.rows["receiver_cup"]["pose"][:3]) + p).tolist()
            self.rows[f"load_{i}"]["bench_force_n"] = 0.0

    def finish(self):
        self.e.flags.add("loaded_source_lifted")
        self.e.update(0.1, self.rows)
        return self.e.update(0.7, self.rows)

    def test_half_fill_has_nonoverlapping_native_spheres_and_expected_payload(self):
        pts = np.array(packing())
        self.assertEqual(len(pts), 200)
        self.assertTrue(all(cup_contains(p, tolerance=-.0003) for p in pts))
        distances = np.linalg.norm(pts[:, None] - pts[None, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        self.assertGreater(distances.min(), .008)
        _, floor, rim = cup_cavity()
        height_fraction = (pts[:, 2].max() + .004 - floor) / (rim - floor)
        self.assertGreater(height_fraction, .45)
        self.assertLess(height_fraction, .55)
        self.assertAlmostEqual(self.spec.parameters["loaded_source_mass_kg"], .0757904591433753)
        with self.assertRaises(ValueError):
            packing(0)

    def test_inner_collider_planes_exclude_rim_wall_and_below_floor(self):
        planes, floor, rim = cup_cavity()
        self.assertEqual(len(planes), 48)
        self.assertTrue(cup_contains((0, 0, floor + .004)))
        for p in ((0, 0, floor), (.04, 0, .02), (0, 0, rim), (0, 0, -.02)):
            self.assertFalse(cup_contains(p), p)

    def test_ten_received_with_residue_passes_without_exact_return(self):
        self.put_received(10)
        self.rows["source_cup"]["pose"][0] -= .06
        self.rows["source_cup"]["angular_velocity"] = [0, 0, 5]
        result = self.finish()
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.metrics["criteria"]["received"], 10)
        self.assertEqual(result.metrics["criteria"]["required_received"], 10)
        self.assertEqual(result.metrics["criteria"]["total"], 200)
        self.assertTrue(result.metrics["criteria"]["receiver_fixture_in_place"])

    def test_displaced_fixed_receiver_cannot_satisfy_final_layout(self):
        self.put_received(200)
        self.rows["receiver_cup"]["pose"][0] += .001
        result = self.finish()
        self.assertFalse(result.metrics["criteria"]["receiver_fixture_in_place"])
        self.assertEqual(result.status, "running")

    def test_insufficient_or_fast_falling_pearls_do_not_pass(self):
        self.put_received(9)
        self.assertEqual(self.finish().status, "running")
        self.put_received(10)
        self.rows["load_0"]["linear_velocity"] = [0, 0, -.4]
        self.assertEqual(self.e.update(.8, self.rows).status, "running")

    def test_requires_loaded_grasp_lift_and_released_supported_source(self):
        self.put_received(200)
        self.assertEqual(self.e.update(.1, self.rows).status, "running")
        self.e.flags.add("loaded_source_lifted")
        self.rows["source_cup"]["fingers"]["right"] = [1, 1]
        self.assertEqual(self.e.update(.2, self.rows).status, "running")
        self.rows["source_cup"]["fingers"]["right"] = [0, 0]
        self.rows["source_cup"]["bench_force_n"] = 0
        self.assertEqual(self.e.update(.3, self.rows).status, "running")

    def test_initial_state_and_empty_cup_lift_do_not_establish_loaded_lift(self):
        self.assertEqual(self.e.update(.1, self.rows).status, "running")
        row = self.rows["source_cup"]
        row["pose"][2] += .04
        row["fingers"]["right"] = [1, 1]
        row["bench_force_n"] = 0
        self.e.update(.2, self.rows)
        self.assertNotIn("loaded_source_lifted", self.e.flags)

    def test_looser_receiving_threshold_does_not_relax_loaded_lift_evidence(self):
        source = self.rows["source_cup"]
        source["pose"][2] += .04
        source["fingers"]["right"] = [1, 1]
        source["bench_force_n"] = 0
        for time_s, count in ((.1, 10), (.2, 160)):
            for i, point in enumerate(packing(count)):
                self.rows[f"load_{i}"]["pose"][:3] = (
                    np.asarray(source["pose"][:3]) + point).tolist()
            self.e.update(time_s, self.rows)
            self.assertEqual("loaded_source_lifted" in self.e.flags, count == 160)

    def test_spills_are_reported_without_turning_one_lost_pearl_into_cup_failure(self):
        self.put_received(199)
        row = self.rows["load_199"]
        row["pose"][2] = .65
        row["corners"] = [[0, 0, .65]]
        result = self.finish()
        self.assertEqual(result.status, "succeeded")
        self.assertEqual(result.metrics["criteria"]["ever_spilled"], 1)


if __name__ == "__main__":
    unittest.main()
