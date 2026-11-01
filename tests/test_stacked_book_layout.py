"""The preview shift must move visual, reset and scoring geometry together."""
from copy import deepcopy
from dataclasses import replace
import unittest

import numpy as np

from robot_tasks.initial_layout import apply_record_to_stage, scene_record
from robot_tasks.paths import PROJECT_ROOT
from robot_tasks.registry import resolve_task
from robot_tasks.tasks.bench.arm_selection import resolve_bench_spec
from robot_tasks.video_delivery import scenes


class StackedBookLayoutTests(unittest.TestCase):
    def configurations(self):
        config = resolve_task("L3-03_v3")[1]
        yield config
        # Exercise another translation without depending on an unapproved
        # paper-preview configuration or introducing another benchmark task.
        raw = deepcopy(config.raw)
        raw["task"]["stacked_book_offset_x_m"] = .03
        yield replace(config, raw=raw)

    def test_original_alignment_and_relative_geometry_are_preserved(self):
        baseline = deepcopy(scenes()["L3-03"])
        _, original = resolve_task("L3-03")
        self.assertIs(scene_record(original.raw["task"]), scenes()["L3-03"])
        for config in self.configurations():
            row = scene_record(config.raw["task"])
            dx = config.raw["task"]["stacked_book_offset_x_m"]
            delta = np.array([dx, 0., 0.])
            for name in ("lower_book", "upper_book"):
                before, after = baseline["tracked"][name], row["tracked"][name]
                np.testing.assert_allclose(after["pose"][:3], np.array(before["pose"][:3]) + delta)
                self.assertEqual(after["pose"][3:], before["pose"][3:])
                for key in ("bounds", "mass"):
                    self.assertEqual(after[key], before[key])
            spec = resolve_bench_spec(config.raw["task"], config.robot)
            self.assertAlmostEqual(spec.parameters["required_stack_turn_rad"], 2*np.pi)
            for key in ("start", "target"):
                np.testing.assert_allclose(row[key+"_vertices_world"],
                                           np.array(baseline[key+"_vertices_world"])+delta[:2])
                centre = np.asarray(row[key+"_centre_xy"])
                np.testing.assert_allclose(spec.parameters[key+"_vertices"] + centre,
                                           row[key+"_vertices_world"])
            for a in spec.design.articles:
                np.testing.assert_allclose(a.initial, row["tracked"][a.name]["pose"])
        self.assertEqual(baseline, scenes()["L3-03"])

    def test_usd_books_and_all_eight_lines_match_scoring_shift_and_are_idempotent(self):
        from pxr import Usd, UsdGeom
        baseline = Usd.Stage.Open(str(PROJECT_ROOT / "assets/tuojing" /
                                     scenes()["L3-03"]["runtime_path"]))
        for config in self.configurations():
            row = scene_record(config.raw["task"])
            stage = Usd.Stage.Open(str(PROJECT_ROOT / "assets/tuojing" / row["runtime_path"]))
            original = stage.GetRootLayer().ExportToString()
            apply_record_to_stage(stage, config.raw["task"], row)
            first = stage.GetSessionLayer().ExportToString()
            apply_record_to_stage(stage, config.raw["task"], row)
            self.assertEqual(first, stage.GetSessionLayer().ExportToString())
            cache = UsdGeom.XformCache()
            for name, record in row["tracked"].items():
                world = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(record["path"]))).T
                np.testing.assert_allclose(world, record["native_world_matrix"], atol=1e-12)
            for kind in ("start", "target"):
                for i in range(4):
                    path = f"/World/L2/Fixture/{kind}_outline_{i}"
                    before = np.asarray(cache.GetLocalToWorldTransform(baseline.GetPrimAtPath(path))).T.copy()
                    before[:3, 3] += row["stacked_book_initial_shift"]["translation_world_m"]
                    after = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))).T
                    np.testing.assert_allclose(after, before, atol=1e-12)
            self.assertEqual(original, stage.GetRootLayer().ExportToString())


if __name__ == "__main__":
    unittest.main()
