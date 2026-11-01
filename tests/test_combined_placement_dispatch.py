"""Baked L3 variants and runtime L1 placements must coexist after merging."""
import unittest

import numpy as np

try:
    from pxr import Usd, UsdGeom
except ImportError:
    Usd = UsdGeom = None

from robot_tasks.initial_layout import scene_record, apply_record_to_stage
from robot_tasks.paths import PROJECT_ROOT
from robot_tasks.registry import resolve_task


@unittest.skipIf(Usd is None, "Optional usd-core")
class CombinedPlacementDispatchTests(unittest.TestCase):
    def test_l3_baked_variants_keep_their_transforms_and_layers(self):
        # The bottle's separate physical closure repair is covered in
        # test_bottle_contacts; it intentionally changes the cap's relative pose.
        for tid in ("L3-02_v3", "L2-07_v3"):
            with self.subTest(task=tid):
                _, config = resolve_task(tid)
                row = scene_record(config.raw["task"])
                self.assertIn("initial_placement_perturbation", row)
                stage = Usd.Stage.Open(str(PROJECT_ROOT / "assets/tuojing" / row["runtime_path"]))
                records = [*row["tracked"].values(), *row["fixtures"]]

                def matrices():
                    cache = UsdGeom.XformCache()
                    return [np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(r["path"])))
                            for r in records]

                before = matrices()
                layers = [stage.GetRootLayer(), stage.GetSessionLayer()]
                text = [layer.ExportToString() for layer in layers]
                apply_record_to_stage(stage, config.raw["task"], row)
                apply_record_to_stage(stage, config.raw["task"], row)
                for actual, expected in zip(matrices(), before):
                    np.testing.assert_array_equal(actual, expected)
                self.assertEqual([layer.ExportToString() for layer in layers], text)

    def test_l1_runtime_variant_still_matches_its_declared_initial_poses(self):
        _, config = resolve_task("L1-01_v3")
        row = scene_record(config.raw["task"])
        stage = Usd.Stage.Open(str(PROJECT_ROOT / "assets/tuojing" / row["runtime_path"]))
        source = stage.GetRootLayer().ExportToString()
        apply_record_to_stage(stage, config.raw["task"], row)
        cache = UsdGeom.XformCache()
        for record in [*row["tracked"].values(), *row["fixtures"]]:
            actual = np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(record["path"]))).T
            np.testing.assert_allclose(actual, record["native_world_matrix"], atol=1e-10)
        self.assertEqual(stage.GetRootLayer().ExportToString(), source)


if __name__ == "__main__":
    unittest.main()
