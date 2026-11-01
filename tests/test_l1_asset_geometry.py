"""Run with Isaac Sim's USD Python libraries; no renderer or model needed."""

import hashlib
import unittest

import numpy as np

try:
    from pxr import Usd, UsdGeom
except ImportError:
    Usd = None

from robot_tasks.tasks.l1.assets import import_asset, verified_asset


@unittest.skipIf(Usd is None, "Requires Isaac Sim USD libraries")
class RackGeometryTests(unittest.TestCase):
    def setUp(self):
        # Retain the importer regression independently of the current L1-01,
        # which now uses procedural geometry and no external rack asset.
        self.asset = "Geometry/test_tube_slot/00000"
        self.size = (0.246, 0.1152, 0.060)
        try:
            path, _ = verified_asset(self.asset)
        except FileNotFoundError:
            self.skipTest("Optional pinned L1 assets missing; run tools/fetch_l1_assets.py outside unit tests")
        self.path = path
        self.before = hashlib.sha256(path.read_bytes()).hexdigest()
        self.stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(self.stage, "/Fixture")
        self.colliders, self.record = import_asset(
            self.stage, "/Fixture", self.asset,
            self.size, (0, 0, 0), dynamic=False,
        )

    def world_points(self, path):
        from pxr import Gf
        prim = self.stage.GetPrimAtPath(path)
        transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        return np.array([transform.Transform(Gf.Vec3d(*p))
                         for p in UsdGeom.Mesh(prim).GetPointsAttr().Get()])

    def test_rendered_vertices_and_collider_have_real_rack_dimensions(self):
        # Inspect actual transformed points, not the cached extent that caused
        # the original false-positive: both visual and collision were tiny.
        visual = self.world_points("/Fixture/Normalized/Source/visual/model")
        collision = np.concatenate([self.world_points(p) for p in self.colliders])
        for vertices in (visual, collision):
            np.testing.assert_allclose(np.ptp(vertices, axis=0), self.size, atol=1e-7)
            np.testing.assert_allclose((vertices.max(0) + vertices.min(0)) / 2, 0, atol=1e-7)
        np.testing.assert_allclose(visual, collision, atol=1e-7)
        self.assertEqual(UsdGeom.Imageable(self.stage.GetPrimAtPath(
            "/Fixture/Normalized/Source/visual/model")).ComputeVisibility(), "inherited")

    def test_extent_repair_preserves_pinned_asset(self):
        self.assertIn("visual/model", self.record["repaired_mesh_extents"])
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), self.before)
        mesh = UsdGeom.Mesh(self.stage.GetPrimAtPath("/Fixture/Normalized/Source/visual/model"))
        vertices = np.asarray(mesh.GetPointsAttr().Get())
        np.testing.assert_allclose(mesh.GetExtentAttr().Get(), [vertices.min(0), vertices.max(0)])


if __name__ == "__main__":
    unittest.main()
