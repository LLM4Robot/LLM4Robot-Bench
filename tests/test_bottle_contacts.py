"""Closure geometry/reset invariants; no Isaac, networking or model calls."""

from copy import deepcopy
import unittest

import numpy as np
from scipy.spatial import ConvexHull

try:
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
except ImportError:
    Usd = None

from robot_tasks.initial_layout import apply_record_to_stage, scene_record
from robot_tasks.paths import PROJECT_ROOT
from robot_tasks.registry import resolve_task
from robot_tasks.robots.transforms import pose_transform
from robot_tasks.tasks.bench.arm_selection import resolve_bench_spec
from robot_tasks.tasks.bench.bottle_contacts import configure, initial_record, ring_pieces
from robot_tasks.video_delivery import scenes


class BottleResetTests(unittest.TestCase):
    def test_original_and_variant_share_engagement_without_moving_the_bottle(self):
        source = deepcopy(scenes())
        for tid in ('L3-01', 'L3-01_variant01', 'L3-01_v3'):
            with self.subTest(task=tid):
                _, config = resolve_task(tid)
                row = scene_record(config.raw['task'])
                old = source[config.scene_id]
                self.assertEqual(row['tracked']['bottle'], old['tracked']['bottle'])
                self.assertEqual(row['fixtures'], old['fixtures'])
                body = pose_transform(row['tracked']['bottle']['pose'])
                cap = pose_transform(row['tracked']['cap']['pose'])
                relative = np.linalg.inv(body) @ cap
                np.testing.assert_allclose(relative[:3, 3], (0, 0, .0045), atol=1e-6)
                np.testing.assert_allclose(relative[:3, :3], np.diag([-1, -1, 1]), atol=1e-6)
                np.testing.assert_allclose(cap, row['tracked']['cap']['native_world_matrix'], atol=1e-12)
                spec = resolve_bench_spec(config.raw['task'], config.robot)
                article = next(a for a in spec.design.articles if a.name == 'cap')
                np.testing.assert_allclose(article.initial, row['tracked']['cap']['pose'])
                self.assertAlmostEqual(config.physics_dt_s, 1 / 1920)
                self.assertAlmostEqual(config.control_dt_s, 1 / 12)
        self.assertEqual(scenes(), source)

    def test_changed_source_requires_review_instead_of_applying_another_offset(self):
        source = deepcopy(scenes()['L3-01'])
        source['tracked']['cap']['pose'][2] += .01
        with self.assertRaisesRegex(ValueError, 'Audit'):
            initial_record(source)

    def test_neck_elbow_does_not_fill_the_cavity(self):
        profile = ((.133, .01815, .0188), (.138, .0167, .0175), (.1565, .0167, .0175))
        sectors = list(ring_pieces(profile))
        self.assertEqual(len(sectors), 64)
        # A broad convex hull over the entire flared profile fills this space.
        # It would overlap the cap at the neck's inward elbow.
        cavity = np.array([.0165, 0., .138])
        outside_neck = np.array([.018, 0., .145])
        for points in sectors:
            equations = ConvexHull(points).equations
            for point in (cavity, outside_neck):
                self.assertTrue(np.any(equations[:, :3] @ point + equations[:, 3] > 0))


@unittest.skipIf(Usd is None, 'Optional usd-core required')
class BottleUSDTests(unittest.TestCase):
    def test_collision_and_reset_overrides_are_session_only_and_idempotent(self):
        for tid in ('L3-01', 'L3-01_variant01'):
            with self.subTest(task=tid):
                _, config = resolve_task(tid)
                row = scene_record(config.raw['task'])
                stage = Usd.Stage.Open(str(PROJECT_ROOT / 'assets/tuojing' / row['runtime_path']))
                root = str(stage.GetPrimAtPath(row['tracked']['bottle']['path']).GetParent().GetPath())
                layers = {layer.identifier: layer.ExportToString() for layer in stage.GetUsedLayers()}
                with self.assertRaisesRegex(ValueError, 'session layer'):
                    configure(stage, root)
                stage.SetEditTarget(stage.GetSessionLayer())
                apply_record_to_stage(stage, config.raw['task'], row)
                report = configure(stage, root)
                authored = stage.GetSessionLayer().ExportToString()
                apply_record_to_stage(stage, config.raw['task'], row)
                self.assertEqual(configure(stage, root), report)
                self.assertEqual(stage.GetSessionLayer().ExportToString(), authored)
                for layer in stage.GetUsedLayers():
                    if layer != stage.GetSessionLayer():
                        self.assertEqual(layer.ExportToString(), layers[layer.identifier])
                cache = UsdGeom.XformCache()
                for name in ('bottle', 'cap'):
                    record = row['tracked'][name]
                    prim = stage.GetPrimAtPath(record['path'])
                    np.testing.assert_allclose(np.asarray(cache.GetLocalToWorldTransform(prim)).T,
                                               record['native_world_matrix'], atol=1e-7)
                self.assertTrue(UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(root + '/Body')).GetKinematicEnabledAttr().Get())
                self.assertFalse(any(p.IsA(UsdPhysics.Joint) for p in Usd.PrimRange(stage.GetPrimAtPath(root))))
                for part in ('Body', 'Cap'):
                    sdf = stage.GetPrimAtPath(f'{root}/{part}/Colliders/{part}_SDF')
                    self.assertFalse(UsdPhysics.CollisionAPI(sdf).GetCollisionEnabledAttr().Get())
                    hulls = list(stage.GetPrimAtPath(f'{root}/{part}/RobustContacts').GetChildren())
                    self.assertEqual(len(hulls), report['convex_shapes'][part])
                    self.assertGreater(len(hulls), 32)
                    for prim in hulls:
                        mesh = UsdGeom.Mesh(prim)
                        points = np.array(mesh.GetPointsAttr().Get())
                        self.assertLessEqual(len(points), 64)
                        self.assertGreater(ConvexHull(points).volume, 0)
                        faces = np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
                        normals = np.cross(points[faces[:, 1]] - points[faces[:, 0]],
                                           points[faces[:, 2]] - points[faces[:, 0]])
                        self.assertTrue(np.all(np.sum(normals * (points[faces].mean(axis=1) - points.mean(axis=0)), axis=1) > 0))
                        self.assertEqual(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get(), 'convexHull')
                        self.assertEqual(mesh.GetVisibilityAttr().Get(), 'invisible')
                        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
                        self.assertEqual(str(material.GetPath()), root + '/Looks/Contact')
                # No mass, material, visual, goal or attachment edits in the override.
                for forbidden in ('def Material', 'physics:mass =', 'over "Visuals"', 'Joint "'):
                    self.assertNotIn(forbidden, authored)


if __name__ == '__main__':
    unittest.main()
