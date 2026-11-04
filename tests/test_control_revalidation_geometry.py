"""Real FCL regressions for solid containment; no CUDA or simulator."""
import importlib.util
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/control_revalidation'))
HAS_GEOMETRY = all(importlib.util.find_spec(name) for name in ('fcl', 'trimesh'))
if HAS_GEOMETRY:
    import fcl
    import trimesh
    import mesh_guard as guard_module
    from mesh_guard import FingerMeshGuard, convex_shape, mesh_shape


@unittest.skipUnless(HAS_GEOMETRY, 'Install tools/control_revalidation/requirements.txt for CPU geometry checks')
class SolidGuardTests(unittest.TestCase):
    def setUp(self):
        self.finger = trimesh.creation.box(extents=[.004] * 3).convex_hull
        self.obstacle = trimesh.creation.box(extents=[.04] * 3)

    def finger_at(self, x):
        return fcl.CollisionObject(convex_shape(self.finger), fcl.Transform(np.array([x, 0., 0.])))

    def test_fully_contained_finger_and_surface_intersection_are_rejected(self):
        other = fcl.CollisionObject(mesh_shape(self.obstacle))
        for x in (0., .02):
            with self.subTest(x=x):
                self.assertLessEqual(FingerMeshGuard.gap(self.finger_at(x), other), 0)
        self.assertAlmostEqual(FingerMeshGuard.gap(self.finger_at(.03), other), .008, places=8)

    def test_obstacle_fully_inside_finger_is_also_rejected(self):
        tiny = trimesh.creation.box(extents=[.001] * 3)
        self.assertLessEqual(FingerMeshGuard.gap(self.finger_at(0), fcl.CollisionObject(mesh_shape(tiny))), 0)

    def test_mesh_pose_is_preserved(self):
        self.obstacle.apply_translation([.1, 0, 0])
        other = fcl.CollisionObject(mesh_shape(self.obstacle))
        self.assertLessEqual(FingerMeshGuard.gap(self.finger_at(.1), other), 0)
        self.assertGreater(FingerMeshGuard.gap(self.finger_at(0), other), .07)

    def test_concave_or_disconnected_enclosures_are_conservative(self):
        left = trimesh.creation.box(extents=[.01] * 3)
        right = left.copy()
        left.apply_translation([-.02, 0, 0]); right.apply_translation([.02, 0, 0])
        mesh = trimesh.util.concatenate([left, right])
        # The finger fits in the actual gap, but the world hull deliberately
        # refuses to certify this concavity instead of risking false clearance.
        self.assertLessEqual(FingerMeshGuard.gap(self.finger_at(0), fcl.CollisionObject(mesh_shape(mesh))), 0)

    def test_degenerate_and_nonfinite_meshes_fail_closed(self):
        flat = trimesh.Trimesh(vertices=[[0,0,0], [1,0,0], [1,1,0], [0,1,0]],
                               faces=[[0,1,2], [0,2,3]], process=False)
        bad = self.obstacle.copy(); bad.vertices[0, 0] = float('nan')
        for mesh in (flat, bad):
            with self.subTest(mesh=mesh), self.assertRaises(ValueError):
                mesh_shape(mesh)

    def guard(self):
        guard = FingerMeshGuard.__new__(FingerMeshGuard)
        guard.w = SimpleNamespace(kin=SimpleNamespace(bases={'right': np.eye(4)}, reach_bound_m=1.),
                                  profile=SimpleNamespace(ee_link='ee'), world_spheres=lambda *args: [])
        guard.arm = 'right'
        guard.request = {'states': {'right': {'finger_positions_m': [0.]}}}
        guard.fingers = [('finger', convex_shape(self.finger), np.asarray(self.finger.vertices),
                          {'origin': np.eye(4), 'axis': [1.,0,0]}, 0)]
        guard.static = [('solid', fcl.CollisionObject(mesh_shape(self.obstacle)), *self.obstacle.bounds)]
        guard.geometry = {}; guard.own_links = []; guard.cache = {}; guard.calls = 0
        guard.minimum_gap = math.inf; guard.closest = None
        return guard

    def test_swept_path_checks_containment_and_between_endpoint_collision(self):
        def transform(kin, q):
            matrix = np.eye(4); matrix[0, 3] = q[0]
            return {'ee': matrix}
        with patch.object(guard_module, 'link_transforms', side_effect=transform):
            for path in ([[0.], [.001]], [[-.03], [.03]]):
                with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'collision/insufficient'):
                    self.guard().validate(path)
            proof = self.guard().validate([[.03], [.04]])
            self.assertGreaterEqual(proof['minimum_mesh_gap_m'], .0079999)
            self.assertEqual(proof['protocol'], 'bounded_finger_solid_narrow_phase_v2')

    def test_nonfinite_paths_are_not_certified(self):
        with self.assertRaisesRegex(ValueError, 'finite path'):
            self.guard().validate([[0.], [float('nan')]])


if __name__ == '__main__':
    unittest.main()
