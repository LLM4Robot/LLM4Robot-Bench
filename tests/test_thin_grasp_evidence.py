"""Thin-grasp evidence must retain prop surfaces while excluding the hand."""
from pathlib import Path
import unittest

import numpy as np

from robot_tasks.cameras.grasp_evidence import (
    Surface, _gripper_meshes, exclude_gripper, match_surfaces,
)
from robot_tasks.robots.control import empty_close_tolerance
from robot_tasks.robots.profile import load_robot_profile

try:
    import trimesh
    import rtree
except ImportError:
    trimesh = None


class ThinGraspEvidenceTests(unittest.TestCase):
    def test_thin_wall_adjoining_surface_remains_observable(self):
        points = np.array([[.04, .01, z] for z in np.linspace(-.02, .02, 12)])
        surface = Surface(points, np.full((12, 3), 120.), 'overview')
        evidence = match_surfaces((surface,), (surface,), .002)
        self.assertTrue(evidence['supported'])
        self.assertEqual(evidence['matching_points'], 12)

    def test_missing_robot_geometry_fails_closed(self):
        self.assertFalse(exclude_gripper(np.ones((12, 3)), .002, None).any())

    def test_thin_wall_is_not_automatically_an_empty_close(self):
        self.assertLess(empty_close_tolerance(0.), .001)
        self.assertEqual(empty_close_tolerance(.012), .0015)

    @unittest.skipIf(trimesh is None, 'requires simulation mesh dependencies')
    def test_real_hand_surfaces_cannot_be_counted_as_a_grasp(self):
        root = Path(__file__).resolve().parents[1]
        profile = load_robot_profile(root/'configs/robots/r5_tuojing_lab.toml')
        meshes = _gripper_meshes(profile.asset_path, profile.package_roots,
                                profile.tcp_pose_wxyz, profile.ee_link)
        for width in (.002, .012):
            for mesh, axis in meshes:
                # Face centres are on the rendered STL, without random sampling.
                samples = mesh.triangles_center[::20] + axis * width / 2
                self.assertFalse(exclude_gripper(samples, width, profile).any())


if __name__ == '__main__':
    unittest.main()
