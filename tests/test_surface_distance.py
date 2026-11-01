"""Actual surfaces must not fill the empty parts of their bounding boxes."""

import unittest

import numpy as np

from robot_tasks.tasks.bench.surface_distance import surface_clearance


class SurfaceDistanceTests(unittest.TestCase):
    def setUp(self):
        self.triangle = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])

    def test_point_in_empty_triangle_bbox_is_not_contact(self):
        distance = surface_clearance([[0.9, 0.9, 0.0]], self.triangle, max_distance=1.0)
        self.assertAlmostEqual(distance, 0.8 / np.sqrt(2))

    def test_face_edge_vertex_and_bounded_far_distance(self):
        for point, expected in [
            ([0.2, 0.2, 0.003], 0.003),
            ([0.5, -0.004, 0], 0.004),
            ([-0.003, -0.004, 0], 0.005),
            ([5.0, 5.0, 5.0], 0.01),
        ]:
            with self.subTest(point=point):
                self.assertAlmostEqual(
                    surface_clearance([point], self.triangle), expected
                )

    def test_rigid_transform_preserves_separation_and_nearest_sample_wins(self):
        angle = 0.8
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1.0],
            ]
        )
        points = np.array([[0.2, 0.2, 0.007], [0.3, 0.3, 0.002]])
        shift = np.array([0.4, 1.2, 0.78])
        result = surface_clearance(
            points @ rotation.T + shift, self.triangle @ rotation.T + shift
        )
        self.assertAlmostEqual(result, 0.002)


if __name__ == "__main__":
    unittest.main()
