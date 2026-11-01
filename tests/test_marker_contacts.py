"""Geometry regressions for the native marker/support contact repair."""

import unittest

import numpy as np

from robot_tasks.integrationtest.diagnostics.marker_contact import penetration_m
from robot_tasks.tasks.bench.marker_contacts import support_prisms


def inside(point, polygon):
    edges = np.roll(polygon, -1, axis=0) - polygon
    offset = np.asarray(point) - polygon
    return np.all(edges[:, 0] * offset[:, 1] - edges[:, 1] * offset[:, 0] >= -1e-10)


class MarkerContactTests(unittest.TestCase):
    def test_solids_preserve_recess_height_and_rounded_outer_boundary(self):
        outline = np.array([(-.095, -.092), (-.087, -.100), (.087, -.100),
                            (.095, -.092), (.095, .092), (.087, .100),
                            (-.087, .100), (-.095, .092)])
        pieces = support_prisms(outline, .0935, .100, (.0765, .0815))
        self.assertEqual(len(pieces), 5)
        for point, expected_height in (((0, 0), .0935), ((.09, 0), .1),
                                        ((0, -.095), .1), ((.095, .1), 0)):
            actual = max((z for _, polygon, z in pieces if inside(point, polygon)), default=0)
            self.assertAlmostEqual(actual, expected_height)
        for _, polygon, _ in pieces:
            self.assertTrue(all(inside(p, outline) for p in polygon))

    def test_penetration_detector_rejects_a_motionless_object_inside_support(self):
        # Upward-facing top at z=0.1; a sleeping body underneath is not settled.
        top = np.array([[[-1., -1., .1], [1., -1., .1], [1., 1., .1]],
                        [[-1., -1., .1], [1., 1., .1], [-1., 1., .1]]])
        self.assertAlmostEqual(penetration_m(np.array([[0, 0, .02]]), top), .08)
        self.assertEqual(penetration_m(np.array([[2, 0, .02], [0, 0, .11]]), top), 0.)


if __name__ == "__main__":
    unittest.main()
