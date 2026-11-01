"""Shallow-contact exits must separate every colliding pair and keep obstacles."""
import unittest
import numpy as np
from robot_tasks.planning.contact_exit import validate_clearances, scene_clearances


class ContactExitTests(unittest.TestCase):
    def test_shallow_contact_separates_and_reaches_clearance(self):
        self.assertAlmostEqual(validate_clearances([
            [-.00027, .02], [-.0001, .019], [.001, .018]]), .00027)

    def test_deep_new_worsening_and_reentering_contacts_are_rejected(self):
        cases = [
            [[-.002], [.001]],
            [[-.00027, .001], [0, -.0001], [.001, .001]],
            [[-.00027], [-.0004], [.001]],
            [[-.00027], [.001], [-.0001], [.001]],
            [[-.00027], [.0001]],
            [[.001], [.002]],
            [[-.00027], [float('nan')]],
        ]
        for gaps in cases:
            with self.subTest(gaps=gaps), self.assertRaises(ValueError):
                validate_clearances(gaps)

    def test_sphere_box_contact_and_other_arm_are_checked(self):
        spheres = np.array([[[0, 0, .1097, .01]], [[0, 0, .1105, .01]], [[0, 0, .112, .01]]])
        box = {'pose': [0, 0, 0, 1, 0, 0, 0], 'dims': [.2, .2, .2]}
        gaps = scene_clearances(spheres, {'cuboid': {'fixture': box}})
        self.assertAlmostEqual(validate_clearances(gaps), .0003)
        with self.assertRaisesRegex(ValueError, 'other arm'):
            scene_clearances(spheres, {'cuboid': {'other_left_0': box}})

    def test_mesh_collision_is_not_replaced_by_its_bounds(self):
        import trimesh
        box = trimesh.creation.box(extents=[.2, .2, .2])
        mesh = {'vertices': box.vertices.tolist(), 'faces': box.faces.tolist(),
                'pose': [0, 0, 0, 1, 0, 0, 0]}
        spheres = np.array([[[0, 0, .1097, .01]], [[0, 0, .112, .01]]])
        gaps = scene_clearances(spheres, {'mesh': {'fixture': mesh}})
        self.assertAlmostEqual(validate_clearances(gaps), .0003)
        mesh['faces'] = mesh['faces'][:-1]
        with self.assertRaisesRegex(ValueError, 'closed collision'):
            scene_clearances(spheres, {'mesh': {'fixture': mesh}})

    def test_touching_closed_parts_keep_their_topology(self):
        import trimesh
        a = trimesh.creation.box(extents=[.1, .2, .2])
        b = a.copy(); b.apply_translation([.1, 0, 0])
        mesh = trimesh.util.concatenate([a, b])
        data = {'vertices': mesh.vertices.tolist(), 'faces': mesh.faces.tolist(),
                'pose': [0, 0, 0, 1, 0, 0, 0]}
        spheres = np.array([[[0, 0, .1097, .01]], [[0, 0, .112, .01]]])
        self.assertAlmostEqual(validate_clearances(scene_clearances(spheres, {'mesh': {'fixture': data}})), .0003)
