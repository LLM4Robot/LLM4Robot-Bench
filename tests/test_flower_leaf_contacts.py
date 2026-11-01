"""A mechanically separated leaf must remain visible to release and collision checks."""
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
from robot_tasks.tasks.bench.delivered_scene import DeliveredScene,flower_payload_surface
from robot_tasks.tasks.bench.native_l2 import NativeL2Scene


class LeafContactTests(unittest.TestCase):
    def test_leaf_contacts_are_aggregated_once_and_cannot_look_released(self):
        scene=DeliveredScene.__new__(DeliveredScene)
        scene.spec=SimpleNamespace(parameters={})
        scene.backend=SimpleNamespace(time_s=lambda:1.)
        scene.config=SimpleNamespace(physics_dt_s=1/240)
        scene.flower_leaf_paths=['/rose/LeafBodies/Leaf0']
        scene.flower_leaf_time=None
        scene.flower_leaf_fingers={'right':SimpleNamespace(get_contact_force_matrix=lambda **kw:np.array([[[.2,0,0]],[[0,0,0]]]))}
        scene.flower_leaf_supports={'Leaf0':SimpleNamespace(get_contact_force_matrix=lambda **kw:np.array([[[0,0,.03]],[[0,0,.1]]]).reshape(1,2,3))}
        scene.flower_leaf_support_targets=['target_cup']
        scene.fixture_paths=[];scene.hook_contact=None
        scene.aux_bodies=[(SimpleNamespace(prim_path=scene.flower_leaf_paths[0],get_world_pose=lambda:(np.zeros(3),np.array([1.,0,0,0]))),None)]
        rows={'rose':{'fingers':{'right':[0.,0.]},'pair_forces':{'target_cup':0.},'bench_force_n':0.,'fixture_force_n':0.}}
        with patch.object(NativeL2Scene,'measure',return_value=rows):
            first=scene.measure();second=scene.measure()
        self.assertEqual(first['rose']['fingers']['right'],[.2,0.])
        self.assertEqual(second['rose']['fingers']['right'],[.2,0.])
        self.assertAlmostEqual(second['rose']['pair_forces']['target_cup'],.1)
        self.assertAlmostEqual(second['rose']['bench_force_n'],.03)
        self.assertIn('Leaf0',second['rose']['flexible_parts'])

    def test_payload_follows_measured_leaf_instead_of_frozen_rest_shape(self):
        fake_shapes=[{'name':'Leaf0Contact0'},{'name':'CoatedStem'}]
        with patch('robot_tasks.tasks.bench.native_geometry.shapes',return_value=fake_shapes), \
             patch('robot_tasks.tasks.bench.native_geometry.polyhedron',return_value=(np.zeros((1,3)),None,0.)):
            points=flower_payload_surface([0,0,0,1,0,0,0],{'Leaf0':{'pose':[.1,0,0,1,0,0,0]}})
        np.testing.assert_allclose(points,[[.1,0,0],[0,0,0]])
