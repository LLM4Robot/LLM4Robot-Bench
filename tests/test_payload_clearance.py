"""Broad phase must preserve penetration rejection and hollow gaps."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np

try:
    import trimesh
except ImportError:
    trimesh = None

from robot_tasks.planning.worker import CuroboWorker


@unittest.skipIf(trimesh is None, "trimesh is supplied by the planner runtime")
class PayloadClearanceTests(unittest.TestCase):
    def setUp(self):
        self.worker=CuroboWorker.__new__(CuroboWorker)
        self.worker.kin=SimpleNamespace(forward=lambda q,arm: (q[0],0,0,1,0,0,0))
        self.path=SimpleNamespace(duration_s=0.,sample=lambda t:[0.])

    def request(self, points, offset=0.):
        mesh=trimesh.creation.box(extents=[.1,.1,.1])
        return {'states':{'left':{'joint_positions_rad':[0.]}},
                'scene':{'table_surface_z_m':-10.,'payloads':{'left':[{'surface_points_world':points}]},
                         'obstacles':{'mesh':{'box':{'vertices':mesh.vertices.tolist(),
                                                   'faces':mesh.faces.tolist(),'pose':[offset,0,0,1,0,0,0]}}}}}

    def test_remote_geometry_never_runs_the_expensive_narrow_phase(self):
        request=self.request([[2.,0,0],[2.,0,0]])
        with patch.object(trimesh.proximity,'signed_distance',side_effect=AssertionError('remote mesh queried')):
            diagnostic={};self.worker.check_payload('left',self.path,request,diagnostic)
        self.assertEqual(diagnostic['payload_mesh_queries'],0)

    def test_real_penetration_is_rejected_but_existing_contact_tolerance_is_retained(self):
        with self.assertRaisesRegex(ValueError,'penetrates fixture'):
            self.worker.check_payload('left',self.path,self.request([[0,0,0]]),{})
        self.worker.check_payload('left',self.path,self.request([[.0495,0,0]]),{})

    def test_cache_invalidates_when_obstacle_pose_changes(self):
        request=self.request([[1.,0,0]])
        self.worker.check_payload('left',self.path,request,{})
        changed=deepcopy(request);changed['scene']['obstacles']['mesh']['box']['pose'][0]=1.
        with self.assertRaisesRegex(ValueError,'penetrates fixture'):
            self.worker.check_payload('left',self.path,changed,{})

    def test_separate_meshes_keep_the_gap_open(self):
        request=self.request([[0,0,0]],offset=-.10)
        other=deepcopy(request['scene']['obstacles']['mesh']['box']);other['pose'][0]=.10
        request['scene']['obstacles']['mesh']['other']=other
        self.worker.check_payload('left',self.path,request,{})

    def test_intermediate_trajectory_collision_is_not_skipped(self):
        request=self.request([[-.1,0,0]])
        self.path=SimpleNamespace(duration_s=.1,sample=lambda t:[2*t])
        with self.assertRaisesRegex(ValueError,'penetrates fixture'):
            self.worker.check_payload('left',self.path,request,{})

    def test_table_guard_still_rejects_payload_below_table(self):
        request=self.request([[2,0,-.01]])
        request['scene']['table_surface_z_m']=0.
        with self.assertRaisesRegex(ValueError,'intersects table'):
            self.worker.check_payload('left',self.path,request,{})

    def rose_pickup(self, downward=False):
        from robot_tasks.tasks.bench.l1 import flower
        from robot_tasks.planning.scene import article_surface
        from robot_tasks.robots.transforms import pose_transform

        # Real episode step 29: native rose is resting above the table, while
        # the former 6-mm straight-stem proxy extended 1.93 mm into it.
        pose = [-.28030720353126526, 1.2197195291519165, .7818540334701538,
                .2691560387611389, -.6684878468513489, .15632425248622894, -.675456702709198]
        transform = pose_transform(pose)
        points = article_surface(flower().design.articles[0])
        world = points @ transform[:3, :3].T + transform[:3, 3]
        self.assertGreater(world[:, 2].min(), .78)
        table = trimesh.creation.box(extents=[2., 2., .08])
        request = self.request(world.tolist())
        request['scene']['table_surface_z_m'] = .78
        request['scene']['obstacles']['mesh'] = {'table': {
            'vertices': table.vertices.tolist(), 'faces': table.faces.tolist(),
            'pose': [0., 1., .74, 1., 0., 0., 0.]}}
        self.worker.kin = SimpleNamespace(forward=lambda q, arm: (0, 0, q[0], 1, 0, 0, 0))
        self.path = SimpleNamespace(duration_s=.1, sample=lambda t: [(-1 if downward else 1)*.25*t])
        return request

    def test_native_rose_can_lift_from_recorded_tabletop_grasp(self):
        request = self.rose_pickup()
        self.worker.check_payload('left', self.path, request, {})

    def test_native_rose_still_cannot_move_down_through_table(self):
        request = self.rose_pickup(downward=True)
        with self.assertRaisesRegex(ValueError, 'intersects table|penetrates fixture'):
            self.worker.check_payload('left', self.path, request, {})

    def test_rose_bud_still_collides_with_an_obstacle_during_lift(self):
        request = self.rose_pickup()
        # Put a solid obstruction around a bud sample while leaving the stem
        # untouched. A stem-only or disabled payload check would miss this.
        point = request['scene']['payloads']['left'][0]['surface_points_world'][-1]
        block = self.request([point])['scene']['obstacles']['mesh']['box']
        block['pose'][:3] = point
        request['scene']['obstacles']['mesh']['bud_obstruction'] = block
        with self.assertRaisesRegex(ValueError, 'bud_obstruction'):
            self.worker.check_payload('left', self.path, request, {})

    def delivered_flower_pickup(self, direction=1):
        from robot_tasks.tasks.bench.delivered_scene import flower_payload_surface
        # L1-05 2026-10-16 formal run, step 6. The whole-body AABB reached
        # z=.8651 despite the actual collider remaining above z=.8829.
        pose = [.39672160148620605, 1.1641508340835571, .8823432922363281,
                -.8336890339851379, -.1822560876607895, .1780846267938614, -.4899296760559082]
        points = flower_payload_surface(pose)
        self.assertGreater(points[:,2].min(), .880)
        request = self.request(points.tolist())
        request['scene']['table_surface_z_m'] = .880
        request['scene']['obstacles']['mesh'] = {}
        self.worker.kin = SimpleNamespace(forward=lambda q, arm: (0,0,q[0],1,0,0,0))
        self.path = SimpleNamespace(duration_s=.1, sample=lambda t:[direction*.25*t])
        return request

    def test_delivered_flower_lift_passes_without_weakening_table_guard(self):
        request = self.delivered_flower_pickup()
        self.worker.check_payload('left', self.path, request, {})
        request = self.delivered_flower_pickup(-1)
        with self.assertRaisesRegex(ValueError, 'intersects table'):
            self.worker.check_payload('left', self.path, request, {})

    def test_delivered_flower_head_is_still_checked_for_fixture_collision(self):
        request = self.delivered_flower_pickup()
        point = max(request['scene']['payloads']['left'][0]['surface_points_world'],key=lambda p:p[2])
        block = self.request([point])['scene']['obstacles']['mesh']['box']
        block['pose'][:3] = point
        request['scene']['obstacles']['mesh']['head_obstruction'] = block
        with self.assertRaisesRegex(ValueError, 'head_obstruction'):
            self.worker.check_payload('left', self.path, request, {})


if __name__=='__main__':unittest.main()
