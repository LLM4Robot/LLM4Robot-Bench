"""Release evidence must distinguish following, stationary and occluded props."""
import unittest
from dataclasses import replace
from unittest.mock import patch
import numpy as np
from test_grasp_guard import keep_fixture_surfaces, observation
from robot_tasks.cameras.grasp_evidence import visible_surfaces
from robot_tasks.cameras.release_evidence import ReleaseEvidence
from robot_tasks.grasp_guard import GraspGuard
from robot_tasks.types import Action, ArmCommand
from fixtures import robot_config


class ReleaseEvidenceTests(unittest.TestCase):
    def make_evidence(self):
        before=observation(rise=.025,object_rise=.025,width=.07)
        pose=before.robot['left'].tcp_pose_world_wxyz
        surfaces=visible_surfaces(before.rgb,pose,.07,0.)
        points=np.concatenate([s.points for s in surfaces])
        colours=np.concatenate([s.colours for s in surfaces])
        evidence=ReleaseEvidence(pose,points,colours,0.,pose,points,colours,0.)
        return evidence,before

    def test_stationary_object_after_probe(self):
        evidence,_=self.make_evidence()
        after=observation(rise=.030,object_rise=.025,width=.07,time_s=1.)
        result=evidence.update(after,'left',0.,None,True)
        self.assertEqual(result['status'],'stationary_after_probe',result)

    def test_caught_object_follows_open_hand(self):
        evidence,_=self.make_evidence()
        after=observation(rise=.030,object_rise=.030,width=.07,time_s=1.)
        result=evidence.update(after,'left',0.,None,True)
        self.assertEqual(result['status'],'object_following',result)

    def test_opening_or_static_hand_never_confirms_release(self):
        evidence,before=self.make_evidence()
        self.assertEqual(evidence.update(before,'left',0.,None,False)['status'],'opening')
        self.assertEqual(evidence.update(before,'left',0.,None,True)['status'],'release_unconfirmed')

    def test_missing_depth_is_unknown_not_released(self):
        evidence,_=self.make_evidence()
        after=observation(rise=.030,object_rise=.025,width=.07,time_s=1.)
        frames=tuple(replace(f,depth=replace(f.depth,data=np.full((f.height,f.width),np.nan,'<f4').tobytes())) for f in after.rgb)
        self.assertEqual(evidence.update(replace(after,rgb=frames),'left',0.,None,True)['status'],'release_unknown')

    def test_indistinguishable_plane_is_unknown(self):
        evidence,_=self.make_evidence()
        # Interior sidewall patches offer no evidence for translation along the
        # wall; excluding boundaries must not manufacture a release decision.
        points=evidence.probe_points
        mask=(abs(points[:,0])<.01) & (abs(points[:,2])<.01)
        evidence.probe_points=points[mask]
        evidence.probe_colours=evidence.probe_colours[mask]
        after=observation(rise=.030,object_rise=.025,width=.07,time_s=1.)
        result=evidence.update(after,'left',0.,None,True)
        self.assertEqual(result['status'],'release_unknown',result)

    @patch(
        "robot_tasks.cameras.grasp_evidence.exclude_gripper",
        new=keep_fixture_surfaces,
    )
    def test_release_is_advisory_and_reclose_starts_new_attempt(self):
        guard=GraspGuard(robot_config())
        before=observation(rise=.025,object_rise=.025,width=.035)
        guard.accepted(Action((ArmCommand('left','gripper',(.07,)),),1/12),before)
        self.assertGreater(len(guard.releases['left'][0].points), 0)
        after=guard.enrich(observation(rise=.025,object_rise=.025,width=.07,time_s=1.))
        self.assertTrue(after.feedback.metrics['release_check']['left']['advisory'])
        motion=Action((ArmCommand('left','cartesian_pose',(.36,.4,.15,*before.robot['left'].tcp_pose_world_wxyz[3:])),),1/12)
        self.assertIsNone(guard.reject_reason(motion,after))
        guard.accepted(Action((ArmCommand('left','gripper',(.012,)),),1/12),after)
        self.assertNotIn('left',guard.releases)
        self.assertIn('left',guard.attempts)

if __name__=='__main__':unittest.main()
