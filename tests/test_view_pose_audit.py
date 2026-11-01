"""Counterexamples for the initial-pose and view comparison diagnostics."""

import unittest
import numpy as np
from PIL import Image

from robot_tasks.integrationtest.diagnostics.robot_reset_audit import pose_error
from tools.diagnostics.audit_view_pose_alignment import contain, joint_closure


class ViewPoseAuditTests(unittest.TestCase):
    def test_display_preserves_a_nonmatching_aspect_ratio_and_can_upscale(self):
        image = Image.new('RGB', (750, 562), 'white')
        result, transform = contain(image, (1280, 960))
        self.assertEqual(result.size, (1280, 960))
        scale = transform['uniform_scale']
        self.assertGreater(scale, 1)
        self.assertAlmostEqual(transform['resized_size'][0] / 750, scale)
        self.assertAlmostEqual(transform['resized_size'][1] / 562, scale)
        self.assertAlmostEqual(transform['padding_xy'][1], (960 - 562*scale) / 2)
        self.assertEqual(result.getpixel((640, 480)), (255, 255, 255))

    def test_pose_comparison_handles_quaternion_sign_and_detects_base_offset(self):
        self.assertLess(pose_error([0,0,0,-1,0,0,0], [0,0,0,1,0,0,0])['rotation_rad'], 1e-7)
        error = pose_error([0,0,.1,.5,0,0,np.sqrt(3)/2], [0,0,0,1,0,0,0])
        self.assertAlmostEqual(error['translation_m'], .1)
        self.assertAlmostEqual(error['rotation_rad'], 2*np.pi/3)

    def test_static_wrist_that_breaks_joint_closure_is_rejected(self):
        try:
            from pxr import Gf, Usd, UsdGeom, UsdPhysics
        except ImportError:
            self.skipTest('Optional usd-core')
        stage = Usd.Stage.CreateInMemory()
        for i in range(9):
            UsdGeom.Xform.Define(stage, '/Robot/b' + str(i))
        for i in range(1, 9):
            cls = UsdPhysics.RevoluteJoint if i < 7 else UsdPhysics.PrismaticJoint
            joint = cls.Define(stage, '/Robot/joints/joint' + str(i))
            joint.CreateAxisAttr('X')
            joint.CreateBody0Rel().SetTargets(['/Robot/b' + str(i-1)])
            joint.CreateBody1Rel().SetTargets(['/Robot/b' + str(i)])
        self.assertTrue(all(r['kinematically_consistent'] for r in joint_closure(stage, '/Robot')))
        wrist = UsdGeom.Xformable(stage.GetPrimAtPath('/Robot/b6'))
        wrist.AddTranslateOp().Set(Gf.Vec3d(0, .042, 0))
        wrist.AddRotateYOp().Set(12)
        row = joint_closure(stage, '/Robot')[5]
        self.assertFalse(row['kinematically_consistent'])
        self.assertAlmostEqual(row['closure_translation_m'], .042)
        self.assertAlmostEqual(row['closure_rotation_rad'], np.deg2rad(12))


if __name__ == '__main__':
    unittest.main()
