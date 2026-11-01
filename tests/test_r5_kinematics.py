"""Independent FK/coordinate and measured completion checks; no alternate inverse solver."""
from dataclasses import replace
import math
import unittest
from unittest.mock import patch
import numpy as np
from fixtures import robot_config
from robot_tasks.robots.adapters import R5Adapter
from robot_tasks.robots.control import width_to_joints
from robot_tasks.robots.kinematics import R5Kinematics
from robot_tasks.robots.transforms import axis_rotation, origin_transform, pose_errors, pose_residual, pose_transform, transform_pose
from robot_tasks.types import ArmState

class R5KinematicsTests(unittest.TestCase):
    def setUp(self):
        self.config = robot_config()
        self.profile = self.config.robot
        self.kin = R5Kinematics(self.profile)
        self.home = self.profile.mounts[0].home_positions_rad
        self.targets = {m.name: m.home_positions_rad + width_to_joints(self.profile,.04) for m in self.profile.mounts}

    def test_zero_pose_matches_hand_summed_urdf_offsets(self):
        # At q=0, joint3's pi roll reverses the following y/z translations;
        # joint6 restores the tool orientation. This does not use the solver.
        expected = (.2402, .42, .2035, 1., 0., 0., 0.)
        np.testing.assert_allclose(self.kin.forward((0.,)*6, "left"), expected, atol=1e-6)

    def test_rpy_uses_fixed_axis_order(self):
        transform = origin_transform((1, 2, 3), (math.pi/2, 0, math.pi/2))
        np.testing.assert_allclose(transform @ [0, 1, 0, 1], [1, 2, 4, 1], atol=1e-12)

    def test_rotation_roundtrip_and_residual_at_pi(self):
        for axis in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 2, -3)):
            for angle in (0., 1e-9, .7, math.pi - 1e-8, math.pi):
                matrix = np.eye(4)
                matrix[:3, :3] = axis_rotation(axis, angle)
                pose = transform_pose(matrix)
                np.testing.assert_allclose(pose_transform(pose), matrix, atol=1e-8)
                self.assertAlmostEqual(np.linalg.norm(pose_residual(np.eye(4), matrix)[3:]), angle, places=7)
                inverse_sign = pose[:3] + tuple(-v for v in pose[3:])
                self.assertLess(pose_errors(pose, inverse_sign)[1], 1e-10)

    def test_forward_model_has_no_legacy_inverse_solver(self):
        self.assertFalse(hasattr(R5Kinematics(self.profile), "solve"))

    def test_cartesian_completion_requires_measured_tcp_accuracy(self):
        robot = R5Adapter()
        robot.config = self.config
        robot.targets = self.targets
        target = self.kin.forward(self.home, "left")
        robot.cartesian_targets = {"left": target}
        states = {name: ArmState(joint_positions_rad=self.home, finger_positions_m=q[6:],
                                 joint_velocities_rad_s=(0.,)*6, finger_velocities_m_s=(0.,)*2,
                                 tcp_pose_world_wxyz=self.kin.forward(self.home, name))
                  for name, q in self.targets.items()}
        with patch.object(robot, "state", return_value=states):
            self.assertEqual(robot.feedback().status, "completed")
            states["left"] = replace(states["left"], tcp_pose_world_wxyz=(target[0]+.01,) + target[1:])
            self.assertEqual(robot.feedback().status, "executing")
