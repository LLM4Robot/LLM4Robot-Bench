"""Offline control contracts; real execution is covered by the GPT pick/place integration case."""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from unittest.mock import Mock, patch
import xml.etree.ElementTree as ET

from robot_tasks.backends.isaacsim import IsaacSimBackend
from robot_tasks.backends.settings import SimulationSettings
from fixtures import robot_config
from robot_tasks.robots.audit import require_joint_sim_ready, require_robot_ready
from robot_tasks.robots.adapters import R5Adapter
from robot_tasks.robots.control import joints_to_width, plan_targets, width_to_joints
from robot_tasks.robots.importer import prepare_urdf
from robot_tasks.robots.profile import load_robot_profile
from robot_tasks.types import Action, ArmCommand, ArmState
from robot_tasks.integrationtest.simulation_process import completed_check


ROOT = Path(__file__).resolve().parents[1]


class R5ControlTests(unittest.TestCase):
    def setUp(self):
        self.config = robot_config()
        self.profile = self.config.robot
        self.targets = {mount.name: mount.home_positions_rad + width_to_joints(self.profile, 0.04)
                        for mount in self.profile.mounts}

    def test_simulation_readiness_does_not_claim_hardware_or_tcp_calibration(self):
        audit = require_joint_sim_ready(self.profile)
        self.assertFalse(audit.ready)
        with self.assertRaisesRegex(ValueError, "hardware_match"):
            require_robot_ready(self.profile)
        hardware = load_robot_profile(ROOT / "configs/robots/r5.toml")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            require_joint_sim_ready(hardware)

    def test_gripper_width_roundtrip_uses_both_fingers(self):
        for width in (0.012, 0.041, 0.070):
            self.assertAlmostEqual(joints_to_width(self.profile, width_to_joints(self.profile, width)), width)
        self.assertAlmostEqual(joints_to_width(self.profile, (0.015, 0.025)), 0.04)
        for bad in (-0.1, 0.08, float("nan"), True):
            with self.subTest(width=bad), self.assertRaises(ValueError):
                width_to_joints(self.profile, bad)

    def test_joint_command_preserves_gripper_and_inactive_arm_targets(self):
        target = (0.1, 1.2, 1.2, 0, 0, 0)
        action = Action((ArmCommand("left", "joint_position", target),), 0.1)
        result = plan_targets(self.profile, ("left",), action, self.targets)
        self.assertEqual(result["left"], target + self.targets["left"][6:])
        self.assertEqual(result["right"], self.targets["right"])
        self.assertEqual(self.targets["left"][0], 0.0)

    def test_bimanual_action_is_validated_atomically(self):
        before = dict(self.targets)
        action = Action((ArmCommand("left", "gripper", (0.06,)),
                         ArmCommand("right", "joint_position", (10, 0, 0, 0, 0, 0))), 0.1)
        with self.assertRaisesRegex(ValueError, "outside configured limits"):
            plan_targets(self.profile, self.config.arms, action, self.targets)
        self.assertEqual(self.targets, before)

    def test_invalid_actions_never_generate_drive_targets(self):
        bad_commands = (
            ArmCommand("right", "gripper", (0.04,)),
            ArmCommand("left", "joint_position", (0.1,)),
            ArmCommand("left", "joint_position", (True, 0, 0, 0, 0, 0)),
            ArmCommand("left", "joint_position", (float("nan"), 0, 0, 0, 0, 0)),
            ArmCommand("left", "gripper", (0.2,)),
        )
        for command in bad_commands:
            with self.subTest(command=command), self.assertRaises(ValueError):
                plan_targets(self.profile, ("left",), Action((command,), 0.1), self.targets)
        command = ArmCommand("left", "gripper", (0.03,))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            plan_targets(self.profile, self.config.arms, Action((command, command), 0.1), self.targets)
        with self.assertRaisesRegex(ValueError, "Cartesian"):
            plan_targets(self.profile, self.config.arms,
                         Action((ArmCommand("left", "cartesian_pose", (0, 0, 0, 1, 0, 0, 0)),), 0.1), self.targets)

    def test_derived_urdf_preserves_source_and_resolves_meshes(self):
        original = hashlib.sha256(self.profile.asset_path.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as temporary:
            output = prepare_urdf(self.profile, Path(temporary) / "resolved.urdf")
            tree = ET.parse(output)
            for mesh in tree.getroot().iter("mesh"):
                self.assertTrue(Path(mesh.get("filename")).is_absolute())
                self.assertTrue(Path(mesh.get("filename")).is_file())
            for joint in tree.getroot().findall("joint"):
                expected = next(limit for limit in self.profile.operating_limits if limit.joint == joint.get("name"))
                self.assertEqual(float(joint.find("limit").get("velocity")), expected.velocity)
        self.assertEqual(hashlib.sha256(self.profile.asset_path.read_bytes()).hexdigest(), original)

    def test_backend_step_obeys_physics_time_and_rejects_fractional_steps(self):
        backend = IsaacSimBackend()
        backend.world, backend.app = Mock(), Mock()
        backend._physics_dt = 1 / 240
        backend.world.current_time = 0.0
        def tick(**kwargs):
            backend.world.current_time += backend._physics_dt
        backend.world.step.side_effect = tick
        backend.step(0.1)
        self.assertEqual(backend.world.step.call_count, 24)
        backend.world.step.reset_mock()
        for bad in (0, -1, 0.001, float("nan"), True):
            with self.subTest(duration=bad), self.assertRaises(ValueError):
                backend.step(bad)
        backend.world.step.assert_not_called()

    def test_cleanup_still_closes_app_if_camera_teardown_fails(self):
        backend = IsaacSimBackend()
        app = Mock()
        backend.app, backend._render_product = app, Mock()
        backend._rgb = SimpleNamespace(detach=Mock(side_effect=RuntimeError("detach failed")))
        with self.assertRaisesRegex(RuntimeError, "detach failed"):
            backend.close()
        app.close.assert_called_once()
        backend.close()
        app.close.assert_called_once()

    def test_simulation_settings_reject_invalid_values_before_kit_start(self):
        for field, value in (("fps", 0), ("headless", "true"), ("graphics_backend", "unknown"),
                             ("arm_stiffness", float("nan")), ("camera_position", [0, 0]),
                             ("solver_position_iterations", 0), ("solver_velocity_iterations", -1),
                             ("solver_velocity_iterations", True)):
            raw = {**self.config.raw, "simulation": {**self.config.raw["simulation"], field: value}}
            with self.subTest(field=field), self.assertRaises(ValueError):
                SimulationSettings.from_config(replace(self.config, raw=raw))

    def test_worker_exit_and_measurement_result_must_both_succeed(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            self.assertFalse(completed_check(output, 0))
            for result, code, expected in (
                ({"status": "completed", "passed": True}, 0, True),
                ({"status": "completed", "passed": False}, 0, False),
                ({"status": "completed", "passed": True}, 5, False),
                ({"status": "starting", "passed": True}, 0, False),
                ([], 0, False),
            ):
                with self.subTest(result=result, code=code):
                    (output / "result.json").write_text(json.dumps(result), encoding="utf-8")
                    self.assertEqual(completed_check(output, code), expected)

    def test_passing_through_target_at_speed_is_not_motion_completion(self):
        robot = R5Adapter()
        robot.config = self.config
        robot.targets = self.targets
        states = {name: ArmState(joint_positions_rad=target[:6], finger_positions_m=target[6:],
                                 joint_velocities_rad_s=(0.0,) * 6, finger_velocities_m_s=(0.0, 0.0))
                  for name, target in self.targets.items()}
        with patch.object(robot, "state", return_value=states):
            self.assertEqual(robot.feedback().status, "completed")
            states["left"] = replace(states["left"], joint_velocities_rad_s=(0.3, 0, 0, 0, 0, 0))
            self.assertEqual(robot.feedback().status, "executing")

    def test_timeout_then_finger_error_feedback_is_json_serializable(self):
        robot = R5Adapter()
        robot.config = self.config
        robot.targets = self.targets.copy()
        # Reproduce the real timeout path reusing a bounded NumPy drive reference.
        robot.drive_targets = {name: tuple(np.float64(v) for v in target)
                               for name, target in self.targets.items()}
        states = {name: ArmState(joint_positions_rad=target[:6],
                                 finger_positions_m=tuple(v+.002 for v in target[6:]),
                                 joint_velocities_rad_s=(0.,)*6,
                                 finger_velocities_m_s=(0.,0.))
                  for name,target in self.targets.items()}
        with patch.object(robot,"state",return_value=states):
            robot.on_action_timeout()
            feedback = robot.feedback()
        # A failed comparison short-circuits to np.bool_ without normalization.
        for details in feedback.metrics['arms'].values():
            self.assertIs(details['gripper_reached'],False)
        json.dumps(asdict(feedback),allow_nan=False)
        self.assertEqual(robot.paths,{})


if __name__ == "__main__":
    unittest.main()
