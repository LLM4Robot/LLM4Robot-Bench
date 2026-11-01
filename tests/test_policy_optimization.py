import base64
from dataclasses import replace
from io import BytesIO
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from robot_tasks.cameras.geometry import project_world, resized_intrinsics
from robot_tasks.config import ExecutionSettings, load_config
from robot_tasks.policies.gpt import GPTPolicy, prepare_frame
from robot_tasks.policies.model_config import GPTSettings
from robot_tasks.policies.operating_notes import manipulation_guidance, task_execution_guidance
from robot_tasks.registry import TASKS, resolve_task
from robot_tasks.policies.responses import ModelReply
from robot_tasks.policies.sparse_action import FIELDS, REFERENCE, decode_sparse_action, orientation_degrees
from robot_tasks.robots.adapters import R5Adapter
from robot_tasks.robots.control import width_to_joints
from robot_tasks.robots.transforms import origin_transform, pose_errors, transform_pose
from robot_tasks.types import Action, ArmCommand, ArmState, Feedback, Observation, PolicyInput, RGBFrame


from fixtures import rgbd, decision_document


ROOT = Path(__file__).resolve().parents[1]


class OptimizationTests(unittest.TestCase):
    def setUp(self):
        base = load_config(ROOT / "configs/checks/gpt_single_arm_pick_place.toml", depth=True)
        self.config = replace(base, cameras=("overview",), observation_arms=())
        t = origin_transform((.3, .4, .2), (0., 0., 0.))
        t[:3, :3] = REFERENCE
        self.pose = transform_pose(t)
        sensor = rgbd(time_s=1., width=100, height=80)
        self.frame = replace(sensor, data=bytes((20, 40, 60)) * 8000,
                             calibration={**sensor.calibration, "intrinsics_fx_fy_cx_cy": [80, 80, 50, 40],
                                          "T_world_camera_pose_wxyz": [0, 0, 1, 1, 0, 0, 0]})
        self.obs = Observation(1., (self.frame,), {"left": ArmState(tcp_pose_world_wxyz=self.pose)}, Feedback("completed"))
        self.settings = GPTSettings("https://example.invalid", "test", max_calls=8)
        self.client = Mock()
        self.client.generate.return_value = ModelReply(json.dumps(decision_document()), "test", "id", {}, 0.)

    def sparse(self, **values):
        row = {"arm": "left", **dict.fromkeys(FIELDS), **values}
        return decode_sparse_action(json.dumps({"commands": [row], "request_stop": False}), self.config, self.obs)

    def test_sparse_position_holds_measured_axes_and_quaternion(self):
        command, = self.sparse(z_m=.25).commands
        self.assertEqual(command.values, (*self.pose[:2], .25, *self.pose[3:]))

    def test_sparse_angles_rotate_about_tcp_without_translating_it(self):
        command, = self.sparse(yaw_deg=45).commands
        self.assertEqual(command.values[:3], self.pose[:3])
        np.testing.assert_allclose(orientation_degrees(command.values), (0, 0, 45), atol=1e-8)

    def test_gripper_only_does_not_create_motion(self):
        command, = self.sparse(gripper=1).commands
        self.assertEqual(command.kind, "gripper")

    def test_sparse_invalid_values_and_unknown_arm_are_rejected(self):
        for values in ({"arm": "right", "z_m": .2}, {"z_m": True}, {"z_m": float("nan")},
                       {"roll_deg": 500}, {"gripper": 1.1}, {}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.sparse(**values)

    def test_partial_motion_requires_measured_tcp(self):
        self.obs = replace(self.obs, robot={})
        with self.assertRaisesRegex(ValueError, "Measured TCP"):
            self.sparse(z_m=.2)

    def test_text_history_survives_image_eviction_and_episode_reset(self):
        policy = GPTPolicy(self.config, self.settings, client=self.client)
        for i in range(4):
            policy.act(PolicyInput(f"instruction_{i}", self.obs, i + 1))
        self.assertEqual(len(policy.history), 4)
        messages = self.client.generate.call_args.kwargs["messages"]
        self.assertIn("instruction_0", json.dumps(messages))
        self.assertFalse(any(item["type"] == "input_image" for item in policy.history[0][0]["content"]))
        self.assertEqual(sum(item["type"] == "input_image" for user, _ in policy.history for item in user["content"]), 4)
        policy.reset(12)
        policy.act(PolicyInput("fresh_episode", self.obs, 1))
        self.assertEqual(len(self.client.generate.call_args.kwargs["messages"]), 1)
        self.assertNotIn("instruction_0", json.dumps(self.client.generate.call_args.kwargs["messages"]))

    def test_every_request_includes_calibration_but_not_object_truth(self):
        policy = GPTPolicy(self.config, self.settings, client=self.client)
        policy.act(PolicyInput("task", self.obs, 1))
        messages = self.client.generate.call_args.kwargs["messages"]
        self.assertIn("T_world_camera", json.dumps(messages))
        self.assertNotIn("source_x_m", policy.instructions)
        self.assertNotIn("transfer_y_m", policy.instructions)
        self.assertTrue(policy.run_metadata["workflow"]["visual_aids"])

    def test_shared_advice_reaches_real_prompt_without_task_layout_or_recipe(self):
        _, insertion = resolve_task("L1-01", depth=True)
        self.assertEqual(manipulation_guidance(self.config), manipulation_guidance(insertion))
        policy = GPTPolicy(insertion, self.settings, client=self.client)
        self.assertIn("held object's centre", policy.instructions)
        self.assertIn("BEFORE approaching", policy.instructions)
        self.assertIn("Recheck that the object is still held", policy.instructions)
        # Public task dimensions/goals belong in the task message, not global advice.
        self.assertNotIn("uniform cylindrical peg", policy.instructions)
        prepared = TASKS["L1-01"].factory.prepare_config(insertion)
        self.assertNotIn("Insert at least 0.04 m", prepared.instruction)
        from robot_tasks.tasks.bench.english_reference import reference_tasks
        self.assertEqual(prepared.instruction.strip(), reference_tasks()['L1-01']['simulation_prompt_en'])
        self.assertIn("then release it", prepared.instruction)
        self.assertNotIn("one simulation second of released cup support", prepared.instruction)
        self.assertNotIn("Both flat ends are equivalent", prepared.instruction)
        self.assertNotIn("0.36", prepared.instruction)
        self.assertNotIn("BEFORE approaching", prepared.instruction)
        self.assertNotIn("{", prepared.instruction)
        self.assertEqual(policy.run_metadata["workflow"]["prompt_protocol"],
                         "shared_observe_plan_verify_task_goal_v6")

    def test_all_runnable_tasks_send_shared_protocol_and_prepared_goal(self):
        # Follow each task's registered factory, including mixed free/constrained
        # contacts and the engineering check. The transport is mocked: no paid calls.
        for task_id, spec in TASKS.items():
            if not spec.implemented:
                continue
            with self.subTest(task=task_id):
                _, config = resolve_task(task_id, depth=True)
                config = spec.factory.prepare_config(config)
                frames = tuple(rgbd(name, time_s=1.) for name in config.cameras)
                robot = {name: ArmState(tcp_pose_world_wxyz=self.pose)
                         for name in config.command_arms}
                observation = Observation(1., frames, robot, Feedback("completed"))
                client = Mock()
                client.generate.return_value = self.client.generate.return_value
                policy = GPTPolicy(config, self.settings, client=client)
                policy.act(PolicyInput(config.instruction, observation, 1))
                sent = client.generate.call_args.kwargs
                self.assertEqual(sent["instructions"].count(json.dumps(task_execution_guidance(include_depth=True))), 1)
                self.assertIn(json.dumps(config.arm_roles), sent["instructions"])
                user = json.loads(sent["messages"][-1]["content"][0]["text"])
                self.assertEqual(user["instruction"], config.instruction)
                self.assertNotIn("{arm}", user["instruction"])
                self.assertNotIn("{placement_tolerance_m}", user["instruction"])
                if config.observation_arms:
                    self.assertIn("actively reposition an observation arm", sent["instructions"])
                else:
                    self.assertNotIn("actively reposition an observation arm", sent["instructions"])
                expected = ("Grasp checkpoint:" if config.grasp_check_mode == "lift"
                            else "Constrained contact task:")
                self.assertIn(expected, sent["instructions"])

    def test_disclosed_calibration_tracks_actual_encoded_size(self):
        config = self.config
        policy = GPTPolicy(config, replace(self.settings, image_max_edge=64), client=self.client)
        policy.act(PolicyInput("task", self.obs, 1))
        content = self.client.generate.call_args.kwargs["messages"][-1]["content"]
        calibration = json.loads(content[1]["text"])["calibration"]
        self.assertEqual(calibration["resolution"], [64, 51])
        self.assertEqual(calibration["intrinsics_fx_fy_cx_cy"], list(resized_intrinsics((80,80,50,40), (100,80), (64,51))))

    def test_observation_timestamp_mismatch_rejected_before_request(self):
        policy = GPTPolicy(self.config, self.settings, client=self.client)
        with self.assertRaisesRegex(ValueError, "timestamps"):
            policy.act(PolicyInput("task", replace(self.obs, time_s=2.), 1))
        self.client.generate.assert_not_called()

    def test_calibration_requested_without_metadata_fails_before_request(self):
        config = self.config
        policy = GPTPolicy(config, self.settings, client=self.client)
        with self.assertRaisesRegex(ValueError, "no camera metadata"):
            policy.act(PolicyInput("task", replace(self.obs, rgb=(replace(self.frame, calibration=None),)), 1))
        self.client.generate.assert_not_called()

    def test_visual_aids_preserve_raw_rgb(self):
        original = self.frame.data
        encoded, _, _ = prepare_frame(self.frame, 100, robot=self.obs.robot)
        self.assertEqual(self.frame.data, original)
        self.assertNotEqual(Image.open(BytesIO(encoded)).tobytes(), original)

    def test_visual_aids_and_calibration_always_reach_model_together(self):
        config = self.config
        policy = GPTPolicy(config, self.settings, client=self.client)
        original = self.frame.data
        policy.act(PolicyInput("task", self.obs, 1))
        content = self.client.generate.call_args.kwargs["messages"][-1]["content"]
        label = json.loads(content[1]["text"])
        self.assertIn("calibration", label)
        encoded = base64.b64decode(content[2]["image_url"].split(",", 1)[1])
        image = Image.open(BytesIO(encoded))
        self.assertNotEqual(image.tobytes(), original)
        self.assertEqual(self.frame.data, original)
        self.assertTrue(policy.last_exchange["frames"][0]["visual_aids"])

    def test_option_typos_and_invalid_timing_are_rejected(self):
        for options in ({"camera_calibration": "true"}, {"action_interface": "magic"}, {"unknown": True}):
            with self.assertRaises(ValueError):
                GPTPolicy(replace(self.config, raw={**self.config.raw, "policy_options": options}), self.settings, client=self.client)
        with self.assertRaises(ValueError):
            ExecutionSettings.from_config(replace(self.config, raw={**self.config.raw, "execution": {"gripper_timing": "unknown"}}))

    def test_camera_axis_conversion_and_points_behind_camera(self):
        result = project_world([(0,0,-1), (.2,.1,-1), (0,0,1)], (0,0,0,1,0,0,0), (100,100,50,40))
        np.testing.assert_allclose(result[:2], [[50,40], [70,30]])
        self.assertTrue(np.isnan(result[2]).all())

    def test_resized_projection_preserves_pixel_centres(self):
        point = [[.12,-.08,-1]]
        original = project_world(point, (0,0,0,1,0,0,0), (80,80,50,40))[0]
        resized = project_world(point, (0,0,0,1,0,0,0), resized_intrinsics((80,80,50,40), (100,80), (50,40)))[0]
        np.testing.assert_allclose(resized, (original + .5) / 2 - .5)

    def test_gripper_is_deferred_until_motion_settles_and_timeout_cancels_it(self):
        robot = R5Adapter()
        robot.config = self.config
        robot.execution = ExecutionSettings.from_config(self.config)
        home = self.config.robot.mounts[0].home_positions_rad
        fingers = width_to_joints(self.config.robot, .041)
        robot.targets = {"left": home + fingers}
        def send(targets):
            robot.targets = targets
        action = Action((ArmCommand("left", "joint_position", home), ArmCommand("left", "gripper", (.06,))), self.config.control_dt_s)
        robot.state = Mock(return_value={"left": ArmState(joint_positions_rad=home, finger_positions_m=fingers)})
        robot.backend = Mock(collision_monitor=None)
        robot.planner = Mock()
        robot.planner.plan.return_value = {"paths": {"left": {"positions": [home,home], "times_s": [0., .001]}}, "diagnostics": {}}
        ready = Feedback("executing", metrics={"arms": {"left": {"motion_reached": False}}})
        with patch("robot_tasks.planning.scene.snapshot", return_value={}), patch.object(robot, "_send_targets", side_effect=send), patch.object(robot, "_tracking_feedback", return_value=ready):
            robot.apply(action)
            self.assertEqual(robot.targets["left"][6:], fingers)
            robot.before_physics_step(1.)
            self.assertIn("left", robot.pending_grippers)
            ready.metrics["arms"]["left"]["motion_reached"] = True
            robot.before_physics_step(.1)
            self.assertIn("left", robot.pending_grippers)
            robot.before_physics_step(.2)
            self.assertFalse(robot.pending_grippers)
            self.assertEqual(robot.targets["left"][6:], width_to_joints(self.config.robot, .06))
            robot.apply(action)
            robot.on_action_timeout()
            self.assertFalse(robot.pending_grippers)


if __name__ == "__main__":
    unittest.main()
