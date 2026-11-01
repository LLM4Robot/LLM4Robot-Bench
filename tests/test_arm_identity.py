"""Real skill names, simulator IDs and camera owners agree without API calls."""

from dataclasses import replace
import json
import unittest
from unittest.mock import Mock

from fixtures import rgbd, robot_config
from robot_tasks.cameras.geometry import project_world
from robot_tasks.cameras.profile import load_camera_rig
from robot_tasks.policies.gpt import GPTPolicy
from robot_tasks.policies.model_config import GPTSettings
from robot_tasks.policies.sparse_action import FIELDS, decode_sparse_action
from robot_tasks.registry import TASKS, resolve_task
from robot_tasks.robot_roles import ARM_IDENTITY_PROTOCOL, arm_identities
from robot_tasks.types import ArmState, Feedback, Observation, PolicyInput


class ArmIdentityTests(unittest.TestCase):
    def config(self, task="L3-01", *, depth=False):
        spec, config = resolve_task(task, depth=depth)
        return spec.factory.prepare_config(config)

    def test_mapping_matches_actual_base_projection_and_camera_parent(self):
        config = self.config()
        identities = arm_identities(config)
        rig = load_camera_rig(config.source.parent / config.raw["simulation"]["camera_profile"])
        overview, = rig.select(("overview",))
        bases = {mount.name: mount.base_pose_wxyz for mount in config.robot.mounts}
        pixels = project_world([bases[a][:3] for a in ("left", "right")],
                               overview.optical_pose_wxyz, overview.intrinsics)
        self.assertLess(pixels[0, 0], pixels[1, 0])
        self.assertEqual(identities["right"]["real_arm"], "left")
        self.assertEqual(identities["right"]["real_wrist_camera_key"], "gemini")
        self.assertEqual(identities["left"]["real_arm"], "right")
        self.assertEqual(identities["left"]["real_wrist_camera_key"], "gemini_right")
        for arm, row in identities.items():
            camera, = rig.select((row["wrist_camera"],))
            self.assertEqual(camera.parent, arm)
            self.assertTrue(camera.existing_prim_path.startswith(
                next(m.prim_path for m in config.robot.mounts if m.name == arm) + "/"))

    def test_every_lab_task_sends_camera_owner_and_prepared_roles(self):
        settings = GPTSettings("https://example.invalid", "offline")
        for task_id, spec in TASKS.items():
            if not spec.implemented:
                continue
            config = self.config(task_id)
            if not arm_identities(config):
                continue
            with self.subTest(task=task_id):
                policy = GPTPolicy(config, settings, client=Mock())
                frames = tuple(replace(rgbd(c), depth=None) for c in config.cameras)
                # Hands may cross the image; ownership must follow the base ID.
                states = {"left": ArmState(tcp_pose_world_wxyz=(.7, 1., 1., 1., 0., 0., 0.)),
                          "right": ArmState(tcp_pose_world_wxyz=(-.1, 1., 1., 1., 0., 0., 0.))}
                context = PolicyInput(config.instruction, Observation(0., frames, states, Feedback("completed")), 1)
                message, frame_info, _, _ = policy.observation_message(context)
                labels = [json.loads(item["text"]) for item in message["content"] if item["type"] == "input_text"]
                labels = {item["camera"]: item for item in labels if "camera" in item}
                for arm, role in config.arm_roles.items():
                    label = labels[arm + "_wrist"]
                    self.assertEqual(label["owner_arm"], arm)
                    self.assertEqual(label["owner_role"], role)
                    self.assertEqual(next(f for f in frame_info if f["camera"] == arm + "_wrist")["owner_arm"], arm)
                self.assertEqual(policy.run_metadata["arm_identity_protocol"], ARM_IDENTITY_PROTOCOL)
                self.assertEqual(policy.run_metadata["arm_identities"], arm_identities(config))
                if config.raw.get("task", {}).get("scene_delivery"):
                    self.assertIn("ROBOT'S perspective", policy.instructions)
                    self.assertIn("LEFT arm is on overview-image RIGHT", policy.instructions)
                else:
                    self.assertIn("do not swap them again", policy.instructions)
                self.assertIn("stationary support or background", policy.instructions)
                policy.client.generate.assert_not_called()

    def test_depth_keeps_same_owner_as_rgb(self):
        config = self.config(depth=True)
        policy = GPTPolicy(config, GPTSettings("https://example.invalid", "offline"), client=Mock())
        frames = tuple(rgbd(c) for c in config.cameras)
        context = PolicyInput(config.instruction, Observation(0., frames, {}, Feedback("completed")), 1)
        message, _, _, _ = policy.observation_message(context)
        labels = [json.loads(i["text"]) for i in message["content"] if i["type"] == "input_text"]
        labels = [x for x in labels if x.get("camera") == "right_wrist"]
        self.assertEqual({x["modality"] for x in labels}, {"rgb", "depth"})
        self.assertTrue(all(x["owner_arm"] == "right" and x["owner_role"] == "manipulation" for x in labels))

    def test_commands_are_not_swapped_again_or_world_targets_mirrored(self):
        config = self.config()
        states = {"right": ArmState(tcp_pose_world_wxyz=(.4, 1.2, 1., 1., 0., 0., 0.))}
        observation = Observation(0., (), states, Feedback("completed"))
        row = {"arm": "right", **dict.fromkeys(FIELDS), "x_m": .35}
        action = decode_sparse_action(json.dumps({"commands": [row], "request_stop": False}), config, observation)
        command, = action.commands
        self.assertEqual(command.arm, "right")
        self.assertEqual(command.values[:3], (.35, 1.2, 1.))
        for wrong in ("can0", "real_left", "gemini"):
            with self.subTest(alias=wrong), self.assertRaises(ValueError):
                decode_sparse_action(json.dumps({"commands": [{**row, "arm": wrong}], "request_stop": False}), config, observation)

    def test_observer_does_not_inherit_real_working_arm_permission(self):
        config = self.config("L2-07")
        row = {"arm": "left", **dict.fromkeys(FIELDS), "gripper": 0.}
        with self.assertRaisesRegex(ValueError, "Observation arm left"):
            decode_sparse_action(json.dumps({"commands": [row], "request_stop": False}), config,
                                 Observation(0., (), {}, Feedback("completed")))

    def test_other_environment_does_not_claim_lab_mapping(self):
        config = robot_config()
        self.assertEqual(arm_identities(config), {})


if __name__ == "__main__":
    unittest.main()
