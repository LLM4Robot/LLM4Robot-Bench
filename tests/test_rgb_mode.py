"""RGB defaults and depth opt-in contracts, without Isaac or provider calls."""

from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from robot_tasks.backends.mock import MockBackend, MockRobot, MockTask
from robot_tasks.cli import main
from robot_tasks.config import load_config
from robot_tasks.policies.gpt import GPTPolicy
from robot_tasks.policies.model_config import GPTSettings
from robot_tasks.policies.responses import ModelReply
from robot_tasks.registry import TASKS, resolve_task
from robot_tasks.runner import run_episode, validate_action
from robot_tasks.types import Action, ArmState, Assessment, DepthQuery, Feedback, Observation, PolicyInput

from fixtures import rgbd, gripper_command


ROOT = Path(__file__).resolve().parents[1]
SETTINGS = GPTSettings("https://example.invalid", "offline", max_calls=3)


class RGBModeTests(unittest.TestCase):
    def test_all_tasks_support_rgb_default_and_explicit_depth(self):
        for task_id, spec in TASKS.items():
            if not spec.implemented:
                continue
            for depth in (None, True, False):
                with self.subTest(task=task_id, depth=depth):
                    _, config = resolve_task(task_id, depth=depth)
                    config = spec.factory.prepare_config(config)
                    self.assertEqual(config.include_depth, bool(depth))
                    self.assertEqual(config.save_depth, bool(depth))
                    frames = tuple(rgbd(c) for c in config.cameras)
                    if not depth:
                        frames = tuple(replace(f, depth=None) for f in frames)
                    policy = GPTPolicy(config, SETTINGS, client=Mock())
                    context = PolicyInput(config.instruction, Observation(0., frames, {}, Feedback("completed")), 1)
                    message, info, _, _ = policy.observation_message(context)
                    count = sum(item["type"] == "input_image" for item in message["content"])
                    self.assertEqual(count, len(config.cameras) * (2 if depth else 1))
                    self.assertEqual("depth_queries" in policy.schema["properties"], bool(depth))
                    self.assertEqual("depth" in policy.run_metadata["observation_modalities"], bool(depth))
                    self.assertEqual(any("depth" in f for f in info), bool(depth))
                    if not depth:
                        self.assertNotIn("depth queries or", policy.instructions)
                        self.assertNotIn("Query depth", config.instruction)
                        self.assertNotIn("RGB-D", config.instruction)
                        self.assertIsNone(policy.run_metadata["workflow"]["release_check"])

    def test_cli_overrides_are_forwarded_and_recorded(self):
        for flag, depth in (([], False), (["--depth"], True), (["--no-depth"], False)):
            with self.subTest(flag=flag), redirect_stdout(StringIO()), \
                    patch("robot_tasks.cli.execute_task", return_value=(ROOT, {"status": "finished"})) as execute:
                self.assertEqual(main(["run", "L1-01", "--output", str(ROOT), *flag]), 0)
                config = execute.call_args.args[1]
                self.assertEqual(config.include_depth, depth)
                self.assertEqual(config.raw["observation"]["include_depth"], depth)
                self.assertEqual(config.save_depth, depth)
            with redirect_stdout(StringIO()) as output:
                self.assertEqual(main(["inspect", "L2-02", *flag]), 0)
            inspected = json.loads(output.getvalue())
            self.assertEqual("depth" in inspected["workflow"]["observation_modalities"], depth)
            self.assertEqual(inspected["workflow"]["grasp_check"], "contact_inspection_no_lift_gate")

    def test_toml_opt_in_and_command_line_override(self):
        from robot_tasks.batch import toml_text

        _, base = resolve_task("GPT-PICK-PLACE", depth=True)
        raw = {**base.raw, "workflow_profile": str(base.workflow_source),
               "robot": {**base.raw["robot"], "profile": str(base.robot.source)},
               "simulation": {**base.raw["simulation"], "camera_profile": str(
                   (base.source.parent / base.raw["simulation"]["camera_profile"]).resolve())}}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "task.toml"
            path.write_text(toml_text(raw), encoding="utf-8")
            self.assertTrue(load_config(path).include_depth)
            off = load_config(path, depth=False)
            self.assertFalse(off.include_depth)
            self.assertFalse(off.save_depth)
            self.assertFalse(off.raw["recording"]["save_depth"])

    def test_rgb_episode_executes_without_depth_guard_or_hidden_depth_recording(self):
        class Backend(MockBackend):
            def capture_rgb(self):
                return tuple(replace(rgbd(c, self.t), depth=None) for c in self.config.cameras)

        _, config = resolve_task("GPT-PICK-PLACE")
        config = replace(config, raw={**config.raw, "execution": {"max_action_steps": 1, "max_decisions": 3}})
        backend, robot, task, client = Backend(), MockRobot(), MockTask(), Mock()
        robot.state = Mock(return_value={arm: ArmState(gripper_width_m=.07,
            tcp_pose_world_wxyz=(.3, .2, .2, 1., 0., 0., 0.)) for arm in config.command_arms})
        robot.apply = Mock()
        task.evaluate = Mock(return_value=Assessment("running", "Offline evaluation, no measured success"))
        documents = [
            {"commands": [gripper_command()], "request_stop": False, "note": "Close in the offline fixture."},
            {"commands": [{**gripper_command(), "gripper": None, "x_m": .4, "z_m": .3}],
             "request_stop": False, "note": "Move in the offline fixture."},
            {"commands": [], "request_stop": True, "note": "Request fixture completion."},
        ]
        client.generate.side_effect = [ModelReply(json.dumps(doc), "offline", "fixture", {}, 0.) for doc in documents]
        policy = GPTPolicy(config, SETTINGS, client=client)
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(StringIO()), \
                patch("robot_tasks.grasp_guard.GraspGuard", side_effect=AssertionError("RGB must not instantiate a depth guard")):
            path, result = run_episode(config, backend, robot, task, policy, Path(temporary))
            self.assertEqual(result["status"], "finished")
            self.assertFalse(result["success"])  # The independent evaluator is still authoritative.
            self.assertEqual(result["executed_actions"], 2)
            self.assertEqual(result["rejected_stop_requests"], 1)
            self.assertEqual(result["grasp_check_protocol"], "model_visual_inspection")
            self.assertIsNone(result["release_check_protocol"])
            self.assertEqual(robot.apply.call_count, 2)
            self.assertFalse((path / "depth").exists())
            manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["observation_modalities"], ["rgb"])
            self.assertFalse(manifest["depth"]["enabled"])
            self.assertFalse(manifest["depth"]["recorded"])
        for call in client.generate.call_args_list:
            observation = json.loads(call.kwargs["messages"][-1]["content"][0]["text"])
            for key in ("depth_queries", "grasp_check", "release_check"):
                self.assertNotIn(key, observation["feedback"]["metrics"])

    def test_depth_queries_cannot_be_executed_in_rgb_mode(self):
        _, config = resolve_task("GPT-PICK-PLACE")
        action = Action((), config.control_dt_s, depth_queries=(DepthQuery("overview", 1, 1, 32, 24),))
        with self.assertRaisesRegex(ValueError, "Depth queries require"):
            validate_action(action, config)
        client = Mock()
        client.generate.return_value = ModelReply(json.dumps({"commands": [], "request_stop": False,
            "depth_queries": [{"camera": "overview", "u": 1, "v": 1}], "note": "Invalid unavailable tool."}),
            "offline", "fixture", {}, 0.)
        policy = GPTPolicy(config, SETTINGS, client=client)
        observation = Observation(0., tuple(replace(rgbd(c), depth=None) for c in config.cameras), {}, Feedback("completed"))
        with self.assertRaises(ValueError):
            policy.act(PolicyInput(config.instruction, observation, 1))


if __name__ == "__main__":
    unittest.main()
