"""Offline coverage of workflow reuse, arm permissions and effective evidence."""

from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from robot_tasks.backends.mock import MockBackend, MockRobot, MockTask
from robot_tasks.config import ExecutionSettings, load_config
from robot_tasks.execution import _gpt_policy
from robot_tasks.policies.gpt import GPTPolicy
from robot_tasks.policies.model_config import GPTSettings, load_gpt_settings
from robot_tasks.policies.responses import ModelReply
from robot_tasks.policies.sparse_action import FIELDS
from robot_tasks.registry import TASKS, get_spec, resolve_task
from robot_tasks.robot_roles import FixedArmMonitor
from robot_tasks.runner import run_episode, validate_action
from robot_tasks.tasks.pick_place import SingleArmPickPlace
from robot_tasks.types import Action, ArmCommand, ArmState, Assessment, Feedback, Observation, PolicyInput
from robot_tasks.workflow import load_workflow, workflow_metadata

from test_depth import rgbd


ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "configs/checks/gpt_single_arm_pick_place.toml"


def states(right_angle=0.):
    return {arm: ArmState(joint_positions_rad=(right_angle if arm == "right" else 0.,) * 6,
                         gripper_width_m=.07, tcp_pose_world_wxyz=(.38, y, .2, 1., 0., 0., 0.))
            for arm, y in (("left", .42), ("right", -.42))}


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(TASK)

    def test_all_tasks_use_the_single_default_and_allow_explicit_model_selection(self):
        default = ROOT / "configs/models/gpt.toml"
        settings = load_gpt_settings(default)
        self.assertEqual(settings.transport, "codex_cli")
        with tempfile.TemporaryDirectory() as temporary:
            override = Path(temporary) / "model.toml"
            override.write_text(default.read_text(encoding="utf-8").replace(
                f'reasoning_effort = "{settings.reasoning_effort}"',
                'reasoning_effort = "high"'), encoding="utf-8")
            with patch("robot_tasks.policies.codex_cli.CodexProcess"), \
                    patch("robot_tasks.policies.codex_cli.CodexCLIClient.require_chatgpt_login"), \
                    patch("robot_tasks.execution.read_api_key", side_effect=AssertionError("No API call")):
                for task_id, spec in TASKS.items():
                    if not spec.implemented:
                        continue
                    _, config = resolve_task(task_id)
                    selected = config.source.parent / config.raw["policy"]["model_config"]
                    self.assertEqual(selected.resolve(), default.resolve())
                    for model, expected in ((None, settings), (override, replace(settings, reasoning_effort="high"))):
                        with self.subTest(task=task_id, model=model):
                            policy = _gpt_policy(config, model)
                            self.assertEqual(policy.settings, replace(
                                expected, max_calls=ExecutionSettings.from_config(config).max_decisions))

    def test_explicit_experiment_reasoning_override_is_preserved(self):
        config = replace(self.config, raw={**self.config.raw, "policy": {
            **self.config.raw["policy"], "reasoning_effort": "high"}})
        with patch("robot_tasks.execution.load_gpt_settings", return_value=GPTSettings("https://example.org", "fixture")), \
                patch("robot_tasks.execution.read_api_key", return_value="offline-fixture"):
            policy = _gpt_policy(config, None)
        self.assertEqual(policy.settings.reasoning_effort, "high")

    def test_default_task_uses_shared_features_and_effective_model_budget(self):
        self.assertEqual(get_spec(self.config).task_id, "GPT-PICK-PLACE")
        self.assertEqual(self.config.arms, ("left",))
        self.assertEqual(self.config.command_arms, ("left", "right"))
        self.assertEqual(self.config.arm_roles, {"left": "manipulation", "right": "observation"})
        self.assertFalse(self.config.include_depth)
        self.assertFalse(self.config.save_depth)
        self.assertEqual(self.config.max_wall_time_s, 18000.0)
        self.assertEqual(ExecutionSettings.from_config(self.config).max_decisions, 150)
        settings = GPTSettings("https://example.invalid", "fixture", reasoning_effort="low", max_calls=2)
        with patch("robot_tasks.execution.load_gpt_settings", return_value=settings), \
                patch("robot_tasks.execution.read_api_key", return_value="offline-fixture"):
            policy = _gpt_policy(self.config, Path("explicit-provider-config.toml"))
        self.assertEqual(policy.settings.reasoning_effort, "low")
        self.assertEqual(policy.settings.max_calls, 150)
        self.assertEqual(policy.run_metadata["workflow"], workflow_metadata())
        self.assertNotIn("depth_queries", policy.schema["required"])
        self.assertIn("note", policy.schema["required"])
        self.assertNotIn("must remain still", SingleArmPickPlace.prepare_config(self.config).instruction)

    def test_profile_can_be_reused_and_overrides_do_not_mutate_defaults(self):
        shared = self.config.workflow_source
        a, _, profile = load_workflow(TASK, {"workflow_profile": str(shared),
            "task": {"id": "unrelated-task"}, "execution": {"max_decisions": 7}})
        b, _, _ = load_workflow(TASK, {"workflow_profile": str(shared), "task": {"id": "another-task"}})
        self.assertEqual(a["execution"]["max_decisions"], 7)
        self.assertEqual(a["observation"], b["observation"])
        self.assertFalse(a["observation"]["include_depth"])
        self.assertEqual(profile["execution"]["max_decisions"], 150)
        self.assertEqual(b["execution"]["max_decisions"], 150)

    def test_profiles_cannot_hide_task_assets_credentials_or_recursive_inheritance(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "workflow.toml"
            for text in ('[task]\nid="hidden"', '[policy]\nmodel_config="private.toml"',
                         'workflow_profile="other.toml"', 'execution=1',
                         '[simulation]\ncamera_profile="hidden.toml"'):
                path.write_text(text)
                with self.subTest(text=text), self.assertRaises(ValueError):
                    load_workflow(TASK, {"workflow_profile": str(path)})

    def test_fixed_capabilities_do_not_depend_on_a_task_id_or_profile(self):
        for task_id in ("GPT-PICK-PLACE", "NEW-TASK"):
            raw, source, _ = load_workflow(TASK, {"task": {"id": task_id}})
            self.assertIsNone(source)
            self.assertEqual(raw["observation"], {
                "include_depth": False, "include_robot_state": True, "include_feedback": True})
            config = replace(self.config, task_id=task_id, workflow_source=None,
                             raw={**self.config.raw, **raw})
            policy = GPTPolicy(config, GPTSettings("https://example.invalid", "fixture"), client=Mock())
            self.assertEqual(policy.run_metadata["workflow"], workflow_metadata())
            self.assertEqual(set(policy.schema["required"]), {"commands", "request_stop", "note"})
            self.assertEqual(ExecutionSettings.from_config(config).gripper_timing, "after_motion")

    def test_task_and_profile_cannot_disable_fixed_capabilities(self):
        changes = [{"policy_options": {"visual_aids": False}},
                   {"execution": {"gripper_timing": "simultaneous"}}]
        changes += [{"observation": {key: False}}
                    for key in ("include_robot_state", "include_feedback")]
        for change in changes:
            for reference in ({}, {"workflow_profile": str(self.config.workflow_source)}):
                with self.subTest(change=change, profile=bool(reference)), self.assertRaises(ValueError):
                    load_workflow(TASK, {**reference, **change})
        with tempfile.TemporaryDirectory() as temporary:
            profile = Path(temporary) / "workflow.toml"
            profile.write_text('[observation]\ninclude_robot_state=false\n')
            with self.assertRaisesRegex(ValueError, "fixed GPT workflow"):
                load_workflow(TASK, {"workflow_profile": str(profile), "observation": {"include_robot_state": True}})

    def test_invalid_roles_and_cameras_are_rejected_before_execution(self):
        text = TASK.read_text().replace('"../workflows/gpt_default.toml"', json.dumps(self.config.workflow_source.as_posix()))
        text = text.replace('"../robots/r5_sim.toml"', json.dumps(self.config.robot.source.as_posix()))
        camera = ROOT / "configs/cameras/r5_orbbec_provisional.toml"
        text = text.replace('"../cameras/r5_orbbec_provisional.toml"', json.dumps(camera.as_posix()))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "task.toml"
            for modified, message in (
                (text.replace('observation_arms = ["right"]', 'observation_arms = ["left"]'), "distinct"),
                (text.replace('observation_arms = ["right"]', 'observation_arms = ["missing"]'), "available"),
                (text + '\n[observation]\ncameras=["overview"]\n', "carry a selected camera")):
                path.write_text(modified)
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    load_config(path)

    def test_observer_motion_uses_same_action_contract_and_never_gets_gripper_permission(self):
        motion = Action((ArmCommand("right", "cartesian_pose", (.3, -.2, .3, 1., 0., 0., 0.)),), self.config.control_dt_s)
        validate_action(motion, self.config)
        with self.assertRaisesRegex(ValueError, "Observation arm"):
            validate_action(replace(motion, commands=(ArmCommand("right", "gripper", (.04,)),)), self.config)
        with self.assertRaisesRegex(ValueError, "Unknown arm"):
            validate_action(motion, replace(self.config, observation_arms=()))
        # A dual-manipulation task uses this exact validator and can grip with either arm.
        both = replace(self.config, arms=("left", "right"), observation_arms=())
        validate_action(replace(motion, commands=(ArmCommand("right", "gripper", (.04,)),)), both)

    def test_generic_fixed_arm_monitor_distinguishes_observation_and_unauthorized_drift(self):
        for status in ("running", "succeeded"):
            assessment = Assessment(status, "Task-specific result", {"task_metric": 1})
            moving = FixedArmMonitor(self.config, .01)
            moving.update(states(), assessment)
            self.assertEqual(moving.update(states(.2), assessment).status, status)
            fixed = FixedArmMonitor(replace(self.config, observation_arms=()), .01)
            fixed.update(states(), assessment)
            result = fixed.update(states(.2), assessment)
            self.assertEqual(result.status, "failed")
            self.assertFalse(result.retryable)
            self.assertEqual(result.metrics["task_metric"], 1)
            fixed.reset()
            self.assertEqual(fixed.update(states(.2), assessment).status, status)

    def test_observer_motion_depth_and_notes_use_common_runner_and_preserve_actual_settings(self):
        self.config = load_config(TASK, depth=True)
        class Backend(MockBackend):
            def capture_rgb(self):
                return tuple(rgbd(c, self.t) for c in self.config.cameras)

        class Robot(MockRobot):
            def state(self):
                return states(self.backend.t)

            def initialize(self, backend, config):
                super().initialize(backend, config)
                self.backend = backend

            def feedback(self):
                return Feedback("completed")

        config = replace(self.config, raw={**self.config.raw, "execution": {
            "max_action_steps": 1, "max_decisions": 3}})
        row = {"arm": "right", **dict.fromkeys(FIELDS), "z_m": .3}
        responses = [
            {"commands": [row], "request_stop": False, "depth_queries": [], "note": "Inspect the target."},
            {"commands": [], "request_stop": False, "depth_queries": [{"camera": "right_wrist", "u": 12, "v": 10}], "note": "Query visible depth."},
            {"commands": [], "request_stop": True, "depth_queries": [], "note": "End the offline fixture."}]
        client = Mock()
        client.generate.side_effect = [ModelReply(json.dumps(v), "fixture", "offline", {}, 0.) for v in responses]
        policy = GPTPolicy(config, GPTSettings("https://example.invalid", "fixture", max_calls=3), client=client)
        robot, task = Robot(), MockTask()
        robot.apply = Mock()
        task.evaluate = Mock(return_value=Assessment("running", "Not yet finished"))
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(StringIO()):
            output, result = run_episode(config, Backend(), robot, task, policy, Path(temporary))
            manifest = json.loads((output / "manifest.json").read_text())
            effective = json.loads((output / "effective_config.json").read_text())
            profile = json.loads((output / "workflow_profile.json").read_text())
            self.assertEqual(manifest["arm_roles"], config.arm_roles)
            self.assertEqual(effective["execution"]["max_decisions"], 3)
            self.assertEqual(profile["execution"]["max_decisions"], 150)
        self.assertEqual(result["model_calls"], 3)
        self.assertEqual(result["completed_steps"], 1)
        self.assertFalse(result["success"])
        robot.apply.assert_called_once()
        self.assertEqual(robot.apply.call_args.args[0].commands[0].arm, "right")
        messages = client.generate.call_args.kwargs["messages"]
        feedback = json.loads(messages[-1]["content"][0]["text"])["feedback"]
        self.assertEqual(feedback["metrics"]["depth_queries"][0]["camera"], "right_wrist")
        self.assertEqual(policy.run_metadata["arm_roles"]["right"], "observation")


if __name__ == "__main__":
    unittest.main()
