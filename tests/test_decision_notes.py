"""Offline decision annotations, including optional camera motion and depth queries."""

from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from robot_tasks.backends.mock import MockBackend, MockRobot, MockTask
from robot_tasks.policies.decision_note import split_decision_note
from robot_tasks.policies.gpt import GPTPolicy
from robot_tasks.policies.model_config import GPTSettings
from robot_tasks.policies.responses import ModelReply
from robot_tasks.policies.sparse_action import FIELDS
from robot_tasks.runner import run_episode
from robot_tasks.types import ArmState, Assessment, Feedback, InvalidPolicyAction, Observation, PolicyInput

from test_depth import configured, rgbd


SETTINGS = GPTSettings("https://example.invalid", "offline-fixture", max_calls=8)


def config():
    base = configured()
    return replace(base, arms=("left", "right"), observation_arms=(), instruction="Offline two-arm action protocol fixture.",
                   raw={**base.raw, "robot": {**base.raw["robot"], "active_arms": ["left", "right"], "observation_arms": []}})


def states(right_angle=0.):
    return {arm: ArmState(joint_positions_rad=(right_angle if arm == "right" else 0.,) * 6,
                         gripper_width_m=.07, tcp_pose_world_wxyz=(.38, y, .2, 1., 0., 0., 0.))
            for arm, y in (("left", .42), ("right", -.42))}


def document(note="The target is occluded. Move the right wrist to inspect it.", **updates):
    value = {"commands": [], "request_stop": False, "depth_queries": [], "note": note}
    value.update(updates)
    return value


def reply(doc):
    return ModelReply(json.dumps(doc), SETTINGS.model, "offline", {}, 0.)


class DecisionNoteTests(unittest.TestCase):
    def setUp(self):
        self.config = config()
        self.client = Mock()
        self.client.generate.return_value = reply(document())
        self.context = PolicyInput(self.config.instruction, Observation(
            0., tuple(rgbd(c) for c in self.config.cameras), states(), Feedback("completed")), 1)

    def test_planning_failure_is_disclosed_to_model_without_calling_provider(self):
        from robot_tasks.planning.failures import failure

        error = failure("left", "start_collision", "Start self-collision.",
                        stage="start", violated_constraints=["self_collision"])
        feedback = Feedback("rejected", str(error), {"planning_failure": error.diagnostic})
        context = replace(self.context, observation=replace(self.context.observation, feedback=feedback))
        policy = GPTPolicy(self.config, SETTINGS, client=self.client)
        message, *_ = policy.observation_message(context)
        disclosed = json.loads(message["content"][0]["text"])
        self.assertEqual(disclosed["feedback"]["metrics"]["planning_failure"], error.diagnostic)
        self.assertIn("start_collision", policy.instructions)
        self.client.generate.assert_not_called()

    def test_note_accepts_motion_wait_query_and_stop_without_altering_commands(self):
        row = {"arm": "right", **dict.fromkeys(FIELDS)}
        row["z_m"] = .25
        cases = [document(commands=[row]), document(), document(request_stop=True),
                 document(depth_queries=[{"camera": "right_wrist", "u": 16, "v": 12}])]
        for doc in cases:
            with self.subTest(doc=doc):
                self.client.generate.return_value = reply(doc)
                policy = GPTPolicy(self.config, SETTINGS, client=self.client)
                action = policy.act(self.context)
                self.assertEqual(action.note, doc["note"])
                self.assertIn("note", policy.schema["required"])
                if doc["commands"]:
                    self.assertEqual(action.commands[0].arm, "right")
                    self.assertEqual(action.commands[0].values[:3], (.38, -.42, .25))
                else:
                    self.assertFalse(action.commands)
                self.assertEqual(action.request_stop, doc["request_stop"])
                self.assertEqual(len(action.depth_queries), len(doc["depth_queries"]))

    def test_invalid_notes_and_extra_commands_fail_closed(self):
        docs = [document(note=value) for value in (None, [], " ", "a"*513, "bad\x00note")]
        docs += [{k: v for k, v in document().items() if k != "note"}, document(code="move_robot()")]
        for doc in docs:
            with self.subTest(doc=doc):
                self.client.generate.return_value = reply(doc)
                policy = GPTPolicy(self.config, SETTINGS, client=self.client)
                with self.assertRaises(InvalidPolicyAction):
                    policy.act(self.context)
                self.assertFalse(policy.history)
        with self.assertRaises(InvalidPolicyAction):
            split_decision_note('{"note":"first", "note":"second"}')

    def test_notes_survive_image_history_trimming_and_clear_on_new_run(self):
        policy = GPTPolicy(self.config, SETTINGS, client=self.client)
        for n in range(5):
            self.client.generate.return_value = reply(document(note=f"Observation {n}. Wait and inspect."))
            policy.act(replace(self.context, step=n+1))
        self.assertEqual(len(policy.history), 5)
        old_user, old_reply = policy.history[0]
        self.assertFalse(any(v["type"] == "input_image" for v in old_user["content"]))
        self.assertIn("Observation 0", old_reply["content"])
        policy.reset(0)
        self.assertFalse(policy.history)
        self.assertEqual(policy.calls, 0)

    def test_note_cannot_be_disabled_by_an_old_task_switch(self):
        current = replace(self.config, raw={**self.config.raw, "policy_options": {"decision_notes": False}})
        with self.assertRaisesRegex(ValueError, "always enabled"):
            GPTPolicy(current, SETTINGS, client=self.client)
        self.client.generate.assert_not_called()

    def test_camera_motion_then_depth_query_then_stop_share_runner_and_record_notes(self):
        class Backend(MockBackend):
            def capture_rgb(self):
                return tuple(rgbd(name, self.t, capture_id=int(self.t*100)+1) for name in self.config.cameras)

        class Robot(MockRobot):
            def state(self):
                return states()

            def feedback(self):
                return Feedback("completed")

        current = replace(self.config, raw={**self.config.raw, "execution": {"max_decisions": 3}})
        robot, task = Robot(), MockTask()
        robot.apply = Mock()
        task.evaluate = Mock(return_value=Assessment("running", "Offline fixture"))
        row = {"arm": "right", **dict.fromkeys(FIELDS), "z_m": .25}
        notes = ["Move the right camera for a clearer view.", "Query the visible object surface.", "End this offline check."]
        self.client.generate.side_effect = [reply(document(notes[0], commands=[row])),
            reply(document(notes[1], depth_queries=[{"camera": "right_wrist", "u": 16, "v": 12}])),
            reply(document(notes[2], request_stop=True))]
        policy = GPTPolicy(current, SETTINGS, client=self.client)
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(StringIO()):
            path, result = run_episode(current, Backend(), robot, task, policy, Path(temporary))
            events = [json.loads(v) for v in (path / "episode.jsonl").read_text().splitlines()]
        self.assertEqual(result["model_calls"], 3)
        self.assertEqual(result["completed_steps"], 1)  # Query and stop do not move physics in this fixture.
        self.assertFalse(result["success"])
        robot.apply.assert_called_once()
        self.assertEqual(robot.apply.call_args.args[0].commands[0].arm, "right")
        decisions = [e["action"] for e in events if e["event"] == "decision"]
        self.assertEqual([a["note"] for a in decisions], notes)
        messages = self.client.generate.call_args_list[-1].kwargs["messages"]
        feedback = json.loads(messages[-1]["content"][0]["text"])["feedback"]
        self.assertEqual(feedback["metrics"]["depth_queries"][0]["camera"], "right_wrist")
        self.assertIn(notes[0], messages[1]["content"])


if __name__ == "__main__":
    unittest.main()
