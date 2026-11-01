"""An unsuccessful stop consumes budget but cannot end an otherwise running task."""

from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from fixtures import robot_config, RGBDBackend
from robot_tasks.backends.mock import MockBackend, MockRobot, MockTask
from robot_tasks.backends.mock import NoopPolicy
from robot_tasks.runner import run_episode
from robot_tasks.types import Action, Assessment


class StopProtocolTests(unittest.TestCase):
    def test_100_rejected_stops_retain_scene_and_only_disclose_incomplete_flag(self):
        base = robot_config()
        config = replace(
            base, max_steps=0, raw={**base.raw, "execution": {"max_decisions": 100}}
        )

        class Task(MockTask):
            def evaluate(self, state, feedback):
                return Assessment(
                    "running", "hidden detailed reason", {"object_pose": [1, 2, 3]}
                )

        class Policy(NoopPolicy):
            def act(self, context):
                self.contexts.append(context)
                return Action((), self.duration_s, True)

        policy = Policy(config.control_dt_s)
        policy.contexts = []
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            output, result = run_episode(
                config, RGBDBackend(), MockRobot(), Task(), policy, Path(directory)
            )
            events = [
                json.loads(x)
                for x in (output / "episode.jsonl").read_text().splitlines()
            ]
        self.assertEqual(result["decisions"], 100)
        self.assertEqual(result["rejected_stop_requests"], 100)
        self.assertEqual(result["completed_steps"], 0)
        self.assertFalse(result["success"])
        self.assertEqual(result["termination_reason"], "model_call_limit")
        self.assertEqual(len([e for e in events if e["event"] == "reset"]), 1)
        for context in policy.contexts[1:]:
            self.assertEqual(context.observation.feedback.status, "stop_rejected")
            self.assertNotIn("object_pose", str(context))
            self.assertNotIn("hidden detailed reason", str(context))
        self.assertEqual(
            policy.contexts[-1].observation.feedback.metrics["remaining_decisions"], 1
        )
