"""Offline grasp checkpoint tests: rendered RGB-D, no simulator truth input/API."""

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from fixtures import decision_document, rgbd, robot_config

from robot_tasks.backends.mock import MockBackend, MockRobot, MockTask
from robot_tasks.grasp_guard import GraspGuard
from robot_tasks.policies.grasp_comparison import comparison_images
from robot_tasks.robots.transforms import rotation_quaternion
from robot_tasks.runner import run_episode
from robot_tasks.types import (
    Action,
    ArmCommand,
    ArmState,
    Assessment,
    DepthFrame,
    Feedback,
    GraspReference,
    Observation,
    PolicyInput,
    RGBFrame,
)

DOWN = (2**-0.5, 0.0, 2**-0.5, 0.0)


def rendered_view(name, eye, height, time_s, *, marker_z=None, object_present=True):
    """Analytic ray/box RGB-D fixture, simulating only a table and a lifted block.

    The checker receives pixels/calibration, never the box position used here.
    """
    eye = np.array(eye)
    backward = eye - [0.36, 0.4, 0.04]
    backward /= np.linalg.norm(backward)
    right = np.cross([0, 0, 1], backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    rotation = np.column_stack((right, up, backward))
    width, h = 240, 180
    v, u = np.mgrid[:h, :width]
    rays = (
        np.stack(((u - width / 2) / 230, -(v - h / 2) / 230, -np.ones_like(u)), axis=-1)
        @ rotation.T
    )
    low, high = (
        np.array([0.3425, 0.3825, height]),
        np.array([0.3775, 0.4175, height + 0.06]),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        plane = -eye[2] / rays[:, :, 2]
        a, b = (low - eye) / rays, (high - eye) / rays
        enter, leave = (
            np.min(np.stack((a, b)), axis=0).max(axis=2),
            np.max(np.stack((a, b)), axis=0).min(axis=2),
        )
    hit = (leave >= enter) & (enter > 0) & (enter < plane) & object_present
    depth = np.where(hit, enter, plane).astype("<f4")
    depth[depth <= 0] = np.nan
    colours = np.zeros((h, width, 3), np.uint8)
    colours[:] = [120, 110, 100]
    colours[hit] = [230, 90, 20]
    if marker_z is not None:
        offset = eye - [0.36, 0.4, marker_z]
        a = np.sum(rays * rays, axis=2)
        b = 2 * np.sum(rays * offset, axis=2)
        c = np.sum(offset * offset) - 0.004**2
        discriminant = b * b - 4 * a * c
        t = (-b - np.sqrt(np.maximum(discriminant, 0))) / (2 * a)
        marker_hit = (discriminant >= 0) & (t > 0) & (t < depth)
        depth[marker_hit] = t[marker_hit]
        colours[marker_hit] = [0, 230, 230]
    capture = int(time_s * 10) + 1
    pose = [*eye, *rotation_quaternion(rotation)]
    calibration = {
        "intrinsics_fx_fy_cx_cy": [230.0, 230.0, width / 2, h / 2],
        "distortion": [0.0] * 5,
        "resolution": [width, h],
        "T_world_camera_pose_wxyz": pose,
        "capture_id": capture,
        "simulation_time_s": time_s,
    }
    return RGBFrame(
        name,
        time_s,
        width,
        h,
        colours.tobytes(),
        calibration,
        DepthFrame(name, time_s, width, h, depth.tobytes(), capture),
    )


def observation(*, rise=0.0, object_rise=0.0, width=0.07, time_s=0.0, eye_shift=0.0):
    frames = (
        rendered_view("overview", [0.70, -0.10, 0.28], object_rise, time_s),
        rendered_view(
            "right_wrist", [0.10 + eye_shift, 0.10, 0.17], object_rise, time_s
        ),
    )
    state = ArmState(
        tcp_pose_world_wxyz=(0.36, 0.4, 0.035 + rise, *DOWN), gripper_width_m=width
    )
    return Observation(
        time_s, frames, {"left": state, "right": state}, Feedback("completed")
    )


def keep_fixture_surfaces(points, width_m, profile):
    """The analytic RGB-D fixture contains no rendered gripper to exclude.

    Isolate mesh filtering, not surface extraction or evidence decisions.
    Real R5 mesh exclusion is covered in test_thin_grasp_evidence.
    """
    return np.ones(len(points), dtype=bool)


def close():
    return Action((ArmCommand("left", "gripper", (0.012,)),), 1 / 12)


def move(x=0.36, y=0.4, z=0.060, arm="left"):
    return Action((ArmCommand(arm, "cartesian_pose", (x, y, z, *DOWN)),), 1 / 12)


class GraspEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.guard = GraspGuard(robot_config())
        self.before = observation()
        self.guard.accepted(close(), self.before)

    def test_visible_lift_is_verified_and_transport_allowed(self):
        after = self.guard.enrich(
            observation(rise=0.025, object_rise=0.025, width=0.035, time_s=1.0)
        )
        evidence = after.feedback.metrics["grasp_check"]["left"]
        self.assertEqual(evidence["status"], "verified", evidence)
        self.assertGreaterEqual(evidence["newly_occupied_points"], 6)
        self.assertGreaterEqual(evidence["vacated_points"], 6)
        self.assertIsNone(self.guard.reject_reason(move(y=0.25, z=0.15), after))
        self.assertEqual(after.grasp_references, ())

    def test_uniform_stationary_object_does_not_pass_even_with_nonminimum_width(self):
        after = self.guard.enrich(observation(rise=0.025, width=0.035, time_s=1.0))
        evidence = after.feedback.metrics["grasp_check"]["left"]
        self.assertEqual(evidence["status"], "unknown", evidence)
        self.assertFalse(evidence["visible_motion_supported"])
        self.assertIsNotNone(self.guard.reject_reason(move(y=0.25), after))

    def test_empty_closure_does_not_gain_success_from_hand_motion_or_note(self):
        after = self.guard.enrich(observation(rise=0.025, width=0.012, time_s=1.0))
        self.assertEqual(
            after.feedback.metrics["grasp_check"]["left"]["status"], "empty_suspected"
        )
        action = replace(
            move(y=0.25), note="I have definitely grasped and lifted the object."
        )
        self.assertIsNotNone(self.guard.reject_reason(action, after))
        self.assertEqual(self.guard.verifications, 0)

    def test_camera_motion_alone_is_not_object_motion(self):
        after = self.guard.enrich(
            observation(rise=0.025, width=0.035, time_s=1.0, eye_shift=0.09)
        )
        self.assertNotEqual(
            after.feedback.metrics["grasp_check"]["left"]["status"], "verified"
        )

    def test_initial_unknown_lift_cannot_be_reset_by_air_release(self):
        unknown = self.guard.enrich(observation(rise=0.025, width=0.035, time_s=1.0))
        opening = Action((ArmCommand("left", "gripper", (0.07,)),), 1/12)
        self.assertIsNotNone(self.guard.reject_reason(opening, unknown))
        self.assertIsNone(self.guard.reject_reason(move(z=0.035), unknown))

    def test_width_change_cannot_be_hidden_by_matching_surfaces(self):
        self.guard.enrich(observation(rise=0.15, object_rise=0.15, width=0.035, time_s=1.0))
        changed = self.guard.enrich(observation(rise=0.15, object_rise=0.15, width=0.041, time_s=2.0))
        self.assertEqual(changed.feedback.metrics["grasp_check"]["left"]["status"], "tracking_uncertain")
        self.assertEqual(self.guard.attempts["left"].last_verified_width, 0.035)

    def test_actual_object_motion_with_changed_camera_extrinsics_is_supported(self):
        after = self.guard.enrich(
            observation(
                rise=0.025, object_rise=0.025, width=0.035, time_s=1.0, eye_shift=0.03
            )
        )
        self.assertEqual(
            after.feedback.metrics["grasp_check"]["left"]["status"], "verified"
        )

    def test_nan_occluded_depth_stays_unknown(self):
        after = observation(rise=0.025, object_rise=0.025, width=0.035, time_s=1.0)
        frames = tuple(
            replace(
                f,
                depth=replace(
                    f.depth, data=np.full((f.height, f.width), np.nan, "<f4").tobytes()
                ),
            )
            for f in after.rgb
        )
        after = self.guard.enrich(replace(after, rgb=frames))
        self.assertEqual(
            after.feedback.metrics["grasp_check"]["left"]["status"], "unknown"
        )

    def test_rendered_tcp_sphere_is_not_counted_as_a_carried_object(self):
        def marker_observation(rise, width, time_s):
            value = observation(rise=rise, width=width, time_s=time_s)
            frame = rendered_view(
                "right_wrist",
                [0.10, 0.10, 0.17],
                0.0,
                time_s,
                marker_z=0.035 + rise,
                object_present=False,
            )
            return replace(value, rgb=(frame,))

        guard = GraspGuard(robot_config())
        guard.accepted(close(), marker_observation(0.0, 0.07, 0.0))
        self.assertEqual(sum(len(s.points) for s in guard.attempts["left"].surfaces), 0)
        after = guard.enrich(marker_observation(0.025, 0.035, 1.0))
        self.assertEqual(
            after.feedback.metrics["grasp_check"]["left"]["status"], "unknown"
        )

    @patch(
        "robot_tasks.cameras.grasp_evidence.exclude_gripper",
        new=keep_fixture_surfaces,
    )
    def test_lost_tracking_blocks_air_release_but_allows_bounded_recovery(self):
        verified = self.guard.enrich(
            observation(rise=0.025, object_rise=0.025, width=0.035, time_s=1.0)
        )
        self.assertEqual(
            verified.feedback.metrics["grasp_check"]["left"]["status"], "verified"
        )
        lost = self.guard.enrich(observation(rise=0.15, width=0.035, time_s=2.0))
        self.assertEqual(
            lost.feedback.metrics["grasp_check"]["left"]["status"], "tracking_uncertain"
        )
        self.assertIsNone(self.guard.reject_reason(move(z=0.185), lost))
        opening = Action(
            (*move(z=0.185).commands, ArmCommand("left", "gripper", (0.07,))), 1 / 12
        )
        self.assertIsNotNone(self.guard.reject_reason(opening, lost))
        self.assertIsNone(self.guard.reject_reason(move(z=0.170), lost))
        self.assertIsNotNone(self.guard.reject_reason(move(z=0.14), lost))
        self.assertIsNotNone(self.guard.reject_reason(move(y=0.42, z=0.17), lost))
        self.assertIsNotNone(self.guard.reject_reason(move(z=0.20), lost))
        self.assertTrue(self.guard.attempts["left"].verified_once)
        # A later clear view restores holding; opening must still be deliberate.
        regained = self.guard.enrich(
            observation(rise=0.15, object_rise=0.15, width=0.035, time_s=3.0)
        )
        self.assertEqual(regained.feedback.metrics["grasp_check"]["left"]["status"], "verified")
        self.assertIsNone(self.guard.reject_reason(opening, regained))
        self.guard.accepted(opening, regained)
        self.assertGreater(len(self.guard.releases["left"][0].points), 0)
        opened = self.guard.enrich(observation(rise=0.15, width=0.07, time_s=4.0))
        self.assertFalse(self.guard.attempts)
        self.assertIsNone(self.guard.reject_reason(move(y=0.25), opened))

    def test_tracking_recovery_returns_to_rotated_high_pose_without_new_pickup_bounds(self):
        self.guard.enrich(observation(rise=0.15, object_rise=0.15, width=0.035, time_s=1.0))
        attempt = self.guard.attempts["left"]
        last_pose = attempt.last_verified_pose
        missing = observation(rise=0.15, width=0.035, time_s=2.0)
        rotated = replace(missing.robot["left"], tcp_pose_world_wxyz=(
            *last_pose[:3], *rotation_quaternion(np.array([
                [0.5, 0, 3**0.5/2], [0, 1, 0], [-3**0.5/2, 0, 0.5]
            ]))))
        missing = self.guard.enrich(replace(missing, robot={**missing.robot, "left": rotated}))
        self.assertEqual(attempt.status, "tracking_uncertain")
        self.assertIsNone(self.guard.reject_reason(move(z=last_pose[2]), missing))
        original = attempt.tracking_surfaces
        # Repeated missing captures do not replace known-good surfaces or anchor.
        self.guard.enrich(replace(missing, time_s=3.0))
        self.assertIs(attempt.tracking_surfaces, original)
        self.assertEqual(attempt.last_verified_pose, last_pose)

    def test_recovery_xy_bound_cannot_be_bypassed_by_repeated_small_steps(self):
        self.guard.enrich(observation(rise=0.15, object_rise=0.15, width=0.035, time_s=1.0))
        lost = self.guard.enrich(observation(rise=0.15, width=0.035, time_s=2.0))
        self.assertIsNone(self.guard.reject_reason(move(x=0.364, z=0.17), lost))
        advanced = replace(lost.robot["left"], tcp_pose_world_wxyz=(0.364, 0.4, 0.17, *DOWN))
        advanced = self.guard.enrich(replace(lost, time_s=3.0, robot={**lost.robot, "left": advanced}))
        self.assertIsNotNone(self.guard.reject_reason(move(x=0.368, z=0.16), advanced))

    def test_empty_close_after_tracking_loss_allows_open_without_false_verification(self):
        self.guard.enrich(observation(rise=0.15, object_rise=0.15, width=0.035, time_s=1.0))
        empty = self.guard.enrich(observation(rise=0.15, width=0.012, time_s=2.0))
        self.assertEqual(empty.feedback.metrics["grasp_check"]["left"]["status"], "empty_suspected")
        self.assertIsNone(self.guard.reject_reason(Action((ArmCommand("left", "gripper", (0.07,)),), 1/12), empty))

    def test_probe_bound_is_cumulative_and_repeated_close_cannot_reset_it(self):
        at_limit = self.guard.enrich(observation(rise=0.039, width=0.012, time_s=1.0))
        self.guard.accepted(close(), at_limit)
        self.assertIsNotNone(self.guard.reject_reason(move(z=0.09), at_limit))
        self.assertIsNone(self.guard.reject_reason(move(z=0.065), at_limit))
        at_limit = self.guard.enrich(observation(rise=0.040, width=0.012, time_s=2.0))
        self.assertIsNotNone(self.guard.reject_reason(move(z=0.076), at_limit))

    def test_observer_move_and_wait_are_not_blocked(self):
        after = self.guard.enrich(observation(rise=0.025, width=0.012, time_s=1.0))
        self.assertIsNone(
            self.guard.reject_reason(move(y=0.2, z=0.3, arm="right"), after)
        )
        self.assertIsNone(self.guard.reject_reason(Action((), 1 / 12), after))

    def test_combined_approach_close_rejected_before_baseline_but_opening_not_bypass(
        self,
    ):
        guard = GraspGuard(robot_config())
        action = Action((*move(z=0.15).commands, *close().commands), 1 / 12)
        self.assertIsNotNone(guard.reject_reason(action, self.before))
        after = self.guard.enrich(observation(width=0.012, time_s=1.0))
        action = Action(
            (*move(y=0.2).commands, ArmCommand("left", "gripper", (0.07,))), 1 / 12
        )
        self.assertIsNotNone(self.guard.reject_reason(action, after))

    def test_reset_clears_reference_and_paired_images_show_actual_before_after(self):
        after = self.guard.enrich(observation(rise=0.025, width=0.012, time_s=1.0))
        comparisons = list(comparison_images(after))
        self.assertEqual(len(comparisons), 2)
        self.assertEqual(comparisons[0][0]["reference_time_s"], 0.0)
        self.assertEqual(comparisons[0][0]["current_time_s"], 1.0)
        self.guard.reset()
        self.assertFalse(self.guard.enrich(observation()).grasp_references)


class GraspRunnerTests(unittest.TestCase):
    def test_paired_evidence_reaches_policy_beyond_rolling_image_history(self):
        from robot_tasks.policies.gpt import GPTPolicy
        from robot_tasks.policies.model_config import GPTSettings
        from robot_tasks.policies.responses import ModelReply

        config = robot_config()
        settings = GPTSettings(
            "https://example.invalid", "fixture", max_calls=8, history_steps=1
        )
        client = Mock()
        client.generate.return_value = ModelReply(
            json.dumps(decision_document()), "fixture", "offline", {}, 0.0
        )
        policy = GPTPolicy(config, settings, client=client)
        reference = GraspReference(
            "left", 0.0, tuple(rgbd(c) for c in config.cameras), {}
        )
        for step in range(1, 5):
            frames = tuple(
                rgbd(c, float(step), capture_id=step + 1) for c in config.cameras
            )
            obs = Observation(
                float(step), frames, {}, Feedback("completed"), (reference,)
            )
            policy.act(PolicyInput("offline", obs, step))
        content = client.generate.call_args.kwargs["messages"][-1]["content"]
        self.assertEqual(sum(x["type"] == "input_image" for x in content), 9)
        labels = [json.loads(x["text"]) for x in content if x["type"] == "input_text"]
        comparisons = [x for x in labels if x.get("modality") == "grasp_comparison"]
        self.assertEqual([c["camera"] for c in comparisons], list(config.cameras))
        self.assertTrue(
            all(
                c["reference_time_s"] == 0.0 and c["current_time_s"] == 4.0
                for c in comparisons
            )
        )
        self.assertEqual(len(policy.last_exchange["grasp_comparisons"]), 3)

    def test_100_calls_do_not_execute_repeated_unverified_transport_or_leak_score(self):
        config = robot_config()
        config = replace(
            config, max_steps=0, raw={**config.raw, "execution": {"max_decisions": 100}}
        )

        class Backend(MockBackend):
            def capture_rgb(self):
                return tuple(rgbd(name, self.t) for name in config.cameras)

        class Robot(MockRobot):
            def initialize(self, backend, config):
                super().initialize(backend, config)
                self.applied = []
                self.width = 0.07

            def state(self):
                return {
                    name: ArmState(
                        tcp_pose_world_wxyz=(0.36, 0.4, 0.1, *DOWN),
                        gripper_width_m=self.width,
                    )
                    for name in self.arms
                }

            def apply(self, action):
                self.applied.append(action)
                self.width = 0.012

            def feedback(self):
                return Feedback("completed")

        class Policy:
            def reset(self, seed):
                self.inputs = []

            def act(self, context):
                self.inputs.append(context)
                return close() if context.step == 1 else move(y=0.25)

        task = MockTask()
        task.evaluate = Mock(
            return_value=Assessment(
                "running", "PRIVATE_SCORE", {"object_pose": [1, 2, 3]}
            )
        )
        robot, policy = Robot(), Policy()
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(StringIO()):
            output, result = run_episode(
                config, Backend(), robot, task, policy, Path(folder)
            )
            events = [
                json.loads(l)
                for l in (output / "episode.jsonl").read_text().splitlines()
            ]
            references = [e for e in events if e["event"] == "grasp_reference"]
            self.assertEqual(len(references), 1)
            self.assertIn("_grasp_left", references[0]["observation"]["rgb"][0]["path"])
        self.assertEqual(result["decisions"], 100)
        self.assertEqual(result["grasp_check_rejections"], 99)
        self.assertEqual(len(robot.applied), 1)
        self.assertNotIn("PRIVATE_SCORE", str(policy.inputs))
        self.assertNotIn("object_pose", str(policy.inputs))


if __name__ == "__main__":
    unittest.main()
