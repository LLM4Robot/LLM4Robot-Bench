"""Offline RGB-D contracts; synthetic sensor fixtures are never task evidence."""

from contextlib import redirect_stdout
from dataclasses import replace
from io import BytesIO, StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image

from robot_tasks.backends.isaacsim import IsaacSimBackend
from robot_tasks.backends.mock import MockBackend, MockRobot, MockTask
from robot_tasks.cameras.depth import depth_array, encode_depth_preview, resolve_depth_queries
from robot_tasks.cameras.geometry import project_world
from robot_tasks.cameras.isaac_rig import IsaacCameraRig
from robot_tasks.cameras.profile import load_camera_rig
from robot_tasks.config import load_config
from robot_tasks.policies.depth_query import split_depth_reply
from robot_tasks.policies.gpt import GPTPolicy
from robot_tasks.policies.model_config import GPTSettings
from robot_tasks.policies.responses import ModelReply
from robot_tasks.recording import EpisodeRecorder
from robot_tasks.runner import run_episode
from robot_tasks.types import Assessment, DepthFrame, DepthQuery, Feedback, Observation, PolicyInput, RGBFrame


from fixtures import rgbd, gripper_command, decision_document


ROOT = Path(__file__).resolve().parents[1]
SETTINGS = GPTSettings("https://example.invalid", "fixture", max_calls=3, image_max_edge=256)


def configured():
    base = load_config(ROOT / "configs/checks/gpt_single_arm_pick_place.toml")
    return replace(base, instruction="Offline RGB-D protocol fixture.", include_depth=True, save_depth=True,
                   raw={**base.raw,
                        "observation": {**base.raw["observation"], "include_depth": True},
                        "recording": {**base.raw["recording"], "save_depth": True},
                        "execution": {"max_action_steps": 1, "max_decisions": 3}})


def reply(document):
    return ModelReply(json.dumps(document), SETTINGS.model, "offline", {}, 0.)


def query_document():
    return decision_document(depth_queries=[{"camera": "overview", "u": 16, "v": 12}])


class DepthGeometryTests(unittest.TestCase):
    def test_preview_preserves_metric_depth_and_fixed_colours(self):
        frame = rgbd(width=5, height=1)
        values = np.array([[.01, .02, 2., 3., np.nan]], dtype="<f4")
        frame = replace(frame, depth=replace(frame.depth, data=values.tobytes()))
        png, _ = encode_depth_preview(frame, (5, 1))
        pixels = np.asarray(Image.open(BytesIO(png)))[0]
        np.testing.assert_array_equal(pixels, [[0, 80, 255], [0, 80, 255],
                                             [255, 40, 0], [255, 40, 0], [0, 0, 0]])
        np.testing.assert_equal(depth_array(frame), values)

    def test_optical_depth_and_usd_world_axes(self):
        frame = rgbd()
        query = DepthQuery("overview", 24, 6, 32, 24)
        result, = resolve_depth_queries((frame,), (query,))
        np.testing.assert_allclose(result["point_camera_opencv_m"], [.125, -.1, .5])
        np.testing.assert_allclose(result["point_world_m"], [1.125, 2.1, 2.5])
        self.assertEqual(result["depth_z_m"], .5)  # NOT length of the off-axis ray.

    def test_scaled_queries_use_same_source_pixel_as_preview(self):
        frame = rgbd(width=640, height=400)
        values = np.linspace(.1, 1.9, 640*400, dtype="<f4").reshape(400, 640)
        frame = replace(frame, depth=replace(frame.depth, data=values.tobytes()))
        query = DepthQuery("overview", 12, 9, 512, 320)
        result, = resolve_depth_queries((frame,), (query,))
        self.assertEqual(result["source_pixel_uv"], [15, 11])
        self.assertEqual(result["depth_z_m"], float(values[11, 15]))
        png, metadata = encode_depth_preview(frame, (512, 320))
        self.assertEqual(Image.open(BytesIO(png)).size, (512, 360))
        self.assertEqual(metadata["image_region_xywh"], [0, 0, 512, 320])
        self.assertEqual(metadata["display_range_m"], [.02, 2.])

    def test_backprojection_roundtrip_with_distortion_and_moving_camera(self):
        frame = rgbd()
        calibration = {**frame.calibration, "distortion": [.05, -.01, .001, -.002, .001],
                       "T_world_camera_pose_wxyz": [.1, -.3, 1., .7071067811865476, 0., .7071067811865476, 0.]}
        frame = replace(frame, calibration=calibration)
        result, = resolve_depth_queries((frame,), (DepthQuery("overview", 25, 5, 32, 24),))
        projected = project_world([result["point_world_m"]], calibration["T_world_camera_pose_wxyz"],
                                  calibration["intrinsics_fx_fy_cx_cy"], calibration["distortion"])
        np.testing.assert_allclose(projected[0], [25, 5], atol=1e-6)

    def test_invalid_depth_is_unknown_not_a_zero_or_neighbour_point(self):
        frame = rgbd()
        values = depth_array(frame).copy()
        values[12, 16] = np.nan
        frame = replace(frame, depth=replace(frame.depth, data=values.tobytes()))
        result, = resolve_depth_queries((frame,), (DepthQuery("overview", 16, 12, 32, 24),))
        self.assertEqual(result["status"], "no_depth")
        self.assertIsNone(result["point_world_m"])
        self.assertIsNone(result["depth_z_m"])
        png, _ = encode_depth_preview(frame, (32, 24))
        self.assertEqual(Image.open(BytesIO(png)).getpixel((16, 12)), (0, 0, 0))
        json.dumps(result, allow_nan=False)

    def test_pairing_rejects_stale_wrong_camera_dimensions_or_capture(self):
        frame = rgbd()
        for change in ({"timestamp_s": 1.}, {"camera": "left_wrist"}, {"width": 31},
                       {"capture_id": 2}, {"data": b"invalid"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                depth_array(replace(frame, depth=replace(frame.depth, **change)))

    def test_schema_rejects_invalid_queries_and_query_motion_combination(self):
        for patch in ({"u": -1}, {"u": 32}, {"v": 24}, {"u": True}, {"u": 1.5},
                      {"camera": "unknown"}, {"camera": []}, {"extra": 1}):
            document = query_document()
            document["depth_queries"][0].update(patch)
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                split_depth_reply(json.dumps(document), {"overview": (32, 24)})
        for update in ({"commands": [1]}, {"request_stop": True}, {"depth_queries": [query_document()["depth_queries"][0]] * 9}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                split_depth_reply(json.dumps({**query_document(), **update}), {"overview": (32, 24)})


class DepthCaptureTests(unittest.TestCase):
    def setUp(self):
        profile = load_camera_rig(ROOT / "configs/cameras/r5_orbbec_provisional.toml")
        self.rig = IsaacCameraRig(None, None, profile, ("left_wrist",), include_depth=True)
        self.spec = self.rig.specs[0]
        width, height = self.spec.resolution
        self.rig.annotators["left_wrist"] = Mock()
        self.rig.annotators["left_wrist"].get_data.return_value = np.full((height, width, 4), 128, np.uint8)
        self.rig.depth_annotators["left_wrist"] = Mock()
        self.rig.world_poses = Mock(return_value={"left_wrist": (0., 0., 1., 1., 0., 0., 0.)})

    def test_render_data_normalizes_invalids_and_pairs_capture_identity(self):
        width, height = self.spec.resolution
        values = np.full((height, width, 1), .4, np.float32)
        values[0, :5, 0] = [np.inf, 0., -1., 30., np.nan]
        self.rig.depth_annotators["left_wrist"].get_data.return_value = values
        frame, = self.rig.read(1.25, 3)
        array = depth_array(frame)
        self.assertTrue(np.isnan(array[0, :5]).all())
        self.assertAlmostEqual(float(array[1, 1]), .4)
        self.assertEqual(frame.calibration["capture_id"], frame.depth.capture_id)
        self.assertEqual(frame.timestamp_s, frame.depth.timestamp_s)

    def test_empty_depth_retries_both_modalities_without_physics(self):
        width, height = self.spec.resolution
        self.rig.depth_annotators["left_wrist"].get_data.side_effect = [np.array([]), np.ones((height, width), np.float32)]
        backend = IsaacSimBackend()
        backend.world, backend._rep = Mock(current_time=2.), Mock()
        backend.settings = Mock(capture_render_passes=1)
        backend.camera_rig = self.rig
        frame, = backend.capture_rgb()
        self.assertEqual(frame.depth.timestamp_s, 2.)
        self.assertEqual(backend.world.render.call_count, 2)
        backend.world.step.assert_not_called()
        self.assertEqual(self.rig.annotators["left_wrist"].get_data.call_count, 2)

    def test_wrong_depth_dimensions_fail_instead_of_reshaping(self):
        self.rig.depth_annotators["left_wrist"].get_data.return_value = np.ones((640, 400))
        with self.assertRaisesRegex(RuntimeError, "depth dimensions"):
            self.rig.read(0., 1)


class DepthPolicyAndRecordingTests(unittest.TestCase):
    def setUp(self):
        self.config = configured()
        self.frames = tuple(rgbd(name) for name in self.config.cameras)
        self.context = PolicyInput("test only", Observation(0., self.frames, {}, Feedback("completed")), 1)
        self.client = Mock()
        self.client.generate.return_value = reply(query_document())

    def test_all_three_rgb_depth_pairs_reach_model_and_query_decoder(self):
        policy = GPTPolicy(self.config, SETTINGS, client=self.client)
        action = policy.act(self.context)
        self.assertEqual(action.commands, ())
        self.assertEqual(action.depth_queries, (DepthQuery("overview", 16, 12, 32, 24),))
        request = self.client.generate.call_args.kwargs
        content = request["messages"][-1]["content"]
        self.assertEqual(sum(v["type"] == "input_image" for v in content), 6)
        labels = [json.loads(v["text"]) for v in content[1:] if v["type"] == "input_text"]
        self.assertEqual([v["modality"] for v in labels], ["rgb", "depth"] * 3)
        self.assertIn("depth_queries", request["schema"]["required"])
        self.assertEqual(len(policy.last_exchange["frames"]), 3)

    def test_missing_or_stale_depth_fails_before_paid_request(self):
        for frame in (replace(self.frames[0], depth=None),
                      replace(self.frames[0], depth=replace(self.frames[0].depth, capture_id=8))):
            policy = GPTPolicy(self.config, SETTINGS, client=self.client)
            context = replace(self.context, observation=replace(self.context.observation, rgb=(frame, *self.frames[1:])))
            with self.assertRaises(ValueError):
                policy.act(context)
        self.client.generate.assert_not_called()

    def test_missing_view_and_missing_calibration_disclosure_are_rejected(self):
        policy = GPTPolicy(self.config, SETTINGS, client=self.client)
        with self.assertRaisesRegex(ValueError, "every configured camera"):
            policy.act(replace(self.context, observation=replace(self.context.observation, rgb=self.frames[:2])))
        frames = (replace(self.frames[0], calibration=None), *self.frames[1:])
        with self.assertRaisesRegex(ValueError, "no camera metadata"):
            policy.act(replace(self.context, observation=replace(self.context.observation, rgb=frames)))
        self.client.generate.assert_not_called()

    def test_rgb_only_sends_three_images_and_has_no_depth_tool(self):
        config = load_config(ROOT / "configs/checks/gpt_single_arm_pick_place.toml")
        frames = tuple(replace(frame, depth=None) for frame in self.frames)
        context = replace(self.context, observation=replace(self.context.observation, rgb=frames))
        document = decision_document()
        del document["depth_queries"]
        self.client.generate.return_value = reply(document)
        policy = GPTPolicy(config, SETTINGS, client=self.client)
        action = policy.act(context)
        self.assertEqual(action.depth_queries, ())
        request = self.client.generate.call_args.kwargs
        self.assertNotIn("depth_queries", request["schema"]["properties"])
        content = request["messages"][-1]["content"]
        self.assertEqual(sum(v["type"] == "input_image" for v in content), 3)
        self.assertNotIn("depth", policy.run_metadata["observation_modalities"])
        self.assertTrue(all("depth" not in f for f in policy.last_exchange["frames"]))
        self.assertIn("no automatic grasp_check/release_check", policy.instructions)

    def test_raw_depth_is_lossless_and_reset_does_not_overwrite_failure(self):
        frame = self.frames[0]
        values = depth_array(frame).copy()
        values[0, 0] = np.nan
        frame = replace(frame, depth=replace(frame.depth, data=values.tobytes()))
        observation = replace(self.context.observation, rgb=(frame,))
        with tempfile.TemporaryDirectory() as temporary:
            recorder = EpisodeRecorder(Path(temporary), self.config, "simulation", 3)
            before = recorder.observation(observation, 1)
            after = recorder.observation(self.context.observation, 1, after_reset=True)
            path = recorder.directory / before["rgb"][0]["depth"]["path"]
            np.testing.assert_equal(np.load(path, allow_pickle=False), values)
            self.assertNotEqual(before["rgb"][0]["depth"]["path"], after["rgb"][0]["depth"]["path"])
            self.assertTrue((recorder.directory / before["rgb"][0]["depth"]["preview_path"]).is_file())
            json.dumps(before, allow_nan=False)

    def test_queries_share_call_budget_without_robot_execution_or_clock_advance(self):
        class Backend(MockBackend):
            def capture_rgb(self):
                return tuple(rgbd(name, self.t) for name in self.config.cameras)
        backend, robot = Backend(), MockRobot()
        robot.apply = Mock()
        task = MockTask()
        task.evaluate = Mock(return_value=Assessment("running", "Offline fixture"))
        policy = GPTPolicy(self.config, SETTINGS, client=self.client)
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(StringIO()):
            path, result = run_episode(self.config, backend, robot, task, policy, Path(temporary))
            self.assertEqual(result["status"], "finished")
            self.assertEqual(result["termination_reason"], "model_call_limit")
            self.assertEqual(result["model_calls"], 3)
            self.assertEqual(result["completed_steps"], 0)
            self.assertEqual(result["executed_actions"], 0)
            self.assertEqual(backend.time_s(), 0.)
            robot.apply.assert_not_called()
            messages = self.client.generate.call_args_list[1].kwargs["messages"]
            feedback = json.loads(messages[-1]["content"][0]["text"])["feedback"]
            self.assertEqual(feedback["metrics"]["depth_queries"][0]["point_world_m"], [1., 2., 2.5])
            self.assertIn("RGB＋深度", (path / "report.md").read_text(encoding="utf-8"))
            events = [json.loads(v) for v in (path / "episode.jsonl").read_text().splitlines()]
            self.assertEqual(sum(v["event"] == "depth_query" for v in events), 3)
            self.assertFalse(any(v["event"] == "physics" for v in events))

    def test_query_then_motion_then_stop_uses_one_existing_execution_loop(self):
        class Backend(MockBackend):
            def capture_rgb(self):
                return tuple(rgbd(name, self.t) for name in self.config.cameras)
        backend, robot, task = Backend(), MockRobot(), MockTask()
        robot.apply = Mock()
        task.evaluate = Mock(return_value=Assessment("running", "Offline fixture"))
        self.client.generate.side_effect = [reply(query_document()), reply(decision_document(commands=[gripper_command()])),
            reply(decision_document(request_stop=True))]
        policy = GPTPolicy(self.config, SETTINGS, client=self.client)
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(StringIO()):
            _, result = run_episode(self.config, backend, robot, task, policy, Path(temporary))
        self.assertEqual(result["status"], "finished")
        self.assertEqual(result["termination_reason"], "model_call_limit")
        self.assertEqual(result["rejected_stop_requests"], 1)
        self.assertEqual(result["model_calls"], 3)
        self.assertEqual(result["completed_steps"], 1)
        robot.apply.assert_called_once()
        self.assertEqual(robot.apply.call_args.args[0].commands[0].values, (.06,))
        final_input = self.client.generate.call_args_list[-1].kwargs["messages"][-1]["content"][0]["text"]
        self.assertNotIn("depth_queries", json.loads(final_input)["feedback"]["metrics"])


class DepthConfigurationTests(unittest.TestCase):
    def test_depth_flags_validate_without_isaac_or_model(self):
        source = ROOT / "configs/checks/gpt_single_arm_pick_place.toml"
        text = source.read_text(encoding="utf-8")
        text += "\n[observation]\ninclude_depth = true\n[recording]\nsave_depth = true\n"
        text = text.replace('"../robots/r5_sim.toml"', json.dumps((ROOT / "configs/robots/r5_sim.toml").as_posix()))
        text = text.replace('"../workflows/gpt_default.toml"', json.dumps((ROOT / "configs/workflows/gpt_default.toml").as_posix()))
        text = text.replace('"../cameras/r5_orbbec_provisional.toml"', json.dumps((ROOT / "configs/cameras/r5_orbbec_provisional.toml").as_posix()))
        cases = (("include_depth = true", "include_depth = 1"),
                 ("save_depth = true", 'save_depth = "yes"'),
                 ("include_depth = true", "include_depth = false"),
                 ('camera_profile = ' + json.dumps((ROOT / "configs/cameras/r5_orbbec_provisional.toml").as_posix()), ""))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.toml"
            for before, after in cases:
                path.write_text(text.replace(before, after), encoding="utf-8")
                with self.subTest(after=after), self.assertRaises(ValueError):
                    load_config(path)
            path.write_text(text, encoding="utf-8")
            config = load_config(path)
        self.assertTrue(config.include_depth)
        self.assertTrue(config.save_depth)


if __name__ == "__main__":
    unittest.main()
