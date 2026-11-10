"""Offline evidence/cleanup regressions for the single-run performance profile."""

from contextlib import nullcontext, redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from fixtures import RGBDBackend, rgbd, decision_document
from robot_tasks.backends.mock import MockRobot, MockTask
from robot_tasks.backends.settings import SimulationSettings
from robot_tasks.policies.gpt import GPTPolicy, prepare_frame
from robot_tasks.policies.model_config import GPTSettings
from robot_tasks.policies.responses import ModelReply
from robot_tasks.registry import resolve_task
from robot_tasks.runtime_performance import mode
from robot_tasks.runtime_performance.cache import FrameEncodingCache, JointReadCache
from robot_tasks.runtime_performance.runner import run_episode
from robot_tasks.runtime_performance.timing import Timings
from robot_tasks.runtime_performance.video import BackgroundVideo, VideoEncodingError
from robot_tasks.types import Action, ArmState, Assessment, Feedback, Observation, PolicyInput, RGBFrame
from robot_tasks.video import MotionVideo


class TimingTests(unittest.TestCase):
    def test_nested_failure_timing_does_not_double_count(self):
        clock = iter([0., 1., 2., 5., 6., 7.])
        timers = Timings(lambda: next(clock))
        with self.assertRaises(ValueError):
            with timers.measure("outer"):
                with timers.measure("inner"):
                    raise ValueError("fixture")
        data = timers.snapshot()
        self.assertEqual(data["phases"]["outer"]["inclusive_s"], 5.)
        self.assertEqual(data["phases"]["outer"]["exclusive_s"], 2.)
        self.assertEqual(data["phases"]["inner"]["exclusive_s"], 3.)
        self.assertEqual(data["phases"]["outer"]["failures"], 1)

    def test_invalid_mode_fails_before_work(self):
        with patch.dict(os.environ, {"GPT_BENCHMARK_RUNTIME_MODE": "typo"}), self.assertRaises(ValueError):
            mode()


class CacheTests(unittest.TestCase):
    def test_native_controller_targets_and_settling_are_identical_with_cache(self):
        from fixtures import robot_config
        from robot_tasks.robots.adapters import R5Adapter
        from robot_tasks.robots.gripper_drive import SymmetricGripDrive
        from robot_tasks.runtime_performance.runner import replace_attribute
        traces, reads = [], []
        for cached in (False, True):
            robot = R5Adapter()
            robot.config = robot_config()
            measured = np.array([0.] * 6 + [.0175, .0175])
            reader = Mock(side_effect=lambda **kw: measured.copy())
            actions = []
            arm = SimpleNamespace(get_joint_positions=reader,
                                  apply_action=lambda action: actions.append(action.joint_positions.tolist()))
            robot.instances = {"left": arm}
            robot.indices = {"left": np.arange(8)}
            robot.targets = {"left": (0.,) * 6 + (.03, .03)}
            robot.drive_targets = {"left": tuple(measured)}
            robot.stable_grips = {"left": SymmetricGripDrive(.0175, .03, .3, .001, .006, .035)}
            cache = JointReadCache(Timings())
            native_types = SimpleNamespace(ArticulationAction=lambda **kw: SimpleNamespace(**kw))
            hook = replace_attribute(arm, "get_joint_positions", cache.reader("left", reader)) if cached else nullcontext()
            with patch.dict("sys.modules", {"isaacsim.core.utils.types": native_types}), hook:
                for index in range(40):
                    measured[6:] = .0175 + index * .00001
                    with cache.scope() if cached else nullcontext():
                        robot.before_physics_step(1 / 1920)
            traces.append((actions, list(robot.settling.history["left"]), robot.drive_targets))
            reads.append(reader.call_count)
        self.assertEqual(traces[0], traces[1])
        self.assertEqual(reads, [80, 40])

    def test_joint_reads_refresh_each_step_and_return_unaliased_arrays(self):
        timers = Timings()
        cache = JointReadCache(timers)
        actual = np.arange(8.)
        original = Mock(side_effect=lambda **kw: actual[np.asarray(kw["joint_indices"])])
        read = cache.reader("left", original)
        indices = np.arange(8)
        with cache.scope():
            first = read(joint_indices=indices)
            first[:] = -100
            np.testing.assert_equal(read(joint_indices=indices), actual)
            np.testing.assert_equal(read(joint_indices=np.arange(6)), actual[:6])
        self.assertEqual(original.call_count, 2)
        actual[:] = 42
        with cache.scope():
            np.testing.assert_equal(read(joint_indices=indices), actual)
        self.assertEqual(original.call_count, 3)
        # Outside the physics callback even identical requests must read Isaac.
        read(joint_indices=indices)
        read(joint_indices=indices)
        self.assertEqual(original.call_count, 5)
        self.assertEqual(cache.values, {})

    def test_joint_scope_exception_cannot_leak_stale_state(self):
        cache = JointReadCache(Timings())
        reader = cache.reader("left", lambda **kw: np.arange(2.))
        with self.assertRaises(RuntimeError):
            with cache.scope():
                reader(joint_indices=(0, 1))
                raise RuntimeError("controller error")
        self.assertFalse(cache.active)
        self.assertFalse(cache.values)

    def test_png_bytes_and_metadata_match_with_all_annotation_inputs(self):
        original = Mock(wraps=prepare_frame)
        cache = FrameEncodingCache(original, Timings())
        frame = rgbd()
        for size in (32, 16):
            expected = prepare_frame(frame, size, robot={})
            self.assertEqual(cache(frame, size, robot={}), expected)
            cached = cache(frame, size, robot={})
            self.assertEqual(cached, expected)
            cached[2]["resolution"][0] = 999  # cannot poison the cache
            self.assertEqual(cache(frame, size, robot={}), expected)
        self.assertEqual(original.call_count, 2)
        changed = replace(frame, data=bytes([200, 10, 20]) * frame.width * frame.height)
        self.assertEqual(cache(changed, 32), prepare_frame(changed, 32))
        calibrated = replace(frame, calibration={**frame.calibration, "capture_id": 999})
        self.assertEqual(cache(calibrated, 32), prepare_frame(calibrated, 32))
        robot = {"left": ArmState(gripper_width_m=.03)}
        self.assertEqual(cache(frame, 32, robot=robot), prepare_frame(frame, 32, robot=robot))
        self.assertEqual(original.call_count, 5)

    def test_image_cache_is_bounded_and_invalid_input_still_rejected(self):
        cache = FrameEncodingCache(prepare_frame, Timings(), max_bytes=2000, max_entries=2)
        for index in range(20):
            frame = replace(rgbd(), data=bytes([index, 10, 20]) * 32 * 24)
            cache(frame, 16)
            self.assertLessEqual(cache.bytes, 2000)
            self.assertLessEqual(len(cache.entries), 2)
        with self.assertRaises(ValueError):
            cache(replace(rgbd(), data=b"bad"), 16)


class BackgroundVideoTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("cv2"), "OpenCV required for real encoding fixture")
    def test_three_camera_mp4s_decode_identically_with_real_encoder(self):
        import cv2
        outputs = []
        with tempfile.TemporaryDirectory() as tmp:
            for background in (False, True):
                videos = []
                for camera in range(3):
                    path = Path(tmp) / f"{background}-{camera}"
                    path.mkdir()
                    settings = SimulationSettings(path, width=160, height=120, fps=12)
                    videos.append(BackgroundVideo(path, settings, factory=MotionVideo, timings=Timings())
                                  if background else MotionVideo(path, settings))
                try:
                    for index in range(36):
                        for camera, video in enumerate(videos):
                            pixels = np.zeros((120, 160, 3), dtype=np.uint8)
                            pixels[:, :, camera] = 30 + index * 5
                            pixels[20:50, index:index+30, :] = (120, 240, 50)
                            video.add(RGBFrame(str(camera), index/12, 160, 120, pixels.tobytes()))
                finally:
                    for video in videos:
                        video.close()
                decoded = []
                for video in videos:
                    capture = cv2.VideoCapture(str(video.output / "motion.mp4"))
                    self.assertTrue(capture.isOpened())
                    self.assertEqual(capture.get(cv2.CAP_PROP_FPS), 12)
                    frames = []
                    try:
                        while True:
                            ok, pixels = capture.read()
                            if not ok:
                                break
                            frames.append(pixels.tobytes())
                    finally:
                        capture.release()
                    self.assertEqual(len(frames), 36)
                    decoded.append((frames, (video.output / "motion.gif").read_bytes()))
                outputs.append(decoded)
        self.assertEqual(outputs[0], outputs[1])

    def fake_cv(self, writes):
        def writer(*args):
            item = Mock()
            item.isOpened.return_value = True
            item.write.side_effect = lambda pixels: writes.append((threading.get_ident(), pixels.copy()))
            return item
        return SimpleNamespace(VideoWriter=writer, VideoWriter_fourcc=lambda *a: 0,
                               cvtColor=lambda pixels, code: pixels, COLOR_RGB2BGR=1)

    def test_all_frames_and_gif_match_synchronous_recording(self):
        main = threading.get_ident()
        outputs = []
        with tempfile.TemporaryDirectory() as tmp:
            for background in (False, True):
                path = Path(tmp) / str(background)
                path.mkdir()
                settings = SimulationSettings(path, width=8, height=6, fps=10)
                writes = []
                with patch.dict("sys.modules", {"cv2": self.fake_cv(writes)}):
                    video = (BackgroundVideo(path, settings, factory=MotionVideo, timings=Timings(), queue_frames=2)
                             if background else MotionVideo(path, settings))
                    for index in range(103):
                        frame = RGBFrame("overview", index / 10, 8, 6,
                                         bytes([index, 255-index, index // 2]) * 48)
                        video.add(frame)
                    video.close()
                    if background:
                        video.close()  # idempotent; worker is fully joined
                        self.assertFalse(video.worker.is_alive())
                    self.assertEqual(video.count, 103)
                    self.assertEqual(len(writes), 103)
                    self.assertTrue(all((thread != main) == background for thread, _ in writes))
                    outputs.append(([pixels.tobytes() for _, pixels in writes],
                                    (path / "motion.gif").read_bytes(),
                                    (path / "first_frame.png").read_bytes(),
                                    (path / "latest_frame.png").read_bytes(),
                                    video.maximum_pixel_change))
        self.assertEqual(outputs[0], outputs[1])

    def test_backpressure_keeps_frames_and_snapshots_mutable_bytes(self):
        blocked, release = threading.Event(), threading.Event()
        frames = []
        class Sink:
            count = 0
            def __init__(self, *args):
                pass
            def add(self, frame):
                blocked.set()
                release.wait(2)
                frames.append(frame.data)
                self.count += 1
            def close(self):
                pass
        with tempfile.TemporaryDirectory() as tmp:
            settings = SimulationSettings(Path(tmp), width=2, height=2)
            video = BackgroundVideo(Path(tmp), settings, factory=Sink, timings=Timings(), queue_frames=1)
            pixels = bytearray([1] * 12)
            try:
                video.add(RGBFrame("x", 0, 2, 2, pixels))
                self.assertTrue(blocked.wait(1))
                pixels[:] = bytes([2] * 12)
                video.add(RGBFrame("x", 1, 2, 2, pixels))
                self.assertEqual(video.queue.qsize(), 1)
            finally:
                release.set()
                video.close()
            self.assertEqual(frames, [bytes([1] * 12), bytes([2] * 12)])

    def test_worker_and_startup_errors_are_not_swallowed(self):
        class Broken:
            def __init__(self, *args):
                self.writer = Mock()
            def add(self, frame):
                raise OSError("disk full")
        with tempfile.TemporaryDirectory() as tmp:
            settings = SimulationSettings(Path(tmp), width=2, height=2)
            video = BackgroundVideo(Path(tmp), settings, factory=Broken, timings=Timings())
            try:
                video.add(RGBFrame("x", 0, 2, 2, bytes(12)))
            except VideoEncodingError:
                pass  # the worker may fail before add returns
            with self.assertRaisesRegex(VideoEncodingError, "disk full"):
                video.close()
            self.assertFalse(video.worker.is_alive())
            video.sink.writer.release.assert_called_once()
            with self.assertRaisesRegex(VideoEncodingError, "startup failed"):
                BackgroundVideo(Path(tmp), settings, factory=Mock(side_effect=OSError("startup failed")), timings=Timings())


class EpisodePerformanceTests(unittest.TestCase):
    def config(self):
        _, config = resolve_task("GPT-PICK-PLACE")
        return replace(config, max_steps=5, include_depth=False, save_depth=False,
            raw={**config.raw, "execution": {"max_action_steps": 1, "wait_steps": 1,
                                           "settle_steps": 0, "max_decisions": 3}})

    def test_model_inputs_events_and_task_results_unchanged_in_all_modes(self):
        from robot_tasks import runner, video
        from robot_tasks.policies import gpt
        from robot_tasks.recording import EpisodeRecorder
        original_prepare = gpt.prepare_frame
        results, event_lists, requests = [], [], []
        with tempfile.TemporaryDirectory() as tmp:
            for selected in ("off", "timing", "optimized"):
                config = self.config()
                sent = []
                def generate(**payload):
                    sent.append(deepcopy({k: v for k, v in payload.items() if k != "on_reasoning"}))
                    decision = decision_document()
                    decision.pop("depth_queries")
                    return ModelReply(json.dumps(decision), "fixture", "offline", {}, 0.)
                policy = GPTPolicy(config, GPTSettings("https://example.invalid", "fixture", max_calls=3),
                                   client=SimpleNamespace(generate=generate))
                backend, robot, task = RGBDBackend(), MockRobot(), MockTask()
                with patch.dict(os.environ, {"GPT_BENCHMARK_RUNTIME_MODE": selected}), redirect_stdout(io.StringIO()):
                    path, result = run_episode(config, backend, robot, task, policy, Path(tmp))
                self.assertEqual(result["status"], "finished", result["error"])
                self.assertIs(gpt.prepare_frame, original_prepare)
                self.assertIs(runner.EpisodeRecorder, EpisodeRecorder)
                self.assertIs(video.MotionVideo, MotionVideo)
                self.assertNotIn("state", vars(robot))
                self.assertNotIn("capture_rgb", vars(backend))
                if selected != "off":
                    data = json.loads((path / "timing.json").read_text())
                    self.assertEqual(data["phases"]["model.transport"]["calls"], 3)
                    self.assertIn("backend.step", data["phases"])
                for key in ("wall_time_s", "runtime_performance"):
                    result.pop(key, None)
                results.append(result)
                events = [json.loads(row) for row in (path / "episode.jsonl").read_text().splitlines()]
                for event in events:
                    event.pop("policy_time_s", None)
                event_lists.append(events)
                requests.append(sent)
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[0], results[2])
        self.assertEqual(event_lists[0], event_lists[1])
        self.assertEqual(event_lists[0], event_lists[2])
        self.assertEqual(requests[0], requests[1])
        self.assertEqual(requests[0], requests[2])

    def test_recording_failure_invalidates_result_and_restores_runtime(self):
        class BrokenVideo:
            def __init__(self, *args, **kwargs):
                self.count = 0
            def add(self, frame):
                self.count += 1
            def close(self):
                raise OSError("disk full on close")
        class Task(MockTask):
            def evaluate(self, state, feedback):
                return Assessment("succeeded", "Synthetic fixture only")
        config = self.config()
        policy = SimpleNamespace(reset=lambda seed: None, act=lambda _: Action((), config.control_dt_s))
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), \
                patch.dict(os.environ, {"GPT_BENCHMARK_RUNTIME_MODE": "optimized"}), \
                patch("robot_tasks.video.MotionVideo", BrokenVideo):
            path, result = run_episode(config, RGBDBackend(), MockRobot(), Task(), policy, Path(tmp), record_video=True)
        self.assertEqual(result["termination_reason"], "recording_error")
        self.assertIsNone(result["success"])
        self.assertFalse(result["task_evaluated"])


if __name__ == "__main__":
    unittest.main()
