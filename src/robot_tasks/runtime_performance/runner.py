"""Instrument the unchanged runner, scoped to one episode per process.

No simulator/UI method moves off its owning thread. Only packed RGB video
processing is asynchronous. CLI parallelism uses independent processes.
"""

from contextlib import contextmanager, ExitStack
from dataclasses import replace
import json
import threading
import time

from . import describe, mode
from .cache import FrameEncodingCache, JointReadCache
from .timing import Timings
from .video import BackgroundVideo

_episode_lock = threading.Lock()


@contextmanager
def replace_attribute(target, name, value):
    """Restore the original descriptor/instance state, including on failure."""
    owned = name in vars(target)
    previous = vars(target).get(name)
    setattr(target, name, value)
    try:
        yield
    finally:
        if owned:
            setattr(target, name, previous)
        else:
            delattr(target, name)


def run_episode(config, backend, robot, task, policy, output_root, **kwargs):
    from .. import runner, video
    from ..policies import gpt
    from ..recording import EpisodeRecorder
    from ..robots.adapters import R5Adapter

    selected = mode()
    if selected == "off" or kwargs.get("dry_run"):
        return runner.run_episode(config, backend, robot, task, policy, output_root, **kwargs)
    if not _episode_lock.acquire(blocking=False):
        raise RuntimeError("Runtime profiling requires one episode per process; use separate CLI processes for parallel runs")
    try:
        timings = Timings()
        description = describe()
        config = replace(config, raw={**config.raw, "runtime_performance": description})
        native_video = video.MotionVideo

        class TimedRecorder(EpisodeRecorder):
            def __init__(self, *args, **kw):
                with timings.measure("recording.initialize"):
                    super().__init__(*args, **kw)
                self._last_timing_checkpoint = 0.0
                self.checkpoint()

            def checkpoint(self):
                value = {"profile": description, **timings.snapshot()}
                path = self.directory / "timing.json"
                temporary = path.with_suffix(".json.tmp")
                temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
                temporary.replace(path)
                self._last_timing_checkpoint = time.monotonic()
                return value

            def observation(self, *args, **kw):
                with timings.measure("recording.observation"):
                    return super().observation(*args, **kw)

            def event(self, value):
                with timings.measure("recording.event"):
                    super().event(value)
                if time.monotonic() - self._last_timing_checkpoint >= 10:
                    self.checkpoint()

            def finish(self, result):
                # Video drains before native finish(), so its failures and final
                # frame counts have already passed through the native error path.
                result["runtime_performance"] = self.checkpoint()
                super().finish(result)

        with ExitStack() as stack:
            def install(target, name, value):
                stack.enter_context(replace_attribute(target, name, value))

            def instrument(target, name, phase):
                function = getattr(target, name, None)
                if callable(function):
                    install(target, name, timings.wrap(phase, function))

            for name in ("open", "reset", "step", "capture_rgb", "save_scene", "pump_events"):
                instrument(backend, name, "backend." + name)
            for name in ("state", "feedback", "apply", "reset"):
                instrument(robot, name, "robot." + name)
            for name in ("build_scene", "reset", "evaluate"):
                instrument(task, name, "task." + name)
            instrument(policy, "act", "model.decision")
            instrument(policy, "observation_message", "model.observation_message")
            if getattr(policy, "client", None) is not None:
                instrument(policy.client, "generate", "model.transport")
                if getattr(policy.client, "host", None) is not None:
                    instrument(policy.client.host, "run", "model.cli_process")

            original_initialize = robot.initialize

            def initialize(*args, **kw):
                with timings.measure("robot.initialize"):
                    result = original_initialize(*args, **kw)
                planner = getattr(robot, "planner", None)
                if planner is not None:
                    instrument(planner, "plan", "planning.plan")
                callback = getattr(backend, "before_physics_step", None)
                if callable(callback):
                    cache = JointReadCache(timings)
                    # Scope only the known native controller, never arbitrary
                    # callers that might directly teleport/set joint positions.
                    use_cache = (selected == "optimized" and isinstance(robot, R5Adapter)
                                 and getattr(callback, "__func__", None) is R5Adapter.before_physics_step)
                    if use_cache:
                        for name, arm in robot.instances.items():
                            install(arm, "get_joint_positions", cache.reader(name, arm.get_joint_positions))

                    def before_step(dt_s):
                        with timings.measure("physics.controller"):
                            if use_cache:
                                with cache.scope():
                                    return callback(dt_s)
                            return callback(dt_s)
                    install(backend, "before_physics_step", before_step)
                return result

            install(robot, "initialize", initialize)
            install(runner, "EpisodeRecorder", TimedRecorder)
            if selected == "optimized":
                install(video, "MotionVideo", lambda output, settings: BackgroundVideo(
                    output, settings, factory=native_video, timings=timings))
                install(gpt, "prepare_frame", FrameEncodingCache(gpt.prepare_frame, timings))
            else:
                class TimedVideo(native_video):
                    def add(self, frame):
                        with timings.measure("video.encode"):
                            return super().add(frame)

                    def close(self):
                        with timings.measure("video.finalize"):
                            return super().close()
                install(video, "MotionVideo", TimedVideo)
                install(gpt, "prepare_frame", timings.wrap("model.image_prepare", gpt.prepare_frame))
            return runner.run_episode(config, backend, robot, task, policy, output_root, **kwargs)
    finally:
        _episode_lock.release()
