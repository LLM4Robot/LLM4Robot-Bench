"""GPT single-arm pick/place; simulator truth never enters the policy input."""

from collections import deque
from dataclasses import replace
import math

from ..config import TaskConfig
from ..types import Assessment, EvaluationState, Feedback
from .pick_place_metrics import lifted, placed
from .pick_place_scene import InteractionScene, InteractionSettings


class PickPlaceEvaluator:
    """Streaming, phase-independent scoring. No scripted action stage is assumed."""

    def __init__(self, settings: InteractionSettings, source, target):
        self.settings, self.source, self.target = settings, tuple(source), tuple(target)
        self.samples = deque()
        self.lift_confirmed = False
        self.drop_events = 0
        self.was_dropping = False
        self.last_time = None

    def update(self, sample: dict) -> Assessment:
        now = sample["time_s"]
        if self.last_time is not None and now <= self.last_time:
            raise ValueError("Evaluator samples must have increasing simulation timestamps")
        self.last_time = now
        self.samples.append(sample)
        while len(self.samples) > 1 and self.samples[1]["time_s"] <= now - self.settings.stability_s + 1e-9:
            self.samples.popleft()
        rows = list(self.samples)
        lift = lifted(rows, self.source[2], self.settings)
        self.lift_confirmed = self.lift_confirmed or lift["passed"]
        placement = placed(rows, self.target, self.settings)
        obj = sample["object"]
        dropping = (self.lift_confirmed and max(obj["finger_normal_force_n"]) < self.settings.contact_threshold_n
                    and obj["linear_velocity_m_s"][2] < -0.05
                    and math.dist(obj["position_m"][:2], self.target[:2]) > self.settings.placement_tolerance_m)
        self.drop_events += int(dropping and not self.was_dropping)
        self.was_dropping = dropping
        metrics = {"lift_confirmed": self.lift_confirmed, "lift": lift, "placement": placement,
                   "drop_events": self.drop_events, "object": obj,
                   "target_m": self.target}
        if obj["position_m"][2] < -self.settings.cube_size_m:
            return Assessment("failed", "Object fell below the table", metrics)
        if self.lift_confirmed and placement["passed"]:
            return Assessment("succeeded", "Measured physical lift, release and stable placement", metrics)
        return Assessment("running", "Waiting for sustained lift and released stable placement", metrics)


class SingleArmPickPlace:
    @staticmethod
    def prepare_config(config: TaskConfig) -> TaskConfig:
        if len(config.arms) != 1:
            raise ValueError("Single-arm pick/place requires exactly one active R5 arm")
        settings = InteractionSettings.from_config(config)
        # Only public task requirements are substituted. No source/target object coordinates.
        fields = {"arm": config.arms[0], "lift_threshold_m": settings.lift_threshold_m,
                  "stability_s": settings.stability_s, "open_width_m": settings.open_width_m,
                  "cube_size_m": settings.cube_size_m,
                  "placement_tolerance_m": settings.placement_tolerance_m}
        return replace(config, instruction=config.instruction.format_map(fields))

    def build_scene(self, backend, config: TaskConfig) -> None:
        self.arm = config.arms[0]
        self.settings = InteractionSettings.from_config(config)
        self.scene = InteractionScene(backend, config, self.settings)

    def reset(self, backend, seed: int) -> None:
        self.scene.reset(self.arm)
        self.evaluator = PickPlaceEvaluator(self.settings, self.scene.source(self.arm), self.scene.destination(self.arm))

    def evaluate(self, state: EvaluationState, feedback: Feedback) -> Assessment:
        robot = state.values["robot"]
        active = robot[self.arm]
        sample = {"time_s": state.values["simulation_time_s"], "object": self.scene.measure(self.arm),
                  "tcp_pose_world_wxyz": active.tcp_pose_world_wxyz, "gripper_width_m": active.gripper_width_m}
        return self.evaluator.update(sample)
