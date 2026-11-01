"""Load a portable TOML episode configuration without simulator dependencies."""

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .robots.profile import RobotProfile, load_robot_profile, names


@dataclass(frozen=True)
class TaskConfig:
    source: Path
    raw: dict
    task_id: str
    instruction: str
    robot: RobotProfile
    arms: tuple[str, ...]
    cameras: tuple[str, ...]
    seed: int
    max_steps: int
    max_wall_time_s: float  # Zero disables the episode wall-time limit.
    control_dt_s: float
    physics_dt_s: float
    include_robot_state: bool
    include_feedback: bool
    save_rgb: bool
    include_depth: bool = False
    save_depth: bool = False
    observation_arms: tuple[str, ...] = ()
    workflow_source: Path | None = None
    workflow_settings: dict | None = None

    # Task capability, assigned by its adapter; not a user workflow switch.
    grasp_check_mode: str = "lift"
    initial_state_description: str = ""

    @property
    def experiment_id(self) -> str:
        """Public run ID; task_id stays the physical/scoring implementation ID."""
        return self.raw["task"].get("variant_id", self.task_id)

    @property
    def scene_id(self) -> str:
        """Versioned scene selection, independent of the scoring implementation."""
        return self.raw["task"].get("scene_variant", self.task_id)

    @property
    def command_arms(self) -> tuple[str, ...]:
        """Arms allowed to move; arms itself identifies task manipulation arms."""
        return self.arms + self.observation_arms

    @property
    def fixed_arms(self) -> tuple[str, ...]:
        return tuple(a for a in self.robot.available_arms if a not in self.command_arms)

    @property
    def arm_roles(self) -> dict[str, str]:
        return {a: "manipulation" if a in self.arms else "observation" if a in self.observation_arms else "fixed"
                for a in self.robot.available_arms}

    @property
    def robot_model(self) -> str:
        return self.robot.model

    @property
    def asset_path(self) -> Path:
        return self.robot.asset_path


@dataclass(frozen=True)
class ExecutionSettings:
    """Control ticks, not model calls or raw physics substeps."""

    max_action_steps: int = 1
    completion_samples: int = 1
    wait_steps: int = 1
    settle_steps: int = 0
    stop_steps: int = 0
    max_decisions: int | None = None  # None inherits the policy budget; 0 disables this cap.
    fixed_arm_tolerance_rad: float = 0.01
    model_budget_basis: str = "all_requests"
    retry_model_errors: bool = False

    @property
    def gripper_timing(self) -> str:
        return "after_motion"

    @classmethod
    def from_config(cls, config: TaskConfig) -> "ExecutionSettings":
        data = dict(config.raw.get("execution", {}))
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown execution settings: {sorted(unknown)}")
        result = cls(**data)
        if result.model_budget_basis not in {"all_requests", "completed_generations"}:
            raise ValueError("execution.model_budget_basis must be all_requests or completed_generations")
        if type(result.retry_model_errors) is not bool:
            raise ValueError("execution.retry_model_errors must be boolean")
        _positive(result.fixed_arm_tolerance_rad, "execution.fixed_arm_tolerance_rad")
        for name in ("max_action_steps", "completion_samples", "wait_steps", "settle_steps", "stop_steps"):
            value = getattr(result, name)
            minimum = 0 if name in {"settle_steps", "stop_steps"} else 1
            if type(value) is not int or value < minimum or (config.max_steps and value > config.max_steps):
                raise ValueError(f"execution.{name} must be an integer in [{minimum}, max_steps]")
        if result.completion_samples > result.max_action_steps:
            raise ValueError("completion_samples cannot exceed max_action_steps")
        if config.max_steps and result.settle_steps >= config.max_steps:
            raise ValueError("settle_steps must leave an action budget")
        if result.max_decisions is not None and (type(result.max_decisions) is not int
                                               or not 0 <= result.max_decisions <= 1000):
            raise ValueError("execution.max_decisions must be in [0, 1000]; zero disables the cap")
        return result


def _names(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(
        not isinstance(v, str) or not v.replace("_", "").isalnum() for v in value
    ):
        raise ValueError(f"{label} must be a nonempty list of simple names")
    if len(set(value)) != len(value):
        raise ValueError(f"{label} contains duplicate names")
    return tuple(value)


def _positive(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return float(value)


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be true or false")
    return value


def load_config(path: Path, *, depth: bool | None = None) -> TaskConfig:
    source = path.expanduser().resolve()
    with source.open("rb") as f:
        raw = tomllib.load(f)
    if raw.get("task", {}).get("simulation_enabled") is False:
        raise ValueError(f"{raw['task'].get('id', source.name)} is real-only and cannot run in simulation")
    if type(raw.get("schema_version")) is not int or raw["schema_version"] != 2:
        raise ValueError("Use schema_version = 2 with an R5 robot.profile and active_arms; legacy robot configs are not supported")
    from .workflow import load_workflow
    raw, workflow_source, workflow_settings = load_workflow(source, raw)
    if depth is not None:
        # Command-line overrides are recorded in the effective configuration.
        raw["observation"]["include_depth"] = _boolean(depth, "depth")
        raw.setdefault("recording", {})["save_depth"] = depth
    if "planner" in raw:
        raise ValueError("R5 uses the shared cuRobo planner; machine paths belong in runtime configuration")
    required = {"schema_version", "task", "robot", "runtime", "observation", "recording"}
    if not required.issubset(raw):
        raise ValueError(f"Missing config sections: {sorted(required - raw.keys())}")
    task, robot = raw["task"], raw["robot"]
    runtime, observation = raw["runtime"], raw["observation"]
    for label, value in (("task.id", task["id"]), ("task.instruction", task["instruction"])):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{label} must be a nonempty string")
    if "variant_id" in task:
        l1 = task["id"] in {"L1-01", "L1-02", "L1-03", "L1-04", "L1-05"}
        l2 = task["id"] in {"L2-01", "L2-02", "L2-03", "L2-04", "L2-05", "L2-06"}
        versioned_scene = task["id"] in {"L2-07", "L3-01", "L3-02", "L3-03", "L3-04"}
        if (task["variant_id"] != task["id"] + "_v3"
                or task.get("scene_delivery") != "video_20261015"
                or not ((l1 and task.get("initial_placement_profile") == "l1_xy_random_p03_20261018")
                        or (l2 and task.get("initial_layout") == "l2_p01")
                        or (versioned_scene and task.get("scene_variant") == task["variant_id"]))):
            raise ValueError("variant_id requires a matching initial layout or versioned scene")
    if "scene_variant" in task:
        from .video_delivery import scenes
        if (task["scene_variant"] != task.get("variant_id")
                or task["scene_variant"] not in scenes()):
            raise ValueError("scene_variant must select the registered variant scene")
    if (not isinstance(robot, dict) or not {"profile", "active_arms"}.issubset(robot)
            or set(robot) - {"profile", "active_arms", "observation_arms"}):
        raise ValueError("robot must contain only profile and active_arms, with optional observation_arms; this project uses R5 exclusively")
    if not isinstance(robot["profile"], str) or not robot["profile"].strip():
        raise ValueError("robot.profile must reference an R5 profile TOML")
    profile = load_robot_profile(source.parent / robot["profile"])
    if task.get("arm_selection", "fixed") != "fixed":
        if task.get("implementation") != "household":
            raise ValueError("Automatic arm selection is supported by household L1 tasks only")
        from .tasks.bench.arm_selection import resolve_bench_spec

        selected = resolve_bench_spec(task, profile)
        # Resolve before validating cameras and before constructing policies,
        # controllers or contact monitors. The effective config records the roles.
        robot["active_arms"] = list(selected.arms)
        robot["observation_arms"] = list(selected.observers)
    arms = names(robot["active_arms"], "robot.active_arms")
    if not set(arms).issubset(profile.available_arms):
        raise ValueError("robot.active_arms must be a nonempty subset of the R5 left/right arms")
    observers = robot.get("observation_arms", [])
    observation_arms = () if observers == [] else names(observers, "robot.observation_arms")
    if not set(observation_arms).issubset(profile.available_arms) or set(arms) & set(observation_arms):
        raise ValueError("observation_arms must be available arms distinct from manipulation active_arms")
    cameras = _names(observation["cameras"], "cameras")
    if observation_arms:
        from .cameras.profile import load_camera_rig
        camera_profile = raw.get("simulation", {}).get("camera_profile")
        if not camera_profile or not observation["include_robot_state"] or not observation["include_feedback"]:
            raise ValueError("Observation arms require a camera profile, robot state and execution feedback")
        rig = load_camera_rig(source.parent / camera_profile)
        parents = {camera.parent for camera in rig.select(cameras)}
        if not set(observation_arms).issubset(parents):
            raise ValueError("Each observation arm must carry a selected camera")
    max_steps, seed = runtime["max_steps"], runtime["seed"]
    if type(max_steps) is not int or max_steps < 0 or type(seed) is not int or seed < 0:
        raise ValueError("max_steps and seed must be nonnegative integers; max_steps=0 disables the total tick limit")
    control_dt = _positive(runtime["control_dt_s"], "control_dt_s")
    physics_dt = _positive(runtime["physics_dt_s"], "physics_dt_s")
    ratio = control_dt / physics_dt
    if not math.isclose(ratio, round(ratio), abs_tol=1e-6) or ratio < 1:
        raise ValueError("control_dt_s must be an integer multiple of physics_dt_s")
    if "save_video" in raw["recording"]:
        _boolean(raw["recording"]["save_video"], "save_video")
    include_depth = _boolean(observation.get("include_depth", False), "include_depth")
    save_depth = _boolean(raw["recording"].get("save_depth", include_depth), "save_depth")
    if save_depth and not include_depth:
        raise ValueError("recording.save_depth requires observation.include_depth")
    if include_depth and not raw.get("simulation", {}).get("camera_profile"):
        raise ValueError("Depth requires simulation.camera_profile for RGB alignment and calibration")
    wall_time = runtime["max_wall_time_s"]
    if (isinstance(wall_time, bool) or not isinstance(wall_time, (int, float))
            or not math.isfinite(wall_time) or wall_time < 0):
        raise ValueError("max_wall_time_s must be finite and nonnegative; zero disables the limit")
    return TaskConfig(
        source=source, raw=raw, task_id=task["id"], instruction=task["instruction"],
        robot=profile, arms=arms, cameras=cameras,
        seed=seed, max_steps=max_steps,
        max_wall_time_s=float(wall_time),
        control_dt_s=control_dt, physics_dt_s=physics_dt,
        include_robot_state=_boolean(observation["include_robot_state"], "include_robot_state"),
        include_feedback=_boolean(observation["include_feedback"], "include_feedback"),
        save_rgb=_boolean(raw["recording"]["save_rgb"], "save_rgb"),
        include_depth=include_depth, save_depth=save_depth,
        observation_arms=observation_arms, workflow_source=workflow_source, workflow_settings=workflow_settings,
    )
