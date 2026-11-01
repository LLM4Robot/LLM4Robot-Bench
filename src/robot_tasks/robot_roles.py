"""Shared role permissions and fixed-arm checks, independent of task scoring."""

from dataclasses import replace

from .config import TaskConfig
from .types import Assessment, ArmState


ARM_IDENTITY_PROTOCOL = "tuojing_base_ids_v3"
REAL_SKILL_REVISION = "1765230ca26ec951f6837ef8759d146f579fec04"


def arm_identities(config: TaskConfig) -> dict[str, dict[str, str]]:
    """Fixed base identities, not a transform of live TCPs or task instructions.

    The real skill names arms from the robot's perspective; this lab's USD and
    existing commands use the fixed overview's left/right. Convert source task
    roles once in control permissions. Current task prose keeps robot-perspective
    names and the system notes explain their mapping to command IDs.
    Other robot layouts do not inherit this lab-specific correspondence.
    """
    if config.raw.get("simulation", {}).get("environment") != "tuojing_lab_v4":
        return {}
    real_arm = {"left": "right", "right": "left"}
    real_camera = {"left": "gemini_right", "right": "gemini"}
    from .video_delivery import enabled
    robot_prompt = enabled(config)
    return {
        arm: {"command_arm": arm, "overview_base_side": arm,
              "real_arm": real_arm[arm], "prompt_arm": real_arm[arm] if robot_prompt else arm,
              "prompt_frame": "robot" if robot_prompt else "command", "wrist_camera": arm + "_wrist",
              "real_wrist_camera_key": real_camera[arm], "role": role}
        for arm, role in config.arm_roles.items()
    }


def validate_role_command(config: TaskConfig, arm: str, kind: str) -> None:
    if arm not in config.command_arms:
        raise ValueError(f"Unknown arm: {arm}; no motion permission")
    if arm in config.observation_arms and kind == "gripper":
        raise ValueError(f"Observation arm {arm} cannot receive gripper commands; use it only to move its camera")


class FixedArmMonitor:
    def __init__(self, config: TaskConfig, tolerance_rad: float):
        self.arms, self.tolerance = config.fixed_arms, tolerance_rad
        self.reset()

    def reset(self) -> None:
        self.reference: dict[str, tuple[float, ...]] | None = None
        self.max_drift = 0.0

    def update(self, state: dict[str, ArmState], assessment: Assessment) -> Assessment:
        if self.reference is None:
            self.reference = {a: state[a].joint_positions_rad for a in self.arms if a in state}
        for arm, original in self.reference.items():
            if arm not in state or len(state[arm].joint_positions_rad) != len(original):
                raise ValueError(f"Missing or invalid fixed arm state: {arm}")
            self.max_drift = max(self.max_drift, max(
                (abs(q - q0) for q, q0 in zip(state[arm].joint_positions_rad, original)), default=0.0))
        metrics = {**assessment.metrics, "fixed_arm_max_drift_rad": self.max_drift}
        if self.max_drift > self.tolerance and assessment.status != "invalid":
            return Assessment("failed", "Fixed arm moved beyond the configured tolerance", metrics)
        return replace(assessment, metrics=metrics)
