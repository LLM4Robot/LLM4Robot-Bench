"""Shared contracts. SI units; quaternion order is always (w, x, y, z)."""

from dataclasses import dataclass, field
from typing import Literal


class InvalidPolicyAction(ValueError):
    """A model returned an invalid executable action, not a service failure."""


class RetryablePolicyError(RuntimeError):
    """No action was produced; the runner may retry the unchanged observation."""


class InvalidSimulationState(RuntimeError):
    """A numerical/readback failure invalidates measurement, not model ability."""


@dataclass(frozen=True)
class DepthFrame:
    """RGB-aligned optical-axis depth, metres, packed little-endian float32.

    NaN denotes an invalid/no-return pixel. This is ideal rendered depth, not
    an emulation of a particular hardware stereo sensor.
    """

    camera: str
    timestamp_s: float
    width: int
    height: int
    data: bytes
    capture_id: int


@dataclass(frozen=True)
class RGBFrame:
    camera: str
    timestamp_s: float
    width: int
    height: int
    data: bytes  # Packed RGB8; renderer-specific tensors stay in the backend.
    calibration: dict | None = None  # Actual simulated camera; disclosure to policy is opt-in.
    depth: DepthFrame | None = None  # Same render product and capture as RGB.


@dataclass(frozen=True)
class ArmState:
    joint_positions_rad: tuple[float, ...] = ()
    gripper_width_m: float | None = None
    joint_velocities_rad_s: tuple[float, ...] = ()
    finger_positions_m: tuple[float, ...] = ()
    finger_velocities_m_s: tuple[float, ...] = ()
    tcp_pose_world_wxyz: tuple[float, ...] | None = None  # xyz + wxyz, measured link pose + configured TCP


@dataclass(frozen=True)
class Feedback:
    status: str  # completed / executing / not_initialized / rejected / error / mock_only
    message: str = ""
    metrics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class GraspReference:
    arm: str
    time_s: float
    rgb: tuple[RGBFrame, ...]
    robot: dict[str, ArmState]


@dataclass(frozen=True)
class Observation:
    time_s: float
    rgb: tuple[RGBFrame, ...]
    robot: dict[str, ArmState]
    feedback: Feedback
    grasp_references: tuple[GraspReference, ...] = ()


@dataclass(frozen=True)
class ArmCommand:
    arm: str  # e.g. arm; or left and right for bimanual tasks
    kind: Literal["joint_position", "cartesian_pose", "gripper"]
    values: tuple[float, ...]
    frame: str = "world"  # Relevant to cartesian_pose only.
    # joint_position: joint angles (rad)
    # cartesian_pose: x,y,z (m), qw,qx,qy,qz
    # gripper: one total opening width (m), NOT per-finger displacement


@dataclass(frozen=True)
class DepthQuery:
    camera: str
    u: int  # Pixel in the RGB image actually sent to the model.
    v: int
    image_width: int
    image_height: int


@dataclass(frozen=True)
class Action:
    commands: tuple[ArmCommand, ...]
    duration_s: float
    request_stop: bool = False  # Policy stopping is NOT an evaluator success.
    depth_queries: tuple[DepthQuery, ...] = ()  # Read-only; cannot accompany motion or stop.
    note: str = ""  # Brief model observation/action rationale; never executable or a success label.


@dataclass(frozen=True)
class PolicyInput:
    instruction: str
    observation: Observation
    step: int
    # Deliberately no hidden object poses, evaluator metrics, or success label.
    # Stateful policies may retain previous inputs/actions within an episode.


@dataclass(frozen=True)
class EvaluationState:
    """Simulator truth; accessible to the task evaluator, never to the policy."""

    values: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Assessment:
    status: Literal["not_evaluated", "running", "succeeded", "failed", "invalid"]
    reason: str
    metrics: dict = field(default_factory=dict)
    retryable: bool = False  # Legacy metadata; the runner never retries terminal task failures.
