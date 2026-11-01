"""Simulator-independent R5 action validation and gripper conversion."""

import math

from ..types import Action
from .profile import RobotProfile


# Use the same measured-error allowance as joint tracking completion. A settled
# loaded joint was accepted by the controller but rejected as the next planning
# start (L2-02: 2.503939 rad; L2-07: 2.506142 rad at a 2.5-rad software limit).
# Only the planning/timeout reference is bounded. Raw observations, requested
# goals, trajectory limits and collision checks remain unchanged.
ARM_JOINT_TRACKING_TOLERANCE_RAD = 0.015
MEASURED_JOINT_LIMIT_TOLERANCE_RAD = ARM_JOINT_TRACKING_TOLERANCE_RAD


def empty_close_tolerance(minimum_width):
    # A 1.5-mm allowance swallowed legitimate 2-mm wall pinches after enabling
    # the URDF's zero-gap range. Keep the old allowance for legacy wide stops.
    return 0.0002 if minimum_width < 0.001 else 0.0015


def bounded_measured_joints(positions, limits) -> tuple[float, ...]:
    """Bound a measured reference within the small solver-settling allowance.

    Raw observations remain untouched. This is never used to repair requested
    model joint goals or planned waypoints, which retain strict limit checks.
    """
    positions, limits = tuple(positions), tuple(limits)
    if len(positions) != len(limits) or any(
        isinstance(q, bool) or not math.isfinite(q) for q in positions
    ):
        raise ValueError("Measured arm joints require finite positions matching the limits")
    result = []
    for q, bound in zip(positions, limits):
        if (q < bound.lower - MEASURED_JOINT_LIMIT_TOLERANCE_RAD
                or q > bound.upper + MEASURED_JOINT_LIMIT_TOLERANCE_RAD):
            raise ValueError(
                f"Measured {bound.joint}: {q:.9g} outside configured limits "
                f"[{bound.lower}, {bound.upper}] beyond solver-settling tolerance"
            )
        result.append(float(min(bound.upper, max(bound.lower, q))))
    return tuple(result)


def normalized_to_width(profile: RobotProfile, value: float) -> float:
    """Model protocol: 0 closes to the configured minimum; 1 opens fully."""
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("gripper must be a finite number in [0, 1]")
    lower, upper = profile.gripper.width_limits_m
    return lower + value * (upper - lower)


def width_to_normalized(profile: RobotProfile, width: float) -> float:
    lower, upper = profile.gripper.width_limits_m
    return (width - lower) / (upper - lower)


def width_to_joints(profile: RobotProfile, width: float) -> tuple[float, float]:
    gripper = profile.gripper
    if (gripper.width_limits_m is None or gripper.joint_scale is None
            or gripper.joint_offset_m is None):
        raise ValueError("Gripper mapping has not been configured")
    if isinstance(width, bool) or not math.isfinite(width) or not gripper.width_limits_m[0] <= width <= gripper.width_limits_m[1]:
        raise ValueError("Gripper width is outside configured limits")
    return tuple(scale * width + offset for scale, offset in zip(gripper.joint_scale, gripper.joint_offset_m))


def joints_to_width(profile: RobotProfile, positions: tuple[float, ...]) -> float:
    gripper = profile.gripper
    if len(positions) != 2 or gripper.joint_scale is None or gripper.joint_offset_m is None:
        raise ValueError("Two finger positions and a calibrated mapping are required")
    # Average the two independent estimates; retain raw finger positions in state.
    return sum((q - offset) / scale for q, offset, scale in zip(
        positions, gripper.joint_offset_m, gripper.joint_scale)) / 2


def plan_targets(profile: RobotProfile, active_arms: tuple[str, ...], action: Action,
                 previous: dict[str, tuple[float, ...]]) -> dict[str, tuple[float, ...]]:
    """Validate the entire bimanual action before changing either arm's targets."""
    if (not isinstance(action, Action) or isinstance(action.duration_s, bool)
            or not math.isfinite(action.duration_s) or action.duration_s <= 0):
        raise ValueError("Expected an Action with positive finite duration")
    planned = dict(previous)
    channels = set()
    limits = {limit.joint: limit for limit in profile.operating_limits}
    joint_names = profile.joint_names + profile.gripper.joint_names
    for command in action.commands:
        if command.arm not in active_arms:
            raise ValueError(f"Arm is not active for this task: {command.arm}")
        if command.kind not in {"joint_position", "cartesian_pose", "gripper"}:
            raise ValueError(f"Unsupported R5 action kind: {command.kind}")
        channel = (command.arm, "gripper" if command.kind == "gripper" else "arm_motion")
        if channel in channels:
            raise ValueError(f"Duplicate action channel: {channel}")
        channels.add(channel)
        values = command.values
        count = {"joint_position": 6, "cartesian_pose": 7, "gripper": 1}[command.kind]
        if len(values) != count or any(isinstance(v, bool) or not isinstance(v, (int, float))
                                      or not math.isfinite(v) for v in values):
            raise ValueError(f"{command.kind} requires {count} finite numeric values")
        current = planned[command.arm]
        if command.kind == "cartesian_pose":
            raise ValueError("Cartesian targets must use the shared trajectory planner")
        elif command.kind == "joint_position":
            target = tuple(values) + current[6:]
        else:
            target = current[:6] + width_to_joints(profile, values[0])
        for name, value in zip(joint_names, target):
            bound = limits[name]
            if not bound.lower <= value <= bound.upper:
                raise ValueError(f"{command.arm}.{name}: target outside configured limits")
        planned[command.arm] = target
    return planned
