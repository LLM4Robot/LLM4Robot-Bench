"""Optional measured return-home gate; never sends robot commands."""

from dataclasses import replace
from typing import Mapping

import numpy as np

from ..config import TaskConfig
from ..types import ArmState, Assessment


def completion_instruction(config: TaskConfig) -> str:
    if not config.raw.get("completion", {}).get("return_home", False):
        return ""
    homes = "; ".join(
        f"{m.name.upper()} joint1..joint6={list(m.home_positions_rad)} radians"
        for m in config.robot.mounts
    )
    return (
        "\nFinal completion protocol: after accomplishing the physical goal, put down "
        "all held task objects on their requested supports and RELEASE them. Withdraw "
        "clear of props, then command BOTH arms back to their initial joint positions: "
        + homes + ". Send joint_positions_rad targets with all TCP fields null when clear; an observation arm must "
        "also return. Verify measured joint state, not just issued commands. Only "
        "then request_stop. Gripper opening at home has no prescribed value. "
        "The program does not automatically release objects or return either arm.\n"
    )


def require_home(
    assessment: Assessment, robot_state: Mapping[str, ArmState], config: TaskConfig
) -> Assessment:
    settings = config.raw.get("completion", {})
    if not settings.get("return_home", False) or assessment.status in {"failed", "invalid"}:
        return assessment
    tolerance = float(settings.get("home_tolerance_rad", 0.05))
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("completion.home_tolerance_rad must be finite and positive")
    errors = {}
    for mount in config.robot.mounts:
        state = robot_state.get(mount.name)
        q = np.asarray(state.joint_positions_rad if state else (), dtype=float)
        home = np.asarray(mount.home_positions_rad)
        errors[mount.name] = (
            float(np.max(np.abs(q - home)))
            if q.shape == home.shape and np.isfinite(q).all() else None
        )
    at_home = bool(errors) and all(v is not None and v <= tolerance for v in errors.values())
    physical_complete = assessment.status == "succeeded"
    metrics = {**assessment.metrics, "completion": {
        "task_and_release_complete": physical_complete,
        "both_arms_home": at_home, "joint_max_errors_rad": errors,
        "home_tolerance_rad": tolerance,
    }}
    if physical_complete and not at_home:
        metrics["goal_complete"] = False
        metrics["stage"] = "return_home"
        return replace(assessment, status="running",
                       reason="Task and release complete; both arms must return home", metrics=metrics)
    return replace(assessment, metrics=metrics)
