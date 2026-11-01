"""Resolve single-arm L1 roles from the declared initial layout, before startup.

This is an episode setup rule, not an IK or reachability certificate. Initial
object geometry is used only for role assignment/audit, never as a policy sensor.
Roles stay fixed for the whole episode, including transport and release.
"""

from dataclasses import replace

import numpy as np

from ...robots.kinematics import R5Kinematics
from ...robots.transforms import pose_transform
from ..l1.specs import local_corners
from .spec import build_spec


def resolve_bench_spec(task, robot):
    spec = build_spec(task["id"])
    from .english_reference import apply_reference, uses_reference
    from ...initial_layout import apply_spec as apply_initial_layout
    if uses_reference(task):
        return apply_initial_layout(apply_reference(spec, task), task)
    mode = task.get("arm_selection", "fixed")
    if mode == "fixed":
        return apply_initial_layout(spec, task)
    if mode != "nearest_initial_object":
        raise ValueError(f"Unknown task.arm_selection: {mode}")
    if not task["id"].startswith("L1-") or len(spec.arms) != 1:
        raise ValueError("Nearest-object arm selection requires a single-arm L1 task")
    target_name = task.get("manipulation_object")
    target = next((a for a in spec.design.articles if a.name == target_name), None)
    if target is None:
        raise ValueError("task.manipulation_object must name a movable task article")
    if set(robot.available_arms) != {"left", "right"}:
        raise ValueError("Nearest-object arm selection requires both R5 arms")

    transform = pose_transform(target.initial)
    centre = transform[:3, :3] @ local_corners(target).mean(axis=0) + transform[:3, 3]
    kinematics = R5Kinematics(robot)
    distances = {}
    for mount in robot.mounts:
        tcp = kinematics.forward(mount.home_positions_rad, mount.name)
        distances[mount.name] = float(np.linalg.norm(np.asarray(tcp[:3]) - centre))
    if not np.isfinite(list(distances.values())).all():
        raise ValueError("Nonfinite initial arm-selection distance")
    # Deterministic tie breaking is independent of mount/config ordering.
    selected = min(distances, key=lambda arm: (distances[arm], arm))
    observer = next(arm for arm in sorted(distances) if arm != selected)
    return replace(spec, arms=(selected,), parameters={
        **spec.parameters,
        "arm_selection": {
            "mode": mode,
            "target": target_name,
            "metric": "3D distance from initial TCP to initial article bounds centre",
            "distances_m": distances,
            "manipulation_arm": selected,
            "observation_arm": observer,
            "tie_break": "arm name in alphabetical order",
            "roles_fixed_for_episode": True,
            "reachability_verified": False,
            "initial_geometry_is_policy_input": False,
        },
    })
