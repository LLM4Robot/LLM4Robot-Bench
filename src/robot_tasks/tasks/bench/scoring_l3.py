"""Relative rotation and tool/contact witnesses, independent of commands or notes."""

import math

import numpy as np

from ...robots.transforms import pose_transform
from .evaluation import (
    held,
    released,
    sustained,
    upright,
)


def _contact_moved_object(e, key, row, active):
    """Distinguish a brief touch from sustained manipulation of a forbidden prop."""
    if not active:
        e.contact_motion.pop(key, None)
        return False
    pose = np.array(row["pose"])
    start_time, start_pose = e.contact_motion.setdefault(
        key, (e.last_time, pose.copy())
    )
    distance = np.linalg.norm(pose[:3] - start_pose[:3])
    angle = 2 * math.acos(float(np.clip(abs(pose[3:] @ start_pose[3:]), 0, 1)))
    return e.last_time - start_time >= 0.15 - 1e-9 and (
        distance > 0.003 or angle > 0.05
    )


def bottle(e, rows):
    body, cap = rows["bottle"], rows["cap"]
    p = e.spec.parameters
    bt = pose_transform(body["pose"])
    local = (np.asarray(cap["corners"]) - bt[:3, 3]) @ bt[:3, :3]
    center = local.mean(axis=0)
    radius = np.linalg.norm(local[:, :2] - center[:2], axis=1).max()
    above = local[:, 2].min() > p["neck_top_z"] + 0.0005
    beside = np.linalg.norm(center[:2]) - radius > p["neck_radius_m"] + 0.0005
    no_neck_contact = cap["pair_forces"].get("bottle", 0) < 0.005
    detached = bool((above or beside) and no_neck_contact)
    # Removal ends this task. A held, geometrically clear cap is sufficient;
    # no extra lift height, placement, release or return-home stage is required.
    cap_held = held(cap, p["cap_arm"])
    gap = float(local[:, 2].min() - p["neck_top_z"])
    if detached and cap_held:
        e.flags.add("cap_detached")
    on_table = cap["bench_force_n"] > 0.005 and released(cap)
    required_lift = p.get("cap_lift_min_m", 0.0)
    body_held = held(body, p["body_arm"])
    lift_ok = (required_lift <= gap <= p["cap_lift_max_m"]
               and cap_held and (p.get("body_fixed", True) or body_held)) if required_lift else True
    complete = detached and "cap_detached" in e.flags and lift_ok
    return complete, {"bottle_fixed": p.get("body_fixed", True), "cap_detached_now": detached,
        "bottle_held": body_held, "cap_held": cap_held,
        "cap_removal_observed": "cap_detached" in e.flags,
        "cap_released_on_table": on_table, "neck_clearance_m": gap,
        "extra_lift_required_m": required_lift,
        "release_required": False, "placement_required": False}, ""


def pushing(e, rows):
    from .push_triangle import containment_margin

    if e.spec.parameters.get("stacked_book_names"):
        from .stacked_books import score
        return score(e, rows)
    block = rows["eraser"]
    article = e.spec.design.articles[0]
    p = e.spec.parameters
    t = pose_transform(block["pose"])
    yaw = math.atan2(t[1, 0], t[0, 0])
    margin = containment_margin(
        block["pose"], article.goal, p["block_vertices"], p["target_vertices"]
    )
    contact = max(block["fingers"][e.spec.arms[0]]) > 0.005
    supported = block["bench_force_n"] > 0.005
    if contact and supported:
        if e.push_contact_pose is None:
            e.push_contact_pose = (t[0, 3], t[1, 3], yaw)
        start = e.push_contact_pose
        translation = np.linalg.norm(t[:2, 3] - start[:2])
        rotation = abs((yaw - start[2] + math.pi) % (2 * math.pi) - math.pi)
        if translation >= p["push_translation_m"] or rotation >= p["push_rotation_rad"]:
            e.flags.add("gripper_pushed_block")
    else:
        e.push_contact_pose = None
    failure = ""
    observer_contact = any(
        max(block["fingers"][arm]) > 0.005 for arm in e.spec.observers
    )
    lifted = min(c[2] for c in block["corners"]) > p["table_z"] + p["max_lift_m"]
    if _contact_moved_object(
        e, "observer_push", block, observer_contact and not contact
    ):
        failure = "The observation arm manipulated the eraser"
    if sustained(
        e, "held_block_lift", lifted and not supported and held(block, e.spec.arms[0])
    ):
        failure = "Eraser was grasped and carried; use planar pushing on the table"
    complete = (
        margin >= -1e-9
        and supported
        and not lifted
        and "gripper_pushed_block" in e.flags
        and released(block)
    )
    return (
        complete,
        {
            "inside_target_outline": margin >= -1e-9,
            "footprint_boundary_margin_m": margin,
            "gripper_push_witness": "gripper_pushed_block" in e.flags,
            "table_supported": supported,
            "released": released(block),
            "target_width_m": float(np.ptp(np.asarray(p["target_vertices"])[:,0])),
            "target_depth_m": float(np.ptp(np.asarray(p["target_vertices"])[:,1])),
            "orientation_requirement": "none",
            "timed_hold_s": 0,
        },
        failure,
    )


def sweeping(e, rows):
    pan, broom = rows["dustpan"], rows["broom"]
    count = e.spec.parameters["count"]
    holds = held(broom, e.spec.arms[0])
    inside, swept = [], []
    failure = ""
    pan_frame = pose_transform(pan["pose"])
    for i in range(count):
        name = f"block_{i}"
        block = rows[name]
        corners = (np.asarray(block["corners"]) - pan_frame[:3, 3]) @ pan_frame[:3, :3]
        # Native pan basket has a round back (70 mm radius) and a straight
        # front lip at +40 mm. The handle at negative Y is NOT a goal region.
        x, y = corners[:, 0], corners[:, 1]
        footprint = (
            (np.abs(x) <= 0.069)
            & (y <= 0.044)
            & ((y >= 0) | (x * x + y * y <= 0.069**2))
        )
        height_min,height_max=-.001,.009
        if 'pan_floor_vertices' in e.spec.parameters:
            polygon=np.asarray(e.spec.parameters['pan_floor_vertices'])
            edges=np.roll(polygon,-1,axis=0)-polygon
            offset=corners[:,None,:2]-polygon[None,:,:]
            side=edges[None,:,0]*offset[:,:,1]-edges[None,:,1]*offset[:,:,0]
            footprint=np.all(side>=-1e-9,axis=1)
            low,high=e.spec.parameters['pan_floor_height_range']
            height_min,height_max=low-.001,high+.005
        contained = bool(
            np.all(footprint)
            and corners[:, 2].min() >= height_min
            and corners[:, 2].min() <= height_max
        )
        support = block["pair_forces"].get("dustpan", 0) > 0.005
        inside.append(contained and support)
        if broom["pair_forces"].get(name, 0) > 0.005 and holds:
            e.flags.add("swept_" + name)
        swept.append("swept_" + name in e.flags)
        direct_contact = any(max(fs) > 0.05 for fs in block["fingers"].values())
        if direct_contact:
            e.flags.add("direct_block_contact")
        if _contact_moved_object(e, "direct_" + name, block, direct_contact):
            failure = "A block was manipulated directly by a gripper"
    # No final grasp, quiet velocity, upright angle or extra dwell requirement.
    # Individual contacts may be transmitted through another swept block.
    witness = any(swept)
    brush_placed = released(broom) and (
        broom["bench_force_n"] + broom["fixture_force_n"] > 0.02)
    complete = sum(inside) >= e.spec.parameters["required"] and witness and (brush_placed or not e.spec.parameters.get("require_brush_release", True))
    return (
        complete,
        {
            "blocks_in_pan": sum(inside),
            "required": e.spec.parameters["required"],
            "tool_use_observed": witness,
            "blocks_with_brush_contact": sum(swept),
            "brush_held_now": holds,
            "brush_released_supported": brush_placed,
            "dustpan_fixed": True,
            "quality_warnings": (
                ["Direct gripper/block contact observed"]
                if "direct_block_contact" in e.flags
                else []
            ),
        },
        failure,
    )


RULES = {
    "L3-01": bottle,
    "L3-03": pushing,
    "L3-04": sweeping,
}
