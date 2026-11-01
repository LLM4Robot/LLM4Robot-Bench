"""Process-aware L2 scoring; force/strain limits are declared simulation proxies."""

import math

import numpy as np

from ...robots.transforms import pose_transform
from .evaluation import (
    held,
    quiet,
    relative,
    released,
    sustained,
    table_placed,
    upright,
)


def axial_quiet(row):
    """Supported round parts may spin about their own axis without relocating."""
    axis = pose_transform(row["pose"])[:3, 2]
    omega = np.asarray(row["angular_velocity"])
    return bool(
        np.linalg.norm(row["linear_velocity"]) < 0.025
        and np.linalg.norm(omega - (omega @ axis) * axis) < 0.4
    )


def marker(e, rows):
    p = e.spec.parameters
    body, cap = rows["body"], rows["cap"]
    b, c = pose_transform(body["pose"]), pose_transform(cap["pose"])
    tip = np.linalg.inv(c) @ b @ np.array([0.0, 0.0, p["body_tip_z"], 1.0])
    gap = float(p["cap_mouth_z"] - tip[2])
    assembly = np.linalg.inv(b) @ c
    # Native body/cap share a remote authoring origin. A small cap tilt can
    # separate those origins by >6 mm while the actual nib stays in the cavity.
    # Test the functional geometry, not agreement between USD root positions.
    tip_in_cap = bool(
        abs(assembly[2, 2]) > 0.95
        and np.linalg.norm(tip[:2]) < 0.0076
        and p["cap_mouth_z"] + 0.002 < tip[2] < p["cap_end_z"]
    )
    if (
        held(body, p["body_arm"])
        and tip_in_cap
        and min(np.min(rows[n]["corners"], axis=0)[2] for n in ("body", "cap"))
        > p["support_z"] + 0.008
    ):
        e.flags.add("assembly_lifted")
    separated = gap >= 0.002
    if separated and held(body, p["body_arm"]) and held(cap, p["cap_arm"]):
        e.flags.add("separated_while_held")
    placed = {
        name: bool(
            released(row)
            and axial_quiet(row)
            and (row["bench_force_n"] + row["fixture_force_n"]) > p[name + "_mass_kg"] * 9.81 * 0.3
        )
        for name, row in (("body", body), ("cap", cap))
    }
    # A later arbitrary rotation must not fake recapping or undo an observed
    # separation. Detect actual insertion of either barrel end into cap cavity.
    ends = np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, p["body_tip_z"], 1.0]])
    local = ends @ (np.linalg.inv(c) @ b).T
    recapped = bool(
        abs(assembly[2, 2]) > 0.95
        and any(
            np.linalg.norm(v[:2]) < 0.0076
            and p["cap_mouth_z"] + 0.002 < v[2] < p["cap_end_z"]
            for v in local
        )
    )
    complete = (
        separated
        and "separated_while_held" in e.flags
        and not recapped
    )
    return (
        complete,
        {
            "assembly_lifted": "assembly_lifted" in e.flags,
            "tip_in_cap_cavity": tip_in_cap,
            "separated_while_held": "separated_while_held" in e.flags,
            "current_axial_gap_m": gap,
            "placed": placed,
            "recapped": recapped,
            "release_required": False,
            "placement_required": False,
            "support_thresholds_n": {
                n: p[n + "_mass_kg"] * 9.81 * 0.3 for n in ("body", "cap")
            },
            "axial_spin_is_not_a_failure": True,
            "stages": {"capped_marker_lifted": "assembly_lifted" in e.flags,
                       "cap_removed_while_held": "separated_while_held" in e.flags,
                       "barrel_placed": "separated_while_held" in e.flags and placed["body"],
                       "cap_placed": "separated_while_held" in e.flags and placed["cap"]},
        },
        "",
    )


def books(e, rows):
    p = e.spec.parameters
    middle = rows["middle_book"]
    outward = np.asarray(p["outward_axis_world"])
    travel = float((np.asarray(middle["pose"][:3]) - e.initial["middle_book"][:3]) @ outward)
    if held(middle, p["manipulation_arm"]) and travel >= p["minimum_travel_m"] - 0.002:
        e.flags.add("middle_extracted_while_held")
    # Book-to-book contact alone is not support: the contact chain must reach
    # the fixed box. Dropping the stack onto the table cannot finish this task.
    supported = {n for n, row in rows.items() if row["fixture_force_n"] > 0.05}
    for _ in rows:
        supported.update(
            n for n, row in rows.items()
            if any(row["pair_forces"].get(other, 0.0) > 0.05 for other in tuple(supported))
        )
    on_box = {n for n in supported if min(c[2] for c in rows[n]["corners"])
              >= p["support_z"] - 0.01}
    adjacent_supported = all(n in on_box and upright(rows[n], 0.45)
                             for n in ("upper_book", "lower_book"))
    release_required = p.get("release_required", True)
    support_held = held(rows["upper_book"], p["support_arm"])
    if p.get("require_top_book_hold"):
        # A support hand may carry the top book's weight. Requiring a force
        # chain through the middle book would contradict that valid support.
        adjacent_supported = bool("lower_book" in on_box and upright(rows["lower_book"], 0.45)
                                  and support_held and quiet(rows["upper_book"]))
    complete = ("middle_extracted_while_held" in e.flags
                and travel >= p["minimum_travel_m"] - 0.002
                and "middle_book" in on_box and adjacent_supported
                and (not release_required or all(released(r) for r in rows.values()))
                and (not p.get("require_top_book_hold") or support_held))
    return complete, {"outward_travel_m": travel, "required_travel_m": p["minimum_travel_m"],
        "middle_extracted": "middle_extracted_while_held" in e.flags,
        "adjacent_books_supported": adjacent_supported,
        "release_required": release_required, "top_book_held": support_held,
        "complete_extraction_required": False,
        "stages": {"visible_extraction": "middle_extracted_while_held" in e.flags,
                   "released_supported": complete and all(released(r) for r in rows.values()),
                   "supported_pull_complete": complete}}, ""


def drawer(e, rows):
    row = rows["drawer"]
    p = e.spec.parameters
    # Rail limits and success use opening from fully CLOSED, not extra travel
    # from the pre-opened episode layout.
    displacement = np.asarray(row["pose"][:3]) - np.asarray(p["closed_pose"][:3])
    outward = np.asarray(p["outward_axis_world"])
    travel = float(displacement @ outward)
    guide_deviation = np.linalg.norm(
        displacement - travel * outward
    ) > 0.008 or not upright(row, 0.15)
    beyond_stop = travel < -0.006 or travel > p["max_travel_m"] + 0.006
    failure = ""
    if sustained(e, "drawer_guide_deviation", guide_deviation):
        failure = "Drawer guide constraint was persistently violated"
    if sustained(e, "drawer_beyond_stop", beyond_stop):
        failure = "Drawer persistently exceeded its mechanical stop"
    # Establish causal pull evidence during a continuous two-finger grasp,
    # independently of the final opening threshold. A last millimetre of passive
    # travel after release must not erase an otherwise measured grasped pull.
    grasped = held(row, p["manipulation_arm"]) and not guide_deviation and not beyond_stop
    pull = e.contact_motion.setdefault("drawer_pull", {"start": None, "net_m": 0.0})
    if grasped:
        if pull["start"] is None:
            pull["start"] = travel
        pull["net_m"] = max(pull["net_m"], travel - pull["start"])
        if pull["net_m"] >= p["minimum_grasped_pull_m"]:
            e.flags.add("pulled_while_held")
    else:
        pull["start"] = None
    if travel >= p["required_open_m"] - 0.002 and "pulled_while_held" in e.flags:
        e.flags.add("opening_reached")
    returned = abs(travel - p["initial_open_m"]) <= p["return_tolerance_m"]
    at_stop = travel < 0.006 or travel > p["max_travel_m"] - 0.006
    impact = at_stop and np.linalg.norm(row["linear_velocity"]) > p["max_speed_m_s"]
    if impact:
        e.flags.add("drawer_speed_reference_exceeded")
    complete = (
        "pulled_while_held" in e.flags
        and "opening_reached" in e.flags
        and returned
        and (released(row) or not p.get("release_required", True))
        and quiet(row)
        and not guide_deviation
        and not beyond_stop
    )
    return (
        complete,
        {
            "opening_m": travel,
            "initial_opening_m": p["initial_open_m"],
            "travel_since_start_m": float((np.asarray(row["pose"][:3])
                                            - e.initial["drawer"][:3]) @ outward),
            "pulled_while_held": "pulled_while_held" in e.flags,
            "maximum_grasped_pull_m": pull["net_m"],
            "minimum_grasped_pull_m": p["minimum_grasped_pull_m"],
            "opening_reached": "opening_reached" in e.flags,
            "returned_to_initial_opening": returned,
            "released": released(row),
            "quiet": bool(quiet(row)),
            "final_state": ("pulled 50 mm, returned to initial opening"
                            + (", released" if p.get("release_required", True) else "")),
            "stages": {"grasped": "drawer" in e.touched,
                       "opened": "opening_reached" in e.flags,
                       "returned_released": bool(complete and released(row)),
                       "returned": bool(complete)},
            "quality_affects_success": False,
            "impact_speed_reference_m_s": p["max_speed_m_s"],
            "peak_speed_m_s": e.peak_speeds.get("drawer", 0.0),
            "quality_warnings": (
                ["Drawer speed at stop exceeded engineering reference"]
                if "drawer_speed_reference_exceeded" in e.flags
                else []
            ),
        },
        failure,
    )


from .pearl_pouring import score as pearl_pouring
from .wiping import score as wiping

RULES = {"L2-01": marker, "L2-02": books, "L2-03": pearl_pouring,
         "L2-05": drawer, "L2-07": wiping}
