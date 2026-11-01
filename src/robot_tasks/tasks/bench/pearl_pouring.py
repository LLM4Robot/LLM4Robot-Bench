"""Cup-to-cup pouring with locked native hollow cups and free rigid pearls."""

import math
from dataclasses import replace

import numpy as np

from ..l1.specs import Article, pose
from .geometry import design
from .native_geometry import cup_cavity
from .native_l2_specs import TABLE_Z
from .spec import BenchSpec

PEARL_RADIUS = 0.004
PEARL_MASS = 0.00030895229571687653
# Native cavity + 8 mm sphere packing: approximately half the usable height.
COUNT = 200
PROMPT_RECEIVED_TARGET = math.ceil(COUNT * 0.8)


def cup_contains(point, radius=PEARL_RADIUS, tolerance=0.0003, receptacle="ripple_paper_cup"):
    """Whole sphere within the delivered cup's 48 inner planes and floor/rim."""
    planes, floor, rim = cup_cavity(receptacle)
    point = np.asarray(point)
    return bool(
        floor - tolerance <= point[2] - radius
        and point[2] + radius <= rim + tolerance
        and np.all(point @ planes[:, :3].T + radius <= planes[:, 3] + tolerance)
    )


def packing(count=COUNT):
    """Bottom-up, centre-out hexagonal layers, with initial contact clearance."""
    if not isinstance(count, int) or count <= 0:
        raise ValueError("Pearl count must be a positive integer")
    _, floor, rim = cup_cavity()
    spacing = 2 * PEARL_RADIUS + 0.0005
    points = []
    for layer in range(int((rim - floor) / spacing)):
        z = floor + PEARL_RADIUS + 0.0004 + layer * spacing * math.sqrt(2 / 3)
        shift = (spacing / 2, spacing * math.sqrt(3) / 6) if layer % 2 else (0, 0)
        candidates = [
            (spacing * (i + j / 2) + shift[0],
             spacing * math.sqrt(3) / 2 * j + shift[1], z)
            for i in range(-6, 7) for j in range(-6, 7)
        ]
        for point in sorted(candidates, key=lambda v: v[0] ** 2 + v[1] ** 2):
            if cup_contains(point, tolerance=-0.0003):
                points.append(point)
                if len(points) == count:
                    return tuple(points)
    raise ValueError("Requested pearls exceed the clearance-qualified cup packing")


def pouring():
    # +X is image-right in the shared overview. Right arm pours toward image-left.
    source, receiver = pose(0.30, 1.18, TABLE_Z + 0.001), pose(0.10, 1.18, TABLE_Z)
    bounds = ((-0.0426, -0.0426, 0.0), (0.0426, 0.0426, 0.107594535))
    articles, instances = [], []
    for name, initial in (("source_cup", source), ("receiver_cup", receiver)):
        articles.append(Article(name, (), initial, initial, 0.014,
                                asset="tuojing:ripple_paper_cup", asset_bounds=bounds))
        instances.append({"name": name, "asset": "ripple_paper_cup", "pose": initial,
                          "tracked": {name: ""}})
        if name == "receiver_cup":
            instances[-1]["static"] = [""]
    points = packing()
    for index, point in enumerate(points):
        name = f"load_{index}"
        initial = pose(*(np.asarray(source[:3]) + point))
        goal = pose(*(np.asarray(receiver[:3]) + point))
        articles.append(Article(name, (), initial, goal, PEARL_MASS,
                                asset="tuojing:pearl_8mm",
                                asset_bounds=((-PEARL_RADIUS,) * 3, (PEARL_RADIUS,) * 3)))
        instances.append({"name": name, "asset": "pearl_8mm", "pose": initial,
                          "tracked": {name: ""}})
    return BenchSpec(
        replace(design("L2-03", "纸杯间倒珍珠", articles, []),
                version="l2-04-overview-right-to-left-v8",
                provenance="Existing L20 ripple paper cups and 8 mm pearls; real-task reference: pearl-pouring.md"),
        arms=("right",),
        mechanism="native_cup_pearl_pouring",
        limitations="Rigid native paper cups: no crushing. Receiver is kinematically fixed to the table; source cup remains dynamic. Native hollow colliders, dimensions and masses retained. Pearl bore replaced by outer sphere; friction 0.03. This is a new simulation protocol, not a reproduction of the real trial's uncounted transfer result.",
        facts=(f"Two identical upright paper cups stand apart on the table: one contains {COUNT} free 8 mm pearls and the other is empty. "
               "The source is initially filled to approximately half the usable cup height; the loaded cup weighs about 76 g. "
               "Each cup is about 108 mm high, 56 mm across at its base and 85 mm across at its mouth. "
               "The R5 maximum opening is 70 mm: approach the lower cup body from the SIDE and close opposing finger pads across its diameter. "
               "Around 40 mm above its base the body is about 67 mm across. Use a slightly downward side approach and check finger/table clearance; do not attempt to surround the wide rim from above or pinch the 3 mm wall. "
               "The loaded source is on the RIGHT of the shared overview image and the empty receiver is on its LEFT. Use the RIGHT robot arm to pour and place the source cup; the LEFT arm may move only to observe. "
               "The receiving cup is fixed upright to the table: do not grasp, reposition or stabilize it. The loaded source cup remains movable. "
               "The source is in the right arm's forward workspace, with 200 mm centre spacing toward image-left. Keep the wrist and both full fingers clear of the receiver during tilting, not just the source lip. "
               "Before pouring, raise the source mouth roughly 150 mm above the receiving rim. As tilt increases, translate the wrist to maintain clearance and aim the stream into the opening; do not rotate about a fixed low TCP. "
               "Pearls move forward after leaving the lip, so the pouring mouth may need to stay back toward the source side rather than directly over the receiver centre. Use the observed landing point to adjust. "
               f"Success requires at least {PROMPT_RECEIVED_TARGET} of the {COUNT} pearls settled inside the fixed receiver "
               "and the source cup released upright, independently resting on the table. Residue is allowed; spills are counted and reported, not required to be zero."),
        parameters={
            "table_z": TABLE_Z, "hold_s": 0.5, "scoring_protocol": "cup_received_count_fixed_receiver_v2",
            "receiver_fixed_to_table": True, "manipulation_arm": "right",
            "native_instances": instances, "load_container_names": ("source_cup", "receiver_cup"),
            "native_cup_names": ("source_cup", "receiver_cup"),
            # Acceptance is deliberately looser than the unchanged model prompt.
            "count": COUNT, "required_received": 10,
            "minimum_loaded_lift_count": math.ceil(COUNT * 0.8),
            "ball_radius_m": PEARL_RADIUS, "ball_mass_kg": PEARL_MASS,
            "ball_friction": 0.03, "pearl_collision": "outer_sphere",
            "initial_packing_top_m": max(v[2] for v in points) + PEARL_RADIUS,
            "loaded_source_mass_kg": 0.014 + COUNT * PEARL_MASS,
            "lift_m": 0.02,
        },
    )


def score(e, rows):
    from ...robots.transforms import pose_transform
    from .evaluation import held, released, relative, sustained, upright

    p = e.spec.parameters
    source, receiver = rows["source_cup"], rows["receiver_cup"]
    counts = {"received": 0, "inside_receiver": 0, "remaining_source": 0,
              "outside_cups": 0, "airborne": 0}
    for i in range(p["count"]):
        row = rows[f"load_{i}"]
        inside = cup_contains(relative(row, receiver)[:3, 3], radius=p.get("ball_radius_m",PEARL_RADIUS), receptacle=p.get("cup_geometry","ripple_paper_cup"))
        remaining = cup_contains(relative(row, source)[:3, 3], radius=p.get("ball_radius_m",PEARL_RADIUS), receptacle=p.get("cup_geometry","ripple_paper_cup"))
        slow = np.linalg.norm(row["linear_velocity"]) < 0.04
        counts["inside_receiver"] += int(inside)
        counts["received"] += int(inside and slow)
        counts["remaining_source"] += int(remaining)
        counts["outside_cups"] += int(not inside and not remaining)
        counts["airborne"] += int(not inside and not remaining and row["bench_force_n"] < 0.0001
                                  and row["pose"][2] > p["table_z"])
        spilled = (row["bench_force_n"] > p.get('ball_mass_kg',PEARL_MASS) * 9.81 * 0.2
                   or row["pose"][2] < p["table_z"] - 0.02)
        if sustained(e, f"pearl_spill_{i}", spilled, seconds=0.15):
            e.flags.add(f"pearl_spilled_{i}")
    if (held(source, p["manipulation_arm"]) and source["bench_force_n"] < 0.05
            and source["pose"][2] >= e.initial["source_cup"][2] + p["lift_m"]
            and counts["remaining_source"] >= p["minimum_loaded_lift_count"]):
        e.flags.add("loaded_source_lifted")
    axis = pose_transform(source["pose"])[:3, 2]
    # An axisymmetric cup spinning in place is not falling or tipping.
    tilt_speed = np.linalg.norm(np.cross(source["angular_velocity"], axis))
    source_placed = bool(released(source) and source["bench_force_n"] > 0.05
                         and np.linalg.norm(source["linear_velocity"]) < 0.025
                         and tilt_speed < 0.3 and upright(source, angle=0.3))
    # Kinematic fixtures do not need gravity/table contact reactions. Validate
    # their measured pose instead of blocking completion on a zero support force.
    receiver_fixed = bool(np.allclose(
        pose_transform(receiver["pose"]), pose_transform(e.initial["receiver_cup"]),
        atol=1e-5, rtol=0))
    placed = {"source": source_placed, "receiver": receiver_fixed and released(receiver)}
    source_empty = counts["remaining_source"] == 0
    enough_received = counts["received"] >= p["required_received"]
    pouring_only = p.get("pouring_success_rule") == "empty_source_half_received_v1"
    if pouring_only:
        # Finish at the measured pouring result, even while the empty source is
        # still held/tilted. Lift history and placement remain diagnostics.
        complete = source_empty and enough_received and receiver_fixed
    else:
        complete = ("loaded_source_lifted" in e.flags
                    and enough_received and all(placed.values())
                    and (not p.get("require_source_empty") or source_empty))
    details = {**counts, "total": p["count"], "required_received": p["required_received"],
               "source_empty": source_empty,
               "success_rule": p.get("pouring_success_rule", "lift_pour_place"),
               "source_placement_required": not pouring_only,
               "received_fraction": counts["received"] / p["count"],
               "ever_spilled": sum(f"pearl_spilled_{i}" in e.flags for i in range(p["count"])),
               "receiver_fixed_to_table": True,
               "receiver_fixture_in_place": receiver_fixed,
               "placed_upright": placed,
               "stages": {"loaded_cup_lifted": "loaded_source_lifted" in e.flags,
                          "enough_received": enough_received,
                          "source_empty": source_empty,
                          "source_released_on_table": source_placed,
                          "receiver_fixture_in_place": receiver_fixed}}
    return bool(complete), details, ""
