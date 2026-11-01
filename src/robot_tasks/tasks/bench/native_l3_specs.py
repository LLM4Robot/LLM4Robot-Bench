"""Delivered rigid props in the laboratory frame; no pregrasp or action script."""

import math
from dataclasses import replace

from ..l1.specs import Article, box, pose
from .geometry import design
from .spec import BenchSpec
from .native_l2_specs import support_box

TABLE_Z = 0.78


def _article(name, asset, initial, mass, bounds):
    return Article(
        name, (), initial, initial, mass, asset="tuojing:" + asset, asset_bounds=bounds
    )


def bottle():
    base = pose(0.19565, 1.30, TABLE_Z)
    # Source cap geometry is expressed in the bottle frame, not at cap centre.
    # Three-start photo-estimated RH threads have 9 mm lead per revolution.
    # +7.5 mm / +300 degrees leaves the last partial engagement, no seal preload.
    lift, turn = 0.0075, 5 * math.pi / 3
    cap = pose(base[0], base[1], base[2] + lift, yaw=turn)
    articles = [
        _article(
            "bottle",
            "oriental_leaf_335ml",
            base,
            0.358,
            ((-0.0267, -0.0267, 0.001), (0.0267, 0.0267, 0.1865)),
        ),
        _article(
            "cap",
            "oriental_leaf_335ml",
            cap,
            0.0035,
            ((-0.02078, -0.02078, 0.1742), (0.02078, 0.02078, 0.1900)),
        ),
    ]
    return BenchSpec(
        replace(
            design("L3-01", "固定瓶身旋下瓶盖", articles),
            version="l3-03-real-left-overview-right-v9",
            release=False,
            provenance="Delivered Oriental Leaf 335 ml body/cap, pre-loosened initial state",
        ),
        arms=("right",),
        mechanism="native_fixed_bottle_loose_cap_contact",
        limitations="Native empty-bottle appearance but delivered body mass still 358 g, not an empty-bottle weighing. Body anchored by task request. Cap pre-loosened 7.5 mm/300 degrees; photo-estimated interrupted threads and friction. No claim of real opening torque.",
        facts="The square Oriental Leaf bottle stands at the centre between both arms and is FIXED to the table. RIGHT unscrews the loose cap counterclockwise as seen from above; LEFT only observes. No support-hand grasp is needed. Body width approximately 52 mm, cap diameter 42 mm. The cap is pre-loosened but still threaded, with 9 mm lead per turn. Complete the task as soon as the cap is fully detached from the neck. Keeping it held is allowed; no additional lift height, placement, release or return home is required. Do not try to move the fixed bottle.",
        parameters={
            "table_z": TABLE_Z,
            "hold_s": 0.0,
            "body_arm": "left", "body_fixed": True,
            "cap_arm": "right",
            "cap_bottom_z": 0.1742,
            "cap_top_z": 0.1900,
            "cap_radius_m": 0.02078,
            "neck_radius_m": 0.019,
            "neck_top_z": 0.1865,
            "thread_lead_m": 0.009,
            "initial_cap_lift_m": lift,
            "native_instances": [
                {
                    "name": "bottle_assembly",
                    "asset": "oriental_leaf_335ml",
                    "pose": base,
                    "tracked": {"bottle": "Body", "cap": "Cap"},
                    "static": ["Body"],
                    "initial_body_transforms": {"Cap": pose(0, 0, lift, yaw=turn)},
                }
            ],
        },
    )


def wiping():
    board = pose(0.25, 1.29, TABLE_Z, yaw=math.pi)
    eraser = pose(0.565, 1.28, TABLE_Z + 0.021, roll=math.pi, yaw=math.pi / 2)
    a = _article("eraser", "board_eraser", eraser, 0.0154,
                 ((-0.06, -0.03, 0), (0.06, 0.03, 0.020)))
    return BenchSpec(
        replace(design("L2-07", "抓板擦擦除原生字迹", [a]),
                version="l3-11-any-ink-removal-v6", release=False),
        arms=("right",), mechanism="native_contact_wiping",
        limitations="Original photo-derived GPT6 mask, rigid eraser and contact-driven single-pass removal. Cleaning proxy, not real ink chemistry. No scripted grasp or automatic cleaning motion.",
        facts="The board lies flat, fixed, in front of the arms; its original red GPT6 writing is upside-down from overview. The loose blue eraser lies pad-down to image-right of the board in the overview camera. RIGHT must pick it up; LEFT observes only. Eraser dimensions 120 x 60 x 20 mm: pinch the 60 mm width with the cleaning pad unobstructed below. Its cloth face is opposite the blue plastic back. Contact and sweep the cloth over the writing; each covered mark disappears on the first genuine wipe, no repeated scrubbing or exact force required. The task succeeds as soon as a genuine contact wipe removes any part of the original writing. No percentage target, further wiping, placement, release or return home is required.",
        parameters={"table_z": TABLE_Z, "hold_s": 0.0,
                    "required_clean_cells": 1, "board_pose": board,
                    "board_surface_z": 0.010, "pad_half_extents": (0.059, 0.029),
                    "pad_z": 0.020,
                    "native_instances": [
                      {"name": "board", "asset": "green_board_20261008", "pose": board,
                       "static": [""], "tracked": {}},
                      {"name": "eraser", "asset": "board_eraser", "pose": eraser,
                       "tracked": {"eraser": ""}}]})


def sweeping():
    # Match overview photo: pan left, green box right, brush leaning against
    # the box's left wall, two loose blue cubes between brush and pan.
    pan = pose(0.02, 1.26, TABLE_Z + 0.00145, yaw=-math.pi / 2)
    # Native +Y points toward bristles. Tilt it down and to image-left,
    # leaving the narrow top of the handle clear of the box for a side pinch.
    broom = pose(0.237, 1.26, TABLE_Z + 0.070, roll=-math.radians(70), yaw=math.pi / 2)
    articles = [
        _article("dustpan", "dustpan", pan, 0.0232,
                 ((-0.0709, -0.1049, -0.00145), (0.0709, 0.04502, 0.0145))),
        _article("broom", "hand_brush", broom, 0.0365,
                 ((-0.0741, -0.0693, -0.00059), (0.0739, 0.07404, 0.00679)))]
    instances = [{"name": a.name, "asset": a.asset.removeprefix("tuojing:"),
                  "pose": a.initial, "tracked": {a.name: ""}} for a in articles]
    instances[0]["static"] = [""]
    instances.append(support_box(0.35, 1.26))
    for i, (x, y) in enumerate(((0.145, 1.295), (0.158, 1.235))):
        a = _article(f"block_{i}", "blue_square", pose(x, y, TABLE_Z + 0.001),
                     0.0026, ((-0.01, -0.01, 0), (0.01, 0.01, 0.020)))
        articles.append(a)
        instances.append({"name": a.name, "asset": "blue_square", "pose": a.initial,
                          "tracked": {a.name: ""}})
    return BenchSpec(
        replace(design("L3-04", "持刷扫两块入固定簸箕", articles),
                version="l3-15-overview-right-brush-v6", release=True),
        arms=("right",), mechanism="native_rigid_brush_sweeping",
        limitations="Rigid native bristle contact, not flexible fibers. Pan/green box anchored. Brush is free and rests against the box, no attachment. Real setup's pan fixation was not measured.",
        facts="From overview: gray dustpan on LEFT, green box on RIGHT, brush leaning against the box's LEFT side with handle high and bristles down, TWO blue 20 mm cubes between pan and brush. The pan mouth faces RIGHT and is fixed. RIGHT picks the brush near its narrow upper handle, brings bristles to table height and sweeps both cubes through the thin lip. LEFT observes only and must not touch the pan. Do not move cubes with the gripper itself. Once both cubes rest inside the pan, put down the brush on the table or against the box and release.",
        parameters={"table_z": TABLE_Z, "hold_s": 0.0, "count": 2, "required": 2,
                    "pan_floor_z": 0.0013, "native_instances": instances})
