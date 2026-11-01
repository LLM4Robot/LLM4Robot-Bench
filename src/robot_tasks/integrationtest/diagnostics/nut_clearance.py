"""Sampled native nut grasp/regrasp clearances; not a task policy or full path test."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .bottle_clearance import Geometry, box_vertices
from .l2_clearance import solve_clear
from ...registry import resolve_task
from ...robots.transforms import origin_transform, pose_transform
from ...tasks.screw.nut import nut_spec


def check():
    _, config = resolve_task("L3-02")
    s = nut_spec()
    if s.parameters["shaft_tip_z"] != 0.09:
        raise ValueError("This retired 90 mm clearance fixture does not validate the current 140 mm asset")
    g = Geometry(config)
    base = np.array(s.design.articles[0].initial[:3])
    initial_transform = pose_transform(s.design.articles[0].initial)
    seeds = {m.name: np.array(m.home_positions_rad) for m in config.robot.mounts}
    samples = []
    support_boxes = None
    for label, lift in (("body_grasp", 0.0), ("body_lift", 0.080)):
        b = base + [0, 0, lift]
        goal = origin_transform(
            b + [0.025, 0, 0.010], (0, math.pi / 2, math.pi)
        )
        body_transform = initial_transform.copy()
        body_transform[2, 3] += lift
        obstacles = [
            (
                "anchor",
                box_vertices(
                    body_transform,
                    [-0.009, -0.009, 0],
                    [0.009, 0.009, 0.094],
                ),
            )
        ]
        q, worst, margin = solve_clear(
            g,
            goal,
            seeds["left"],
            0.012,
            obstacles,
            arm="left",
            table_z=0.78,
            contact_pairs=(("link7", "anchor"), ("link8", "anchor")),
        )
        seeds["left"] = q
        support_boxes, _ = g.boxes("left", q, 0.012)
        samples.append(
            {
                "phase": label,
                "arm": "left",
                "tcp_position": goal[:3, 3].tolist(),
                "rpy": [0, math.pi / 2, math.pi],
                "width_m": 0.012,
                "joints": q.tolist(),
                "table_clearance_m": margin,
                "worst": worst,
            }
        )
    # Keep the horizontal starting axis; sample separated hand approaches.
    b = np.array([base[0], base[1], 0.96])
    body_transform = origin_transform(b, (0, math.pi / 2, 0))
    whole = box_vertices(body_transform, [-0.009, -0.009, 0], [0.009, 0.009, 0.094])
    left_goal = origin_transform(b + [0.020, 0, 0.0], (0, math.pi / 2, math.pi))
    ql, worst, margin = solve_clear(
        g,
        left_goal,
        seeds["left"],
        0.012,
        [("anchor", whole)],
        arm="left",
        table_z=0.78,
        contact_pairs=(("link7", "anchor"), ("link8", "anchor")),
    )
    support_boxes, _ = g.boxes("left", ql, 0.012)
    samples.append(
        {
            "phase": "body_horizontal",
            "arm": "left",
            "tcp_position": left_goal[:3, 3].tolist(),
            "rpy": [0, math.pi / 2, math.pi],
            "width_m": 0.012,
            "joints": ql.tolist(),
            "table_clearance_m": margin,
            "worst": worst,
        }
    )
    obstacles = [("left/" + name, shape) for name, shape in support_boxes.items()]
    obstacles.append(
        (
            "anchor_below_nut",
            box_vertices(body_transform, [-0.009, -0.009, 0], [0.009, 0.009, 0.0742]),
        )
    )
    right_goal = origin_transform(b + [0.093, 0, 0], (0, 0, math.pi))
    qr, worst, margin = solve_clear(
        g, right_goal, seeds["right"], 0.013, obstacles, arm="right", table_z=0.78
    )
    samples.append(
        {
            "phase": "nut_grasp",
            "arm": "right",
            "tcp_position": right_goal[:3, 3].tolist(),
            "rpy": [0, 0, math.pi],
            "width_m": 0.013,
            "joints": qr.tolist(),
            "table_clearance_m": margin,
            "worst": worst,
        }
    )
    # A small common upward move supplies a separate physical-grasp diagnostic.
    lift_joints = {}
    from ...robots.transforms import transform_pose

    for arm, goal, q in (("left", left_goal, ql), ("right", right_goal, qr)):
        target = goal.copy()
        target[2, 3] += 0.010
        lift_joints[arm] = g.solve(arm, transform_pose(target), q).tolist()
    return {
        "passed": True,
        "task": "L3-02",
        "gpt_called": False,
        "samples": samples,
        "held_body_pose": [*b, *transform_pose(body_transform)[3:]],
        "lift_joints": lift_joints,
        "scope": "Sampled horizontal table pickup and raised bimanual IK with conservative arm/wrist/prop boxes. Not continuous paths or physical grip success.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = check()
    except Exception as exc:
        result = {"passed": False, "error": str(exc)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
