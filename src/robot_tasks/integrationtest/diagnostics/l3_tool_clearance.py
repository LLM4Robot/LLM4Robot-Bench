"""Sample native L3 tool grasps against CAD, table and passive rests; no robot run."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from ...registry import resolve_task
from ...robots.transforms import origin_transform, pose_transform
from ...tasks.bench.spec import build_spec
from .bottle_clearance import Geometry, box_vertices
from .l2_clearance import solve_clear


def check():
    result = []
    for task_id in ("L2-07", "L3-04"):
        _, config = resolve_task(task_id)
        spec = build_spec(task_id)
        geometry = Geometry(config)
        articles = {a.name: a for a in spec.design.articles}
        obstacles = [
            (
                p.name,
                box_vertices(
                    origin_transform(p.position, p.rotation),
                    -np.asarray(p.size) / 2,
                    np.asarray(p.size) / 2,
                ),
            )
            for p in spec.design.fixtures
            if p.collision
        ]
        samples = []
        if task_id == "L2-07":
            a = articles["eraser"]
            t = pose_transform(a.initial)
            center = t[:3, 3] + t[:3, :3] @ [0, 0, -0.001]
            samples.append(("eraser_grasp", spec.arms[0], center, 0.050))
            board = spec.parameters["board_pose"]
            obstacles.append(
                (
                    "board",
                    box_vertices(
                        pose_transform(board), [-0.205, -0.14, 0], [0.205, 0.14, 0.010]
                    ),
                )
            )
            # Two sample tool contact positions; not a supplied wiping policy.
            for dx in (-0.030, 0.030):
                samples.append(
                    (
                        "pad_contact",
                        spec.arms[0],
                        (board[0] + dx, board[1], board[2] + 0.010 + 0.019),
                        0.050,
                    )
                )
        else:
            for name, arm, local, width in (
                ("broom", spec.arms[0], (0, -0.068, 0.004), 0.028),
            ):
                t = pose_transform(articles[name].initial)
                center = t[:3, 3] + t[:3, :3] @ local
                samples.append((name + "_handle", arm, center, width))
        for name, arm, xyz, width in samples:
            seed = np.array(
                next(m for m in config.robot.mounts if m.name == arm).home_positions_rad
            )
            q, worst, margin = solve_clear(
                geometry,
                origin_transform(
                    xyz,
                    (0, math.pi / 2, -math.pi / 2 if task_id == "L2-07" else math.pi),
                ),
                seed,
                width,
                obstacles,
                arm=arm,
                table_z=0.78,
            )
            result.append(
                {
                    "task": task_id,
                    "sample": name,
                    "joints": q.tolist(),
                    "table_margin_m": margin,
                    "fixture_margin_m": worst[2],
                    "nearest_fixture": worst[:2],
                }
            )
    return {
        "passed": True,
        "gpt_called": False,
        "scope": "sampled IK and conservative arm/finger/camera boxes vs table/fixed props; not continuous trajectories, stable grasps or complete robot task evidence",
        "samples": result,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    output = p.parse_args().output
    try:
        result = check()
    except Exception as exc:
        result = {"passed": False, "gpt_called": False, "error": str(exc)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
