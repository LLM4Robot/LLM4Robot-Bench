"""Native board eraser pushed into an intentionally generous triangular target."""

from dataclasses import replace
import math

import numpy as np

from ...robots.transforms import pose_transform
from ..l1.specs import Article, box, pose
from .geometry import design
from .spec import BenchSpec

VERTICES = tuple((x, y, z) for z in (0.0, 0.020)
                 for x in (-0.06, 0.06) for y in (-0.03, 0.03))
TARGET = ((0.0, 0.15), (-0.16, -0.13), (0.16, -0.13))


def build():
    a = Article(
        "eraser",
        (),
        pose(0.30, 1.24, 0.801, roll=math.pi),
        pose(0.18, 1.10, 0.78),
        0.0154,
        asset="tuojing:board_eraser",
        asset_bounds=((-0.06, -0.03, 0), (0.06, 0.03, 0.020)),
    )
    outline = []
    for i, (u, v) in enumerate(zip(TARGET, (*TARGET[1:], TARGET[0]))):
        dx, dy = v[0] - u[0], v[1] - u[1]
        outline.append(
            box(
                f"target_outline_{i}",
                (math.hypot(dx, dy), 0.002, 0.0004),
                (a.goal[0] + (u[0] + v[0]) / 2, a.goal[1] + (u[1] + v[1]) / 2, 0.7802),
                (0.7, 0.1, 0.08),
                (0, 0, math.atan2(dy, dx)),
                collision=False,
            )
        )
    return BenchSpec(
        replace(
            design("L3-03", "夹爪推板擦入宽容目标区", [a], outline),
            version="l3-14-overview-right-eraser-push-v6",
            release=True,
            provenance="2026-10-11 delivered blue eraser, original geometry and mass",
        ),
        arms=("right",),
        mechanism="native_rigid_eraser_pushing",
        facts="Use RIGHT gripper exterior to PUSH the loose blue board eraser into the large triangular outline. LEFT observes. The eraser starts above-right of the outline in overview; target is below-left. Eraser 120 x 60 x 20 mm; target 320 x 280 mm. Do not grasp, lift or carry. The complete footprint must enter the outline on the tabletop; any in-plane orientation is allowed. Withdraw without moving it back out.",
        limitations="Rigid planar pushing with a generous nonphysical target. This simplified geometric task alone does not establish L3 force adaptation; native friction is estimated.",
        parameters={
            "table_z": 0.78,
            "hold_s": 0.0,
            "target_vertices": TARGET,
            "block_vertices": VERTICES,
            "push_translation_m": 0.003,
            "push_rotation_rad": math.radians(3),
            "max_lift_m": 0.008,
            "native_instances": [
                {
                    "name": "eraser",
                    "asset": "board_eraser",
                    "pose": a.initial,
                    "tracked": {"eraser": ""},
                }
            ],
        },
    )


def containment_margin(actual_pose, goal, vertices=VERTICES, target=TARGET):
    """Signed metres to convex target sides; actual prism vertices, not an AABB.

    The very same CCW polygon is used for the rendered outline. Negative means
    some part protrudes; centre-only or enclosing-rectangle tests are insufficient.
    """
    t = np.linalg.inv(pose_transform(goal)) @ pose_transform(actual_pose)
    points = (np.asarray(vertices) @ t[:3, :3].T + t[:3, 3])[:, :2]
    margins = []
    for a, b in zip(target, (*target[1:], target[0])):
        edge = np.subtract(b, a)
        d = points - a
        margins.extend((edge[0] * d[:, 1] - edge[1] * d[:, 0]) / np.linalg.norm(edge))
    return float(min(margins))
