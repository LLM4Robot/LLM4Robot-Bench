"""Optional annotated input: known table coordinates and measured TCP projections.

No segmentation, target object truth, grasp point, or depth is used. Raw recorded
RGB is unchanged. Projected markers can be occluded in the underlying image.
"""

import numpy as np
from PIL import ImageDraw

from .geometry import project_world
from ..backends.scene_geometry import bench_geometry


def annotate(image, calibration, robot, *, bench=None):
    if calibration is None:
        raise ValueError("Visual aids require actual camera parameters")
    draw = ImageDraw.Draw(image)
    size = image.size
    bench = bench or bench_geometry()

    def pixel(point):
        uv = project_world([point], calibration["T_world_camera_pose_wxyz"],
                           calibration["intrinsics_fx_fy_cx_cy"], calibration["distortion"])[0]
        if np.isfinite(uv).all() and 0 <= uv[0] < size[0] and 0 <= uv[1] < size[1]:
            return tuple(float(v) for v in uv)
        return None

    labels = []
    axes = [np.arange(np.ceil((c - s / 2) * 10) / 10, c + s / 2 + 1e-8, .1)
            for c, s in zip(bench.centre[:2], bench.size[:2])]
    if bench.prim_path == "/World/Bench":
        axes = [np.arange(.0, .81, .1), np.arange(-.6, .61, .1)]
    for x in axes[0]:
        for y in axes[1]:
            uv = pixel((x, y, bench.top + .001))
            if uv:
                u, v = uv
                draw.ellipse((u - 1, v - 1, u + 1, v + 1), fill="white")
                # Keep the grid, but do not cover the objects with 117 labels.
                label = f"{x:.1f},{y:.1f}".replace("-0.0", "0.0")
                box = draw.textbbox((u + 3, v + 2), label, stroke_width=1)
                if box[2] < size[0] and box[3] < size[1]:
                    labels.append((u, v, label, box))
    occupied = [(0, 0, size[0], 18)]
    for name, state in robot.items():
        if state.tcp_pose_world_wxyz is not None:
            uv = pixel(state.tcp_pose_world_wxyz[:3])
            if uv:
                u, v = uv
                draw.line((u - 5, v, u + 5, v), fill="cyan", width=2)
                draw.line((u, v - 5, u, v + 5), fill="cyan", width=2)
                draw.text((u + 6, v - 8), f"{name} TCP", fill="cyan", stroke_width=1, stroke_fill="black")
                occupied.append((u - 7, v - 12, u + 75, v + 12))
    # Spread a small number of labels across the view using screen geometry
    # only. Never use task object poses or masks to place the annotations.
    selected = []
    def distance(label):
        u, v = label[:2]
        if not selected:
            return -((u - size[0] / 2) ** 2 + (v - size[1] / 2) ** 2)
        return min((u - x) ** 2 + (v - y) ** 2 for x, y in selected)

    while labels and len(selected) < 12:
        candidate = max(labels, key=distance)
        labels.remove(candidate)
        u, v, label, box = candidate
        if any(box[0] < b[2] + 8 and box[2] > b[0] - 8 and
               box[1] < b[3] + 8 and box[3] > b[1] - 8 for b in occupied):
            continue
        draw.text((u + 3, v + 2), label, fill="white", stroke_width=1, stroke_fill="black")
        occupied.append(box)
        selected.append((u, v))
    draw.text((4, 4), "Projected table XY (m) / TCP; may be occluded", fill="white", stroke_width=1, stroke_fill="black")
    return image
