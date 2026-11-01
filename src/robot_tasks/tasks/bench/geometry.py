"""Open hollow solids made of convex pieces; no hull seals a visible aperture."""

import math

import numpy as np

from ..l1.specs import BLUE, RED, Part

# Closed hexahedron faces, oriented outward below.
_FACES = (
    (0, 3, 2, 1),
    (4, 5, 6, 7),
    (0, 1, 5, 4),
    (1, 2, 6, 5),
    (2, 3, 7, 6),
    (3, 0, 4, 7),
)


def closed_piece(points):
    points = np.asarray(points, dtype=float)
    faces = []
    center = points.mean(axis=0)
    for quad in _FACES:
        for tri in ((quad[0], quad[1], quad[2]), (quad[0], quad[2], quad[3])):
            a, b, c = points[list(tri)]
            if np.dot(np.cross(b - a, c - a), (a + b + c) / 3 - center) < 0:
                tri = tuple(reversed(tri))
            faces.append(tri)
    return points, np.asarray(faces, dtype=int)


def cylinder(name, radius, height, xyz=(0, 0, 0), color=RED, rpy=(0, 0, 0)):
    return Part(
        name, (radius, radius, height), tuple(xyz), tuple(rpy), "cylinder", color
    )


def cup_parts(
    name="cup",
    *,
    bottom_radius=0.022,
    top_radius=0.030,
    height=0.065,
    wall=0.003,
    floor=0.004,
    color=BLUE,
    segments=24,
):
    """Origin is the bottom centre; radii describe the inner cavity."""
    parts = [
        cylinder(name + "_floor", bottom_radius + wall, floor, (0, 0, floor / 2), color)
    ]
    for i in range(segments):
        points = []
        for angle in (i * 2 * math.pi / segments, (i + 1) * 2 * math.pi / segments):
            for radius, z in (
                (bottom_radius, floor),
                (bottom_radius + wall, floor),
                (top_radius + wall, height),
                (top_radius, height),
            ):
                points.append((radius * math.cos(angle), radius * math.sin(angle), z))
        vertices, faces = closed_piece(points)
        parts.append(
            Part(
                f"{name}_wall_{i}",
                (0, 0, 0),
                shape="mesh",
                color=color,
                vertices=tuple(tuple(float(v) for v in row) for row in vertices),
                faces=tuple(tuple(int(v) for v in row) for row in faces),
            )
        )
    return tuple(parts)


def design(task_id, name, articles, fixtures=(), *, friction=0.35):
    from ..l1.specs import Design

    return Design(
        task_id,
        name,
        "household",
        tuple(articles),
        tuple(fixtures),
        "",
        version=task_id.lower() + "-household-v1",
        fixture_friction=friction,
    )
