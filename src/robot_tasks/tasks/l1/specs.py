"""Shared rigid article geometry for the maintained task scenes."""

from dataclasses import dataclass

import numpy as np

from ...robots.transforms import origin_transform, transform_pose

RED = (0.85, 0.16, 0.05)
BLUE = (0.08, 0.3, 0.8)
GREY = (0.38, 0.43, 0.48)
GREEN = (0.08, 0.65, 0.25)
YELLOW = (0.95, 0.7, 0.08)


def pose(x, y, z, roll=0.0, pitch=0.0, yaw=0.0):
    return transform_pose(origin_transform((x, y, z), (roll, pitch, yaw)))


@dataclass(frozen=True)
class Part:
    name: str
    size: tuple[float, float, float]
    position: tuple = (0.0, 0.0, 0.0)
    rotation: tuple = (0.0, 0.0, 0.0)
    shape: str = "box"
    color: tuple = RED
    collision: bool = True
    vertices: tuple = ()
    faces: tuple = ()


@dataclass(frozen=True)
class Article:
    name: str
    parts: tuple[Part, ...]
    initial: tuple
    goal: tuple
    mass: float = 0.06
    asset: str = ""
    asset_size: tuple = ()
    asset_rotation: tuple = (0.0, 0.0, 0.0)
    asset_bounds: tuple = ()  # Optional (min_xyz, max_xyz), for native uncentred metre assets.


@dataclass(frozen=True)
class Design:
    task_id: str
    name: str
    kind: str
    articles: tuple[Article, ...]
    fixtures: tuple[Part, ...]
    instruction: str
    fixture_top: float = 0.0
    clearance: float = 0.005
    release: bool = True
    fixture_asset: str = ""
    fixture_size: tuple = ()
    fixture_pose: tuple = ()
    provenance: str = "Project-authored parametric rigid test article v1"
    version: str = "l1-rigid-v1"
    fixture_friction: float = 0.8


def box(name, size, xyz=(0.0, 0.0, 0.0), color=GREY, rpy=(0.0, 0.0, 0.0), collision=True):
    return Part(name, tuple(size), tuple(xyz), tuple(rpy), color=color, collision=collision)


def local_corners(article):
    """Conservative visual/collision envelope for process and containment checks."""
    from itertools import product

    if article.asset:
        if article.asset_bounds:
            return np.array(list(product(*zip(*article.asset_bounds))))
        return np.array(list(product(*[(-s / 2, s / 2) for s in article.asset_size])))
    points = []
    for part in article.parts:
        if not part.collision:
            continue
        if part.shape == "mesh":
            t = origin_transform(part.position, part.rotation)
            points.extend(np.asarray(part.vertices) @ t[:3, :3].T + t[:3, 3])
            continue
        size = part.size if part.shape == "box" else (2 * part.size[0], 2 * part.size[1], part.size[2])
        corners = np.array(list(product(*[(-s / 2, s / 2) for s in size])))
        t = origin_transform(part.position, part.rotation)
        points.extend(corners @ t[:3, :3].T + t[:3, 3])
    return np.asarray(points)
