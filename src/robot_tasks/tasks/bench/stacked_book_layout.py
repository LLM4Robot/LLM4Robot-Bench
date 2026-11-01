"""Translate the original aligned book stack and both rectangles at reset only.

The immutable delivered variants are kept for historical playback. An explicit
task setting restores the original book alignment before translating the complete
group; rotation, contact constraints, and containment rules are unchanged.
"""
from copy import deepcopy
import math

import numpy as np

from ...video_delivery import scenes

_KEY = "stacked_book_offset_x_m"
_STAMP = "robot_tasks_stacked_book_initial_shift"


def initial_record(original, task):
    if _KEY not in task:
        return original
    dx = task[_KEY]
    if (isinstance(dx, bool) or not isinstance(dx, (int, float))
            or not math.isfinite(dx) or abs(dx) > .30):
        raise ValueError("Stacked-book offset must be finite and within +/- 30 cm")
    baseline = scenes()["L3-03"]
    row = deepcopy(original)
    delta = np.array([dx, 0., 0.])
    for name in ("lower_book", "upper_book"):
        record = deepcopy(baseline["tracked"][name])
        record["pose"][:3] = (np.asarray(record["pose"][:3]) + delta).tolist()
        world = np.asarray(record["native_world_matrix"]).copy()
        world[:3, 3] += delta
        record["native_world_matrix"] = world.tolist()
        row["tracked"][name] = record
    for key in ("start_centre_xy", "target_centre_xy", "start_vertices_world",
                "target_vertices_world"):
        row[key] = (np.asarray(baseline[key]) + delta[:2]).tolist()
    row["target_outline_centres"] = (
        np.asarray(baseline["target_outline_centres"]) + delta[:2]).tolist()
    row["stacked_book_initial_shift"] = {
        "baseline_runtime_sha256": baseline["runtime_sha256"],
        "translation_world_m": delta.tolist(),
        "restored_original_book_alignment": True,
        "frame_orientation_and_dimensions_preserved": True,
    }
    return row


def apply_initial_pose(stage, row):
    plan = row.get("stacked_book_initial_shift")
    if plan is None:
        return
    from pxr import Gf, Usd, UsdGeom

    session = stage.GetSessionLayer()
    stamp = plan["baseline_runtime_sha256"] + ":" + repr(plan["translation_world_m"])
    previous = session.customLayerData.get(_STAMP)
    if previous:
        if previous != stamp:
            raise ValueError("Another stacked-book placement is already applied")
        return
    cache = UsdGeom.XformCache()
    targets = {row["tracked"][name]["path"]:
               np.asarray(row["tracked"][name]["native_world_matrix"])
               for name in ("lower_book", "upper_book")}
    # Both delivered versions have identical rectangle geometry. Restore from
    # the original stage explicitly so even prior preview rotations cannot leak
    # into this placement or disagree with the scoring polygon.
    from ...video_delivery import verified_scene
    baseline = Usd.Stage.Open(str(verified_scene("L3-03")))
    source_cache = UsdGeom.XformCache()
    for kind in ("start", "target"):
        for i in range(4):
            path = f"/World/L2/Fixture/{kind}_outline_{i}"
            source = baseline.GetPrimAtPath(path)
            if not source:
                raise ValueError("Missing original book rectangle: " + path)
            world = np.asarray(source_cache.GetLocalToWorldTransform(source)).T.copy()
            world[:3, 3] += plan["translation_world_m"]
            targets[path] = world
    with Usd.EditContext(stage, session):
        for path, world in targets.items():
            prim = stage.GetPrimAtPath(path)
            if not prim:
                raise ValueError("Missing stacked-book placement prim: " + path)
            xf = UsdGeom.Xformable(prim)
            reset = xf.GetResetXformStack()
            parent = np.asarray(cache.GetLocalToWorldTransform(prim.GetParent())).T
            local = world if reset else np.linalg.inv(parent) @ world
            xf.MakeMatrixXform().Set(Gf.Matrix4d(local.T.tolist()))
            xf.SetResetXformStack(reset)
    session.customLayerData = {**session.customLayerData, _STAMP: stamp}
