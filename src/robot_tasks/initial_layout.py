"""Deterministic initial placements and declared physical setup repairs.

The original delivery, robot, cameras, materials and task rules remain intact.
No pose writes occur during manipulation. Reset and geometric scoring use the
same transformed initial records as the scene, including passive assemblies.
"""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json

import numpy as np

from .paths import PROJECT_ROOT
from .robots.transforms import pose_transform, transform_pose
from .video_delivery import scenes


def layout(task):
    name = task.get("initial_layout")
    if not name:
        return None
    if name != "l2_p01":
        raise ValueError("Unknown initial layout: " + str(name))
    path = PROJECT_ROOT / "configs/task_references/l2_initial_p01.json"
    data = path.read_bytes()
    record = json.loads(data)["tasks"][task["id"]]
    if record["base_runtime_sha256"] != scenes()[task["id"]]["runtime_sha256"]:
        raise ValueError("Initial layout needs review against the updated asset: " + task["id"])
    return {**record, "name": name, "sha256": hashlib.sha256(data).hexdigest()}


def matrix(operation):
    angle = np.deg2rad(operation.get("yaw_deg", 0.0))
    c, s = np.cos(angle), np.sin(angle)
    result = np.eye(4)
    result[:3, :3] = ((c, -s, 0), (s, c, 0), (0, 0, 1))
    pivot = np.asarray(operation.get("pivot_world_m", (0, 0, 0)), dtype=float)
    shift = np.asarray(operation.get("translation_world_m", (0, 0, 0)), dtype=float)
    result[:3, 3] = pivot - result[:3, :3] @ pivot + shift
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite initial transform")
    return result


def delta_for(path, selected):
    matches = [matrix(op) for op in selected["operations"]
               if any(path == root or path.startswith(root + "/") for root in op["roots"])]
    if len(matches) > 1:
        raise ValueError("Overlapping initial placement roots: " + path)
    return matches[0] if matches else np.eye(4)


def scene_record(task):
    if task["id"].startswith("L1-"):
        from .tasks.bench.initial_placement import scene_record as l1_scene_record
        return l1_scene_record(task)
    from .tasks.bench.measured_assets import measured_record
    original = measured_record(scenes()[task.get("scene_variant", task["id"])], task["id"])
    if task["id"] == "L3-03":
        from .tasks.bench.stacked_book_layout import initial_record
        return initial_record(original, task)
    if task["id"] == "L3-01":
        from .tasks.bench.bottle_contacts import initial_record
        return initial_record(original)
    selected = layout(task)
    if selected is None:
        return original
    row = deepcopy(original)
    records = [*row["tracked"].values(), *row["fixtures"]]
    records.extend(row[key] for key in ("orange", "jelly") if key in row)
    for record in records:
        delta = delta_for(record["path"], selected)
        record["pose"] = list(transform_pose(delta @ pose_transform(record["pose"])))
        if "native_world_matrix" in record:
            record["native_world_matrix"] = (delta @ record["native_world_matrix"]).tolist()
    return row


def apply_record_to_stage(stage, task, row):
    """Apply runtime placements and the bottle's declared closure repair.

    Most L3 placement perturbations are baked into their USD layers. Their
    provenance is not an L1 runtime move plan. Bottle closure alignment and
    explicitly configured stacked-book translations use session-layer changes.
    L2 placement is applied separately by the backend's apply_stage hook.
    """
    if task["id"].startswith("L1-"):
        from .tasks.bench.initial_placement import apply_to_stage
        apply_to_stage(stage, row)
    elif task["id"] == "L3-01":
        from .tasks.bench.bottle_contacts import apply_initial_pose
        apply_initial_pose(stage, row)
    elif task["id"] == "L3-03":
        from .tasks.bench.stacked_book_layout import apply_initial_pose
        apply_initial_pose(stage, row)


def apply_spec(spec, task):
    selected = layout(task)
    if selected is None:
        return spec
    original = scenes()[task["id"]]
    records = dict(original["tracked"])
    records.update({key: original[key] for key in ("orange", "jelly") if key in original})
    articles = []
    for article in spec.design.articles:
        delta = delta_for(records[article.name]["path"], selected)
        articles.append(replace(article,
            initial=transform_pose(delta @ pose_transform(article.initial)),
            goal=transform_pose(delta @ pose_transform(article.goal))))
    parameters = dict(spec.parameters)
    # These are coordinate frames derived from placement, not success thresholds.
    anchor = "middle_book" if task["id"] == "L2-02" else "drawer"
    if anchor in records and "outward_axis_world" in parameters:
        delta = delta_for(records[anchor]["path"], selected)
        parameters["outward_axis_world"] = tuple(delta[:3, :3] @ parameters["outward_axis_world"])
        if "closed_pose" in parameters:
            parameters["closed_pose"] = transform_pose(delta @ pose_transform(parameters["closed_pose"]))
    design = spec.design
    if original["fixtures"] and design.fixture_pose:
        delta = delta_for(original["fixtures"][0]["path"], selected)
        design = replace(design, fixture_pose=transform_pose(delta @ pose_transform(design.fixture_pose)))
    return replace(spec, design=replace(design, articles=tuple(articles)), parameters=parameters)


def apply_stage(stage, task):
    selected = layout(task)
    if selected is None:
        return None
    from pxr import Gf, UsdGeom
    if stage.GetEditTarget().GetLayer() != stage.GetSessionLayer():
        raise ValueError("Initial placement must only author the session layer")
    cache = UsdGeom.XformCache()
    writes = []
    for operation in selected["operations"]:
        delta = matrix(operation)
        for path in operation["roots"]:
            if not path.startswith("/World/L2/"):
                raise ValueError("Initial placement may only move L2 props")
            prim = stage.GetPrimAtPath(path)
            if not prim:
                raise ValueError("Missing initial placement root: " + path)
            world = np.asarray(cache.GetLocalToWorldTransform(prim)).T
            parent = np.asarray(cache.GetLocalToWorldTransform(prim.GetParent())).T
            writes.append((prim, np.linalg.inv(parent) @ delta @ world))
    for prim, value in writes:
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTransformOp(opSuffix="initial_layout").Set(Gf.Matrix4d(value.T.tolist()))
    return selected
