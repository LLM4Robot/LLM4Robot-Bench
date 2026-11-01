#!/usr/bin/env python3
"""Rebuild original L3 placements from the recorded, invertible v3 transforms.

Run with usd-core and numpy. Only the three original-placement overlay layers
and their metadata are written; the imported corrected source layers stay intact.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from pxr import Gf, Usd, UsdGeom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from robot_tasks.robots.transforms import pose_transform, transform_pose


def main():
    path = ROOT / "assets/tuojing/video_20261015.json"
    visual_path = ROOT / "configs/task_references/video_visual_setup_20261015.json"
    data = json.loads(path.read_text())
    visual = json.loads(visual_path.read_text())
    roots = {"L3-01": "/World/L2/bottle_assembly",
             "L3-02": "/World/L2/anchor_assembly",
             "L2-07": "/World/L3Wiping/board"}
    for tid, prim_path in roots.items():
        variant = data["tasks"][tid + "_v3"]
        change = variant["initial_placement_perturbation"]
        delta = np.eye(4)
        if "transform" in change:
            delta = np.array(change["transform"], dtype=float)
        else:
            delta[:3, 3] = change["world_translation_m"]
        inverse = np.linalg.inv(delta)
        source = ROOT / "assets/tuojing" / variant["runtime_path"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != variant["runtime_sha256"]:
            raise ValueError("Changed source: " + str(source))
        stage = Usd.Stage.Open(str(source))
        cache = UsdGeom.XformCache()
        prim = stage.GetPrimAtPath(prim_path)
        world = np.array(cache.GetLocalToWorldTransform(prim)).T
        parent = np.array(cache.GetLocalToWorldTransform(prim.GetParent())).T
        local = np.linalg.inv(parent) @ inverse @ world
        destination = source.with_name("runtime_original.usda")
        overlay = Usd.Stage.CreateInMemory()
        overlay.GetRootLayer().subLayerPaths = [source.name]
        target = overlay.OverridePrim(prim_path)
        xf = UsdGeom.Xformable(target)
        xf.ClearXformOpOrder()
        xf.AddTransformOp(opSuffix="original_layout").Set(Gf.Matrix4d(local.T.tolist()))
        overlay.GetRootLayer().Export(str(destination))
        row = deepcopy(variant)
        for record in [*row["tracked"].values(), *row["fixtures"]]:
            if record["path"] == prim_path or record["path"].startswith(prim_path + "/"):
                record["pose"] = list(transform_pose(inverse @ pose_transform(record["pose"])))
                if "native_world_matrix" in record:
                    record["native_world_matrix"] = (inverse @ record["native_world_matrix"]).tolist()
        row.pop("initial_placement_perturbation")
        row["original_layout_restoration"] = {"root": prim_path, "inverse_transform": inverse.tolist(),
                                             "source_variant": tid + "_v3"}
        row["runtime_dependencies"] = [{"path": variant["runtime_path"],
                                        "sha256": variant["runtime_sha256"],
                                        "bytes": variant["runtime_bytes"]}]
        row["runtime_path"] = destination.relative_to(ROOT / "assets/tuojing").as_posix()
        row["runtime_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
        row["runtime_bytes"] = destination.stat().st_size
        data["tasks"][tid] = row
        visual["tasks"][tid]["runtime_scene_sha256"] = row["runtime_sha256"]
    for filename, record in ((path, data), (visual_path, visual)):
        filename.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
