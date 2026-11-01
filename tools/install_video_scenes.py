"""Install the complete 2026-10-15 delivery, preserving USDZ bytes and history.

Run in a Python environment with usd-core. Only installation reads/writes assets;
runtime references the checked task-local scene instead of importing older props.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


ROOTS = {
    "L1-01": {"marker": "/World/Household/marker"},
    "L1-02": {"plug": "/World/Household/plug", "plug_storage_base": "/World/Household/plug_storage_base"},
    "L1-03": {"hat": "/World/Household/hat", "source_box": "/World/Household/source_box"},
    "L1-04": {"lower_cup": "/World/Household/lower_cup", "upper_cup": "/World/Household/upper_cup"},
    "L1-05": {n: "/World/Household/" + n for n in ("rose", "source_cup", "target_cup", "source_charger")},
    "L2-01": {"body": "/World/L2/marker/Body", "cap": "/World/L2/marker/Cap"},
    "L2-02": {n: "/World/L2/" + n for n in ("lower_book", "middle_book", "upper_book")},
    "L2-03": {n: "/World/L2/" + n for n in ("source_cup", "receiver_cup", *[f"load_{i}" for i in range(5)])},
    "L2-05": {"drawer": "/World/L2/organizer/Drawer1"},
    "L3-01": {"bottle": "/World/L2/bottle_assembly/Body", "cap": "/World/L2/bottle_assembly/Cap"},
    "L3-02": {"anchor": "/World/L2/anchor_assembly/AnchorBody", "nut": "/World/L2/anchor_assembly/Nut"},
    "L2-07": {"eraser": "/World/L3Wiping/eraser"},
    "L3-03": {"eraser": "/World/L2/eraser"},
    "L3-04": {n: "/World/L2/" + n for n in ("dustpan", "broom", "block_0", "block_1")},
}

FIXTURES = {
    **{t: ["/World/Household/Fixture"] for t in ("L1-01", "L1-02", "L1-03")},
    "L1-04": [], "L1-05": [],
    **{t: ["/World/L2/support_box"] for t in ("L2-01", "L2-02", "L3-02", "L3-04")},
    "L2-03": [], "L2-04": ["/World/L2/Pedestal"],
    "L2-05": ["/World/L2/organizer/Body", "/World/L2/organizer/Drawer0", "/World/L2/support_box"],
    "L3-01": [], "L2-07": ["/World/L3Wiping/board"], "L3-03": [],
}


def install(delivery, root):
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics
    from robot_tasks.robots.transforms import transform_pose
    from robot_tasks.task_numbering import from_legacy

    delivery, root = delivery.resolve(), root.resolve()
    destination = root / "assets/tuojing/source"
    staging = root / "build/reference_alignment_20261015/source_latest_staging"
    staging.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "assets/tuojing/video_20261015.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"tasks": {}}
    manifest = {"schema_version": 1, "delivery": "L1-L3_assets_20261015", "tasks": {}}
    for path in sorted(delivery.glob("*.usdz")):
        # Vendor archives keep their original labels; manifest keys use the
        # current catalogue. Never infer an asset directory from a new ID.
        source_id = path.stem
        tid = from_legacy(source_id)
        s = Usd.Stage.Open(str(path))
        cache = UsdGeom.XformCache()
        bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])

        from video_scene_metadata import record as body_record
        def record(prim_path):
            return body_record(s, prim_path)

        task = {"path": f"source/{source_id}/scene.usdz", "bytes": path.stat().st_size,
                "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
                "source_units_metadata": {"up": UsdGeom.GetStageUpAxis(s), "meters_per_unit": UsdGeom.GetStageMetersPerUnit(s)},
                "effective_units": {"up": "Z", "meters_per_unit": 1.0},
                "tracked": {n: record(p) for n, p in ROOTS.get(tid, {}).items()},
                "fixtures": [record(p) for p in FIXTURES[tid]],
                "robots": {n: record("/World/Robots/R5_" + n.title()) for n in ("left", "right")},
                "cameras": {}}
        for name, parent, prim_path, resolution in (
            ("overview", "world", "/World/Cameras/ThirdPerson", (1280, 960)),
            ("left_wrist", "left", "/World/Robots/R5_Left/link6/WristSensor/Sensor/Camera", (1280, 720)),
            ("right_wrist", "right", "/World/Robots/R5_Right/link6/WristSensor/Sensor/Camera", (1280, 720)),
        ):
            camera = UsdGeom.Camera(s.GetPrimAtPath(prim_path))
            matrix = np.asarray(cache.GetLocalToWorldTransform(camera.GetPrim())).T
            if parent != "world":
                frame = s.GetPrimAtPath("/World/Robots/R5_" + parent.title() + "/link6")
                matrix = np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(frame)).T) @ matrix
            f, w, h = camera.GetFocalLengthAttr().Get(), camera.GetHorizontalApertureAttr().Get(), camera.GetVerticalApertureAttr().Get()
            task["cameras"][name] = {"path": prim_path, "parent": parent, "pose": list(transform_pose(matrix)),
                                     "resolution": resolution, "intrinsics": [resolution[0]*f/w, resolution[1]*f/h, resolution[0]/2, resolution[1]/2],
                                     "clipping_m": list(camera.GetClippingRangeAttr().Get())}
        if tid == "L2-04":
            task["orange"] = record("/World/L2/orange")
        if tid == "L1-05":
            task["cup_contents"] = [str(p.GetPath()) for p in s.Traverse()
                                    if str(p.GetPath()).startswith("/World/Household/") and any(x in str(p.GetPath()).lower() for x in ("pearl", "marble", "charger", "filler"))]
        from video_scene_metadata import enrich
        enrich(s, task, tid)
        old = previous["tasks"].get(tid, {})
        if old.get("sha256") == task["sha256"]:
            task.update({k: v for k, v in old.items() if k.startswith("runtime_")
                         or k in ('visual_overlay','effective_tracked_overrides','derived_contents')})
            task['tracked'].update(task.get('effective_tracked_overrides',{}))
            task['tracked'].update(task.get('derived_contents',{}).get('added',{}))
        folder = staging / source_id
        folder.mkdir(exist_ok=True)
        shutil.copy2(path, folder / "scene.usdz")
        manifest["tasks"][tid] = task
        print(tid, task["bytes"], "installed", flush=True)
    if len(manifest["tasks"]) != 15:
        raise ValueError("Expected all 15 delivered scenes")
    backup = root / "build/reference_alignment_20261015/source_before_latest"
    if destination.exists():
        if backup.exists():
            for tid, row in manifest["tasks"].items():
                installed = root / "assets/tuojing" / row["path"]
                if not installed.is_file() or hashlib.sha256(installed.read_bytes()).hexdigest() != row["sha256"]:
                    raise ValueError("Existing latest source differs; preserved backup and local source: " + tid)
            shutil.rmtree(staging)
        else:
            destination.rename(backup)
            staging.rename(destination)
    else:
        staging.rename(destination)
    for key in ('source_archive','visual_release','visual_textures'):
        if key in previous:
            manifest[key] = previous[key]
    temporary = manifest_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("delivery", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    install(args.delivery, args.root)
