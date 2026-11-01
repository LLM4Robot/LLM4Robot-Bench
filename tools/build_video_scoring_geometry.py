"""Extract collider geometry from current complete scenes without editing them."""

import json
import hashlib
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from pxr import Usd, UsdGeom, UsdPhysics

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "whiteboard_marker": ("L1-01", "/World/Household/marker"),
    "black_ceramic_mug": ("L1-01", "/World/Household/Fixture"),
    "usb_charger": ("L1-02", "/World/Household/plug"),
    "ripple_paper_cup": ("L1-04", "/World/Household/lower_cup"),
    "artificial_rose": ("L1-05", "/World/Household/rose"),
    "yellow_visor": ("L1-03", "/World/Household/hat"),
    "wooden_rack": ("L1-03", "/World/Household/Fixture"),
}
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--prop', choices=tuple(SOURCES), action='append')
args = parser.parse_args()
output = ROOT / 'assets/tuojing/video_scoring_geometry.json'
result = json.loads(output.read_text()) if args.prop else {"delivery": "video_20261015", "props": {}}
manifest = json.loads((ROOT / 'assets/tuojing/video_20261015.json').read_text())
for name, (tid, root) in SOURCES.items():
    if args.prop and name not in args.prop:
        continue
    scene_file = ROOT / 'assets/tuojing' / manifest['tasks'][tid]['path']
    if name in {'black_ceramic_mug', 'wooden_rack', 'artificial_rose', 'yellow_visor'}:
        scene_file = ROOT / 'assets/tuojing' / manifest['tasks'][tid]['runtime_path']
    stage = Usd.Stage.Open(str(scene_file))
    cache = UsdGeom.XformCache()
    inverse = np.linalg.inv(np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(root))).T)
    shapes = []
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            continue
        if name == "wooden_rack" and prim.GetName() != "Hook0":
            continue
        row = {"name": prim.GetName(), "kind": prim.GetTypeName().lower(),
               "transform": (inverse @ np.array(cache.GetLocalToWorldTransform(prim)).T).tolist()}
        if prim.IsA(UsdGeom.Mesh):
            points = np.array(UsdGeom.Mesh(prim).GetPointsAttr().Get(), dtype=float)
        elif prim.IsA(UsdGeom.Cube):
            from itertools import product
            half = UsdGeom.Cube(prim).GetSizeAttr().Get() / 2
            points = np.array(list(product((-half, half), repeat=3)))
            row["kind"] = "mesh"
        else:
            points = None
        if points is not None:
            hull = ConvexHull(points)
            chosen = sorted(hull.vertices)
            mapping = {v: i for i, v in enumerate(chosen)}
            edges = sorted({tuple(sorted((mapping[a], mapping[b])))
                            for tri in hull.simplices for a, b in zip(tri, np.roll(tri, -1))})
            row.update(points=points[chosen].tolist(), edges=edges,
                       faces=[[mapping[v] for v in tri] for tri in hull.simplices])
        elif prim.IsA(UsdGeom.Capsule) or prim.IsA(UsdGeom.Cylinder):
            schema = UsdGeom.Capsule(prim) if prim.IsA(UsdGeom.Capsule) else UsdGeom.Cylinder(prim)
            row.update(radius=schema.GetRadiusAttr().Get(), height=schema.GetHeightAttr().Get(), axis=schema.GetAxisAttr().Get())
        elif prim.IsA(UsdGeom.Sphere):
            # A sphere is a zero-height capsule, retaining the same radius.
            row.update(kind="capsule", radius=UsdGeom.Sphere(prim).GetRadiusAttr().Get(), height=0., axis="Z")
        else:
            raise ValueError(str(prim.GetPath()) + ": " + prim.GetTypeName())
        shapes.append(row)
    source = ROOT / "assets/tuojing" / manifest['tasks'][tid]['path']
    result["props"]["video:" + name] = {"task": tid, "root": root, "shapes": shapes,
        "source_scene_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    if name in {'black_ceramic_mug', 'wooden_rack', 'artificial_rose', 'yellow_visor'}:
        result['props']['video:' + name]['runtime_scene_sha256'] = hashlib.sha256(scene_file.read_bytes()).hexdigest()
    print(name, len(shapes), flush=True)
(ROOT / "assets/tuojing/video_scoring_geometry.json").write_text(json.dumps(result, separators=(",", ":")) + "\n", encoding="utf-8")
