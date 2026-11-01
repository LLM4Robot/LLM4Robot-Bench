"""Tessellate supplier STEP (mm) into a centred, visual-only USD asset (m).

Run in the separate camera-assets Conda environment with CadQuery and usd-core.
No simulator dependency. CAD coordinates and source hashes remain in the manifest.
"""

import argparse
import hashlib
import json
from pathlib import Path


def convert(source: Path, output: Path, origin: str) -> None:
    import cadquery as cq
    import numpy as np
    from pxr import Gf, Usd, UsdGeom, Vt

    shape = cq.importers.importStep(str(source)).val()
    bbox = shape.BoundingBox()
    centre = np.array([(bbox.xmin + bbox.xmax) / 2, (bbox.ymin + bbox.ymax) / 2, (bbox.zmin + bbox.zmax) / 2])
    vertices, triangles = shape.tessellate(0.15, 0.15)
    points = (np.array([v.toTuple() for v in vertices]) - centre) / 1000
    if len(points) == 0 or len(triangles) == 0:
        raise ValueError("CAD tessellation produced no mesh")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    root = UsdGeom.Xform.Define(stage, "/CameraHousing")
    stage.SetDefaultPrim(root.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/CameraHousing/Geometry")
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    mesh.CreateFaceVertexCountsAttr([3] * len(triangles))
    mesh.CreateFaceVertexIndicesAttr(np.asarray(triangles, dtype=np.int32).flatten().tolist())
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.18, 0.20, 0.22)])
    mesh.CreateExtentAttr([Gf.Vec3f(*points.min(axis=0)), Gf.Vec3f(*points.max(axis=0))])
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    stage.SetMetadata("customLayerData", {"source_file": source.name, "source_sha256": source_hash,
                                         "source_url": origin, "purpose": "visual_only_uncalibrated"})
    stage.GetRootLayer().Save()
    manifest = {
        "source_file": source.name, "source_url": origin, "source_sha256": source_hash,
        "source_units": "mm", "source_bbox_center_mm": centre.tolist(),
        "dimensions_m": [bbox.xlen / 1000, bbox.ylen / 1000, bbox.zlen / 1000],
        "vertices": len(points), "triangles": len(triangles),
        "cadquery_version": cq.__version__, "tessellation_tolerance_mm": 0.15, "angular_tolerance_rad": 0.15,
        "usd_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "appearance": "uniform neutral colour; CAD surfaces preserved, original materials not reproduced",
        "physics": "visual only; no collision, mass or sensor optics inferred from CAD",
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--origin", required=True)
    args = parser.parse_args()
    convert(args.source.resolve(), args.output.resolve(), args.origin)
