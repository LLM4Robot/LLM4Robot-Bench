#!/usr/bin/env python3
"""Render actual task meshes in a studio view, without physics or model calls.

Run with the Isaac Python environment. Documentation cameras and lights live
only in the session layer; the task's observation rig and source USD are intact.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", choices=("drawer", "marker", "jelly"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from isaacsim import SimulationApp
    from robot_tasks.runtime import configure_windows_storage
    storage = configure_windows_storage()
    extra = [f"--/app/vulkan={'false' if os.name == 'nt' else 'true'}"]
    if storage is not None:
        extra.extend(storage.kit_arguments())
    app = SimulationApp({"headless": True, "renderer": "PathTracing",
                         "width": 1600, "height": 1200, "multi_gpu": False,
                         "extra_args": extra})
    try:
        import carb
        import numpy as np
        import omni.replicator.core as rep
        import omni.usd
        from PIL import Image
        from pxr import Gf, Usd, UsdGeom, UsdLux, UsdShade
        from robot_tasks.video_delivery import verified_scene, configure_stage
        from robot_tasks.tasks.bench.measured_assets import calibrate_jelly, calibrate_marker
        tid, path = {"drawer": ("L2-05", "/World/L2/organizer"),
                     "marker": ("L2-01", "/World/L2/marker"),
                     "jelly": ("L2-06", "/World/L2/jelly")}[args.asset]
        omni.usd.get_context().open_stage(str(verified_scene(tid)))
        stage = omni.usd.get_context().get_stage()
        configure_stage(stage, tid)
        evidence = {"task": tid, "asset": path, "model_called": False,
                    "physics_advanced": False, "source_layers_saved": False}
        if args.asset == "marker":
            from robot_tasks.tasks.bench.marker_contacts import refresh_delivered_groups
            refresh_delivered_groups(stage, path)
            evidence["calibration"] = calibrate_marker(stage, path)
        elif args.asset == "jelly":
            evidence["calibration"] = calibrate_jelly(stage)
        # Isolate the native object without re-exporting its materials/textures.
        for prim in stage.Traverse():
            name = str(prim.GetPath())
            if prim.HasAPI(UsdLux.LightAPI):
                UsdLux.LightAPI(prim).GetIntensityAttr().Set(0.)
            if prim.IsA(UsdGeom.Imageable) and not (
                    name == path or name.startswith(path + "/") or path.startswith(name + "/")):
                UsdGeom.Imageable(prim).MakeInvisible()
        root = UsdGeom.Xformable(stage.GetPrimAtPath(path))
        native = np.asarray(root.GetLocalTransformation()).T
        native_scale = np.linalg.norm(native[:3, :3], axis=0)
        matrix = Gf.Matrix4d().SetScale(Gf.Vec3d(*map(float, native_scale)))
        if args.asset == "marker":
            matrix = matrix * Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), 90))
        root.MakeMatrixXform().Set(matrix)
        bounds = UsdGeom.BBoxCache(0., ["default", "render"]).ComputeWorldBound(root.GetPrim()).ComputeAlignedRange()
        lo, hi = np.array(bounds.GetMin()), np.array(bounds.GetMax())
        target = (lo + hi) / 2
        span = float(np.linalg.norm(hi-lo))
        evidence.update(bounds_m=[lo.tolist(), hi.tolist()], preserved_asset_scale=native_scale.tolist())
        floor = UsdGeom.Cube.Define(stage, "/Studio/Floor")
        floor.CreateSizeAttr(1.)
        xf = UsdGeom.Xformable(floor)
        xf.AddTranslateOp().Set(Gf.Vec3d(float(target[0]), float(target[1]), float(lo[2]-.006)))
        xf.AddScaleOp().Set(Gf.Vec3f(5., 5., .01))
        mat = UsdShade.Material.Define(stage, "/Studio/GroundMaterial")
        shader = UsdShade.Shader.Define(stage, "/Studio/GroundMaterial/Surface")
        from pxr import Sdf
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.68,.72,.77))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.65)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(floor.GetPrim()).Bind(mat)
        dome = UsdLux.DomeLight.Define(stage, "/Studio/Dome")
        dome.CreateIntensityAttr(500.)
        dome.CreateColorAttr(Gf.Vec3f(.88,.94,1.))
        for name, offset, intensity, size in (
                ("Key", (-1., -1.2, 2.5), 1500., .8),
                ("Rim", (1., .6, 1.8), 2000., .5)):
            light = rep.create.light(light_type="disk", position=tuple(target+np.array(offset)*span),
                                     look_at=tuple(target), intensity=intensity,
                                     scale=(size*span, size*span, size*span))
        if args.asset == "drawer":
            eye = target + np.array([-.8, -1.05, 1.7]) * span
        elif args.asset == "marker":
            eye = target + np.array([.1, -1.0, .85]) * span
        else:
            eye = target + np.array([.9, -1.35, 1.]) * span
        camera = rep.create.camera(position=tuple(eye), look_at=tuple(target),
                                   focal_length={"drawer": 30., "marker": 24., "jelly": 36.}[args.asset],
                                   clipping_range=(.001, 10.))
        product = rep.create.render_product(camera, (1600, 1200))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb"); rgb.attach(product)
        settings = carb.settings.get_settings()
        settings.set_string('/rtx/rendermode', 'PathTracing')
        settings.set_int('/rtx/pathtracing/spp', 64)
        settings.set_int('/rtx/pathtracing/totalSpp', 512)
        variants = ["original", "upper_open"] if args.asset == "drawer" else ["asset"]
        for variant in variants:
            if variant == "upper_open":
                # Original task starts 30 mm open; extend the upper drawer by 50 mm.
                for name in ["Drawer1"]+[f"Contents/Cube{i:02}" for i in range(1,6)]:
                    prim=stage.GetPrimAtPath(path+'/'+name)
                    if not prim: continue
                    xf=UsdGeom.Xformable(prim);m=xf.GetLocalTransformation()
                    t=m.ExtractTranslation();m.SetTranslateOnly(t+Gf.Vec3d(0,-.05/float(native_scale[1]),0))
                    xf.MakeMatrixXform().Set(m)
            for _ in range(30):
                app.update()
            for _ in range(6):
                rep.orchestrator.step(delta_time=0., rt_subframes=8, pause_timeline=True)
                data=np.asarray(rgb.get_data())
            if data.ndim != 3 or data[...,:3].std()<1: raise RuntimeError("Empty asset render")
            Image.fromarray(np.ascontiguousarray(data[...,:3])).save(args.output/(variant+'.png'))
        evidence.update(camera_position_m=eye.tolist(), camera_target_m=target.tolist(),
                        images=[s+'.png' for s in variants], resolution=[1600,1200])
        (args.output/'render.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
        print('ASSET_RENDERED', args.asset, flush=True)
    finally:
        app.close()


if __name__ == "__main__":
    main()
