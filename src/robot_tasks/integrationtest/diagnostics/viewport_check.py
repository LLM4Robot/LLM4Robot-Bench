"""GUI/overview alignment and sensor-isolation check; no planner or GPT calls."""

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np


def check(task_id, output):
    from ...backends.isaacsim import IsaacSimBackend
    from ...backends.viewport import VIEW_CAMERA_PATH
    from ...registry import resolve_task

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    output.mkdir(parents=True, exist_ok=True)
    entry, config = resolve_task(task_id)
    config = entry.factory.prepare_config(config)
    config = replace(config, raw={**config.raw, "simulation": {
        **config.raw["simulation"], "headless": False}})
    backend, task = IsaacSimBackend(), entry.factory()
    backend.run_directory = output
    result = {"task": task_id, "passed": False, "gpt_called": False}
    try:
        backend.open(config)
        backend.add_r5(config.robot)
        task.build_scene(backend, config)
        backend.reset(config.seed)
        for _ in range(12):
            backend.step(config.control_dt_s)
        from omni.kit.viewport.utility import get_active_viewport
        from pxr import Gf, UsdGeom

        stage = backend.world.stage
        viewer = UsdGeom.Camera(stage.GetPrimAtPath(VIEW_CAMERA_PATH))
        source = UsdGeom.Camera(stage.GetPrimAtPath(backend.camera_rig.paths["overview"]))

        def matrix(camera):
            return UsdGeom.XformCache().GetLocalToWorldTransform(camera.GetPrim())

        before = matrix(source)
        actual = matrix(viewer)
        np.testing.assert_allclose(actual, before, atol=1e-8, rtol=0)
        assert str(get_active_viewport().camera_path) == VIEW_CAMERA_PATH
        for name in ("focalLength", "horizontalAperture", "verticalAperture", "clippingRange"):
            np.testing.assert_allclose(viewer.GetPrim().GetAttribute(name).Get(),
                                       source.GetPrim().GetAttribute(name).Get())
        # Simulate navigating only the GUI camera, then verify the policy camera
        # is unchanged and a physics step does not snap the viewer back.
        moved = Gf.Matrix4d(actual)
        moved.SetTranslateOnly(actual.ExtractTranslation() + Gf.Vec3d(.1, 0, 0))
        op = viewer.GetOrderedXformOps()[0]
        op.Set(moved)
        backend.step(config.control_dt_s)
        np.testing.assert_allclose(matrix(source), before, atol=1e-8, rtol=0)
        np.testing.assert_allclose(matrix(viewer), moved, atol=1e-8, rtol=0)
        op.Set(actual)
        from PIL import Image

        frame = next(f for f in backend.capture_rgb() if f.camera == "overview")
        Image.frombytes("RGB", (frame.width, frame.height), frame.data).save(output / "overview.png")
        result.update(passed=True, viewport=backend.scene_report["gui_viewport"],
                      world_eye=list(before.ExtractTranslation()),
                      lens_matches=True, navigation_preserves_sensor=True,
                      overview_rgb_saved=True)
    except Exception as exc:  # noqa: BLE001 - preserve failed checks without a model verdict
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        backend.close()
    print("VIEWPORT_CHECK", json.dumps(result), flush=True)
    return result["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if check(args.task, args.output) else 2)
