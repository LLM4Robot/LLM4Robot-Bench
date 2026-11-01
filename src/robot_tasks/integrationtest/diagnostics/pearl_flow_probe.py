"""Gravity drainage qualification only, NOT a robot trajectory or GPT baseline.

The source cup is explicitly fixed at the selected angle above the receiver BEFORE
physics starts. All pearls are free bodies. This isolates the delivered cavity
and friction; it cannot establish a robot's ability to align, tilt or complete.
Use l2_execution_probe separately to qualify the actual loaded R5 grasp.
"""

import argparse
import json
import math
import os
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np


def check(output, *, tilt_degrees=120.0, count=None, minimum_drained_fraction=0.9):
    from ...backends.isaacsim import IsaacSimBackend
    from ...registry import resolve_task
    from ...robots.adapters import R5Adapter
    from ...robots.transforms import pose_transform, transform_pose
    from ...tasks.bench.native_l2 import NativeL2Scene
    from ...tasks.bench.pearl_pouring import cup_contains, pouring
    from ...tasks.l1.specs import pose

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    output.mkdir(parents=True, exist_ok=True)
    entry, config = resolve_task("L2-03")
    config = entry.factory.prepare_config(config)
    spec = pouring()
    p = deepcopy(spec.parameters)
    if not 0 <= tilt_degrees <= 180 or not 0 <= minimum_drained_fraction <= 1:
        raise ValueError("Tilt must be in [0, 180] degrees and drainage fraction in [0, 1]")
    count = p["count"] if count is None else count
    if not 1 <= count <= p["count"]:
        raise ValueError("Diagnostic count must be between 1 and the task's current pearl count")
    # Count overrides are fixture comparisons only; never alter the task config.
    p["native_instances"] = p["native_instances"][:2 + count]
    p["count"] = count
    p["required_received"] = math.ceil(count * 0.8)
    p["loaded_source_mass_kg"] = .014 + count * p["ball_mass_kg"]
    spec = replace(spec, design=replace(spec.design, articles=spec.design.articles[:2 + count]))
    source = p["native_instances"][0]
    original = pose_transform(source["pose"])
    p["initial_packing_top_m"] = max(
        item["pose"][2] - source["pose"][2] + p["ball_radius_m"]
        for item in p["native_instances"][2:])
    receiver_xyz = np.asarray(p["native_instances"][1]["pose"][:3])
    direction = float(np.sign(receiver_xyz[0] - original[0, 3]))
    tilted = pose(*(receiver_xyz + [-.135 * direction, 0, .231]), pitch=direction * math.radians(tilt_degrees))
    source["pose"], source["static"] = tilted, [""]
    transform = pose_transform(tilted) @ np.linalg.inv(original)
    for item in p["native_instances"]:
        if item["name"].startswith("load_"):
            item["pose"] = transform_pose(transform @ pose_transform(item["pose"]))
    spec = replace(spec, parameters=p)
    backend, robot = IsaacSimBackend(), R5Adapter()
    backend.run_directory = output
    result = {"scope": "Fixed tilted cup gravity-drain fixture; no robot pour or GPT success claim",
              "gpt_called": False, "passed": False, "source_kinematic": True,
              "tilt_degrees": tilt_degrees, "count": count,
              "minimum_drained_fraction": minimum_drained_fraction,
              "samples": []}
    try:
        backend.open(config)
        robot.initialize(backend, config)
        scene = NativeL2Scene(backend, config, spec)
        from pxr import UsdPhysics

        result["receiver_kinematic"] = bool(UsdPhysics.RigidBodyAPI(
            backend.world.stage.GetPrimAtPath(scene.paths["receiver_cup"])
        ).GetKinematicEnabledAttr().Get())
        result["receiver_planning_colliders"] = sum(
            path.startswith(scene.paths["receiver_cup"] + "/")
            for path in backend.planning_fixture_paths)
        scene.reset(config.seed)
        robot.reset(config.seed)
        for step in range(61):
            if step:
                backend.step(config.control_dt_s)
            if step % 12:
                continue
            rows = scene.measure()
            counts = {}
            for name in ("source_cup", "receiver_cup"):
                inverse = np.linalg.inv(pose_transform(rows[name]["pose"]))
                counts[name] = sum(cup_contains((inverse @ np.array([*rows[f"load_{i}"]["pose"][:3], 1]))[:3])
                                   for i in range(p["count"]))
            result["samples"].append({"time_s": backend.time_s(), **counts,
                                      "receiver_displacement_m": float(np.linalg.norm(
                                          np.asarray(rows["receiver_cup"]["pose"][:3])
                                          - scene.initial["receiver_cup"][:3])),
                                      "receiver_up_z": float(pose_transform(rows["receiver_cup"]["pose"])[2, 2])})
            from PIL import Image
            for frame in backend.capture_rgb():
                if frame.camera == "overview":
                    Image.frombytes("RGB", (frame.width, frame.height), frame.data).save(output / f"flow_{step:02}.png")
            print("PEARL_FLOW", step, counts, flush=True)
        result["drained"] = p["count"] - counts["source_cup"]
        result["drained_fraction"] = result["drained"] / p["count"]
        result["passed"] = (result["samples"][0]["source_cup"] == p["count"]
                            and result["drained_fraction"] >= minimum_drained_fraction)
        result["max_receiver_displacement_m"] = max(s["receiver_displacement_m"] for s in result["samples"])
        result["passed"] &= (result["receiver_kinematic"]
                              and result["receiver_planning_colliders"] > 0
                              and result["max_receiver_displacement_m"] < 1e-5
                              and all(s["receiver_up_z"] > .9999 for s in result["samples"]))
        result["received_in_fixture"] = counts["receiver_cup"]
        result["note"] = "Fixed-angle drainage gate, not a gradual tilt, robot pour, or formal task score; fixture alignment is not a policy input."
    except Exception as exc:  # noqa: BLE001 - persist simulator failures
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        robot.close()
        backend.close()
    return result["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tilt-degrees", type=float, default=120.0)
    parser.add_argument("--count", type=int, help="Diagnostic-only subset of the task load")
    parser.add_argument("--minimum-drained-fraction", type=float, default=0.9)
    args = parser.parse_args()
    raise SystemExit(0 if check(args.output, tilt_degrees=args.tilt_degrees,
                              count=args.count,
                              minimum_drained_fraction=args.minimum_drained_fraction) else 2)
