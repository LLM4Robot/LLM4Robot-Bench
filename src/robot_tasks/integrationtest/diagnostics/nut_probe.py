"""Isolated thread fixture: bounded loads, no GPT or production robot control."""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .scene_smoke import check
from ...robots.transforms import pose_transform


def probe(backend, task, output):
    from pxr import Gf, UsdPhysics

    scene = task.scene
    if scene.spec.parameters["shaft_tip_z"] != 0.09:
        raise ValueError("This retired 90 mm nut fixture does not validate the current 140 mm asset")
    body, nut = scene.bodies["anchor"], scene.bodies["nut"]
    start = scene.measure()
    body_pose, nut_pose = start["anchor"]["pose"], start["nut"]["pose"]
    root = scene.root + "/Diagnostic"
    anchor = UsdPhysics.FixedJoint.Define(scene.stage, root + "/BodyAnchor")
    anchor.CreateBody1Rel().SetTargets([scene.paths["anchor"]])
    anchor.CreateLocalPos0Attr(Gf.Vec3f(*body_pose[:3]))
    anchor.CreateLocalRot0Attr(Gf.Quatf(body_pose[3], Gf.Vec3f(*body_pose[4:])))
    anchor.CreateLocalPos1Attr(Gf.Vec3f(0))
    anchor.CreateLocalRot1Attr(Gf.Quatf(1))
    guide = UsdPhysics.Joint.Define(scene.stage, root + "/NutGuide")
    guide.CreateBody1Rel().SetTargets([scene.paths["nut"]])
    guide.CreateLocalPos0Attr(Gf.Vec3f(*nut_pose[:3]))
    guide.CreateLocalPos1Attr(Gf.Vec3f(0))
    for axis in ("transX", "transY", "rotX", "rotY"):
        lim = UsdPhysics.LimitAPI.Apply(guide.GetPrim(), axis)
        lim.CreateLowAttr(1.0)
        lim.CreateHighAttr(-1.0)
    turning = False
    previous = backend.before_physics_step

    def load(dt):
        if previous:
            previous(dt)
        omega = nut.get_angular_velocity()[2]
        target = 4 * math.pi if turning else 0.0
        torque = float(np.clip((target - omega) * 0.00015, -0.002, 0.002))
        nut._rigid_prim_view.apply_forces_and_torques_at_pos(
            forces=np.array([[0, 0, 0.08]], dtype=np.float32),
            torques=np.array([[0, 0, torque]], dtype=np.float32),
            is_global=True,
        )

    samples = []
    try:
        backend.before_physics_step = load
        z0 = nut_pose[2]
        for _ in range(12):
            backend.step(scene.config.control_dt_s)
        before = scene.measure()["nut"]
        axial_rise = before["pose"][2] - z0
        if abs(axial_rise) > 0.0005:
            raise AssertionError(
                f"Straight pulling must remain blocked: rise={axial_rise}"
            )
        turning = True
        last_yaw = math.atan2(
            pose_transform(before["pose"])[1, 0], pose_transform(before["pose"])[0, 0]
        )
        turn = 0.0
        for _ in range(72):
            backend.step(scene.config.control_dt_s)
            rows = scene.measure()
            relative = np.linalg.inv(
                pose_transform(rows["anchor"]["pose"])
            ) @ pose_transform(rows["nut"]["pose"])
            yaw = math.atan2(relative[1, 0], relative[0, 0])
            turn += math.atan2(math.sin(yaw - last_yaw), math.cos(yaw - last_yaw))
            last_yaw = yaw
            samples.append(
                {
                    "relative_bottom_z_m": float(relative[2, 3]),
                    "turns": turn / (2 * math.pi),
                    "pair_contact_n": rows["nut"]["pair_forces"].get("anchor", 0),
                }
            )
            if relative[2, 3] > 0.091:
                break
        if samples[-1]["relative_bottom_z_m"] <= 0.0903 or turn <= 0.2:
            raise AssertionError(
                {"error": "Nut did not physically screw off", "last": samples[-1]}
            )
        return {
            "passed": True,
            "gpt_called": False,
            "scope": "Diagnostic fixed body and coaxial nut guide with .08 N pull and <= .002 Nm torque; no axial position/velocity command. Not R5 or GPT success.",
            "straight_pull_rise_m": axial_rise,
            "samples": samples,
        }
    finally:
        backend.before_physics_step = previous
        scene.stage.RemovePrim(root)


def hook(backend, task, output):
    result = probe(backend, task, output)
    (output / "thread_probe.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    task.reset(backend, task.scene.config.seed)
    for _ in range(24):
        backend.step(task.scene.config.control_dt_s)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return check("L3-02", args.output, prepare=False, capture_hook=hook)


if __name__ == "__main__":
    raise SystemExit(main())
