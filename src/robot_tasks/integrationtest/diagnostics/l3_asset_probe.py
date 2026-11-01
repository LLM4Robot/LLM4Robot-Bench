"""Passive prop contact diagnostics, never a robot policy or benchmark trial.

Positions/velocities are injected ONLY in these isolated fixtures. The wiping
ledger receives a declared test-grasp flag; real contacts/poses come from PhysX.
No robot is controlled, no GPT is called, and no task-success rate is inferred.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from .scene_smoke import check
from ...robots.transforms import pose_transform
from ...tasks.l1.specs import pose


def place(body, p):
    body.set_world_pose(np.asarray(p[:3]), np.asarray(p[3:]))
    body.set_linear_velocity(np.zeros(3))
    body.set_angular_velocity(np.zeros(3))


def bottle_probe(scene):
    """Passive coaxial jig + bounded loads; only closure mechanics, no robot."""
    from pxr import Gf, UsdPhysics

    backend = scene.backend
    cap = scene.bodies["cap"]
    first = scene.measure()["cap"]
    start = first["pose"]
    if pose_transform(start)[2, 2] < 0.99:
        raise AssertionError("Native loose cap is tilted before the diagnostic")
    path = scene.root + "/DiagnosticCapGuide"
    guide = UsdPhysics.Joint.Define(scene.stage, path)
    guide.CreateBody1Rel().SetTargets([scene.paths["cap"]])
    guide.CreateLocalPos0Attr(Gf.Vec3f(*start[:3]))
    guide.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    guide.CreateLocalRot0Attr(Gf.Quatf(1))
    guide.CreateLocalRot1Attr(Gf.Quatf(start[3], Gf.Vec3f(*(-np.asarray(start[4:])))))
    for axis in ("transX", "transY", "rotX", "rotY"):
        limit = UsdPhysics.LimitAPI.Apply(guide.GetPrim(), axis)
        limit.CreateLowAttr(1.0)
        limit.CreateHighAttr(-1.0)
    # The production bottle is free. Anchor it only in this declared closure
    # mechanics jig, substituting for the hand; never in the task scene builder.
    anchor_path = scene.root + "/DiagnosticBottleAnchor"
    body_pose = scene.measure()["bottle"]["pose"]
    anchor = UsdPhysics.FixedJoint.Define(scene.stage, anchor_path)
    anchor.CreateBody1Rel().SetTargets([scene.paths["bottle"]])
    anchor.CreateLocalPos0Attr(Gf.Vec3f(*body_pose[:3]))
    anchor.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    anchor.CreateLocalRot0Attr(Gf.Quatf(body_pose[3], Gf.Vec3f(*body_pose[4:])))
    anchor.CreateLocalRot1Attr(Gf.Quatf(1))
    previous = backend.before_physics_step

    def load(dt):
        if previous:
            previous(dt)
        angular = cap.get_angular_velocity()[2]
        torque = float(np.clip((0.7 - angular) * 0.004, -0.003, 0.003))
        cap._rigid_prim_view.apply_forces_and_torques_at_pos(
            forces=np.array([[0, 0, 0.055]], dtype=np.float32),
            torques=np.array([[0, 0, torque]], dtype=np.float32),
            is_global=True,
        )

    samples = []
    angle0 = math.atan2(pose_transform(start)[1, 0], pose_transform(start)[0, 0])
    turn = 0.0
    try:
        backend.before_physics_step = load
        for i in range(180):
            backend.step(scene.config.control_dt_s)
            row = scene.measure()["cap"]
            t = pose_transform(row["pose"])
            angle = math.atan2(t[1, 0], t[0, 0])
            turn += (angle - angle0 + math.pi) % (2 * math.pi) - math.pi
            angle0 = angle
            low = min(c[2] for c in row["corners"])
            gap = low - scene.initial["bottle"][2] - 0.1865
            samples.append(
                {
                    "turn_rad": turn,
                    "rise_m": row["pose"][2] - start[2],
                    "neck_clearance_m": gap,
                    "contact_n": row["pair_forces"].get("bottle"),
                }
            )
            if gap > 0.001 and row["pair_forces"].get("bottle", 0) < 0.005:
                break
        result = {
            "scope": "temporary bottle anchor/coaxial cap jig and torque <=.003 Nm/upward load .055 N; no robot/GPT, not benchmark success",
            "samples": samples,
        }
        if samples[-1]["neck_clearance_m"] <= 0.001:
            raise AssertionError(result)
        return result
    finally:
        backend.before_physics_step = previous
        scene.stage.RemovePrim(path)
        scene.stage.RemovePrim(anchor_path)


def wipe_probe(scene):
    from ...tasks.bench.wiping import WipeLedger

    backend = scene.backend
    board = scene.spec.parameters["board_pose"]
    ledger = WipeLedger(board)
    body = scene.bodies["eraser"]
    peak = 0.0
    start_x = float(ledger.world[:, 0].min()) - .025
    end_x = float(ledger.world[:, 0].max()) + .025
    for y in (-0.023, 0.022):
        place(body, pose(start_x, board[1] + y, board[2] + 0.029, roll=math.pi))
        for _ in range(6):
            backend.step(scene.config.control_dt_s)
        # Friction changes distance per control step. Cover the measured ink
        # extent rather than assuming the old two-character timed stroke fits.
        for _ in range(400):
            body.set_linear_velocity(np.array([0.12, 0, 0]))
            backend.step(scene.config.control_dt_s)
            row = scene.measure()["eraser"]
            forces = np.asarray(
                scene.pad_contact.get_contact_force_matrix(dt=scene.config.physics_dt_s)
            )
            f = float(np.linalg.norm(forces))
            peak = max(peak, f)
            ledger.update(backend.time_s(), row["pose"], pad_contact_n=f, held=True)
            if row["pose"][0] >= end_x:
                break
    result = {
        "diagnostic_clean_fraction": ledger.metrics()["clean_fraction"],
        "real_board_contact_peak_n": peak,
        "production_unheld_clean_fraction": scene.ledger.metrics()["clean_fraction"],
        "grasp_signal": "synthetic fixture flag for local ledger only; production sees no robot grasp",
    }
    if (
        peak <= 0.005
        or result["diagnostic_clean_fraction"] < 0.9
        or result["production_unheld_clean_fraction"] != 0
    ):
        raise AssertionError(result)
    return result


def sweep_probe(scene):
    from ...tasks.bench.evaluation import BenchEvaluator

    pan_pose = scene.initial["dustpan"]
    t = pose_transform(pan_pose)
    centers = (
        (-0.04, 0.02, 0.0012),
        (0, 0.02, 0.0012),
        (0.04, 0.02, 0.0012),
        (-0.02, -0.02, 0.0012),
        (0.02, -0.02, 0.0012),
    )
    for i, c in enumerate(centers[: scene.spec.parameters["count"]]):
        xyz = np.asarray(c) @ t[:3, :3].T + t[:3, 3] + [0, 0, 0.002]
        place(scene.bodies[f"block_{i}"], pose(*xyz))
    for _ in range(36):
        scene.backend.step(scene.config.control_dt_s)
    rows = scene.measure()
    e = BenchEvaluator(scene.spec, scene.initial)
    # Test only final basket geometry and physical support; not sweep acquisition.
    e.flags.update(("swept_block_0",))
    r = e.update(scene.backend.time_s(), rows)
    result = {
        "supported_blocks": r.metrics["criteria"]["blocks_in_pan"],
        "fixture_final_assessment": r.status,
        "synthetic_evidence": "tool-use flags only; actual pan/block poses and contacts measured",
        "pair_forces_n": {
            n: v["pair_forces"].get("dustpan")
            for n, v in rows.items()
            if n.startswith("block_")
        },
    }
    if result["supported_blocks"] != scene.spec.parameters["required"]:
        raise AssertionError(result)
    return result



def hook(backend, task, output):
    if task.scene.config.task_id == "L3-02":
        from .nut_probe import hook as nut_hook
        return nut_hook(backend, task, output)
    result = {
        "L3-01": bottle_probe,
        "L2-07": wipe_probe,
        "L3-04": sweep_probe,
    }[task.scene.config.task_id](task.scene)
    (output / "contact_probe.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    # Restore ordinary initial conditions before the shared initial-state audit.
    task.reset(backend, task.scene.config.seed)
    for _ in range(24):
        backend.step(task.scene.config.control_dt_s)
    return {"passed": True, "scope": __doc__, **result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("L3-01", "L3-02", "L2-07", "L3-04"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    return check(args.task, args.output, prepare=False, capture_hook=hook)


if __name__ == "__main__":
    raise SystemExit(main())
