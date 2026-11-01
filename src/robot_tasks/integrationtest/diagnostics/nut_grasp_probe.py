"""Diagnostic prepositioned R5 grip + 10 mm lift. No GPT, no attachment to props.

Only this fixture prepositions both robots/parts. The production task starts empty
with a horizontal assembly on the table. It does not import this module.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from .scene_smoke import check
from ...robots.control import width_to_joints
from ...robots.transforms import pose_transform, transform_pose
from ...tasks.l1.specs import pose


def run(backend, task, output, fixture):
    from isaacsim.core.utils.types import ArticulationAction
    from PIL import Image

    scene = task.scene
    if scene.spec.parameters["shaft_tip_z"] != 0.09:
        raise ValueError("This retired 90 mm grasp fixture does not validate the current 140 mm asset")
    cfg = scene.config
    indices, targets = {}, {}
    saved = {}
    for arm, phase in (("left", "body_horizontal"), ("right", "nut_grasp")):
        instance = backend.articulations[arm]
        names = cfg.robot.joint_names + cfg.robot.gripper.joint_names
        indices[arm] = [instance.get_dof_index(n) for n in names]
        saved[arm] = instance.get_joint_positions(joint_indices=indices[arm]).copy()
        q = next(s["joints"] for s in fixture["samples"] if s["phase"] == phase)
        targets[arm] = np.array([*q, *width_to_joints(cfg.robot, 0.016)])
        instance.set_joint_positions(targets[arm], joint_indices=indices[arm])
        instance.set_joint_velocities(np.zeros(8), joint_indices=indices[arm])
        instance.apply_action(
            ArticulationAction(joint_positions=targets[arm], joint_indices=indices[arm])
        )
    body_pose = fixture["held_body_pose"]
    nut_pose = transform_pose(
        pose_transform(body_pose) @ pose_transform(pose(0, 0, 0.0875))
    )
    for name, p in (("anchor", body_pose), ("nut", nut_pose)):
        b = scene.bodies[name]
        b.set_world_pose(np.asarray(p[:3]), np.asarray(p[3:]))
        b.set_linear_velocity(np.zeros(3))
        b.set_angular_velocity(np.zeros(3))
    checkpoints = []

    def capture(label):
        rows = scene.measure()
        checkpoints.append(
            {
                "label": label,
                "objects": rows,
                "joint_positions": {
                    arm: backend.articulations[arm]
                    .get_joint_positions(joint_indices=indices[arm])
                    .tolist()
                    for arm in indices
                },
            }
        )
        for f in backend.capture_rgb():
            Image.frombytes("RGB", (f.width, f.height), f.data).save(
                output / (label + "_" + f.camera + ".png")
            )
        return rows

    try:
        for i in range(32):
            width = 0.016 - (i + 1) / 32 * 0.006
            for arm in targets:
                targets[arm][6:] = width_to_joints(cfg.robot, width)
                backend.articulations[arm].apply_action(
                    ArticulationAction(
                        joint_positions=targets[arm], joint_indices=indices[arm]
                    )
                )
            backend.step(cfg.physics_dt_s)
        capture("closing")
        for _ in range(3):
            backend.step(cfg.control_dt_s)
        before = capture("gripped")
        for name, arm in (("anchor", "left"), ("nut", "right")):
            if min(before[name]["fingers"][arm]) <= 0.005:
                raise AssertionError(
                    f'No bilateral {arm}/{name} grip: {before[name]["fingers"][arm]}'
                )
        starts = {a: q[:6].copy() for a, q in targets.items()}
        for i in range(24):
            for arm in targets:
                targets[arm][:6] = (
                    starts[arm]
                    + (np.asarray(fixture["lift_joints"][arm]) - starts[arm])
                    * (i + 1)
                    / 24
                )
                backend.articulations[arm].apply_action(
                    ArticulationAction(
                        joint_positions=targets[arm], joint_indices=indices[arm]
                    )
                )
            backend.step(cfg.control_dt_s)
        after = capture("lifted")
        for name, arm in (("anchor", "left"), ("nut", "right")):
            if min(after[name]["fingers"][arm]) <= 0.005:
                raise AssertionError(f"Grip was lost during lift: {arm}/{name}")
        rises = {n: after[n]["pose"][2] - before[n]["pose"][2] for n in before}
        if min(rises.values()) < 0.005:
            raise AssertionError(f"Objects did not follow 10 mm paired lift: {rises}")
        result = {
            "passed": True,
            "gpt_called": False,
            "rise_m": rises,
            "checkpoints": checkpoints,
            "scope": "Prepositioned fixture, actual R5 joint drives close fingers then lift both parts 10 mm. No object attachment. Does not verify autonomous pickup, reorientation, or full unscrewing.",
        }
        (output / "grasp_probe.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        return result
    finally:
        (output / "checkpoints.json").write_text(
            json.dumps(checkpoints, indent=2), encoding="utf-8"
        )
        for arm, q in saved.items():
            inst = backend.articulations[arm]
            inst.set_joint_positions(q, joint_indices=indices[arm])
            inst.apply_action(
                ArticulationAction(joint_positions=q, joint_indices=indices[arm])
            )
        task.reset(backend, cfg.seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poses", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixture = json.loads(args.poses.read_text())
    if not fixture.get("passed"):
        raise ValueError("Requires passed sampled clearance evidence")

    def hook(backend, task, output):
        return run(backend, task, output, fixture)

    return check("L3-02", args.output, prepare=False, capture_hook=hook)


if __name__ == "__main__":
    raise SystemExit(main())
