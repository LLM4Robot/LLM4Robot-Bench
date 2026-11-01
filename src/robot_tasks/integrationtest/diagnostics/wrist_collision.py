"""Replay one recorded arm path to test wrist/fixture contact without GPT.

This intentionally bypasses planning so the physical collision stop is tested
independently. It is a controller regression, never task-solving evidence.
"""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--step", type=int, default=37)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--isolate-box", action="store_true", help="Remove brush contact only for isolated box regression")
    parser.add_argument("--check-retreat", action="store_true", help="After stopping, test a diagnostic retreat to the recorded clear pose")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (args.run / "episode.jsonl").open(encoding="utf-8")]
    start = next(r for r in rows if r.get("event") == "transition" and r.get("step") == args.step-1)
    plan = next(r for r in rows if r.get("event") == "motion_plan" and r.get("step") == args.step)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from ...backends.isaacsim import IsaacSimBackend
    from ...config import ExecutionSettings
    from ...registry import resolve_task
    from ...robots.adapters import R5Adapter
    from ...robots.kinematics import R5Kinematics
    from ...planning.trajectory import JointPath

    spec, config = resolve_task("L3-04")
    config = spec.factory.prepare_config(config)
    backend, robot, task = IsaacSimBackend(), R5Adapter(), spec.factory()
    result = {"gpt_called": False, "baseline": args.baseline, "passed": False,
              "source_run": str(args.run), "source_step": args.step,
              "scope": "recorded joint path replay; collision regression only", "isolate_box": args.isolate_box}
    try:
        backend.open(config)
        from isaacsim.core.utils.types import ArticulationAction
        from PIL import Image
        from pxr import Usd, UsdGeom, UsdPhysics
        import numpy as np

        robot.config, robot.backend = config, backend
        robot.execution = ExecutionSettings.from_config(config)
        robot.kinematics = R5Kinematics(config.robot)
        robot.instances = backend.add_r5(config.robot)
        task.build_scene(backend, config)
        if args.isolate_box:
            prim = backend.world.stage.GetPrimAtPath(task.scene.paths["broom"])
            UsdGeom.Imageable(prim).MakeInvisible()
            for child in Usd.PrimRange(prim):
                if child.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(child).CreateCollisionEnabledAttr(False)
        if args.baseline:
            for row in backend.environment_report["wrist_collision"].values():
                for path in row["colliders"]:
                    UsdPhysics.CollisionAPI(backend.world.stage.GetPrimAtPath(path)).CreateCollisionEnabledAttr(False)
        backend.reset(config.seed)
        task.reset(backend, config.seed)
        robot.reset(config.seed)
        for name, row in start["observation"]["robot"].items():
            q = tuple(row["joint_positions_rad"]) + tuple(row["finger_positions_m"])
            arm, indices = robot.instances[name], robot.indices[name]
            arm.set_joint_positions(np.array(q), joint_indices=indices)
            arm.set_joint_velocities(np.zeros(8), joint_indices=indices)
            arm.apply_action(ArticulationAction(joint_positions=np.array(q), joint_indices=indices))
            robot.targets[name] = robot.drive_targets[name] = q
        backend.before_physics_step = robot.before_physics_step
        for _ in range(3):
            backend.step(config.control_dt_s)
        from ...planning.scene import snapshot
        decision = next(r for r in rows if r.get("event") == "decision" and r.get("step") == args.step)
        planning_request = {"id": 1, "commands": decision["action"]["commands"],
                            "states": {n: asdict(s) for n, s in robot.state().items()},
                            "scene": snapshot(backend)}
        (args.output / "planning_request.json").write_text(json.dumps(planning_request), encoding="utf-8")
        result["initial_collision"] = backend.collision_monitor.blocked
        backend.collision_monitor.begin_action()
        robot.collision_stop = None
        for name, data in plan["paths"].items():
            robot.paths[name] = JointPath(tuple(tuple(p) for p in data["positions"]), tuple(data["times_s"]))
        robot.motion_arms = set(robot.paths)
        robot.path_time_s = 0.
        robot.settling = type(robot.settling)()
        duration = max(p.duration_s for p in robot.paths.values())
        feedback = None
        for _ in range(int((duration + 2) / config.control_dt_s)):
            backend.step(config.control_dt_s)
            feedback = robot.feedback()
            if feedback.status == "collision_blocked":
                break
        result.update(feedback=asdict(feedback), robot={n: asdict(s) for n, s in robot.state().items()},
                      peak_contact=backend.collision_monitor.peak,
                      contact_events=backend.collision_monitor.events,
                      passed=bool(not args.baseline and not result["initial_collision"]
                                  and feedback.status == "collision_blocked"
                                  and feedback.metrics["collision"]["component"] == "wrist_camera_or_mount"))
        for frame in backend.capture_rgb():
            Image.frombytes("RGB", (frame.width, frame.height), frame.data).save(args.output / (frame.camera + ".png"))
        (args.output / "attachments.json").write_text(json.dumps(backend.wrist_geometry), encoding="utf-8")
        if args.check_retreat and result["passed"]:
            # Explicit diagnostic command, never automatic production recovery.
            # Verify the contact latch does not prevent leaving the obstacle.
            arm_name = feedback.metrics["collision"]["arm"]
            measured = robot.state()[arm_name].joint_positions_rad
            target = tuple(start["observation"]["robot"][arm_name]["joint_positions_rad"])
            limits_by_name = {limit.joint: limit for limit in config.robot.operating_limits}
            limits = [limits_by_name[name] for name in config.robot.joint_names]
            retreat = JointPath.from_waypoints([measured, target], measured, limits)
            backend.collision_monitor.begin_action(allow_retreat=True)
            robot.collision_stop = None
            robot.paths = {arm_name: retreat}
            robot.motion_arms = {arm_name}
            robot.path_time_s = 0.
            robot.settling = type(robot.settling)()
            for _ in range(int((retreat.duration_s + 2) / config.control_dt_s)):
                backend.step(config.control_dt_s)
                retreat_feedback = robot.feedback()
                if retreat_feedback.status == "collision_blocked":
                    break
            error = max(abs(a-b) for a, b in zip(robot.state()[arm_name].joint_positions_rad, target))
            result["retreat"] = {"feedback": asdict(retreat_feedback), "joint_error_rad": error,
                                 "passed": retreat_feedback.status != "collision_blocked" and error < .02}
            result["passed"] = result["passed"] and result["retreat"]["passed"]
    finally:
        (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        backend.close()
    print("WRIST_COLLISION", json.dumps({k: result[k] for k in ("baseline", "passed", "feedback", "initial_collision")}))
    return 0 if args.baseline or result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
