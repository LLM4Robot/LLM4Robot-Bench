"""Isolated R5 contact/short-motion diagnostic, never a scripted task baseline.

An explicit sampled-IK fixture starts OPEN fingers around the prop. Subsequent
closing and motion use the production adapter/planner. Props are never attached
or teleported. This does not validate visual grasp selection or full task success.
Run in Isaac Python, with --poses pointing to l2_clearance's diagnostic JSON.
"""

import argparse
import json
import os
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np


def check(task_id, poses, output, width=None, velocity_iterations=None, travel=0.020, marker_pair=False, step_close=False, finish=False, physics_hz=None, soft_physics=None):
    from ...backends.isaacsim import IsaacSimBackend
    from ...registry import resolve_task
    from ...robots.adapters import R5Adapter
    from ...robots.control import width_to_joints
    from ...types import Action, ArmCommand, EvaluationState

    if task_id == "L2-05":
        raise ValueError(
            "The added drawer handle has been removed; old grasp fixtures are invalid. "
            "Qualify a native-lip contact fixture before running this diagnostic."
        )
    if step_close and not marker_pair:
        raise ValueError("Old step-close control requires the paired marker diagnostic")
    if marker_pair and task_id != "L2-01":
        raise ValueError("Paired marker diagnostic only supports L2-01")
    if finish and task_id not in {"L2-03", "L2-04", "L2-05"}:
        raise ValueError("Finish qualification supports only pouring, orange and drawer")
    maximum = .085 if finish and task_id == "L2-05" else (.080 if task_id == "L2-04" else .05)
    if not np.isfinite(travel) or not 0 < travel <= maximum:
        raise ValueError(f"Probe travel must be between 0 and {maximum} m")
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    output.mkdir(parents=True, exist_ok=True)
    spec, config = resolve_task(task_id)
    config = spec.factory.prepare_config(config)
    if soft_physics is not None:
        if task_id != "L2-04":
            raise ValueError("Soft physics overrides require L2-04")
        overrides = json.loads(soft_physics.read_text(encoding="utf-8"))
        config = replace(config, raw={**config.raw, "orange_physics": {
            **config.raw.get("orange_physics", {}), **overrides}})
    if physics_hz is not None:
        if physics_hz not in (240, 480, 960):
            raise ValueError("Diagnostic physics rate must be 240, 480 or 960 Hz")
        config = replace(config, physics_dt_s=1.0 / physics_hz)
    if velocity_iterations is not None:
        config = replace(
            config,
            raw={
                **config.raw,
                "simulation": {
                    **config.raw["simulation"],
                    "solver_velocity_iterations": velocity_iterations,
                },
            },
        )
    fixture = next(
        r for r in json.loads(poses.read_text())["results"] if r["task"] == task_id
    )
    active = (config.arms[0],)
    initial = {a: next(s for s in fixture["samples"] if s["arm"] == a) for a in active}
    config = replace(
        config,
        robot=replace(
            config.robot,
            mounts=tuple(
                replace(m, home_positions_rad=tuple(initial[m.name]["joints"]))
                if m.name in initial
                else m
                for m in config.robot.mounts
            ),
        ),
    )
    backend, robot, task = IsaacSimBackend(), R5Adapter(), spec.factory()
    backend.run_directory = output
    result = {
        "task": task_id,
        "passed": False,
        "gpt_called": False,
        "diagnostic_step_close": step_close,
        "scope": "open-finger fixture, actual R5 grip and short motion; not full task or GPT success",
        "fixture": initial,
        "finish_requested": finish,
        "checkpoints": [],
    }

    last_time, last_assessment = None, None

    def assess():
        nonlocal last_time, last_assessment
        now = backend.time_s()
        if now != last_time:
            last_assessment = task.evaluate(
                EvaluationState({"simulation_time_s": now}), robot.feedback()
            )
            last_time = now
        return last_assessment

    def capture(label):
        from PIL import Image

        rows = task.scene.measure()
        assessment = assess()
        objects = {
            n: {k: v for k, v in row.items() if k != "corners"}
            for n, row in rows.items()
            if not n.startswith("load_")
        }
        result["checkpoints"].append(
            {
                "label": label,
                "simulation_time_s": backend.time_s(),
                "objects": objects,
                "criteria": assessment.metrics.get("criteria"),
                "assessment_status": assessment.status,
                "assessment_reason": assessment.reason,
                "observer_contact": assessment.metrics.get("observer_contact"),
                "robot": {a: asdict(s) for a, s in robot.state().items()},
                "feedback": asdict(robot.feedback()),
                "planning": robot.last_plan,
            }
        )
        for f in backend.capture_rgb():
            Image.frombytes("RGB", (f.width, f.height), f.data).save(
                output / f"{label}_{f.camera}.png"
            )
        (output / "progress.json").write_text(
            json.dumps(result, ensure_ascii=False), encoding="utf-8"
        )
        print("PROBE_CHECKPOINT", label, assessment.status, flush=True)
        if assessment.status == "invalid":
            raise RuntimeError(assessment.reason)
        return rows

    def execute(action, diagnostic_joint_goal=None):
        if diagnostic_joint_goal is None:
            robot.apply(action)
            if step_close and any(c.arm == "right" and c.kind == "gripper" and c.values[0] == .012 for c in action.commands):
                robot._advance_drive_targets(10.)  # Diagnostic old reference jump.
        else:
            # Contact isolation only: a predeclared joint-space fixture lift
            # bypasses planner clearance gates, never physical collision.
            from ...planning.trajectory import JointPath
            limits = [next(x for x in config.robot.operating_limits if x.joint == n)
                      for n in config.robot.joint_names]
            start = robot.state()["left"].joint_positions_rad
            robot.paths = {"left": JointPath.from_waypoints([start, diagnostic_joint_goal], start, limits)}
            robot.path_time_s = 0.
            robot.cartesian_targets.clear()
            robot.motion_arms = {"left"}
        completed = 0
        for tick in range(120):
            backend.step(config.control_dt_s)
            assessment = assess()
            if assessment.status in {"failed", "invalid"}:
                capture("terminal")
                raise RuntimeError(assessment.reason)
            if finish and task_id == "L2-04" and assessment.status == "succeeded":
                return
            completed = completed + 1 if robot.feedback().status == "completed" else 0
            if tick % 24 == 23:
                print("PROBE_ACTION", tick + 1, robot.feedback().status, flush=True)
            if completed >= 3:
                return
        capture("timeout")
        robot.on_action_timeout()
        raise RuntimeError("R5 action timeout: " + str(asdict(robot.feedback())))

    try:
        backend.open(config)
        result["solver_velocity_iterations"] = (
            backend.settings.solver_velocity_iterations
        )
        result["finger_collision_overrides"] = backend.environment_report.get(
            "finger_collision_overrides", []
        )
        robot.initialize(backend, config)
        task.build_scene(backend, config)
        result["scene_physics"] = backend.scene_report.get("household", {}).get("orange_physics")
        result["task_version"] = task.spec.design.version
        result["scoring_protocol"] = config.raw["task"].get("scoring_protocol")
        result["physics_dt_s"] = config.physics_dt_s
        backend.reset(config.seed)
        task.reset(backend, config.seed)
        robot.reset(config.seed)
        # Fixture assignment only, before physics advances; actual close is later.
        fixture_targets = dict(robot.targets)
        for arm in active:
            q = robot.targets[arm][:6] + width_to_joints(config.robot, 0.07)
            robot.instances[arm].set_joint_positions(
                np.asarray(q), joint_indices=robot.indices[arm]
            )
            fixture_targets[arm] = q
        robot.drive_targets = dict(fixture_targets)
        robot._send_targets(fixture_targets)
        for _ in range(12):
            backend.step(config.control_dt_s)
        before = capture("open")
        close = (
            width
            if width is not None
            else {
                "L2-01": 0.012,
                "L2-02": 0.015,
                "L2-04": 0.023,
                "L2-05": 0.035,
                "L2-03": 0.055,
            }[task_id]
        )
        result["close_width_m"] = close
        widths = [close]
        if task_id == "L2-04":
            widths = [w for w in (0.030, 0.026, 0.023, 0.021, 0.019) if w > close] + [
                close
            ]
        result["close_sequence_m"] = widths
        for opening in widths:
            execute(
                Action(
                    tuple(ArmCommand(a, "gripper", (opening,)) for a in active),
                    config.control_dt_s,
                )
            )
        capture("close_command_finished")
        motion_start = {a: robot.state()[a].tcp_pose_world_wxyz for a in active}
        # Deliberately local, not a task-solving trajectory or goal-position helper.
        delta = (
            travel * np.asarray(task.spec.parameters["outward_axis_world"])
            if task_id in {"L2-02", "L2-05"}
            else np.array([0.0, 0.0, travel])
        )
        result["requested_displacement_m"] = None if marker_pair else delta.tolist()
        motion_steps = 2 if marker_pair else max(2, int(np.ceil(travel / .02)))
        for i in range(motion_steps):
            commands = []
            for arm in active:
                pose = list(motion_start[arm])
                pose[:3] = (np.asarray(pose[:3]) + delta * ((i + 1) / motion_steps)).tolist()
                commands.append(
                    ArmCommand(arm, "cartesian_pose", tuple(pose), frame="world")
                )
            goal = None
            if marker_pair:
                lifted = [s for s in fixture["samples"] if s["arm"] == "left"][1]["joints"]
                start = np.array(initial["left"]["joints"])
                goal = start + (np.array(lifted) - start) * ((i + 1) / 2)
            execute(Action(tuple(commands), config.control_dt_s), diagnostic_joint_goal=goal)
            after = capture(f"move_{i + 1}")
            if finish and task_id == "L2-04" and assess().status == "succeeded":
                break
        if marker_pair:
            if task_id != "L2-01":
                raise ValueError("Paired cap check requires L2-01")
            from ...robots.transforms import pose_transform, origin_transform, transform_pose
            # The prop has been lifted physically by the fixture joint path.
            result["paired_lift_mode"] = "diagnostic joint path; planning clearance bypassed, physical collision retained"
            result["paired_lift_target_joints"] = list(lifted)
            capture("paired_raised")
            side = next(s for s in fixture["samples"] if s["arm"] == "right")
            cap = pose_transform(task.scene.measure()["cap"]["pose"])
            centre = (cap @ np.array([0., 0., .137, 1.]))[:3]
            for offset in (.08, .02, 0.):
                target = centre + np.array([offset, 0., 0.])
                pose = transform_pose(origin_transform(target, side["rpy"]))
                execute(Action((ArmCommand("right", "cartesian_pose", pose, frame="world"),
                                ArmCommand("right", "gripper", (.07,))), config.control_dt_s))
            capture("paired_open")
            execute(Action((ArmCommand("right", "gripper", (.012,)),), config.control_dt_s))
            pair = capture("paired_closed")
            result["paired_contact"] = {n: pair[n]["fingers"] for n in ("body", "cap")}
            result["paired_close_passed"] = bool(min(pair["body"]["fingers"]["left"]) > .05
                                                    and min(pair["cap"]["fingers"]["right"]) > .05)
        name = {
            "L2-01": "body",
            "L2-02": "middle_book",
            "L2-04": "orange",
            "L2-05": "drawer",
            "L2-03": "source_cup",
        }[task_id]
        key = "centroid" if task_id == "L2-04" else "pose"
        actual = np.asarray(after[name][key][:3]) - before[name][key][:3]
        result["measured_object_displacement_m"] = actual.tolist()
        result["passed"] = bool(np.linalg.norm(actual - delta) < 0.01)
        if marker_pair:
            result["passed"] = result.get("paired_close_passed", False)
        if finish and result["passed"] and task_id == "L2-03":
            from .pearl_pour_motion import finish_pour

            result["scope"] = "Prepositioned open-finger R5 fixture, physical grip/pour/release through production planner; no GPT or visual grasp-selection success claim"
            result["passed"] = False
            finish_pour(backend, robot, task, config, execute, capture, assess)
            result["passed"] = assess().status == "succeeded"
        if finish and task_id == "L2-04":
            result["scope"] = "Prepositioned open-finger R5 grasp and 50 mm lift qualification through the production adapter; no GPT or autonomous approach evidence"
            result["passed"] = assess().status == "succeeded"
            capture("final_held_lift")
        if finish and result["passed"] and task_id == "L2-05":
            result["passed"] = False
            execute(Action((ArmCommand("left", "gripper", (.07,)),), config.control_dt_s))
            capture("released")
            # Production also stops at the first terminal assessment. An extra
            # optional retreat after success must not invalidate a completed task.
            if assess().status != "succeeded":
                pose = list(robot.state()["left"].tcp_pose_world_wxyz)
                pose[1 if task_id == "L2-05" else 2] += .03
                execute(Action((ArmCommand("left", "cartesian_pose", tuple(pose), frame="world"),), config.control_dt_s))
                for _ in range(12):
                    backend.step(config.control_dt_s)
                    assess()
            capture("final")
            result["passed"] = assess().status == "succeeded"
        if any(
            c["assessment_status"] in {"invalid", "failed"}
            for c in result["checkpoints"]
        ):
            result["passed"] = False
    except Exception as exc:  # noqa: BLE001 - diagnostics must retain simulator/planner failures
        result["passed"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
        if task_id == "L2-04" and result["checkpoints"]:
            try:
                capture("exception_state")
            except Exception:
                pass  # Keep the original failure even if readback is invalid.
    finally:
        if hasattr(task, "evaluator"):
            result["peak_object_speeds_m_s"] = getattr(task.evaluator, "peak_speeds", {})
            result["peak_finger_force_n"] = getattr(task.evaluator, "peak_force", None)
        (output / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        robot.close()
        backend.close()
    print(
        "L2_EXECUTION_PROBE",
        task_id,
        result["passed"],
        result.get("error", ""),
        flush=True,
    )
    return result["passed"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=["L2-01", "L2-02", "L2-03", "L2-04", "L2-05"])
    p.add_argument("--poses", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--width", type=float)
    p.add_argument("--marker-pair", action="store_true")
    p.add_argument("--step-close", action="store_true", help="Diagnostic only: reproduce old direct right closure")
    p.add_argument("--finish", action="store_true", help="Qualify cup pouring, orange held lift or drawer pull/release from the explicit grip fixture, without GPT")
    p.add_argument("--physics-hz", type=int, choices=[240, 480, 960], help="Explicit local step-size diagnostic override")
    p.add_argument("--soft-physics", type=Path, help="L2-04 physics override JSON; diagnostic only")
    p.add_argument("--velocity-iterations", type=int)
    p.add_argument(
        "--travel",
        type=float,
        default=0.020,
        help="Local motion in metres, at most 0.05",
    )
    args = p.parse_args()
    return (
        0
        if check(
            args.task,
            args.poses,
            args.output,
            args.width,
            args.velocity_iterations,
            args.travel,
            args.marker_pair,
            args.step_close,
            args.finish,
            args.physics_hz,
            args.soft_physics,
        )
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
