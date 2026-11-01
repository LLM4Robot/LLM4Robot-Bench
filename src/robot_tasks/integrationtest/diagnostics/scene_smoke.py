"""Manual Isaac scene/startup check. Never loads an API key or calls GPT.

Run in the Isaac environment: python -m robot_tasks.integrationtest.diagnostics.scene_smoke --all
Each scene uses a fresh process; videos/model success rates are separate experiments.
"""

import argparse
import json
import os
import subprocess
import sys
import traceback
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from ...paths import PROJECT_ROOT


def probe_planner(backend, robot):
    """Exercise the normal planner from measured state without executing a path."""
    from ...planning.scene import snapshot
    from ...types import Action, ArmCommand

    state = robot.state()
    commands = []
    for arm in robot.config.command_arms:
        goal = list(state[arm].tcp_pose_world_wxyz)
        goal[2] += .01
        commands.append(ArmCommand(arm, "cartesian_pose", tuple(goal)))
    result = {
        "scope": "Normal cuRobo plans a 10 mm upward TCP movement for all permitted arms; no trajectory execution or task-success claim",
        "no_gpt": True, "passed": False,
        "commands": [{"arm": c.arm, "pose": list(c.values)} for c in commands],
    }
    try:
        reply = robot.planner.plan(Action(tuple(commands), .5), state, snapshot(backend))
        if set(reply["paths"]) != {c.arm for c in commands}:
            raise ValueError("Planner did not return every requested arm")
        result.update(passed=True,
            paths={arm: {"points": len(path["positions"]), "duration_s": path["times_s"][-1]}
                   for arm, path in reply["paths"].items()},
            diagnostics=reply.get("diagnostics", {}))
    except Exception as exc:
        result.update(error=str(exc), diagnostic=getattr(exc, "diagnostic", None))
    return result


def check(task_id, output, *, prepare=True, prop_probe=False, capture_hook=None,
          capture_before_observations=False, depth=None, plan_probe=False, config_path=None):
    import numpy as np
    from PIL import Image

    from ...backends.isaacsim import IsaacSimBackend
    from ...registry import resolve_task
    from ...robots.adapters import R5Adapter
    from ...robots.preparation import preparation_action
    from ...types import EvaluationState

    if plan_probe and not prepare:
        raise ValueError("A planner probe requires normal robot preparation")
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    output.mkdir(parents=True, exist_ok=True)
    spec, config = resolve_task(task_id, config_path=config_path, depth=depth)
    config = spec.factory.prepare_config(config)
    backend, robot, task = IsaacSimBackend(), R5Adapter(), spec.factory()
    backend.run_directory = output
    result = {
        "task": task_id,
        "passed": False,
        "gpt_called": False,
        "scope": "scene, physical reset, camera input and initial scoring; not task-solving evidence",
        "observation_modalities": ["rgb", "depth"] if config.include_depth else ["rgb"],
        "preparation_requested": prepare,
        "initial_layout": config.raw["task"].get("initial_layout"),
    }
    from ...video_delivery import enabled, scenes
    if enabled(config):
        result['runtime_scene_sha256'] = scenes()[config.scene_id]['runtime_sha256']
        from ...video_delivery import setup_digest, visual_setup
        result['visual_setup_sha256'] = setup_digest()
        result['visual_setup_revision'] = visual_setup()['revision']
    try:
        backend.open(config)
        if prepare:
            robot.initialize(backend, config)
        else:
            # Physical reset/camera checks do not need a CUDA planning worker.
            from ...config import ExecutionSettings
            from ...robots.kinematics import R5Kinematics

            robot.config, robot.backend = config, backend
            robot.execution = ExecutionSettings.from_config(config)
            robot.kinematics = R5Kinematics(config.robot)
            robot.instances = backend.add_r5(config.robot)
        task.build_scene(backend, config)
        backend.reset(config.seed)
        task.reset(backend, config.seed)
        robot.reset(config.seed)
        for _ in range(24):
            backend.step(config.control_dt_s)
        from .robot_reset_audit import capture as capture_robot_reset
        result["robot_reset_audit"] = capture_robot_reset(backend, robot)
        if not result["robot_reset_audit"]["passed"]:
            raise RuntimeError("Robot reset differs from its configured base/joints or URDF FK")
        if prepare:
            robot.apply(
                getattr(spec.factory, "preparation_action", preparation_action)(config)
            )
            completed = 0
            for _ in range(120):
                backend.step(config.control_dt_s)
                completed = (
                    completed + 1 if robot.feedback().status == "completed" else 0
                )
                if completed >= 3:
                    break
            if completed < 3:
                raise RuntimeError(
                    "Startup preparation failed: " + str(asdict(robot.feedback()))
                )
        rows = task.scene.measure()
        from ...tasks.bench.english_reference import initial_state_issue
        issue = initial_state_issue(task.scene.spec, rows)
        if issue:
            raise RuntimeError(issue)
        if plan_probe:
            result["planner_probe"] = probe_planner(backend, robot)
            if not result["planner_probe"]["passed"]:
                raise RuntimeError("Normal planner request failed: " + result["planner_probe"]["error"])
        # Finish documentation render warm-up before saving the observation views;
        # otherwise newly loaded robot meshes can be absent in the first capture.
        if capture_hook is not None and capture_before_observations:
            result["preview"] = capture_hook(backend, task, output)
        frames = backend.capture_rgb()
        result["cameras"] = []
        result["camera_capture"] = {}
        from ...backends.scene_geometry import bench_geometry
        from ...policies.gpt import prepare_frame
        from ...policies.operating_notes import operating_notes

        result["operating_notes"] = operating_notes(config)
        state = robot.state()
        for frame in frames:
            from ...cameras.depth import depth_array

            depth_values = depth_array(frame) if config.include_depth else None
            if depth_values is not None and not np.any(np.isfinite(depth_values) & (depth_values > 0)):
                raise RuntimeError("Empty depth: " + frame.camera)
            if not config.include_depth and frame.depth is not None:
                raise RuntimeError("Depth captured while disabled: " + frame.camera)
            Image.frombytes("RGB", (frame.width, frame.height), frame.data).save(
                output / (frame.camera + ".png")
            )
            pixels = np.frombuffer(frame.data, dtype=np.uint8).reshape(-1, 3)
            lit_fraction = float(np.mean(pixels.mean(axis=1) > 15))
            if pixels.std() < 1 or lit_fraction < .05:
                raise RuntimeError("Blank camera: " + frame.camera)
            result["cameras"].append(frame.camera)
            # Use the same overlay and calibration resizing as the policy, but
            # never instantiate a model client or send a request in this check.
            preview, size, calibration = prepare_frame(frame, 768, robot=state, bench=bench_geometry(config))
            (output / ("model_preview_" + frame.camera + ".png")).write_bytes(preview)
            if depth_values is not None:
                np.save(output / (frame.camera + "_depth.npy"), depth_values)
            result["camera_capture"][frame.camera] = {
                "lit_pixel_fraction": lit_fraction,
                "calibration": frame.calibration, "preview_calibration": calibration,
                "preview_resolution": size,
                "valid_depth_fraction": (float(np.mean(np.isfinite(depth_values) & (depth_values > 0)))
                                         if depth_values is not None else None),
            }
        if capture_hook is not None and not capture_before_observations:
            result["preview"] = capture_hook(backend, task, output)
        # Initial state must be running, never already successful or physically failed.
        assessments = []
        for _ in range(14):
            backend.step(config.control_dt_s)
            assessment = task.evaluate(
                EvaluationState(
                    {"simulation_time_s": backend.time_s(), "robot": robot.state()}
                ),
                robot.feedback(),
            )
            assessments.append(assessment.status)
        if any(s != "running" for s in assessments):
            raise RuntimeError(
                "Invalid initial task assessment: " + str(asdict(assessment))
            )
        monitor = getattr(backend, "collision_monitor", None)
        if monitor is not None:
            result["initial_rigid_collision"] = {
                "blocked": monitor.blocked, "peak_contact": monitor.peak,
                "physics_steps": monitor.steps, "events": monitor.events}
            if monitor.blocked is not None:
                raise RuntimeError("Unexpected arm/attachment contact in initial scene: " + str(monitor.blocked))
        result.update(
            passed=True,
            initial_assessment=asdict(assessment),
            objects_before=rows,
            scene=backend.scene_report,
            prepared=prepare,
        )
        if prop_probe:
            from .prop_physics import run

            result["prop_probe"] = run(task.scene)
            for frame in backend.capture_rgb():
                Image.frombytes("RGB", (frame.width, frame.height), frame.data).save(
                    output / ("prop_probe_" + frame.camera + ".png")
                )
    except Exception as exc:  # noqa: BLE001 - a diagnostic must persist every worker failure
        traceback.print_exc()
        result["passed"] = False
        result["error"] = str(exc)
    finally:
        try:
            (output / "result.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
                encoding="utf-8",
            )
            print(
                "SCENE_SMOKE",
                task_id,
                "PASS" if result["passed"] else "FAIL",
                flush=True,
            )
        finally:
            try:
                robot.close()
            finally:
                backend.close()
    return 0 if result["passed"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", nargs="?")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--no-preparation", action="store_true")
    parser.add_argument("--prop-probe", action="store_true")
    parser.add_argument("--plan-probe", action="store_true",
                        help="Request a small real cuRobo plan without executing it or calling a model")
    parser.add_argument("--depth", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.plan_probe and args.no_preparation:
        parser.error("--plan-probe requires normal preparation")
    root = args.output or PROJECT_ROOT / "runs" / (
        "scene_smoke_" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    )
    root.mkdir(parents=True, exist_ok=True)
    if args.all:
        from ...registry import TASKS

        results = []
        for spec in TASKS.values():
            if not spec.task_id.startswith("L") or not spec.implemented or spec.base_task_id is not None:
                continue
            command = [
                sys.executable,
                "-m",
                __package__ + ".scene_smoke",
                spec.task_id,
                "--output",
                str(root / spec.task_id),
            ]
            if args.no_preparation:
                command.append("--no-preparation")
            if args.prop_probe:
                command.append("--prop-probe")
            if args.plan_probe:
                command.append("--plan-probe")
            if args.depth is not None:
                command.append("--depth" if args.depth else "--no-depth")
            with (root / (spec.task_id + ".log")).open("w", encoding="utf-8") as log:
                process = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=PROJECT_ROOT,
                    check=False,
                )
            result_path = root / spec.task_id / "result.json"
            result = (
                json.loads(result_path.read_text(encoding="utf-8"))
                if result_path.exists()
                else {"error": "worker did not produce result"}
            )
            entry = {
                "task": spec.task_id,
                "passed": result.get("passed", False),
                "returncode": process.returncode,
                "error": result.get("error"),
            }
            results.append(entry)
            (root / "summary.json").write_text(
                json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(json.dumps(entry, ensure_ascii=False), flush=True)
        return 0 if all(r["passed"] and r["returncode"] == 0 for r in results) else 2
    if not args.task:
        parser.error("Provide a task ID or --all")
    return check(
        args.task, root, prepare=not args.no_preparation, prop_probe=args.prop_probe,
        depth=args.depth, plan_probe=args.plan_probe
    )


if __name__ == "__main__":
    raise SystemExit(main())
