"""Laboratory acceptance through the production backend/adapter, without GPT.

Isaac Python: python -m robot_tasks.integrationtest.diagnostics.tuojing_lab [--gui] [--hold]
Results go to build/tuojing_lab_check, never into benchmark runs or reports.
"""

import argparse
from dataclasses import asdict, replace
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from ...paths import PROJECT_ROOT


def check(output, *, gui=False, hold=False):
    import numpy as np
    from PIL import Image

    from ...backends.isaacsim import IsaacSimBackend
    from ...cameras.depth import depth_array
    from ...config import load_config
    from ...planning.scene import snapshot
    from ...robots.adapters import R5Adapter
    from ...robots.transforms import pose_transform
    from ...types import Action, ArmCommand

    config = load_config(PROJECT_ROOT / "configs/checks/tuojing_lab.toml", depth=True)
    config = replace(config, raw={**config.raw, "simulation": {**config.raw["simulation"], "headless": not gui}})
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    backend, robot = IsaacSimBackend(), R5Adapter()
    backend.run_directory = output
    result = {"passed": False, "gpt_called": False,
              "scope": "shared laboratory, real adapter/planner, RGB-D and camera attachment; no benchmark task",
              "captures": {}, "motions": []}

    def capture(label):
        frames = backend.capture_rgb()
        records, images = {}, {}
        for frame in frames:
            rgb = np.frombuffer(frame.data, dtype=np.uint8).reshape(frame.height, frame.width, 3)
            depth = depth_array(frame)
            fraction = float(np.isfinite(depth).mean())
            if rgb.std() < 1 or fraction < 0.1:
                raise RuntimeError(f"{label}/{frame.camera}: blank RGB or missing depth")
            if frame.depth.timestamp_s != frame.timestamp_s or frame.depth.capture_id != frame.calibration["capture_id"]:
                raise RuntimeError("RGB/depth capture IDs or timestamps differ")
            Image.fromarray(rgb).save(output / f"{label}_{frame.camera}.png")
            np.save(output / f"{label}_{frame.camera}_depth.npy", depth)
            records[frame.camera] = {"resolution": [frame.width, frame.height], "valid_depth_fraction": fraction,
                                     "calibration": frame.calibration}
            images[frame.camera] = rgb.copy()
        result["captures"][label] = records
        return images

    def verify_fk():
        errors = {}
        for arm, state in robot.state().items():
            actual = pose_transform(state.tcp_pose_world_wxyz)
            expected = pose_transform(robot.kinematics.forward(state.joint_positions_rad, arm))
            position = float(np.linalg.norm(expected[:3, 3] - actual[:3, 3]))
            rotation = float(np.linalg.norm(expected[:3, :3] - actual[:3, :3]))
            errors[arm] = {"position_m": position, "rotation_matrix_error": rotation}
            if position > 0.001 or rotation > 0.003:
                raise RuntimeError(f"Imported {arm} articulation does not match URDF FK: {errors[arm]}")
        return errors

    def execute(action):
        robot.apply(action)
        completed = 0
        for _ in range(config.raw["execution"]["max_action_steps"]):
            backend.step(config.control_dt_s)
            feedback = robot.feedback()
            if feedback.status in {"rejected", "error"}:
                raise RuntimeError(str(asdict(feedback)))
            completed = completed + 1 if feedback.status == "completed" else 0
            if completed >= 3:
                return asdict(feedback)
        raise RuntimeError("Motion did not settle: " + str(asdict(robot.feedback())))

    try:
        backend.open(config)
        robot.initialize(backend, config)
        backend.reset(config.seed)
        robot.reset(config.seed)
        for _ in range(24):
            backend.step(config.control_dt_s)
        from pxr import UsdPhysics

        stage = backend.world.stage
        roots = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
        scenes = [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
        if len(roots) != 2 or len(scenes) != 1 or stage.GetPrimAtPath("/World/Bench"):
            raise RuntimeError(f"Duplicate scene components: roots={roots}, physics={scenes}")
        result["articulation_roots"], result["physics_scenes"] = roots, scenes
        result["initial_fk_errors"] = verify_fk()
        scene = snapshot(backend)
        if "bench" in scene["obstacles"]["cuboid"] or "_World_Furniture_Worktable_Top" not in scene["obstacles"]["mesh"]:
            raise RuntimeError("Planner does not use the delivered tabletop")
        result["planner_environment_colliders"] = len(scene["obstacles"]["mesh"])
        initial_images = capture("initial")
        initial_poses = backend.camera_rig.world_poses()
        result["initial_camera_poses"] = initial_poses
        for arm, delta in (("left", -0.10), ("right", 0.10)):
            q = list(robot.state()[arm].joint_positions_rad)
            q[0] += delta
            feedback = execute(Action((ArmCommand(arm, "joint_position", tuple(q)),), 2.0,
                                      note="Engineering check: bounded joint motion away from the other arm"))
            images = capture(arm + "_moved")
            poses = backend.camera_rig.world_poses()
            camera = arm + "_wrist"
            displacement = float(np.linalg.norm(np.array(poses[camera][:3]) - initial_poses[camera][:3]))
            pixel_change = float(np.abs(images[camera].astype(float) - initial_images[camera]).mean())
            # The camera must preserve its complete transform relative to link6.
            ee = pose_transform(backend.read_ee_pose(arm))
            local = np.linalg.inv(ee) @ pose_transform(poses[camera])
            spec = next(s for s in backend.camera_rig.specs if s.name == camera)
            follow_error = float(np.max(np.abs(local - pose_transform(spec.optical_pose_wxyz))))
            if displacement < 0.005 or pixel_change < 0.5 or follow_error > 1e-4:
                raise RuntimeError(f"{camera} did not follow physical motion: {displacement}, {pixel_change}, {follow_error}")
            if not np.allclose(pose_transform(poses["overview"]), pose_transform(initial_poses["overview"]), atol=1e-6):
                raise RuntimeError("The fixed overview camera moved")
            result["motions"].append({"arm": arm, "feedback": feedback,
                                      "camera_displacement_m": displacement, "image_mean_absolute_change": pixel_change,
                                      "camera_attachment_error": follow_error, "fk_errors": verify_fk()})
            print(f"LAB_CHECK {arm}: motion, FK and camera follow passed", flush=True)
        result["grippers"] = execute(Action(tuple(ArmCommand(arm, "gripper", (0.06,)) for arm in config.arms), 1.0))
        for arm, state in robot.state().items():
            if abs(state.gripper_width_m - 0.06) > 0.003:
                raise RuntimeError(f"{arm}: gripper opening was not reached")
        pose = list(robot.state()["left"].tcp_pose_world_wxyz)
        pose[2] += 0.015
        result["cartesian_motion"] = execute(Action((ArmCommand("left", "cartesian_pose", tuple(pose)),), 2.0,
                                                    note="Engineering check: raise TCP 15 mm in world Z"))
        measured = robot.state()["left"].tcp_pose_world_wxyz
        error = float(np.linalg.norm(np.array(measured[:3]) - pose[:3]))
        if error > 0.003:
            raise RuntimeError(f"Cartesian target was not reached: {error} m")
        result["cartesian_position_error_m"] = error
        capture("final")
        result["scene"] = backend.scene_report
        result["passed"] = True
        print("LAB_CHECK PASS; evidence: " + str(output), flush=True)
    except Exception as exc:
        traceback.print_exc()
        result["error"] = str(exc)
    finally:
        result["status"] = "completed" if result["passed"] else "error"
        (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        try:
            if result["passed"] and hold:
                print("GUI preview; close the window or press Ctrl+C to exit.", flush=True)
                while backend.app.is_running():
                    backend.world.step(render=True)
        except KeyboardInterrupt:
            pass
        finally:
            try:
                robot.close()
            finally:
                backend.close()
    return 0 if result["passed"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--hold", action="store_true", help="Keep a successful GUI preview open")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.hold and not args.gui:
        parser.error("--hold requires --gui")
    output = (args.output or PROJECT_ROOT / "build/tuojing_lab_check" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S_%fZ")).resolve()
    if args.worker:
        return check(output, gui=args.gui, hold=args.hold)
    # Kit fast_shutdown can exit zero even after an exception. Supervise a
    # fresh output directory and decide success from the persisted assessment.
    from ..simulation_process import _monitor_worker, _stop_worker, completed_check

    output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-u", "-m", __spec__.name, "--worker", "--output", str(output)]
    command += [flag for flag, enabled in (("--gui", args.gui), ("--hold", args.hold)) if enabled]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace", cwd=PROJECT_ROOT,
                               creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
                               start_new_session=os.name != "nt")
    try:
        with (output / "worker.log").open("w", encoding="utf-8") as log:
            returncode = _monitor_worker(process, log, None)
        return 0 if completed_check(output, returncode) else 2
    except KeyboardInterrupt:
        _stop_worker(process)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
