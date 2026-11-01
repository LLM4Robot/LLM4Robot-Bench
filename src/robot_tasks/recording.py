"""Persistent episode evidence. No GIF/video is fabricated by the scaffold."""

import hashlib
import json
import platform
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .config import TaskConfig
from .runtime import snapshot_runtime_config, snapshot_storage_config
from .types import Observation


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def finish_run(directory: Path, result: dict) -> None:
    """Flush the result and its readable report before native process shutdown."""
    from .reporting import write_run_report
    write_json(directory / "result.json", result)
    write_run_report(directory)


class EpisodeRecorder:
    def __init__(self, root: Path, config: TaskConfig, mode: str, step_limit: int,
                 *, run_directory: Path | None = None):
        name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.directory = run_directory or root / f"{config.experiment_id}_{mode}_{name}_{uuid4().hex[:8]}"
        self.directory.mkdir(parents=True, exist_ok=run_directory is not None)
        if any(p.name != "process.log" for p in self.directory.iterdir()):
            raise ValueError("Episode output must be a fresh directory")
        # Both direct task runs and engineering checks use this recorder.
        # Save before Kit starts, since fast shutdown can end the worker process.
        snapshot_runtime_config(self.directory)
        snapshot_storage_config(self.directory)
        self.save_rgb = config.save_rgb
        self.require_depth = config.include_depth and mode != "dry_run"
        self.save_depth = config.save_depth and mode != "dry_run"
        config_bytes = config.source.read_bytes()
        (self.directory / "config.toml").write_bytes(config_bytes)
        # The task file alone no longer describes the effective workflow.
        write_json(self.directory / "effective_config.json", config.raw)
        if config.raw["task"].get("initial_layout"):
            from .initial_layout import layout
            write_json(self.directory / "initial_layout.json", layout(config.raw["task"]))
        if config.workflow_settings is not None:
            write_json(self.directory / "workflow_profile.json", config.workflow_settings)
        robot_profile_bytes = config.robot.source.read_bytes()
        (self.directory / "robot_profile.toml").write_bytes(robot_profile_bytes)
        source_lock_bytes = config.robot.asset_lock.read_bytes()
        (self.directory / "asset_source.lock.json").write_bytes(source_lock_bytes)
        camera_profile = config.raw.get("simulation", {}).get("camera_profile")
        if camera_profile:
            (self.directory / "camera_profile.toml").write_bytes((config.source.parent / camera_profile).read_bytes())
        package = Path(__file__).parent
        sources = {
            str(p.relative_to(package)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(package.rglob("*.py"))
        }
        write_json(self.directory / "manifest.json", {
            "task_id": config.experiment_id, "mode": mode,
            **({"base_task_id": config.task_id} if config.experiment_id != config.task_id else {}),
            "evidence_type": "framework_dry_run" if mode == "dry_run" else "simulation_attempt",
            "mock": mode == "dry_run", "seed": config.seed, "step_limit": step_limit,
            "config_path": str(config.source), "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "workflow_profile_path": str(config.workflow_source) if config.workflow_source else None,
            "effective_config_sha256": hashlib.sha256((self.directory / "effective_config.json").read_bytes()).hexdigest(),
            "robot_asset": str(config.asset_path) if config.asset_path else None,
            "robot_profile_path": str(config.robot.source),
            "robot_profile_sha256": hashlib.sha256(robot_profile_bytes).hexdigest(),
            "asset_source_lock_sha256": hashlib.sha256(source_lock_bytes).hexdigest(),
            "available_arms": config.robot.available_arms, "active_arms": config.arms,
            "command_arms": config.command_arms, "arm_roles": config.arm_roles,
            "python": platform.python_version(), "platform": platform.platform(),
            "source_sha256": sources,
            "units": {"distance": "m", "angle": "rad", "time": "s", "quaternion": "wxyz"},
            "video_status": "pending",
            "instruction": config.instruction,
            "grasp_check_mode": config.grasp_check_mode,
            "initial_state_description": config.initial_state_description,
            "physics_during_inference": "paused",
            "observation_modalities": ["rgb", "depth"] if config.include_depth else ["rgb"],
            "depth": {"enabled": config.include_depth, "recorded": self.save_depth,
                      "source": "ideal_simulation_render" if self.require_depth else None,
                      "quantity": "optical_axis_z", "units": "m", "hardware_emulated": False,
                      "recording_cadence": "decision observations and reset/final observations; not every physics tick",
                      "queries_share_model_budget": config.include_depth},
            "execution": config.raw.get("execution", {}),
            "episode_isolation": "fresh scene/robot/task and policy history at run start; unique output; no previous runs loaded",
            "retry_protocol": "no automatic retry",
        })
        (self.directory / "instruction.txt").write_text(config.instruction, encoding="utf-8")

    def observation(self, observation: Observation, step: int, *, after_reset: bool = False,
                    reference_arm: str | None = None) -> dict:
        if reference_arm is not None and not reference_arm.replace("_", "").isalnum():
            raise ValueError("Invalid reference arm name")
        suffix = f"_grasp_{reference_arm}" if reference_arm else ("_reset" if after_reset else "")
        frames = []
        for frame in observation.rgb:
            if len(frame.data) != frame.width * frame.height * 3 or min(frame.width, frame.height) <= 0:
                raise ValueError("Invalid packed RGB8 frame")
            saved_path = None
            if self.save_rgb:
                # Camera names are sanitized independently of adapter inputs.
                if not frame.camera.replace("_", "").isalnum():
                    raise ValueError("Invalid camera name")
                # Preserve the failed decision's images when its reset is logged.
                relative = Path("frames") / frame.camera / f"{step:06d}{suffix}.ppm"
                target = self.directory / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                header = f"P6\n{frame.width} {frame.height}\n255\n".encode("ascii")
                target.write_bytes(header + frame.data)
                saved_path = relative.as_posix()
            frames.append({"camera": frame.camera, "time_s": frame.timestamp_s,
                           "width": frame.width, "height": frame.height, "path": saved_path,
                           "calibration": frame.calibration})
            if self.require_depth:
                from .cameras.depth import depth_array, depth_metadata, encode_depth_preview
                metadata = depth_metadata(frame)
                metadata.update(path=None, preview_path=None)
                if self.save_depth:
                    import numpy as np
                    if not frame.camera.replace("_", "").isalnum():
                        raise ValueError("Invalid camera name")
                    relative = Path("depth") / frame.camera / f"{step:06d}{suffix}.npy"
                    target = self.directory / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    np.save(target, depth_array(frame), allow_pickle=False)
                    preview, display = encode_depth_preview(frame, (frame.width, frame.height))
                    target.with_suffix(".png").write_bytes(preview)
                    metadata.update(path=relative.as_posix(), preview_path=relative.with_suffix(".png").as_posix(),
                                    preview=display)
                frames[-1]["depth"] = metadata
        return {"time_s": observation.time_s, "rgb": frames,
                "robot": {k: asdict(v) for k, v in observation.robot.items()},
                "feedback": asdict(observation.feedback)}

    def event(self, value: dict) -> None:
        exchange = value.get("exchange", {})
        if value.get("event") == "model_exchange" and exchange.get("provider_diagnostic"):
            directory = self.directory / "model_errors"
            directory.mkdir(exist_ok=True)
            write_json(directory / f"request_{exchange['call']:04d}.json", exchange["provider_diagnostic"])
        with (self.directory / "episode.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")

    def finish(self, result: dict) -> None:
        write_json(self.directory / "result.json", result)
        manifest_path = self.directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["video_status"] = result["video_status"]
        write_json(manifest_path, manifest)
        from .reporting import write_run_report
        write_run_report(self.directory)
