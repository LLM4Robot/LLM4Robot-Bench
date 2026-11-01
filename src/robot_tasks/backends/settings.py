"""Explicit simulation settings, loadable without importing Isaac Sim."""

from dataclasses import dataclass, replace
import math
import os
from pathlib import Path

from ..config import TaskConfig


@dataclass(frozen=True)
class SimulationSettings:
    asset_directory: Path
    headless: bool = True
    renderer: str = "RaytracedLighting"
    active_gpu: int = 0
    physics_gpu: int = 0
    robot_gravity_compensation: bool = True
    cartesian_completion_tolerance_m: float = 0.001
    graphics_backend: str = "d3d12" if os.name == "nt" else "vulkan"
    width: int = 960
    height: int = 720
    fps: int = 12
    arm_stiffness: float = 1500.0
    arm_damping: float = 80.0
    finger_stiffness: float = 2000.0
    finger_damping: float = 40.0
    stable_gripper_drive: bool = False
    finger_max_acceleration: float = 0.15
    finger_closing_lead: float = 0.0015
    finger_effort_limit: float = 4.0
    self_collision: bool = False
    gpu_dynamics: bool = False  # Required for native volume FEM; rendering alone is not GPU physics.
    solver_position_iterations: int = 32
    solver_velocity_iterations: int = 0
    camera_position: tuple[float, ...] = (1.6, 0.0, 1.4)
    camera_target: tuple[float, ...] = (0.28, 0.0, 0.2)
    camera_reference_frame: str = "world"
    camera_resolution: tuple[int, int] | None = None
    camera_intrinsics: tuple[float, ...] | None = None
    camera_profile: str = ""  # Relative to episode config; empty keeps legacy overview.
    capture_render_passes: int = 2
    environment: str = "plain"
    robot_appearance: str = "vendor_white"

    @classmethod
    def from_config(cls, config: TaskConfig) -> "SimulationSettings":
        data = dict(config.raw.get("simulation", {}))
        if ("camera_position" in data) != ("camera_target" in data):
            raise ValueError("Set camera_position and camera_target together")
        if ("camera_resolution" in data) != ("camera_intrinsics" in data):
            raise ValueError("Set camera_resolution and camera_intrinsics together")
        if "camera_resolution" in data and ("camera_position" not in data or not data.get("camera_profile")):
            raise ValueError("Shared camera optics require camera_position/target and camera_profile")
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown simulation settings: {sorted(unknown)}")
        directory = data.pop("asset_directory", "../../build/r5")
        if not isinstance(directory, str) or not directory:
            raise ValueError("simulation.asset_directory must be a path")
        # Host-local generated USD/planner caches must not race when several
        # workers share one project checkout. Task assets remain read-only.
        directory = os.environ.get("GPT_BENCHMARK_ASSET_CACHE", directory)
        if not directory.strip():
            raise ValueError("GPT_BENCHMARK_ASSET_CACHE must not be empty")
        result = cls(asset_directory=(config.source.parent / directory).resolve(), **data)
        if result.environment not in {"plain", "lab_room_v1", "tuojing_lab_v4"}:
            raise ValueError("simulation.environment must be plain, lab_room_v1 or tuojing_lab_v4")
        if result.environment == "tuojing_lab_v4" and not result.camera_profile:
            raise ValueError("tuojing_lab_v4 requires its existing-camera profile")
        if result.robot_appearance not in {"vendor_white", "r5_black_silver_v1"}:
            raise ValueError("simulation.robot_appearance must be vendor_white or r5_black_silver_v1")
        if not isinstance(result.camera_profile, str):
            raise ValueError("simulation.camera_profile must be a path string")
        if type(result.capture_render_passes) is not int or not 1 <= result.capture_render_passes <= 8:
            raise ValueError("capture_render_passes must be in [1, 8]")
        if result.graphics_backend not in {"vulkan", "d3d12"}:
            raise ValueError("graphics_backend must be vulkan or d3d12")
        if result.renderer not in {"RaytracedLighting", "PathTracing"}:
            raise ValueError("renderer must be RaytracedLighting or PathTracing")
        for name in ("active_gpu", "physics_gpu"):
            if type(getattr(result, name)) is not int or getattr(result, name) < 0:
                raise ValueError(f"simulation.{name} must be a nonnegative integer")
        for name in ("headless", "self_collision", "gpu_dynamics", "robot_gravity_compensation"):
            if type(getattr(result, name)) is not bool:
                raise ValueError(f"simulation.{name} must be boolean")
        for name in ("width", "height", "fps"):
            value = getattr(result, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"simulation.{name} must be a positive integer")
        for name, minimum in (("solver_position_iterations", 1), ("solver_velocity_iterations", 0)):
            value = getattr(result, name)
            if type(value) is not int or not minimum <= value <= 255:
                raise ValueError(f"simulation.{name} must be an integer in [{minimum}, 255]")
        for name in ("arm_stiffness", "arm_damping", "finger_stiffness", "finger_damping", "finger_max_acceleration", "finger_closing_lead", "finger_effort_limit", "cartesian_completion_tolerance_m"):
            value = getattr(result, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"simulation.{name} must be finite and positive")
        for name in ("camera_position", "camera_target"):
            value = getattr(result, name)
            if not isinstance(value, (tuple, list)) or len(value) != 3 or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value
            ):
                raise ValueError(f"simulation.{name} must contain three finite numbers")
        if result.camera_reference_frame not in {"world", "tuojing_lab"}:
            raise ValueError("camera_reference_frame must be world or tuojing_lab")
        if "camera_resolution" in data:
            resolution, intrinsics = result.camera_resolution, result.camera_intrinsics
            if not isinstance(resolution, (tuple, list)) or len(resolution) != 2 or any(
                type(v) is not int or v <= 0 for v in resolution
            ):
                raise ValueError("camera_resolution must contain two positive integers")
            if not isinstance(intrinsics, (tuple, list)) or len(intrinsics) != 4 or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in intrinsics
            ):
                raise ValueError("camera_intrinsics must contain four finite numbers")
            fx, fy, cx, cy = intrinsics
            if min(fx, fy) <= 0 or not (0 <= cx < resolution[0] and 0 <= cy < resolution[1]):
                raise ValueError("camera_intrinsics has invalid focal lengths or principal point")
            result = replace(result, camera_resolution=tuple(resolution), camera_intrinsics=tuple(intrinsics))
        from ..cameras.overview import resolve_view
        position, target = resolve_view(result, config)
        result = replace(result, camera_position=position, camera_target=target)
        return result

    def application_config(self) -> dict:
        return {
            "headless": self.headless, "width": self.width, "height": self.height,
            "renderer": self.renderer,
            **({"samples_per_pixel_per_frame": 64, "max_bounces": 12,
                "max_specular_transmission_bounces": 12, "denoiser": True}
               if self.renderer == "PathTracing" else {}),
            "active_gpu": self.active_gpu, "physics_gpu": self.physics_gpu, "multi_gpu": False,
            # Kit runs in an isolated worker; fast shutdown avoids Windows DLL teardown crashes.
            "fast_shutdown": True,
            "extra_args": [f"--/app/vulkan={'true' if self.graphics_backend == 'vulkan' else 'false'}"],
        }
