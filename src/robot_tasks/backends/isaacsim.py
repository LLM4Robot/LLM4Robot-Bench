"""Isaac Sim lifecycle; importing this module never starts Kit."""

import math
import time
from pathlib import Path

from ..config import TaskConfig
from ..types import EvaluationState, RGBFrame
from ..cameras import RGBNotReady
from ..runtime import configure_windows_storage, storage_report
from .settings import SimulationSettings
from .scene_geometry import BENCH_POSITION, BENCH_SIZE


class IsaacSimBackend:
    def __init__(self) -> None:
        self.app = self.world = self.settings = None
        self._rgb = self._render_product = self._rep = None
        self._physics_dt = None
        self.articulations = {}
        self.ee_views = {}
        self.scene_report = {}
        self.import_report = {}
        self.runtime_storage = None
        self.camera_profile = self.camera_rig = None
        self.camera_names = ()
        self._capture_id = 0
        self.include_depth = False
        self.before_physics_step = None
        self.environment_report = {"profile": "plain"}
        self.lab_asset = None
        self.planning_environment_paths = ()
        self.planning_legacy_bench = True
        self.wrist_geometry = {}
        self.collision_monitor = None

    def open(self, config: TaskConfig) -> None:
        if self.app is not None:
            raise RuntimeError("Isaac Sim backend is already open")
        self.allow_actor_recovery = config.task_id.startswith("L1-")
        self.settings = SimulationSettings.from_config(config)
        from ..video_delivery import enabled
        self.video_delivery = enabled(config)
        if self.video_delivery:
            from ..video_delivery import scenes
            if scenes()[config.scene_id].get('sdf_colliders') and not self.settings.gpu_dynamics:
                raise ValueError('Delivered SDF colliders require simulation.gpu_dynamics = true')
        if self.settings.environment == "tuojing_lab_v4":
            if self.video_delivery:
                from ..video_delivery import verified_scene
                self.lab_asset = verified_scene(config.scene_id)
            else:
                from .tuojing_lab import require_lab_assets
                self.lab_asset = require_lab_assets()
        self.environment_report = {"profile": self.settings.environment}
        self.camera_names = config.cameras
        self.include_depth = config.include_depth
        if self.include_depth and not self.settings.camera_profile:
            raise ValueError("Depth capture requires a camera rig profile")
        if self.settings.camera_profile:
            from ..cameras.profile import load_camera_rig
            self.camera_profile = load_camera_rig(config.source.parent / self.settings.camera_profile)
            if self.video_delivery:
                from ..video_delivery import camera_profile
                self.camera_profile = camera_profile(self.camera_profile, config.scene_id)
            elif "camera_position" in config.raw.get("simulation", {}):
                from ..cameras.overview import override_overview
                self.camera_profile = override_overview(self.camera_profile, self.settings)
            self.camera_profile.select(config.cameras)
        elif config.cameras != ("overview",):
            raise ValueError("Multiple cameras require simulation.camera_profile")
        self.runtime_storage = configure_windows_storage()
        self._physics_dt = config.physics_dt_s
        try:
            from isaacsim import SimulationApp
        except ImportError as exc:
            raise RuntimeError("Run simulation with the existing Isaac Sim Conda environment") from exc
        application_config = self.settings.application_config()
        if self.runtime_storage is not None:
            application_config["extra_args"].extend(self.runtime_storage.kit_arguments())
        self.app = SimulationApp(application_config)
        try:
            import numpy as np
            from isaacsim.core.api import World
            from isaacsim.core.api.objects import FixedCuboid
            from pxr import PhysxSchema, UsdLux

            world_options = {}
            if self.lab_asset is not None:
                import omni.usd
                from isaacsim.core.utils.stage import open_stage
                from .tuojing_lab import configure_stage, validate_robot_binding

                if self.video_delivery:
                    if not open_stage(str(self.lab_asset)):
                        raise RuntimeError(f"Could not open latest scene: {self.lab_asset}")
                    stage = omni.usd.get_context().get_stage()
                    from ..video_delivery import configure_stage as configure_video_stage
                    self.environment_report = configure_video_stage(stage, config.scene_id)
                    from ..initial_layout import apply_stage as apply_initial_layout
                    placement = apply_initial_layout(stage, config.raw["task"])
                    if placement is not None:
                        self.environment_report["initial_layout"] = placement
                else:
                    if not open_stage(str(self.lab_asset)):
                        raise RuntimeError(f"Could not open laboratory: {self.lab_asset}")
                    stage = omni.usd.get_context().get_stage()
                    self.environment_report = configure_stage(stage, self.camera_profile)
                from ..robots.wrist_collision import configure_wrist_collisions
                self.wrist_geometry, wrist_report = configure_wrist_collisions(
                    stage, config.robot, expected_shapes=21 if self.video_delivery else 5)
                self.environment_report["wrist_collision"] = wrist_report
                if config.task_id.startswith(("L2-", "L3-")):
                    from .tuojing_lab import apply_l2_finger_collisions
                    self.environment_report["finger_collision_overrides"] = apply_l2_finger_collisions(stage, config.robot)
                elif config.task_id.startswith("L1-"):
                    # Thin-object tuning is specific to L1. Keep the existing
                    # L2/L3 overrides intact.
                    from ..robots.finger_collision import configure_finger_collisions
                    self.environment_report["finger_collision_by_arm"] = {
                        mount.name: configure_finger_collisions(stage, mount.prim_path,
                            torsional_patch_radius=.004 if config.task_id == 'L1-05' else 0.)
                        for mount in config.robot.mounts
                    }
                validate_robot_binding(stage, config.robot)
                if not self.camera_profile or any(not s.existing_prim_path for s in self.camera_profile.select(config.cameras)):
                    raise ValueError("Laboratory cameras must bind to the supplied USD cameras")
                self.planning_environment_paths = self.environment_report["planning_environment_paths"]
                self.planning_legacy_bench = False
                world_options["physics_prim_path"] = "/World/PhysicsScene"
            # Import Replicator after opening the final stage. Importing it on
            # the temporary startup stage can leave stale OmniGraph handles.
            import omni.replicator.core as rep
            self.world = World(
                stage_units_in_meters=1.0, physics_dt=config.physics_dt_s,
                rendering_dt=1 / self.settings.fps, backend="numpy", device="cpu",
                sim_params={"use_gpu_pipeline": False, "use_fabric": False,
                            "enable_scene_query_support": True},
                **world_options,
            )
            physics = self.world.get_physics_context()
            physics.enable_gpu_dynamics(self.settings.gpu_dynamics)
            if self.settings.gpu_dynamics:
                import carb

                physics.set_broadphase_type("GPU")
                carb.settings.get_settings().set_bool("/physics/updateToUsd", True)
                carb.settings.get_settings().set_bool("/physics/updateVelocitiesToUsd", True)
                carb.settings.get_settings().set_bool("/physics/suppressReadback", False)
            physics.set_solver_type("TGS")
            # Permit zero TGS velocity iterations; record actual per-arm values below.
            scene_api = PhysxSchema.PhysxSceneAPI.Apply(physics.get_current_physics_scene_prim())
            scene_api.CreateMinVelocityIterationCountAttr(0)
            if self.lab_asset is None:
                self.world.scene.add(FixedCuboid(
                    prim_path="/World/Bench", name="bench", position=np.array(BENCH_POSITION),
                    scale=np.array(BENCH_SIZE), size=1.0, color=np.array([0.24, 0.29, 0.34]),
                ))
                UsdLux.DomeLight.Define(self.world.stage, "/World/Light").CreateIntensityAttr(1100)
            if self.settings.environment == "lab_room_v1":
                from .lab_room import build_lab_room
                self.environment_report = build_lab_room(self.world.stage)
            if self.camera_profile is None:
                camera = rep.create.camera(
                    name="overview", position=self.settings.camera_position,
                    look_at=self.settings.camera_target, clipping_range=(0.005, 100),
                )
                self._render_product = rep.create.render_product(camera, (self.settings.width, self.settings.height))
                self._rgb = rep.AnnotatorRegistry.get_annotator("rgb")
                self._rgb.attach(self._render_product)
            rep.orchestrator.set_capture_on_play(False)
            self._rep = rep
            if not self.settings.headless and self.camera_profile is None:
                from isaacsim.core.utils.viewports import set_camera_view
                set_camera_view(eye=np.array(self.settings.camera_position), target=np.array(self.settings.camera_target))
        except Exception:
            # Kit fast shutdown may terminate before a caller can save errors.
            # Persist initialization failures before closing the application.
            import json
            import traceback
            error = traceback.format_exc()
            print(error, flush=True)
            directory = getattr(self, "run_directory", None)
            if directory is not None:
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "initialization_error.json").write_text(
                    json.dumps({"passed": False, "error": error}), encoding="utf-8")
            self.close()
            raise

    def add_r5(self, profile) -> dict:
        """Create both platform arms before reset initializes the physics handles."""
        import numpy as np
        from isaacsim.core.api.objects import FixedCuboid
        from isaacsim.core.prims import SingleArticulation, SingleRigidPrim
        from isaacsim.core.utils.stage import add_reference_to_stage
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics
        from ..robots.importer import convert_r5

        if self.world is None or self.articulations:
            raise RuntimeError("R5 loading requires an open, empty backend")
        if self.lab_asset is None:
            usd_path, self.import_report = convert_r5(profile, self.settings)
        else:
            usd_path = self.lab_asset
            self.import_report = {"source": str(usd_path), "binding": "existing_articulations",
                                  "source_lock": "assets/tuojing/video_20261015.json" if self.video_delivery else "assets/tuojing/source.lock.json"}
        self.scene_report = {"asset": str(usd_path), "physics_device": "cuda" if self.settings.gpu_dynamics else "cpu", "arms": {},
                             "environment": self.environment_report,
                             "robot_appearance": ("complete_video_20261015" if self.video_delivery else "delivered_tuojing_lab_v4") if self.lab_asset else self.settings.robot_appearance,
                             "storage": storage_report(self.runtime_storage, kit_started=True),
                             "self_collision": self.settings.self_collision,
                             "drive_type": "force", "gravity_m_s2": 9.81,
                             "solver_type": "TGS",
                             "solver_position_iterations": self.settings.solver_position_iterations,
                             "solver_velocity_iterations": self.settings.solver_velocity_iterations,
                             "collision_approximation": "delivered_USD; see per-arm approximations" if self.lab_asset else "convex_decomposition"}
        for mount in profile.mounts:
            position, quaternion = mount.base_pose_wxyz[:3], mount.base_pose_wxyz[3:]
            if self.lab_asset is None:
                root = add_reference_to_stage(str(usd_path), mount.prim_path)
                transform = UsdGeom.Xformable(root)
                transform.ClearXformOpOrder()
                transform.AddTranslateOp().Set(Gf.Vec3d(*position))
                transform.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
                    Gf.Quatd(quaternion[0], Gf.Vec3d(*quaternion[1:])))
            else:
                root = self.world.stage.GetPrimAtPath(mount.prim_path)
            finger_collision = self.environment_report.get("finger_collision_by_arm", {}).get(mount.name)
            descendants = list(Usd.PrimRange(root, Usd.TraverseInstanceProxies()))
            roots = [prim for prim in descendants if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
            if len(roots) != 1:
                raise RuntimeError(f"{mount.name}: expected one articulation root, found {len(roots)}")
            # A joint anchored to the world must follow the configured base pose.
            for prim in descendants:
                if prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.PrismaticJoint):
                    # Physical PD effort gains must not inherit an acceleration-drive default.
                    drive_kind = "angular" if prim.IsA(UsdPhysics.RevoluteJoint) else "linear"
                    UsdPhysics.DriveAPI.Apply(prim, drive_kind).CreateTypeAttr("force")
                    if self.settings.stable_gripper_drive and prim.GetName() in profile.gripper.joint_names:
                        limit = next(v for v in profile.operating_limits if v.joint == prim.GetName())
                        joint = UsdPhysics.PrismaticJoint(prim)
                        joint.CreateLowerLimitAttr(limit.lower)
                        joint.CreateUpperLimitAttr(limit.upper)
                        PhysxSchema.PhysxJointAPI.Apply(prim).CreateMaxJointVelocityAttr(limit.velocity)
                if prim.IsA(UsdPhysics.FixedJoint):
                    joint = UsdPhysics.FixedJoint(prim)
                    if not joint.GetBody0Rel().GetTargets() and joint.GetBody1Rel().GetTargets():
                        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*position))
                        joint.CreateLocalRot0Attr().Set(Gf.Quatf(quaternion[0], Gf.Vec3f(*quaternion[1:])))
            api = PhysxSchema.PhysxArticulationAPI.Apply(roots[0])
            api.CreateEnabledSelfCollisionsAttr(self.settings.self_collision)
            api.CreateSolverPositionIterationCountAttr(self.settings.solver_position_iterations)
            api.CreateSolverVelocityIterationCountAttr(self.settings.solver_velocity_iterations)
            instance = SingleArticulation(str(roots[0].GetPath()), name=f"r5_{mount.name}")
            self.world.scene.add(instance)
            self.articulations[mount.name] = instance
            tips = [prim for prim in descendants if prim.GetName() == profile.ee_link
                    and prim.HasAPI(UsdPhysics.RigidBodyAPI)]
            if len(tips) != 1:
                raise RuntimeError(f"{mount.name}: expected one rigid ee_link: {profile.ee_link}")
            tip_path = str(tips[0].GetPath())
            # Read-only physics views; do not add them to Scene's reset list or set link poses.
            self.ee_views[mount.name] = SingleRigidPrim(
                tip_path, name=f"r5_{mount.name}_ee", reset_xform_properties=False)
            if self.lab_asset is None:
                self.world.scene.add(FixedCuboid(
                    prim_path=f"/World/Stand_{mount.name}", name=f"stand_{mount.name}",
                    position=np.array([position[0], position[1], position[2] / 2]),
                    scale=np.array([0.16, 0.18, position[2]]), size=1.0,
                    color=np.array([0.16, 0.46, 0.70] if mount.name == "left" else [0.8, 0.38, 0.16]),
                ))
            colliders = [str(prim.GetPath()) for prim in descendants if prim.HasAPI(UsdPhysics.CollisionAPI)]
            if not colliders:
                raise RuntimeError(f"{mount.name}: imported robot contains no collision geometry")
            self.scene_report["arms"][mount.name] = {
                "prim_path": mount.prim_path, "articulation_root": str(roots[0].GetPath()),
                "collider_count": len(colliders), "colliders": colliders,
                "collision_approximations": sorted({
                    str(UsdPhysics.MeshCollisionAPI(p).GetApproximationAttr().Get())
                    for p in descendants if p.HasAPI(UsdPhysics.CollisionAPI)
                }),
                "configured_base_pose_wxyz": mount.base_pose_wxyz,
                "ee_link_path": tip_path, "tcp_pose_in_ee_wxyz": profile.tcp_pose_wxyz,
                "finger_collision": finger_collision,
            }
            if self.lab_asset is None and self.settings.robot_appearance == "r5_black_silver_v1":
                from .appearance import apply_r5_appearance
                self.scene_report["arms"][mount.name]["appearance"] = apply_r5_appearance(
                    self.world.stage, mount.prim_path)
        if self.camera_profile is not None:
            if getattr(self, "video_delivery", False):
                from ..video_delivery import reconcile_render_graph
                self.scene_report["saved_scene_render_graph_reconciliation"] = reconcile_render_graph()
            from ..cameras.isaac_rig import IsaacCameraRig
            self.camera_rig = IsaacCameraRig(self.world.stage, self._rep, self.camera_profile, self.camera_names,
                                             include_depth=self.include_depth)
            self.scene_report["camera_rig"] = self.camera_rig.create(
                {name: arm["ee_link_path"] for name, arm in self.scene_report["arms"].items()})
            if not self.settings.headless:
                from .viewport import initialize_overview_view

                self.scene_report["gui_viewport"] = initialize_overview_view(
                    self.world.stage, self.camera_rig.paths["overview"])
        if self.lab_asset is not None:
            from ..robots.collision_feedback import CollisionMonitor
            self.collision_monitor = CollisionMonitor(self.world.stage, profile,
                allow_actor_recovery=self.allow_actor_recovery)
            self.scene_report["collision_feedback"] = self.collision_monitor.metadata()
        return dict(self.articulations)

    def reset(self, seed: int) -> None:
        if self.world is None:
            raise RuntimeError("Backend is not open")
        import numpy as np
        np.random.seed(seed)
        if self.collision_monitor is not None:
            self.collision_monitor.reset()
        self.world.reset()
        for view in self.ee_views.values():
            view.initialize()
        for name, arm in self.articulations.items():
            self.scene_report["arms"][name]["actual_solver_iterations"] = {
                "position": int(arm.get_solver_position_iteration_count()),
                "velocity": int(arm.get_solver_velocity_iteration_count()),
            }

    def read_ee_pose(self, arm: str) -> tuple[float, ...]:
        """Actual rigid-link pose from PhysX, independent of our FK implementation."""
        position, quaternion = self.ee_views[arm].get_world_pose()
        return tuple(float(v) for v in (*position, *quaternion))

    def step(self, duration_s: float) -> None:
        if self.world is None or self._physics_dt is None:
            raise RuntimeError("Backend is not open")
        if isinstance(duration_s, bool) or not math.isfinite(duration_s) or duration_s <= 0:
            raise ValueError("Simulation duration must be finite and positive")
        count = round(duration_s / self._physics_dt)
        if count < 1 or not math.isclose(count * self._physics_dt, duration_s, abs_tol=1e-9):
            raise ValueError("Duration must be an integer multiple of physics_dt_s")
        if not self.app.is_running():
            raise KeyboardInterrupt("Simulation window was closed")
        for _ in range(count):
            self._wait_until_playing()
            before = self.time_s()
            if self.before_physics_step is not None:
                self.before_physics_step(self._physics_dt)
            self.world.step(render=False)
            if self.collision_monitor is not None:
                self.collision_monitor.after_step(self._physics_dt)
            after = self.time_s()
            if not math.isfinite(after) or after <= before:
                raise RuntimeError(
                    f"Physics step did not advance the simulation clock: {before} -> {after}; "
                    "this is a simulation execution error, not a task failure"
                )

    def _wait_until_playing(self) -> None:
        """Honor GUI pause without advancing actions, scores or model calls."""
        notified = False
        while not self.world.is_playing():
            if not self.app.is_running() or self.world.is_stopped():
                raise KeyboardInterrupt("Simulation window was closed or timeline stopped")
            if self.settings.headless:
                raise RuntimeError("Headless simulation timeline is unexpectedly paused")
            if not notified:
                print("R5_SIMULATION_PAUSED: press Play in Isaac Sim to resume", flush=True)
                notified = True
            self.world.render()
            time.sleep(0.01)
        if notified:
            print("R5_SIMULATION_RESUMED", flush=True)

    def pump_events(self) -> None:
        """Service the GUI on its owning thread without ticking or capturing."""
        if self.app is None or not self.app.is_running():
            raise KeyboardInterrupt("Simulation window was closed")
        if self.settings.headless:
            return
        timestamp = self.time_s()
        # SimulationContext.render disables playSimulations during app.update;
        # plain app.update would risk advancing physics on the frozen observation.
        self.world.render()
        if not math.isclose(self.time_s(), timestamp, abs_tol=1e-9):
            raise RuntimeError("GUI refresh unexpectedly advanced the simulation clock")

    def capture_rgb(self) -> tuple[RGBFrame, ...]:
        import numpy as np
        if self._rgb is None and self.camera_rig is None:
            raise RuntimeError("Camera has not been initialized")
        timestamp = self.time_s()
        self._capture_id += 1
        # First-use shader/render-product initialization may finish after the first
        # render on Linux. Retry empty buffers only, without advancing physics.
        attempts = 8
        # Flush render latency at the SAME physics state. Readiness alone does not
        # establish freshness. The camera integration check verifies pixel changes.
        passes = getattr(self.settings, "capture_render_passes", 1)
        for _ in range(passes - 1):
            self.world.render()
            self._rep.orchestrator.step(delta_time=0.0, pause_timeline=False, rt_subframes=2)
            if not math.isclose(self.time_s(), timestamp, abs_tol=1e-9):
                raise RuntimeError("RGB capture unexpectedly advanced the simulation clock")
        for attempt in range(attempts):
            self.world.render()
            self._rep.orchestrator.step(delta_time=0.0, pause_timeline=False, rt_subframes=2)
            if not math.isclose(self.time_s(), timestamp, abs_tol=1e-9):
                raise RuntimeError("RGB capture unexpectedly advanced the simulation clock")
            try:
                if self.camera_rig is not None:
                    return self.camera_rig.read(timestamp, self._capture_id)
                data = np.asarray(self._rgb.get_data())
                if data.size == 0:
                    raise RGBNotReady("overview: RGB buffer is not ready")
                if (data.ndim != 3 or data.shape[:2] != (self.settings.height, self.settings.width)
                        or data.shape[2] not in {3, 4}):
                    raise RuntimeError(f"Camera returned invalid RGB dimensions: {data.shape}")
                rgb = np.ascontiguousarray(data[..., :3], dtype=np.uint8)
                return (RGBFrame("overview", timestamp, self.settings.width, self.settings.height, rgb.tobytes()),)
            except RGBNotReady:
                if attempt == attempts - 1:
                    raise
                print(f"R5_RGB_WARMUP retry={attempt + 1}", flush=True)

    def time_s(self) -> float:
        return 0.0 if self.world is None else float(self.world.current_time)

    def evaluation_state(self) -> EvaluationState:
        return EvaluationState({"simulation_time_s": self.time_s()})

    def save_scene(self, path: Path) -> None:
        if self.world is None:
            raise RuntimeError("Backend is not open")
        self.world.stage.Export(str(path))

    def close(self) -> None:
        app, self.app = self.app, None
        try:
            if self.collision_monitor is not None:
                self.collision_monitor.close()
            if self.camera_rig is not None:
                self.camera_rig.close()
            if self._rgb is not None and self._render_product is not None:
                self._rgb.detach(self._render_product)
            if self._render_product is not None:
                self._render_product.destroy()
            if self.world is not None:
                self.world.stop()
                self.world.clear_instance()
        finally:
            self.world = None
            self._rgb = self._render_product = self._rep = None
            self.camera_profile = self.camera_rig = None
            self.camera_names = ()
            self.before_physics_step = None
            self.articulations.clear()
            self.ee_views.clear()
            self.lab_asset = None
            self.planning_environment_paths = ()
            self.planning_legacy_bench = True
            self.wrist_geometry = {}
            self.collision_monitor = None
            self.__dict__.pop("_planner_static_scene", None)
            if app is not None:
                app.close()
