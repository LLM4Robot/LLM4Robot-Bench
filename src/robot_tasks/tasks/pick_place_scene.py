"""Physical cube scene and scoring parameters for GPT pick/place."""

from dataclasses import dataclass
import math

import numpy as np

from ..config import TaskConfig


@dataclass(frozen=True)
class InteractionSettings:
    cube_size_m: float = 0.04
    cube_mass_kg: float = 0.08
    friction: float = 0.8
    source_x_m: float = 0.38
    transfer_y_m: float = 0.15
    open_width_m: float = 0.070
    lift_threshold_m: float = 0.07
    placement_tolerance_m: float = 0.025
    stability_s: float = 1.0
    contact_threshold_n: float = 0.05
    placement_height_tolerance_m: float = 0.006
    placement_speed_m_s: float = 0.02
    placement_angular_speed_rad_s: float = 0.2
    gripper_open_tolerance_m: float = 0.003

    @classmethod
    def from_config(cls, config: TaskConfig) -> "InteractionSettings":
        data = dict(config.raw.get("interaction", {}))
        unknown = set(data) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown interaction settings: {sorted(unknown)}")
        result = cls(**data)
        for name in cls.__dataclass_fields__:
            value = getattr(result, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"interaction.{name} must be finite and positive")
        lower, upper = config.robot.gripper.width_limits_m
        if not lower < result.cube_size_m < result.open_width_m <= upper:
            raise ValueError("The object must fit the R5 gripper width limits")
        if result.transfer_y_m <= 2 * result.placement_tolerance_m:
            raise ValueError("Source and destination acceptance regions must be separate")
        return result


class InteractionScene:
    """A free dynamic cube. No attachment joint or pose writes during execution."""

    cube_path = "/World/Interaction/Cube"

    def __init__(self, backend, config: TaskConfig, settings: InteractionSettings):
        from isaacsim.core.api.materials import PhysicsMaterial
        from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid
        from isaacsim.core.prims import RigidPrim
        from pxr import PhysxSchema, UsdShade

        self.backend, self.config, self.settings = backend, config, settings
        backend.planning_payloads = self.planning_payloads
        self.material = PhysicsMaterial("/World/Interaction/Material", static_friction=settings.friction,
                                        dynamic_friction=settings.friction, restitution=0.0)
        self.cube = backend.world.scene.add(DynamicCuboid(
            self.cube_path, name="interaction_cube", size=settings.cube_size_m,
            position=np.array(self.source("left")), mass=settings.cube_mass_kg,
            color=np.array([0.85, 0.16, 0.05]), physics_material=self.material,
        ))
        # Thin visual markers have no collision; the physical supporting surface is the bench.
        for arm in config.arms:
            destination = self.destination(arm)
            backend.world.scene.add(VisualCuboid(
                f"/World/Interaction/Target_{arm}", name=f"target_{arm}", size=1.0,
                scale=np.array([0.10, 0.10, 0.001]), position=np.array([*destination[:2], 0.0005]),
                color=np.array([0.1, 0.55, 0.25]),
            ))
        self.contacts = {}
        for arm in config.arms:
            base = next(m.prim_path for m in config.robot.mounts if m.name == arm)
            self.contacts[arm] = RigidPrim(
                prim_paths_expr=f"{base}/link[78]", name=f"{arm}_finger_contacts",
                reset_xform_properties=False, track_contact_forces=True,
                contact_filter_prim_paths_expr=[self.cube_path], max_contact_count=128,
                disable_stablization=False,
            )
        # One explicit material and small contact offset make centimetre-scale geometry meaningful.
        paths = [self.cube_path, "/World/Bench"] + [p for a in backend.scene_report["arms"].values()
                                                    for p in a["colliders"]]
        for path in paths:
            prim = backend.world.stage.GetPrimAtPath(path)
            # Imported collision meshes are instances. Material/offset overrides
            # belong to this check's stage, never the pinned manufacturer asset.
            while prim.IsInstanceProxy():
                ancestor = prim.GetParent()
                while ancestor and not ancestor.IsInstance():
                    ancestor = ancestor.GetParent()
                if not ancestor:
                    raise RuntimeError(f"Cannot configure instance collision: {path}")
                ancestor.SetInstanceable(False)
                prim = backend.world.stage.GetPrimAtPath(path)
            collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            collision.CreateContactOffsetAttr(0.001)
            collision.CreateRestOffsetAttr(0.0)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                self.material.material, bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                materialPurpose="physics")
        body = PhysxSchema.PhysxRigidBodyAPI.Apply(self.cube.prim)
        body.CreateSolverPositionIterationCountAttr(32)
        body.CreateSolverVelocityIterationCountAttr(0)
        backend.scene_report["interaction"] = {
            "cube_path": self.cube_path, "cube_size_m": settings.cube_size_m,
            "cube_mass_kg": settings.cube_mass_kg, "friction": settings.friction,
            "contact_offset_m": 0.001, "rest_offset_m": 0.0,
            "finger_contact_api": "RigidPrim.get_contact_force_matrix(dt=physics_dt_s)",
            "contact_sample_rate_hz": 1 / config.control_dt_s,
            "object_attached": False, "object_kinematic": False,
            "ground_truth_use": "Cube state/contacts serve evaluation and privileged collision-only payload checks; never model input.",
        }

    def source(self, arm: str) -> tuple[float, float, float]:
        mount = next(m for m in self.config.robot.mounts if m.name == arm)
        return (self.settings.source_x_m, mount.base_pose_wxyz[1], self.settings.cube_size_m / 2)

    def destination(self, arm: str) -> tuple[float, float, float]:
        x, y, z = self.source(arm)
        return (x, y + (-1 if arm == "left" else 1) * self.settings.transfer_y_m, z)

    def reset(self, arm: str) -> None:
        # This is the only object pose setter; called before the next episode begins.
        self.cube.set_default_state(position=np.array(self.source(arm)), orientation=np.array([1., 0., 0., 0.]),
                                    linear_velocity=np.zeros(3), angular_velocity=np.zeros(3))
        self.backend.reset(self.config.seed)
        for view in self.contacts.values():
            view.initialize()

    def measure(self, arm: str) -> dict:
        position, quaternion = self.cube.get_world_pose()
        matrix = np.asarray(self.contacts[arm].get_contact_force_matrix(dt=self.config.physics_dt_s))
        if matrix.shape != (2, 1, 3) or not np.isfinite(matrix).all():
            raise RuntimeError(f"Invalid finger contact matrix: {matrix}")
        return {
            "position_m": [float(v) for v in position], "quaternion_wxyz": [float(v) for v in quaternion],
            "linear_velocity_m_s": [float(v) for v in self.cube.get_linear_velocity()],
            "angular_velocity_rad_s": [float(v) for v in self.cube.get_angular_velocity()],
            "finger_normal_force_vectors_n": matrix[:, 0, :].tolist(),
            "finger_normal_force_n": np.linalg.norm(matrix[:, 0, :], axis=1).tolist(),
        }

    def planning_payloads(self):
        from ..planning.scene import box_surface
        from ..robots.transforms import pose_transform
        result = {}
        for arm in self.config.arms:
            sample = self.measure(arm)
            if min(sample["finger_normal_force_n"]) <= .05:
                continue
            transform = pose_transform((*sample["position_m"], *sample["quaternion_wxyz"]))
            local = box_surface([self.settings.cube_size_m]*3)
            result[arm] = [{"surface_points_world": (local@transform[:3,:3].T+transform[:3,3]).tolist(),
                            "source": "bilateral measured contact; not proof of successful lift"}]
        return result
