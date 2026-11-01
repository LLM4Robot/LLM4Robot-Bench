"""Shared rigid scene primitives. Pose writes occur only in episode reset."""

from dataclasses import asdict, dataclass

import numpy as np

from ...robots.transforms import origin_transform, pose_transform, transform_pose
from .assets import import_asset
from .specs import local_corners
from ...backends.scene_geometry import bench_geometry


@dataclass(frozen=True)
class ResetSettings:
    position_jitter_m: float = 0.0
    yaw_jitter_rad: float = 0.0


class L1Scene:
    root = "/World/L1"

    report_key = "l1"

    def __init__(self, backend, config, settings, *, design):
        from isaacsim.core.api.materials import PhysicsMaterial
        from isaacsim.core.prims import RigidPrim, SingleRigidPrim
        from pxr import PhysxSchema, UsdGeom, UsdPhysics, UsdShade

        self.backend, self.config, self.settings = backend, config, settings
        self.bench = bench_geometry(config)
        backend.planning_payloads = self.planning_payloads
        self.design = design
        self.bodies, self.contacts = {}, {}
        self.material = PhysicsMaterial(
            self.root + "/Material", static_friction=0.8, dynamic_friction=0.8, restitution=0.0
        )
        self.fixture_material = PhysicsMaterial(
            self.root + "/FixtureMaterial", static_friction=self.design.fixture_friction,
            dynamic_friction=self.design.fixture_friction, restitution=0.0,
        )
        # Make the contact-pair rule explicit. With the zero-friction socket,
        # min(0, .8) is zero, while gripper/peg and bench/peg remain .8.
        for material in (self.material, self.fixture_material):
            PhysxSchema.PhysxMaterialAPI.Apply(material.material.GetPrim()).CreateFrictionCombineModeAttr("min")
        stage = backend.world.stage
        self.stage = stage
        self.colliders = []
        evidence = []
        # Build at origin: normalization must not include the world placement.
        UsdGeom.Xform.Define(stage, self.root + "/Fixture")
        if self.design.fixture_asset:
            paths, record = import_asset(
                stage,
                self.root + "/Fixture",
                self.design.fixture_asset,
                self.design.fixture_size,
                (0, 0, 0),
                dynamic=False,
                task_id=config.task_id,
            )
            self.colliders.extend(paths)
            evidence.append(record)
            self.set_pose(self.root + "/Fixture", self.design.fixture_pose)
        for part in self.design.fixtures:
            self.add_part(self.root + "/Fixture/" + part.name, part)
        for article in self.design.articles:
            path = self.root + "/" + article.name
            UsdGeom.Xform.Define(stage, path)
            if article.asset:
                paths, record = import_asset(
                    stage, path, article.asset, article.asset_size, article.asset_rotation,
                    dynamic=True, task_id=config.task_id,
                )
                self.colliders.extend(paths)
                evidence.append(record)
            else:
                for part in article.parts:
                    self.add_part(path + "/" + part.name, part)
            prim = stage.GetPrimAtPath(path)
            UsdPhysics.RigidBodyAPI.Apply(prim).CreateKinematicEnabledAttr(False)
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(article.mass)
            body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            body.CreateSolverPositionIterationCountAttr(backend.settings.solver_position_iterations)
            body.CreateSolverVelocityIterationCountAttr(backend.settings.solver_velocity_iterations)
            self.bodies[article.name] = backend.world.scene.add(
                SingleRigidPrim(
                    path,
                    name="l1_" + article.name,
                    position=np.array(article.initial[:3]),
                    orientation=np.array(article.initial[3:]),
                )
            )
        # Keep grasp and bench friction; the fixture has its own material.
        paths = (
            self.colliders
            + [self.bench.prim_path]
            + [p for a in backend.scene_report["arms"].values() for p in a["colliders"]]
        )
        for path in paths:
            prim = stage.GetPrimAtPath(path)
            while prim.IsInstanceProxy():
                ancestor = prim.GetParent()
                while ancestor and not ancestor.IsInstance():
                    ancestor = ancestor.GetParent()
                if not ancestor:
                    raise RuntimeError(f"Cannot deinstance collider {path}")
                ancestor.SetInstanceable(False)
                prim = stage.GetPrimAtPath(path)
            api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            # Native thin walls and finger recesses need a sub-millimetre skin.
            api.CreateContactOffsetAttr(0.0002 if getattr(backend, "lab_asset", None) is not None else 0.001)
            api.CreateRestOffsetAttr(0.0)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                (self.fixture_material if path.startswith(self.root + "/Fixture/") else self.material).material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                materialPurpose="physics",
            )
        arm = config.arms[0]
        base = next(m.prim_path for m in config.robot.mounts if m.name == arm)
        self.fingers = RigidPrim(
            base + "/link[78]",
            name="l1_finger_contacts",
            reset_xform_properties=False,
            track_contact_forces=True,
            contact_filter_prim_paths_expr=[self.root + "/" + a.name for a in self.design.articles],
            max_contact_count=512,
            disable_stablization=False,
        )
        self.support_filters = [p for p in self.colliders if p.startswith(self.root + "/Fixture/")] + [
            self.bench.prim_path
        ]
        backend.planning_fixture_paths = tuple(self.support_filters[:-1])
        self.supports = {
            a.name: RigidPrim(
                self.root + "/" + a.name,
                name="l1_support_" + a.name,
                reset_xform_properties=False,
                track_contact_forces=True,
                contact_filter_prim_paths_expr=self.support_filters,
                max_contact_count=512,
                disable_stablization=False,
            )
            for a in self.design.articles
        }
        backend.scene_report[self.report_key] = {
            "version": self.design.version,
            "design": asdict(self.design),
            "assets": evidence,
            "reset": asdict(settings),
            "contact_materials": {"gripper_peg_and_bench": {"static": 0.8, "dynamic": 0.8},
                                  "fixture": {"static": self.design.fixture_friction,
                                              "dynamic": self.design.fixture_friction},
                                  "friction_combine_mode": "min", "restitution": 0.0},
            "colliders": self.colliders,
            "ground_truth_use": "evaluator plus privileged collision-only planner; never object state in model input",
            "controller": {
                "adapter": "R5Adapter / cuRobo trajectory",
                "motion_mode": "collision_checked_trajectory",
                "gripper_timing": "after_motion",
                "startup_preparation": "hold_joint_home_v1",
                "observation_preparation": "model_controlled_from_joint_home",
            },
            "arm_roles": config.arm_roles,
            "physics_variant": "CPU PhysX, fixed fixture colliders, dynamic primitives/convex shapes; see asset overrides",
            "object_attachment": False,
            "initial_poses": {},
        }

    def set_pose(self, path, pose):
        from pxr import Gf, UsdGeom

        x = UsdGeom.Xformable(self.stage.GetPrimAtPath(path))
        x.ClearXformOpOrder()
        x.AddTranslateOp().Set(Gf.Vec3d(*pose[:3]))
        x.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(pose[3], Gf.Vec3d(*pose[4:])))

    def add_part(self, path, part):
        from pxr import Gf, UsdGeom, UsdPhysics

        if part.shape == "mesh":
            shape = UsdGeom.Mesh.Define(self.stage, path)
            shape.CreatePointsAttr([Gf.Vec3f(*v) for v in part.vertices])
            shape.CreateFaceVertexCountsAttr([3] * len(part.faces))
            shape.CreateFaceVertexIndicesAttr(np.asarray(part.faces).ravel().tolist())
            shape.CreateSubdivisionSchemeAttr("none")
            if part.collision:
                UsdPhysics.MeshCollisionAPI.Apply(shape.GetPrim()).CreateApproximationAttr("convexHull")
        elif part.shape == "box":
            shape = UsdGeom.Cube.Define(self.stage, path)
            shape.CreateSizeAttr(1.0)
        elif part.shape == "cylinder":
            shape = UsdGeom.Cylinder.Define(self.stage, path)
            shape.CreateRadiusAttr(part.size[0])
            shape.CreateHeightAttr(part.size[2])
            shape.CreateAxisAttr("Z")
        elif part.shape == "sphere":
            shape = UsdGeom.Sphere.Define(self.stage, path)
            shape.CreateRadiusAttr(part.size[0])
        else:
            raise ValueError(f"Unsupported L1 part shape: {part.shape}")
        self.set_pose(path, transform_pose(origin_transform(part.position, part.rotation)))
        if part.shape == "box":
            UsdGeom.Xformable(shape).AddScaleOp().Set(Gf.Vec3f(*part.size))
        shape.CreateDisplayColorAttr([Gf.Vec3f(*part.color)])
        if part.collision:
            UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
            self.colliders.append(path)

    def reset(self, seed):
        rng = np.random.default_rng(seed)
        self.initial = {}
        for a in self.design.articles:
            t = pose_transform(a.initial)
            t[:2, 3] += rng.uniform(-self.settings.position_jitter_m, self.settings.position_jitter_m, 2)
            yaw = rng.uniform(-self.settings.yaw_jitter_rad, self.settings.yaw_jitter_rad)
            t[:3, :3] = origin_transform((0, 0, 0), (0, 0, yaw))[:3, :3] @ t[:3, :3]
            p = transform_pose(t)
            self.initial[a.name] = p
            self.bodies[a.name].set_default_state(
                position=np.array(p[:3]),
                orientation=np.array(p[3:]),
                linear_velocity=np.zeros(3),
                angular_velocity=np.zeros(3),
            )
        self.backend.reset(seed)
        self.fingers.initialize()
        for view in self.supports.values():
            view.initialize()
        self.backend.scene_report[self.report_key]["initial_poses"] = self.initial

    def planning_payloads(self):
        """Collision-only privilege; no task evaluator is queried or changed."""
        from ...planning.scene import article_surface
        forces = np.asarray(self.fingers.get_contact_force_matrix(dt=self.config.physics_dt_s))
        payloads = []
        for i, article in enumerate(self.design.articles):
            if not np.all(np.linalg.norm(forces[:,i,:],axis=1) > .05):
                continue
            xyz, q = self.bodies[article.name].get_world_pose()
            transform = pose_transform((*xyz,*q))
            local = article_surface(article)
            payloads.append({"surface_points_world": (local@transform[:3,:3].T+transform[:3,3]).tolist(),
                             "source": "bilateral measured contact; sampled geometry, not a grasp-success label"})
        return {self.config.arms[0]: payloads}

    def measure(self):
        dt = self.config.physics_dt_s
        forces = np.asarray(self.fingers.get_contact_force_matrix(dt=dt))
        if forces.shape != (2, len(self.bodies), 3) or not np.isfinite(forces).all():
            raise RuntimeError("Invalid L1 finger contact data")
        contact_data = self.fingers.get_contact_force_data(dt=dt)
        points = np.asarray(contact_data[1])
        counts = np.asarray(contact_data[4])
        starts = np.asarray(contact_data[5])
        result = {}
        for i, a in enumerate(self.design.articles):
            support = np.asarray(self.supports[a.name].get_contact_force_matrix(dt=dt))
            if support.shape != (1, len(self.support_filters), 3) or not np.isfinite(support).all():
                raise RuntimeError("Invalid L1 support contact data")
            body = self.bodies[a.name]
            xyz, q = body.get_world_pose()
            t = pose_transform((*xyz, *q))
            local_contacts = []
            for f in range(2):
                n = int(counts[f, i])
                start = int(starts[f, i])
                local_contacts.append(((points[start : start + n] - xyz) @ t[:3, :3]).tolist())
            result[a.name] = {
                "pose": [*map(float, xyz), *map(float, q)],
                "linear_velocity": body.get_linear_velocity().tolist(),
                "angular_velocity": body.get_angular_velocity().tolist(),
                "finger_forces": np.linalg.norm(forces[:, i, :], axis=1).tolist(),
                "finger_contacts_local": local_contacts,
                "fixture_force_n": float(np.linalg.norm(support[0, :-1].sum(axis=0))),
                "bench_force_n": float(np.linalg.norm(support[0, -1])),
                "corners": (local_corners(a) @ t[:3, :3].T + xyz).tolist(),
            }
        return result
