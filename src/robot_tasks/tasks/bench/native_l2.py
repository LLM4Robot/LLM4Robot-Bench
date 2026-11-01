"""Native L20 L2 rigid assets, preserving their separate bodies and materials.

Only episode initialization writes poses. Native joints stay passive; delivered
handles, mass and inertia are preserved without robot attachments.
"""

from dataclasses import asdict
from itertools import product

import numpy as np

from ...asset_delivery import verified_prop
from ...backends.scene_geometry import bench_geometry
from ...robots.transforms import pose_transform, transform_pose
from ...types import InvalidSimulationState
from ..l1.scene import L1Scene



class NativeL2Scene:
    root = "/World/L2"
    report_key = "household"
    set_pose = L1Scene.set_pose
    add_part = L1Scene.add_part

    def __init__(self, backend, config, spec):
        from isaacsim.core.prims import RigidPrim, SingleRigidPrim, SingleXFormPrim
        from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

        self.backend, self.config, self.spec = backend, config, spec
        self.design, self.stage = spec.design, backend.world.stage
        self.bench = bench_geometry(config)
        self.bodies, self.initial, self.paths, self.bounds = {}, {}, {}, {}
        self.colliders, self.evidence, self.deliveries = [], [], {}
        self.cached_time, self.cached_rows = None, None
        self.auxiliary = []
        self.fixed_names = set()
        fixed_body_paths = []
        UsdGeom.Xform.Define(self.stage, self.root)
        for item in spec.parameters["native_instances"]:
            parent = self.root + "/" + item["name"]
            if item["asset"] not in self.deliveries:
                self.deliveries[item["asset"]] = verified_prop(item["asset"], task_id=config.task_id)
            path, evidence = self.deliveries[item["asset"]]
            prim = self.stage.DefinePrim(parent)
            prim.GetReferences().AddReference(str(path))
            self.set_pose(parent, item["pose"])
            # Compose a declared initial rigid transform once (e.g. a cap
            # pre-loosened along its helix). Never couple motion during play.
            for relative, initial in item.get("initial_body_transforms", {}).items():
                body = self.stage.GetPrimAtPath(parent + "/" + relative)
                if not body or not body.HasAPI(UsdPhysics.RigidBodyAPI):
                    raise ValueError(f"Missing native initial body: {parent}/{relative}")
                xform = UsdGeom.Xformable(body)
                local = np.asarray(xform.GetLocalTransformation()).T
                composed = pose_transform(initial) @ local
                xform.MakeMatrixXform().Set(Gf.Matrix4d(composed.T.tolist()))
            # Initial joint displacement only: retain authored closed-position
            # joint frames/limits and all native geometry, mass and materials.
            for relative, offset in item.get("initial_body_offsets", {}).items():
                body = self.stage.GetPrimAtPath(parent + "/" + relative)
                if not body or not body.HasAPI(UsdPhysics.RigidBodyAPI):
                    raise ValueError(f"Missing native initial body: {parent}/{relative}")
                xform = UsdGeom.Xformable(body)
                transform = xform.GetLocalTransformation()
                transform.SetTranslateOnly(transform.ExtractTranslation() + Gf.Vec3d(*offset))
                xform.MakeMatrixXform().Set(transform)
            self.evidence.append(
                {
                    **evidence,
                    "instance": item["name"],
                    "static_bodies": item.get("static", []),
                    "initial_body_offsets": item.get("initial_body_offsets", {}),
                    "initial_body_transforms": item.get("initial_body_transforms", {}),
                    "native_dimensions": True,
                    "native_materials": not item["name"].startswith("load_"),
                }
            )
            for relative in item.get("static", []):
                body = self.stage.GetPrimAtPath(
                    parent + ("/" + relative if relative else "")
                )
                if not body:
                    raise ValueError(f"Missing native fixed body: {parent}/{relative}")
                # Kinematic anchoring preserves native joint frames and bodies.
                UsdPhysics.RigidBodyAPI(body).CreateKinematicEnabledAttr(True)
                PhysxSchema.PhysxRigidBodyAPI.Apply(body).CreateEnableCCDAttr(False)
                fixed_body_paths.append(str(body.GetPath()))
            for relative in item.get("disabled_joints", []):
                UsdPhysics.Joint(
                    self.stage.GetPrimAtPath(parent + "/" + relative)
                ).CreateJointEnabledAttr(False)
            if item["name"].startswith("load_") and "ball_friction" in spec.parameters:
                # The transport protocol does not use the 1 mm bore. Hundreds
                # of 16-hull pearls made CPU contact stepping impractical.
                # Preserve native visuals/mass; explicitly declare this proxy.
                if spec.parameters.get("pearl_collision") == "outer_sphere":
                    for shape in Usd.PrimRange(prim):
                        if shape.HasAPI(UsdPhysics.CollisionAPI):
                            UsdPhysics.CollisionAPI(shape).CreateCollisionEnabledAttr(
                                False
                            )
                    sphere = UsdGeom.Sphere.Define(
                        self.stage, parent + "/TransportCollider"
                    )
                    sphere.CreateRadiusAttr(spec.parameters["ball_radius_m"])
                    sphere.CreateVisibilityAttr("invisible")
                    UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
                    contact = PhysxSchema.PhysxCollisionAPI.Apply(sphere.GetPrim())
                    contact.CreateContactOffsetAttr(0.00015)
                    contact.CreateRestOffsetAttr(0.0)
                if not hasattr(self, "ball_material"):
                    from isaacsim.core.api.materials import PhysicsMaterial

                    f = spec.parameters["ball_friction"]
                    self.ball_material = PhysicsMaterial(
                        self.root + "/PearlMaterial",
                        static_friction=f,
                        dynamic_friction=f,
                        restitution=0.0,
                    )
                    PhysxSchema.PhysxMaterialAPI.Apply(
                        self.ball_material.material.GetPrim()
                    ).CreateFrictionCombineModeAttr("min")
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                    self.ball_material.material,
                    bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                    materialPurpose="physics",
                )
            for name, relative in item["tracked"].items():
                body_path = parent + ("/" + relative if relative else "")
                self.paths[name] = body_path
            tracked_paths = set(self.paths.values())
            for body in Usd.PrimRange(prim):
                if (
                    body.HasAPI(UsdPhysics.RigidBodyAPI)
                    and str(body.GetPath()) not in tracked_paths
                ):
                    self.auxiliary.append(str(body.GetPath()))

        marker_contacts = None
        if spec.design.task_id == "L2-01":
            from .marker_contacts import configure_marker_box_contacts

            marker_contacts = configure_marker_box_contacts(
                self.stage, self.root + "/marker", self.root + "/support_box"
            )

        if spec.parameters.get("native_cup_names"):
            from .native_geometry import catalog, verify_stage

            if self.deliveries["ripple_paper_cup"][1]["sha256"] != catalog()["ripple_paper_cup"]["sha256"]:
                raise ValueError("Cup scoring geometry is from a different locked asset")
            verify_stage(self.stage, [("ripple_paper_cup", self.paths[name])
                                      for name in spec.parameters["native_cup_names"]])

        for part in self.design.fixtures:
            self.add_part(self.root + "/Fixture/" + part.name, part)
        transforms = UsdGeom.XformCache()
        bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        for name, path in self.paths.items():
            prim = self.stage.GetPrimAtPath(path)
            world = np.asarray(transforms.GetLocalToWorldTransform(prim)).T
            self.initial[name] = transform_pose(world)
            # USD's aligned root bound can inflate after a root rotation.
            # Measure visible vertices in the body's own frame, not a world
            # AABB transformed back (which rejects correctly placed objects).
            points = []
            inverse = np.linalg.inv(world)
            for shape in Usd.PrimRange(prim):
                if not shape.IsA(UsdGeom.Mesh):
                    continue
                if UsdGeom.Imageable(shape).ComputeVisibility() == "invisible":
                    continue
                vertices = np.asarray(UsdGeom.Mesh(shape).GetPointsAttr().Get())
                if vertices.ndim != 2 or vertices.shape[1] != 3:
                    continue
                relative = inverse @ np.asarray(transforms.GetLocalToWorldTransform(shape)).T
                points.append(vertices @ relative[:3, :3].T + relative[:3, 3])
            if points:
                vertices = np.concatenate(points)
                lo, hi = vertices.min(axis=0), vertices.max(axis=0)
            else:
                bound = bbox.ComputeUntransformedBound(prim).ComputeAlignedRange()
                lo, hi = np.asarray(bound.GetMin()), np.asarray(bound.GetMax())
            if not np.isfinite([lo, hi]).all() or max(hi - lo) <= 0:
                raise ValueError(f"Native tracked body has no finite geometry: {path}")
            self.bounds[name] = np.array(list(product(*zip(lo, hi))))
            if not name.startswith("load_"):
                fixed = path in fixed_body_paths
                if fixed:
                    self.fixed_names.add(name)
                self.bodies[name] = backend.world.scene.add(
                    (SingleXFormPrim if fixed else SingleRigidPrim)(path, name="native_l2_" + name)
                )
        self.load_names = [n for n in self.paths if n.startswith("load_")]
        self.load_view = None
        self.load_containers = tuple(spec.parameters.get("load_container_names", ("bowl",)))
        if self.load_names:
            self.load_view = backend.world.scene.add(
                RigidPrim(
                    self.root + "/load_.*",
                    name="native_l2_loads",
                    reset_xform_properties=False,
                    track_contact_forces=True,
                    contact_filter_prim_paths_expr=[
                        self.bench.prim_path,
                        *[self.paths[n] for n in self.load_containers],
                    ],
                    max_contact_count=16384,
                    disable_stablization=False,
                )
            )
            self.load_names = [p.rsplit("/", 1)[-1] for p in self.load_view.prim_paths]
        self.aux_bodies = []
        for index, path in enumerate(dict.fromkeys(self.auxiliary)):
            prim = self.stage.GetPrimAtPath(path)
            if UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get():
                continue  # Fixed fixtures do not need velocity/reset writes.
            world = np.asarray(transforms.GetLocalToWorldTransform(prim)).T
            body = backend.world.scene.add(
                SingleRigidPrim(path, name=f"native_l2_aux_{index}")
            )
            self.aux_bodies.append((body, transform_pose(world)))

        self.manipulated = [n for n in self.paths if not n.startswith("load_")]
        filters = [self.paths[n] for n in self.manipulated]
        self.arm_contacts = {
            m.name: RigidPrim(
                m.prim_path + "/link[78]",
                name="native_l2_fingers_" + m.name,
                reset_xform_properties=False,
                track_contact_forces=True,
                contact_filter_prim_paths_expr=filters,
                max_contact_count=4096,
                disable_stablization=False,
            )
            for m in config.robot.mounts
        }
        self.supports, self.support_names = {}, {}
        fixture_paths = [
            p
            for p in self.auxiliary
            if UsdPhysics.RigidBodyAPI(self.stage.GetPrimAtPath(p))
            .GetKinematicEnabledAttr()
            .Get()
        ]
        fixture_paths += [
            self.root + "/Fixture/" + p.name
            for p in self.design.fixtures
            if p.collision
        ]
        self.fixture_paths = fixture_paths
        # Include every moving arm link: camera/bracket colliders belong to
        # link6. Finger-only sensing misses an observer pushing with its wrist.
        observer_targets = list(dict.fromkeys(filters + fixture_paths))
        self.observer_views = {
            arm: RigidPrim(
                next(m.prim_path for m in config.robot.mounts if m.name == arm)
                + "/link[1-8]",
                name="native_l2_observer_" + arm,
                reset_xform_properties=False,
                track_contact_forces=True,
                contact_filter_prim_paths_expr=observer_targets,
                max_contact_count=4096,
                disable_stablization=False,
            )
            for arm in config.observation_arms
        }
        self.observer_targets = observer_targets
        for name, path in self.paths.items():
            if name.startswith("load_"):
                continue
            others = [n for n in self.manipulated if n != name]
            targets = (
                [self.bench.prim_path] + fixture_paths + [self.paths[n] for n in others]
            )
            self.support_names[name] = others
            self.supports[name] = RigidPrim(
                path,
                name="native_l2_support_" + name,
                reset_xform_properties=False,
                track_contact_forces=True,
                contact_filter_prim_paths_expr=targets,
                max_contact_count=2048,
                disable_stablization=False,
            )
        backend.planning_payloads = self.planning_payloads
        # Contact filters use body roots, while the planner needs actual shape
        # paths. Native assets put most collision meshes beneath Xform roots.
        backend.planning_fixture_paths = tuple(
            str(shape.GetPath())
            for path in dict.fromkeys(fixture_paths + fixed_body_paths)
            for shape in Usd.PrimRange(self.stage.GetPrimAtPath(path))
            if shape.HasAPI(UsdPhysics.CollisionAPI)
            and UsdPhysics.CollisionAPI(shape).GetCollisionEnabledAttr().Get()
        )
        backend.scene_report[self.report_key] = {
            "version": self.design.version,
            "mechanism": spec.mechanism,
            "limitations": spec.limitations,
            "assets": self.evidence,
            "parameters": asdict(spec)["parameters"],
            "public_facts": spec.facts,
            "initial_poses": self.initial,
            "physics_overrides": {
                "fixed_bodies": fixed_body_paths,
                "marker_box_contacts": marker_contacts,
                "pearl_collision": spec.parameters.get("pearl_collision"),
                "pearl_friction": spec.parameters.get("ball_friction"),
            },
            "object_attachment": False,
            "arm_roles": config.arm_roles,
            "grasp_adapters": [],
            "ground_truth_use": "scoring and collision geometry only; no object poses supplied to the policy",
        }

    def reset(self, seed):
        self.cached_time, self.cached_rows = None, None
        for name, body in self.bodies.items():
            p = self.initial[name]
            if name in self.fixed_names:
                body.set_default_state(position=np.asarray(p[:3]), orientation=np.asarray(p[3:]))
                continue
            body.set_default_state(
                position=np.asarray(p[:3]),
                orientation=np.asarray(p[3:]),
                linear_velocity=np.zeros(3),
                angular_velocity=np.zeros(3),
            )
        for body, p in self.aux_bodies:
            body.set_default_state(
                position=np.asarray(p[:3]),
                orientation=np.asarray(p[3:]),
                linear_velocity=np.zeros(3),
                angular_velocity=np.zeros(3),
            )
        if self.load_view is not None:
            poses = np.asarray([self.initial[n] for n in self.load_names])
            zeros = np.zeros((len(self.load_names), 3))
            self.load_view.set_default_state(
                positions=poses[:, :3],
                orientations=poses[:, 3:],
                linear_velocities=zeros,
                angular_velocities=zeros,
            )
        self.backend.reset(seed)
        for view in (*self.arm_contacts.values(), *self.supports.values()):
            view.initialize()
        for view in self.observer_views.values():
            view.initialize()
        if self.load_view is not None:
            self.load_view.initialize()
            if list(self.load_view.prim_paths) != [
                self.paths[n] for n in self.load_names
            ]:
                raise RuntimeError("Unexpected native pearl view order")

    def measure(self):
        now = self.backend.time_s()
        if self.cached_time == now and self.cached_rows is not None:
            return self.cached_rows
        forces = {
            a: np.asarray(v.get_contact_force_matrix(dt=self.config.physics_dt_s))
            for a, v in self.arm_contacts.items()
        }
        expected = (2, len(self.manipulated), 3)
        if any(
            f.shape != expected or not np.isfinite(f).all() for f in forces.values()
        ):
            raise InvalidSimulationState(
                "Invalid native L2 finger contact measurements"
            )
        rows = {}
        for name, body in self.bodies.items():
            xyz, q = body.get_world_pose()
            t = pose_transform((*xyz, *q))
            contacts = np.asarray(
                self.supports[name].get_contact_force_matrix(
                    dt=self.config.physics_dt_s
                )
            )[0]
            nf = len(self.fixture_paths)
            if not np.isfinite(contacts).all():
                raise InvalidSimulationState("Nonfinite native asset contacts: " + name)
            fingers = (
                {
                    a: np.linalg.norm(
                        f[:, self.manipulated.index(name)], axis=1
                    ).tolist()
                    for a, f in forces.items()
                }
                if name in self.manipulated
                else {a: [0.0, 0.0] for a in forces}
            )
            pairs = dict(
                zip(
                    self.support_names[name],
                    np.linalg.norm(contacts[1 + nf :], axis=1).tolist(),
                )
            )
            rows[name] = {
                "pose": [*map(float, xyz), *map(float, q)],
                "linear_velocity": [0., 0., 0.] if name in self.fixed_names else body.get_linear_velocity().tolist(),
                "angular_velocity": [0., 0., 0.] if name in self.fixed_names else body.get_angular_velocity().tolist(),
                "fingers": fingers,
                "pair_forces": pairs,
                "bench_force_n": float(np.linalg.norm(contacts[0])),
                "fixture_force_n": float(
                    np.linalg.norm(contacts[1 : 1 + nf].sum(axis=0))
                ),
                "corners": (self.bounds[name] @ t[:3, :3].T + xyz).tolist(),
            }
        if self.load_view is not None:
            positions, orientations = self.load_view.get_world_poses()
            velocities = self.load_view.get_velocities()
            contacts = np.asarray(
                self.load_view.get_contact_force_matrix(dt=self.config.physics_dt_s)
            )
            net_contacts = np.asarray(
                self.load_view.get_net_contact_forces(dt=self.config.physics_dt_s)
            )
            if (
                net_contacts.shape != (len(self.load_names), 3)
                or not np.isfinite(net_contacts).all()
                or contacts.shape != (len(self.load_names), 1 + len(self.load_containers), 3)
                or not np.isfinite(contacts).all()
            ):
                raise InvalidSimulationState("Invalid bulk pearl contact measurement")
            for i, name in enumerate(self.load_names):
                xyz, q = positions[i], orientations[i]
                t = pose_transform((*xyz, *q))
                rows[name] = {
                    "pose": [*map(float, xyz), *map(float, q)],
                    "linear_velocity": velocities[i, :3].tolist(),
                    "angular_velocity": velocities[i, 3:].tolist(),
                    "fingers": {a: [0.0, 0.0] for a in forces},
                    "fixture_force_n": 0.0,
                    "bench_force_n": float(np.linalg.norm(contacts[i, 0])),
                    "pair_forces": {
                        container: float(np.linalg.norm(contacts[i, j + 1]))
                        for j, container in enumerate(self.load_containers)
                    },
                    "net_contact_force_world_n": net_contacts[i].tolist(),
                    "corners": (self.bounds[name] @ t[:3, :3].T + xyz).tolist(),
                }
        self.cached_time, self.cached_rows = now, rows
        return rows

    def observer_evidence(self):
        from ...types import InvalidSimulationState

        result = {}
        for arm, view in self.observer_views.items():
            forces = np.asarray(
                view.get_contact_force_matrix(dt=self.config.physics_dt_s)
            )
            if (
                forces.shape != (8, len(self.observer_targets), 3)
                or not np.isfinite(forces).all()
            ):
                raise InvalidSimulationState("Invalid observer rigid contact readback")
            magnitudes = np.linalg.norm(forces, axis=2)
            result[arm] = {
                "mode": "rigid_contact",
                "max_force_n": float(magnitudes.max()),
                "targets_n": {
                    path: float(magnitudes[:, i].max())
                    for i, path in enumerate(self.observer_targets)
                    if magnitudes[:, i].max() > 0.01
                },
                "scope": "all arm links including wrist-mounted housing; task props and fixtures",
            }
        return result

    def planning_payloads(self):
        result = {a: [] for a in self.config.arms}
        for row in self.measure().values():
            for arm, payloads in result.items():
                if min(row["fingers"][arm]) > 0.05:
                    payloads.append(
                        {
                            "surface_points_world": row["corners"],
                            "source": "native geometry with measured contact; collision only",
                        }
                    )
        return result
