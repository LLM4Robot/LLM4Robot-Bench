"""Shared rigid task scenes. Object pose writes are restricted to episode reset.

No success detector attaches, moves, releases or teleports task objects.
"""

from dataclasses import asdict

import numpy as np

from ...robots.transforms import pose_transform, transform_pose
from ..l1.scene import L1Scene, ResetSettings


class BenchScene(L1Scene):
    root = "/World/Household"
    report_key = "household"

    def __init__(self, backend, config, spec):
        from isaacsim.core.prims import RigidPrim

        self.spec = spec
        settings = ResetSettings()
        super().__init__(backend, config, settings, design=spec.design)
        if spec.parameters.get("fixed_articles"):
            from pxr import PhysxSchema, UsdPhysics
            for name in spec.parameters["fixed_articles"]:
                prim = self.stage.GetPrimAtPath(self.root + "/" + name)
                UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateEnableCCDAttr(False)
                prefix = self.root + "/" + name + "/"
                backend.planning_fixture_paths += tuple(path for path in self.colliders if path.startswith(prefix))
        if spec.design.task_id == "L1-01":
            self.configure_marker_contacts()
        if spec.design.task_id in {"L1-01", "L1-02", "L1-03", "L1-04", "L1-05"}:
            from .native_geometry import catalog, verify_stage

            roots = [(a.asset.split(":", 1)[1], self.root + "/" + a.name)
                     for a in self.design.articles if a.asset.startswith("tuojing:")]
            if self.design.fixture_asset in {"tuojing:ripple_paper_cup", "tuojing:black_ceramic_mug", "tuojing:wooden_rack"}:
                roots.append((self.design.fixture_asset.split(":", 1)[1], self.root + "/Fixture"))
            source_hashes = {record.get("prop"): record.get("sha256")
                             for record in backend.scene_report[self.report_key]["assets"]}
            for name, _ in roots:
                if source_hashes.get(name) != catalog()[name]["sha256"]:
                    raise ValueError("Scoring geometry is from a different locked delivery: " + name)
            verify_stage(self.stage, roots)
            backend.scene_report[self.report_key]["native_scoring_geometry"] = {
                "source_sha256": {name: catalog()[name]["sha256"] for name, _ in roots},
                "verified_against_imported_stage": True,
                "primitive_surface_error_budget_m": 0.00005,
                "geometry_does_not_enter_policy_input": True,
            }
        if spec.design.task_id == "L1-02":
            from pxr import PhysxSchema

            # A 1-mm contact margin exceeds the native 0.55-mm side clearance.
            # Restore the delivered 0.2-mm margin without enlarging the slots.
            for path in self.colliders:
                PhysxSchema.PhysxCollisionAPI.Apply(self.stage.GetPrimAtPath(path)).CreateContactOffsetAttr(0.0002)
        paths = [self.root + "/" + a.name for a in self.design.articles]
        self.arm_contacts = {config.arms[0]: self.fingers}
        for mount in config.robot.mounts:
            if mount.name not in self.arm_contacts:
                self.arm_contacts[mount.name] = RigidPrim(
                    mount.prim_path + "/link[78]",
                    name="household_fingers_" + mount.name,
                    reset_xform_properties=False,
                    track_contact_forces=True,
                    contact_filter_prim_paths_expr=paths,
                    max_contact_count=4096,
                    disable_stablization=False,
                )
        self.pairs, self.pair_names = {}, {}
        for article in self.design.articles:
            others = [a.name for a in self.design.articles if a.name != article.name]
            self.pair_names[article.name] = others
            if others:
                self.pairs[article.name] = RigidPrim(
                    self.root + "/" + article.name,
                    name="household_pairs_" + article.name,
                    reset_xform_properties=False,
                    track_contact_forces=True,
                    contact_filter_prim_paths_expr=[
                        self.root + "/" + n for n in others
                    ],
                    max_contact_count=4096,
                    disable_stablization=False,
                )
        self.drives = {}
        for joint in spec.joints:
            self.create_joint(joint)
        report = backend.scene_report[self.report_key]
        report.update(
            mechanism=spec.mechanism,
            limitations=spec.limitations,
            public_facts=spec.facts,
            parameters=spec.parameters,
            goal_rule=(spec.design.instruction if spec.parameters.get("scoring_protocol") == "feishu_english_20261015_v1" else
                       "marker end inside the main cup cavity; release, support and stability are diagnostic only"
                       if spec.design.task_id == "L1-01" else
                       "task-specific physical geometry, contact/process evidence and stable final state"),
            joints=[asdict(j) for j in spec.joints],
            scoring={
                "version": "household-v1",
                "scoring_protocol": spec.parameters.get("scoring_protocol", "recoverable_progress_v1"),
                "stable_s": spec.parameters.get("hold_s", 1.0),
                "damage": "proxy thresholds only; no material fracture",
                "quality_references": "uncalibrated force/speed excursions are advisory, not damage or automatic failure",
            },
        )

    def configure_marker_contacts(self):
        """Keep the delivered cup friction and contact margin for passive settling."""
        from pxr import PhysxSchema, UsdPhysics

        native = UsdPhysics.MaterialAPI(
            self.stage.GetPrimAtPath(self.root + "/Fixture/Looks/Contact")
        )
        static, dynamic = native.GetStaticFrictionAttr().Get(), native.GetDynamicFrictionAttr().Get()
        if not all(value is not None and np.isfinite(value) and value > 0
                   for value in (static, dynamic)):
            raise ValueError("L1-01 requires positive delivered cup contact friction")
        self.fixture_material.set_static_friction(static)
        self.fixture_material.set_dynamic_friction(dynamic)
        for path in self.colliders:
            PhysxSchema.PhysxCollisionAPI.Apply(
                self.stage.GetPrimAtPath(path)
            ).CreateContactOffsetAttr(0.0002)
        report = self.backend.scene_report[self.report_key]
        report["contact_materials"]["fixture"] = {"static": static, "dynamic": dynamic}
        report["marker_contact_stability"] = {
            "cup_friction_source": "delivered Fixture/Looks/Contact; restitution remains zero",
            "contact_offset_m": 0.0002,
            "solver_velocity_iterations": self.backend.settings.solver_velocity_iterations,
            "angular_damping": "delivered value retained",
            "sleep_disabled": "required by contact-force sampling; do not lose support reports",
            "source_usd_modified": False,
        }

    def create_joint(self, spec):
        from pxr import Gf, UsdPhysics

        path = self.root + "/Joints/" + spec.name
        hinge_axis = next(iter(spec.axes)) if len(spec.axes) == 1 else None
        is_hinge = hinge_axis is not None and hinge_axis.startswith("rot")
        if is_hinge:
            # A single-axis hinge uses a revolute joint, not a D6 swing drive.
            # D6 Y/Z swing drives share native PhysX parameters.
            joint = UsdPhysics.RevoluteJoint.Define(self.stage, path)
            joint.CreateAxisAttr(hinge_axis[-1])
        else:
            joint = UsdPhysics.Joint.Define(self.stage, path)
        joint.CreateCollisionEnabledAttr(False)
        transforms = {a.name: pose_transform(a.initial) for a in self.design.articles}
        world = np.eye(4)
        world[:3, 3] = spec.anchor
        for index, name in enumerate((spec.body0, spec.body1)):
            if name:
                getattr(joint, f"CreateBody{index}Rel")().SetTargets(
                    [self.root + "/" + name]
                )
            local = np.linalg.inv(transforms[name]) @ world if name else world
            p = transform_pose(local)
            getattr(joint, f"CreateLocalPos{index}Attr")(Gf.Vec3f(*p[:3]))
            getattr(joint, f"CreateLocalRot{index}Attr")(
                Gf.Quatf(p[3], Gf.Vec3f(*p[4:]))
            )
        for axis in ("transX", "transY", "transZ", "rotX", "rotY", "rotZ"):
            if is_hinge and axis != hinge_axis:
                continue
            lower, upper, stiffness, damping = spec.axes.get(
                axis, (1.0, -1.0, 0.0, 0.0)
            )
            if is_hinge:
                joint.CreateLowerLimitAttr(lower)
                joint.CreateUpperLimitAttr(upper)
            else:
                limit = UsdPhysics.LimitAPI.Apply(joint.GetPrim(), axis)
                limit.CreateLowAttr(lower)
                limit.CreateHighAttr(upper)
            if stiffness or damping:
                drive = UsdPhysics.DriveAPI.Apply(
                    joint.GetPrim(), "angular" if is_hinge else axis
                )
                drive.CreateTypeAttr("force")
                drive.CreateStiffnessAttr(stiffness)
                drive.CreateDampingAttr(damping)
                drive.CreateTargetPositionAttr(0.0)
                drive.CreateTargetVelocityAttr(0.0)
                self.drives[spec.name, axis] = drive

    def reset(self, seed):
        for drive in self.drives.values():
            drive.CreateTargetPositionAttr(0.0)
        super().reset(seed)
        for view in (*self.arm_contacts.values(), *self.pairs.values()):
            view.initialize()

    def measure(self):
        rows = super().measure()
        dt = self.config.physics_dt_s
        forces = {
            arm: np.asarray(view.get_contact_force_matrix(dt=dt))
            for arm, view in self.arm_contacts.items()
        }
        for values in forces.values():
            if values.shape != (2, len(rows), 3) or not np.isfinite(values).all():
                raise RuntimeError("Invalid household finger contacts")
        for i, (name, row) in enumerate(rows.items()):
            row["fingers"] = {
                arm: np.linalg.norm(f[:, i], axis=1).tolist()
                for arm, f in forces.items()
            }
            row["pair_forces"] = {}
            if name in self.pairs:
                matrix = np.asarray(self.pairs[name].get_contact_force_matrix(dt=dt))
                if not np.isfinite(matrix).all():
                    raise RuntimeError("Invalid household pair contacts")
                row["pair_forces"] = dict(
                    zip(
                        self.pair_names[name],
                        map(float, np.linalg.norm(matrix[0], axis=1)),
                    )
                )
        if self.spec.design.task_id == "L1-03":
            suffix = self.spec.parameters["target_hook_suffix"]
            indices = [i for i, path in enumerate(self.support_filters) if path.endswith(suffix)]
            if len(indices) != 1:
                raise RuntimeError("L1-03 requires exactly one native Hook0 contact filter")
            force = np.asarray(self.supports["hat"].get_contact_force_matrix(dt=dt))[0, indices[0]]
            rows["hat"]["target_hook_force_n"] = float(np.linalg.norm(force))
            rows["hat"]["target_hook_vertical_force_n"] = float(max(0., force[2]))
        return rows

    def planning_payloads(self):
        from ...planning.scene import article_surface

        rows = self.measure()
        payloads = {a: [] for a in self.config.arms}
        for article in self.design.articles:
            if article.name in self.spec.parameters.get("fixed_articles", ()):
                continue
            row = rows[article.name]
            t = pose_transform(row["pose"])
            for arm in self.config.arms:
                if min(row["fingers"][arm]) > 0.05:
                    payloads[arm].append(
                        {
                            "surface_points_world": (
                                article_surface(article) @ t[:3, :3].T + t[:3, 3]
                            ).tolist(),
                            "source": "measured bilateral contact; collision only, never task-success evidence",
                        }
                    )
        return payloads
