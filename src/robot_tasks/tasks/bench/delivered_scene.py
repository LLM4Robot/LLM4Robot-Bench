"""Read-only bindings to complete delivered scenes; no duplicate old props."""

from dataclasses import asdict
from itertools import product
import numpy as np

from ...backends.scene_geometry import bench_geometry
from ...video_delivery import scenes
from .native_l2 import NativeL2Scene


def flower_payload_surface(pose, flexible_parts=None):
    """Use the actual component surfaces, not empty corners of a tilted AABB."""
    from .native_geometry import shapes, polyhedron
    from ...robots.transforms import pose_transform
    prop = 'video:artificial_rose'
    surfaces = []
    for i, shape in enumerate(shapes(prop)):
        part = next((value for name, value in (flexible_parts or {}).items()
                     if shape['name'].startswith(name + 'Contact')), None)
        transform = pose_transform(part['pose'] if part else pose)
        local = polyhedron(prop, i)[0]
        surfaces.append(local @ transform[:3,:3].T + transform[:3,3])
    return np.concatenate(surfaces)


def contact_filter_paths(stage, roots):
    """Static Xforms are not PhysX bodies: filter their actual collider paths."""
    from pxr import Usd, UsdPhysics
    result=[]
    for root in roots:
        prim=stage.GetPrimAtPath(root)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) or prim.HasAPI(UsdPhysics.CollisionAPI):
            result.append(root)
        else:
            result.extend(str(p.GetPath()) for p in Usd.PrimRange(prim)
                if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get())
    return list(dict.fromkeys(result))


class DeliveredScene(NativeL2Scene):
    """Reuse measured contact/reset interfaces with the supplied physical bodies."""

    def __init__(self, backend, config, spec):
        from isaacsim.core.prims import RigidPrim, SingleRigidPrim, SingleXFormPrim
        from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics
        from ...robots.transforms import transform_pose

        self.backend, self.config, self.spec, self.design = backend, config, spec, spec.design
        self.stage = backend.world.stage
        self.root = ("/World/Household" if config.task_id.startswith("L1-") else
                     "/World/L3Wiping" if config.task_id == "L2-07" else "/World/L2")
        self.bench = bench_geometry(config)
        from ...initial_layout import scene_record, apply_record_to_stage
        row = scene_record(config.raw['task'])
        apply_record_to_stage(self.stage, config.raw['task'], row)
        bottle_contacts = None
        if config.task_id == 'L3-01':
            from .bottle_contacts import configure
            bottle_contacts = configure(self.stage, self.root + '/bottle_assembly')
        leaf_records = row.get('physical_repairs', {}).get('flower_passive_leaves', {}).get('leaf_bodies', [])
        self.flower_leaf_paths = [part['path'] for part in leaf_records]
        self.flower_leaf_time = None
        self.flower_leaf_fingers, self.flower_leaf_supports = {}, {}
        self.paths = {n: record["path"] for n, record in row["tracked"].items()}
        self.initial = {n: tuple(record["pose"]) for n, record in row["tracked"].items()}
        self.bounds = {n: np.array(list(product(*zip(*record["bounds"]))))
                       for n, record in row["tracked"].items()}
        marker_filters = None
        if config.task_id == 'L2-01':
            from .marker_contacts import refresh_delivered_groups
            from .measured_assets import calibrate_marker
            marker_filters = refresh_delivered_groups(self.stage, self.root+'/marker')
            marker_filters['measurement_calibration'] = calibrate_marker(self.stage)
        self.bodies, self.fixed_names = {}, set()
        self.cached_time = self.cached_rows = None
        self.load_names, self.load_view = [], None
        self.aux_bodies = []
        fixture_roots = [record["path"] for record in row["fixtures"]]
        self.fixture_paths = contact_filter_paths(self.stage,fixture_roots)
        for n, path in self.paths.items():
            prim = self.stage.GetPrimAtPath(path)
            if config.task_id == "L3-01" and n == "bottle":
                # The user clarified that the bottle is a fixed fixture.
                UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(True)
                PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateEnableCCDAttr(False)
            fixed = bool(UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get())
            if fixed:
                self.fixed_names.add(n)
            self.bodies[n] = backend.world.scene.add(
                (SingleXFormPrim if fixed else SingleRigidPrim)(path, name="delivered_" + n))
        transforms = UsdGeom.XformCache()
        for prim in Usd.PrimRange(self.stage.GetPrimAtPath(self.root)):
            if (not prim.HasAPI(UsdPhysics.RigidBodyAPI) or str(prim.GetPath()) in self.paths.values()
                    or UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get()):
                continue
            matrix = np.asarray(transforms.GetLocalToWorldTransform(prim)).T.copy()
            u, _, vt = np.linalg.svd(matrix[:3,:3]);matrix[:3,:3] = u @ vt
            options = ({'reset_xform_properties': False}
                       if str(prim.GetPath()) in self.flower_leaf_paths else {})
            # A leaf has a world/reset transform below the stem body. Importer
            # normalization must not remove that reset and nest rigid bodies.
            body = backend.world.scene.add(SingleRigidPrim(str(prim.GetPath()),
                  name='delivered_aux_'+str(len(self.aux_bodies)), **options))
            self.aux_bodies.append((body,transform_pose(matrix)))
        self.manipulated = list(self.paths)
        filters = list(self.paths.values())
        self.arm_contacts = {
            m.name: RigidPrim(m.prim_path + "/link[78]", name="delivered_fingers_" + m.name,
                             reset_xform_properties=False, track_contact_forces=True,
                             contact_filter_prim_paths_expr=filters, max_contact_count=8192,
                             disable_stablization=False)
            for m in config.robot.mounts
        }
        if self.flower_leaf_paths:
            self.flower_leaf_fingers = {
                m.name: RigidPrim(m.prim_path + '/link[78]', name='flower_leaf_fingers_' + m.name,
                    reset_xform_properties=False, track_contact_forces=True,
                    contact_filter_prim_paths_expr=self.flower_leaf_paths, max_contact_count=4096,
                    disable_stablization=False) for m in config.robot.mounts}
            self.flower_leaf_support_targets = [n for n in self.paths if n != 'rose']
            for i, path in enumerate(self.flower_leaf_paths):
                self.flower_leaf_supports[path] = RigidPrim(path, name='flower_leaf_support_' + str(i),
                    reset_xform_properties=False, track_contact_forces=True,
                    contact_filter_prim_paths_expr=[self.bench.prim_path, *self.fixture_paths,
                        *[self.paths[n] for n in self.flower_leaf_support_targets]],
                    max_contact_count=4096, disable_stablization=False)
        self.supports, self.support_names = {}, {}
        for n, path in self.paths.items():
            others = [x for x in self.manipulated if x != n]
            self.support_names[n] = others
            self.supports[n] = RigidPrim(path, name="delivered_support_" + n,
                reset_xform_properties=False, track_contact_forces=True,
                contact_filter_prim_paths_expr=[self.bench.prim_path, *self.fixture_paths,
                                                *[self.paths[x] for x in others]],
                max_contact_count=8192, disable_stablization=False)
        self.observer_targets = list(dict.fromkeys(filters + self.fixture_paths + self.flower_leaf_paths))
        self.observer_views = {
            arm: RigidPrim(next(m.prim_path for m in config.robot.mounts if m.name == arm) + "/link[1-8]",
                           name="delivered_observer_" + arm, reset_xform_properties=False,
                           track_contact_forces=True, contact_filter_prim_paths_expr=self.observer_targets,
                           max_contact_count=8192, disable_stablization=False)
            for arm in config.observation_arms
        }
        self.colliders = [str(p.GetPath()) for p in self.stage.Traverse()
                          if str(p.GetPath()).startswith(self.root + "/")
                          and p.HasAPI(UsdPhysics.CollisionAPI)
                          and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()]
        planning_fixture_roots = fixture_roots + [self.paths[n] for n in self.fixed_names]
        backend.planning_fixture_paths = tuple(
            path for path in self.colliders
            if any(path.startswith(root + "/") or path == root for root in planning_fixture_roots))
        backend.planning_payloads = self.planning_payloads
        self.report_key = "household"
        backend.scene_report[self.report_key] = {
            "version": self.design.version, "mechanism": spec.mechanism,
            "delivery": "video_20261015", "scene_sha256": row["sha256"],
            "assets": [{"task_id": config.task_id, "path": row["path"], "sha256": row["sha256"]}],
            "parameters": spec.parameters, "limitations": spec.limitations,
            "initial_poses": self.initial, "tracked_paths": self.paths,
            "fixed_bodies": [self.paths[n] for n in self.fixed_names],
            "delivery_repairs": row.get("physical_repairs", {}),
            "initial_placement_perturbation": row.get('initial_placement_perturbation'),
            "object_attachment": False, "older_props_imported": bool(spec.parameters.get("stacked_book_names")),
            "arm_roles": config.arm_roles,
            "ground_truth_use": "independent scoring and collision planner; privileged smoke controller only when explicitly invoked",
            "physics_overrides": {"bottle_fixed_by_task_setup": config.task_id == "L3-01",
                                  "bottle_contacts": bottle_contacts,
                                  "marker_contact_groups_refreshed": marker_filters},
            "reset_auxiliary_dynamic_bodies": [str(body.prim_path) for body,_ in self.aux_bodies],
            "fixture_contact_filters":self.fixture_paths,
        }
        self.hook_contact = None
        if config.task_id == "L1-03":
            self.hook_contact = RigidPrim(self.paths["hat"], name="delivered_hook_support",
                reset_xform_properties=False, track_contact_forces=True,
                contact_filter_prim_paths_expr=["/World/Household/Fixture/Colliders/Hook0"],
                max_contact_count=2048, disable_stablization=False)

    def reset(self, seed):
        self.flower_leaf_time = None
        super().reset(seed)
        for view in (*self.flower_leaf_fingers.values(), *self.flower_leaf_supports.values()):
            view.initialize()
        if self.hook_contact is not None:
            self.hook_contact.initialize()

    def measure(self):
        rows = super().measure()
        if self.spec.parameters.get("required_stack_turn_rad"):
            forces = {arm: np.asarray(view.get_contact_force_matrix(dt=self.config.physics_dt_s))
                      for arm, view in self.arm_contacts.items()}
            for i, name in enumerate(self.manipulated):
                rows[name]["finger_force_vectors"] = {arm: force[:, i].tolist()
                                                      for arm, force in forces.items()}
        if self.flower_leaf_paths and self.flower_leaf_time != self.backend.time_s():
            # A bent leaf is still part of the manipulated flower. Preserve all
            # finger/support/observer evidence instead of making its contacts
            # disappear when splitting it from the stem's rigid body.
            rose = rows['rose']
            for arm, view in self.flower_leaf_fingers.items():
                forces = np.asarray(view.get_contact_force_matrix(dt=self.config.physics_dt_s))
                if forces.shape != (2, len(self.flower_leaf_paths), 3) or not np.isfinite(forces).all():
                    raise ValueError('Invalid passive flower leaf contact measurement')
                rose['fingers'][arm] = (np.asarray(rose['fingers'][arm]) +
                    np.linalg.norm(forces, axis=2).sum(axis=1)).tolist()
            nf = len(self.fixture_paths)
            for view in self.flower_leaf_supports.values():
                forces = np.asarray(view.get_contact_force_matrix(dt=self.config.physics_dt_s))[0]
                if not np.isfinite(forces).all():
                    raise ValueError('Nonfinite passive leaf support forces')
                rose['bench_force_n'] += float(np.linalg.norm(forces[0]))
                rose['fixture_force_n'] += float(np.linalg.norm(forces[1:1+nf].sum(axis=0)))
                for i, name in enumerate(self.flower_leaf_support_targets):
                    rose['pair_forces'][name] += float(np.linalg.norm(forces[1+nf+i]))
            rose['flexible_parts'] = {}
            for body, _ in self.aux_bodies:
                if str(body.prim_path) in self.flower_leaf_paths:
                    xyz, q = body.get_world_pose()
                    rose['flexible_parts'][str(body.prim_path).rsplit('/', 1)[-1]] = {
                        'pose': [*map(float, xyz), *map(float, q)]}
            self.flower_leaf_time = self.backend.time_s()
        if self.hook_contact is not None:
            contact = np.asarray(self.hook_contact.get_contact_force_matrix(dt=self.config.physics_dt_s))
            rows["hat"]["target_hook_vertical_force_n"] = float(max(0., contact[0, 0, 2]))
        return rows

    def planning_payloads(self):
        result = {arm: [] for arm in self.config.arms}
        for name, row in self.measure().items():
            if name in self.fixed_names:
                continue
            for arm, payloads in result.items():
                gripping = min(row["fingers"][arm]) > .05
                if self.spec.parameters.get("required_stack_turn_rad"):
                    from .stacked_books import grasp_contact
                    gripping = grasp_contact(row, arm)
                if gripping:
                    flower = self.config.task_id == 'L1-05' and name == 'rose'
                    points = flower_payload_surface(row['pose'], row.get('flexible_parts')).tolist() if flower else row['corners']
                    payloads.append({"surface_points_world": points,
                                     "source": "current delivered collider surfaces, measured bilateral contact"
                                     if flower else "current delivered body, measured bilateral contact"})
        return result


class DeliveredWipingScene(DeliveredScene):
    """Replace only the writing display with contact-removable delivered marks."""

    def __init__(self, backend, config, spec):
        from isaacsim.core.prims import RigidPrim
        from pxr import UsdGeom, Sdf
        from .resistant_wiping import ResistantWipeLedger
        import json
        from PIL import Image
        from zipfile import ZipFile
        from ...paths import PROJECT_ROOT

        super().__init__(backend, config, spec)
        stage = self.stage
        photo = UsdGeom.Mesh(stage.GetPrimAtPath(self.root + "/board/Visuals/PhotoWriting"))
        points = np.array(photo.GetPointsAttr().Get())
        # Sample the actual currently bound ink texture in the delivery; never
        # reuse the older board's differently scaled mask or create new glyphs.
        from pxr import UsdShade
        material, _ = UsdShade.MaterialBindingAPI(photo.GetPrim()).ComputeBoundMaterial()
        texture = stage.GetPrimAtPath(str(material.GetPath()) + "/Texture")
        asset = texture.GetAttribute("inputs:file").Get()
        internal = asset.resolvedPath.split("[", 1)[1].rsplit("]", 1)[0]
        package = PROJECT_ROOT / "assets/tuojing" / scenes()[config.scene_id]["path"]
        with ZipFile(package) as archive:
            rgba = np.array(Image.open(archive.open(internal)).convert("RGBA"))
        height, width = rgba.shape[:2]
        low, high = points.min(axis=0), points.max(axis=0)
        cells, weights = [], []
        pitch = 5
        for v in range(0, height, pitch):
            for u in range(0, width, pitch):
                weight = float(rgba[v:v+pitch, u:u+pitch, 3].sum())
                if weight < 127.5:
                    continue
                cells.append((low[0]+(u+pitch/2)/width*(high[0]-low[0]),
                              high[1]-(v+pitch/2)/height*(high[1]-low[1]),
                              float(low[2])+.00005))
                weights.append(weight)
        centres = np.array(cells)
        if not len(centres):
            raise ValueError("Delivered whiteboard has no visible ink")
        UsdGeom.Imageable(photo).MakeInvisible()
        self.ink = UsdGeom.Mesh.Define(stage, self.root + "/board/RemovableInk")
        half_x, half_y = (high[0]-low[0])/width*pitch/2, (high[1]-low[1])/height*pitch/2
        offsets = np.array([[-half_x,-half_y,0.],[half_x,-half_y,0.],
                            [half_x,half_y,0.],[-half_x,half_y,0.]])
        points = (centres[:,None,:]+offsets).reshape(-1,3)
        self.ink_indices = np.arange(len(points)).reshape(-1,4)
        self.ink.CreatePointsAttr(points.tolist())
        self.ink.CreateFaceVertexCountsAttr([4]*len(centres))
        self.ink.CreateFaceVertexIndicesAttr(self.ink_indices.ravel().tolist())
        self.ink.CreateSubdivisionSchemeAttr("none")
        self.ink.CreateDoubleSidedAttr(True)
        # Match the bound source handwriting material, including blue ink.
        ink_shader = UsdShade.Shader(stage.GetPrimAtPath(str(material.GetPath()) + '/Surface'))
        ink_color = ink_shader.GetInput('diffuseColor').Get()
        if ink_color is None:
            raise ValueError('Delivered handwriting needs an explicit ink color')
        self.ink.CreateDisplayColorAttr([ink_color])
        # Bind an explicit surface: RTX's fallback display-color material does
        # not reliably consume per-face displayOpacity.
        wear_material = UsdShade.Material.Define(stage, self.root + "/board/Looks/InkWear")
        wear_surface = UsdShade.Shader.Define(stage, str(wear_material.GetPath()) + "/Surface")
        wear_surface.CreateIdAttr("UsdPreviewSurface")
        wear_surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(ink_color)
        wear_surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.38)
        wear_surface.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.)
        wear_alpha = UsdShade.Shader.Define(stage, str(wear_material.GetPath()) + "/Alpha")
        wear_alpha.CreateIdAttr("UsdPrimvarReader_float")
        wear_alpha.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayOpacity")
        wear_alpha.CreateInput("fallback", Sdf.ValueTypeNames.Float).Set(1.)
        wear_alpha.CreateOutput("result", Sdf.ValueTypeNames.Float)
        wear_surface.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(wear_alpha.ConnectableAPI(), "result")
        wear_material.CreateSurfaceOutput().ConnectToSource(wear_surface.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(self.ink.GetPrim()).Bind(wear_material)
        self.centers = centres
        resistance = json.loads((PROJECT_ROOT / "configs/task_references/l3_wiping_resistance.json").read_text())
        self.wiping_revision = resistance.pop("revision")
        backend.scene_report["l2_07_ink_wear"] = {"revision": self.wiping_revision, **resistance}
        self.ledger = ResistantWipeLedger(spec.parameters["board_pose"], centres,
                                 pad_z=spec.parameters["pad_z"],
                                 half_extents=spec.parameters["pad_half_extents"], weights=weights,
                                 facing_direction=1, **resistance)
        self.ink.CreateDisplayOpacityPrimvar("uniform")
        self._render_ink()
        self.pad_contact = RigidPrim(self.paths["eraser"], name="delivered_wiping_contact",
             reset_xform_properties=False, track_contact_forces=True,
             contact_filter_prim_paths_expr=[self.root + "/board/Colliders/WritingSurface"],
             max_contact_count=4096, disable_stablization=False)

    def reset(self, seed):
        super().reset(seed)
        self.pad_contact.initialize()
        self.ledger.coverage.fill(0)
        self.ledger.dirty[:] = True
        self.ledger.previous = self.ledger.last_time = None
        self.ledger.previous_gain = 0.0
        self.ledger.last_normal_force_n = self.ledger.last_force_gain = 0.0
        self._render_ink()

    def _render_ink(self):
        alive = self.ink_indices[self.ledger.dirty]
        # Removing every face can leave a stale RTX mesh. Deactivate the prim
        # explicitly when clean; reset reactivates it before restoring faces.
        self.ink.GetPrim().SetActive(bool(len(alive)))
        if not len(alive):
            return
        self.ink.GetFaceVertexCountsAttr().Set([4] * len(alive))
        self.ink.GetFaceVertexIndicesAttr().Set(alive.ravel().tolist())
        self.ink.GetDisplayOpacityPrimvar().Set(self.ledger.opacity[self.ledger.dirty].tolist())

    def measure(self):
        rows = super().measure()
        row = rows["eraser"]
        force = np.asarray(self.pad_contact.get_contact_force_matrix(dt=self.config.physics_dt_s))
        if force.shape != (1, 1, 3) or not np.isfinite(force).all():
            raise ValueError("Invalid eraser/writing-surface contact force")
        # Force on the eraser, projected onto the outward board normal.
        normal_force = max(0.0, float(force[0, 0] @ self.ledger.normal))
        changed = self.ledger.update(self.backend.time_s(), row["pose"],
                                    pad_contact_n=normal_force,
                                    held=min(row["fingers"][self.spec.arms[0]]) > .05)
        if changed:
            self._render_ink()
        row["wiping"] = {**self.ledger.metrics(), "revision": self.wiping_revision,
                         "pad_board_contact_vector_n": force[0, 0].tolist()}
        return rows
