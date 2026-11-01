"""Native GPU FEM orange; no rigid proxy, attachments or rupture scene swapping."""

import numpy as np

from ...asset_delivery import verified_prop
from ...types import InvalidSimulationState
from ..l1.scene import L1Scene
from .orange_evaluation import (
    FINGER_ARM_ORDER, arm_finger_distances, deformation_metrics, motion_metrics,
)


class OrangeScene:
    root = "/World/L2"
    object_name = "orange"
    set_pose = L1Scene.set_pose
    add_part = L1Scene.add_part

    def __init__(self, backend, config, spec):
        from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade

        if not backend.settings.gpu_dynamics:
            raise ValueError(
                "L2-04 native orange requires simulation.gpu_dynamics=true"
            )
        self.backend, self.config, self.spec, self.design = (
            backend,
            config,
            spec,
            spec.design,
        )
        self.stage = backend.world.stage
        self.colliders = []
        self.path = self.root + "/orange"
        from ...video_delivery import enabled, scenes
        latest = enabled(config)
        if latest:
            scene = scenes()[config.task_id]
            evidence = {"delivery": "video_20261015", "path": scene["path"], "sha256": scene["sha256"]}
            prim = self.stage.GetPrimAtPath(self.path)
        else:
            asset, evidence = verified_prop("orange_segment", task_id=config.task_id)
            prim = self.stage.DefinePrim(self.path)
            prim.GetReferences().AddReference(str(asset))
        self.placement = self.design.articles[0].initial
        if not latest:
            self.set_pose(self.path, self.placement)
        pedestal_path = self.root + "/Pedestal"
        if latest:
            pedestal_evidence = {**evidence, "prim": pedestal_path}
            pedestal = self.stage.GetPrimAtPath(pedestal_path)
        else:
            pedestal_asset, pedestal_evidence = verified_prop(
                spec.parameters["pedestal_asset"], task_id=config.task_id
            )
            pedestal = self.stage.DefinePrim(pedestal_path)
            pedestal.GetReferences().AddReference(str(pedestal_asset))
        if not latest:
            self.set_pose(pedestal_path, spec.parameters["pedestal_pose"])
        # Passive fixed support only; never attach or drive the orange.
        UsdPhysics.RigidBodyAPI(pedestal).CreateKinematicEnabledAttr(True)
        PhysxSchema.PhysxRigidBodyAPI.Apply(pedestal).CreateEnableCCDAttr(False)
        pedestal_colliders = tuple(
            str(p.GetPath())
            for p in Usd.PrimRange(pedestal)
            if p.HasAPI(UsdPhysics.CollisionAPI)
        )
        if not pedestal_colliders:
            raise ValueError("Native orange pedestal has no collision geometry")
        for part in self.design.fixtures:
            self.add_part(self.root + "/Fixture/" + part.name, part)
        self.visual = self.stage.GetPrimAtPath(self.path + "/Visual")
        # The delivered FEM mesh has photo subsets but no mesh-level material.
        # RTX's dynamic mesh path otherwise displays the unbound grey fallback.
        # Keep both native photo materials; remove only non-rendering lobe groups
        # from this session to avoid overlapping GeomSubset face assignments.
        photo = UsdShade.Material(self.stage.GetPrimAtPath(self.path + "/Looks/FrontPhoto"))
        if not photo:
            raise ValueError("Delivered orange front-photo material is missing")
        UsdShade.MaterialBindingAPI.Apply(self.visual).Bind(photo)
        hidden_groups = []
        for child in self.visual.GetChildren():
            if child.IsA(UsdGeom.Subset) and child.GetAttribute("familyName").Get() == "anatomicalLobes":
                hidden_groups.append(child.GetName())
                child.SetActive(False)
        self.sim = self.stage.GetPrimAtPath(self.path + "/Simulation")
        self.collision = self.stage.GetPrimAtPath(self.path + "/Collision")
        from .orange_physics import apply

        physics = apply(self.stage, self.path, config.raw.get("orange_physics", {}))
        self.rest = np.asarray(
            self.sim.GetAttribute("omniphysics:restShapePoints").Get(), dtype=float
        )
        self.tets = np.asarray(
            self.sim.GetAttribute("omniphysics:restTetVtxIndices").Get(), dtype=int
        )
        r = self.rest[self.tets]
        edges = np.stack([r[:, i] - r[:, 0] for i in (1, 2, 3)], axis=2)
        self.rest_inverse = np.linalg.inv(edges)
        self.initial = {"orange": self.placement}
        from .surface_distance import link_triangles

        self.finger_paths, self.finger_surfaces = [], []
        bounds = UsdGeom.BBoxCache(0.0, ["default", "render"])
        for arm in FINGER_ARM_ORDER:
            mount = next(m for m in config.robot.mounts if m.name == arm)
            for link in ("link7", "link8"):
                path = mount.prim_path + "/" + link
                self.finger_paths.append(path)
                self.finger_surfaces.append(link_triangles(self.stage, path))
        self.previous = self.cached = None
        self.observer_bounds = {}
        for arm in config.observation_arms:
            mount = next(m for m in config.robot.mounts if m.name == arm)
            parts = []
            for index in range(1, 9):
                path = mount.prim_path + f"/link{index}"
                b = bounds.ComputeUntransformedBound(
                    self.stage.GetPrimAtPath(path)
                ).ComputeAlignedRange()
                parts.append((path, np.asarray(b.GetMin()), np.asarray(b.GetMax())))
            self.observer_bounds[arm] = parts
        backend.planning_fixture_paths = pedestal_colliders
        backend.planning_payloads = self.planning_payloads
        backend.scene_report["household"] = {
            "version": self.design.version,
            "mechanism": spec.mechanism,
            "assets": [evidence, pedestal_evidence],
            "pedestal": {
                "pose": spec.parameters["pedestal_pose"],
                "fixed": True,
                "native_geometry": True,
                "source_asset_modified": False,
            },
            "limitations": spec.limitations,
            "parameters": spec.parameters,
            "initial_poses": self.initial,
            "native_materials_preserved": all(
                physics["native"][key] == physics["effective"][key]
                for key in (
                    "dynamic_friction",
                    "youngs_modulus_pa",
                    "poissons_ratio",
                    "elasticity_damping",
                )
            ),
            "visual_material_adapter": {
                "mesh_fallback": "native FrontPhoto",
                "native_photo_subsets_preserved": True,
                "nonrendering_lobe_groups_hidden": hidden_groups,
                "source_asset_modified": False,
            },
            "orange_physics": physics,
            "object_attachment": False,
            "contact_overrides_m": {
                "contact_offset": physics["effective"]["contact_offset_m"],
                "rest_offset": physics["effective"]["rest_offset_m"],
            },
            "rupture_demo_executed": False,
            "soft_contact_measurement": "FEM collision vertices to delivered finger triangle surfaces, distances capped at 10 mm; unsigned proximity, not force or grasp proof",
            "deformation": "95th-percentile principal element strain, rigid-motion invariant; engineering reference only",
        }

    def reset(self, seed):
        # A new episode constructs a fresh stage. No state-switch demo or
        # mid-episode vertex rewrite is used to recover from manipulation.
        self.backend.reset(seed)
        self.previous = self.cached = None

    def fem_points(self):
        return (np.asarray(self.sim.GetAttribute("points").Get(), dtype=float),
                np.asarray(self.collision.GetAttribute("points").Get(), dtype=float))

    def measure(self):
        from omni.physx import get_physx_interface
        from pxr import UsdGeom

        now = self.backend.time_s()
        if self.cached is not None and self.previous[0] == now:
            return self.cached
        get_physx_interface().update_transformations(False, True, True, False)
        points = np.asarray(self.visual.GetAttribute("points").Get(), dtype=float)
        simulation, collision = self.fem_points()
        cache = UsdGeom.XformCache()
        t = np.asarray(cache.GetLocalToWorldTransform(self.visual)).T
        world = points @ t[:3, :3].T + t[:3, 3]
        ct = np.asarray(cache.GetLocalToWorldTransform(self.collision)).T
        physical = collision @ ct[:3, :3].T + ct[:3, 3]
        if (
            world.shape[0] < 4
            or physical.shape[0] < 4
            or not np.isfinite(world).all()
            or not np.isfinite(simulation).all()
            or not np.isfinite(physical).all()
        ):
            raise InvalidSimulationState("Missing/nonfinite native FEM readback")
        motion = motion_metrics(
            world,
            world if self.previous is None else self.previous[1],
            1.0 if self.previous is None else now - self.previous[0],
        )
        distances = []
        from .surface_distance import surface_clearance

        for path, triangles in zip(self.finger_paths, self.finger_surfaces):
            finger = np.asarray(
                cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(path))
            ).T
            local = (physical - finger[:3, 3]) @ finger[:3, :3]
            distances.append(surface_clearance(local, triangles))
        row = {
            "bounds_min": world.min(axis=0).tolist(),
            "bounds_max": world.max(axis=0).tolist(),
            "collision_bounds_min": physical.min(axis=0).tolist(),
            "collision_bounds_max": physical.max(axis=0).tolist(),
            "visual_world_transform": t.tolist(),
            "centroid": world.mean(axis=0).tolist(),
            **motion,
            "finger_distances_m": distances,
            **deformation_metrics(self.rest_inverse, simulation, self.tets),
        }
        row["observer_proximity"] = {}
        for arm, parts in self.observer_bounds.items():
            nearest = float("inf")
            for path, lo, hi in parts:
                transform = np.asarray(
                    cache.GetLocalToWorldTransform(self.stage.GetPrimAtPath(path))
                ).T
                local = (physical - transform[:3, 3]) @ transform[:3, :3]
                distance = np.linalg.norm(
                    np.maximum(np.maximum(lo - local, local - hi), 0.0), axis=1
                ).min()
                nearest = min(nearest, float(distance))
            row["observer_proximity"][arm] = {
                "mode": "soft_geometry_proximity",
                "distance_m": nearest,
                "scope": "soft collision surface to observer link bounds; contact ambiguous",
            }
        self.previous, self.cached = (now, world.copy()), {self.object_name: row}
        return self.cached

    def observer_evidence(self):
        return self.measure()[self.object_name]["observer_proximity"]

    def planning_payloads(self):
        row = self.measure()[self.object_name]
        payloads = {a: [] for a in self.config.arms}
        arm = self.spec.arms[0]
        if (
            all(d < 0.003 for d in arm_finger_distances(row, arm))
            and row["bounds_min"][2] > self.spec.parameters["support_z"] + 0.003
        ):
            from itertools import product

            payloads[arm].append(
                {
                    "surface_points_world": list(
                        product(*zip(row["bounds_min"], row["bounds_max"]))
                    ),
                    "source": "measured FEM bounds near fingers; conservative collision proxy, not force evidence",
                }
            )
        return payloads
