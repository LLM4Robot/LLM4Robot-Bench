"""Physical wrist attachments and matching planner geometry from the live stage.

Optical cameras are not collision shapes. The delivered housing, lenses and
CAD mount are, and belong to the existing link6 body (no extra rigid body).
"""

import numpy as np


def configure_wrist_collisions(stage, profile, *, expected_shapes=5):
    from pxr import PhysxSchema, Usd, UsdGeom, UsdPhysics

    from ..planning.scene import collider_meshes

    geometry, report = {}, {}
    for mount in profile.mounts:
        link = stage.GetPrimAtPath(mount.prim_path + "/" + profile.ee_link)
        root = stage.GetPrimAtPath(str(link.GetPath()) + "/WristSensor")
        if not root:
            raise ValueError(f"Missing delivered wrist assembly: {mount.name}")
        shapes = []
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdGeom.Gprim):
                continue
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
            contact = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            contact.CreateContactOffsetAttr(.001)
            contact.CreateRestOffsetAttr(0.)
            if prim.IsA(UsdGeom.Mesh):
                # A single hull would fill the open CAD bracket.
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexDecomposition")
                decomp = PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(prim)
                decomp.CreateMaxConvexHullsAttr(32)
                decomp.CreateShrinkWrapAttr(True)
                decomp.CreateMinThicknessAttr(.0002)
            shapes.append(str(prim.GetPath()))
        if len(shapes) != expected_shapes:
            raise ValueError(f"Audit changed wrist asset before use: {mount.name}, {shapes}")
        # Reuse the same mesh export as fixture planning. Express it in link6,
        # after camera mounting adjustments, rather than assuming URDF includes it.
        from types import SimpleNamespace
        snapshot = collider_meshes(SimpleNamespace(
            world=SimpleNamespace(stage=stage), planning_legacy_bench=False,
            planning_fixture_paths=shapes))
        inverse = np.linalg.inv(np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(link)).T)
        geometry[mount.name] = []
        for shape in shapes:
            mesh = snapshot["mesh"][shape.replace("/", "_")]
            vertices = np.asarray(mesh["vertices"]) @ inverse[:3, :3].T + inverse[:3, 3]
            geometry[mount.name].append({"name": shape.split("/WristSensor/")[1],
                                         "vertices": vertices.tolist(), "faces": mesh["faces"]})
        report[mount.name] = {"colliders": shapes, "rigid_body": str(link.GetPath()),
                              "mount_collision": "convex decomposition of delivered CAD",
                              "sensor_collision": "native housing, face and lens primitives",
                              "source_asset_modified": False, "mass_inertia_unchanged": True}
    return geometry, report
