"""Native FEM contact/release diagnostic, not a robot or model task solve."""

import numpy as np


def run(scene):
    from isaacsim.core.prims import SingleRigidPrim
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    initial = scene.measure()["orange"]
    floor = scene.spec.parameters["support_z"]
    ztop = initial["collision_bounds_max"][2]
    if abs(initial["collision_bounds_min"][2] - floor) > 0.0015:
        raise RuntimeError(
            "Native soft resting contact disagrees with table: " + str(initial)
        )
    x, y = initial["centroid"][:2]
    path = scene.root + "/DiagnosticPlaten"
    cube = UsdGeom.Cube.Define(scene.stage, path)
    cube.CreateSizeAttr(1.0)
    UsdGeom.XformCommonAPI(cube).SetScale(Gf.Vec3f(0.075, 0.06, 0.01))
    UsdGeom.XformCommonAPI(cube).SetTranslate(Gf.Vec3d(x, y, ztop + 0.015))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    api = PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
    api.CreateContactOffsetAttr(0.002)
    api.CreateRestOffsetAttr(0.0)
    UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim()).CreateKinematicEnabledAttr(True)
    # No attachment to the soft body. Only the test platen receives poses.
    scene.backend.step(scene.config.control_dt_s)
    press = SingleRigidPrim(path, name="diagnostic_platen")
    press.initialize()
    h = ztop - floor
    rows = [initial]
    for start, end, duration in (
        (0.0, 0.2 * h, 1.0),
        (0.2 * h, 0.2 * h, 0.25),
        (0.2 * h, 0.0, 1.0),
    ):
        for i in range(round(duration / scene.config.control_dt_s)):
            f = (i + 1) / round(duration / scene.config.control_dt_s)
            dz = start + (end - start) * f * f * (3 - 2 * f)
            press.set_world_pose(position=np.array([x, y, ztop + 0.005 - dz]))
            scene.backend.step(scene.config.control_dt_s)
        rows.append(scene.measure()["orange"])
    press.set_world_pose(position=np.array([x, y, ztop + 0.05]))
    for _ in range(12):
        scene.backend.step(scene.config.control_dt_s)
    rows.append(scene.measure()["orange"])
    span = lambda r: r["collision_bounds_max"][2] - r["collision_bounds_min"][2]
    compressed = min(span(r) for r in rows[1:3])
    ratio = compressed / span(initial)
    out = {
        "task": "L2-04",
        "gpt_called": False,
        "scope": "Quasi-static platen compression/release; no robot grasp or task success",
        "samples": rows,
        "compression_height_ratio": ratio,
        "recovery_height_ratio": span(rows[-1]) / span(initial),
    }
    if not 0.55 < ratio < 0.96 or not all(r["minimum_volume_ratio"] > 0 for r in rows):
        raise RuntimeError("Native soft compression response invalid: " + str(out))
    if any(abs(r["collision_bounds_min"][2] - floor) > 0.0015 for r in rows):
        raise RuntimeError("Soft-floor contact penetration/hover exceeded 1.5 mm")
    out["passed"] = True
    return out
