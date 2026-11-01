"""Manual cap/support contact regression; no model calls or task-success scoring.

Uses recorded release poses plus passive centre/rim drops. Each fixture gets a
fresh physics stage. The embedded pose is an intentional penetration stress
test, not a valid initial state for an experiment.
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from ...paths import PROJECT_ROOT
from ...robots.transforms import pose_transform, transform_pose


RECORDED_RELEASE = (
    .12610569596290588, 1.2200736999511719, .8903547525405884,
    .694839358329773, .06648162752389908, .7126677632331848, .069880411028862,
)
RECORDED_EMBEDDED = (
    .060572944581508636, 1.213922381401062, .9246077537536621,
    .09632924199104309, -.833308219909668, .15682242810726166, -.5212721824645996,
)


def cases():
    from ...tasks.l1.specs import pose

    yield "recorded_release", RECORDED_RELEASE
    yield "embedded_stress", RECORDED_EMBEDDED
    for name, centre, pitch in (
        ("centre_drop", (.19565, 1.30, .93), math.pi / 2),
        ("front_rim_drop", (.19565, 1.207, .94), math.pi / 2),
        ("corner_drop", (.266, 1.224, .94), math.pi / 4),
        ("mouth_down_drop", (.19565, 1.30, .94), 0),
    ):
        matrix = pose_transform(pose(*centre, pitch=pitch))
        matrix[:3, 3] -= matrix[:3, :3] @ np.array([0, 0, .128])
        yield name, transform_pose(matrix)


def penetration_m(points, triangles):
    """Vertical depth inside the closed, non-overhanging support shell.

    Evaluate the original visible surface, not the new physics approximation.
    This catches a motionless object sleeping inside the box as a failure.
    """
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    u, v = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2]
    det = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    valid = det > 1e-12  # Only outward/upward faces.
    a, b, c, u, v, det = (x[valid] for x in (a, b, c, u, v, det))
    depth = 0.0
    for point in points:
        offset = point[:2] - a[:, :2]
        s = (offset[:, 0] * v[:, 1] - offset[:, 1] * v[:, 0]) / det
        t = (u[:, 0] * offset[:, 1] - u[:, 1] * offset[:, 0]) / det
        inside = (s >= -1e-7) & (t >= -1e-7) & (s + t <= 1 + 1e-7)
        if point[2] >= 0 and inside.any():
            top = a[:, 2] + s * (b[:, 2] - a[:, 2]) + t * (c[:, 2] - a[:, 2])
            depth = max(depth, float(top[inside].max() - point[2]))
    return depth


def run_case(name, initial, *, fixed, hz, seconds):
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import FixedCuboid
    from isaacsim.core.prims import RigidPrim, SingleRigidPrim
    from isaacsim.core.utils.stage import create_new_stage
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    from ...asset_delivery import verified_prop
    from ...tasks.bench.marker_contacts import configure_marker_box_contacts
    from ...tasks.bench.native_l2_specs import support_box

    World.clear_instance()
    create_new_stage()
    dt = 1 / hz
    world = World(stage_units_in_meters=1, physics_dt=dt, rendering_dt=1 / 12,
                  backend="numpy", device="cpu",
                  sim_params={"use_gpu_pipeline": False, "use_fabric": False})
    physics = world.get_physics_context()
    physics.enable_gpu_dynamics(False)
    physics.set_solver_type("TGS")
    scene_api = PhysxSchema.PhysxSceneAPI.Apply(physics.get_current_physics_scene_prim())
    scene_api.CreateEnableCCDAttr(True)
    scene_api.CreateEnableStabilizationAttr(False)
    scene_api.CreateMinVelocityIterationCountAttr(0)
    stage = world.stage
    world.scene.add(FixedCuboid(
        prim_path="/World/Table", name="table", position=np.array([.2, 1.3, .73]),
        scale=np.array([2., 2., .1]), size=1.))
    box = stage.DefinePrim("/World/Box")
    box.GetReferences().AddReference(str(verified_prop("green_support_box", task_id="L2-01")[0]))
    box_transform = pose_transform(support_box()["pose"])
    UsdGeom.Xformable(box).MakeMatrixXform().Set(Gf.Matrix4d(box_transform.T.tolist()))
    UsdPhysics.RigidBodyAPI(box).CreateKinematicEnabledAttr(True)
    PhysxSchema.PhysxRigidBodyAPI.Apply(box).CreateEnableCCDAttr(False)
    marker = stage.DefinePrim("/World/Marker")
    marker.GetReferences().AddReference(str(verified_prop("pullable_marker", task_id="L2-01")[0]))
    UsdGeom.Xformable(marker).MakeMatrixXform().Set(Gf.Matrix4d(pose_transform(initial).T.tolist()))
    stage.GetPrimAtPath("/World/Marker/Body").SetActive(False)
    override = configure_marker_box_contacts(stage, "/World/Marker", "/World/Box") if fixed else None
    cap = world.scene.add(SingleRigidPrim("/World/Marker/Cap", name="cap"))
    contacts = RigidPrim(
        "/World/Marker/Cap", name="contacts", reset_xform_properties=False,
        track_contact_forces=True, contact_filter_prim_paths_expr=["/World/Box"],
        max_contact_count=4096, disable_stablization=False)

    shell = UsdGeom.Mesh(stage.GetPrimAtPath("/World/Box/Visuals/SteppedMoldedShell"))
    triangles = np.asarray(shell.GetPointsAttr().Get())[
        np.asarray(shell.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)]
    angles = np.linspace(0, 2 * math.pi, 48, endpoint=False)
    local = np.array([(0.0087 * math.cos(a), 0.0087 * math.sin(a), z)
                      for z in np.linspace(.105, .151, 9) for a in angles])
    box_inverse = np.linalg.inv(box_transform)
    world.reset()
    contacts.initialize()
    samples = []
    try:
        for step in range(round(seconds * hz)):
            world.step(render=False)
            if (step + 1) % round(hz / 12):
                continue
            xyz, q = cap.get_world_pose()
            transform = box_inverse @ pose_transform((*xyz, *q))
            points = local @ transform[:3, :3].T + transform[:3, 3]
            samples.append({
                "time_s": (step + 1) * dt,
                "pose": [*map(float, xyz), *map(float, q)],
                "speed_m_s": float(np.linalg.norm(cap.get_linear_velocity())),
                "angular_speed_rad_s": float(np.linalg.norm(cap.get_angular_velocity())),
                "box_force_n": float(np.linalg.norm(contacts.get_contact_force_matrix(dt=dt))),
                "penetration_m": penetration_m(points, triangles),
            })
        late = samples[-12:]
        metrics = {key: max(s[key] for s in late)
                   for key in ("box_force_n", "speed_m_s", "penetration_m")}
        finite = all(np.isfinite(s["pose"] + [s["speed_m_s"], s["box_force_n"]]).all()
                     for s in samples)
        # Mechanical regression tolerances, never task-success conditions.
        passed = (finite and metrics["penetration_m"] < .0008
                  and metrics["box_force_n"] < .5 and metrics["speed_m_s"] < .03)
        return {"case": name, "fixed": fixed, "passed": bool(passed), "hz": hz,
                "last_second_max": metrics, "override": override, "samples": samples}
    finally:
        world.stop()
        world.clear()
        World.clear_instance()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "build/marker_contact")
    parser.add_argument("--baseline", action="store_true", help="Also demonstrate the original embedded failure")
    parser.add_argument("--hz", type=int, default=1920)
    parser.add_argument("--seconds", type=float, default=5.)
    args = parser.parse_args()
    if args.hz <= 0 or args.hz % 12 or args.seconds < 2:
        parser.error("Use a positive multiple of 12 Hz and at least 2 seconds")
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from ...runtime import configure_windows_storage
    from ...backends.settings import SimulationSettings

    storage = configure_windows_storage()
    from isaacsim import SimulationApp

    settings = SimulationSettings(asset_directory=PROJECT_ROOT / "assets", headless=True,
                                  width=640, height=480).application_config()
    if storage:
        settings["extra_args"].extend(storage.kit_arguments())
    app = SimulationApp(settings)
    report = {"scope": "cap/box physical regression only; not GPT task success",
              "gpt_called": False, "cases": []}
    try:
        if args.baseline:
            report["baseline"] = run_case("embedded_stress", RECORDED_EMBEDDED,
                                          fixed=False, hz=args.hz, seconds=args.seconds)
        for name, pose in cases():
            row = run_case(name, pose, fixed=True, hz=args.hz, seconds=args.seconds)
            report["cases"].append(row)
            print("MARKER_CONTACT", name, row["passed"], row["last_second_max"], flush=True)
        report["passed"] = all(r["passed"] for r in report["cases"])
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return 0 if report["passed"] else 2
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
