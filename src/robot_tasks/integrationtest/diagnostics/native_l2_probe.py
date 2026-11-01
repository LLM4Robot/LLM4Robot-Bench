"""Native-asset mechanism checks, never policy baselines or GPT success runs."""

import numpy as np


def advance(scene, seconds):
    for _ in range(round(seconds / scene.config.control_dt_s)):
        scene.backend.step(scene.config.control_dt_s)


def run(scene):
    result = {
        "task": scene.config.task_id,
        "gpt_called": False,
        "scope": "mechanism-only diagnostic; direct test actuation, not robot task success",
    }
    if scene.config.task_id == "L2-05":
        from pxr import UsdPhysics

        drive = UsdPhysics.DriveAPI(
            scene.stage.GetPrimAtPath(scene.root + "/organizer/Joints/Drawer1"),
            "linear",
        )
        old = {
            k: getattr(drive, "Get" + k + "Attr")().Get()
            for k in ("Stiffness", "Damping", "MaxForce", "TargetPosition")
        }
        start = scene.measure()["drawer"]["pose"][:3]
        try:
            drive.CreateStiffnessAttr(120.0)
            drive.CreateDampingAttr(6.0)
            drive.CreateMaxForceAttr(4.0)
            drive.CreateTargetPositionAttr(-0.085)
            advance(scene, 3)
        finally:
            for name, value in old.items():
                getattr(drive, "Create" + name + "Attr")(value)
        advance(scene, 1)
        final = scene.measure()["drawer"]
        axis = np.asarray(scene.spec.parameters["outward_axis_world"])
        travel = float((np.asarray(final["pose"][:3]) - start) @ axis)
        opening = float((np.asarray(final["pose"][:3])
                         - scene.spec.parameters["closed_pose"][:3]) @ axis)
        lateral = np.asarray(final["pose"][:3]) - start - travel * axis
        result.update(
            opening_m=opening,
            travel_since_start_m=travel,
            lateral_drift_m=float(np.linalg.norm(lateral)),
            released_speed_m_s=float(np.linalg.norm(final["linear_velocity"])),
        )
        if not 0.075 <= opening <= 0.093 or np.linalg.norm(lateral) > 0.002:
            raise RuntimeError(
                "Native drawer rail failed opening/release diagnostic: " + str(result)
            )
    elif scene.config.task_id == "L2-03":
        from ...tasks.bench.evaluation import relative
        from ...tasks.bench.pearl_pouring import cup_contains
        from pxr import UsdPhysics

        advance(scene, 1)
        rows = scene.measure()
        count = sum(cup_contains(relative(rows[n], rows["source_cup"])[:3, 3])
                    for n in scene.load_names)
        fixed = bool(UsdPhysics.RigidBodyAPI(
            scene.stage.GetPrimAtPath(scene.paths["receiver_cup"])
        ).GetKinematicEnabledAttr().Get())
        result.update(source_contained=count, receiver_fixed=fixed,
                      scope="Initial cup packing and fixed receiver check; no robot pouring")
        if count != scene.spec.parameters["count"] or not fixed:
            raise RuntimeError("Cup pouring initial layout failed: " + str(result))
    elif scene.config.task_id == "L2-01":
        from pxr import Gf, UsdPhysics

        from ...robots.transforms import pose_transform

        rows = scene.measure()
        for name in ("body", "cap"):
            p = rows[name]["pose"]
            cls = UsdPhysics.FixedJoint if name == "body" else UsdPhysics.PrismaticJoint
            joint = cls.Define(scene.stage, scene.root + "/Diagnostic_" + name)
            joint.CreateBody1Rel().SetTargets([scene.paths[name]])
            joint.CreateLocalPos0Attr(Gf.Vec3f(*p[:3]))
            joint.CreateLocalRot0Attr(Gf.Quatf(p[3], Gf.Vec3f(*p[4:])))
            joint.CreateLocalPos1Attr(Gf.Vec3f(0.0))
            if name == "cap":
                joint.CreateAxisAttr("Z")
                joint.CreateLowerLimitAttr(0.0)
                joint.CreateUpperLimitAttr(0.065)
                drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
                drive.CreateStiffnessAttr(100.0)
                drive.CreateDampingAttr(0.5)
                drive.CreateMaxForceAttr(4.0)
                drive.CreateTargetPositionAttr(0.05)
        advance(scene, 3)
        rows = scene.measure()
        b, c = (pose_transform(rows[n]["pose"]) for n in ("body", "cap"))
        tip = np.linalg.inv(c) @ b @ np.array([0.0, 0.0, 0.128, 1.0])
        gap = 0.105 - float(tip[2])
        result.update(
            axial_clearance_m=gap,
            test_force_limit_n=4.0,
            fixture="Fixed barrel and axially guided cap in diagnostic only",
        )
        if gap < 0.002:
            raise RuntimeError(
                "Native cap fit did not release under diagnostic pull: " + str(result)
            )
    elif scene.config.task_id == "L2-02":
        from pxr import Gf, UsdPhysics

        start = scene.measure()["middle_book"]["pose"][:3]
        joint = UsdPhysics.PrismaticJoint.Define(
            scene.stage, scene.root + "/DiagnosticPull"
        )
        joint.CreateBody1Rel().SetTargets([scene.paths["middle_book"]])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*start))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0))
        joint.CreateAxisAttr("X")
        joint.CreateLowerLimitAttr(0.0)
        joint.CreateUpperLimitAttr(0.14)
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "linear")
        drive.CreateStiffnessAttr(150.0)
        drive.CreateDampingAttr(8.0)
        drive.CreateMaxForceAttr(8.0)
        # Follow the task's extraction axis. This fixture only
        # checks sliding; it is not the bimanual task or its success evaluator.
        direction = float(scene.spec.parameters["outward_axis_world"][0])
        joint.CreateLowerLimitAttr(min(0.0, direction * 0.14))
        joint.CreateUpperLimitAttr(max(0.0, direction * 0.14))
        drive.CreateTargetPositionAttr(direction * 0.05)
        try:
            advance(scene, 4.0)
            row = scene.measure()["middle_book"]
            p = scene.spec.parameters
            travel = float((np.asarray(row["pose"][:3]) - start) @ p["outward_axis_world"])
            result.update(
                travel_m=travel,
                required_travel_m=p["minimum_travel_m"],
            )
            if travel < p["minimum_travel_m"] - 0.002:
                raise RuntimeError(
                    "Native book did not slide the required distance: " + str(result)
                )
        finally:
            joint.CreateJointEnabledAttr(False)
    else:
        raise ValueError("No native mechanism diagnostic for " + scene.config.task_id)
    result["passed"] = True
    return result
