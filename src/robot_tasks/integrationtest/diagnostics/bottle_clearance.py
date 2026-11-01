"""Offline L3-01 workspace diagnostic; never imported by the robot policy.

Run in the planner environment (numpy/scipy/trimesh). Numerical IK only proposes
joint-limit-respecting inspection poses; it is NOT the production planner.
Conservative CAD/URDF boxes check sampled inter-arm and camera/prop clearance.
This is neither continuous path certification nor physical grasp/task success.
"""

import argparse
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from ...cameras.profile import load_camera_rig
from ...paths import PROJECT_ROOT
from ...robots.control import width_to_joints
from ...robots.kinematics import R5Kinematics
from ...robots.transforms import (
    origin_transform,
    pose_residual,
    pose_transform,
    transform_pose,
)
from ...robots.urdf import read_urdf
from ...registry import resolve_task
from ...tasks.bench.spec import build_spec


def box_vertices(transform, low, high):
    points = np.array(
        [
            (x, y, z)
            for x in (low[0], high[0])
            for y in (low[1], high[1])
            for z in (low[2], high[2])
        ]
    )
    return points @ transform[:3, :3].T + transform[:3, 3]


def separation(first, second):
    """SAT separating-plane margin for two orthogonal boxes, in metres.

    Positive is a conservative lower bound on distance. Nonpositive means box
    overlap, which does not establish collision between the enclosed meshes.
    """
    a = np.array([first[i] - first[0] for i in (4, 2, 1)])
    b = np.array([second[i] - second[0] for i in (4, 2, 1)])
    axes = np.concatenate((a, b, np.cross(a[:, None], b).reshape(-1, 3)))
    norms = np.linalg.norm(axes, axis=1)
    axes = axes[norms > 1e-10] / norms[norms > 1e-10, None]
    x, y = first @ axes.T, second @ axes.T
    return float(np.maximum(x.min(0) - y.max(0), y.min(0) - x.max(0)).max())


class Geometry:
    def __init__(self, config):
        import trimesh

        self.profile = config.robot
        self.kin = R5Kinematics(config.robot)
        self.urdf = read_urdf(self.profile.asset_path, dict(self.profile.package_roots))
        self.mesh_bounds = {}
        root = ET.parse(self.profile.asset_path).getroot()
        for link in root.findall("link"):
            collision = link.find("collision")
            node = collision.find("geometry/mesh")
            uri = node.get("filename")
            if uri.startswith("package://"):
                package, relative = uri[10:].split("/", 1)
                path = dict(self.profile.package_roots)[package] / relative
            else:
                path = self.profile.asset_path.parent / uri
            mesh = trimesh.load(path, force="mesh")
            mesh.apply_scale(tuple(map(float, node.get("scale", "1 1 1").split())))
            o = collision.find("origin")
            t = (
                origin_transform(
                    tuple(map(float, o.get("xyz", "0 0 0").split())),
                    tuple(map(float, o.get("rpy", "0 0 0").split())),
                )
                if o is not None
                else np.eye(4)
            )
            self.mesh_bounds[link.get("name")] = (t, mesh.bounds)
        self.rig = load_camera_rig(
            config.source.parent / config.raw["simulation"]["camera_profile"]
        )
        assets = PROJECT_ROOT / "assets/cameras"
        self.housing_dims = np.array(
            json.loads((assets / "gemini305.json").read_text())["dimensions_m"]
        )
        self.bracket_dims = np.array(
            json.loads((assets / "r5_wrist_camera_mount.json").read_text())[
                "dimensions_m"
            ]
        )
        self.bracket_pose = pose_transform(
            json.loads((assets / "gemini305_r5_mounted.json").read_text())[
                "bracket_centred_cad_in_assembly_wxyz"
            ]
        )

    def boxes(self, arm, q, width):
        positions = dict(zip(self.profile.joint_names, q))
        positions.update(
            zip(self.profile.gripper.joint_names, width_to_joints(self.profile, width))
        )
        transforms = {self.profile.base_link: self.kin.bases[arm]}
        pending = list(self.urdf.joints)
        while pending:
            for joint in list(pending):
                if joint.parent not in transforms:
                    continue
                t = origin_transform(joint.origin_xyz, joint.origin_rpy)
                motion = np.eye(4)
                value = positions.get(joint.name, 0.0)
                if joint.kind == "prismatic":
                    motion[:3, 3] = np.asarray(joint.axis) * value
                elif joint.kind in ("revolute", "continuous"):
                    from ...robots.transforms import axis_rotation

                    motion[:3, :3] = axis_rotation(joint.axis, value)
                transforms[joint.child] = transforms[joint.parent] @ t @ motion
                pending.remove(joint)
        boxes = {
            name: box_vertices(transforms[name] @ t, *bounds)
            for name, (t, bounds) in self.mesh_bounds.items()
        }
        camera = next(c for c in self.rig.cameras if c.parent == arm)
        optical = transforms[self.profile.ee_link] @ pose_transform(
            camera.optical_pose_wxyz
        )
        assembly = optical @ pose_transform(camera.asset_pose_wxyz)
        boxes["camera"] = box_vertices(
            assembly, -self.housing_dims / 2, self.housing_dims / 2
        )
        boxes["bracket"] = box_vertices(
            assembly @ self.bracket_pose, -self.bracket_dims / 2, self.bracket_dims / 2
        )
        return boxes, optical

    def solve(self, arm, pose, seed):
        from scipy.optimize import least_squares

        goal = pose_transform(pose)

        def residual(q):
            r = pose_residual(pose_transform(self.kin.forward(q, arm)), goal)
            r[3:] *= 0.2
            return r

        rng = np.random.default_rng(0)
        guesses = [seed, *rng.uniform(self.kin.lower, self.kin.upper, (24, 6))]
        for guess in guesses:
            r = least_squares(
                residual, guess, bounds=(self.kin.lower, self.kin.upper), max_nfev=150
            )
            if np.linalg.norm(residual(r.x)) < 1e-5:
                return r.x
        raise RuntimeError(f"No diagnostic IK pose: {arm} {pose}")


def check(output):
    from .l2_clearance import solve_clear

    _, config = resolve_task("L3-01")
    geometry = Geometry(config)
    spec = build_spec("L3-01")
    body, cap = spec.design.articles
    # Bounds for the free body/neck; only the support fingers may touch the body.
    obstacles = [
        (
            "body",
            box_vertices(
                pose_transform(body.initial),
                (-0.031, -0.023, 0.001),
                (0.031, 0.023, 0.157),
            ),
        ),
        (
            "neck",
            box_vertices(
                pose_transform(body.initial),
                (-0.019, -0.019, 0.157),
                (0.019, 0.019, 0.1865),
            ),
        ),
    ]
    seeds = {m.name: np.asarray(m.home_positions_rad) for m in config.robot.mounts}
    support_arm, cap_arm = spec.parameters["body_arm"], spec.parameters["cap_arm"]
    # Search a small set of diagnostic hand orientations. Only the actual
    # intended finger/body contact is exempt; cameras and the other hand are not.
    failures = []
    rows = []
    for height, roll in ((0.055, 0.0), (0.125, math.pi), (0.145, math.pi)):
        support_goal = origin_transform(
            (body.initial[0] - 0.045, body.initial[1], body.initial[2] + height),
            (roll, 0, 0),
        )
        try:
            support_q, support_worst, support_table = solve_clear(
                geometry,
                support_goal,
                seeds[support_arm],
                0.046,
                obstacles,
                arm=support_arm,
                table_z=0.78,
                contact_pairs=(("link7", "body"), ("link8", "body")),
            )
        except RuntimeError as exc:
            failures.append(str(exc))
            continue
        support_boxes, _ = geometry.boxes(support_arm, support_q, 0.046)
        paired_obstacles = obstacles + [
            (support_arm + "/" + name, shape) for name, shape in support_boxes.items()
        ]
        for angle in (math.pi / 3, math.pi / 2, -math.pi / 3, -math.pi / 2, math.pi):
            center = (cap.initial[0], cap.initial[1], cap.initial[2] + 0.200)
            try:
                q, worst, table_margin = solve_clear(
                    geometry,
                    origin_transform(center, (0, math.pi / 2, angle)),
                    seeds[cap_arm],
                    0.042,
                    paired_obstacles,
                    arm=cap_arm,
                    table_z=0.78,
                )
            except RuntimeError as exc:
                failures.append(str(exc))
                continue
            rows.append(
                {
                    "yaw_rad": angle,
                    "joints": q.tolist(),
                    "fixture_margin_m": worst[2],
                    "table_margin_m": table_margin,
                }
            )
            break
        if rows:
            break
    if not rows:
        raise RuntimeError("No sampled bimanual clearance fixture: " + str(failures))
    result = {
        "passed": True,
        "gpt_called": False,
        "version": spec.design.version,
        "scope": "sampled bimanual support/cap poses and inter-arm clearance; not physical grip or continuous path certification",
        "support": {
            "arm": support_arm,
            "joints": support_q.tolist(),
            "table_margin_m": support_table,
            "fixture_margin_m": support_worst[2],
        },
        "cap_arm": cap_arm,
        "samples": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports/native_bottle_clearance.json",
    )
    args = parser.parse_args()
    try:
        report = check(args.output)
    except Exception as exc:
        report = {"passed": False, "error": str(exc), "gpt_called": False}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)
    raise SystemExit(0 if report["passed"] else 2)
