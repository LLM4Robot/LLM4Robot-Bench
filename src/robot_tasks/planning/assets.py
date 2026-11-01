"""Derive planner geometry from immutable R5 meshes, using the configured TCP/limits."""

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

from ..robots.importer import prepare_urdf
from ..robots.kinematics import R5Kinematics
from ..robots.transforms import origin_transform

TCP_FRAME = "robot_tasks_tcp"
REVISION = "78fd485fa82d9b9a063fb4985e371814587e666a"


def build_assets(profile, directory, attachments=None):
    import trimesh
    from curobo.scene import Mesh

    kin = R5Kinematics(profile)
    identity = hashlib.sha256(
        profile.source.read_bytes()
        + profile.asset_lock.read_bytes()
        + Path(__file__).read_bytes()
        + json.dumps(attachments or {}, sort_keys=True).encode()
    ).hexdigest()[:16]
    output = directory.resolve() / identity
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "robot.json"
    if cache.is_file():
        return json.loads(cache.read_text()), kin
    urdf = prepare_urdf(profile, output / "resolved.urdf")
    root = ET.parse(urdf).getroot()
    spheres, fingers = {}, {}
    for link in root.findall("link"):
        name = link.get("name")
        collision = link.find("collision")
        node = collision.find("geometry/mesh")
        mesh = trimesh.load(node.get("filename"), force="mesh")
        mesh.apply_scale([float(x) for x in node.get("scale", "1 1 1").split()])
        origin = collision.find("origin")
        if origin is not None:
            mesh.apply_transform(
                origin_transform(
                    tuple(map(float, origin.get("xyz", "0 0 0").split())),
                    tuple(map(float, origin.get("rpy", "0 0 0").split())),
                )
            )
        if name == profile.base_link:
            centre = mesh.bounds.mean(0)
            rows = [
                {
                    "center": centre.tolist(),
                    "radius": float(np.linalg.norm(mesh.extents) / 2),
                }
            ]
        else:
            geometry = Mesh(
                name=name,
                vertices=mesh.vertices.tolist(),
                faces=mesh.faces.tolist(),
                pose=[0, 0, 0, 1, 0, 0, 0],
            )
            fit = geometry.get_bounding_spheres(num_spheres=12, surface_radius=0.001)
            rows = [{"center": s.pose[:3], "radius": s.radius} for s in fit]
        if name in ("link7", "link8"):
            joint = next(
                j for j in root.findall("joint") if j.find("child").get("link") == name
            )
            origin = joint.find("origin")
            fingers[name] = {
                "spheres": rows,
                "origin": origin_transform(
                    tuple(map(float, origin.get("xyz", "0 0 0").split())),
                    tuple(map(float, origin.get("rpy", "0 0 0").split())),
                ).tolist(),
                "axis": list(map(float, joint.find("axis").get("xyz").split())),
            }
        else:
            spheres[name] = rows
    # Finger collision spheres follow actual measured opening, expressed in link6.
    # They are NOT exaggerated to the fully-open swept envelope while holding.
    link6_static = deepcopy(spheres[profile.ee_link])
    wrist_spheres = {}
    for arm, parts in (attachments or {}).items():
        fitted = []
        for part in parts:
            mesh = Mesh(name=part["name"].replace("/", "_"), vertices=part["vertices"],
                        faces=part["faces"], pose=[0, 0, 0, 1, 0, 0, 0])
            count = 32 if part["name"].startswith("Mount/") else 12
            fitted.extend({"center": s.pose[:3], "radius": s.radius}
                          for s in mesh.get_bounding_spheres(num_spheres=count, surface_radius=.002))
        wrist_spheres[arm] = fitted
    wrist_count = max((len(rows) for rows in wrist_spheres.values()), default=0)
    for rows in wrist_spheres.values():
        rows.extend({"center": [0, 0, 0], "radius": -1.0} for _ in range(wrist_count-len(rows)))
    # A fixed-size sphere buffer supports different left/right attachment poses.
    spheres[profile.ee_link] += next(iter(wrist_spheres.values()), [])
    finger_count = sum(len(f["spheres"]) for f in fingers.values())
    spheres[profile.ee_link] += [
        {"center": [0, 0, 0], "radius": -1.0} for _ in range(finger_count)
    ]
    for node in list(root.findall("joint")):
        if node.get("name") in profile.gripper.joint_names:
            root.remove(node)
    for node in list(root.findall("link")):
        if node.get("name") in fingers:
            root.remove(node)
    ET.SubElement(root, "link", name=TCP_FRAME)
    fixed = ET.SubElement(root, "joint", name="tcp_fixed", type="fixed")
    ET.SubElement(fixed, "parent", link=profile.ee_link)
    ET.SubElement(fixed, "child", link=TCP_FRAME)
    r = kin.tcp[:3, :3]
    rpy = (
        math.atan2(r[2, 1], r[2, 2]),
        math.atan2(-r[2, 0], math.hypot(r[0, 0], r[1, 0])),
        math.atan2(r[1, 0], r[0, 0]),
    )
    ET.SubElement(
        fixed,
        "origin",
        xyz=" ".join(map(str, kin.tcp[:3, 3])),
        rpy=" ".join(map(str, rpy)),
    )
    arm_urdf = output / "arm.urdf"
    ET.ElementTree(root).write(arm_urdf, encoding="utf-8")
    # The anchored base never moves with a joint trajectory. Its conservative
    # bounding sphere protrudes below the physical base and can intersect the
    # mounting surface, falsely invalidating every start state in a lab scene.
    # Exclude only this fixed link from the active robot's world/self checks;
    # keep its geometry in `spheres` for the OTHER arm's obstacle model.
    links = [name for name in spheres if name != profile.base_link]
    ignore = {a: links[i + 1 : i + 3] for i, a in enumerate(links)}
    cfg = {
        "robot_cfg": {
            "kinematics": {
                "format_version": 2.0,
                "urdf_path": str(arm_urdf),
                "asset_root_path": str(output),
                "base_link": profile.base_link,
                "tool_frames": [TCP_FRAME],
                "collision_link_names": links,
                "collision_spheres": {name: spheres[name] for name in links},
                "collision_sphere_buffer": 0.0,
                "self_collision_ignore": ignore,
                "self_collision_buffer": {a: 0.0 for a in links},
                "mesh_link_names": links,
                "cspace": {
                    "joint_names": list(profile.joint_names),
                    "default_joint_position": list(
                        profile.mounts[0].home_positions_rad
                    ),
                    "null_space_weight": [1.0] * 6,
                    "cspace_distance_weight": [1.0] * 6,
                    "max_acceleration": 2.0,
                    "max_jerk": 30.0,
                },
            }
        }
    }
    result = {
        "config": cfg,
        "spheres": spheres,
        "finger_geometry": fingers,
        "link6_static": link6_static,
        "wrist_spheres": wrist_spheres,
        "metadata": {
            "solver": "cuRobo V2",
            "revision": REVISION,
            "tcp_frame": TCP_FRAME,
            "geometry": "URDF mesh-fitted spheres; approximate, not exact collision geometry",
            "self_collision": "nonadjacent links; immediate two neighbours excluded",
            "fixed_base_collision": "anchored base excluded from own motion checks; retained as other-arm obstacle",
            "finger_geometry": "updated from measured finger positions",
            "wrist_geometry": "live mounted USD housing/lenses/CAD bracket, link6-frame fitted spheres" if wrist_spheres else "no attachments supplied",
            "wrist_sphere_count_per_arm": wrist_count,
            "fallback": None,
            "asset_identity": identity,
        },
    }
    cache.write_text(json.dumps(result, indent=2))
    return result, kin
