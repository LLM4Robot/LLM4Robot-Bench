"""Read robot descriptions without ROS, Isaac Sim, or mesh-processing imports.

This checks the description, not its dynamic accuracy or hardware calibration.
"""

from dataclasses import dataclass
import math
from pathlib import Path
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class Joint:
    name: str
    kind: str
    parent: str
    child: str
    lower: float | None
    upper: float | None
    effort: float | None
    velocity: float | None
    mimic: str | None
    origin_xyz: tuple[float, ...] = (0.0, 0.0, 0.0)
    origin_rpy: tuple[float, ...] = (0.0, 0.0, 0.0)
    axis: tuple[float, ...] = (1.0, 0.0, 0.0)


@dataclass(frozen=True)
class Description:
    name: str
    root_link: str
    links: tuple[str, ...]
    joints: tuple[Joint, ...]
    meshes: tuple[Path, ...]

    def joint(self, name: str) -> Joint:
        for joint in self.joints:
            if joint.name == name:
                return joint
        raise ValueError(f"Joint does not exist in URDF: {name}")

    def chain(self, base: str, tip: str) -> tuple[str, ...]:
        if base not in self.links or tip not in self.links:
            raise ValueError(f"Unknown link in chain {base} -> {tip}")
        by_child = {joint.child: joint for joint in self.joints}
        result = []
        while tip != base:
            if tip not in by_child:
                raise ValueError(f"No chain from {base} to requested tip")
            joint = by_child[tip]
            if joint.kind != "fixed":
                result.append(joint.name)
            tip = joint.parent
        return tuple(reversed(result))


def _number(text: str | None, label: str) -> float:
    try:
        value = float(text)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: expected a number") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label}: expected a finite number")
    return value


def _name(node: ET.Element, attribute: str, label: str) -> str:
    value = node.get(attribute, "")
    if not value or value.strip() != value:
        raise ValueError(f"{label}: missing or invalid {attribute}")
    return value


def _mesh_path(uri: str, source: Path, packages: dict[str, Path]) -> Path:
    if uri.startswith("package://"):
        package, separator, relative = uri[len("package://"):].partition("/")
        if not separator or package not in packages:
            raise ValueError(f"Unmapped mesh package: {uri}")
        root = packages[package].resolve()
        target = (root / relative).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"Mesh escapes its package: {uri}")
        return target
    if "://" in uri:
        raise ValueError(f"Unsupported mesh URI: {uri}; use a package or relative path")
    return (source.parent / uri).resolve()


def read_urdf(source: Path, packages: dict[str, Path]) -> Description:
    try:
        root = ET.parse(source).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Invalid URDF XML: {source}: {exc}") from exc
    if root.tag != "robot":
        raise ValueError("URDF root must be <robot>")
    name = _name(root, "name", "robot")
    links = tuple(_name(node, "name", "link") for node in root.findall("link"))
    if not links or len(set(links)) != len(links):
        raise ValueError("URDF must have uniquely named links")
    joints = []
    for node in root.findall("joint"):
        joint_name = _name(node, "name", "joint")
        kind = _name(node, "type", joint_name)
        if kind not in {"fixed", "revolute", "continuous", "prismatic"}:
            raise ValueError(f"{joint_name}: unsupported joint type {kind}")
        parent, child = node.find("parent"), node.find("child")
        if parent is None or child is None:
            raise ValueError(f"{joint_name}: missing parent/child")
        parent_name = _name(parent, "link", joint_name)
        child_name = _name(child, "link", joint_name)
        if parent_name not in links or child_name not in links:
            raise ValueError(f"{joint_name}: parent/child references an unknown link")
        lower = upper = effort = velocity = None
        axis_values = (1.0, 0.0, 0.0)
        if kind != "fixed":
            limit = node.find("limit")
            if limit is None:
                raise ValueError(f"{joint_name}: missing limits")
            effort = _number(limit.get("effort"), f"{joint_name}.effort")
            velocity = _number(limit.get("velocity"), f"{joint_name}.velocity")
            if effort <= 0 or velocity <= 0:
                raise ValueError(f"{joint_name}: effort and velocity limits must be positive")
            if kind != "continuous":
                lower = _number(limit.get("lower"), f"{joint_name}.lower")
                upper = _number(limit.get("upper"), f"{joint_name}.upper")
                if lower >= upper:
                    raise ValueError(f"{joint_name}: lower limit must be below upper limit")
            axis = node.find("axis")
            values = (axis.get("xyz", "1 0 0") if axis is not None else "1 0 0").split()
            if len(values) != 3 or sum(_number(v, f"{joint_name}.axis") ** 2 for v in values) == 0:
                raise ValueError(f"{joint_name}: invalid joint axis")
            axis_values = tuple(_number(v, f"{joint_name}.axis") for v in values)
        origin = node.find("origin")
        origin_vectors = []
        for key in ("xyz", "rpy"):
            values = (origin.get(key, "0 0 0") if origin is not None else "0 0 0").split()
            if len(values) != 3:
                raise ValueError(f"{joint_name}.origin.{key} must have three components")
            origin_vectors.append(tuple(_number(v, f"{joint_name}.origin.{key}") for v in values))
        mimic = node.find("mimic")
        joints.append(Joint(joint_name, kind, parent_name, child_name, lower, upper,
                            effort, velocity, None if mimic is None else _name(mimic, "joint", joint_name),
                            *origin_vectors, axis_values))
    names = {joint.name for joint in joints}
    if len(names) != len(joints):
        raise ValueError("Duplicate URDF joint names")
    parents = {}
    for joint in joints:
        if joint.child in parents:
            raise ValueError(f"Link has multiple parents: {joint.child}")
        parents[joint.child] = joint.parent
        if joint.mimic is not None and (joint.mimic not in names or joint.mimic == joint.name):
            raise ValueError(f"{joint.name}: invalid mimic reference")
    mimic_targets = {joint.name: joint.mimic for joint in joints if joint.mimic is not None}
    for joint_name in mimic_targets:
        seen = set()
        current = joint_name
        while current in mimic_targets:
            if current in seen:
                raise ValueError("URDF mimic graph contains a cycle")
            seen.add(current)
            current = mimic_targets[current]
    roots = set(links) - parents.keys()
    if len(roots) != 1:
        raise ValueError("URDF must have exactly one root link")
    root_link = next(iter(roots))
    for link in links:
        seen = set()
        current = link
        while current in parents:
            if current in seen:
                raise ValueError("URDF link graph contains a cycle")
            seen.add(current)
            current = parents[current]
        if current != root_link:
            raise ValueError("Disconnected URDF link graph")
    # Validate all origin and inertia values, including ones an IK reader ignores.
    for origin in root.iter("origin"):
        for key in ("xyz", "rpy"):
            values = origin.get(key, "0 0 0").split()
            if len(values) != 3:
                raise ValueError(f"origin.{key} must have three components")
            for value in values:
                _number(value, f"origin.{key}")
    for link_node in root.findall("link"):
        inertial = link_node.find("inertial")
        if inertial is None:
            raise ValueError(f"{link_node.get('name')}: missing inertial parameters")
        mass, inertia = inertial.find("mass"), inertial.find("inertia")
        if mass is None or inertia is None or _number(mass.get("value"), "mass") <= 0:
            raise ValueError(f"{link_node.get('name')}: invalid mass/inertia")
        xx, yy, zz, xy, xz, yz = (
            _number(inertia.get(k), f"inertia.{k}")
            for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
        )
        determinant = xx * yy * zz + 2 * xy * xz * yz - xx * yz ** 2 - yy * xz ** 2 - zz * xy ** 2
        if xx <= 0 or xx * yy - xy ** 2 <= 0 or determinant <= 0:
            raise ValueError(f"{link_node.get('name')}: inertia is not positive definite")
        if link_node.find("visual/geometry") is None or link_node.find("collision/geometry") is None:
            raise ValueError(f"{link_node.get('name')}: missing visual/collision geometry")
    meshes = set()
    for mesh in root.iter("mesh"):
        uri = _name(mesh, "filename", "mesh")
        scale = mesh.get("scale", "1 1 1").split()
        if len(scale) != 3 or any(_number(v, "mesh.scale") <= 0 for v in scale):
            raise ValueError(f"Invalid mesh scale: {uri}")
        path = _mesh_path(uri, source, packages)
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty mesh: {path}")
        meshes.add(path)
    return Description(name, root_link, links, tuple(joints), tuple(sorted(meshes)))
