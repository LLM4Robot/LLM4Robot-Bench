"""R5 configuration shared by tasks; unset calibration never becomes a default."""

from dataclasses import dataclass
import math
from pathlib import Path
import re
import tomllib


@dataclass(frozen=True)
class ArmMount:
    name: str
    prim_path: str
    base_pose_wxyz: tuple[float, ...] | None
    home_positions_rad: tuple[float, ...] | None


@dataclass(frozen=True)
class OperatingLimit:
    joint: str
    lower: float
    upper: float
    effort: float
    velocity: float


@dataclass(frozen=True)
class Gripper:
    joint_names: tuple[str, ...]
    width_limits_m: tuple[float, ...] | None
    joint_scale: tuple[float, ...] | None
    joint_offset_m: tuple[float, ...] | None
    calibration_source: str


@dataclass(frozen=True)
class RobotProfile:
    source: Path
    model: str
    asset_path: Path
    asset_robot_name: str
    asset_lock: Path
    package_roots: tuple[tuple[str, Path], ...]
    base_link: str
    ee_link: str
    joint_names: tuple[str, ...]
    tcp_pose_wxyz: tuple[float, ...] | None
    tcp_source: str
    gripper: Gripper
    mounts: tuple[ArmMount, ...]
    operating_limits: tuple[OperatingLimit, ...]
    limits_source: str
    hardware_match: str

    @property
    def available_arms(self) -> tuple[str, ...]:
        return tuple(mount.name for mount in self.mounts)


def names(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(
        not isinstance(item, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", item)
        for item in value
    ):
        raise ValueError(f"{label}: expected nonempty names")
    if len(set(value)) != len(value):
        raise ValueError(f"{label}: duplicate names")
    return tuple(value)


def text(value: object, label: str, *, optional: bool = False) -> str:
    if not isinstance(value, str) or (not optional and not value.strip()):
        raise ValueError(f"{label}: expected {'optional ' if optional else ''}text")
    return value.strip()


def vector(value: object, size: int, label: str) -> tuple[float, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != size or any(
        isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v)
        for v in value
    ):
        raise ValueError(f"{label}: expected {size} finite numbers")
    return tuple(float(v) for v in value)


def pose(value: object, label: str) -> tuple[float, ...] | None:
    result = vector(value, 7, label)
    if result is not None and not math.isclose(sum(v * v for v in result[3:]), 1, abs_tol=1e-6):
        raise ValueError(f"{label}: quaternion must be normalized (xyz + wxyz)")
    return result


def _fields(data: dict, allowed: set[str], label: str, required: set[str] | None = None) -> None:
    if not isinstance(data, dict):
        raise ValueError(f"{label}: expected a TOML table")
    unknown = data.keys() - allowed
    missing = (required or set()) - data.keys()
    if unknown or missing:
        raise ValueError(f"{label}: unknown fields={sorted(unknown)}, missing fields={sorted(missing)}")


def load_robot_profile(path: Path) -> RobotProfile:
    source = path.expanduser().resolve()
    with source.open("rb") as stream:
        data = tomllib.load(stream)
    _fields(data, {"schema_version", "model", "asset", "kinematics", "gripper", "arms",
                   "operating_limits", "limits_source", "hardware_match"}, "robot profile",
            {"schema_version", "model", "asset", "kinematics", "gripper", "arms"})
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError("Robot profile requires schema_version = 1")
    if data["model"] != "r5":
        raise ValueError("This project requires ARX R5; other robot models are not supported")
    asset = data["asset"]
    _fields(asset, {"path", "robot_name", "lock_file", "package_roots"}, "asset",
            {"path", "robot_name", "lock_file", "package_roots"})
    asset_path = (source.parent / text(asset["path"], "asset.path")).resolve()
    if asset_path.suffix.lower() != ".urdf":
        raise ValueError("Step 1 requires a URDF source; CAD/STEP is not a robot description")
    package_roots = asset["package_roots"]
    if not isinstance(package_roots, dict) or not package_roots:
        raise ValueError("asset.package_roots must map ROS package names to local directories")
    packages = tuple((text(k, "package name"), (source.parent / text(v, k)).resolve())
                     for k, v in package_roots.items())
    kinematics = data["kinematics"]
    _fields(kinematics, {"base_link", "ee_link", "joint_names", "tcp_pose_wxyz", "tcp_source"},
            "kinematics", {"base_link", "ee_link", "joint_names"})
    joints = names(kinematics["joint_names"], "kinematics.joint_names")
    gripper = data["gripper"]
    _fields(gripper, {"joint_names", "width_limits_m", "joint_scale", "joint_offset_m",
                      "calibration_source"}, "gripper", {"joint_names"})
    finger_joints = names(gripper["joint_names"], "gripper.joint_names")
    if len(finger_joints) != 2 or set(finger_joints) & set(joints):
        raise ValueError("R5 gripper must name two distinct finger joints outside the arm chain")
    width = vector(gripper.get("width_limits_m"), 2, "gripper.width_limits_m")
    if width is not None and not 0 <= width[0] < width[1]:
        raise ValueError("gripper.width_limits_m requires 0 <= min < max")
    scale = vector(gripper.get("joint_scale"), 2, "gripper.joint_scale")
    offset = vector(gripper.get("joint_offset_m"), 2, "gripper.joint_offset_m")
    if any(v is not None for v in (width, scale, offset)) and not all(
        v is not None for v in (width, scale, offset)
    ):
        raise ValueError("Gripper width limits, scales and offsets must be provided together")
    arms = data["arms"]
    _fields(arms, {"left", "right"}, "arms", {"left", "right"})
    mounts = []
    for name in ("left", "right"):
        arm = arms[name]
        _fields(arm, {"prim_path", "base_pose_wxyz", "home_positions_rad"}, f"arms.{name}", {"prim_path"})
        prim = text(arm["prim_path"], f"arms.{name}.prim_path")
        if not re.fullmatch(r"(?:/[A-Za-z_][A-Za-z0-9_]*)+", prim):
            raise ValueError(f"arms.{name}.prim_path: invalid absolute USD prim path")
        mounts.append(ArmMount(
            name, prim, pose(arm.get("base_pose_wxyz"), f"arms.{name}.base_pose_wxyz"),
            vector(arm.get("home_positions_rad"), len(joints), f"arms.{name}.home_positions_rad"),
        ))
    left_path, right_path = (mount.prim_path for mount in mounts)
    if (left_path == right_path or left_path.startswith(right_path + "/")
            or right_path.startswith(left_path + "/")):
        raise ValueError("R5 arm prim paths must be distinct and not nested")
    limits_data = data.get("operating_limits", {})
    _fields(limits_data, set(joints + finger_joints), "operating_limits")
    limits = []
    for name, entry in limits_data.items():
        keys = {"lower", "upper", "effort", "velocity"}
        _fields(entry, keys, f"operating_limits.{name}", keys)
        values = vector([entry[k] for k in ("lower", "upper", "effort", "velocity")], 4, name)
        assert values is not None
        lower, upper, effort, velocity = values
        if lower >= upper or effort <= 0 or velocity <= 0:
            raise ValueError(f"operating_limits.{name}: invalid range/effort/velocity")
        limits.append(OperatingLimit(name, lower, upper, effort, velocity))
    return RobotProfile(
        source=source, model="r5", asset_path=asset_path,
        asset_robot_name=text(asset["robot_name"], "asset.robot_name"),
        asset_lock=(source.parent / text(asset["lock_file"], "asset.lock_file")).resolve(),
        package_roots=packages, base_link=text(kinematics["base_link"], "kinematics.base_link"),
        ee_link=text(kinematics["ee_link"], "kinematics.ee_link"), joint_names=joints,
        tcp_pose_wxyz=pose(kinematics.get("tcp_pose_wxyz"), "kinematics.tcp_pose_wxyz"),
        tcp_source=text(kinematics.get("tcp_source", ""), "kinematics.tcp_source", optional=True),
        gripper=Gripper(finger_joints, width, scale, offset,
                        text(gripper.get("calibration_source", ""), "gripper.calibration_source", optional=True)),
        mounts=tuple(mounts), operating_limits=tuple(limits),
        limits_source=text(data.get("limits_source", ""), "limits_source", optional=True),
        hardware_match=text(data.get("hardware_match", ""), "hardware_match", optional=True),
    )
