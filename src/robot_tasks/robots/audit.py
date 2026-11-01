"""Offline R5 preflight. A valid asset is not a calibrated, runnable robot."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Literal

from .profile import RobotProfile
from .urdf import Description, read_urdf


@dataclass(frozen=True)
class Issue:
    severity: Literal["error", "pending", "warning"]
    field: str
    message: str


@dataclass(frozen=True)
class RobotAudit:
    description: Description | None
    integrity_valid: bool
    issues: tuple[Issue, ...]

    @property
    def valid(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def ready(self) -> bool:
        return self.valid and not any(issue.severity == "pending" for issue in self.issues)

    def as_dict(self) -> dict:
        description = self.description
        return {
            "asset_integrity_valid": self.integrity_valid,
            "description_parsed": description is not None,
            "asset_and_bindings_valid": self.valid,
            "execution_configuration_ready": self.ready,
            "simulation_validated": False,
            "description": None if description is None else {
                "name": description.name, "root_link": description.root_link,
                "links": description.links, "joints": [asdict(j) for j in description.joints],
                "meshes": [str(path) for path in description.meshes],
            },
            "issues": [asdict(issue) for issue in self.issues],
        }


def _verify_sources(profile: RobotProfile) -> set[Path]:
    lock = json.loads(profile.asset_lock.read_text(encoding="utf-8"))
    if (not isinstance(lock, dict) or type(lock.get("schema_version")) is not int
            or lock["schema_version"] != 1 or not isinstance(lock.get("files"), list)
            or not lock["files"]):
        raise ValueError("Invalid or empty asset source lock")
    root = profile.asset_lock.parent.resolve()
    verified = set()
    for entry in lock["files"]:
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root) or path in verified:
            raise ValueError(f"Invalid or duplicate source lock path: {path}")
        data = path.read_bytes()
        digest = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if digest != entry["git_blob_sha1"]:
            raise ValueError(f"Manufacturer asset modified or corrupt: {path}")
        verified.add(path)
    if profile.asset_path not in verified:
        raise ValueError("Configured URDF is not covered by the source lock")
    return verified


def audit_robot(profile: RobotProfile) -> RobotAudit:
    issues: list[Issue] = []
    description = None
    integrity_valid = False
    verified: set[Path] = set()
    try:
        verified = _verify_sources(profile)
        integrity_valid = True
    except (OSError, ValueError, KeyError, TypeError) as exc:
        issues.append(Issue("error", "asset.lock_file", str(exc)))
    try:
        description = read_urdf(profile.asset_path, dict(profile.package_roots))
        if description.name != profile.asset_robot_name:
            raise ValueError(f"Wrong URDF robot: {description.name}; expected {profile.asset_robot_name}")
        if integrity_valid and not set(description.meshes).issubset(verified):
            raise ValueError("A referenced mesh is not covered by the source lock")
        if description.root_link != profile.base_link:
            raise ValueError(f"Configured base_link is not the URDF root: {profile.base_link}")
        if description.chain(profile.base_link, profile.ee_link) != profile.joint_names:
            raise ValueError("Arm joint_names must match the ordered URDF chain from base_link to ee_link")
        for name in profile.joint_names:
            if description.joint(name).kind not in {"revolute", "continuous"}:
                raise ValueError(f"R5 arm joint is not rotational: {name}")
        if len(profile.joint_names) != 6:
            raise ValueError("R5 requires six arm joints per instance")
        for name in profile.gripper.joint_names:
            joint = description.joint(name)
            if joint.kind != "prismatic" or joint.parent != profile.ee_link:
                raise ValueError(f"{name}: expected a prismatic gripper joint attached to ee_link")
        movable = {joint.name for joint in description.joints if joint.kind != "fixed"}
        if movable != set(profile.joint_names + profile.gripper.joint_names):
            raise ValueError("Some movable joints are unmapped or mapped more than once")
    except (OSError, ValueError) as exc:
        issues.append(Issue("error", "asset.description", str(exc)))

    def pending(field: str, message: str) -> None:
        issues.append(Issue("pending", field, message))

    if not profile.hardware_match:
        pending("hardware_match", "Confirm the R5a description and gripper revision match the installed R5 arms")
    if profile.tcp_pose_wxyz is None or not profile.tcp_source:
        pending("kinematics.tcp_pose_wxyz", "Define the actual tool-center pose relative to ee_link and its source")
    limits = {limit.joint: limit for limit in profile.operating_limits}
    for name in profile.joint_names + profile.gripper.joint_names:
        if name not in limits:
            pending(f"operating_limits.{name}", "Set operating bounds with a source; do not use exporter limits as hardware limits")
    if not profile.limits_source:
        pending("limits_source", "Record the SDK/manual or explicit simulation operating-envelope source")
    for mount in profile.mounts:
        if mount.base_pose_wxyz is None:
            pending(f"arms.{mount.name}.base_pose_wxyz", "Set this arm's measured or explicitly designed base pose in the world frame")
        if mount.home_positions_rad is None:
            pending(f"arms.{mount.name}.home_positions_rad", "Set a reviewed initial joint configuration")
        else:
            for name, position in zip(profile.joint_names, mount.home_positions_rad):
                limit = limits.get(name)
                if limit is not None and not limit.lower <= position <= limit.upper:
                    issues.append(Issue("error", f"arms.{mount.name}.home_positions_rad",
                                        f"{name} initial position is outside operating limits"))
    gripper = profile.gripper
    if gripper.width_limits_m is None or not gripper.calibration_source:
        pending("gripper.calibration", "Calibrate total opening width -> each finger joint; the URDF alone does not define a zero gap")
    # Limits in the vendor model describe the asset; operating limits are separate.
    if description is not None:
        for limit in profile.operating_limits:
            try:
                joint = description.joint(limit.joint)
                if ((joint.lower is not None and limit.lower < joint.lower)
                        or (joint.upper is not None and limit.upper > joint.upper)
                        or (joint.effort is not None and limit.effort > joint.effort)
                        or (joint.velocity is not None and limit.velocity > joint.velocity)):
                    raise ValueError(f"{limit.joint}: operating limits exceed the source description")
            except ValueError as exc:
                issues.append(Issue("error", "operating_limits", str(exc)))
        if (gripper.width_limits_m is not None and gripper.joint_scale is not None
                and gripper.joint_offset_m is not None):
            for name, scale, offset in zip(gripper.joint_names, gripper.joint_scale, gripper.joint_offset_m):
                try:
                    joint = description.joint(name)
                    if scale == 0:
                        raise ValueError(f"{name}: a zero scale does not control gripper opening")
                    for width in gripper.width_limits_m:
                        position = scale * width + offset
                        limit = limits.get(name)
                        lower = limit.lower if limit else joint.lower
                        upper = limit.upper if limit else joint.upper
                        if lower is None or upper is None or not lower <= position <= upper:
                            raise ValueError(f"{name}: width mapping exceeds joint limits")
                except ValueError as exc:
                    issues.append(Issue("error", "gripper.calibration", str(exc)))
        if any(j.velocity is not None and j.velocity >= 100 for j in description.joints):
            issues.append(Issue("warning", "asset.joint_limits",
                                "Vendor URDF contains very broad exporter limits; these are not calibrated operating limits"))
    return RobotAudit(description, integrity_valid, tuple(issues))


def require_robot_ready(profile: RobotProfile) -> RobotAudit:
    audit = audit_robot(profile)
    if not audit.ready:
        unresolved = "; ".join(f"{issue.field}: {issue.message}" for issue in audit.issues
                               if issue.severity != "warning")
        raise ValueError(f"R5 configuration is not ready for execution: {unresolved}")
    return audit


def require_joint_sim_ready(profile: RobotProfile) -> RobotAudit:
    """Joint-only simulation needs no TCP or confirmation of the physical revision.

    Preserve those pending items in the returned report; this is not permission
    for Cartesian control or for a hardware adapter.
    """
    audit = audit_robot(profile)
    deferred = {"hardware_match", "kinematics.tcp_pose_wxyz"}
    blockers = [issue for issue in audit.issues if issue.severity == "error"
                or (issue.severity == "pending" and issue.field not in deferred)]
    if blockers:
        details = "; ".join(f"{issue.field}: {issue.message}" for issue in blockers)
        raise ValueError(f"R5 joint simulation configuration is incomplete: {details}")
    return audit


def require_cartesian_sim_ready(profile: RobotProfile) -> RobotAudit:
    """Simulation pose control needs a TCP, but not a physical hardware match."""
    audit = require_joint_sim_ready(profile)
    if profile.tcp_pose_wxyz is None or not profile.tcp_source:
        raise ValueError("R5 Cartesian simulation requires a defined TCP and its source")
    return audit
