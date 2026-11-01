"""Relocate a verified fitted collision model without random refitting."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from ..robots.kinematics import R5Kinematics


def geometry_digest(attachments):
    # USD world/local transforms differ below a micrometre across platforms.
    # Quantize only the identity; the stored collision spheres stay unchanged.
    def normalize(value):
        if isinstance(value, dict):
            return {k: normalize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [normalize(v) for v in value]
        return round(value, 5) + 0.0 if isinstance(value, float) else value
    return hashlib.sha256(json.dumps(normalize(attachments or {}), sort_keys=True).encode()).hexdigest()


def build_reference_assets(profile, directory, attachments, reference_path):
    raw = Path(reference_path).read_bytes()
    reference = json.loads(raw)
    actual = {
        "profile_sha256": hashlib.sha256(profile.source.read_bytes()).hexdigest(),
        "asset_lock_sha256": hashlib.sha256(profile.asset_lock.read_bytes()).hexdigest(),
        "attachments_sha256_10um": geometry_digest(attachments),
    }
    if reference.get("schema") != 1 or reference["inputs"] != actual:
        raise ValueError("Pinned planner geometry does not match the robot profile/assets/wrist geometry; export a matching reference")
    # Include this loader so implementation changes cannot reuse stale paths.
    identity = hashlib.sha256(raw + Path(__file__).read_bytes()).hexdigest()[:16]
    output = directory.resolve() / ("reference_" + identity)
    output.mkdir(parents=True, exist_ok=True)
    result = deepcopy(reference["robot"])
    root = ET.fromstring(reference["arm_urdf"])
    packages = dict(profile.package_roots)
    for mesh in root.iter("mesh"):
        uri = mesh.get("filename", "")
        if not uri.startswith("package://"):
            raise ValueError("Pinned geometry contains a non-portable mesh path")
        package, relative = uri[len("package://"):].split("/", 1)
        resolved = (packages[package] / relative).resolve()
        if not resolved.is_relative_to(packages[package].resolve()) or not resolved.is_file():
            raise ValueError("Pinned geometry references a missing/outside mesh")
        mesh.set("filename", resolved.as_posix())
    arm = output / "arm.urdf"
    ET.ElementTree(root).write(arm, encoding="utf-8")
    cfg = result["config"]["robot_cfg"]["kinematics"]
    cfg["urdf_path"], cfg["asset_root_path"] = str(arm), str(output)
    result["metadata"].update(asset_identity=identity,
        fitted_reference_sha256=hashlib.sha256(raw).hexdigest(),
        fitted_reference_source=reference["source_asset_identity"])
    (output / "robot.json").write_text(json.dumps(result, indent=2))
    return result, R5Kinematics(profile)
