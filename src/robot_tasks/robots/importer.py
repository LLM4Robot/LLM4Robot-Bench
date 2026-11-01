"""Convert immutable vendor URDF into a separately versioned Isaac Sim asset."""

from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from ..backends.settings import SimulationSettings
from .audit import audit_robot
from .profile import RobotProfile


def prepare_urdf(profile: RobotProfile, destination: Path) -> Path:
    """Resolve package URIs and apply the explicitly configured simulation limits."""
    audit = audit_robot(profile)
    if not audit.valid:
        raise ValueError(f"R5 source asset validation failed: {audit.issues}")
    tree = ET.parse(profile.asset_path)
    packages = dict(profile.package_roots)
    for mesh in tree.getroot().iter("mesh"):
        uri = mesh.get("filename", "")
        if uri.startswith("package://"):
            package, relative = uri[len("package://"):].split("/", 1)
            resolved = (packages[package] / relative).resolve()
        else:
            resolved = (profile.asset_path.parent / uri).resolve()
        mesh.set("filename", resolved.as_posix())
    limits = {limit.joint: limit for limit in profile.operating_limits}
    for joint in tree.getroot().findall("joint"):
        if joint.get("name") in limits:
            limit = limits[joint.get("name")]
            element = joint.find("limit")
            for field in ("lower", "upper", "effort", "velocity"):
                element.set(field, str(getattr(limit, field)))
    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="utf-8", xml_declaration=True)
    return destination


def convert_r5(profile: RobotProfile, settings: SimulationSettings) -> tuple[Path, dict]:
    """Call only after SimulationApp startup. The output is regenerated if inputs change."""
    import omni.kit.commands
    from isaacsim.core.utils.extensions import enable_extension
    from pxr import Usd, UsdGeom

    version = importlib.metadata.version("isaacsim")
    inputs = {
        "schema_version": 1, "isaacsim_version": version,
        "platform": sys.platform, "source_urdf_path": str(profile.asset_path),
        "source_lock_sha256": hashlib.sha256(profile.asset_lock.read_bytes()).hexdigest(),
        "importer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "profile_sha256": hashlib.sha256(profile.source.read_bytes()).hexdigest(),
        "limits": [asdict(limit) for limit in profile.operating_limits],
        "fixed_base": True, "convex_decomposition": True,
        "self_collision": settings.self_collision,
        "arm_stiffness": settings.arm_stiffness, "arm_damping": settings.arm_damping,
    }
    identity = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:16]
    directory = settings.asset_directory / identity
    usd_path = directory / "r5.usd"
    manifest_path = directory / "import.json"
    audit = audit_robot(profile)
    if not audit.valid:
        raise ValueError(f"R5 source asset validation failed: {audit.issues}")
    if usd_path.is_file() and manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        artifacts = existing.get("generated_sha256", {})
        if existing.get("inputs") == inputs and artifacts and all(
            (directory / name).is_file()
            and hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest
            for name, digest in artifacts.items()
        ):
            return usd_path, existing
        raise ValueError(f"Generated asset cache is incomplete or changed: {directory}; use a new asset_directory")
    urdf = prepare_urdf(profile, directory / "resolved.urdf")
    enable_extension("isaacsim.asset.importer.urdf")
    ok, options = omni.kit.commands.execute("URDFCreateImportConfig")
    if not ok:
        raise RuntimeError("Unable to create Isaac Sim URDF import configuration")
    options.fix_base = True
    options.merge_fixed_joints = False
    options.make_default_prim = True
    options.create_physics_scene = False
    options.import_inertia_tensor = True
    options.convex_decomp = True
    options.self_collision = settings.self_collision
    options.default_drive_strength = settings.arm_stiffness
    options.default_position_drive_damping = settings.arm_damping
    ok, imported_path = omni.kit.commands.execute(
        "URDFParseAndImportFile", urdf_path=str(urdf), import_config=options, dest_path=str(usd_path),
    )
    if not ok or not usd_path.is_file():
        raise RuntimeError(f"R5 URDF import failed: {imported_path}")
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None or not stage.GetDefaultPrim().IsValid():
        raise RuntimeError("Imported R5 USD has no default prim")
    if abs(UsdGeom.GetStageMetersPerUnit(stage) - 1.0) > 1e-9:
        raise RuntimeError("R5 USD units are not meters")
    manifest = {
        "inputs": inputs, "source_urdf": str(profile.asset_path),
        "default_prim": str(stage.GetDefaultPrim().GetPath()),
        "generated_sha256": {
            path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in directory.rglob("*") if path.is_file() and path != manifest_path
        },
        "scope": "R5a simulation import; no measured hardware calibration",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return usd_path, manifest
