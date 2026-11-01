"""Portable RGB camera profiles, independent of Kit and hardware SDKs."""

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import tomllib

from ..robots.transforms import finite_vector, pose_transform


@dataclass(frozen=True)
class CameraSpec:
    name: str
    model: str
    parent: str
    resolution: tuple[int, int]
    intrinsics: tuple[float, float, float, float]  # fx, fy, cx, cy in pixels
    distortion: tuple[float, ...]  # OpenCV k1,k2,p1,p2,k3[,k4,k5,k6]
    optical_pose_wxyz: tuple[float, ...]  # metres, USD axes: -Z forward, +Y up
    intrinsics_source: str
    pose_source: str
    asset: Path
    asset_pose_wxyz: tuple[float, ...]  # CAD frame relative to optical frame
    asset_pose_source: str
    clipping_m: tuple[float, float] = (0.005, 20.0)
    existing_prim_path: str = ""  # Bind a camera supplied by the environment, without duplicating it.

    def metadata(self) -> dict:
        return {**asdict(self), "asset": str(self.asset),
                "asset_sha256": hashlib.sha256(self.asset.read_bytes()).hexdigest() if self.asset.is_file() else None,
                "axes": "USD: +X right, +Y up, -Z forward",
                "housing_physics": "visual_only; no added mass or collision",
                "image_model": "OpenCV pinhole; no exposure/noise/rolling-shutter simulation"}


@dataclass(frozen=True)
class CameraRigProfile:
    source: Path
    calibration_status: str
    notes: str
    cameras: tuple[CameraSpec, ...]

    def select(self, names: tuple[str, ...]) -> tuple[CameraSpec, ...]:
        available = {c.name: c for c in self.cameras}
        if not set(names).issubset(available):
            raise ValueError(f"Cameras missing from rig: {sorted(set(names) - available.keys())}")
        return tuple(available[name] for name in names)


def load_camera_rig(path: Path) -> CameraRigProfile:
    source = path.expanduser().resolve()
    with source.open("rb") as stream:
        raw = tomllib.load(stream)
    if set(raw) != {"schema_version", "calibration_status", "notes", "cameras"}:
        raise ValueError("Camera rig requires schema_version, calibration_status, notes and cameras")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ValueError("Camera rig schema_version must be 1")
    # Measured profiles will need a calibration artifact and provenance contract.
    if raw["calibration_status"] != "provisional_uncalibrated":
        raise ValueError("Only provisional_uncalibrated camera rigs are currently supported")
    if not isinstance(raw["notes"], str) or not raw["notes"].strip():
        raise ValueError("Camera rig notes must explain its assumptions")
    if not isinstance(raw["cameras"], dict) or not raw["cameras"]:
        raise ValueError("Camera rig must define at least one camera")
    result = []
    fields = set(CameraSpec.__dataclass_fields__) - {"name"}
    required = fields - {"clipping_m", "existing_prim_path"}
    for name, data in raw["cameras"].items():
        if not name.replace("_", "").isalnum():
            raise ValueError("Camera names must be simple identifiers")
        if not isinstance(data, dict) or set(data) - fields or not required.issubset(data):
            raise ValueError(f"{name}: invalid or missing camera fields")
        for key in ("model", "intrinsics_source", "pose_source", "asset", "asset_pose_source"):
            if not isinstance(data[key], str) or not data[key].strip():
                raise ValueError(f"{name}.{key} must be nonempty")
        if data["parent"] not in {"world", "left", "right"}:
            raise ValueError(f"{name}: parent must be world, left or right")
        existing = data.get("existing_prim_path", "")
        if not isinstance(existing, str) or (existing and (
            not existing.startswith("/World/") or ".." in existing.split("/")
        )):
            raise ValueError(f"{name}: existing_prim_path must be an absolute /World prim path")
        resolution = data["resolution"]
        if not isinstance(resolution, list) or len(resolution) != 2 or any(
            type(v) is not int or v <= 0 for v in resolution
        ):
            raise ValueError(f"{name}: resolution requires two positive integers")
        intrinsic = finite_vector(data["intrinsics"], 4, f"{name}.intrinsics")
        if min(intrinsic[:2]) <= 0 or not (0 <= intrinsic[2] < resolution[0] and 0 <= intrinsic[3] < resolution[1]):
            raise ValueError(f"{name}: invalid focal lengths or principal point")
        distortion = data["distortion"]
        if len(distortion) not in {4, 5, 8}:
            raise ValueError(f"{name}: OpenCV distortion requires 4, 5 or 8 coefficients")
        finite_vector(distortion, len(distortion), f"{name}.distortion")
        for key in ("optical_pose_wxyz", "asset_pose_wxyz"):
            pose_transform(data[key])
        near, far = finite_vector(data.get("clipping_m", (0.005, 20.0)), 2, f"{name}.clipping_m")
        if not 0 < near < far:
            raise ValueError(f"{name}: clipping must satisfy 0 < near < far")
        asset = (source.parent / data["asset"]).resolve()
        if (not asset.is_file() and not existing) or asset.suffix not in {".usd", ".usda", ".usdc"}:
            raise ValueError(f"{name}: camera USD asset does not exist: {asset}")
        values = {**data, "name": name, "asset": asset}
        for key in ("resolution", "intrinsics", "distortion", "optical_pose_wxyz", "asset_pose_wxyz", "clipping_m"):
            if key in values:
                values[key] = tuple(values[key])
        result.append(CameraSpec(**values))
    return CameraRigProfile(source, raw["calibration_status"], raw["notes"], tuple(result))


def intrinsics_from_fov(width: int, height: int, horizontal_deg: float, vertical_deg: float) -> tuple[float, ...]:
    """Nominal, centred pinhole estimate; this is not per-device calibration."""
    if min(width, height) <= 0 or not all(math.isfinite(v) and 0 < v < 180 for v in (horizontal_deg, vertical_deg)):
        raise ValueError("Image dimensions and FOV must be valid")
    return (width / (2 * math.tan(math.radians(horizontal_deg) / 2)),
            height / (2 * math.tan(math.radians(vertical_deg) / 2)), width / 2, height / 2)
