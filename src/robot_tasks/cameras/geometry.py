"""Explicit USD/OpenCV camera geometry, independent of Isaac and model APIs.

T_world_camera maps camera points to world. USD optical axes: right/up/back;
OpenCV optical axes: right/down/forward. Pixels refer to image pixel centres.
"""

import numpy as np

from ..robots.transforms import finite_vector, pose_transform


def resized_intrinsics(intrinsics, original_size, new_size) -> tuple[float, ...]:
    fx, fy, cx, cy = finite_vector(intrinsics, 4, "Camera intrinsics")
    if min(*original_size, *new_size) <= 0:
        raise ValueError("Image sizes must be positive")
    sx, sy = new_size[0] / original_size[0], new_size[1] / original_size[1]
    # PIL resize aligns pixel centres (not upper-left pixel corners).
    return (float(fx * sx), float(fy * sy), float((cx + .5) * sx - .5), float((cy + .5) * sy - .5))


def project_world(points, camera_pose, intrinsics, distortion=()) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if not np.isfinite(points).all():
        raise ValueError("World points must be finite")
    transform = pose_transform(camera_pose)
    local = (points - transform[:3, 3]) @ transform[:3, :3]
    cv = local * np.array([1., -1., -1.])
    with np.errstate(divide="ignore", invalid="ignore"):
        x, y = cv[:, 0] / cv[:, 2], cv[:, 1] / cv[:, 2]
        k1, k2, p1, p2, k3, k4, k5, k6 = list(distortion) + [0.] * (8 - len(distortion))
        r2 = x*x + y*y
        radial = (1 + k1*r2 + k2*r2**2 + k3*r2**3) / (1 + k4*r2 + k5*r2**2 + k6*r2**3)
        xd = x*radial + 2*p1*x*y + p2*(r2 + 2*x*x)
        yd = y*radial + p1*(r2 + 2*y*y) + 2*p2*x*y
        fx, fy, cx, cy = intrinsics
        pixels = np.column_stack((fx*xd + cx, fy*yd + cy))
    pixels[cv[:, 2] <= 0] = np.nan
    return pixels


def calibration_metadata(spec, world_pose, timestamp: float, capture_id: int) -> dict:
    return {"intrinsics_fx_fy_cx_cy": list(spec.intrinsics), "distortion": list(spec.distortion),
            "resolution": list(spec.resolution), "T_world_camera_pose_wxyz": list(world_pose),
            "camera_axes": "USD: +X right, +Y up, -Z forward",
            "pixel_axes": "u right, v down; zero-based pixel centres",
            "source": "simulation_camera_parameters", "hardware_calibrated": False,
            "capture_id": capture_id, "simulation_time_s": timestamp}
