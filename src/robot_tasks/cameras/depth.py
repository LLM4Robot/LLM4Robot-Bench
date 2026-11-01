"""Metric RGB-D validation, display and pixel queries; no object/evaluator truth."""

from io import BytesIO
import math

import numpy as np
from PIL import Image, ImageDraw

from ..robots.transforms import pose_transform
from ..types import DepthQuery, RGBFrame


# Fixed display scale across cameras and time. This is NOT a hardware range.
DISPLAY_RANGE_M = (0.02, 2.0)
MAX_DEPTH_QUERIES = 8


def depth_array(frame: RGBFrame) -> np.ndarray:
    """Reject unpaired/stale data before recording, inference or backprojection."""
    depth, calibration = frame.depth, frame.calibration
    if depth is None or calibration is None:
        raise ValueError(f"{frame.camera}: RGB-D requires depth and calibration")
    if (depth.camera != frame.camera or (depth.width, depth.height) != (frame.width, frame.height)
            or min(depth.width, depth.height) <= 0 or len(depth.data) != depth.width * depth.height * 4):
        raise ValueError(f"{frame.camera}: depth/RGB dimensions or camera disagree")
    if (not math.isfinite(frame.timestamp_s)
            or not math.isclose(depth.timestamp_s, frame.timestamp_s, rel_tol=0, abs_tol=1e-9)
            or not math.isclose(calibration.get("simulation_time_s", math.nan), frame.timestamp_s,
                                rel_tol=0, abs_tol=1e-9)
            or type(depth.capture_id) is not int or depth.capture_id != calibration.get("capture_id")
            or tuple(calibration.get("resolution", ())) != (frame.width, frame.height)):
        raise ValueError(f"{frame.camera}: depth/RGB capture or calibration disagree")
    values = np.frombuffer(depth.data, dtype="<f4").reshape(depth.height, depth.width)
    if np.isinf(values).any() or np.any(np.isfinite(values) & (values <= 0)):
        raise ValueError("Depth must contain positive metres or NaN for invalid pixels")
    return values


def depth_metadata(frame: RGBFrame) -> dict:
    values = depth_array(frame)
    valid = values[np.isfinite(values)]
    return {"source": "ideal_simulation_render", "quantity": "optical_axis_z", "unit": "m",
            "dtype": "float32", "invalid": "NaN", "aligned_to": "rgb", "hardware_emulated": False,
            "capture_id": frame.depth.capture_id, "time_s": frame.timestamp_s,
            "resolution": [frame.width, frame.height], "valid_fraction": float(valid.size / values.size),
            "minimum_m": float(valid.min()) if valid.size else None,
            "maximum_m": float(valid.max()) if valid.size else None}


def _colour(values: np.ndarray) -> np.ndarray:
    near, far = DISPLAY_RANGE_M
    t = np.clip((np.nan_to_num(values, nan=near) - near) / (far - near), 0, 1)
    anchors = np.array([[0, 80, 255], [0, 220, 220], [255, 230, 0], [255, 40, 0]])
    # Avoid three simultaneous float64 channels plus a full float64 RGB copy.
    # Keep the exact colour mapping; the raw metric depth is never quantized.
    rgb = np.empty((*t.shape, 3), dtype=np.uint8)
    for i in range(3):
        rgb[..., i] = np.interp(t, [0, 1/3, 2/3, 1], anchors[:, i])
    rgb[~np.isfinite(values)] = 0
    return rgb


def encode_depth_preview(frame: RGBFrame, size: tuple[int, int]) -> tuple[bytes, dict]:
    """Nearest-pixel sampling avoids inventing depths across object boundaries.

    RGB coordinates are unchanged; a legend is appended BELOW the depth image.
    Raw float32 depth is saved separately and never quantized for measurements.
    """
    values = depth_array(frame)
    width, height = size
    if min(size) <= 0:
        raise ValueError("Depth preview dimensions must be positive")
    columns = np.minimum(((np.arange(width) + .5) * frame.width / width).astype(int), frame.width - 1)
    rows = np.minimum(((np.arange(height) + .5) * frame.height / height).astype(int), frame.height - 1)
    image = Image.new("RGB", (width, height + 40), "#202020")
    image.paste(Image.fromarray(_colour(values[rows[:, None], columns])), (0, 0))
    gradient = np.linspace(*DISPLAY_RANGE_M, width, dtype=np.float32)[None, :]
    image.paste(Image.fromarray(_colour(np.repeat(gradient, 6, axis=0))), (0, height))
    draw = ImageDraw.Draw(image)
    draw.text((2, height + 8), "Z: 0.02m blue -> 2.00m red (clipped display)", fill="white")
    draw.text((2, height + 23), "Black: no depth. Query pixels in the RGB image.", fill="white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue(), {"image_region_xywh": [0, 0, width, height],
                              "encoded_size": list(image.size), "display_range_m": list(DISPLAY_RANGE_M),
                              "display_clipped": True, "invalid_colour": "black",
                              "sampling": "nearest source pixel; no depth interpolation"}


def validate_depth_queries(queries: tuple[DepthQuery, ...], cameras: tuple[str, ...]) -> None:
    if len(queries) > MAX_DEPTH_QUERIES:
        raise ValueError(f"At most {MAX_DEPTH_QUERIES} depth queries per decision")
    for query in queries:
        if (not isinstance(query, DepthQuery) or query.camera not in cameras
                or any(type(v) is not int for v in (query.u, query.v, query.image_width, query.image_height))
                or not 0 <= query.u < query.image_width or not 0 <= query.v < query.image_height):
            raise ValueError("Depth query requires a known camera and an in-bounds integer RGB pixel")


def _undistort_pixel(u: int, v: int, calibration: dict) -> tuple[float, float]:
    fx, fy, cx, cy = calibration["intrinsics_fx_fy_cx_cy"]
    if not all(math.isfinite(x) for x in (fx, fy, cx, cy)) or min(fx, fy) <= 0:
        raise ValueError("Invalid depth backprojection intrinsics")
    coefficients = list(calibration.get("distortion", ()))
    if len(coefficients) not in {0, 4, 5, 8} or not all(math.isfinite(x) for x in coefficients):
        raise ValueError("Depth queries require OpenCV pinhole distortion coefficients")
    k1, k2, p1, p2, k3, k4, k5, k6 = coefficients + [0.] * (8 - len(coefficients))
    xd, yd = (u - cx) / fx, (v - cy) / fy
    x, y = xd, yd
    for _ in range(40):
        r2 = x*x + y*y
        numerator = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
        denominator = 1 + k4*r2 + k5*r2*r2 + k6*r2*r2*r2
        if not math.isfinite(numerator + denominator) or abs(denominator) < 1e-12:
            break
        radial = numerator / denominator
        dx, dy = 2*p1*x*y + p2*(r2 + 2*x*x), p1*(r2 + 2*y*y) + 2*p2*x*y
        if max(abs((x*radial + dx - xd)*fx), abs((y*radial + dy - yd)*fy)) < 1e-6:
            return x, y
        if abs(radial) < 1e-12:
            break
        x, y = (xd - dx) / radial, (yd - dy) / radial
    raise ValueError("Pixel undistortion did not converge; no 3D point returned")


def resolve_depth_queries(frames: tuple[RGBFrame, ...], queries: tuple[DepthQuery, ...]) -> list[dict]:
    """Read the visible surface only, never a task object's hidden centre/pose."""
    validate_depth_queries(queries, tuple(f.camera for f in frames))
    by_name = {f.camera: f for f in frames}
    results = []
    for query in queries:
        frame = by_name[query.camera]
        values = depth_array(frame)
        u = min(int((query.u + .5) * frame.width / query.image_width), frame.width - 1)
        v = min(int((query.v + .5) * frame.height / query.image_height), frame.height - 1)
        result = {"camera": query.camera, "rgb_pixel_uv": [query.u, query.v],
                  "rgb_image_size": [query.image_width, query.image_height], "source_pixel_uv": [u, v],
                  "capture_id": frame.depth.capture_id, "simulation_time_s": frame.timestamp_s,
                  "source": "ideal_simulation_depth", "surface_point_only": True,
                  "depth_z_m": None, "point_world_m": None, "point_camera_opencv_m": None}
        z = float(values[v, u])
        if not math.isfinite(z):
            result["status"] = "no_depth"
        else:
            x, y = _undistort_pixel(u, v, frame.calibration)
            cv = np.array([x*z, y*z, z])
            transform = pose_transform(frame.calibration["T_world_camera_pose_wxyz"])
            world = transform[:3, :3] @ (cv * [1., -1., -1.]) + transform[:3, 3]
            result.update(status="ok", depth_z_m=z, point_camera_opencv_m=cv.tolist(), point_world_m=world.tolist())
        results.append(result)
    return results
