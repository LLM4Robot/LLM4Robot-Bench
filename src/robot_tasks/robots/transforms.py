"""Rigid transforms in metres; poses are xyz + scalar-first quaternion wxyz."""

import math
from numbers import Real

import numpy as np


def finite_vector(values, size: int, label: str) -> np.ndarray:
    if len(values) != size or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real)
                                   or not math.isfinite(v) for v in values):
        raise ValueError(f"{label} requires {size} finite numeric values")
    return np.asarray(values, dtype=float)


def axis_rotation(axis, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew)


def origin_transform(xyz, rpy) -> np.ndarray:
    """URDF fixed-axis roll/pitch/yaw: Rz(yaw) Ry(pitch) Rx(roll)."""
    roll, pitch, yaw = rpy
    result = np.eye(4)
    result[:3, :3] = (axis_rotation((0, 0, 1), yaw) @ axis_rotation((0, 1, 0), pitch)
                      @ axis_rotation((1, 0, 0), roll))
    result[:3, 3] = xyz
    return result


def pose_transform(pose) -> np.ndarray:
    values = finite_vector(pose, 7, "Pose (xyz + wxyz)")
    q = values[3:]
    norm = np.linalg.norm(q)
    if not math.isclose(norm, 1.0, abs_tol=1e-3):
        raise ValueError("Pose quaternion must be normalized (wxyz)")
    w, x, y, z = q / norm
    result = np.eye(4)
    result[:3, :3] = [
        [1 - 2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1 - 2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1 - 2*(x*x+y*y)],
    ]
    result[:3, 3] = values[:3]
    return result


def rotation_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Largest-diagonal conversion remains well-conditioned at 180 degrees."""
    r = rotation
    candidates = np.array([1 + np.trace(r), 1 + r[0, 0] - r[1, 1] - r[2, 2],
                           1 - r[0, 0] + r[1, 1] - r[2, 2],
                           1 - r[0, 0] - r[1, 1] + r[2, 2]])
    i = int(np.argmax(candidates))
    s = 2 * math.sqrt(max(0.0, candidates[i]))
    if i == 0:
        q = np.array([s/4, (r[2, 1]-r[1, 2])/s, (r[0, 2]-r[2, 0])/s, (r[1, 0]-r[0, 1])/s])
    elif i == 1:
        q = np.array([(r[2, 1]-r[1, 2])/s, s/4, (r[0, 1]+r[1, 0])/s, (r[0, 2]+r[2, 0])/s])
    elif i == 2:
        q = np.array([(r[0, 2]-r[2, 0])/s, (r[0, 1]+r[1, 0])/s, s/4, (r[1, 2]+r[2, 1])/s])
    else:
        q = np.array([(r[1, 0]-r[0, 1])/s, (r[0, 2]+r[2, 0])/s, (r[1, 2]+r[2, 1])/s, s/4])
    q /= np.linalg.norm(q)
    return -q if q[0] < 0 else q


def transform_pose(transform: np.ndarray) -> tuple[float, ...]:
    return tuple(float(v) for v in np.r_[transform[:3, 3], rotation_quaternion(transform[:3, :3])])


def pose_residual(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Translation and shortest rotation vector, both in the common parent frame."""
    q = rotation_quaternion(target[:3, :3] @ current[:3, :3].T)
    sine = np.linalg.norm(q[1:])
    rotation = 2 * q[1:] if sine < 1e-10 else q[1:] * (2 * math.atan2(sine, q[0]) / sine)
    return np.r_[target[:3, 3] - current[:3, 3], rotation]


def pose_errors(current, target) -> tuple[float, float]:
    residual = pose_residual(pose_transform(current), pose_transform(target))
    return float(np.linalg.norm(residual[:3])), float(np.linalg.norm(residual[3:]))
