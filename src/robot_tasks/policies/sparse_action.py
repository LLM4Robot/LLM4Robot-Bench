"""Absolute named TCP axes; null holds the latest measured value, never an old goal."""

import json
import math

import numpy as np

from ..robots.transforms import origin_transform, pose_transform, transform_pose
from ..types import Action, ArmCommand, InvalidPolicyAction
from ..runner import validate_action
from ..robots.control import normalized_to_width
from .action_codec import _unique_object


FIELDS = ("x_m", "y_m", "z_m", "roll_deg", "pitch_deg", "yaw_deg", "gripper")
# R5 TCP +X points down, +Y closes along world Y at zero angles.
REFERENCE = origin_transform((0., 0., 0.), (0., math.pi / 2, 0.))[:3, :3]


def orientation_degrees(pose) -> tuple[float, float, float]:
    r = pose_transform(pose)[:3, :3] @ REFERENCE.T
    pitch = math.asin(float(np.clip(-r[2, 0], -1, 1)))
    if abs(math.cos(pitch)) < 1e-7:
        roll, yaw = 0., math.atan2(-r[0, 1], r[1, 1])
    else:
        roll, yaw = math.atan2(r[2, 1], r[2, 2]), math.atan2(r[1, 0], r[0, 0])
    return tuple(math.degrees(v) for v in (roll, pitch, yaw))


def sparse_schema(arms: tuple[str, ...], *, joint_targets: bool = False) -> dict:
    properties = {name: {"type": ["number", "null"]} for name in FIELDS}
    fields = list(FIELDS)
    if joint_targets:
        fields.append("joint_positions_rad")
        properties["joint_positions_rad"] = {
            "type": ["array", "null"], "items": {"type": "number"},
            "minItems": 6, "maxItems": 6,
        }
    properties["gripper"].update(minimum=0, maximum=1)
    properties["arm"] = {"type": "string", "enum": list(arms)}
    return {"type": "object", "additionalProperties": False, "required": ["commands", "request_stop"],
            "properties": {"request_stop": {"type": "boolean"}, "commands": {
                "type": "array", "maxItems": len(arms), "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["arm", *fields], "properties": properties}}}}


def decode_sparse_action(text, config, observation) -> Action:
    try:
        doc = json.loads(text, object_pairs_hook=_unique_object)
        if not isinstance(doc, dict) or set(doc) != {"commands", "request_stop"}:
            raise ValueError("Expected commands and request_stop only")
        rows, stop = doc["commands"], doc["request_stop"]
        if not isinstance(rows, list) or len(rows) > len(config.command_arms) or type(stop) is not bool:
            raise ValueError("Invalid commands or stop flag")
        commands, seen = [], set()
        for row in rows:
            keys = {"arm", *FIELDS}
            allowed = [keys]
            if config.raw.get("completion", {}).get("return_home", False):
                allowed.append(keys | {"joint_positions_rad"})
            if not isinstance(row, dict) or set(row) not in allowed:
                raise ValueError("Each sparse command must contain arm and all nullable target fields")
            arm = row["arm"]
            if not isinstance(arm, str) or arm not in config.command_arms or arm in seen:
                raise ValueError("Each active arm may appear only once")
            seen.add(arm)
            for field in FIELDS:
                value = row[field]
                if value is not None and (type(value) not in (int, float) or not math.isfinite(value)):
                    raise ValueError(f"{field} must be finite or null")
            motion = any(row[f] is not None for f in FIELDS[:6])
            joints = row.get("joint_positions_rad")
            if joints is not None:
                if motion:
                    raise ValueError("Joint targets and Cartesian targets are mutually exclusive")
                if (not isinstance(joints, list) or len(joints) != 6
                        or any(type(v) not in (int, float) or not math.isfinite(v) for v in joints)):
                    raise ValueError("joint_positions_rad requires six finite numbers")
                limits = {v.joint: v for v in config.robot.operating_limits}
                if any(not limits[n].lower <= v <= limits[n].upper
                       for n, v in zip(config.robot.joint_names, joints)):
                    raise ValueError("Joint target outside configured R5 limits")
                commands.append(ArmCommand(arm, "joint_position", tuple(float(v) for v in joints)))
            width = None if row["gripper"] is None else normalized_to_width(config.robot, row["gripper"])
            if not motion and joints is None and width is None:
                raise ValueError("Use commands=[] to wait; an all-null command has no target")
            if motion:
                state = observation.robot.get(arm)
                if state is None or state.tcp_pose_world_wxyz is None:
                    raise ValueError("Measured TCP is required for sparse motion")
                pose = list(state.tcp_pose_world_wxyz)
                for i, field in enumerate(FIELDS[:3]):
                    if row[field] is not None:
                        pose[i] = float(row[field])
                if any(row[f] is not None for f in FIELDS[3:6]):
                    angles = list(orientation_degrees(pose))
                    for i, field in enumerate(FIELDS[3:6]):
                        if row[field] is not None:
                            if not -180 <= row[field] <= 180:
                                raise ValueError("Orientation angles must be in [-180, 180] degrees")
                            angles[i] = row[field]
                    transform = origin_transform(pose[:3], tuple(math.radians(a) for a in angles))
                    transform[:3, :3] = transform[:3, :3] @ REFERENCE
                    pose = transform_pose(transform)
                commands.append(ArmCommand(arm, "cartesian_pose", tuple(pose)))
            if width is not None:
                commands.append(ArmCommand(arm, "gripper", (float(width),)))
        action = Action(tuple(commands), config.control_dt_s, stop)
        validate_action(action, config)
        return action
    except (ValueError, TypeError, KeyError) as exc:
        raise InvalidPolicyAction(str(exc)) from exc
