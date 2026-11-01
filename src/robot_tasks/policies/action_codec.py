"""A closed, executable-data-only action contract for model output."""

import json
import math

from ..config import TaskConfig
from ..robots.control import width_to_joints
from ..runner import validate_action
from ..types import Action, ArmCommand, InvalidPolicyAction


def action_schema(arms: tuple[str, ...]) -> dict:
    return {"type": "object", "additionalProperties": False,
            "required": ["commands", "request_stop"], "properties": {
                "request_stop": {"type": "boolean"},
                "commands": {"type": "array", "maxItems": len(arms) * 2, "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["arm", "kind", "values", "frame"], "properties": {
                        "arm": {"type": "string", "enum": list(arms)},
                        "kind": {"type": "string", "enum": ["joint_position", "cartesian_pose", "gripper"]},
                        "values": {"type": "array", "items": {"type": "number"}, "minItems": 1, "maxItems": 7},
                        "frame": {"type": "string", "enum": ["world", "robot_base"]},
                    }}}}}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def decode_action(text: str, config: TaskConfig) -> Action:
    """Reject malformed/extra fields; never repair, clip, eval or execute code."""
    try:
        return _decode_action(text, config)
    except ValueError as exc:
        raise InvalidPolicyAction(str(exc)) from exc


def _decode_action(text: str, config: TaskConfig) -> Action:
    try:
        document = json.loads(text, object_pairs_hook=_unique_object)
    except (ValueError, TypeError):
        raise ValueError("Model action must be one valid JSON object") from None
    if not isinstance(document, dict) or set(document) != {"commands", "request_stop"}:
        raise ValueError("Model action has missing or unexpected fields")
    rows, stop = document["commands"], document["request_stop"]
    if not isinstance(rows, list) or len(rows) > 2 * len(config.command_arms) or type(stop) is not bool:
        raise ValueError("Invalid command count or request_stop type")
    if stop and rows:
        raise ValueError("A stop request cannot contain motion commands")
    commands = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"arm", "kind", "values", "frame"}:
            raise ValueError("Invalid command fields")
        if not all(isinstance(row[k], str) for k in ("arm", "kind", "frame")):
            raise ValueError("Command arm/kind/frame must be strings")
        kind, values = row["kind"], row["values"]
        if (kind not in {"joint_position", "cartesian_pose", "gripper"}
                or not isinstance(values, list)
                or len(values) != {"joint_position": 6, "cartesian_pose": 7, "gripper": 1}[kind]
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)):
            raise ValueError("Command values have wrong count or contain nonfinite/non-numeric values")
        if row["frame"] not in {"world", "robot_base"}:
            raise ValueError("Unsupported command frame")
        if kind == "gripper":
            width_to_joints(config.robot, values[0])
        elif kind == "joint_position":
            limits = {v.joint: v for v in config.robot.operating_limits}
            if any(not limits[name].lower <= v <= limits[name].upper
                   for name, v in zip(config.robot.joint_names, values)):
                raise ValueError("Joint target outside configured R5 limits")
        commands.append(ArmCommand(row["arm"], kind, tuple(float(v) for v in values), row["frame"]))
    action = Action(tuple(commands), config.control_dt_s, stop)
    validate_action(action, config)
    return action
