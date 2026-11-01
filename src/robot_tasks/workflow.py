"""One GPT workflow with an explicit, opt-in depth observation channel."""

from copy import deepcopy
from pathlib import Path
import tomllib


SECTIONS = {"runtime", "observation", "recording", "execution", "policy", "simulation"}


def workflow_metadata(config=None) -> dict:
    """Describe implemented behavior for inspection and run records, not switches."""
    depth = config.include_depth if config is not None else False
    constrained = config is not None and config.grasp_check_mode == "contact_inspection"
    guard = depth and not constrained
    return {
        "version": 14, "action_interface": "sparse_eef_normalized_gripper", "ik": "curobo",
        "generation_retries": "unlimited retries; 2/4/8s backoff then 8s; unchanged observation; shared configured call and wall-time budgets; user cancellation supported",
        "motion": "collision_checked_joint_trajectory",
        "stop_protocol": "evaluator_gated_continue_within_total_call_budget",
        "planner_information": "privileged collision geometry; no task goal or score",
        "prompt_protocol": "shared_observe_plan_verify_task_goal_v6",
        "task_prompt_protocol": (config.raw.get("task", {}).get("prompt_protocol")
                                 if config is not None else None),
        "task_reference": ({key: value for key, value in config.raw.get("task", {}).items()
                            if key.startswith("reference_")} if config is not None else {}),
        "grasp_check": ("rgbd_surface_comotion_v3_tracking_recovery" if guard else
                        "contact_inspection_no_lift_gate" if constrained else "model_visual_inspection"),
        "release_check": "rgbd_stationary_vs_following_advisory_v1" if guard else None,
        "grasp_check_information": (
            "visible RGB-D, measured TCP/opening, known robot cavity and bench height; no object truth or contacts"
            if guard else "model inspects current images and measured robot state; no automatic depth gate"),
        "observation_modalities": ["rgb"] + (["depth"] if depth else []) + ["camera_calibration", "robot_state", "feedback"],
        "retain_text_history": True, "operating_notes": True, "visual_aids": True,
        "decision_notes": True, "gripper_timing": "after_motion",
        "active_observation": "according_to_task_arm_roles",
        "startup_preparation": "hold_joint_home_v1",
        "observation_preparation": "model_controlled_from_joint_home",
    }


def _fixed_inputs(raw: dict) -> dict:
    if "policy_options" in raw:
        raise ValueError("Remove [policy_options]: GPT workflow capabilities are always enabled")
    if "gripper_timing" in raw.get("execution", {}):
        raise ValueError("Remove execution.gripper_timing: motion always precedes the gripper command")
    result = deepcopy(raw)
    observation = result.setdefault("observation", {})
    depth = observation.setdefault("include_depth", False)
    if type(depth) is not bool:
        raise ValueError("observation.include_depth must be true or false")
    for key in ("include_robot_state", "include_feedback"):
        if key in observation and observation[key] is not True:
            raise ValueError(f"observation.{key} is required by the fixed GPT workflow; remove this override")
        observation[key] = True
    return result


def validate_gpt_workflow(config) -> None:
    """Also validate programmatic callers before constructing a paid request."""
    _fixed_inputs(config.raw)
    if not (config.include_robot_state and config.include_feedback):
        raise ValueError("GPT workflow requires RGB, measured robot state and execution feedback")
    if config.save_depth and not config.include_depth:
        raise ValueError("recording.save_depth requires observation.include_depth")
    if not config.raw.get("simulation", {}).get("camera_profile"):
        raise ValueError("GPT workflow requires a camera rig profile with calibration")


def load_workflow(source: Path, task: dict) -> tuple[dict, Path | None, dict | None]:
    """Task fields override profile fields. A profile cannot define task assets/roles.

    Keep this to one explicit profile, without recursive inheritance or hidden defaults.
    File references remain in the task so their base directory is unambiguous.
    """
    reference = task.get("workflow_profile")
    if reference is None:
        return _fixed_inputs(task), None, None
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("workflow_profile must reference a TOML file")
    path = (source.parent / reference).resolve()
    with path.open("rb") as stream:
        profile = tomllib.load(stream)
    if not profile or set(profile) - SECTIONS:
        raise ValueError("Workflow profiles may only define experiment parameters and shared camera settings")
    if any(not isinstance(v, dict) for v in profile.values()):
        raise ValueError("Workflow profile sections must be tables")
    if set(profile.get("policy", {})) - {"reasoning_effort"}:
        raise ValueError("Workflow policy only configures reasoning_effort; model_config belongs to the task")
    if set(profile.get("simulation", {})) - {"camera_position", "camera_target", "camera_reference_frame", "camera_resolution", "camera_intrinsics"}:
        raise ValueError("Workflow simulation only configures the overview camera; scene assets belong to the task")
    # Validate both sources so a task cannot mask obsolete profile switches.
    _fixed_inputs(profile)
    _fixed_inputs(task)
    merged = deepcopy(profile)
    for section, value in task.items():
        if section in profile:
            if not isinstance(value, dict):
                raise ValueError(f"Task override {section} must be a table")
            merged[section].update(deepcopy(value))
        else:
            merged[section] = deepcopy(value)
    return _fixed_inputs(merged), path, profile
