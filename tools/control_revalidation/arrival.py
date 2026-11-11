"""Measured arrival feedback, advisory only and never an episode-stop rule."""
from dataclasses import asdict, replace
import math


def annotate_arrival(feedback, previous_action=None, reporting_tolerance_m=0.0005):
    """Keep every arm's measured residual, including unchanged held targets.

    No object poses, contact forces or evaluator flags are inspected. A native
    completed command need not have reached its target precisely, and neither
    completion nor blocked gripper closure proves that an object is held.
    """
    if not math.isfinite(reporting_tolerance_m) or reporting_tolerance_m <= 0:
        raise ValueError("Reporting tolerance must be finite and positive")
    if feedback.status != "completed":
        return feedback
    commanded = {c.arm for c in previous_action.commands} if previous_action else set()
    rows = {}
    for arm, values in feedback.metrics.get("tcp", {}).items():
        position = values.get("position_error_m")
        if position is None:
            continue
        if not math.isfinite(position) or position < 0:
            raise ValueError("Invalid measured arrival residual")
        rows[arm] = {
            "position_error_m": position,
            "orientation_error_rad": values.get("orientation_error_rad"),
            "requested_pose_world_wxyz": values.get("requested_pose_world_wxyz"),
            "measured_pose_world_wxyz": values.get("measured_pose_world_wxyz"),
            "newly_commanded_arm": arm in commanded,
            "within_reporting_tolerance": position <= reporting_tolerance_m,
        }
    metrics = {**feedback.metrics, "arrival_feedback": {
        "arms": rows,
        "reporting_tolerance_m": reporting_tolerance_m,
        "episode_stop_on_residual": False,
        "native_completed_is_not_task_success": True,
        "holding_confirmed": None,
    }}
    if any(not row["within_reporting_tolerance"] for row in rows.values()):
        return replace(feedback, status="partial_execution", metrics=metrics,
                       message="Native execution ended with a measured target residual. "
                       "Inspect the measured poses and per-arm errors before the next action. "
                       "This is not precise arrival, proof of holding or task success; "
                       "the reporting threshold does not terminate the episode.")
    return replace(feedback, metrics=metrics)


def policy_class():
    """Load model code only when a caller explicitly constructs a policy."""
    from robot_tasks.policies.gpt import GPTPolicy

    class ArrivalFeedbackPolicy(GPTPolicy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._previous_action = None

        def reset(self, seed):
            super().reset(seed)
            self._previous_action = None

        def act(self, context):
            feedback = annotate_arrival(context.observation.feedback, self._previous_action)
            observed = replace(context.observation, feedback=feedback)
            action = super().act(replace(context, observation=observed))
            self._previous_action = action
            return action

    return ArrivalFeedbackPolicy
