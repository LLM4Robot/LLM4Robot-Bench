"""Explain rejected plans without changing collision checks or executing recovery."""

import math


class PlanningFailure(ValueError):
    """JSON-safe planner diagnostic preserved across the worker process boundary."""

    def __init__(self, diagnostic):
        self.diagnostic = diagnostic
        super().__init__(
            f"{diagnostic['arm']}: [{diagnostic['code']}] {diagnostic['message']}"
        )


def failure(arm, code, message, **details):
    return PlanningFailure({
        "arm": arm,
        "code": code,
        "message": message + " No commands from this action were applied.",
        "source": "planner collision model and finite solver search; not task scoring",
        **details,
    })


def state_violations(planner, state):
    """Use the pinned planner's feasibility constraints, not optimization costs.

    The auxiliary rollout avoids resizing/replaying the main planning CUDA graph.
    Constraint values can be weighted or binary; they are NOT penetration metres.
    """
    graph = planner.graph_planner
    if graph is None:
        raise RuntimeError("Planner has no state feasibility diagnostic")
    metrics = graph.auxiliary_rollout.compute_metrics_from_action(
        state.position.unsqueeze(1)
    )
    constraints = metrics.costs_and_constraints.constraints
    violations = []
    for name, values in zip(constraints.names, constraints.values, strict=True):
        maximum = float(values.max().item())
        if not math.isfinite(maximum):
            raise ValueError(f"Non-finite planner constraint: {name}")
        if maximum > 0:
            violations.append(name)
    return violations


def state_failure(arm, stage, violations):
    collision = bool({"self_collision", "scene_collision"}.intersection(violations))
    code = f"{stage}_collision" if collision else f"{stage}_state_invalid"
    detail = ", ".join(violations)
    if stage == "start":
        message = f"Current measured start state is invalid in the planner ({detail}). "
        if "self_collision" in violations:
            message += (
                "The robot's approximate collision shapes overlap within this arm. "
                "This is not proof of physical mesh contact. Changing this arm's goal "
                "or camera view alone cannot clear the reported self-collision; "
                "execution-side inspection/recovery is required."
            )
        else:
            message += (
                "Inspect the current state/obstacle constraint before retrying; "
                "a different goal from the unchanged invalid start may also fail."
            )
    else:
        message = (
            f"Requested joint goal violates planner constraints ({detail}). "
            "Choose a different goal; the current start state passed the diagnostic."
        )
    return failure(arm, code, message, stage=stage, violated_constraints=violations)


def diagnose_failure(planner, arm, kind, start, goal, result, *,
                     goal_distance_m=None, reach_bound_m=None):
    """Diagnose only after a plan fails, preserving successful planning behavior.

    A None plan can mean failed IK OR failed graph seeding. A separate, bounded
    IK probe is needed to distinguish these; neither finite search proves that
    a reachable solution/path cannot exist.
    """
    try:
        violations = state_violations(planner, start)
        if violations:
            return state_failure(arm, "start", violations)
        if kind == "joint_position":
            violations = state_violations(planner, goal)
            if violations:
                return state_failure(arm, "goal", violations)
        elif kind == "cartesian_pose":
            if (goal_distance_m is not None and reach_bound_m is not None
                    and goal_distance_m > reach_bound_m + 1e-6):
                return failure(
                    arm, "goal_out_of_reach",
                    "Requested TCP position exceeds the URDF chain's conservative "
                    "maximum reach from the robot base. Choose a closer position.",
                    stage="goal", goal_distance_m=goal_distance_m,
                    reach_bound_m=reach_bound_m,
                )
            if result is None:
                ik = planner.ik_solver.solve_pose(
                    goal, return_seeds=1, current_state=start
                )
                if not bool(ik.success.any()):
                    return failure(
                        arm, "goal_ik_not_found",
                        "No collision-free goal IK solution was found in the finite "
                        "search. This can reflect position, orientation, limits, "
                        "collision or nonconvergence; it does not prove global "
                        "unreachability. Revise the target position/orientation.",
                        stage="goal",
                    )
        return failure(
            arm, "path_not_found",
            "The start state passed the diagnostic, but no valid trajectory was "
            "found within the planning attempts. This is not proof that no path "
            "exists. Consider an intermediate pose or a different approach.",
            stage="path",
        )
    except Exception as exc:
        # A failed diagnostic must not mislabel the original rejection as a
        # collision/unreachable target, or allow an unvalidated motion through.
        return failure(
            arm, "planning_failed_unknown",
            "Planning failed and its diagnostic was unavailable. The cause is "
            "unclassified; do not infer collision or target unreachability.",
            stage="unknown", diagnostic_error=f"{type(exc).__name__}: {exc}",
        )
