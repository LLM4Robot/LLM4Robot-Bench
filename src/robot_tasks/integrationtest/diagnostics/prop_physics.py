"""Manual fixture diagnostics, never imported by the task runner.

These checks deliberately set up fixtures or apply test drives. They do not
control a robot, do not call GPT and are NOT task-success evidence. They test
whether the declared contact geometry can support the required physical state.
"""

import numpy as np


def _pose(scene, name, pose):
    body = scene.bodies[name]
    body.set_world_pose(np.array(pose[:3]), np.array(pose[3:]))
    body.set_linear_velocity(np.zeros(3))
    body.set_angular_velocity(np.zeros(3))


def _advance(scene, seconds):
    for _ in range(round(seconds / scene.config.control_dt_s)):
        scene.backend.step(scene.config.control_dt_s)


def run(scene):
    """Return measurements, failing if a claimed passive mechanism is absent."""
    task = scene.config.task_id
    if task == "L2-04":
        from .orange_probe import run as probe_orange

        return probe_orange(scene)
    if hasattr(scene, "spec") and "native_instances" in scene.spec.parameters:
        from .native_l2_probe import run as probe_native

        return probe_native(scene)
    result = {"task": task, "scope": "fixture-only, no robot policy or success score"}
    if task == "L1-02":
        from .plug_holder_probe import run as probe_holder

        result["source_holder"] = probe_holder(scene)
    if task in {"L1-01", "L1-02", "L1-03", "L1-04", "L1-05"}:
        article = (next(a for a in scene.design.articles if a.name == "plug")
                   if task == "L1-02" else scene.design.articles[-1])
        goal = list(article.goal)
        # Native socket clearance is a seating check, not an impact-drop test.
        goal[2] += 0.0005 if task == "L1-02" else 0.008
        if task == "L1-04":
            # Check a reachable release *inside* the mouth before fingers on
            # the narrower lower body would hit the other cup's rim. This is a
            # gravity/contact diagnostic, not a scripted robot grasp or success.
            lower = scene.measure()["lower_cup"]["pose"]
            p = scene.spec.parameters
            goal[:3] = [lower[0], lower[1], lower[2] + p["cup_height"] - p["entry_depth"]]
            result["release_entry_depth_m"] = p["entry_depth"]
        _pose(scene, article.name, goal)
        _advance(scene, 4 if task == "L1-05" else 2)
        row = scene.measure()[article.name]
        support = row["fixture_force_n"] + sum(row["pair_forces"].values())
        drift = float(np.linalg.norm(np.array(row["pose"][:2]) - goal[:2]))
        result.update(
            support_n=support, horizontal_drift_m=drift, final_pose=row["pose"]
        )
        from ...tasks.bench.evaluation import BenchEvaluator
        from ...tasks.bench.scoring_l1 import RULES

        # Inspect geometry criteria without inventing a grasp witness or success.
        _, criteria, _ = RULES[task](
            BenchEvaluator(scene.spec, scene.initial), scene.measure()
        )
        result["geometry_criteria"] = criteria
        key = {
            "L1-01": "inside_holder",
            "L1-02": "inside_socket",
            "L1-03": "hook_through_head_opening",
            "L1-04": "nested",
            "L1-05": "stem_end_inside",
        }[task]
        if not criteria[key]:
            raise RuntimeError(
                "Physical resting pose rejected by geometry scorer: " + str(result)
            )
        if task == "L1-03" and not criteria["target_hook_supported"]:
            raise RuntimeError("Hat is not carried by the selected lowest peg: " + str(result))
        drift_limit = 0.02
        if task == "L1-05":
            # A rigid rose can lean until its tip rests at the cup's inner wall.
            drift_limit = scene.spec.parameters["radius"] + scene.spec.parameters["stem_radius"]
        if task == "L1-03":
            # A hanging visor naturally slides inward along a sloping peg and
            # pivots around its rear band. Bound that excursion by the actual peg
            # and visor span, not the old miniature J-hook's 8-cm heuristic.
            from ...tasks.l1.specs import local_corners

            peg_span = np.linalg.norm(np.array(scene.spec.parameters["hook_end"])
                                      - scene.spec.parameters["hook_start"])
            drift_limit = float(peg_span + np.linalg.norm(np.ptp(local_corners(article), axis=0)))
        result["horizontal_drift_limit_m"] = drift_limit
        if support < scene.spec.parameters.get("min_support_n", 0.05) or drift > drift_limit:
            raise RuntimeError("Open-cavity/hook support probe failed: " + str(result))
    else:
        result["scope"] = "initial physical scene check only; no separate fixture probe"
    result["passed"] = True
    return result
