"""Passive seating and force-only extraction diagnostic; never model execution."""

import numpy as np

from ...robots.transforms import pose_transform


def run(scene):
    for _ in range(round(3 / scene.config.control_dt_s)):
        scene.backend.step(scene.config.control_dt_s)
    rows = scene.measure()
    plug, base = rows["plug"], rows["plug_storage_base"]
    support = plug["pair_forces"]["plug_storage_base"]
    relative = np.linalg.inv(pose_transform(base["pose"])) @ pose_transform(plug["pose"])
    result = {
        "scope": "fixture diagnostic only; 0.65 N upward force at charger COM, no robot grasp",
        "charger_support_n": support,
        "base_bench_support_n": base["bench_force_n"],
        "seated_relative_origin_m": relative[:3, 3].tolist(),
        "seated_axis_cosine": float(relative[2, 2]),
    }
    if not (support > .25 and base["bench_force_n"] > .25 and relative[2, 2] > .98
            and np.linalg.norm(relative[:2, 3]) < .003 and 0 <= relative[2, 3] < .006):
        raise RuntimeError("Charger not naturally supported upright in holder: " + str(result))

    initial_base = np.asarray(base["pose"][:3])
    previous = scene.backend.before_physics_step

    def pull(dt):
        if previous is not None:
            previous(dt)
        scene.pairs["plug"].apply_forces_and_torques_at_pos(
            forces=np.array([[0., 0., .65]], dtype=np.float32), is_global=True)

    cleared = False
    max_base_drift = 0.
    scene.backend.before_physics_step = pull
    try:
        for _ in range(round(1 / scene.config.control_dt_s)):
            scene.backend.step(scene.config.control_dt_s)
            rows = scene.measure()
            base_pos = np.asarray(rows["plug_storage_base"]["pose"][:3])
            max_base_drift = max(max_base_drift, float(np.linalg.norm(base_pos - initial_base)))
            if min(c[2] for c in rows["plug"]["corners"]) > base_pos[2] + .057:
                cleared = True
                break
    finally:
        scene.backend.before_physics_step = previous
    result.update(extracted=cleared, max_base_drift_m=max_base_drift,
                  passed=cleared and max_base_drift < .003)
    if not result["passed"]:
        raise RuntimeError("Charger extraction/holder separation failed: " + str(result))
    return result
