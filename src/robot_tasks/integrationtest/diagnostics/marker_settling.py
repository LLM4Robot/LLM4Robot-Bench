"""Passive L1-01 contact regression; no GPT or autonomous success evidence.

Run in Isaac Python with -m robot_tasks.integrationtest.diagnostics.marker_settling.
The diagnostic explicitly poses/spins the marker, measures passive decay, then
restores the initial scene. Production runners never import this module.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from ...paths import PROJECT_ROOT
from ...robots.transforms import pose_transform
from .scene_smoke import check


def probe(backend, task, output):
    from ...tasks.bench.evaluation import BenchEvaluator
    from ...tasks.bench.scoring_l1 import marker

    scene = task.scene
    body = scene.bodies["marker"]
    p = scene.spec.parameters
    theta = np.deg2rad(25.77648)
    # Released leaning pose observed in the failed run. Both spin directions
    # must settle without finger/table support, while contact reporting stays on.
    leaning = [p["holder_xy"][0], p["holder_xy"][1] + .0186585,
               p["floor"] + .0035224, np.cos(theta / 2), np.sin(theta / 2), 0, 0]
    upright = [*p["holder_xy"], p["floor"] + .008, 1, 0, 0, 0]
    cases = []
    for name, pose, spin in (("upright_drop", upright, 0),
                             ("leaning_spin_positive", leaning, 10),
                             ("leaning_spin_negative", leaning, -10)):
        body.set_world_pose(np.array(pose[:3]), np.array(pose[3:]))
        body.set_linear_velocity(np.zeros(3))
        body.set_angular_velocity(pose_transform(pose)[:3, 2] * spin)
        samples = []
        for i in range(round(15 / scene.config.control_dt_s)):
            backend.step(scene.config.control_dt_s)
            row = scene.measure()["marker"]
            samples.append({
                "time_s": (i + 1) * scene.config.control_dt_s,
                "pose": row["pose"],
                "angular_speed_rad_s": float(np.linalg.norm(row["angular_velocity"])),
                "linear_speed_m_s": float(np.linalg.norm(row["linear_velocity"])),
                "cup_support_n": row["fixture_force_n"],
                "bench_support_n": row["bench_force_n"],
                "finger_force_n": max(f for fs in row["fingers"].values() for f in fs),
            })
        tail = [s for s in samples if s["time_s"] > 13]
        passed = all(s["angular_speed_rad_s"] < .3 and s["linear_speed_m_s"] < .025
                     and s["cup_support_n"] > p["min_support_n"]
                     and s["bench_support_n"] < .05 and s["finger_force_n"] < .05 for s in tail)
        goal_met, criteria, _ = marker(BenchEvaluator(scene.spec, scene.initial), {"marker": row})
        entry = {
            "name": name, "passed": passed,
            "final_angular_speed_rad_s": samples[-1]["angular_speed_rad_s"],
            "tail_max_angular_speed_rad_s": max(s["angular_speed_rad_s"] for s in tail),
            "tail_min_cup_support_n": min(s["cup_support_n"] for s in tail),
            "final_geometry_criteria": criteria,
            "final_goal_conditions_met": bool(goal_met),
            "samples": samples,
        }
        cases.append(entry)
        print("MARKER_CONTACT_CASE", name, "PASS" if passed else "FAIL",
              "final_omega", entry["final_angular_speed_rad_s"], flush=True)
    report = {
        "scope": "passive contact settling, not autonomous task completion or geometry acceptance",
        "gpt_called": False, "passed": all(c["passed"] for c in cases),
        "cases": cases, "scene_contacts": backend.scene_report["household"]["contact_materials"],
    }
    (output / "marker_settling.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    task.reset(backend, 0)
    for _ in range(24):
        backend.step(scene.config.control_dt_s)
    return {k: v for k, v in report.items() if k != "cases"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "build/marker_settling")
    args = parser.parse_args()
    code = check("L1-01", args.output, prepare=False, capture_hook=probe)
    report = args.output / "marker_settling.json"
    passed = report.exists() and json.loads(report.read_text(encoding="utf-8"))["passed"]
    return 0 if code == 0 and passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
