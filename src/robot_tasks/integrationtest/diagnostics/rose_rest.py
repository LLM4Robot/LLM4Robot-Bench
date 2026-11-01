"""L1-05 passive startup plus 30 seconds of free rest; no model or prop forcing."""

import argparse
import json
from pathlib import Path

import numpy as np

from .scene_smoke import check
from ...robots.transforms import pose_transform


def probe(backend, task, output):
    task.initialize_episode(
        backend, None, lambda **_: backend.step(task.scene.config.control_dt_s))
    ready = backend.scene_report['household']['initial_prop_settling']
    start = backend.time_s()
    anchor = pose_transform(task.scene.measure()['rose']['pose'])
    rows = []
    for _ in range(round(30 / task.scene.config.control_dt_s)):
        backend.step(task.scene.config.control_dt_s)
        r = task.scene.measure()['rose']
        t = pose_transform(r['pose'])
        rows.append({'time_s': backend.time_s()-start,
                     'linear_speed_m_s': float(np.linalg.norm(r['linear_velocity'])),
                     'angular_speed_rad_s': float(np.linalg.norm(r['angular_velocity'])),
                     'position_drift_m': float(np.linalg.norm(t[:3,3]-anchor[:3,3])),
                     'rotation_drift_rad': float(np.arccos(np.clip(
                         (np.trace(anchor[:3,:3].T @ t[:3,:3])-1)/2,-1,1))),
                     'finger_force_n': max(r['finger_forces']),
                     'bench_force_n': r['bench_force_n']})
    limits = {'linear_speed_m_s': .001, 'angular_speed_rad_s': .05,
              'position_drift_m': .0005, 'rotation_drift_rad': .02,
              'finger_force_n': .01}
    peaks = {key: max(r[key] for r in rows) for key in limits}
    passed = all(peaks[key] < limit for key,limit in limits.items())
    result = {'passed': passed, 'gpt_called': False, 'startup': ready,
              'duration_s': 30., 'peaks': peaks, 'limits': limits,
              'pose_or_velocity_forced': False}
    (output/'rest_trace.json').write_text(json.dumps(rows,indent=2))
    (output/'rest_result.json').write_text(json.dumps(result,indent=2))
    if not passed:
        raise AssertionError('Rose did not remain at rest after startup: '+str(peaks))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    return check('L1-05',args.output,prepare=False,capture_hook=probe)


if __name__ == '__main__':
    raise SystemExit(main())
