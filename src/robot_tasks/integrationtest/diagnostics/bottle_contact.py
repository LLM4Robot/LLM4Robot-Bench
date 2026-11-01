"""Independent closure load/drop regression; never a task policy or GPT trial."""

import argparse
import json
from pathlib import Path

import numpy as np

from ...paths import PROJECT_ROOT
from ...robots.transforms import pose_transform
from .scene_smoke import check


def probe(backend, task, output):
    from PIL import Image

    scene = task.scene
    cap = scene.bodies['cap']
    initial = np.asarray(scene.initial['cap'])
    initial_transform = pose_transform(initial)
    # The delivered cap root is near the bottle base, ~15 cm below the cap.
    # Its apparent translation under small tilts is not displacement of the cap.
    initial_center = (initial_transform[:3, :3] @ np.mean(scene.bounds['cap'], axis=0)
                      + initial_transform[:3, 3])
    config = scene.config
    previous_hook = backend.before_physics_step
    cases = (
        ('rest', (0., 0., 0.), (0., 0., 0.)),
        ('side_005N', (.05, 0., 0.), (0., 0., 0.)),
        ('down_1N', (0., 0., -1.), (0., 0., 0.)),
        ('down_5N', (0., 0., -5.), (0., 0., 0.)),
        ('up_015N', (0., 0., .15), (0., 0., 0.)),
        ('unscrew', (0., 0., .055), (0., 0., .003)),
        ('drop_above_neck', (0., 0., 0.), (0., 0., 0.)),
    )
    report = {
        'scope': 'isolated external loads and drop; not robot grasp or task success',
        'gpt_called': False, 'physics_dt_s': config.physics_dt_s,
        'sample_dt_s': config.control_dt_s, 'initial_pose': initial.tolist(),
        'initial_center_world_m': initial_center.tolist(),
        'displacement_reference': 'center of the cap geometry, not the offset USD root',
        'physics_overrides': backend.scene_report['household']['physics_overrides'],
        'cases': [], 'passed': False,
    }

    def sample():
        row = scene.measure()['cap']
        return {k: row[k] for k in ('pose', 'corners', 'linear_velocity',
                                   'angular_velocity', 'pair_forces')}

    try:
        for name, force, torque in cases:
            backend.before_physics_step = previous_hook
            cap.set_world_pose(initial[:3], initial[3:])
            cap.set_linear_velocity(np.zeros(3))
            cap.set_angular_velocity(np.zeros(3))
            for _ in range(round(1 / config.control_dt_s)):
                backend.step(config.control_dt_s)
            if name == 'drop_above_neck':
                elevated = initial.copy()
                elevated[2] += .05
                cap.set_world_pose(elevated[:3], elevated[3:])
                cap.set_linear_velocity(np.zeros(3))
                cap.set_angular_velocity(np.zeros(3))
                scene.cached_time = None
            before = sample()
            start = backend.time_s()

            def load(dt):
                if previous_hook is not None:
                    previous_hook(dt)
                if backend.time_s() - start < .5:
                    cap._rigid_prim_view.apply_forces_and_torques_at_pos(
                        forces=np.asarray([force], dtype=np.float32),
                        torques=np.asarray([torque], dtype=np.float32), is_global=True)

            backend.before_physics_step = load
            samples = []
            for _ in range(round(2 / config.control_dt_s)):
                backend.step(config.control_dt_s)
                samples.append(sample())
            backend.before_physics_step = previous_hook
            positions = np.asarray([np.mean(s['corners'], axis=0) for s in samples])
            dz = positions[:, 2] - initial_center[2]
            final = samples[-1]
            checks = {'finite': bool(np.isfinite(positions).all())}
            if name == 'unscrew':
                # A deliberately strong torque pulse checks that the repair
                # has not glued the cap to the bottle. It is not a realistic
                # gripper trajectory or an opening-torque calibration.
                checks.update(released=bool(dz.max() > .02),
                              above_table=bool(np.min(np.asarray(final['corners'])[:, 2]) > .879),
                              bottle_contact_ended=final['pair_forces']['bottle'] < .01)
            elif name == 'drop_above_neck':
                checks.update(no_deep_neck_penetration=bool(dz.min() > -.001),
                              settled=bool(np.linalg.norm(final['linear_velocity']) < .02),
                              supported=final['pair_forces']['bottle'] > .01)
            else:
                checks.update(retained=bool(np.max(np.linalg.norm(positions - initial_center, axis=1)) < .002),
                              no_deep_neck_penetration=bool(dz.min() > -.001),
                              settled=bool(np.linalg.norm(final['linear_velocity']) < .02))
            case = {'name': name, 'force_n': force, 'torque_nm': torque,
                    'force_duration_s': .5, 'before': before, 'samples': samples,
                    'min_delta_z_m': float(dz.min()), 'max_delta_z_m': float(dz.max()),
                    'checks': checks, 'passed': all(checks.values())}
            report['cases'].append(case)
            (output / 'bottle_contact.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
            print('BOTTLE_CONTACT', name, 'PASS' if case['passed'] else 'FAIL', checks, flush=True)
            if name in {'unscrew', 'drop_above_neck'}:
                for frame in backend.capture_rgb():
                    Image.frombytes('RGB', (frame.width, frame.height), frame.data).save(
                        output / f'{name}_{frame.camera}.png')
        report['passed'] = all(c['passed'] for c in report['cases'])
        (output / 'bottle_contact.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        return {k: v for k, v in report.items() if k != 'cases'}
    finally:
        backend.before_physics_step = previous_hook
        task.reset(backend, config.seed)
        for _ in range(round(2 / config.control_dt_s)):
            backend.step(config.control_dt_s)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', nargs='?', default='L3-01',
                        choices=('L3-01', 'L3-01_variant01', 'L3-01_v3'))
    parser.add_argument('--output', type=Path, default=PROJECT_ROOT / 'build/bottle_contact')
    args = parser.parse_args()
    code = check(args.task, args.output, prepare=False, capture_hook=probe)
    result = args.output / 'bottle_contact.json'
    passed = result.exists() and json.loads(result.read_text(encoding='utf-8'))['passed']
    return 0 if code == 0 and passed else 2


if __name__ == '__main__':
    raise SystemExit(main())
