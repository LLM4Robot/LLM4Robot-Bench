#!/usr/bin/env python3
"""Explicit bottle initial-state experiment; check never starts Isaac or a model.

The frozen benchmark and its assets are not edited. The existing closure repair
offset is overridden only in this single episode process, before resolving any
scene, scoring geometry or model inputs. The override is recorded with the run.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def load_experiment(path):
    value = json.loads(Path(path).read_text())
    fields = {'schema_version', 'kind', 'experiment_id', 'base_task',
              'preloosen_degrees', 'gpu_index', 'max_calls', 'max_wall_time_s'}
    if set(value) != fields or value['schema_version'] != 1 or value['kind'] != 'bottle_preloosen':
        raise ValueError('Unsupported experimental configuration')
    if value['base_task'] != 'L3-01':
        raise ValueError('This experiment only supports original L3-01')
    angle = value['preloosen_degrees']
    if type(angle) not in (int, float) or not math.isfinite(angle) or not 0 <= angle <= 300:
        raise ValueError('preloosen_degrees must be a finite number in [0, 300]')
    if value['experiment_id'] != f'L3-01_preloosen{angle:g}':
        raise ValueError('Experiment ID must disclose the preloosen angle')
    # This deployment was explicitly prepared for physical GPU 0; do not silently
    # remap a different Vulkan device through CUDA_VISIBLE_DEVICES.
    if type(value['gpu_index']) is not int or value['gpu_index'] != 0:
        raise ValueError('This prepared deployment uses physical GPU 0')
    if type(value['max_calls']) is not int or not 1 <= value['max_calls'] <= 1000:
        raise ValueError('max_calls must be in [1, 1000]')
    wall = value['max_wall_time_s']
    if type(wall) not in (int, float) or not math.isfinite(wall) or wall < 0:
        raise ValueError('max_wall_time_s must be finite and nonnegative')
    return value


@contextmanager
def engagement(angle):
    from robot_tasks.tasks.bench import bottle_contacts
    original = bottle_contacts.REENGAGE_M
    bottle_contacts.REENGAGE_M = bottle_contacts.THREAD_LEAD_M * (300 - angle) / 360
    try:
        yield
    finally:
        bottle_contacts.REENGAGE_M = original


def prepare(value):
    from robot_tasks.registry import resolve_task
    spec, config = resolve_task(value['base_task'])
    raw = deepcopy(config.raw)
    raw['task']['experimental_initial_state'] = {
        'experiment_id': value['experiment_id'],
        'preloosen_degrees': value['preloosen_degrees'],
        'formal_default_preloosen_degrees': 180.0,
        'source_asset_preloosen_degrees': 300.0,
        'not_a_scored_paper_episode': True,
    }
    raw['simulation'].update(active_gpu=value['gpu_index'], physics_gpu=value['gpu_index'])
    raw['execution']['max_decisions'] = value['max_calls']
    raw['runtime']['max_wall_time_s'] = value['max_wall_time_s']
    return spec, replace(config, raw=raw, max_wall_time_s=value['max_wall_time_s'])


def inspect(value):
    import numpy as np
    from robot_tasks.initial_layout import scene_record
    from robot_tasks.robots.transforms import pose_transform
    from robot_tasks.tasks.bench.arm_selection import resolve_bench_spec
    from robot_tasks.policies.model_config import load_gpt_settings
    with engagement(value['preloosen_degrees']):
        spec, config = prepare(value)
        prepared = spec.factory.prepare_config(config)
        row = scene_record(config.raw['task'])
        body = pose_transform(row['tracked']['bottle']['pose'])
        cap = pose_transform(row['tracked']['cap']['pose'])
        relative = np.linalg.inv(body) @ cap
        angle = math.radians(value['preloosen_degrees'])
        expected = pose_transform((0, 0, .009 * value['preloosen_degrees'] / 360,
                                   math.cos(angle / 2), 0, 0, math.sin(angle / 2)))
        np.testing.assert_allclose(relative, expected, atol=1e-6)
        scoring = resolve_bench_spec(config.raw['task'], config.robot)
        article = next(a for a in scoring.design.articles if a.name == 'cap')
        np.testing.assert_allclose(article.initial, row['tracked']['cap']['pose'], atol=1e-10)
        model = load_gpt_settings(ROOT / 'configs/models/gpt.toml')
        return {**value, 'initial_cap_lift_m': float(relative[2, 3]),
                'scene_and_scoring_initial_pose_match': True,
                'instruction': prepared.instruction, 'active_arms': prepared.arms,
                'model': model.model, 'reasoning_effort': model.reasoning_effort,
                'physics_dt_s': prepared.physics_dt_s,
                'base_config_sha256': hashlib.sha256(config.source.read_bytes()).hexdigest(),
                'experimental_initial_state': config.raw['task']['experimental_initial_state'],
                'limitation': '270-degree preloosen diagnostic on ROG allowed direct upward pull-off; enforced screw removal is not established.',
                'simulation_started': False, 'model_requests': 0}


def require_idle_gpu(index):
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used',
                                    '--format=csv,noheader,nounits'], text=True, timeout=15)
    rows = [line.split(',') for line in text.splitlines() if line.strip()]
    gpu = next((r for r in rows if int(r[0]) == index), None)
    if gpu is None or int(gpu[2]) > 256:
        raise RuntimeError(f'GPU {index} is missing or in use; experiment not started')
    jobs = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader'], text=True, timeout=15)
    if gpu[1].strip() in jobs:
        raise RuntimeError(f'GPU {index} has an active compute process; experiment not started')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['check', 'run'])
    parser.add_argument('--experiment-config', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--max-calls', type=int)
    parser.add_argument('--wall-time-s', type=float)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    value = load_experiment(args.experiment_config)
    if args.max_calls is not None:
        if not 1 <= args.max_calls <= 1000:parser.error('--max-calls must be in [1, 1000]')
        value['max_calls'] = args.max_calls
    if args.wall_time_s is not None:
        if not math.isfinite(args.wall_time_s) or args.wall_time_s < 0:parser.error('Invalid wall time')
        value['max_wall_time_s'] = args.wall_time_s
    if args.action == 'run' and not args.worker:
        from robot_tasks.runtime_config import load_runtime
        from robot_tasks.batch import run_job
        if args.output is None:parser.error('run requires --output')
        require_idle_gpu(value['gpu_index'])
        python = load_runtime().python
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(value['gpu_index']))
        args.output.mkdir(parents=True, exist_ok=False)
        command = [python, str(Path(__file__).resolve()), *sys.argv[1:], '--worker']
        wall = value['max_wall_time_s']
        if wall:
            # Count simulator startup, model waiting and execution against the
            # same budget. The runner checks this deadline during model calls.
            env['GPT_BENCHMARK_WALL_DEADLINE_MONOTONIC'] = str(time.monotonic() + wall)
        def started(record):
            (args.output/'worker.json').write_text(json.dumps(record, indent=2) + '\n')
        # Keep the framework supervisor alive: Isaac's python.sh starts a child
        # instead of exec-ing it. Reuse the benchmark's bounded tree cleanup so
        # fleet stop also reaps the simulator and separately sessioned planner.
        process = run_job(command, args.output/'process', env,
                          wall + 300 if wall else None, started)
        # The extra supervisor time is only for cancellation, flushing videos
        # and cleanup, matching batch_run; the episode deadline remains wall.
        started(process)
        return process['returncode'] if process['cleanup_complete'] else 2
    report = inspect(value)
    if args.action == 'check':
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(report, indent=2));return 0
    if args.output is None:parser.error('run requires --output')
    if not os.environ.get('GPT_BENCHMARK_CODEX_BRIDGE_URL'):
        parser.error('This prepared experiment requires the local Codex SSH bridge')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'experiment.json').write_text(json.dumps(report, indent=2) + '\n')
    from robot_tasks.execution import execute_task
    with engagement(value['preloosen_degrees']):
        spec, config = prepare(value)
        directory, result = execute_task(spec, config, args.output)
    print(json.dumps({'output': str(directory), 'experiment_id': value['experiment_id'], **result}, indent=2))
    return 0 if result['status'] == 'finished' else 2


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
