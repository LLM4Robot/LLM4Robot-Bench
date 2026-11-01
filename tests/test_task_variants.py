"""Third-round IDs keep original physics and rules; no model calls."""
from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest

from robot_tasks.batch import freeze_config
from robot_tasks.execution import execute_task
from robot_tasks.initial_layout import layout, scene_record
from robot_tasks.paths import PROJECT_ROOT
from robot_tasks.registry import TASKS, resolve_task
from robot_tasks.tasks.bench.arm_selection import resolve_bench_spec


BASES = ('L2-01', 'L2-02', 'L2-03', 'L2-04', 'L2-05', 'L2-06')


class TaskVariantTests(unittest.TestCase):
    def test_registration_and_original_rules_are_preserved(self):
        self.assertEqual({s.base_task_id for s in TASKS.values()
                          if s.base_task_id and s.base_task_id.startswith('L2-')}, set(BASES) | {'L2-07'})
        for base in BASES:
            with self.subTest(base=base):
                _, original = resolve_task(base)
                spec, variant = resolve_task(base + '_v3')
                self.assertEqual(variant.task_id, base)
                self.assertEqual(spec.base_task_id, base)
                self.assertEqual(variant.experiment_id, base + '_v3')
                a, b = deepcopy(original.raw), deepcopy(variant.raw)
                b['task'].pop('variant_id'); b['task'].pop('initial_layout')
                for raw in (a, b):
                    for key in ('launch_command', 'launch_command_conda'):
                        raw['task'].pop(key, None)
                self.assertEqual(a, b)
                self.assertEqual(spec.factory.prepare_config(original).instruction,
                                 spec.factory.prepare_config(variant).instruction)
                before = resolve_bench_spec(original.raw['task'], original.robot)
                after = resolve_bench_spec(variant.raw['task'], variant.robot)
                for key in before.parameters:
                    if key not in ('outward_axis_world', 'closed_pose'):
                        self.assertEqual(before.parameters[key], after.parameters[key], key)
                reviewed = {**original.raw['task'], 'initial_layout': 'l2_p01'}
                self.assertEqual(scene_record(reviewed), scene_record(variant.raw['task']))

    def test_snapshot_dispatch_and_episode_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for base in BASES:
                tid = base + '_v3'
                path = root / (tid + '.toml')
                freeze_config(tid, path, PROJECT_ROOT / 'configs/models/gpt.toml',
                              PROJECT_ROOT / 'configs/workflows/gpt_default.toml',
                              {'reasoning_effort': 'xhigh', 'seed': 0,
                               'wall_time_s': 18000, 'max_calls': 0})
                spec, config = resolve_task(tid, path)
                directory, result = execute_task(spec, config, root, dry_run=True, step_limit=1)
                self.assertTrue(directory.name.startswith(tid + '_dry_run_'))
                self.assertEqual(result['task_id'], tid)
                self.assertEqual(result['base_task_id'], base)
                manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
                self.assertEqual(manifest['task_id'], tid)
                self.assertEqual(manifest['base_task_id'], base)
                self.assertEqual(json.loads((directory / 'initial_layout.json').read_text(encoding='utf-8')),
                                 layout(config.raw['task']))
                with self.assertRaisesRegex(ValueError, 'mismatch'):
                    resolve_task(base, path)
                path.write_text(path.read_text(encoding='utf-8').replace('"l2_p01"', '"unknown"'),
                                encoding='utf-8')
                with self.assertRaisesRegex(ValueError, 'variant_id'):
                    resolve_task(tid, path)


if __name__ == '__main__':
    unittest.main()
