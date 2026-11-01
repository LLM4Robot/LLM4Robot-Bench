"""Boundary cases for empty-source, half-received scoring; no GPT or physics."""
import unittest
import numpy as np

from robot_tasks.registry import resolve_task
from robot_tasks.robots.transforms import pose_transform, transform_pose
from robot_tasks.tasks.bench.arm_selection import resolve_bench_spec
from robot_tasks.tasks.bench.evaluation import BenchEvaluator
from test_household_tasks import measurement


class PouringAcceptanceTests(unittest.TestCase):
    def scene(self, task_id, received, remaining=0, moving=False):
        _, config = resolve_task(task_id)
        spec = resolve_bench_spec(config.raw['task'], config.robot)
        rows = measurement(spec)
        source, receiver = rows['source_cup'], rows['receiver_cup']
        src, dst = pose_transform(source['pose']), pose_transform(receiver['pose'])
        # Preserve the asset's validated packing offsets when moving pearls.
        local = [np.linalg.inv(src) @ pose_transform(rows[f'load_{i}']['pose'])
                 for i in range(spec.parameters['count'])]
        source['pose'][2] += .25
        source['pose'][3:] = [np.cos(np.pi/4), 0., np.sin(np.pi/4), 0.]
        source['fingers'][spec.parameters['manipulation_arm']] = [1., 1.]
        source['bench_force_n'] = 0.
        for i in range(spec.parameters['count']):
            row = rows[f'load_{i}']
            if i < received:
                row['pose'] = list(transform_pose(dst @ local[i]))
                if moving:
                    row['linear_velocity'] = [0., 0., .1]
            elif i < received + remaining:
                row['pose'] = list(transform_pose(pose_transform(source['pose']) @ local[i]))
            else:
                # Spilled pearls may be outside both cups, including off-table.
                row['pose'] = [1., 1. + i * .02, .1, 1., 0., 0., 0.]
        evaluator = BenchEvaluator(spec, {a.name: a.initial for a in spec.design.articles})
        return config, spec, rows, evaluator

    def test_six_and_seven_succeed_with_empty_source_still_held(self):
        for task_id in ('L2-03', 'L2-03_v3'):
            for received in (6, 7, 12):
                with self.subTest(task=task_id, received=received):
                    _, spec, rows, evaluator = self.scene(task_id, received)
                    self.assertEqual(spec.parameters['required_received'], 6)
                    result = evaluator.update(.1, rows)
                    self.assertEqual(result.status, 'succeeded')
                    c = result.metrics['criteria']
                    self.assertEqual(c['received'], received)
                    self.assertTrue(c['source_empty'])
                    self.assertFalse(c['source_placement_required'])
                    self.assertFalse(c['placed_upright']['source'])
                    self.assertEqual(result.metrics['scoring_protocol'], 'empty_source_half_received_v1')

    def test_five_received_residue_and_unsettled_balls_do_not_succeed(self):
        for task_id in ('L2-03', 'L2-03_v3'):
            for received, remaining, moving in ((5, 0, False), (6, 1, False), (6, 0, True)):
                with self.subTest(task=task_id, state=(received, remaining, moving)):
                    _, _, rows, evaluator = self.scene(task_id, received, remaining, moving)
                    result = evaluator.update(.1, rows)
                    self.assertEqual(result.status, 'running')
                    self.assertEqual(result.metrics['criteria']['remaining_source'], remaining)

    def test_initial_state_does_not_succeed_and_prompt_is_unchanged(self):
        for tid in ('L2-03', 'L2-03_v3'):
            entry, config = resolve_task(tid)
            spec = resolve_bench_spec(config.raw['task'], config.robot)
            e = BenchEvaluator(spec, {a.name: a.initial for a in spec.design.articles})
            result = e.update(.1, measurement(spec))
            self.assertEqual(result.status, 'running')
            self.assertEqual(result.metrics['criteria']['remaining_source'], 12)
            self.assertEqual(entry.factory.prepare_config(config).instruction, config.instruction)


if __name__ == '__main__':
    unittest.main()
