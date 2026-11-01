"""Placement variants must preserve contacts, geometry and the baseline task."""
from copy import deepcopy
from itertools import product
import json
import unittest

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics

from robot_tasks.paths import PROJECT_ROOT
from robot_tasks.registry import resolve_task
from robot_tasks.robots.transforms import pose_transform
from robot_tasks.tasks.bench.arm_selection import resolve_bench_spec
from robot_tasks.tasks.bench.initial_placement import scene_record, apply_to_stage, PROFILES, yaw_about_pivot
from robot_tasks.video_delivery import scenes

TASKS = ('L1-01', 'L1-02', 'L1-03', 'L1-04', 'L1-05')


class InitialPlacementTests(unittest.TestCase):
    def test_variants_preserve_baselines_body_types_heights_and_declared_orientations(self):
        before = json.dumps(scenes(), sort_keys=True)
        for tid in TASKS:
            with self.subTest(task=tid):
                _, config = resolve_task(tid + "_v3")
                row = scene_record(config.raw['task'])
                self.assertIsNot(row, scenes()[tid])
                self.assertEqual(row, scene_record(config.raw['task']))
                plan = row['initial_placement_perturbation']
                rotated = {r['path'] for r in plan['rotations']}
                for item in row['initial_placement_perturbation']['changed_records']:
                    delta = np.array(item['after_pose']) - item['before_pose']
                    self.assertLessEqual(np.linalg.norm(delta[:2]), .09)
                    self.assertAlmostEqual(delta[2], 0., places=12)
                    if item['path'] not in rotated:
                        np.testing.assert_array_equal(delta[3:], np.zeros(4))
                for name, record in row['tracked'].items():
                    for key in ('kinematic', 'mass', 'bounds', 'inherited_scale'):
                        self.assertEqual(record[key], scenes()[tid]['tracked'][name][key])
                baseline_task = {**config.raw['task'], 'initial_placement_profile': 'baseline'}
                self.assertIs(scene_record(baseline_task), scenes()[tid])
                p01 = scene_record({**config.raw['task'], 'initial_placement_profile': 'l1_xy_small_p01_20261018'})
                self.assertFalse(p01['initial_placement_perturbation']['rotations'])
                for item in p01['initial_placement_perturbation']['changed_records']:
                    distance = np.linalg.norm(np.array(item['after_pose'][:2]) - item['before_pose'][:2])
                    self.assertGreater(distance, .01)
                    self.assertLess(distance, .015)
        self.assertEqual(json.dumps(scenes(), sort_keys=True), before)

    def test_stage_contacts_follow_groups_and_reset_leaf_bodies_without_accumulation(self):
        for tid, variant in product(TASKS, range(1, 6)):
            with self.subTest(task=tid, variant=variant):
                _, config = resolve_task(tid + "_v3")
                row = scene_record({**config.raw['task'], 'initial_placement_variant': variant})
                stage = Usd.Stage.Open(str(PROJECT_ROOT / 'assets/tuojing' / row['runtime_path']))
                root_text = stage.GetRootLayer().ExportToString()
                def world(path):
                    return np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path))).T
                groups = row['initial_placement_perturbation']['groups']
                rotations = row['initial_placement_perturbation']['rotations']
                before = {path: world(path).copy() for g in groups for path in g['paths']}
                leaf_resets = {path: UsdGeom.Xformable(stage.GetPrimAtPath(path)).GetResetXformStack()
                               for path in before}
                body_types = {str(p.GetPath()): UsdPhysics.RigidBodyAPI(p).GetKinematicEnabledAttr().Get()
                              for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
                expected = {path: transform.copy() for path, transform in before.items()}
                for group in groups:
                    for path in group['paths']:
                        expected[path][:3, 3] += group['delta_m']
                for rotation in rotations:
                    expected[rotation['path']] = yaw_about_pivot(rotation['yaw_deg'],
                        expected[rotation['pivot_path']][:3, 3]) @ expected[rotation['path']]
                apply_to_stage(stage, row)
                for group in groups:
                    for path in group['paths']:
                        np.testing.assert_allclose(world(path), expected[path], atol=1e-12)
                        self.assertEqual(UsdGeom.Xformable(stage.GetPrimAtPath(path)).GetResetXformStack(), leaf_resets[path])
                    origin = group['paths'][0]
                    for path in group['paths'][1:]:
                        if not any(r['path'] in (origin, path) for r in rotations):
                            np.testing.assert_allclose(np.linalg.inv(world(origin)) @ world(path),
                                                       np.linalg.inv(before[origin]) @ before[path], atol=1e-12)
                for record in [*row['tracked'].values(), *row['fixtures']]:
                    np.testing.assert_allclose(world(record['path']), record['native_world_matrix'], atol=1e-12)
                for path, value in body_types.items():
                    self.assertEqual(UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(path)).GetKinematicEnabledAttr().Get(), value)
                after = {path: world(path).copy() for path in before}
                apply_to_stage(stage, row)
                for path in before:
                    np.testing.assert_array_equal(world(path), after[path])
                self.assertEqual(stage.GetRootLayer().ExportToString(), root_text)

    def test_p02_matches_reviewed_five_centimetre_plan_and_hat_yaw(self):
        expected = {'L1-01': [(-.04,-.03,0.),(0.,.03,0.)],
                    'L1-02': [(-.04,-.03,0.),(0.,.03,0.)],
                    'L1-03': [(0.,-.05,0.),(0.,0.,0.)],
                    'L1-04': [(-.04,-.02,0.),(.04,-.03,0.)],
                    'L1-05': [(-.04,-.03,0.),(0.,.03,0.)]}
        for tid, deltas in expected.items():
            _, config = resolve_task(tid + "_v3")
            row = scene_record({**config.raw['task'], 'initial_placement_profile': 'l1_xy_5cm_hat15_p02_20261018'})
            plan = row['initial_placement_perturbation']
            self.assertEqual([tuple(g['delta_m']) for g in plan['groups']], deltas)
            if tid == 'L1-03':
                self.assertEqual(plan['rotations'][0]['yaw_deg'], 15.)
            else:
                self.assertEqual(plan['rotations'], [])

    def test_active_task_poses_and_fixed_table_centre_are_consistent(self):
        for tid in TASKS:
            _, config = resolve_task(tid + "_v3")
            task = config.raw['task']
            spec = resolve_bench_spec(task, config.robot)
            row = scene_record(task)
            for article in spec.design.articles:
                np.testing.assert_array_equal(article.initial, row['tracked'][article.name]['pose'])
            baseline_spec = resolve_bench_spec({**task, 'initial_placement_profile': 'baseline'}, config.robot)
            self.assertEqual(spec.design.instruction, baseline_spec.design.instruction)
            self.assertEqual(spec.arms, baseline_spec.arms)
            self.assertEqual(spec.observers, baseline_spec.observers)
            if row['fixtures']:
                np.testing.assert_array_equal(spec.design.fixture_pose, row['fixtures'][0]['pose'])
            if tid == 'L1-04':
                np.testing.assert_array_equal(spec.parameters['centre_xy'], baseline_spec.parameters['centre_xy'])
            if tid == 'L1-05':
                self.assertEqual(spec.parameters['fixed_articles'], ('source_cup', 'target_cup'))
                np.testing.assert_array_equal(spec.parameters['vase_xy'], row['tracked']['target_cup']['pose'][:2])

    def test_unknown_profile_and_changed_baseline_are_rejected(self):
        _, config = resolve_task('L1-05_v3')
        with self.assertRaisesRegex(ValueError, 'Unknown initial placement'):
            scene_record({**config.raw['task'], 'initial_placement_profile': 'unknown'})
        row = scenes()['L1-05']
        previous = row['runtime_sha256']
        try:
            row['runtime_sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'reviewed baseline'):
                scene_record(config.raw['task'])
        finally:
            row['runtime_sha256'] = previous

    def test_p03_all_twenty_five_layouts_are_bounded_reproducible_and_independent(self):
        from tools.generate_l1_position_variants import geometry_screen
        profile = json.loads(PROFILES['l1_xy_random_p03_20261018'].read_text(encoding='utf-8'))
        seeds = set()
        for tid in TASKS:
            _, config = resolve_task(tid + "_v3")
            self.assertEqual(config.raw['task']['initial_placement_profile'], profile['id'])
            self.assertIn(config.raw['task']['initial_placement_variant'], range(1, 6))
            task = profile['tasks'][tid]
            self.assertEqual(len(task['variants']), 5)
            for number, variant in enumerate(task['variants'], 1):
                with self.subTest(task=tid, variant=number):
                    self.assertEqual(variant['variant'], number)
                    self.assertNotIn(variant['seed'], seeds)
                    seeds.add(variant['seed'])
                    raw = {**config.raw['task'], 'initial_placement_variant': number}
                    row = scene_record(raw)
                    self.assertIsNone(geometry_screen(tid, config, scenes()[tid], variant))
                    self.assertEqual(row, scene_record(raw))
                    self.assertEqual(row['initial_placement_perturbation']['seed'], variant['seed'])
                    for group, (low, high) in zip(variant['groups'], task['translation_ranges_m']):
                        radius = np.linalg.norm(group['delta_m'])
                        self.assertGreaterEqual(radius, low - 1e-8)
                        self.assertLessEqual(radius, high + 1e-8)
                        self.assertEqual(group['delta_m'][2], 0.)
                    shifts = [np.array(g['delta_m']) for g in variant['groups']]
                    self.assertGreaterEqual(np.linalg.norm(shifts[0] - shifts[1]), .015)
                    self.assertTrue(variant['reachability_screen']['passed'])
                    for check in variant['reachability_screen']['checks']:
                        self.assertLessEqual(check['position_error_m'], .003)
                    if tid == 'L1-03':
                        self.assertGreaterEqual(abs(variant['rotations'][0]['yaw_deg']), 10.)
                        self.assertLessEqual(abs(variant['rotations'][0]['yaw_deg']), 25.)
                    else:
                        self.assertFalse(row['initial_placement_perturbation']['rotations'])
                    spec = resolve_bench_spec(raw, config.robot)
                    for article in spec.design.articles:
                        np.testing.assert_array_equal(article.initial, row['tracked'][article.name]['pose'])
                    if tid == 'L1-04':
                        cups = row['tracked']
                        right, left = cups['lower_cup']['pose'], cups['upper_cup']['pose']
                        self.assertGreater(right[0] - left[0], .09)
                        self.assertGreaterEqual(abs(right[1] - left[1]), .01)
                        self.assertLessEqual(abs(right[1] - left[1]), .05)
                    if tid == 'L1-05':
                        self.assertEqual(spec.parameters['fixed_articles'], ('source_cup','target_cup'))
                        self.assertGreater(row['tracked']['source_cup']['pose'][0], row['tracked']['target_cup']['pose'][0])
        self.assertEqual(len(seeds), 25)

    def test_p03_selection_validation_and_stage_identity_include_variant(self):
        _, config = resolve_task('L1-05_v3')
        for invalid in (0, 6, True, 1.5, '2', None):
            with self.subTest(variant=invalid), self.assertRaises(ValueError):
                scene_record({**config.raw['task'], 'initial_placement_variant': invalid})
        row = scene_record(config.raw['task'])
        stage = Usd.Stage.Open(str(PROJECT_ROOT / 'assets/tuojing' / row['runtime_path']))
        apply_to_stage(stage, row)
        other = scene_record({**config.raw['task'], 'initial_placement_variant': 2})
        with self.assertRaisesRegex(ValueError, 'already been applied'):
            apply_to_stage(stage, other)

    def test_p03_generator_recreates_saved_seeded_samples(self):
        from tools.generate_l1_position_variants import generate
        from contextlib import redirect_stdout
        from io import StringIO
        saved = json.loads(PROFILES['l1_xy_random_p03_20261018'].read_text(encoding='utf-8'))
        with redirect_stdout(StringIO()):
            regenerated = generate(saved['master_seed'])
        for tid in TASKS:
            for old, new in zip(saved['tasks'][tid]['variants'], regenerated['tasks'][tid]['variants']):
                for key in ('variant','seed','sampling_attempt','groups','rotations','rejections_before_acceptance'):
                    self.assertEqual(old.get(key), new.get(key))


if __name__ == '__main__':
    unittest.main()
