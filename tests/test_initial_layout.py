"""Initial-layout isolation and mechanical-frame consistency, without GPT."""

from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np

from robot_tasks.initial_layout import apply_spec, apply_stage, scene_record
from robot_tasks.paths import PROJECT_ROOT
from robot_tasks.registry import resolve_task
from robot_tasks.robots.transforms import pose_transform
from robot_tasks.tasks.bench.arm_selection import resolve_bench_spec
from robot_tasks.video_delivery import scenes


class InitialLayoutTests(unittest.TestCase):
    def configs(self):
        for path in sorted((PROJECT_ROOT / "configs/tasks").glob("l2*.v3.toml")):
            entry, after = resolve_task(config_path=path)
            if after.raw['task'].get('initial_layout') != 'l2_p01':
                continue
            _, before = resolve_task(after.task_id)
            yield before, after, entry

    def test_opt_in_keeps_defaults_prompt_arms_and_all_nonplacement_settings(self):
        manifest = deepcopy(scenes())
        self.assertEqual(len(list(self.configs())), 6)
        for before, after, entry in self.configs():
            with self.subTest(task=before.task_id):
                raw = deepcopy(after.raw)
                raw['task'].pop('initial_layout')
                raw['task'].pop('variant_id')
                original = deepcopy(before.raw)
                for record in (raw, original):
                    record['task'].pop('launch_command', None)
                    record['task'].pop('launch_command_conda', None)
                self.assertEqual(raw, original)
                self.assertEqual(entry.factory.prepare_config(after).instruction,
                                 entry.factory.prepare_config(before).instruction)
                spec = resolve_bench_spec(before.raw['task'], before.robot)
                self.assertIs(apply_spec(spec, before.raw['task']), spec)
                base_record = scene_record(before.raw['task'])
                source = scenes()[before.task_id]
                if before.task_id in ('L2-01', 'L2-06'):
                    # The original and variant both inherit the confirmed
                    # asset calibration; placement opt-in must add no other
                    # changes and must never mutate the historical delivery.
                    restored = deepcopy(base_record)
                    if before.task_id == 'L2-01':
                        for name in ('body', 'cap'):
                            restored['tracked'][name]['bounds'] = source['tracked'][name]['bounds']
                    else:
                        restored['jelly']['mass'] = source['jelly']['mass']
                    self.assertEqual(restored, source)
                else:
                    self.assertIs(base_record, source)
                changed = resolve_bench_spec(after.raw['task'], after.robot)
                self.assertEqual(spec.arms, changed.arms)
                for original_article, variant_article in zip(spec.design.articles, changed.design.articles):
                    self.assertEqual(original_article.asset_bounds, variant_article.asset_bounds)
                    self.assertEqual(original_article.mass, variant_article.mass)
                for key, value in spec.parameters.items():
                    if key not in ('outward_axis_world', 'closed_pose'):
                        self.assertEqual(value, changed.parameters[key], key)
        self.assertEqual(manifest, scenes())

    def test_loaded_cup_preserves_all_twelve_ball_relative_positions(self):
        task={'id':'L2-03','initial_layout':'l2_p01'}
        old=scenes()['L2-03']['tracked'];new=scene_record(task)['tracked']
        balls=[n for n in old if n.startswith('load_')]
        self.assertEqual(len(balls),12)
        for name in balls:
            np.testing.assert_allclose(np.linalg.inv(pose_transform(new['source_cup']['pose'])) @ pose_transform(new[name]['pose']),
                np.linalg.inv(pose_transform(old['source_cup']['pose'])) @ pose_transform(old[name]['pose']), atol=1e-12)
        self.assertAlmostEqual(new['source_cup']['pose'][0],new['receiver_cup']['pose'][0])

    def test_reset_records_and_scoring_initial_poses_match_stage_without_asset_edits(self):
        try:
            from pxr import Usd, UsdGeom
        except ImportError:
            self.skipTest('Requires usd-core')
        for _, config, _ in self.configs():
            with self.subTest(task=config.task_id):
                row=scene_record(config.raw['task'])
                stage=Usd.Stage.Open(str(PROJECT_ROOT/'assets/tuojing'/row['runtime_path']))
                original=stage.GetRootLayer().ExportToString()
                # All authored properties outside transforms must stay byte-identical.
                attrs={str(p.GetPath()): {a.GetName():str(a.Get()) for a in p.GetAttributes()
                       if not a.GetName().startswith('xformOp')} for p in stage.Traverse()}
                stage.SetEditTarget(stage.GetSessionLayer())
                apply_stage(stage,config.raw['task'])
                cache=UsdGeom.XformCache()
                spec=resolve_bench_spec(config.raw['task'],config.robot)
                records={**row['tracked'],**{k:row[k] for k in ('orange','jelly') if k in row}}
                for name,record in records.items():
                    world=np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(record['path']))).T
                    np.testing.assert_allclose(world[:3,3],record['pose'][:3],atol=1e-8)
                    article=next(a for a in spec.design.articles if a.name==name)
                    np.testing.assert_allclose(article.initial,record['pose'],atol=1e-8)
                self.assertEqual(original,stage.GetRootLayer().ExportToString())
                for p in stage.Traverse():
                    self.assertEqual(attrs[str(p.GetPath())],{a.GetName():str(a.Get()) for a in p.GetAttributes()
                        if not a.GetName().startswith('xformOp')})

    def test_drawer_closed_frame_and_books_extraction_axis_rotate_with_props(self):
        for before,after,_ in self.configs():
            if before.task_id not in ('L2-02','L2-05'):
                continue
            old=resolve_bench_spec(before.raw['task'],before.robot)
            new=resolve_bench_spec(after.raw['task'],after.robot)
            name='drawer' if before.task_id=='L2-05' else 'middle_book'
            a=pose_transform(next(a.initial for a in old.design.articles if a.name==name))
            b=pose_transform(next(a.initial for a in new.design.articles if a.name==name))
            np.testing.assert_allclose(a[:3,:3].T @ old.parameters['outward_axis_world'],
                                      b[:3,:3].T @ new.parameters['outward_axis_world'],atol=1e-10)
            if name=='drawer':
                np.testing.assert_allclose(np.linalg.inv(a) @ pose_transform(old.parameters['closed_pose']),
                    np.linalg.inv(b) @ pose_transform(new.parameters['closed_pose']),atol=1e-10)


if __name__=='__main__':
    unittest.main()
