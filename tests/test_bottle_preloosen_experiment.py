"""Offline checks for an explicit initial-state experiment, never a model/GPU run."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from robot_tasks.initial_layout import scene_record
from robot_tasks.registry import resolve_task
from robot_tasks.robots.transforms import pose_transform
from robot_tasks.tasks.bench import bottle_contacts
from robot_tasks.video_delivery import scenes

ROOT=Path(__file__).resolve().parents[1]
loader=importlib.util.spec_from_file_location('bottle_experiment',ROOT/'tools/experiments/bottle_preloosen.py')
experiment=importlib.util.module_from_spec(loader);loader.loader.exec_module(experiment)
CONFIG=ROOT/'configs/experiments/l3_01_preloosen_270.json'


class BottleExperimentTests(unittest.TestCase):
    def test_270_changes_only_initial_cap_engagement_and_restores_default(self):
        original=deepcopy(scenes())
        value=experiment.load_experiment(CONFIG)
        spec,base=resolve_task('L3-01')
        default=scene_record(base.raw['task'])
        with experiment.engagement(270):
            _,config=experiment.prepare(value)
            row=scene_record(config.raw['task'])
            self.assertEqual(spec.factory.prepare_config(config).instruction,
                             spec.factory.prepare_config(base).instruction)
            self.assertEqual(row['tracked']['bottle'],default['tracked']['bottle'])
            self.assertEqual(row['fixtures'],default['fixtures'])
            relative=np.linalg.inv(pose_transform(row['tracked']['bottle']['pose'])) @ pose_transform(row['tracked']['cap']['pose'])
            np.testing.assert_allclose(relative[:3,3],(0,0,.00675),atol=1e-6)
            np.testing.assert_allclose(relative[:3,:3],[[0,1,0],[-1,0,0],[0,0,1]],atol=1e-6)
        self.assertEqual(scene_record(base.raw['task']),default)
        self.assertEqual(scenes(),original)
        self.assertEqual(bottle_contacts.REENGAGE_M,.003)

    def test_override_is_restored_after_failure(self):
        with self.assertRaisesRegex(RuntimeError,'test abort'):
            with experiment.engagement(270):raise RuntimeError('test abort')
        self.assertEqual(bottle_contacts.REENGAGE_M,.003)

    def test_check_is_offline_and_does_not_spawn_a_process(self):
        with patch.object(experiment.subprocess,'Popen',side_effect=AssertionError('no child allowed')):
            report=experiment.inspect(experiment.load_experiment(CONFIG))
        self.assertTrue(report['scene_and_scoring_initial_pose_match'])
        self.assertFalse(report['simulation_started'])
        self.assertEqual(report['model_requests'],0)
        self.assertEqual(report['max_calls'],150)
        self.assertEqual(report['max_wall_time_s'],18000)

    def test_budget_reaches_both_runner_and_model_without_generation(self):
        from robot_tasks.execution import _gpt_policy
        _,config=experiment.prepare(experiment.load_experiment(CONFIG))
        self.assertEqual(config.max_wall_time_s,18000)
        self.assertEqual(config.raw['execution']['max_decisions'],150)
        client=SimpleNamespace(require_chatgpt_login=lambda:None)
        with patch('robot_tasks.policies.codex_bridge.create_codex_client',return_value=client), \
                patch('robot_tasks.execution.GPTPolicy') as policy:
            _gpt_policy(config,None)
        self.assertEqual(policy.call_args.args[1].max_calls,150)

    def test_invalid_angles_and_task_fail_before_launch(self):
        value=experiment.load_experiment(CONFIG)
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'experiment.json'
            for edit in [{'preloosen_degrees':float('nan')},{'preloosen_degrees':301},
                         {'preloosen_degrees':True},{'base_task':'L3-02'},{'gpu_index':1}]:
                p.write_text(json.dumps({**value,**edit}))
                with self.subTest(edit=edit),self.assertRaises(ValueError):experiment.load_experiment(p)

    def test_run_keeps_the_existing_process_supervisor_for_safe_stop(self):
        for returncode,complete,expected in [(0,True,0),(130,True,130),(0,False,2)]:
            with tempfile.TemporaryDirectory() as temp, \
                    patch('sys.argv',['bottle_preloosen.py','run','--experiment-config',str(CONFIG),
                                      '--output',str(Path(temp)/'episode')]), \
                    patch.object(experiment,'require_idle_gpu'), \
                    patch.object(experiment.time,'monotonic',return_value=500), \
                    patch('robot_tasks.runtime_config.load_runtime',return_value=SimpleNamespace(python='/fake/isaac-python')), \
                    patch('robot_tasks.batch.run_job',return_value={'returncode':returncode,'cleanup_complete':complete}) as supervise:
                self.assertEqual(experiment.main(),expected)
                command,directory,env,timeout,callback=supervise.call_args.args
                self.assertEqual(command[0],'/fake/isaac-python')
                self.assertEqual(command[-1],'--worker')
                self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'0')
                self.assertEqual(timeout,18300)
                self.assertEqual(float(env['GPT_BENCHMARK_WALL_DEADLINE_MONOTONIC']),18500)
                saved=json.loads((Path(temp)/'episode/worker.json').read_text())
                self.assertEqual(saved['returncode'],returncode)


if __name__=='__main__':unittest.main()
