"""Offline force-law and time-only budget regressions; no model/network calls."""
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

from fixtures import RGBDBackend, decision_document, robot_config
from robot_tasks import batch
from robot_tasks.backends.mock import MockRobot, MockTask
from robot_tasks.config import ExecutionSettings, load_config
from robot_tasks.policies.gpt import GPTPolicy
from robot_tasks.policies.model_config import GPTSettings, load_gpt_settings
from robot_tasks.policies.responses import ModelReply, RetryableModelError
from robot_tasks.runner import run_episode
from robot_tasks.runtime_config import RuntimeConfig
from robot_tasks.tasks.bench.resistant_wiping import ResistantWipeLedger
from robot_tasks.types import Assessment, Feedback, PolicyInput, Observation


class ForceWearTests(unittest.TestCase):
    def sweep(self, force, passes=1):
        l=ResistantWipeLedger((0,0,0,1,0,0,0),[[0,0,.018]],facing_direction=1)
        t=0
        for n in range(passes):
            for x in np.linspace(-.07,.07,281)[::(-1 if n%2 else 1)]:
                t+=.01
                l.update(t,(x,0,0,1,0,0,0),pad_contact_n=force,held=True)
        return l

    def test_light_pressure_strong_pressure_and_saturation(self):
        forces=[0,.05,.1,.2,.5,1,2,3,5,10,1000]
        wear=[1-self.sweep(f).residue[0] for f in forces]
        self.assertEqual(wear[:3],[0,0,0])
        self.assertLess(wear[3],.002)
        self.assertTrue(all(a<b for a,b in zip(wear[2:],wear[3:])))
        self.assertLess(wear[-1],.61)
        self.assertGreater(self.sweep(.2,3).residue[0],.99)
        self.assertFalse(self.sweep(3,3).dirty[0])
        self.assertFalse(self.sweep(10,2).dirty[0])

    def test_force_spike_cannot_erase_prior_light_segment(self):
        l=self.sweep(0)
        l.update(4,(0,0,0,1,0,0,0),held=True,pad_contact_n=.2)
        l.update(5,(.01,0,0,1,0,0,0),held=True,pad_contact_n=100)
        self.assertLess(l.coverage[0],.001)

    def test_force_parameters_and_nonfinite_input(self):
        for k,v in [('force_deadband_n',-1),('force_half_gain_n',0),('force_max_gain',float('nan'))]:
            with self.assertRaises(ValueError):
                ResistantWipeLedger((0,0,0,1,0,0,0),**{k:v})
        for f in [-2,float('nan'),float('inf')]:
            self.assertEqual(self.sweep(f).coverage[0],0)


class UnlimitedBudgetTests(unittest.TestCase):
    def run_budget(self, *, fail=False, external=False, legacy_retry=False,
                   call_cap=0, seconds_per_call=149.):
        cfg=robot_config()
        cfg=replace(cfg,task_id='L2-07',max_steps=0,max_wall_time_s=18000,
            raw={**cfg.raw,'execution':{'max_decisions':call_cap,'max_action_steps':1,'completion_samples':1}})
        if legacy_retry:
            cfg.raw['execution']['retry_model_errors']=False
        self.assertEqual(ExecutionSettings.from_config(cfg).max_decisions,call_cap)
        clock=[0.]
        settings=GPTSettings('https://example.invalid','fixture',max_calls=call_cap,history_steps=0)
        client=Mock()
        def generate(*a,**kw):
            clock[0]+=seconds_per_call
            if fail:raise RetryableModelError('offline retry')
            return ModelReply(json.dumps(decision_document(commands=[])), 'fixture','offline',{},0.)
        client.generate.side_effect=generate
        policy=GPTPolicy(cfg,settings,client=client)
        robot=MockRobot();robot.apply=Mock();robot.feedback=Mock(return_value=Feedback('completed'))
        task=MockTask();task.evaluate=Mock(return_value=Assessment('running','fixture'))
        env={'GPT_BENCHMARK_PERSISTENT_MODEL_RETRY':'1'}
        if external:env['GPT_BENCHMARK_WALL_DEADLINE_MONOTONIC']='1000'
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()), \
             patch.dict('os.environ',env), patch('robot_tasks.runner.time.monotonic',side_effect=lambda:clock[0]), \
             patch('robot_tasks.runner.time.sleep'):
            _,result=run_episode(cfg,RGBDBackend(),robot,task,policy,Path(d))
        call_limit_first = bool(call_cap and call_cap * seconds_per_call < (1000 if external else 18000))
        self.assertEqual(result['termination_reason'],'model_call_limit' if call_limit_first else 'wall_time_limit')
        self.assertEqual(client.generate.call_count,call_cap if call_limit_first else (7 if external else 121))
        self.assertIsNone(result['error'])
        if fail:self.assertEqual(robot.apply.call_count,0)
        else:self.assertEqual(robot.apply.call_count,call_cap if call_limit_first else (6 if external else 120))

    def test_150_call_budget_stops_before_the_next_request(self):
        self.run_budget(call_cap=150, seconds_per_call=1.)

    def test_five_hours_stops_before_150_calls_and_discards_late_response(self):
        self.run_budget(call_cap=150)

    def test_no_hundred_request_cap_and_late_response_discarded(self):self.run_budget()
    def test_retries_are_unlimited_until_deadline(self):self.run_budget(fail=True)
    def test_startup_included_in_external_deadline(self):self.run_budget(external=True)

    def test_reclassified_wiping_retains_legacy_persistent_retry(self):
        self.run_budget(fail=True,legacy_retry=True)

    def test_policy_has_no_thousand_request_cap(self):
        cfg=robot_config();client=Mock()
        client.generate.return_value=ModelReply(json.dumps(decision_document(commands=[])),'fixture','offline',{},0.)
        p=GPTPolicy(cfg,GPTSettings('https://example.invalid','fixture',max_calls=0),client=client)
        p.calls=1001
        backend=RGBDBackend();backend.open(cfg)
        try:
            p.act(PolicyInput(cfg.instruction,Observation(0.,backend.capture_rgb(),{},Feedback('completed')),1002))
        finally:backend.close()
        self.assertEqual(p.calls,1002)

    def test_batch_preserves_zero_in_both_configs(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); model=root/'model.toml'
            model.write_text('[gpt]\nbase_url="https://example.invalid"\nmodel="fixture"\n')
            runtime=RuntimeConfig('/unused',root,root,root/'runtime.toml',True)
            args=Mock(tasks=['L2-07'],max_calls=0,wall_time_s=18000,startup_grace_s=120,
                      seed=0,reasoning_effort='xhigh',dry_run=False,gui=False,depth=None)
            manifest,_=batch.create_batch(root/'batch',args,model,runtime)
            self.assertEqual(manifest['options']['max_calls'],0)
            cfg=load_config(root/'batch/configs/L2-07.toml')
            self.assertEqual(cfg.max_wall_time_s,18000)
            self.assertEqual(ExecutionSettings.from_config(cfg).max_decisions,0)
            self.assertEqual(load_gpt_settings(root/'batch/configs/model.toml').max_calls,0)

    def test_time_only_batch_installs_deadline_and_persistent_retry(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            manifest={'python':'unused','options':{'max_calls':0,'wall_time_s':18000.,
                'startup_grace_s':60.,'dry_run':False,'gui':False},
                'tasks':[{'id':'L2-07','config':'unused','cameras':[]}]}
            environment={'KEPT':'yes'}
            captured={}
            def job(command,directory,env,timeout,started):
                captured.update(env);started({'worker_pid':1})
                return {'interrupted':False,'cleanup_complete':True}
            counts={k:0 for k in ['infrastructure_error','timeout','interrupted','pending','video_incomplete','video_unverified']}
            with patch.object(batch,'run_job',side_effect=job), \
                 patch.object(batch,'collect_result',return_value={'outcome':'failed','model_calls':121,'video_complete':True}), \
                 patch.object(batch,'summarize',return_value=counts), \
                 patch.object(batch.time,'monotonic',return_value=123.), redirect_stdout(io.StringIO()):
                self.assertEqual(batch.execute_batch(root,manifest,{},environment),0)
            self.assertEqual(captured['GPT_BENCHMARK_WALL_DEADLINE_MONOTONIC'],'18123.0')
            self.assertEqual(captured['GPT_BENCHMARK_PERSISTENT_MODEL_RETRY'],'1')
            self.assertEqual(environment,{'KEPT':'yes'})

if __name__=='__main__':unittest.main()
