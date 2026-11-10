"""Branch launch/profile regressions. All policy/Isaac execution is mocked."""
from contextlib import redirect_stdout
from copy import deepcopy
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from robot_tasks.control_revalidation import profile
from robot_tasks.registry import resolve_task, TASKS
from robot_tasks.tasks.bench import bottle_contacts
from robot_tasks.recording import EpisodeRecorder

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/experiments/l3_01_preloosen_270.json'
loader = importlib.util.spec_from_file_location('control_run_integration', ROOT / 'tools/control_revalidation/run.py')
launcher = importlib.util.module_from_spec(loader); loader.loader.exec_module(launcher)


class ProfileIntegrationTests(unittest.TestCase):
    def test_all_task_profiles_preserve_prompts_and_change_only_gravity_option(self):
        for name in sorted(k for k in TASKS if k.startswith('L')):
            with self.subTest(task=name):
                spec, original = resolve_task(name)
                before = deepcopy(original.raw)
                configured = profile.configure(original)
                expected = deepcopy(before); expected['simulation']['robot_gravity_compensation'] = True
                self.assertEqual(configured.raw, expected)
                self.assertEqual(original.raw, before)
                report = profile.describe(spec, original)
                self.assertTrue(report['task_control_schema_unchanged'])
                self.assertEqual(report['max_calls'], 150)
                self.assertEqual(report['max_wall_time_s'], 18000)

    def test_cli_defaults_to_profile_and_dry_run_does_not_activate_it(self):
        from robot_tasks import cli
        self.assertIs(cli.execute_task, profile.execute_task)
        spec, config = resolve_task('L3-01')
        with patch.object(profile, 'activated', side_effect=AssertionError('no runtime allowed')), \
                patch.object(profile.native, 'execute_task', return_value=('out', {})) as execute:
            profile.execute_task(spec, config, Path('unused'), dry_run=True)
        self.assertIs(execute.call_args.args[1], config)

    def test_activation_reuses_runtime_and_restores_on_exception(self):
        old_policy = profile.native.POLICIES['gpt']
        old_runner = profile.native.run_episode
        with patch.object(profile.sys, 'platform', 'linux'), \
                patch.dict(os.environ, {'GPT_BENCHMARK_PLANNER_PYTHON': sys.executable}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'fixture'):
                with profile.activated():
                    self.assertEqual(os.environ['R5_PRECISION_PYTHON'], sys.executable)
                    self.assertTrue(os.environ['GPT_BENCHMARK_PLANNER_PYTHON'].endswith('planner_python.py'))
                    self.assertIsNot(profile.native.POLICIES['gpt'], old_policy)
                    self.assertIsNot(profile.native.run_episode, old_runner)
                    raise RuntimeError('fixture')
            self.assertEqual(os.environ['GPT_BENCHMARK_PLANNER_PYTHON'], sys.executable)
            self.assertNotIn('R5_PRECISION_PYTHON', os.environ)
            self.assertIs(profile.native.POLICIES['gpt'], old_policy)
            self.assertIs(profile.native.run_episode, old_runner)

    def test_shim_cannot_recursively_select_itself(self):
        with patch.dict(os.environ, {'R5_PRECISION_PYTHON': str(profile.WORKER_ROOT / 'planner_python.py')}):
            with self.assertRaisesRegex(ValueError, 'not the shim'):
                profile.planner_python()

    def test_adapter_uses_existing_client_and_budget_without_model_generation(self):
        from robot_tasks.policies.gpt import GPTPolicy
        spec, config = resolve_task('L3-01')
        config = spec.factory.prepare_config(profile.configure(config))
        client = SimpleNamespace(require_chatgpt_login=lambda: None)
        with patch.object(profile.sys, 'platform', 'linux'), \
                patch.dict(os.environ, {'GPT_BENCHMARK_PLANNER_PYTHON': sys.executable}, clear=True), \
                patch('robot_tasks.policies.codex_bridge.create_codex_client', return_value=client):
            with profile.activated():
                policy = profile.native.POLICIES['gpt'](config, None)
                self.assertIs(policy.client, client)
                self.assertEqual(policy.settings.max_calls, 150)
                self.assertTrue(hasattr(policy, '_previous_action'))
                self.assertIsInstance(policy, GPTPolicy)
                self.assertEqual(policy.instructions, GPTPolicy(config, policy.settings, client=client).instructions)

    def test_general_worker_builds_valid_gpu_settings(self):
        from robot_tasks.backends.settings import SimulationSettings
        def execute(spec, config, output, **kwargs):
            settings = SimulationSettings.from_config(config)
            self.assertEqual(settings.active_gpu, 1)
            self.assertEqual(settings.physics_gpu, 0)
            return output, {'status': 'finished'}
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '1'}), \
                patch.object(launcher, 'execute_task', side_effect=execute), redirect_stdout(io.StringIO()):
            self.assertEqual(launcher.main(['run', '--task', 'L2-01', '--gpu', '1',
                                            '--worker', '--output', 'unused']), 0)

    def test_270_worker_reaches_profile_preserving_initial_state_and_recording(self):
        bottle = launcher.bottle_launcher()
        seen = {}
        def fake_execute(spec, config, output, **kwargs):
            # Exercise the real native recorder's freshness check and config
            # snapshots; do not construct a simulator or call a model.
            self.assertAlmostEqual(bottle_contacts.REENGAGE_M, .00075)
            config = spec.factory.prepare_config(config)
            recorder = EpisodeRecorder(output, config, 'simulation', config.max_steps,
                                       run_directory=kwargs['run_directory'])
            seen.update(config=config, directory=recorder.directory)
            return recorder.directory, {'status': 'finished', 'success': False}
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()), \
                patch.object(profile.sys, 'platform', 'linux'), \
                patch.dict(os.environ, {'GPT_BENCHMARK_PLANNER_PYTHON': sys.executable,
                                       'GPT_BENCHMARK_CODEX_BRIDGE_URL': 'http://127.0.0.1:1',
                                       'GPT_BENCHMARK_WALL_DEADLINE_MONOTONIC': '18500'}, clear=True), \
                patch.object(profile.native, 'execute_task', side_effect=fake_execute):
            self.assertEqual(bottle.main(['run', '--experiment-config', str(CONFIG),
                                          '--output', temp, '--worker']), 0)
            config = seen['config']
            self.assertTrue(config.raw['simulation']['robot_gravity_compensation'])
            self.assertEqual(config.raw['task']['experimental_initial_state']['preloosen_degrees'], 270)
            self.assertEqual(config.raw['execution']['max_decisions'], 150)
            self.assertEqual(config.max_wall_time_s, 18000)
            saved = json.loads((seen['directory'] / 'effective_config.json').read_text())['control_revalidation']
            sidecar = next((Path(temp) / 'control-profiles').glob('*.json'))
            self.assertEqual(saved, json.loads(sidecar.read_text()))
            self.assertEqual(saved['shared_wall_deadline_monotonic'], '18500')
            self.assertEqual(saved['experimental_initial_state']['preloosen_degrees'], 270)
        self.assertEqual(bottle_contacts.REENGAGE_M, .003)

    def test_general_run_uses_supervisor_and_shared_budget(self):
        for code, cleanup, expected in ((0, True, 0), (130, True, 130), (0, False, 2)):
            with self.subTest(code=code, cleanup=cleanup), tempfile.TemporaryDirectory() as temp, \
                    patch.object(launcher, 'bottle_launcher', return_value=SimpleNamespace(require_idle_gpu=lambda _: None)), \
                    patch('robot_tasks.runtime_config.load_runtime', return_value=SimpleNamespace(python='/fake/isaac-python')), \
                    patch.object(launcher.time, 'monotonic', return_value=500), \
                    patch('robot_tasks.batch.run_job', return_value={'returncode':code, 'cleanup_complete':cleanup}) as supervise:
                output = Path(temp) / 'new'
                self.assertEqual(launcher.main(['run', '--task', 'L2-01', '--gpu', '0', '--output', str(output)]), expected)
                command, _, env, timeout, _ = supervise.call_args.args
                self.assertEqual(command[0], '/fake/isaac-python')
                self.assertEqual(command[-1], '--worker')
                self.assertEqual(float(env['GPT_BENCHMARK_WALL_DEADLINE_MONOTONIC']), 18500)
                self.assertEqual(timeout, 18300)

    def test_explicit_270_check_uses_existing_configuration_with_no_process(self):
        with patch('subprocess.Popen', side_effect=AssertionError('no child allowed')), redirect_stdout(io.StringIO()) as text:
            self.assertEqual(launcher.main(['check', '--experiment-config', str(CONFIG)]), 0)
        value = json.loads(text.getvalue())
        self.assertEqual(value['preloosen_degrees'], 270)
        self.assertAlmostEqual(value['initial_cap_lift_m'], .00675, delta=1e-6)
        self.assertEqual(value['control_profile']['max_calls'], 150)
        self.assertEqual(value['control_profile']['max_wall_time_s'], 18000)
        self.assertFalse(value['simulation_started'])
        self.assertEqual(value['model_requests'], 0)

    def test_wrong_task_for_270_rejected_before_any_worker(self):
        with patch('subprocess.Popen', side_effect=AssertionError('no child allowed')), self.assertRaises(SystemExit):
            launcher.main(['check', '--task', 'L2-01', '--experiment-config', str(CONFIG)])


if __name__ == '__main__':
    unittest.main()
