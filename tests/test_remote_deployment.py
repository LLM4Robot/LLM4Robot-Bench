"""Optional SSH helpers must honor caller paths without contacting a host/model."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / 'skills/r5-remote-deploy'
spec = importlib.util.spec_from_file_location('remote_fleet_test', SKILL / 'scripts/fleet.py')
fleet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fleet)


class RemoteDeploymentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = {'host': 'example.test', 'user': 'reader',
                       'project': '/srv/my project', 'runtime': '/srv/my runtime',
                       'machines': {'worker': 22}, 'bridge_port': 28901,
                       'isaac_root': '/opt/isaac sim', 'planner_root': '/opt/planner'}
        self.enterContext(patch.object(fleet, 'CFG', self.config))
        self.enterContext(patch.object(fleet, 'CONFIG', self.root / 'remote.local.json'))

    def test_configured_port_and_runtime_are_forwarded_to_health_check(self):
        token = self.root / 'token'
        token.write_text('synthetic-test-token')
        with patch.object(fleet, 'TOKEN', token), patch.object(fleet, 'remote') as remote, \
                patch.object(fleet.subprocess, 'run') as run:
            fleet.connect('worker')
        health = remote.call_args.args[1]
        self.assertIn('GPT_BENCHMARK_RUNTIME_ROOT=/srv/my runtime', health)
        self.assertIn('GPT_BENCHMARK_CODEX_BRIDGE_URL=http://127.0.0.1:28901', health)
        self.assertIn('GPT_BENCHMARK_CODEX_BRIDGE_TOKEN_FILE=/srv/my runtime/bridge.token', health)
        self.assertIn('127.0.0.1:28901:127.0.0.1:28901', run.call_args.args[0])

    def test_remote_model_access_does_not_start_local_bridge(self):
        self.config['use_local_codex'] = False
        with patch('sys.argv', ['fleet.py', 'connect', '--machine', 'worker']), \
                patch.object(fleet, 'remote') as remote, patch.object(fleet, 'ensure_bridge') as bridge:
            fleet.main()
        bridge.assert_not_called()
        remote.assert_called_once_with('worker', ['true'])

    def test_busy_project_prevents_transfer(self):
        with patch('sys.argv', ['fleet.py', 'sync', '--machine', 'worker']), \
                patch.object(fleet, 'require_idle', side_effect=RuntimeError('busy')), \
                patch.object(fleet.subprocess, 'run') as transfer:
            with self.assertRaisesRegex(RuntimeError, 'busy'):
                fleet.main()
        transfer.assert_not_called()

    def test_explicit_experiment_path_is_forwarded_without_starting_bridge_in_dry_run(self):
        with patch('sys.argv', ['fleet.py','start','--machine','worker','--tasks','L3-01',
                                '--experiment-config','configs/experiments/l3_01_preloosen_270.json','--dry-run']), \
                patch.object(fleet,'remote') as remote,patch.object(fleet,'ensure_bridge') as bridge:
            fleet.main()
        bridge.assert_not_called()
        command=remote.call_args.args[1]
        self.assertIn('--experiment-config',command)
        self.assertEqual(command[-1],'configs/experiments/l3_01_preloosen_270.json')

    def test_bootstrap_gets_the_same_runtime_and_installed_interpreters(self):
        with patch('sys.argv', ['fleet.py', 'bootstrap', '--machine', 'worker']), \
                patch.object(fleet, 'require_idle'), patch.object(fleet, 'remote') as remote:
            fleet.main()
        command = remote.call_args.args[1]
        self.assertIn('GPT_BENCHMARK_RUNTIME_ROOT=/srv/my runtime', command)
        self.assertIn('GPT_BENCHMARK_ISAAC_ROOT=/opt/isaac sim', command)
        self.assertIn('GPT_BENCHMARK_PLANNER_ROOT=/opt/planner', command)

    @unittest.skipIf(os.name == 'nt', 'Remote wrappers execute on Linux')
    def test_real_wrapper_uses_custom_runtime_and_can_disable_forwarding(self):
        project = self.root / 'project with spaces'
        script = project / 'skills/r5-remote-deploy/scripts/remote_entry.sh'
        script.parent.mkdir(parents=True)
        shutil.copyfile(SKILL / 'scripts/remote_entry.sh', script)
        runtime = self.root / 'runtime with spaces'
        python = runtime / 'framework/bin/python'
        python.parent.mkdir(parents=True)
        python.symlink_to(sys.executable)
        base = {k: v for k, v in os.environ.items() if not k.startswith('GPT_BENCHMARK_')}
        for bridge in ('0', '1'):
            env = dict(base, GPT_BENCHMARK_RUNTIME_ROOT=str(runtime),
                       GPT_BENCHMARK_USE_LOCAL_CODEX=bridge,
                       GPT_BENCHMARK_CODEX_BRIDGE_URL='http://127.0.0.1:28901')
            code = 'import os,json; print(json.dumps(dict(os.environ)))'
            result = subprocess.run(['bash', str(script), '-c', code], env=env,
                                    capture_output=True, text=True, check=True, timeout=10)
            values = json.loads(result.stdout)
            self.assertEqual(values['GPT_BENCHMARK_PYTHON'], str(runtime / 'bin/isaac-python'))
            self.assertEqual(values['PYTHONPATH'], str(project / 'src'))
            self.assertEqual(values.get('GPT_BENCHMARK_CODEX_BRIDGE_URL'),
                             'http://127.0.0.1:28901' if bridge == '1' else None)

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Process inspection uses Linux /proc')
    def test_idle_check_detects_an_external_project_process(self):
        self.config['project'] = str(self.root)
        worker = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], cwd=self.root)
        try:
            with patch.object(fleet, 'remote') as remote:
                fleet.require_idle('worker')
            command = remote.call_args.args[1]
            # Enumerate only the process owned by this test. Its real /proc
            # metadata still drives the check, without requiring access to
            # unrelated users' processes on a shared CI runner.
            probe = command[2]
            command[2] = (
                'from pathlib import Path\nfrom unittest.mock import patch\n'
                f'with patch.object(Path, "glob", return_value=[Path("/proc/{worker.pid}")]):\n'
                f'    exec({probe!r})\n'
            )
            busy = subprocess.run(command, capture_output=True, text=True, timeout=10)
            self.assertNotEqual(busy.returncode, 0)
            self.assertIn(str(worker.pid), busy.stderr)
        finally:
            worker.terminate()
            worker.wait(timeout=5)
        idle = subprocess.run(command, capture_output=True, text=True, timeout=10)
        self.assertEqual(idle.returncode, 0, idle.stderr)

    @unittest.skipUnless(sys.platform.startswith('linux'), 'Process inspection uses Linux /proc')
    def test_idle_check_refuses_deployment_when_process_permissions_are_unknown(self):
        self.config['project'] = str(self.root)
        with patch.object(fleet, 'remote') as remote:
            fleet.require_idle('worker')
        command = remote.call_args.args[1]
        probe = command[2]
        command[2] = (
            'from pathlib import Path\nfrom unittest.mock import patch\n'
            'with patch.object(Path, "glob", return_value=[Path("/proc/999999999")]), '
            'patch.object(Path, "read_bytes", side_effect=PermissionError):\n'
            f'    exec({probe!r})\n'
        )
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Cannot inspect project processes', result.stderr)


if __name__ == '__main__':
    unittest.main()
