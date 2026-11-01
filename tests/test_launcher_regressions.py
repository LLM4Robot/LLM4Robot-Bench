"""Interpreter identity, option parsing and cancellation regression tests."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import venv
from unittest.mock import Mock, patch

from robot_tasks.paths import PROJECT_ROOT
from robot_tasks.runtime_config import load_runtime

SPEC = importlib.util.spec_from_file_location('portable_launcher', PROJECT_ROOT / 'tools/launch.py')
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class InterpreterTests(unittest.TestCase):
    @unittest.skipIf(os.name == 'nt', 'POSIX venv symlinks')
    def test_venv_interpreter_keeps_its_environment_from_toml_and_env(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            venv.EnvBuilder(with_pip=False, symlinks=True).create(root / 'venv')
            python = root / 'venv/bin/python'
            config = root / 'local.toml'
            config.write_text('[isaacsim]\npython="venv/bin/python"')
            for overrides in ({}, {'GPT_BENCHMARK_PYTHON': str(python)}):
                with self.subTest(overrides=overrides), patch.dict(os.environ, overrides, clear=True):
                    chosen = load_runtime(config).python
                self.assertEqual(chosen, str(python))
                prefix = subprocess.check_output([chosen, '-c', 'import sys; print(sys.prefix)'], text=True).strip()
                self.assertEqual(prefix, str(root / 'venv'))

    @unittest.skipIf(os.name == 'nt', 'POSIX execve dispatch')
    def test_launcher_uses_local_python_and_forwards_only_existing_cli_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'local.toml'
            config.write_text('[isaacsim]\npython=' + json.dumps(sys.executable))
            args = ['check', 'gpt-pick-place', '--gui']
            with patch.dict(os.environ, {'GPT_BENCHMARK_RUNTIME_CONFIG': str(config)}, clear=True), \
                    patch.object(sys, 'argv', ['launch.py', *args]), patch.object(launcher.os, 'execve') as execute:
                launcher.main()
                self.assertEqual(execute.call_args.args[0], sys.executable)
                self.assertEqual(execute.call_args.args[1][3:], args)


class WindowsCancellationTests(unittest.TestCase):
    @unittest.skipUnless(os.name == 'nt', 'Windows console inheritance')
    def test_native_launcher_clears_inherited_ignore_ctrl_c_flag(self):
        kernel = Mock()
        kernel.SetConsoleCtrlHandler.return_value = True
        process = Mock()
        process.wait.side_effect = [subprocess.TimeoutExpired('fixture', .2), 0]
        with patch('ctypes.WinDLL', return_value=kernel), \
                patch.object(launcher.subprocess, 'Popen', return_value=process):
            self.assertEqual(launcher.run_windows(['fixture'], {}, 2), 0)
        kernel.SetConsoleCtrlHandler.assert_called_once_with(None, False)
        self.assertTrue(all(call.kwargs['timeout'] == .2 for call in process.wait.call_args_list))

    def run_case(self, wait_results):
        process = Mock(pid=987)
        process.wait.side_effect = wait_results
        with patch.object(launcher.subprocess, 'Popen', return_value=process) as popen, \
                patch.object(launcher.subprocess, 'run') as kill_tree, \
                patch.object(subprocess, 'CREATE_NEW_PROCESS_GROUP', 512, create=True), \
                patch.object(signal, 'CTRL_BREAK_EVENT', 1, create=True), \
                patch.object(signal, 'signal'):
            result = launcher.run_windows(['python', 'fixture'], {}, 2)
        self.assertEqual(result, 130)
        popen.assert_called_once_with(['python', 'fixture'], env={}, creationflags=512)
        process.send_signal.assert_called_once_with(1)
        return process, kill_tree

    def test_cancel_allows_supervisor_to_finish_cleanup(self):
        process, kill_tree = self.run_case([KeyboardInterrupt(), 130])
        self.assertEqual(process.wait.call_args.kwargs['timeout'], 6)
        kill_tree.assert_not_called()

    def test_cancel_timeout_uses_bounded_tree_termination(self):
        process, kill_tree = self.run_case([KeyboardInterrupt(), subprocess.TimeoutExpired('fixture', 6), 1])
        self.assertEqual(kill_tree.call_args.args[0], ['taskkill', '/PID', '987', '/T', '/F'])
        self.assertEqual(kill_tree.call_args.kwargs['timeout'], 2)
        self.assertEqual(process.wait.call_args.kwargs['timeout'], 2)


@unittest.skipIf(os.name == 'nt', 'POSIX signal/session integration')
class PosixCancellationTests(unittest.TestCase):
    def test_actual_launcher_preserves_slow_supervisor_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            package = root / 'src/robot_tasks'
            package.mkdir(parents=True)
            (package / '__init__.py').write_text('')
            pidfile = root / 'worker.pid'
            (package / '__main__.py').write_text('''import signal,subprocess,sys,time
from pathlib import Path
worker=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'], start_new_session=True)
def stop(*args):
    time.sleep(0.6)
    worker.kill()
    worker.wait(timeout=2)
    raise SystemExit(130)
signal.signal(signal.SIGINT,stop)
Path(sys.argv[1]).write_text(str(worker.pid))
while worker.poll() is None:
    time.sleep(.05)
''')
            bootstrap = root / 'bootstrap.py'
            bootstrap.write_text(f'''import importlib.util,sys
from pathlib import Path
sys.path.insert(0, {str(PROJECT_ROOT / 'src')!r})
spec=importlib.util.spec_from_file_location('launch', {str(PROJECT_ROOT / 'tools/launch.py')!r})
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.ROOT=Path({str(root)!r})
from types import SimpleNamespace
m.load_runtime=lambda: SimpleNamespace(python=sys.executable)
sys.argv=['launch.py', {str(pidfile)!r}]
raise SystemExit(m.main())
''')
            process = subprocess.Popen([sys.executable, str(bootstrap)], start_new_session=True,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            worker = None
            try:
                deadline = time.monotonic() + 5
                while not pidfile.exists() and time.monotonic() < deadline:
                    time.sleep(.02)
                worker = int(pidfile.read_text())
                os.killpg(process.pid, signal.SIGINT)
                self.assertEqual(process.wait(timeout=5), 130)
                with self.assertRaises(ProcessLookupError):
                    os.kill(worker, 0)
            finally:
                if worker:
                    try:
                        os.killpg(worker, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


@unittest.skipUnless(os.name == 'nt' and (shutil.which('pwsh') or shutil.which('powershell')),
                     'Requires native Windows PowerShell')
class PowerShellFallbackTests(unittest.TestCase):
    def test_storage_mapping_accepts_the_callers_absolute_directory(self):
        shell = shutil.which('pwsh') or shutil.which('powershell')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tools = root / 'tools'
            tools.mkdir()
            runtime = root / 'configs/runtime'
            runtime.mkdir(parents=True)
            shutil.copyfile(PROJECT_ROOT / 'tools/research_storage.ps1', tools / 'research_storage.ps1')
            cache = root / 'custom cache'
            (runtime / 'windows.json').write_text(json.dumps({
                'root': str(cache), 'environment': {'TEMP': 'tmp'}}))
            probe = tools / 'probe.ps1'
            probe.write_text('''. "$PSScriptRoot/research_storage.ps1"
$ErrorActionPreference = 'Stop'
$previous = Set-ResearchStorageEnvironment
try { @{ temp = $env:TEMP } | ConvertTo-Json -Compress }
finally { Restore-ResearchStorageEnvironment $previous }
''')
            completed = subprocess.run([shell, '-NoProfile', '-ExecutionPolicy', 'Bypass',
                                        '-File', str(probe)], capture_output=True, text=True, timeout=20)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(Path(json.loads(completed.stdout)['temp']).resolve(), (cache / 'tmp').resolve())
            self.assertTrue((cache / 'tmp').is_dir())

    def test_legacy_default_bootstraps_without_path_python_and_respects_local_selection(self):
        shell = shutil.which('pwsh') or shutil.which('powershell')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tools = root / 'tools'
            tools.mkdir()
            runtime = root / 'configs/runtime'
            runtime.mkdir(parents=True)
            # Emulate the company's existing default interpreter at a temporary fixture path.
            source = (PROJECT_ROOT / 'tools/run_r5_check.ps1').read_text()
            source = source.replace("'D:\\conda\\envs\\isaac51\\python.exe'",
                                    "'" + sys.executable.replace("'", "''") + "'")
            script = tools / 'run_r5_check.ps1'
            script.write_text(source)
            (tools / 'launch.py').write_text(
                "import os,sys,json; print(json.dumps({'python':sys.executable,"
                "'override':os.environ.get('GPT_BENCHMARK_PYTHON'),"
                "'eula':os.environ.get('OMNI_KIT_ACCEPT_EULA'),"
                "'no_user_site':os.environ.get('PYTHONNOUSERSITE')}))")
            env = {k: v for k, v in os.environ.items() if not k.startswith('GPT_BENCHMARK_')}
            env['PATH'] = ''  # Original working invocation did not need PATH Python.
            for local, explicit in ((False, False), (True, False), (True, True)):
                with self.subTest(local=local, explicit=explicit):
                    if local:
                        (runtime / 'local.toml').write_text('[isaacsim]\npython="current"')
                    command = [shell, '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(script)]
                    if explicit:
                        command += ['-TaskPython', sys.executable]
                    completed = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
                    self.assertEqual(completed.returncode, 0, completed.stderr)
                    report = json.loads(completed.stdout.strip())
                    self.assertEqual(Path(report['python']), Path(sys.executable))
                    self.assertEqual(report['override'], sys.executable if explicit or not local else None)
                    self.assertEqual(report['eula'], 'YES')
                    self.assertEqual(report['no_user_site'], '1')


if __name__ == '__main__':
    unittest.main()
