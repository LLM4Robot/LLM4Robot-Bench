"""Remote launch checks never start a real simulation or model request."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(os.name == 'posix', 'Remote workers use Linux')
class RemoteRuntimeTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location(
            'remote_job_test', Path(__file__).resolve().parents[1] / 'skills/r5-remote-deploy/scripts/remote_job.py')
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.launcher = self.root / 'launcher'
        self.launcher.write_text('#!/bin/sh\nexit 0\n')
        self.launcher.chmod(0o755)
        self.enterContext(patch.dict(os.environ, {
            'GPT_BENCHMARK_PYTHON': str(self.launcher),
            'GPT_BENCHMARK_PLANNER_PYTHON': str(self.launcher),
        }))

    def test_checks_the_actual_child_interpreters(self):
        with patch.object(self.module.subprocess, 'run', wraps=subprocess.run) as run:
            self.module.check_runtime()
        self.assertEqual(run.call_count, 2)
        self.assertIn('import numpy, PIL', run.call_args_list[0].args[0][-1])
        self.assertIn('import torch, curobo', run.call_args_list[1].args[0][-1])

    def test_broken_wrapper_aborts_before_batch_or_job_record(self):
        self.launcher.write_text('#!/bin/sh\necho "python.sh missing" >&2\nexit 127\n')
        with (patch.object(self.module, 'STATE', self.root / 'state'),
              patch('sys.argv', ['remote_job.py', 'start', '--tasks', 'L2-01']),
              patch.object(self.module.subprocess, 'Popen', wraps=subprocess.Popen) as popen,
              self.assertRaisesRegex(SystemExit, 'Isaac runtime preflight failed')):
            self.module.main()
        self.assertFalse((self.root / 'state/job.json').exists())
        self.assertEqual(popen.call_count, 1)  # Only preflight, never batch_run.py.

    def test_missing_launcher_is_explicit(self):
        self.launcher.unlink()
        with self.assertRaisesRegex(SystemExit, 'Isaac runtime missing'):
            self.module.check_runtime()

    def test_other_deployment_busy_blocks_start_before_runtime(self):
        original = self.root / 'another-experiment'
        job = original / 'runs/remote' / os.uname().nodename / 'job.json'
        job.parent.mkdir(parents=True)
        job.write_text(json.dumps({'pid': os.getpid(),
                                  'identity': self.module.identity(os.getpid()), 'tasks': ['L2-05']}))
        with patch.object(self.module, 'ROOT', self.root / 'current-project'), \
                patch.dict(os.environ, {'GPT_BENCHMARK_GUARD_PROJECTS': json.dumps([str(original)])}):
            with self.assertRaisesRegex(SystemExit, 'busy'):
                self.module.require_other_deployment_idle()
            job.write_text(json.dumps({'pid': os.getpid(), 'identity': 'stale'}))
            self.module.require_other_deployment_idle()


if __name__ == '__main__':
    unittest.main()
