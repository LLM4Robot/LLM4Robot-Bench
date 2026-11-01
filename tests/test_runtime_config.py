"""Machine paths must not change experiment settings or require Windows files."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from robot_tasks.runtime_config import load_runtime
from robot_tasks.runtime import configure_windows_storage, snapshot_storage_config, snapshot_runtime_config, storage_report


class RuntimeConfigTests(unittest.TestCase):
    def test_remote_generated_asset_cache_does_not_change_scene_settings(self):
        from dataclasses import asdict
        from robot_tasks.backends.settings import SimulationSettings
        from robot_tasks.registry import resolve_task

        _, config = resolve_task('L2-02')
        with patch.dict(os.environ, {}, clear=True):
            baseline = asdict(SimulationSettings.from_config(config))
        for machine in ('worker_3', 'worker_7'):
            cache = self.root / machine / 'generated_assets'
            with patch.dict(os.environ, {'GPT_BENCHMARK_ASSET_CACHE': str(cache)}):
                settings = asdict(SimulationSettings.from_config(config))
            self.assertEqual(settings.pop('asset_directory'), cache)
            self.assertEqual(settings, {k: v for k, v in baseline.items() if k != 'asset_directory'})

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        # Match the config loader's canonical directory, including 8.3 aliases.
        self.root = Path(self.temp.name).resolve()
        # These tests clear the environment; Windows then has no user-directory
        # variables. Model a home directory without depending on the host user.
        home = patch('robot_tasks.runtime_config.Path.home', return_value=self.root / 'user')
        home.start()
        self.addCleanup(home.stop)
        self.path = self.root / 'local.toml'
        self.path.write_text('')
        self.env = patch.dict(os.environ, {'GPT_BENCHMARK_RUNTIME_CONFIG': str(self.path)}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_relative_paths_are_resolved_against_the_declaring_file(self):
        self.path.write_text('[isaacsim]\npython="venv/bin/python"\n[paths]\ncache_dir="cache"\noutput_dir="out"')
        cfg = load_runtime()
        self.assertEqual(cfg.python, str(self.root / 'venv/bin/python'))
        self.assertEqual(cfg.cache_dir, self.root / 'cache')
        self.assertEqual(cfg.output_dir, self.root / 'out')

    def test_only_explicit_python_override_is_accepted(self):
        self.path.write_text('[isaacsim]\npython="unused"')
        with patch.dict(os.environ, {'GPT_BENCHMARK_PYTHON': str(self.root / 'chosen')}):
            self.assertEqual(load_runtime().python, str(self.root / 'chosen'))

    def test_experiment_model_and_unknown_configuration_are_rejected(self):
        for text in ('[simulation]\ngraphics_backend="auto"', '[model]\nprovider_config="model.toml"',
                     '[paths]\noutput_dir=""', '[isaacsim]\npython=true'):
            with self.subTest(text=text):
                self.path.write_text(text)
                with self.assertRaises(ValueError):
                    load_runtime()
        with self.assertRaises(FileNotFoundError):
            load_runtime(self.root / 'missing.toml')

    def test_missing_default_local_config_uses_current_python_and_project_output(self):
        import sys
        with patch.dict(os.environ, {}, clear=True), patch('robot_tasks.runtime_config.PROJECT_ROOT', self.root):
            cfg = load_runtime()
        self.assertEqual(cfg.python, sys.executable)
        self.assertEqual(cfg.output_dir, self.root / 'runs')
        self.assertFalse(cfg.cache_explicit)

    def test_config_load_is_read_only(self):
        self.path.write_text('[paths]\ncache_dir="not_created"\noutput_dir="also_not_created"')
        load_runtime()
        self.assertFalse((self.root / 'not_created').exists())
        self.assertFalse((self.root / 'also_not_created').exists())

    def test_explicit_system_cache_does_not_require_a_home_directory(self):
        for platform, variable in (('win32', 'LOCALAPPDATA'), ('linux', 'XDG_CACHE_HOME')):
            with self.subTest(platform=platform), \
                    patch('robot_tasks.runtime_config.sys.platform', platform), \
                    patch.dict(os.environ, {variable: str(self.root / 'system_cache')}), \
                    patch('robot_tasks.runtime_config.Path.home', side_effect=RuntimeError('No home directory')) as home:
                cfg = load_runtime()
                self.assertEqual(cfg.cache_dir, self.root / 'system_cache/GPT_Benchmark')
                home.assert_not_called()

    def test_portable_storage_uses_configured_cache_without_changing_home(self):
        self.path.write_text('[paths]\ncache_dir="cache"')
        original = os.environ.get('HOME')
        storage = configure_windows_storage()
        self.assertFalse(storage.legacy)
        self.assertEqual(storage.kit_root, self.root / 'cache/kit')
        self.assertTrue(storage.kit_root.is_dir())
        self.assertEqual(os.environ.get('HOME'), original)

    def test_linux_snapshot_does_not_read_windows_private_file(self):
        with patch('robot_tasks.runtime.sys.platform', 'linux'), patch('robot_tasks.runtime.STORAGE_CONFIG', self.root / 'missing.json'):
            snapshot_storage_config(self.root)
        self.assertFalse(json.loads((self.root / 'storage_config.json').read_text())['windows_storage_configured'])

    def test_python_only_local_config_preserves_legacy_windows_storage(self):
        self.path.write_text('[isaacsim]\npython="current"')
        source = self.root / 'windows.json'
        source.write_text('{"root":"D:/example"}')
        with patch('robot_tasks.runtime.sys.platform', 'win32'), patch('robot_tasks.runtime.STORAGE_CONFIG', source):
            snapshot_storage_config(self.root)
        self.assertEqual(source.read_bytes(), (self.root / 'storage_config.json').read_bytes())

    def test_legacy_storage_uses_the_configured_root_without_a_drive_restriction(self):
        source = self.root / 'windows.json'
        cache = self.root / 'chosen cache'
        source.write_text(json.dumps({'root': str(cache), 'kit_root': 'kit',
                                      'environment': {'TEMP': 'tmp', 'PYTHONPYCACHEPREFIX': 'bytecode'},
                                      'junctions': []}))
        with patch('robot_tasks.runtime.sys.platform', 'win32'), \
                patch('robot_tasks.runtime.STORAGE_CONFIG', source), \
                patch('tempfile.tempdir'), patch('sys.pycache_prefix'):
            storage = configure_windows_storage()
            report = storage_report(storage)
        self.assertTrue(storage.legacy)
        self.assertEqual(storage.root, cache)
        self.assertEqual(Path(storage.environment['TEMP']), cache / 'tmp')
        self.assertEqual(report['configured_drive'], cache.drive.upper())
        self.assertNotIn('all_research_paths_on_d', report)

    def test_legacy_storage_still_rejects_a_cache_path_escaping_its_root(self):
        source = self.root / 'windows.json'
        source.write_text(json.dumps({'root': str(self.root / 'cache'), 'kit_root': 'kit',
                                      'environment': {'TEMP': '../escape'}, 'junctions': []}))
        with patch('robot_tasks.runtime.sys.platform', 'win32'), \
                patch('robot_tasks.runtime.STORAGE_CONFIG', source), \
                self.assertRaisesRegex(ValueError, 'escapes configured root'):
            configure_windows_storage()

    def test_snapshot_records_actual_python_and_omits_credentials(self):
        import sys
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'do-not-record-this'}):
            snapshot_runtime_config(self.root)
        text = (self.root / 'runtime_config.json').read_text()
        self.assertNotIn('do-not-record-this', text)
        self.assertEqual(json.loads(text)['python'], sys.executable)


if __name__ == '__main__':
    unittest.main()
