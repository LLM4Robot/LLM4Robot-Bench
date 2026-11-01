"""Named task/check dispatch without Isaac Sim, credentials or paid API calls."""

from contextlib import chdir, redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
import re
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from robot_tasks.backends.mock import MockBackend, MockRobot, MockTask, NoopPolicy
from robot_tasks.cli import main
from robot_tasks.integrationtest.debug_check import main as debug_main
from robot_tasks.integrationtest.registry import CHECKS, CheckSpec
from robot_tasks.integrationtest.simulation_process import run_check_process
from robot_tasks.integrationtest.simulation_worker import main as worker_main
from robot_tasks.registry import TASKS, resolve_task


ROOT = Path(__file__).resolve().parents[1]


class NamedEntryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        # Windows TEMP can use an 8.3 alias; the CLI returns resolved paths.
        self.directory = Path(self.temp.name).resolve()

    def config_variant(self, task="GPT-PICK-PLACE", *, task_id=None):
        spec = TASKS[task]
        content = spec.default_config.read_text(encoding="utf-8")
        if 'workflow_profile' in content:
            content = content.replace('"../workflows/gpt_default.toml"',
                                      json.dumps((ROOT / "configs/workflows/gpt_default.toml").as_posix()))
            content += ('\n[runtime]\nmax_steps = 3\n'
                        '\n[execution]\nmax_action_steps=1\ncompletion_samples=1\nwait_steps=1\nsettle_steps=0\nstop_steps=0\n'
                        '\n[recording]\nsave_depth=false\nsave_video=false\n')
        content = content.replace('"../robots/r5_sim.toml"', json.dumps((ROOT / "configs/robots/r5_sim.toml").as_posix()))
        content = re.sub(r"max_steps = \d+", "max_steps = 3", content)
        content = content.replace('"../cameras/r5_orbbec_provisional.toml"',
                                  json.dumps((ROOT / "configs/cameras/r5_orbbec_provisional.toml").as_posix()))
        content = content.replace("save_video = true", "save_video = false")
        content = content.replace("settle_steps = 12", "settle_steps = 0")
        if 'workflow_profile' not in content:
            content = re.sub(r"\[execution\]\n.*?(?=\n\[)", "[execution]\n", content, flags=re.S)
        if task_id is not None:
            content = content.replace(f'id = "{task}"', f'id = "{task_id}"')
        path = self.directory / "variant.toml"
        path.write_text(content, encoding="utf-8")
        return path

    def test_named_defaults_and_config_only_work_outside_project_directory(self):
        with chdir(self.directory):
            for task_id, expected in TASKS.items():
                if expected.evidence == "real_only":
                    with self.assertRaisesRegex(ValueError, "real-only"):
                        resolve_task(task_id)
                    continue
                spec, config = resolve_task(task_id)
                self.assertIs(spec, expected)
                self.assertEqual(config.source, expected.default_config)
                self.assertEqual(resolve_task(config_path=expected.default_config)[0], spec)
        variant = self.config_variant()
        spec, config = resolve_task("GPT-PICK-PLACE", variant)
        self.assertEqual(config.source, variant)
        self.assertEqual(config.max_steps, 3)

    def test_wrong_task_or_missing_identity_is_rejected_before_execution(self):
        for arguments, message in (
            (["run", "missing"], "Unknown task ID"),
            (["run"], "Provide a task ID"),
            (["run", "GPT-PICK-PLACE", "--config", str(TASKS["L2-01"].default_config)], "Task/config mismatch"),
        ):
            with self.subTest(arguments=arguments), patch("robot_tasks.cli.execute_task") as execute, \
                    redirect_stderr(io.StringIO()) as error:
                with self.assertRaises(SystemExit) as raised:
                    main(arguments)
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(message, error.getvalue())
                execute.assert_not_called()

    def test_dry_run_by_name_never_calls_a_model_or_reports_task_success(self):
        with patch("robot_tasks.execution.GPTPolicy") as policy, redirect_stdout(io.StringIO()) as output:
            code = main(["dry-run", "GPT-PICK-PLACE", "--output", str(self.directory)])
        policy.assert_not_called()
        result = json.loads(output.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(result["status"], "dry_run_complete")
        self.assertFalse(result["task_evaluated"])
        self.assertIsNone(result["success"])

    def test_adding_a_registered_task_needs_no_cli_change(self):
        from fixtures import RGBDBackend
        from robot_tasks.robots.preparation import preparation_action
        from robot_tasks.types import Feedback
        path = self.config_variant(task_id="TEST-NEW")
        factory = Mock(return_value=MockTask(), spec=[])
        spec = replace(TASKS["GPT-PICK-PLACE"], task_id="TEST-NEW", default_config=path,
                       factory=factory, implemented=True)
        policy_factory = Mock(side_effect=lambda config, model: NoopPolicy(config.control_dt_s))
        backend = RGBDBackend()
        robot = MockRobot()
        robot.apply = Mock()
        robot.feedback = Mock(return_value=Feedback("completed"))
        with patch.dict(TASKS, {spec.task_id: spec}), \
                patch.dict("robot_tasks.execution.POLICIES", {"fixture": policy_factory}), \
                patch("robot_tasks.execution.IsaacSimBackend", return_value=backend), \
                patch("robot_tasks.execution.R5Adapter", return_value=robot), \
                patch("robot_tasks.runner.print"), \
                redirect_stdout(io.StringIO()) as output:
            code = main(["run", spec.task_id, "--policy", "fixture", "--gui", "--output", str(self.directory)])
        self.assertEqual(code, 0)
        factory.assert_called_once_with()
        config = policy_factory.call_args.args[0]
        self.assertEqual(robot.apply.call_args_list[0].args[0], preparation_action(config))
        self.assertFalse(config.raw["simulation"]["headless"])
        self.assertEqual(json.loads(output.getvalue())["task_id"], spec.task_id)
        self.assertTrue(backend.closed)

    def test_all_checks_dispatch_their_registered_implementation_in_worker(self):
        for name, spec in CHECKS.items():
            with self.subTest(check=name):
                function = Mock(return_value=(self.directory, {"passed": True}))
                module = SimpleNamespace(**{spec.entry_point.split(":")[1]: function})
                args = ["--check", name, "--config", str(spec.default_config),
                        "--run-directory", str(self.directory)]
                if spec.uses_simulation:
                    args.append("--gui")
                if spec.uses_model:
                    args += ["--model-config", str(self.directory / "model.toml")]
                with patch("robot_tasks.integrationtest.registry.import_module", return_value=module):
                    self.assertEqual(worker_main(args), 0)
                self.assertEqual(function.call_args.args, (spec.default_config, self.directory.parent))
                self.assertEqual(function.call_args.kwargs["gui"], spec.uses_simulation)
                self.assertEqual(function.call_args.kwargs["run_directory"], self.directory)
                if spec.uses_model:
                    self.assertEqual(function.call_args.kwargs["model_path"], self.directory / "model.toml")

    def test_check_overrides_cannot_silently_change_kind_or_ignore_flags(self):
        for name, path, options, message in (
            ("gpt-pick-place", TASKS["L2-01"].default_config, {}, "Check/config mismatch"),
            ("joint", None, {}, "Unknown check"),
            ("gpt", None, {}, "Unknown check"),
        ):
            with self.subTest(name=name), patch("robot_tasks.integrationtest.simulation_process.subprocess.Popen") as process:
                with self.assertRaisesRegex(ValueError, message):
                    run_check_process(name, self.directory, config_path=path, **options)
                process.assert_not_called()

    def test_check_cli_passes_name_and_overrides_to_common_process(self):
        config = CHECKS["gpt-pick-place"].default_config
        model = self.directory / "model.toml"
        with patch("robot_tasks.cli.run_check_process", return_value=(self.directory, True)) as execute:
            self.assertEqual(main(["check", "gpt-pick-place", "--config", str(config), "--model-config", str(model),
                                   "--gui", "--output", str(self.directory)]), 0)
        execute.assert_called_once_with("gpt-pick-place", self.directory, config_path=config, gui=True, model_config=model)

    def test_process_passes_resolved_config_and_model_to_worker(self):
        observed = []

        def fake_process(command, **kwargs):
            observed.append(command)
            output = Path(command[command.index("--run-directory") + 1])
            (output / "result.json").write_text('{"status":"completed","passed":true}', encoding="utf-8")
            process = Mock(stdout=io.StringIO(), pid=123, returncode=0)
            process.wait.return_value = 0
            return process

        with chdir(self.directory), patch("robot_tasks.integrationtest.simulation_process.subprocess.Popen", side_effect=fake_process), \
                redirect_stdout(io.StringIO()):
            output, passed = run_check_process("gpt-pick-place", self.directory, model_config=Path("model.toml"), gui=True)
        self.assertTrue(passed)
        command = observed[0]
        self.assertEqual(command[command.index("--config") + 1], str(CHECKS["gpt-pick-place"].default_config))
        self.assertEqual(command[command.index("--model-config") + 1], str(self.directory / "model.toml"))
        self.assertIn("--gui", command)
        self.assertTrue((output / "process.json").is_file())

    def test_debugger_uses_same_registry_and_debug_default_or_override(self):
        debugger = Mock()
        for override in (None, CHECKS["gpt-pick-place"].default_config):
            args = ["--check", "gpt-pick-place", "--gui", "--output", str(self.directory)]
            if override is not None:
                args += ["--config", str(override)]
            with self.subTest(override=override), patch.dict("sys.modules", {"debugpy": debugger}), \
                    patch("robot_tasks.integrationtest.debug_check.sys.platform", "win32"), \
                    patch("robot_tasks.runtime.configure_windows_storage"), \
                    patch.object(CheckSpec, "run", return_value=(self.directory, {"passed": True})) as execute, \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(debug_main(args), 0)
                self.assertEqual(execute.call_args.args[0], override or CHECKS["gpt-pick-place"].default_config)
        self.assertEqual(debugger.wait_for_client.call_count, 2)


if __name__ == "__main__":
    unittest.main()
