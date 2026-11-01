"""Batch lifecycle tests use mock episodes/short subprocesses, never GPT or Isaac."""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from robot_tasks import batch
from robot_tasks.config import load_config
from robot_tasks.policies.model_config import load_gpt_settings
from robot_tasks.runtime_config import RuntimeConfig


class BatchTests(unittest.TestCase):
    def test_request_log_mirror_handles_partial_and_escaped_lines(self):
        stream = io.BytesIO()
        mirror = batch.RequestLogMirror(stream)
        output = io.StringIO()

        def append(data):
            position = stream.tell()
            stream.seek(0, 2)
            stream.write(data)
            stream.seek(position)

        with redirect_stdout(output):
            append(b"native startup\nR5_POLICY_REQUEST 1\\r")
            mirror.poll()
            self.assertEqual(output.getvalue(), "")
            append(b"\\n[request 1] thinking...\r\n[request 1] response received")
            mirror.poll()
            mirror.poll(final=True)
        self.assertEqual(output.getvalue().splitlines(), [
            "R5_POLICY_REQUEST 1", "[request 1] thinking...", "[request 1] response received"])

    def test_run_job_mirrors_requests_and_keeps_full_log(self):
        directory = self.root / "mirror"
        script = "print('startup noise', flush=True); print('R5_POLICY_REQUEST 1', flush=True); print('[request 1] response received', flush=True)"
        output = io.StringIO()
        with redirect_stdout(output):
            result = batch.run_job([sys.executable, "-c", script], directory, dict(os.environ), 15, lambda _: None)
        self.assertEqual(result["returncode"], 0)
        self.assertIn("R5_POLICY_REQUEST 1", output.getvalue())
        self.assertIn("[request 1] response received", output.getvalue())
        self.assertNotIn("startup noise", output.getvalue())
        self.assertIn("startup noise", (directory / "process.log").read_text(encoding="utf-8"))

    def setUp(self):
        # Reproduce Windows CI's legacy text-file default on every platform.
        # Explicit UTF-8 must work without relying on PYTHONUTF8 or the OS locale.
        original_open = Path.open

        def legacy_locale_open(path, mode="r", buffering=-1, encoding=None, errors=None, newline=None):
            if "b" not in mode and encoding in (None, "locale"):
                encoding = "cp1252"
            return original_open(path, mode, buffering, encoding, errors, newline)

        self.enterContext(patch.object(Path, "open", legacy_locale_open))
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.model = self.root / "model.toml"
        self.model.write_text('[gpt]\nbase_url="https://example.org/v1"\nmodel="offline-fixture"\n', encoding="utf-8")
        self.runtime = RuntimeConfig(sys.executable, self.root, self.root, self.root / "runtime.toml", True)

    def main(self, *args):
        with patch.object(batch, "load_runtime", return_value=self.runtime), redirect_stdout(io.StringIO()):
            return batch.main(list(args))

    def dry_batch(self, name="batch", tasks=("L1-01", "L2-07")):
        root = self.root / name
        self.assertEqual(self.main("--tasks", *tasks, "--output", str(root), "--model-config", str(self.model),
                                  "--dry-run", "--max-calls", "7", "--wall-time-s", "30"), 0)
        return root

    def test_env_never_executes_substitutions_and_exports_take_precedence(self):
        env = self.root / ".env"
        env.write_text('export OPENAI_API_KEY="file-secret" # comment\n'
                       'LITERAL=$(touch unexpected) # stays literal\nSINGLE=\'hello # world\'\n', encoding="utf-8")
        loaded = batch.load_env(env, {"OPENAI_API_KEY": "exported-secret"})
        self.assertEqual(loaded["OPENAI_API_KEY"], "exported-secret")
        self.assertEqual(loaded["LITERAL"], "$(touch unexpected)")
        self.assertEqual(loaded["SINGLE"], "hello # world")
        self.assertFalse((self.root / "unexpected").exists())
        env.write_text('SECRET_VALUE_WITHOUT_ASSIGNMENT', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "line 1") as error:
            batch.load_env(env, {})
        self.assertNotIn("SECRET_VALUE", str(error.exception))

    def test_reasoning_uses_selected_model_unless_cli_explicitly_overrides(self):
        with self.model.open("a", encoding="utf-8") as stream:
            stream.write('reasoning_effort="xhigh"\n')
        for name, arguments, expected in (
                ("model-default", (), "xhigh"),
                ("explicit-override", ("--reasoning-effort", "medium"), "medium")):
            with self.subTest(name=name):
                root = self.root / name
                self.assertEqual(self.main("--tasks", "L1-01", "--output", str(root),
                    "--model-config", str(self.model), "--dry-run", *arguments), 0)
                manifest = json.loads((root / "batch.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["options"]["reasoning_effort"], expected)
                config = load_config(root / "configs/L1-01.toml")
                self.assertEqual(config.raw["policy"]["reasoning_effort"], expected)

    def test_default_budget_and_explicit_zero_wall_time(self):
        original = batch.run_job
        for name, flags, wall in (("paper-default", (), 18000),
                                  ("unlimited-zero", ("--wall-time-s", "0"), 0)):
            with self.subTest(name=name):
                root = self.root / name
                with patch.object(batch, "run_job", wraps=original) as run:
                    self.assertEqual(self.main("--tasks", "L1-01", "L3-03", "--output", str(root),
                        "--model-config", str(self.model), "--dry-run", *flags), 0)
                self.assertEqual(run.call_count, 2)
                for call in run.call_args_list:
                    self.assertEqual(call.args[3], wall + 300 if wall else None)
                    self.assertEqual("GPT_BENCHMARK_WALL_DEADLINE_MONOTONIC" in call.args[2], bool(wall))
                for task in ("L1-01", "L3-03"):
                    config = load_config(root / f"configs/{task}.toml")
                    self.assertEqual(config.max_wall_time_s, wall)
                    self.assertEqual(config.raw["execution"]["max_decisions"], 150)
                self.assertEqual(load_gpt_settings(root / "configs/model.toml").max_calls, 150)
                manifest = json.loads((root / "batch.json").read_text(encoding="utf-8"))
                self.assertEqual(manifest["options"]["wall_time_s"], wall)
                self.assertEqual(manifest["options"]["max_calls"], 150)

    def test_zero_call_cap_and_five_hours_are_frozen_for_each_task(self):
        root = self.root / "five-hours"
        with patch.object(batch, "run_job", wraps=batch.run_job) as run:
            self.assertEqual(self.main("--tasks", "L2-01", "L2-05", "--output", str(root),
                "--model-config", str(self.model), "--dry-run", "--max-calls", "0",
                "--wall-time-s", "18000"), 0)
        self.assertEqual(run.call_count, 2)
        for call in run.call_args_list:
            self.assertEqual(call.args[3], 18300)  # Separate deadline plus startup grace for each task.
        for task in ("L2-01", "L2-05"):
            config = load_config(root / f"configs/{task}.toml")
            self.assertEqual(config.max_wall_time_s, 18000)
            self.assertEqual(config.raw["execution"]["max_decisions"], 0)
        self.assertIn("不限调用次数", (root / "summary.md").read_text(encoding="utf-8"))

    def test_depth_mode_is_frozen_for_batch_and_not_changed_on_resume(self):
        for name, flags, expected in (("rgb", (), False), ("rgbd", ("--depth",), True)):
            with self.subTest(name=name):
                root = self.root / name
                self.assertEqual(self.main("--tasks", "L1-01", "--output", str(root),
                    "--model-config", str(self.model), "--dry-run", *flags), 0)
                config = load_config(root / "configs/L1-01.toml")
                self.assertEqual(config.include_depth, expected)
                self.assertEqual(config.save_depth, expected)
                self.assertEqual(self.main("--resume", str(root), "--no-depth"), 2)

    def test_dry_batch_snapshots_overrides_and_skips_real_only(self):
        from dataclasses import replace
        real_only = replace(batch.TASKS["L2-07"], implemented=False, evidence="real_only")
        with patch.dict(batch.TASKS, {"L2-07": real_only}):
            root = self.dry_batch()
        config = load_config(root / "configs/L1-01.toml")
        self.assertEqual(config.max_wall_time_s, 30)
        self.assertEqual(config.raw["execution"]["max_decisions"], 7)
        self.assertTrue(config.raw["recording"]["save_video"])
        self.assertEqual(config.raw["policy"]["reasoning_effort"], "low")
        self.assertTrue(config.robot.asset_path.is_absolute())
        result = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(result["counts"]["dry_run"], 1)
        self.assertEqual(result["counts"]["skipped"], 1)
        self.assertIsNone(result["results"][0]["success"])
        self.assertTrue((root / "summary.csv").exists())
        self.assertTrue((root / "summary.md").exists())

    def test_resume_skips_completed_jobs_and_does_not_allow_changed_conditions(self):
        root = self.dry_batch()
        with patch.object(batch, "run_job", side_effect=AssertionError("must not rerun")):
            self.assertEqual(self.main("--resume", str(root)), 0)
        with redirect_stdout(io.StringIO()), patch.object(batch.sys, "stderr", io.StringIO()):
            self.assertEqual(self.main("--resume", str(root), "--max-calls", "8"), 2)
            self.model.write_text(self.model.read_text(encoding="utf-8") + "# changed\n", encoding="utf-8")
            self.assertEqual(self.main("--resume", str(root)), 2)
        self.assertEqual(self.main("--summarize", str(root)), 0)  # Offline reporting is still possible.

    def evidence(self, name, result):
        directory = self.root / name
        episode = directory / "episodes/run"
        episode.mkdir(parents=True)
        (episode / "result.json").write_text(json.dumps(result), encoding="utf-8")
        return directory, episode

    def test_interrupted_job_restarts_in_new_directory_and_preserves_old_evidence(self):
        root = self.dry_batch(tasks=("L1-01",))
        state = json.loads((root / "state.json").read_text(encoding="utf-8"))
        job = state["L1-01"]
        old = Path(job["directory"])
        process = json.loads((old / "process.json").read_text(encoding="utf-8"))
        process.update(interrupted=True, returncode=130)
        batch.write_json(old / "process.json", process)
        job.update(status="interrupted")
        batch.write_json(root / "state.json", state)
        self.assertEqual(self.main("--resume", str(root)), 0)
        new = json.loads((root / "state.json").read_text(encoding="utf-8"))["L1-01"]
        self.assertNotEqual(new["directory"], str(old))
        self.assertTrue(list((old / "episodes").glob("*/result.json")))
        self.assertEqual(len(list((root / "tasks/L1-01").glob("run_*"))), 2)

    def test_saved_process_result_is_recovered_without_duplicate_calls(self):
        root = self.dry_batch(tasks=("L1-01",))
        state = json.loads((root / "state.json").read_text(encoding="utf-8"))
        process_path = Path(state["L1-01"]["directory"]) / "process.json"
        process = json.loads(process_path.read_text(encoding="utf-8"))
        process["note"] = "保留中文记录"
        batch.write_json(process_path, process)
        state["L1-01"].update(status="running")
        state["L1-01"].pop("summary")
        batch.write_json(root / "state.json", state)
        with patch.object(batch, "run_job", side_effect=AssertionError("must recover")):
            self.assertEqual(self.main("--resume", str(root)), 0)
        recovered = json.loads((root / "state.json").read_text(encoding="utf-8"))["L1-01"]
        self.assertEqual(recovered["status"], "done")
        self.assertEqual(recovered["summary"]["process"]["note"], "保留中文记录")

    def test_process_liveness_probe_preserves_live_process_and_accepts_exited_pid(self):
        process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
        try:
            self.assertTrue(batch.pid_alive(process.pid))
            self.assertIsNone(process.poll())
        finally:
            process.terminate()
            process.wait(timeout=5)
        self.assertFalse(batch.pid_alive(process.pid))

    def test_missing_key_fails_before_creating_batch_or_launching_worker(self):
        root = self.root / "no_key"
        with patch.dict(os.environ, {}, clear=True), patch.object(batch, "run_job") as run, \
                patch.object(batch.sys, "stderr", io.StringIO()):
            self.assertEqual(self.main("--tasks", "L1-01", "--output", str(root), "--model-config", str(self.model),
                                       "--env-file", str(self.root / "missing.env")), 2)
        run.assert_not_called()
        self.assertFalse(root.exists())

    def test_exit_zero_is_not_task_success_and_unknown_usage_is_not_zero(self):
        directory, episode = self.evidence("failed", {"status": "finished", "task_evaluated": True,
                                                     "success": False, "model_calls": 2})
        events = [
            {"event": "model_exchange", "exchange": {"usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}, "latency_s": 2.}},
            {"event": "model_exchange", "exchange": {"status": "rejected"}},
        ]
        (episode / "episode.jsonl").write_text("\n".join(map(json.dumps, events)) + '\n{"partial":', encoding="utf-8")
        row = batch.collect_result(directory, {"returncode": 0}, False, ["overview"])
        self.assertEqual(row["outcome"], "failed")
        self.assertFalse(row["success"])
        self.assertEqual(row["total_tokens"], 15)
        self.assertEqual(row["usage_samples"], 1)
        self.assertEqual(row["api_latency_s"], 2.)
        self.assertFalse(row["video_complete"])
        self.assertIsNone(row["reasoning_tokens"])
        self.assertEqual(batch.collect_result(directory, {"returncode": 124, "timed_out": True}, False, [])['outcome'], "timeout")
        row = batch.collect_result(self.root / "missing", {"returncode": 0}, False, [])
        self.assertEqual(row["outcome"], "infrastructure_error")
        self.assertIsNone(row["model_calls"])
        self.assertIsNone(row["api_latency_s"])

    def test_video_metadata_is_checked_instead_of_trusting_result_flag(self):
        video = self.root / "motion.mp4"
        video.write_bytes(b"fixture")
        with patch.object(batch.shutil, "which", return_value="ffprobe"), \
                patch.object(batch.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, json.dumps({"streams": [{"width": 640, "height": 480, "duration": "1.2"}]}))):
            self.assertTrue(batch.inspect_video(video)["valid"])
        with patch.object(batch.shutil, "which", return_value=None), \
                patch.object(batch, "inspect_video_with_opencv", return_value={"valid": None, "reason": "video_decoder_unavailable"}):
            self.assertIsNone(batch.inspect_video(video)["valid"])
        self.assertFalse(batch.inspect_video(self.root / "absent.mp4")["valid"])

    def test_video_decode_without_ffprobe_checks_all_recorded_frames(self):
        try:
            import cv2
            import numpy as np
        except ImportError:
            self.skipTest("OpenCV is required for this real video fixture")
        video = self.root / "recording.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 12, (64, 48))
        self.assertTrue(writer.isOpened())
        for i in range(12):
            writer.write(np.full((48, 64, 3), i * 16, dtype=np.uint8))
        writer.release()
        with patch.object(batch.shutil, "which", return_value=None):
            result = batch.inspect_video(video, expected_frames=12)
            self.assertTrue(result["valid"])
            self.assertEqual(result["decoded_frames"], 12)
            self.assertEqual(result["duration"], 1.0)
            self.assertFalse(batch.inspect_video(video, expected_frames=13)["valid"])
            broken = self.root / "truncated.mp4"
            broken.write_bytes(video.read_bytes()[:video.stat().st_size // 2])
            self.assertFalse(batch.inspect_video(broken, expected_frames=12)["valid"])

    def test_unknown_video_verification_is_not_reported_as_incomplete(self):
        directory, episode = self.evidence("unknown_video", {"status": "finished", "task_evaluated": True, "success": False})
        with patch.object(batch, "inspect_video", return_value={"valid": None, "reason": "video_decoder_unavailable"}):
            row = batch.collect_result(directory, {"returncode": 0}, False, ["overview"])
        self.assertIsNone(row["video_complete"])

    def test_invalid_simulation_is_separate_from_model_failure(self):
        directory, _ = self.evidence('invalid', {'status': 'invalid', 'success': None,
            'task_evaluated': False, 'termination_reason': 'simulation_invalid'})
        for returncode in (0, 2):
            row = batch.collect_result(directory, {'returncode': returncode}, False, [])
            self.assertEqual(row['outcome'], 'invalid_episode')
            self.assertIsNone(row['success'])
        row = batch.collect_result(directory, {'returncode': 137}, False, [])
        self.assertEqual(row['outcome'], 'infrastructure_error')

    def test_provider_quota_error_stops_before_next_job(self):
        root = self.dry_batch(tasks=("L1-01", "L1-02"))
        manifest = json.loads((root / "batch.json").read_text(encoding="utf-8"))
        manifest["options"]["dry_run"] = False
        def failed_job(command, directory, environment, timeout, started):
            directory.mkdir(parents=True)
            started({"worker_pid": 99999999})
            episode = directory / "episodes/run"
            episode.mkdir(parents=True)
            (episode / "result.json").write_text(json.dumps({"status": "error", "error": "insufficient_quota"}), encoding="utf-8")
            return {"returncode": 2, "interrupted": False, "cleanup_complete": True}
        with patch.object(batch, "run_job", side_effect=failed_job) as run, redirect_stdout(io.StringIO()):
            self.assertEqual(batch.execute_batch(root, manifest, {}, {}), 2)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(json.loads((root / "summary.json").read_text(encoding="utf-8"))["counts"]["pending"], 1)

    @unittest.skipUnless(sys.platform.startswith("linux"), "Linux process-session ownership fixture")
    def test_timeout_reclaims_separate_session_planner_child(self):
        pidfile = self.root / "child.pid"
        script = ("import subprocess,sys,time; from pathlib import Path; "
                  "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True); "
                  f"Path({str(pidfile)!r}).write_text(str(p.pid),encoding='utf-8'); time.sleep(60)")
        process = batch.run_job([sys.executable, "-c", script], self.root / "timeout", dict(os.environ), 1.5, lambda _: None)
        self.assertTrue(process["timed_out"])
        self.assertTrue(process["cleanup_complete"])
        self.assertIsNone(batch.process_identity(int(pidfile.read_text(encoding="utf-8"))))


if __name__ == "__main__":
    unittest.main()
