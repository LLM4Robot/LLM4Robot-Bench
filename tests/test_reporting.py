"""Reports preserve score semantics and distinguish task error from TCP tracking."""

import json
from pathlib import Path
import tempfile
import unittest

from robot_tasks.recording import finish_run, write_json
from robot_tasks.reporting import collect_run_evidence, write_run_report


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)

    def events(self, values):
        (self.run / "episode.jsonl").write_text(
            "".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")

    def result(self, **updates):
        result = {"task_id": "TEST", "status": "finished", "task_evaluated": True,
                  "success": False, "false_completion": True, "termination_reason": "policy_requested_stop",
                  "assessment": {"status": "failed", "reason": "Target not reached",
                                 "metrics": {"placement": {"max_xy_error_m": .034}}}}
        return {**result, **updates}

    def test_tracking_statistics_use_action_end_feedback_not_physics_or_ik_errors(self):
        def event(kind, position, orientation):
            return {"event": kind, "observation": {"feedback": {"metrics": {"tcp": {
                "left": {"position_error_m": position, "orientation_error_rad": orientation}}}}}}
        self.events([event("physics", 100., 100.), event("transition", .003, .01),
                     event("transition", .009, .03), event("action_rejected", 99., 99.),
                     {"event": "model_exchange", "exchange": {"latency_s": 12.}},
                     {"event": "model_exchange", "exchange": {}},
                     {"event": "policy_configuration", "configuration": {"provider": {
                         "model": "test-model", "reasoning_effort": "medium", "base_url": "PRIVATE_ENDPOINT"}}}])
        finish_run(self.run, self.result(invalid_model_actions=2))
        data = collect_run_evidence(self.run)
        metric = data["tracking"]["left"]["position_error_m"]
        self.assertEqual(metric["samples"], 2)
        self.assertAlmostEqual(metric["mean"], .006)
        self.assertAlmostEqual(metric["max"], .009)
        self.assertEqual(data["api_latency_total_s"], 12.)
        self.assertEqual(data["api_latency_samples"], 1)
        self.assertEqual(data["logged_model_exchanges"], 2)
        text = (self.run / "report.md").read_text(encoding="utf-8")
        self.assertIn("| 任务评价 | 失败 |", text)
        self.assertIn("placement.max_xy_error_m | 0.034", text)
        self.assertIn("| 错误宣告完成 | 是 |", text)
        self.assertIn("| 无效模型动作数（计入调用预算） | 2 |", text)
        self.assertNotIn("PRIVATE_ENDPOINT", text)

    def test_solver_name_comes_from_recorded_scene_not_current_defaults(self):
        self.events([])
        for solver in ("numerical", "lula"):
            write_json(self.run / "scene.json", {"ik_solver": solver})
            finish_run(self.run, self.result())
            text = (self.run / "report.md").read_text(encoding="utf-8")
            self.assertIn(f"| IK 求解器 | {solver} |", text)

    def test_no_measurements_are_unknown_not_zero_and_partial_log_is_disclosed(self):
        self.events([])
        (self.run / "episode.jsonl").write_text('{"event":', encoding="utf-8")
        finish_run(self.run, self.result(task_evaluated=False, success=None, status="error", error="API timeout"))
        data = collect_run_evidence(self.run)
        self.assertIsNone(data["api_latency_total_s"])
        self.assertEqual(data["tracking"], {})
        text = (self.run / "report.md").read_text(encoding="utf-8")
        self.assertIn("未评估", text)
        self.assertIn("第 1 行不完整", text)
        self.assertIn("不按零误差处理", text)
        self.assertIn("API timeout", text)

    def test_refresh_after_cleanup_error_preserves_original_artifacts(self):
        self.events([])
        video = self.run / "video" / "left_wrist" / "motion.mp4"
        video.parent.mkdir(parents=True)
        video.write_bytes(b"video fixture, never presented as a real recording")
        finish_run(self.run, self.result(success=True, false_completion=False))
        self.assertIn("| 任务评价 | 成功 |", (self.run / "report.md").read_text(encoding="utf-8"))
        finish_run(self.run, self.result(task_evaluated=False, success=None,
                                       status="error", termination_reason="cleanup_error", error="close failed"))
        text = (self.run / "report.md").read_text(encoding="utf-8")
        self.assertNotIn("| 任务评价 | 成功 |", text)
        self.assertIn("(<video/left_wrist/motion.mp4>)", text)
        self.assertEqual(video.read_bytes(), b"video fixture, never presented as a real recording")
        self.assertEqual((self.run / "episode.jsonl").read_text(), "")
        self.assertFalse((self.run / "report.md.tmp").exists())

    def test_expected_failed_check_exit_is_not_mislabelled_as_process_crash(self):
        write_json(self.run / "process.json", {"returncode": 2, "passed": False, "timed_out": False})
        finish_run(self.run, self.result(status="completed", passed=False))
        text = (self.run / "report.md").read_text(encoding="utf-8")
        self.assertIn("| 任务评价 | 失败 |", text)
        self.assertNotIn("进程退出异常或超时", text)
        write_json(self.run / "process.json", {"returncode": -9, "passed": False, "timed_out": True})
        write_run_report(self.run)
        self.assertIn("进程退出异常或超时", (self.run / "report.md").read_text(encoding="utf-8"))

    def test_engineering_pass_and_missing_result_do_not_claim_task_success(self):
        finish_run(self.run, {"status": "completed", "passed": True, "task_evaluated": False,
                              "pose_checks": [{"position_error_m": .002, "passed": True}]})
        text = (self.run / "report.md").read_text(encoding="utf-8")
        self.assertIn("| 工程检查通过 | 是 |", text)
        self.assertIn("未评估", text)
        self.assertIn('"position_error_m": 0.002', text)
        (self.run / "result.json").unlink()
        write_run_report(self.run)
        self.assertIn("没有完整的 result.json", (self.run / "report.md").read_text(encoding="utf-8"))

    def test_retry_report_exposes_first_failure_even_when_final_attempt_succeeds(self):
        self.events([])
        finish_run(self.run, self.result(success=True, evaluation_protocol="retry_with_history",
            reset_count=1, failed_attempts=1, first_attempt_success=False, success_attempt=2,
            attempts=[{"attempt": 1, "status": "failed", "decision_start": 1, "decision_end": 3,
                       "assessment": {"reason": "Cylinder was knocked over"}},
                      {"attempt": 2, "status": "succeeded", "decision_start": 4, "decision_end": 9,
                       "assessment": {"reason": "Seated"}}]))
        text = (self.run / "report.md").read_text(encoding="utf-8")
        self.assertIn("| 首次尝试成功 | 否 |", text)
        self.assertIn("| 失败尝试数 | 1 |", text)
        self.assertIn("| 成功发生于第几次尝试 | 2 |", text)
        self.assertIn("Cylinder was knocked over", text)
        self.assertIn("不能当作独立单次尝试的成功率", text)

    def test_physical_goal_and_process_violation_do_not_become_full_success(self):
        self.events([])
        assessment = {"status": "failed", "reason": "Physical goal reached; load spilled",
                      "metrics": {"scoring_protocol": "recoverable_progress_v1",
                                  "goal_complete": True, "process_constraints_satisfied": False,
                                  "constraint_violations": ["Load spilled"],
                                  "quality_warnings": ["Speed reference exceeded"]}}
        finish_run(self.run, self.result(assessment=assessment, false_completion=False))
        text = (self.run / "report.md").read_text(encoding="utf-8")
        self.assertIn("| 任务评价 | 失败 |", text)
        self.assertIn("| 目标状态满足（仍需满足稳定时间） | 是 |", text)
        self.assertIn("| 必要过程约束满足 | 否 |", text)
        self.assertIn("Load spilled", text)
        self.assertIn("Speed reference exceeded", text)
        self.assertIn("不代表已测得材料损坏", text)

    def test_invalid_l2_report_keeps_stage_progress_without_scoring_model_failure(self):
        self.events([])
        assessment = {'status': 'invalid', 'reason': 'Inverted FEM elements', 'metrics': {
            'scoring_protocol': 'native_l2_validity_v2', 'stages': {'lifted': True},
            'invalid_reason': 'simulation_invalid'}}
        finish_run(self.run, self.result(status='invalid', task_evaluated=False, success=None,
                                        evaluation_validity='invalid', assessment=assessment))
        text = (self.run / 'report.md').read_text(encoding='utf-8')
        self.assertIn('不计入模型成功／失败分母', text)
        self.assertIn('| lifted | 是 |', text)
        self.assertIn('Inverted FEM elements', text)


if __name__ == "__main__":
    unittest.main()
