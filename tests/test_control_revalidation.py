"""CPU regressions: no provider, Isaac startup or robot actuation."""
import ast
from dataclasses import asdict
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools/control_revalidation"))
from arrival import annotate_arrival
from robot_tasks.types import Action, ArmCommand, Feedback
import planner_python


class ArrivalTests(unittest.TestCase):
    def test_old_precision_stops_are_feedback_not_termination(self):
        for error in [0.000584369933496461, 0.00492686931760456, 0.01]:
            with self.subTest(error=error):
                raw = Feedback("completed", "native", {"tcp": {"left": {"position_error_m": error}}})
                before = asdict(raw)
                result = annotate_arrival(raw)
                self.assertEqual(result.status, "partial_execution")
                self.assertEqual(result.metrics["tcp"]["left"]["position_error_m"], error)
                self.assertFalse(result.metrics["arrival_feedback"]["episode_stop_on_residual"])
                self.assertEqual(asdict(raw), before)

    def test_retained_working_arm_is_not_lost_on_observer_move(self):
        raw = Feedback("completed", "native", {"tcp": {
            "left": {"position_error_m": .000622492785},
            "right": {"position_error_m": .000072431736},
        }})
        action = Action((ArmCommand("right", "cartesian_pose", (0, 0, 1, 1, 0, 0, 0)),), .1)
        rows = annotate_arrival(raw, action).metrics["arrival_feedback"]["arms"]
        self.assertFalse(rows["left"]["newly_commanded_arm"])
        self.assertTrue(rows["right"]["newly_commanded_arm"])
        self.assertFalse(rows["left"]["within_reporting_tolerance"])
        self.assertTrue(rows["right"]["within_reporting_tolerance"])

    def test_native_errors_not_masked_and_no_three_partial_stop(self):
        for status in ["action_timeout", "collision_blocked", "error", "rejected"]:
            raw = Feedback(status, "native", {})
            self.assertIs(annotate_arrival(raw), raw)
        raw = Feedback("completed", "native", {"tcp": {"left": {"position_error_m": .001}}})
        for _ in range(5):
            self.assertEqual(annotate_arrival(raw).status, "partial_execution")

    def test_invalid_measurements_fail_instead_of_claiming_arrival(self):
        for value in [float("nan"), float("inf"), -.001]:
            with self.assertRaises(ValueError):
                annotate_arrival(Feedback("completed", "native", {"tcp": {"left": {"position_error_m": value}}}))


class ProfileTests(unittest.TestCase):
    def test_opt_in_profile_preserves_prompt_scoring_and_native_tolerance(self):
        spec = importlib.util.spec_from_file_location(
            "control_profile_run", ROOT / "tools/control_revalidation/run.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        from robot_tasks.registry import resolve_task
        _, original = resolve_task("L3-01")
        before = original.source.read_bytes()
        report = module.inspect_profile("L3-01")
        self.assertTrue(report["robot_gravity_compensation"])
        self.assertTrue(report["task_control_schema_unchanged"])
        self.assertEqual(report["native_completion_tolerance_m"], .005)
        self.assertFalse(report["stop_on_arrival_error"])
        self.assertEqual(report["model_calls"], 0)
        self.assertEqual(original.source.read_bytes(), before)

    def test_native_mirror_has_only_reviewed_precision_and_residual_changes(self):
        native = (ROOT / "src/robot_tasks/planning/worker.py").read_text()
        expected = native.replace("position_tolerance=0.001,", "position_tolerance=0.0001,")
        expected = expected.replace("error[0] > 0.002", "error[0] > 0.00025")
        expected = expected.replace('"waypoints": len(path.positions),',
            '"waypoints": len(path.positions),\n'
            '                "requested_endpoint_position_error_m": error[0] if command["kind"] == "cartesian_pose" else None,\n'
            '                "requested_endpoint_orientation_error_rad": error[1] if command["kind"] == "cartesian_pose" else None,')
        self.assertNotEqual(native, expected)
        mirror = (ROOT / "tools/control_revalidation/native_precision.py").read_text()
        self.assertEqual(ast.dump(ast.parse(expected)), ast.dump(ast.parse(mirror)))
        sha = hashlib.sha256(native.encode()).hexdigest()
        self.assertIn(sha, (ROOT / "tools/control_revalidation/planner_worker.py").read_text())

    def test_shim_rejects_arbitrary_python_commands(self):
        with tempfile.NamedTemporaryFile() as runtime:
            argv = ["-u", "-m", "robot_tasks.planning.worker", "--profile", "robot.toml"]
            result = planner_python.command(argv, runtime.name)
            self.assertEqual(result[:2], [runtime.name, "-u"])
            self.assertEqual(result[-2:], ["--profile", "robot.toml"])
            with self.assertRaises(ValueError):
                planner_python.command(["-c", "print('unexpected')"], runtime.name)
        with self.assertRaises(ValueError):
            planner_python.command(argv, None)


if __name__ == "__main__":
    unittest.main()
