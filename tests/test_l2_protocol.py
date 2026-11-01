"""Episode validity and observer-only roles; entirely offline, no model calls."""

import unittest
from unittest.mock import Mock

from test_native_l2_scoring import soft_row
from test_single_episode import SingleEpisodeTests, TerminalTask, TrialPolicy

from robot_tasks.tasks.bench.observer_contact import ObserverContactMonitor
from robot_tasks.tasks.bench.orange_evaluation import OrangeEvaluator
from robot_tasks.tasks.bench.spec import build_spec
from robot_tasks.tasks.bench.task import BenchTask
from robot_tasks.types import Assessment, EvaluationState, Feedback, InvalidSimulationState


class InvalidTask(TerminalTask):
    def __init__(self, *, early=False, exception=False):
        super().__init__()
        self.early, self.exception = early, exception

    def evaluate(self, state, feedback):
        if self.early or state.values["simulation_time_s"] > 0:
            if self.exception:
                raise InvalidSimulationState("Nonfinite measured state")
            return Assessment(
                "invalid", "Invalid FEM", {"invalid_reason": "simulation_invalid"}
            )
        return Assessment("running", "Initial state", {"stages": {"grasped": True}})


class ValidityTests(unittest.TestCase):
    setUp = SingleEpisodeTests.setUp
    run_case = SingleEpisodeTests.run_case

    def test_invalid_measurements_end_without_scoring_model_failure(self):
        for early in (False, True):
            for exception in (False, True):
                with self.subTest(early=early, exception=exception):
                    result = self.run_case(
                        InvalidTask(early=early, exception=exception), TrialPolicy()
                    )
                    self.assertEqual(result["status"], "invalid")
                    self.assertEqual(result["evaluation_validity"], "invalid")
                    self.assertIsNone(result["success"])
                    self.assertFalse(result["task_evaluated"])
                    self.assertEqual(result["model_calls"], 0 if early else 1)
                    if exception and not early:
                        self.assertTrue(
                            result["assessment"]["metrics"]["stages"]["grasped"]
                        )


class ObserverTests(unittest.TestCase):
    def test_repeated_task_frame_reuses_score_and_observer_checks(self):
        task = BenchTask()
        task.spec = build_spec("L2-04")
        from robot_tasks.registry import resolve_task
        from types import SimpleNamespace
        task.config = resolve_task("L2-04")[1]
        homes = {m.name: SimpleNamespace(joint_positions_rad=m.home_positions_rad) for m in task.config.robot.mounts}
        task.scene = Mock()
        task.scene.measure.return_value = soft_row(task.spec)
        task.scene.observer_evidence.return_value = {
            "left": {"mode": "soft_geometry_proximity", "distance_m": .25}
        }
        task.reset(Mock(), 0)
        state = EvaluationState({"simulation_time_s": .1, "robot": homes})
        first = task.evaluate(state, Feedback("completed"))
        for _ in range(3):
            self.assertIs(task.evaluate(state, Feedback("completed")), first)
        task.scene.measure.assert_called_once()
        task.scene.observer_evidence.assert_called_once()
        task.scene.measure.return_value = soft_row(task.spec, lifted=True, touching=True)
        task.evaluate(EvaluationState({"simulation_time_s": .2, "robot": homes}), Feedback("completed"))
        task.scene.measure.return_value = soft_row(task.spec)
        self.assertEqual(task.evaluate(EvaluationState({"simulation_time_s": .3, "robot": homes}),
                                       Feedback("completed")).status, "succeeded")
        task.reset(Mock(), 0)
        # A new episode cannot inherit the cached result or previous lift.
        self.assertEqual(task.evaluate(state, Feedback("completed")).status, "running")

    def test_l2_source_prompts_are_not_extended_with_legacy_task_notes(self):
        from robot_tasks.registry import resolve_task
        from robot_tasks.tasks.bench.l2_protocol import OPERATING_NOTES

        for task_id in ("L2-01", "L2-02", "L2-03", "L2-04", "L2-05"):
            spec, config = resolve_task(task_id)
            config = spec.factory.prepare_config(config)
            self.assertNotIn(OPERATING_NOTES, config.instruction)
            self.assertEqual(config.raw["task"]["prompt_protocol"], "feishu_english_20261015_v1")
            from robot_tasks.tasks.bench.english_reference import reference_tasks
            self.assertEqual(config.instruction.strip(), reference_tasks()[task_id]['simulation_prompt_en'])

    def test_transient_contact_does_not_fail_but_sustained_contact_does(self):
        monitor = ObserverContactMonitor()

        def update(t, force):
            return monitor.update(
                t,
                {"right": {"mode": "rigid_contact", "max_force_n": force}},
                Assessment("running", "grasping", {"stages": {"grasped": True}}),
            )

        self.assertEqual(update(0.1, 1).status, "running")
        self.assertEqual(update(0.2, 0).status, "running")
        self.assertEqual(update(0.3, 1).status, "running")
        result = update(0.6, 1)
        self.assertEqual(result.status, "failed")
        self.assertEqual(result.metrics["failure_category"], "observer_role_violation")
        self.assertTrue(result.metrics["stages"]["grasped"])

    def test_soft_bounds_nearness_requires_review_not_fake_contact_failure(self):
        monitor = ObserverContactMonitor()
        evidence = {"left": {"mode": "soft_geometry_proximity", "distance_m": 0.002}}
        monitor.update(0.1, evidence, Assessment("running", ""))
        result = monitor.update(0.4, evidence, Assessment("succeeded", ""))
        self.assertEqual(result.status, "invalid")
        self.assertEqual(
            result.metrics["invalid_reason"], "observer_contact_requires_review"
        )
        self.assertFalse(
            result.metrics["observer_contact"]["geometry_is_contact_proof"]
        )

    def test_numeric_invalid_takes_precedence_over_contact(self):
        monitor = ObserverContactMonitor()
        evidence = {"right": {"mode": "rigid_contact", "max_force_n": 1}}
        monitor.update(0.1, evidence, Assessment("running", ""))
        result = monitor.update(0.4, evidence, Assessment("invalid", "Nonfinite readback"))
        self.assertEqual(result.reason, "Nonfinite readback")

    def test_inverted_soft_element_is_recorded_and_allows_lift(self):
        spec = build_spec("L2-04")
        evaluator = OrangeEvaluator(spec)
        rows = soft_row(spec, strain=0.8)
        rows["orange"]["minimum_volume_ratio"] = -0.1
        result = evaluator.update(0.1, rows)
        self.assertEqual(result.status, "running")
        self.assertNotIn("invalid_reason", result.metrics)
        self.assertTrue(result.metrics["simulation_quality"]["mesh_inversion_observed"])
        self.assertTrue(result.metrics["quality_warnings"])
        evaluator.update(0.2, soft_row(spec, lifted=True, touching=True))
        # A recovered mesh still retains the diagnostic history.
        result = evaluator.update(0.3, soft_row(spec))
        self.assertEqual(result.status, "succeeded")
        self.assertTrue(result.metrics["quality_warnings"])
        self.assertEqual(result.metrics["simulation_quality"]["minimum_volume_ratio_seen"], -.1)
        # Nor does an inverted but finite sample veto a held lift.
        rows = soft_row(spec, lifted=True, touching=True)
        rows["orange"]["minimum_volume_ratio"] = -.2
        self.assertEqual(evaluator.update(.4, rows).status, "running")
        self.assertTrue(evaluator.lifted)


if __name__ == "__main__":
    unittest.main()
