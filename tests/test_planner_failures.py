"""Offline failure attribution; no simulator, CUDA or model service required."""

import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from robot_tasks.planning.failures import diagnose_failure, state_violations


def metrics(**values):
    return SimpleNamespace(costs_and_constraints=SimpleNamespace(
        constraints=SimpleNamespace(
            names=list(values), values=[np.array([[[v]]]) for v in values.values()]
        )
    ))


class PlannerFailureTests(unittest.TestCase):
    def setUp(self):
        self.planner = Mock()
        self.rollout = self.planner.graph_planner.auxiliary_rollout
        self.rollout.compute_metrics_from_action.return_value = metrics(
            cspace=0, self_collision=0, scene_collision=0
        )
        self.start = Mock()
        self.goal = Mock()
        self.planner.ik_solver.solve_pose.return_value = SimpleNamespace(
            success=np.array([True])
        )

    def diagnose(self, kind="cartesian_pose", result=None, **kwargs):
        return diagnose_failure(self.planner, "left", kind, self.start,
                                self.goal, result, **kwargs)

    def test_invalid_start_takes_precedence_over_unreachable_goal(self):
        self.rollout.compute_metrics_from_action.return_value = metrics(
            self_collision=1.0000024, scene_collision=0
        )
        error = self.diagnose(goal_distance_m=10., reach_bound_m=.7)
        self.assertEqual(error.diagnostic["code"], "start_collision")
        self.assertEqual(error.diagnostic["violated_constraints"], ["self_collision"])
        self.assertIn("not proof of physical mesh contact", str(error))
        self.assertIn("No commands", str(error))
        self.planner.ik_solver.solve_pose.assert_not_called()
        # Solver constraint costs are not reported as physical penetration.
        self.assertNotIn("penetration_m", json.dumps(error.diagnostic))

    def test_environment_collision_is_not_reported_as_self_collision(self):
        self.rollout.compute_metrics_from_action.return_value = metrics(scene_collision=1)
        error = self.diagnose()
        self.assertEqual(error.diagnostic["code"], "start_collision")
        self.assertEqual(error.diagnostic["violated_constraints"], ["scene_collision"])
        self.assertNotIn("reported self-collision", str(error))

    def test_joint_limit_failure_is_not_reported_as_collision(self):
        self.rollout.compute_metrics_from_action.return_value = metrics(cspace=1)
        self.assertEqual(self.diagnose().diagnostic["code"], "start_state_invalid")

    def test_joint_goal_collision_is_distinct_from_start_collision(self):
        self.rollout.compute_metrics_from_action.side_effect = [
            metrics(self_collision=0), metrics(scene_collision=1)
        ]
        error = self.diagnose(kind="joint_position")
        self.assertEqual(error.diagnostic["code"], "goal_collision")
        self.assertEqual(error.diagnostic["stage"], "goal")
        self.assertIn("current start state passed", str(error))
        self.planner.ik_solver.solve_pose.assert_not_called()

    def test_position_beyond_conservative_reach_has_geometric_evidence(self):
        error = self.diagnose(goal_distance_m=5., reach_bound_m=.7)
        self.assertEqual(error.diagnostic["code"], "goal_out_of_reach")
        self.assertEqual(error.diagnostic["goal_distance_m"], 5.)
        self.planner.ik_solver.solve_pose.assert_not_called()

    def test_none_plan_does_not_automatically_mean_failed_ik(self):
        error = self.diagnose()
        self.assertEqual(error.diagnostic["code"], "path_not_found")
        self.planner.ik_solver.solve_pose.assert_called_once_with(
            self.goal, return_seeds=1, current_state=self.start
        )

    def test_ik_failure_reports_finite_search_not_proven_unreachability(self):
        self.planner.ik_solver.solve_pose.return_value.success = np.array([False])
        error = self.diagnose()
        self.assertEqual(error.diagnostic["code"], "goal_ik_not_found")
        self.assertIn("does not prove global unreachability", str(error))

    def test_failed_trajectory_result_does_not_rerun_ik(self):
        error = self.diagnose(result=SimpleNamespace(success=np.array([False])))
        self.assertEqual(error.diagnostic["code"], "path_not_found")
        self.planner.ik_solver.solve_pose.assert_not_called()

    def test_broken_or_nonfinite_diagnostic_does_not_invent_a_cause(self):
        for value in [float("nan"), float("inf")]:
            with self.subTest(value=value):
                self.rollout.compute_metrics_from_action.return_value = metrics(self_collision=value)
                error = self.diagnose()
                self.assertEqual(error.diagnostic["code"], "planning_failed_unknown")
                json.dumps(error.diagnostic, allow_nan=False)
        self.rollout.compute_metrics_from_action.side_effect = RuntimeError("diagnostic unavailable")
        error = self.diagnose()
        self.assertEqual(error.diagnostic["code"], "planning_failed_unknown")
        self.assertIn("RuntimeError", error.diagnostic["diagnostic_error"])

    def test_state_check_only_consumes_feasibility_constraints(self):
        result = metrics(self_collision=0, scene_collision=0)
        result.costs_and_constraints.costs = metrics(self_collision=10).costs_and_constraints.constraints
        self.rollout.compute_metrics_from_action.return_value = result
        self.assertEqual(state_violations(self.planner, self.start), [])
