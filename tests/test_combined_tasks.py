"""Original/v3 isolation and shared-core regressions, without model calls."""
from copy import deepcopy
from dataclasses import replace
import tempfile
from pathlib import Path
import unittest

import numpy as np

from robot_tasks.backends.settings import SimulationSettings
from robot_tasks.batch import freeze_config
from robot_tasks.config import ExecutionSettings
from robot_tasks.initial_layout import scene_record
from robot_tasks.paths import PROJECT_ROOT
from robot_tasks.registry import TASKS, resolve_task
from robot_tasks.tasks.bench.arm_selection import resolve_bench_spec
from robot_tasks.tasks.bench.evaluation import BenchEvaluator
from robot_tasks.tasks.bench.stacked_books import score
from robot_tasks.tasks.l1.specs import local_corners, pose
from robot_tasks.robots.transforms import pose_transform
from robot_tasks.video_delivery import effective_cameras, scenes
from test_household_tasks import measurement

L1 = ("L1-01", "L1-02", "L1-03", "L1-04", "L1-05")
L2 = ("L2-01", "L2-02", "L2-03", "L2-04", "L2-05", "L2-06")
L3 = ("L3-01", "L3-02", "L3-03", "L3-04")
VERSIONED_SCENES = ("L2-07",) + L3


class CombinedTaskTests(unittest.TestCase):
    def test_all_originals_and_variants_are_independent_registered_tasks(self):
        self.assertEqual({s.base_task_id for s in TASKS.values() if s.base_task_id}, set(L1 + L2 + VERSIONED_SCENES))
        for tid in L1 + L2 + VERSIONED_SCENES:
            original, a = resolve_task(tid)
            variant, b = resolve_task(tid + "_v3")
            self.assertEqual(original.factory, variant.factory)
            self.assertEqual(b.task_id, a.task_id)
            self.assertNotEqual(b.experiment_id, a.experiment_id)
            self.assertNotEqual(a.source, b.source)
            self.assertNotEqual(scene_record(a.raw["task"]), scene_record(b.raw["task"]))
            self.assertEqual(a.robot.source, b.robot.source)

    def test_l1_canonical_entries_preserve_frozen_selected_layout_and_rules(self):
        selected = {"L1-01":3, "L1-02":3, "L1-03":2, "L1-04":1, "L1-05":1}
        for tid, number in selected.items():
            _, a = resolve_task(tid)
            _, b = resolve_task(tid + "_v3")
            self.assertEqual(a.raw["task"]["initial_placement_profile"], "baseline")
            self.assertEqual(b.raw["task"]["initial_placement_variant"], number)
            self.assertEqual(a.instruction, b.instruction)
            self.assertEqual(effective_cameras(a.scene_id), effective_cameras(b.scene_id))
            for cfg in (a,b):
                settings = SimulationSettings.from_config(cfg)
                self.assertFalse(settings.robot_gravity_compensation)
                self.assertEqual(settings.cartesian_completion_tolerance_m, .005)
            spec_a = resolve_bench_spec(a.raw["task"],a.robot)
            spec_b = resolve_bench_spec(b.raw["task"],b.robot)
            self.assertEqual(spec_a.design.instruction, spec_b.design.instruction)
            self.assertEqual(spec_a.mechanism, spec_b.mechanism)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / (tid + "_v3.toml")
                freeze_config(tid + "_v3", path, PROJECT_ROOT / "configs/models/gpt.toml",
                              PROJECT_ROOT / "configs/workflows/gpt_default.toml",
                              {"reasoning_effort":"medium", "seed":0, "wall_time_s":18000, "max_calls":150})
                _, frozen = resolve_task(tid + "_v3", path)
                self.assertEqual(scene_record(frozen.raw["task"]), scene_record(b.raw["task"]))

    def test_versioned_scene_selection_and_shared_camera_are_not_overwritten(self):
        for tid in VERSIONED_SCENES:
            _, a = resolve_task(tid)
            _, b = resolve_task(tid + "_v3")
            self.assertEqual(a.scene_id, tid)
            self.assertEqual(b.scene_id, tid + "_v3")
            self.assertNotEqual(scenes()[a.scene_id]["runtime_sha256"],
                                scenes()[b.scene_id]["runtime_sha256"])
            self.assertEqual(effective_cameras(a.scene_id), effective_cameras(b.scene_id))

    def test_renderer_and_gpu_selection_coexist(self):
        _, cfg = resolve_task("L2-06_v3")
        raw = deepcopy(cfg.raw)
        raw["simulation"].update(active_gpu=7, physics_gpu=7)
        settings = SimulationSettings.from_config(replace(cfg, raw=raw))
        self.assertEqual(settings.renderer, "PathTracing")
        self.assertEqual(settings.application_config()["active_gpu"], 7)
        self.assertEqual(settings.application_config()["physics_gpu"], 7)

    def test_controller_and_budget_profiles_are_explicit(self):
        for tid in L2 + VERSIONED_SCENES:
            for suffix in ("", "_v3"):
                _, cfg = resolve_task(tid + suffix)
                sim = SimulationSettings.from_config(cfg)
                execution = ExecutionSettings.from_config(cfg)
                self.assertEqual(sim.robot_gravity_compensation, tid in L2)
                self.assertEqual(sim.cartesian_completion_tolerance_m,
                                 .001 if tid in L2 else .005)
                self.assertEqual(execution.model_budget_basis,
                                 "all_requests" if tid in L2 else "completed_generations")

    def test_l2_pouring_retains_empty_source_and_half_received(self):
        for tid in ("L2-03", "L2-03_v3"):
            _, cfg = resolve_task(tid)
            spec = resolve_bench_spec(cfg.raw["task"], cfg.robot)
            self.assertEqual(spec.parameters["count"], 12)
            self.assertEqual(spec.parameters["required_received"], 6)
            self.assertEqual(spec.parameters["pouring_success_rule"], "empty_source_half_received_v1")

    def test_wiping_reclassification_preserves_its_scoring_protocol(self):
        for tid in ("L2-07", "L2-07_variant01"):
            _, cfg = resolve_task(tid)
            spec = resolve_bench_spec(cfg.raw["task"], cfg.robot)
            evaluator = BenchEvaluator(spec, {a.name: a.initial for a in spec.design.articles})
            result = evaluator.update(0., measurement(spec))
            self.assertEqual(result.status, "running")
            self.assertEqual(result.metrics["scoring_protocol"], "feishu_english_20261015_v1")

    def test_stacked_books_original_and_variant_share_paper_goal(self):
        original_prompt = resolve_task("L3-03")[1].instruction
        for tid in ("L3-03", "L3-03_v3", "L3-03_variant01"):
            _, cfg = resolve_task(tid)
            spec = resolve_bench_spec(cfg.raw["task"], cfg.robot)
            self.assertEqual(spec.parameters["boundary_reference"], "both_books")
            self.assertEqual(cfg.instruction, original_prompt)
            rows = measurement(spec)
            for name, row in rows.items():
                row["finger_force_vectors"] = {a: [[0., 0., 0.], [0., 0., 0.]] for a in spec.arms}
            rows["lower_book"]["bench_force_n"] = .4
            rows["upper_book"]["pair_forces"]["lower_book"] = .2
            rows["lower_book"]["pair_forces"]["upper_book"] = .2
            evaluator = BenchEvaluator(spec, {a.name: a.initial for a in spec.design.articles})
            evaluator.last_time = 0.
            done, details, reason = score(evaluator, rows)
            self.assertFalse(done)
            self.assertFalse(reason)
            self.assertTrue(details["started_in_start_rectangle"])
            self.assertTrue(details["upper_book_containment_required"])

    def test_stacked_book_transfer_requires_full_turn_and_both_footprints(self):
        # Synthetic contact/pose streams exercise scoring, not robot feasibility.
        for tid in ("L3-03", "L3-03_v3", "L3-03_variant01"):
            _, cfg = resolve_task(tid)
            spec = resolve_bench_spec(cfg.raw["task"], cfg.robot)
            articles = spec.design.articles
            start = np.asarray(spec.parameters["start_centre_xy"])
            target = np.asarray(articles[0].goal[:2])
            initial = pose_transform(articles[0].initial)
            yaw = np.arctan2(initial[1, 0], initial[0, 0])
            for turns, overhang, expected in ((1., 0., True), (1., .04, False),
                                               (.5, 0., False)):
                with self.subTest(task=tid, turns=turns, overhang=overhang):
                    rows = measurement(spec)
                    evaluator = BenchEvaluator(spec, {a.name: a.initial for a in articles})
                    for step in range(73):
                        fraction = step / 72
                        xy = start + fraction * (target - start)
                        for article in articles:
                            row = rows[article.name]
                            offset = overhang if step == 72 and article.name == "upper_book" else 0.
                            row["pose"] = list(pose(xy[0] + offset, xy[1], article.initial[2],
                                                    yaw=yaw + 2*np.pi*turns*fraction))
                            transform = pose_transform(row["pose"])
                            row["corners"] = (local_corners(article) @ transform[:3, :3].T
                                              + transform[:3, 3]).tolist()
                            row["finger_force_vectors"] = {
                                arm: [[.1, 0., 0.], [0., 0., 0.]] for arm in spec.arms}
                            row["fingers"] = {arm: [.1, 0.] for arm in spec.arms}
                        rows["lower_book"]["bench_force_n"] = .4
                        rows["upper_book"]["pair_forces"]["lower_book"] = .2
                        rows["lower_book"]["pair_forces"]["upper_book"] = .2
                        evaluator.last_time = step / 12
                        done, details, reason = score(evaluator, rows)
                    self.assertFalse(reason)
                    self.assertTrue(details["upper_book_supported_by_lower"])
                    self.assertTrue(details["simultaneous_push_witness"])
                    self.assertEqual(details["full_turn_observed"], turns == 1.)
                    self.assertEqual(done, expected)
                    if overhang:
                        self.assertTrue(details["lower_book_in_target"])
                        self.assertEqual(details["books_in_target"], 1)

    def test_sweeping_requires_all_three_blocks(self):
        for tid in ("L3-04", "L3-04_v3"):
            _, cfg = resolve_task(tid)
            spec = resolve_bench_spec(cfg.raw["task"], cfg.robot)
            self.assertEqual(spec.parameters["required"], 3)
            self.assertEqual(sum(a.name.startswith("block_") for a in spec.design.articles), 3)
            self.assertFalse(spec.parameters["require_brush_release"])

    def test_frozen_versioned_scenes_keep_their_identity_and_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            for tid in VERSIONED_SCENES:
                variant = tid + "_v3"
                path = Path(tmp) / (variant + ".toml")
                freeze_config(variant, path, PROJECT_ROOT / "configs/models/gpt.toml",
                              PROJECT_ROOT / "configs/workflows/gpt_default.toml",
                              {"reasoning_effort": "xhigh", "seed": 0,
                               "wall_time_s": 18000, "max_calls": 150})
                _, cfg = resolve_task(variant, path)
                self.assertEqual(cfg.scene_id, variant)
                self.assertEqual(cfg.experiment_id, variant)
                self.assertEqual(ExecutionSettings.from_config(cfg).max_decisions, 150)
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    resolve_task(tid, path)


if __name__ == "__main__":
    unittest.main()
