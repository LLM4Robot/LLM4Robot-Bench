"""English source goals and physical counterexamples, without model calls/GPU."""

import unittest
import json
from dataclasses import asdict
import numpy as np

from robot_tasks.registry import resolve_task
from robot_tasks.robots.transforms import pose_transform, transform_pose
from robot_tasks.tasks.bench.arm_selection import resolve_bench_spec
from robot_tasks.tasks.bench.english_reference import apply_reference, reference_tasks, initial_state_issue
from robot_tasks.tasks.bench.evaluation import BenchEvaluator
from robot_tasks.tasks.bench.scoring_l1 import marker, plug, cups, flower
from robot_tasks.tasks.bench.scoring_l3 import bottle
from robot_tasks.tasks.bench.wiping import score as wiping
from robot_tasks.tasks.l1.specs import local_corners, pose
from robot_tasks.tasks.screw.nut import nut_spec, NutEvaluator
from test_household_tasks import measurement
from test_nut_task import sample


def scene(task):
    _, config = resolve_task(task)
    task = {k:v for k,v in config.raw['task'].items() if k != 'scene_delivery'}
    spec = resolve_bench_spec(task, config.robot)
    rows = measurement(spec)
    evaluator = BenchEvaluator(spec, {a.name: a.initial for a in spec.design.articles})
    return spec, rows, evaluator


def place(spec, rows, name, at):
    article = next(a for a in spec.design.articles if a.name == name)
    t = pose_transform(at)
    rows[name]['pose'] = list(at)
    rows[name]['corners'] = (local_corners(article) @ t[:3, :3].T + t[:3, 3]).tolist()


class EnglishReferenceTests(unittest.TestCase):
    def test_all_source_instructions_reach_policy_without_old_task_guidance(self):
        for tid, source in reference_tasks().items():
            with self.subTest(task=tid):
                entry, config = resolve_task(tid)
                prepared = entry.factory.prepare_config(config)
                self.assertEqual(prepared.instruction.strip(), source['simulation_prompt_en'])
                self.assertEqual(prepared.arms, tuple(source['active_arms']))
                self.assertEqual(prepared.observation_arms, tuple(source['observation_arms']))
                self.assertFalse(config.raw.get('completion', {}).get('return_home', False))
                self.assertNotIn('Declared prop facts and guidance', prepared.instruction)
                self.assertFalse(any('\u4e00' <= c <= '\u9fff' for c in prepared.instruction))
        text = reference_tasks()['L1-04']['simulation_prompt_en']
        self.assertIn('left arm to place the left cup', text)
        self.assertIn('right arm to nest the right cup', text)

    def test_marker_entry_while_held_does_not_satisfy_release_goal(self):
        spec, rows, e = scene('L1-01')
        a = spec.design.articles[0]
        place(spec, rows, 'marker', a.goal)
        r = rows['marker']; r['bench_force_n'] = 0.; r['fixture_force_n'] = .22
        e.grasped.add('marker'); r['fingers']['right'] = [1., 1.]
        self.assertFalse(marker(e, rows)[0])
        r['fingers']['right'] = [0., 0.]
        self.assertTrue(marker(e, rows)[0])
        r['fixture_force_n'] = 0.
        self.assertFalse(marker(e, rows)[0])

    def test_prongs_can_align_without_entering_but_a_lateral_miss_fails(self):
        spec, rows, e = scene('L1-02')
        fixture = pose_transform(spec.design.fixture_pose)
        local = pose(0., 0., spec.parameters['rim'] - fixture[2, 3] + .01)
        place(spec, rows, 'plug', transform_pose(fixture @ pose_transform(local)))
        rows['plug']['fingers']['right'] = [1., 1.]; e.grasped.add('plug')
        done, criteria, _ = plug(e, rows)
        self.assertTrue(done); self.assertFalse(criteria['insertion_required'])
        local = pose(.003, 0., local[2])
        place(spec, rows, 'plug', transform_pose(fixture @ pose_transform(local)))
        self.assertFalse(plug(e, rows)[0])

    def test_cup_pair_needs_initial_centre_placement_and_assigned_hands(self):
        spec, rows, e = scene('L1-04')
        for a in spec.design.articles: place(spec, rows, a.name, a.goal)
        base, top = rows['lower_cup'], rows['upper_cup']
        base['bench_force_n'] = .275; top['pair_forces']['lower_cup'] = .14
        self.assertFalse(cups(e, rows)[0])
        base['fingers']['right'] = [1., 1.]; cups(e, rows)
        base['fingers']['right'] = [0., 0.]; cups(e, rows)
        top['fingers']['left'] = [1., 1.]; cups(e, rows)
        top['fingers']['left'] = [0., 0.]
        self.assertTrue(cups(e, rows)[0])

    def test_flower_starts_in_a_separate_source_cup(self):
        spec, rows, e = scene('L1-05')
        self.assertIn('source_cup', rows)
        self.assertEqual(spec.parameters['fixed_articles'], ('source_cup',))
        self.assertIsNone(initial_state_issue(spec, rows))
        original = rows['source_cup']['pose']
        rows['source_cup']['pose'] = list(pose(*original[:3], roll=np.pi/2))
        self.assertIsNotNone(initial_state_issue(spec, rows))
        rows['source_cup']['pose'] = original
        from types import SimpleNamespace
        from robot_tasks.tasks.bench.scene import BenchScene
        backend_scene = BenchScene.__new__(BenchScene)
        backend_scene.spec, backend_scene.design = spec, spec.design
        backend_scene.config = SimpleNamespace(arms=('right',))
        backend_scene.measure = lambda: rows
        rows['source_cup']['fingers']['right'] = [1., 1.]
        self.assertEqual(backend_scene.planning_payloads(), {'right': []})
        self.assertLess(rows['source_cup']['pose'][0], spec.parameters['vase_xy'][0])
        self.assertFalse(flower(e, rows)[0])
        place(spec, rows, 'rose', spec.design.articles[0].goal)
        rows['rose']['fixture_force_n'] = .08; rows['rose']['bench_force_n'] = 0.
        self.assertTrue(flower(e, rows)[0])

    def test_fixed_bottle_needs_held_cap_and_three_to_five_cm_clearance(self):
        spec, rows, e = scene('L3-01')
        self.assertEqual(spec.mechanism, 'native_fixed_bottle_cap_contact')
        self.assertIn('Body', spec.parameters['native_instances'][0]['static'])
        self.assertTrue(spec.parameters['body_fixed'])
        body_z = rows['bottle']['pose'][2]
        for gap, expected in ((.006, False), (.04, True), (.06, False)):
            with self.subTest(gap=gap):
                at = pose(rows['cap']['pose'][0], rows['cap']['pose'][1],
                          body_z + spec.parameters['neck_top_z'] + gap - spec.parameters['cap_bottom_z'])
                place(spec, rows, 'cap', at)
                rows['bottle']['fingers']['right'] = [0., 0.]
                rows['cap']['fingers']['left'] = [1., 1.]
                self.assertEqual(bool(bottle(e, rows)[0]), expected)
        at = pose(rows['cap']['pose'][0], rows['cap']['pose'][1],
                  body_z + spec.parameters['neck_top_z'] + .04 - spec.parameters['cap_bottom_z'])
        place(spec, rows, 'cap', at)
        self.assertTrue(bottle(e, rows)[0])
        rows['cap']['fingers']['left'] = [0., 0.]
        self.assertFalse(bottle(e, rows)[0])

    def test_wiping_one_mark_is_not_erasing_the_writing(self):
        spec, rows, e = scene('L2-07')
        self.assertTrue(spec.parameters['whiteboard_surface'])
        m = rows['eraser']['wiping']; m['remaining_cells'] = m['initial_cells'] - 1
        self.assertFalse(wiping(e, rows)[0])
        m['remaining_cells'] = 0
        self.assertTrue(wiping(e, rows)[0])

    def test_pouring_layout_count_and_unheld_receiver_are_consistent(self):
        spec, rows, _ = scene('L2-03')
        self.assertLess(rows['source_cup']['pose'][0], rows['receiver_cup']['pose'][0])
        self.assertEqual(spec.parameters['count'], 10)
        self.assertEqual(spec.parameters['required_received'], 10)
        self.assertTrue(spec.parameters['require_source_empty'])
        self.assertEqual(sum(a.name.startswith('load_') for a in spec.design.articles), 10)

    def test_support_hand_can_hold_top_book_without_box_contact(self):
        from robot_tasks.tasks.bench.scoring_l2 import books
        spec, rows, e = scene('L2-02')
        p = spec.parameters
        middle = rows['middle_book']; axis = np.asarray(p['outward_axis_world'])
        middle['pose'][:3] = (np.asarray(middle['pose'][:3]) + .05 * axis).tolist()
        middle['fingers'][p['manipulation_arm']] = [1., 1.]
        middle['fixture_force_n'] = .5
        rows['lower_book']['fixture_force_n'] = 1.
        rows['upper_book']['fingers'][p['support_arm']] = [1., 1.]
        done, criteria, _ = books(e, rows)
        self.assertTrue(done)
        self.assertTrue(criteria['stages']['supported_pull_complete'])
        self.assertFalse(criteria['stages']['released_supported'])
        result = e.update(.1, rows)
        self.assertEqual(result.status, 'succeeded')
        json.dumps(asdict(result), allow_nan=False)
        rows['upper_book']['fingers'][p['support_arm']] = [0., 0.]
        self.assertFalse(books(e, rows)[0])

    def test_nut_detachment_finishes_while_parts_remain_held(self):
        _, config = resolve_task('L3-02')
        spec = apply_reference(nut_spec(), {k:v for k,v in config.raw['task'].items() if k != 'scene_delivery'}); e = NutEvaluator(spec)
        self.assertEqual(e.update(.1, sample()).status, 'running')
        result = e.update(.2, sample(turn=.2, rise=.008, contact=0.))
        self.assertEqual(result.status, 'succeeded')
        self.assertFalse(result.metrics['criteria']['release_required'])
        self.assertFalse(result.metrics['criteria']['parts_released_supported'])

    def test_push_target_uses_one_rectangle_for_geometry_and_outline(self):
        spec, _, _ = scene('L3-03')
        self.assertEqual(len(spec.parameters['target_vertices']), 4)
        self.assertEqual(len(spec.design.fixtures), 4)
        self.assertTrue(all(not p.collision for p in spec.design.fixtures))


if __name__ == '__main__':
    unittest.main()
