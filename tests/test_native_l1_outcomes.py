"""Boundary regressions against locked collider geometry; no model requests."""

import hashlib
from dataclasses import replace
import unittest
from pathlib import Path
import numpy as np

from robot_tasks.planning.scene import article_surface
from robot_tasks.robots.transforms import pose_transform, transform_pose
from robot_tasks.tasks.bench.evaluation import BenchEvaluator
from robot_tasks.tasks.bench.native_geometry import catalog, cavity_fit, cup_cavity, shapes, transformed_poly
from robot_tasks.tasks.bench.scoring_l1 import cups, flower, hat, plug, marker
from robot_tasks.tasks.bench.spec import build_spec
from robot_tasks.tasks.l1.specs import local_corners, pose
from test_household_tasks import measurement


def goal(task):
    spec=build_spec(task); rows=measurement(spec)
    for a in spec.design.articles:
        r=rows[a.name]; r['pose']=list(a.goal)
        t=pose_transform(a.goal); r['corners']=(local_corners(a)@t[:3,:3].T+t[:3,3]).tolist()
        r['bench_force_n']=0.;r['fixture_force_n']=a.mass*9.81
    return spec,rows,BenchEvaluator(spec,{a.name:a.initial for a in spec.design.articles})


class NativeOutcomeTests(unittest.TestCase):
    def test_barrel_fits_near_floor_wall_without_using_cap_radius(self):
        # Preserve the known paper-cup regression independently of L1-01's
        # new ceramic receptacle: only the barrel reaches this narrow floor.
        planes, floor, _ = cup_cavity()
        t = pose_transform(pose(.0177, 0, floor))
        self.assertTrue(cavity_fit('whiteboard_marker', t, .001)[0])
        self.assertGreater(.0177 + .00898, .025 * np.cos(np.pi/48) + .001)
        t[0,3] += .003
        self.assertFalse(cavity_fit('whiteboard_marker', t, .001)[0])

    def test_cap_and_barrel_are_distinct_collision_shapes(self):
        parts=shapes('whiteboard_marker')
        self.assertEqual([(s['kind'],s['radius']) for s in parts],[('cylinder',.0081),('capsule',.0087)])

    def test_flower_slow_drift_cannot_accumulate_one_second_stability(self):
        spec,rows,e=goal('L1-05');r=rows['rose']
        self.assertEqual(e.update(.1,rows).status,'running')
        r['pose'][0]+=.0008
        self.assertEqual(e.update(.6,rows).status,'running')
        r['pose'][0]+=.0008
        self.assertEqual(e.update(1.2,rows).status,'running')
        self.assertEqual(e.update(1.3,rows).status,'running')
        self.assertEqual(e.update(2.4,rows).status,'succeeded')

    def test_flower_losing_support_resets_stable_interval(self):
        spec,rows,e=goal('L1-05');r=rows['rose']
        e.update(.1,rows);r['fixture_force_n']=0
        self.assertEqual(e.update(.9,rows).status,'running')
        r['fixture_force_n']=.22
        self.assertEqual(e.update(1.,rows).status,'running')
        self.assertEqual(e.update(1.9,rows).status,'running')
        self.assertEqual(e.update(2.1,rows).status,'succeeded')

    def test_shallow_held_moving_charger_succeeds_without_seating_or_wait(self):
        spec,rows,e=goal('L1-02');r=rows['plug']
        r['pose'][2]=spec.parameters['rim']-.001
        r['fingers']['right']=[1.,1.]
        r['fixture_force_n']=0.
        r['linear_velocity']=[.01,0.,-.03]
        r['angular_velocity']=[0.,0.,.5]
        done,c,_=plug(e,rows)
        self.assertTrue(c['inside_socket']);self.assertFalse(c['housing_seated']);self.assertTrue(done)
        self.assertFalse(c['release_required']);self.assertFalse(c['released'])
        self.assertFalse(c['fixture_supported']);self.assertFalse(c['grasp_lift_recorded'])
        self.assertEqual(e.update(.1,rows).status,'succeeded')
        self.assertEqual(spec.parameters['hold_s'],0.)

    def test_charger_alignment_contact_and_numerical_noise_are_not_entry(self):
        for depth in (-.002,0.,.0001):
            with self.subTest(depth=depth):
                spec,rows,e=goal('L1-02')
                rows['plug']['pose'][2]=spec.parameters['rim']-depth
                self.assertFalse(plug(e,rows)[0])
                self.assertEqual(e.update(.1,rows).status,'running')

    def test_released_shallow_charger_can_succeed_without_support(self):
        spec,rows,e=goal('L1-02')
        rows['plug']['pose'][2]=spec.parameters['rim']-.0005
        rows['plug']['fixture_force_n']=0.
        self.assertEqual(e.update(.1,rows).status,'succeeded')

    def test_only_one_blade_entering_is_incomplete(self):
        spec,rows,e=goal('L1-02')
        local=pose_transform(pose(0,0,.03015-.0002,pitch=.03))
        rows['plug']['pose']=list(transform_pose(pose_transform(spec.design.fixture_pose) @ local))
        done,criteria,_=plug(e,rows)
        self.assertEqual(sum(c['inside_slot'] for c in criteria['blades']),1)
        self.assertFalse(done)

    def test_entry_cannot_ignore_floor_or_housing_penetration(self):
        spec,rows,e=goal('L1-02')
        rows['plug']['pose'][2]=spec.parameters['floor']-.001
        self.assertFalse(plug(e,rows)[0])
        local=pose_transform(pose(0,0,.03015+.017,pitch=np.pi))
        rows['plug']['pose']=list(transform_pose(pose_transform(spec.design.fixture_pose) @ local))
        done,criteria,_=plug(e,rows)
        self.assertFalse(criteria['housing_clear_of_strip'])
        self.assertFalse(done)

    def test_real_blade_chamfers_and_both_slots_are_checked(self):
        self.assertGreater(len(shapes('usb_charger')[1]['points']),8)
        spec,rows,e=goal('L1-02');rows['plug']['pose'][1]+=.0008
        _,c,_=plug(e,rows)
        self.assertFalse(c['inside_socket']);self.assertEqual(len(c['blades']),2)

    def test_charger_score_follows_fixture_translation_and_yaw(self):
        spec,rows,e=goal('L1-02')
        self.assertTrue(plug(e,rows)[0])
        for yaw in (0., .37, -1.2):
            shift=pose_transform(pose(.13,-.04,.08,yaw=yaw))
            fixture=shift @ pose_transform(spec.design.fixture_pose)
            moved=replace(spec, design=replace(spec.design,fixture_pose=transform_pose(fixture)),
                          parameters={**spec.parameters,'rim':spec.parameters['rim']+.08,
                                      'floor':spec.parameters['floor']+.08})
            rows['plug']['pose']=list(transform_pose(shift @ pose_transform(spec.design.articles[0].goal)))
            evaluator=BenchEvaluator(moved,e.initial)
            self.assertTrue(plug(evaluator,rows)[0])
            # A miss across the narrow slot dimension remains a miss at every yaw.
            rows['plug']['pose'][:3]=(np.asarray(rows['plug']['pose'][:3])+fixture[:3,0]*.002).tolist()
            self.assertFalse(plug(evaluator,rows)[0])

    def test_tilted_housing_corner_contact_does_not_count_as_seated(self):
        spec,rows,e=goal('L1-02');r=rows['plug']
        r['pose']=list(pose(*r['pose'][:3],pitch=.05))
        _,c,_=plug(e,rows)
        self.assertFalse(c['housing_seated'])

    def test_hat_is_native_and_planner_keeps_opening_unfilled(self):
        spec=build_spec('L1-03');a=spec.design.articles[0]
        self.assertEqual(a.asset,'tuojing:yellow_visor')
        self.assertEqual(len(shapes('yellow_visor')),208)
        samples=article_surface(a)
        near_plane=samples[np.abs(samples[:,2]-.035)<.002]
        self.assertGreater(len(near_plane),0)
        self.assertTrue(np.all(np.linalg.norm(near_plane[:,:2],axis=1)>.080))
        self.assertEqual(spec.parameters['hold_s'],0.)

    def test_hat_round_tip_balancing_and_wrong_support_are_rejected(self):
        spec,rows,e=goal('L1-03');r=rows['hat'];r['target_hook_vertical_force_n']=.64
        self.assertTrue(hat(e,rows)[0])
        r['target_hook_vertical_force_n']=0
        self.assertFalse(hat(e,rows)[0])
        r['target_hook_vertical_force_n']=.64
        # Move the plane to the peg's rounded endpoint: contact is insufficient.
        t=pose_transform(r['pose']);end=np.asarray(spec.parameters['hook_end'])
        t[:3,3]=end-t[:3,:3]@np.array([0,.088,.035]);r['pose']=list(transform_pose(t))
        self.assertFalse(hat(e,rows)[1]['hook_through_head_opening'])

    def test_native_cup_spin_can_pass_but_tumbling_cannot(self):
        spec,rows,e=goal('L1-04');top,base=rows['upper_cup'],rows['lower_cup']
        top['pair_forces']['lower_cup']=.14;base['bench_force_n']=.28
        top['pose']=list(base['pose']);top['pose'][2]+=.03
        top['angular_velocity']=[0,0,10];base['angular_velocity']=[0,0,-12]
        self.assertEqual(e.update(.1,rows).status,'running')
        self.assertEqual(e.update(1.2,rows).status,'succeeded')
        top['angular_velocity']=[1,0,0]
        self.assertEqual(e.update(1.3,rows).status,'running')

    def test_native_cup_tilt_and_wall_penetration_boundaries(self):
        spec,rows,e=goal('L1-04');top,base=rows['upper_cup'],rows['lower_cup']
        top['pair_forces']['lower_cup']=.14;base['bench_force_n']=.28
        top['pose']=list(transform_pose(pose_transform(base['pose'])@pose_transform(pose(.002,0,.05,pitch=.05))))
        self.assertTrue(cups(e,rows)[1]['nested'])
        top['pose'][0]+=.025
        self.assertFalse(cups(e,rows)[1]['nested'])
        top['pose']=list(base['pose']);top['pose'][2]+=.003
        self.assertFalse(cups(e,rows)[1]['nested'])

    def test_rose_segments_have_real_bends_and_head_cannot_enter_cup(self):
        stems=[s for s in shapes('artificial_rose') if s['name'].startswith('Stem')]
        self.assertEqual(len(stems),4)
        self.assertNotEqual(stems[1]['transform'][0][3],0.)
        spec,rows,e=goal('L1-05');r=rows['rose']
        self.assertTrue(flower(e,rows)[0]) # Final evidence does not depend on history samples.
        r['pose'][0]+=.06
        self.assertFalse(flower(e,rows)[1]['stem_end_inside'])
        r['pose']=list(pose(*spec.parameters['vase_xy'],spec.parameters['floor'],pitch=np.pi/2))
        self.assertFalse(flower(e,rows)[1]['head_above_rim'])

    def test_bent_rose_near_wall_is_not_rejected_as_a_straight_stem(self):
        spec,rows,e=goal('L1-05')
        rows['rose']['pose']=list(pose(spec.parameters['vase_xy'][0]+.0251,
                                      spec.parameters['vase_xy'][1],
                                      spec.parameters['floor']+.0012,pitch=-.3))
        done,criteria,_=flower(e,rows)
        self.assertTrue(done,criteria)
        # This pose was rejected by the former straight-centreline criterion.
        self.assertGreater(.0251+.00105,.025*np.cos(np.pi/48)+.001)

    def test_hat_pickup_pads_use_fixture_local_coordinates(self):
        from robot_tasks.robots.transforms import origin_transform

        spec=build_spec('L1-03');rack=pose_transform(spec.design.fixture_pose)
        positions=[]
        for pad in spec.design.fixtures:
            world=rack@origin_transform(pad.position,pad.rotation)
            self.assertAlmostEqual(world[2,3]+pad.size[2]/2,.825)
            np.testing.assert_allclose(world[:3,:3],np.eye(3),atol=1e-9)
            positions.append(world[:2,3])
        self.assertEqual(len(positions),4)
        np.testing.assert_allclose(np.mean(positions,axis=0),spec.design.articles[0].initial[:2])

    def test_cached_colliders_are_from_exact_locked_source_usds(self):
        import json
        root = Path(__file__).resolve().parents[1] / 'assets/tuojing'
        entries = {e['id']: e for e in json.loads((root / 'catalog.json').read_text(encoding='utf-8'))['entries']}
        packages = {p['id']: p for p in json.loads((root / 'source.lock.json').read_text(encoding='utf-8'))['packages']}
        for name, record in catalog().items():
            if name.startswith("video:"):
                continue  # Complete scene geometry has a separate hash-locked manifest test.
            entry = entries[name]
            package = packages[entry['package']]
            self.assertEqual(package['files'][entry['path']]['sha256'], record['sha256'])
            source = root / 'packages' / package['directory'] / entry['path']
            if source.exists():
                self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), record['sha256'])
        planes,floor,rim=cup_cavity()
        self.assertEqual(len(planes),48)
        self.assertAlmostEqual(floor,.004,places=8);self.assertAlmostEqual(rim,.1075,places=8)


if __name__=='__main__': unittest.main()
