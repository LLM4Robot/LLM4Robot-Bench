"""Current scene contracts and physical scoring counterexamples; no model calls."""

from dataclasses import replace
from pathlib import Path
import unittest
import numpy as np

from robot_tasks.paths import PROJECT_ROOT
from robot_tasks.registry import resolve_task
from robot_tasks.robots.transforms import pose_transform, transform_pose
from robot_tasks.tasks.bench.arm_selection import resolve_bench_spec
from robot_tasks.tasks.bench.evaluation import BenchEvaluator
from robot_tasks.tasks.bench.native_geometry import catalog, band_opening
from robot_tasks.tasks.bench.scoring_l1 import flower, marker
from robot_tasks.tasks.l1.specs import pose
from robot_tasks.video_delivery import scenes, camera_profile, PROMPT_FRAME
from test_household_tasks import measurement


def scene(tid):
    _, config = resolve_task(tid)
    spec = resolve_bench_spec(config.raw['task'], config.robot)
    return spec, measurement(spec), BenchEvaluator(spec, {a.name:a.initial for a in spec.design.articles})


class VideoDeliveryTests(unittest.TestCase):
    def test_static_fixture_contact_filters_use_shapes_and_keep_rigid_body_roots(self):
        try:
            from pxr import Usd, UsdGeom, UsdPhysics
        except ImportError:
            self.skipTest('Optional usd-core')
        from robot_tasks.tasks.bench.delivered_scene import contact_filter_paths
        stage=Usd.Stage.CreateInMemory()
        stage.DefinePrim('/World/Mug','Xform')
        for name in ('Floor','Wall'):
            UsdPhysics.CollisionAPI.Apply(UsdGeom.Cube.Define(stage,'/World/Mug/'+name).GetPrim())
        UsdPhysics.RigidBodyAPI.Apply(stage.DefinePrim('/World/Box','Xform'))
        UsdPhysics.CollisionAPI.Apply(UsdGeom.Cube.Define(stage,'/World/Box/Solid').GetPrim())
        self.assertEqual(contact_filter_paths(stage,['/World/Mug','/World/Box']),
            ['/World/Mug/Floor','/World/Mug/Wall','/World/Box'])

    def test_every_current_task_selects_its_complete_scene_and_source_prompt_frame(self):
        self.assertEqual(len(scenes()), 21)  # 16 base scenes and five delivered alternative scenes.
        for tid, row in scenes().items():
            with self.subTest(task=tid):
                entry, cfg = resolve_task(tid)
                self.assertEqual(cfg.raw['task']['scene_delivery'], 'video_20261015')
                self.assertEqual(cfg.robot.source.name, 'r5_video_20261015.toml')
                # Jelly reuses the orange scene's room/support with a new prop.
                source_tid = 'L2-04' if tid == 'L2-06' else tid.removesuffix('_v3')
                from robot_tasks.task_numbering import to_legacy
                self.assertEqual(row['path'], f'source/{to_legacy(source_tid)}/scene.usdz')
                self.assertEqual(entry.factory.prepare_config(cfg).instruction.strip(), cfg.instruction.strip())
                self.assertTrue(all('legacy' not in r['path'] for r in row['tracked'].values()))
        self.assertIn('LEFT arm is on overview-image RIGHT', PROMPT_FRAME)

    def test_geometry_is_locked_to_the_current_scene_and_band_has_a_real_opening(self):
        for name, row in catalog().items():
            if name.startswith('video:'):
                self.assertEqual(row['source_scene_sha256'], scenes()[row['task']]['sha256'])
        planes = band_opening('video:yellow_visor')
        self.assertEqual(planes.shape, (32, 3))
        self.assertTrue(np.isfinite(planes).all())
        self.assertTrue(np.all(planes[:,2] > 0))
        # The simplified band still admits a central 65 mm radius disk;
        # incorrectly selecting two radial vertices collapses this opening.
        self.assertTrue(np.all(planes[:,2] > .065))
        self.assertTrue(np.all(planes[:,2] < .082))

    def test_current_marker_needs_release_and_actual_mug_support(self):
        spec, rows, evaluator = scene('L1-01')
        t = pose_transform(spec.design.fixture_pose)
        t[2,3] += .01
        r = rows['marker']
        r['pose'] = transform_pose(t)
        r['bench_force_n'], r['fixture_force_n'] = 0., .2
        evaluator.grasped.add('marker')
        r['fingers']['right'] = [1.,1.]
        self.assertFalse(marker(evaluator,rows)[0])
        r['fingers']['right'] = [0.,0.]
        self.assertTrue(marker(evaluator,rows)[0])
        r['fixture_force_n'] = 0.
        self.assertFalse(marker(evaluator,rows)[0])

    def test_flower_containment_uses_measured_target_cup_frame(self):
        spec, rows, evaluator = scene('L1-05')
        self.assertEqual(spec.parameters['fixed_articles'], ('source_cup', 'target_cup'))
        self.assertGreater(rows['source_cup']['pose'][0], rows['target_cup']['pose'][0])
        target = pose_transform(rows['target_cup']['pose'])
        target[:3,3] += [.12,.03,.06]
        rows['target_cup']['pose'] = transform_pose(target)
        rows['rose']['pose'] = transform_pose(target @ pose_transform(pose(0,0,.01)))
        rows['rose']['bench_force_n'] = 0.
        rows['rose']['pair_forces']['target_cup'] = .1
        self.assertTrue(flower(evaluator,rows)[0])
        miss = pose_transform(rows['rose']['pose']);miss[0,3]+=.08
        rows['rose']['pose'] = transform_pose(miss)
        self.assertFalse(flower(evaluator,rows)[0])

    def test_fixed_flower_cup_still_requires_actual_flower_support(self):
        spec, rows, evaluator = scene('L1-05')
        target=pose_transform(rows['target_cup']['pose'])
        rows['rose']['pose']=transform_pose(target @ pose_transform(pose(0,0,.01)))
        rows['rose']['bench_force_n']=0.
        rows['target_cup']['bench_force_n']=0.
        rows['rose']['pair_forces']['target_cup']=.1
        self.assertTrue(flower(evaluator,rows)[0])
        rows['rose']['pair_forces']['target_cup']=0.
        self.assertFalse(flower(evaluator,rows)[0])
        rows['rose']['pair_forces']['target_cup']=.1
        evaluator.spec=replace(spec,parameters={**spec.parameters,'fixed_articles':()})
        self.assertFalse(flower(evaluator,rows)[0])

    def test_inherited_drawer_scale_and_video_camera_overrides(self):
        row = scenes()['L2-05']
        spec, _, _ = scene('L2-05')
        self.assertAlmostEqual(spec.parameters['initial_open_m'], .03*row['fixtures'][0]['inherited_scale'][1])
        from robot_tasks.cameras.profile import load_camera_rig
        _, cfg = resolve_task('L1-01')
        rig = camera_profile(load_camera_rig(cfg.source.parent / cfg.raw['simulation']['camera_profile']), 'L1-01')
        from robot_tasks.video_delivery import effective_cameras
        self.assertTrue(np.allclose(rig.cameras[0].intrinsics,
                                    effective_cameras('L1-01')['overview']['intrinsics']))
        # Immutable source metadata remains available beneath the runtime fit.
        self.assertAlmostEqual(scenes()['L1-01']['cameras']['overview']['intrinsics'][2],738.,places=3)

    def test_current_push_measurements_keep_full_3d_footprint(self):
        spec, rows, evaluator = scene('L3-03')
        from robot_tasks.tasks.bench.scoring_l3 import pushing
        for vertices in spec.parameters['book_vertices'].values():
            self.assertEqual(np.asarray(vertices).shape, (8, 3))
        for row in rows.values():
            row['finger_force_vectors'] = {a: [[0.,0.,0.], [0.,0.,0.]] for a in spec.arms}
        evaluator.last_time = 0.
        done, criteria, failure = pushing(evaluator,rows)
        self.assertFalse(done)
        self.assertFalse(failure)
        self.assertFalse(criteria['both_arms_push_witness'])

    def test_current_dustpan_basket_accepts_new_width_but_rejects_the_handle(self):
        from itertools import product
        from robot_tasks.tasks.bench.scoring_l3 import sweeping
        spec,rows,evaluator=scene('L3-04')
        pan=pose_transform(rows['dustpan']['pose'])
        def place(name,centre):
            local=np.array(list(product([-.003,.003],repeat=3)))+centre
            rows[name]['corners']=(local@pan[:3,:3].T+pan[:3,3]).tolist()
            rows[name]['pair_forces']['dustpan']=.1
        place('block_0',[.080,.040,.005])
        place('block_1',[-.060,.050,.005])
        place('block_2',[.01,.03,.005])
        rows['broom']['bench_force_n']=.1
        evaluator.flags.add('swept_block_0')
        self.assertTrue(sweeping(evaluator,rows)[0])
        place('block_0',[0.,-.090,.016])
        self.assertFalse(sweeping(evaluator,rows)[0])

    def test_documented_counts_use_current_glass_and_filler_support_needs_the_cup(self):
        for tid in ('L1-05','L2-03'):
            packing=scenes()[tid]['derived_contents']['packing_validation']
            self.assertGreaterEqual(packing['minimum_wall_clearance_m'],.0002)
            self.assertGreaterEqual(packing['minimum_floor_clearance_m'],.0002)
            self.assertGreaterEqual(packing['minimum_centre_distance_m'],.0164)
        pouring, rows, _ = scene('L2-03')
        self.assertEqual(pouring.parameters['count'],12)
        self.assertEqual(pouring.parameters['required_received'],6)
        self.assertEqual(pouring.parameters['minimum_loaded_lift_count'],12)
        self.assertEqual(sum(n.startswith('load_') for n in rows),12)
        native=scenes()['L2-03']['tracked']
        self.assertAlmostEqual(pouring.parameters['ball_mass_kg'],native['load_0']['mass'])
        self.assertAlmostEqual(pouring.parameters['loaded_source_mass_kg'],
            native['source_cup']['mass']+sum(r['mass'] for n,r in native.items() if n.startswith('load_')))
        spec, rows, evaluator = scene('L1-05')
        self.assertEqual(len(spec.parameters['target_filler_names']),10)
        target = pose_transform(rows['target_cup']['pose'])
        rows['rose']['pose'] = transform_pose(target @ pose_transform(pose(0,0,.01)))
        rows['rose']['bench_force_n'] = 0.
        name = spec.parameters['target_filler_names'][0]
        rows['rose']['pair_forces'][name] = .1
        rows[name]['pair_forces']['target_cup'] = .1
        self.assertTrue(flower(evaluator,rows)[0])
        rows[name]['pair_forces']['target_cup'] = 0.
        self.assertFalse(flower(evaluator,rows)[0])
        rows[name]['pair_forces']['target_cup'] = .1
        rows[name]['pose'][0] += .1
        self.assertFalse(flower(evaluator,rows)[0])

    def test_sdf_tasks_reject_cpu_before_starting_kit(self):
        from robot_tasks.backends.isaacsim import IsaacSimBackend
        for tid in ('L3-01','L3-02'):
            _, cfg = resolve_task(tid)
            self.assertTrue(cfg.raw['simulation']['gpu_dynamics'])
            raw = {**cfg.raw,'simulation':{**cfg.raw['simulation'],'gpu_dynamics':False}}
            with self.assertRaisesRegex(ValueError,'SDF colliders require'):
                IsaacSimBackend().open(replace(cfg,raw=raw))

    def test_offset_intrinsics_agree_with_usd_projection(self):
        try:
            from pxr import Usd, UsdGeom
        except ImportError:
            self.skipTest('Optional usd-core')
        row = scenes()['L1-01']
        source = PROJECT_ROOT / 'assets/tuojing' / row['path']
        if not source.exists():
            self.skipTest('Private delivery is not installed')
        stage = Usd.Stage.Open(str(source))
        camera = UsdGeom.Camera(stage.GetPrimAtPath(row['cameras']['overview']['path']))
        projection = np.asarray(camera.GetCamera().frustum.ComputeProjectionMatrix())
        point = np.array([.12,.08,-1.3,1.])
        clip = point @ projection
        ndc = clip[:2]/clip[3]
        w,h = row['cameras']['overview']['resolution']
        fx,fy,cx,cy = row['cameras']['overview']['intrinsics']
        expected = [fx*point[0]/-point[2]+cx, -fy*point[1]/-point[2]+cy]
        np.testing.assert_allclose([(ndc[0]+1)*w/2,(1-ndc[1])*h/2],expected,atol=1e-4)

    def test_visual_revision_preserves_collisions_and_rejects_a_physical_change(self):
        try:
            from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf
        except ImportError:
            self.skipTest('Optional usd-core')
        import tempfile
        from tools.video_visual_r3 import apply
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene_file = root/'release/L1-01/static/scene.usdc'
            scene_file.parent.mkdir(parents=True)
            revision = Usd.Stage.CreateNew(str(scene_file))
            revision.DefinePrim('/World','Xform')
            revision.DefinePrim('/World/Materials','Scope')
            mat = UsdShade.Material.Define(revision,'/World/Materials/Test')
            mat.GetPrim().CreateAttribute('reviewed_color',Sdf.ValueTypeNames.Float3).Set((.2,.3,.4))
            revision.GetRootLayer().Save()
            current = Usd.Stage.CreateInMemory()
            current.DefinePrim('/World','Xform')
            current.DefinePrim('/World/Materials','Scope')
            UsdShade.Material.Define(current,'/World/Materials/Test')
            cube = UsdGeom.Cube.Define(current,'/World/Object')
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
            cube.CreateSizeAttr(.017)
            result = apply(current,root/'release','L1-01',root/'assets')
            self.assertEqual(result['physical_signature_before'],result['physical_signature_after'])
            self.assertAlmostEqual(cube.GetSizeAttr().Get(),.017)
            np.testing.assert_allclose(current.GetPrimAtPath('/World/Materials/Test').GetAttribute('reviewed_color').Get(),(.2,.3,.4),atol=1e-7)
            revision.DefinePrim('/World/ReferenceDetail','Xform')
            bad = UsdGeom.Cube.Define(revision,'/World/ReferenceDetail/TableSurface')
            UsdPhysics.CollisionAPI.Apply(bad.GetPrim())
            revision.GetRootLayer().Save()
            with self.assertRaisesRegex(ValueError,'changed physical'):
                apply(current,root/'release','L1-01',root/'assets')
