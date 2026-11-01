"""Physical-binding regressions: one visible support, detachable pen, fixed bottle."""
import unittest
import numpy as np
from robot_tasks.video_delivery import scenes
from robot_tasks.paths import PROJECT_ROOT

try:
    from pxr import Usd, UsdGeom, UsdPhysics
except ImportError:
    Usd = None

@unittest.skipIf(Usd is None, 'USD required for delivery geometry contracts')
class VideoSceneRepairs(unittest.TestCase):
    def stage(self, task):
        return Usd.Stage.Open(str(PROJECT_ROOT/'assets/tuojing'/scenes()[task]['runtime_path']))

    def test_hat_support_is_free_and_open_after_contact_simplification(self):
        from scipy.spatial import ConvexHull
        from robot_tasks.tasks.bench.native_geometry import verify_stage
        stage = self.stage('L1-03')
        original = Usd.Stage.Open(str(PROJECT_ROOT/'assets/tuojing'/scenes()['L1-03']['path']))
        root = '/World/Household'
        box = stage.GetPrimAtPath(root+'/source_box')
        hat = stage.GetPrimAtPath(root+'/hat')
        self.assertFalse(UsdPhysics.RigidBodyAPI(box).GetKinematicEnabledAttr().Get())
        self.assertFalse(UsdPhysics.RigidBodyAPI(hat).GetKinematicEnabledAttr().Get())
        self.assertAlmostEqual(UsdPhysics.MassAPI(box).GetMassAttr().Get(), .6, places=6)
        self.assertAlmostEqual(UsdPhysics.MassAPI(hat).GetMassAttr().Get(), .065, places=6)
        boxes = list(stage.GetPrimAtPath(root+'/source_box/Colliders').GetChildren())
        self.assertEqual(len(boxes), 5)
        self.assertTrue(all(p.IsA(UsdGeom.Cube) and
            UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() for p in boxes))
        # The real, projected convex colliders must leave space around the
        # head-opening centre, independently of the scorer's opening formula.
        for prim in stage.GetPrimAtPath(root+'/hat/Colliders').GetChildren():
            points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
            faces = ConvexHull(points[:, :2]).equations
            self.assertGreater(float(faces[:, 2].max()), .04)
        verify_stage(stage, [('video:yellow_visor', root+'/hat')])
        # Physics simplification must not simplify the visible hat or box.
        for name in ('hat', 'source_box'):
            for prim in Usd.PrimRange(stage.GetPrimAtPath(root+'/'+name)):
                if prim.IsA(UsdGeom.Mesh) and '/Colliders/' not in str(prim.GetPath()):
                    before = UsdGeom.Mesh(original.GetPrimAtPath(prim.GetPath()))
                    np.testing.assert_array_equal(UsdGeom.Mesh(prim).GetPointsAttr().Get(),
                                                  before.GetPointsAttr().Get())

    def test_mug17_preserves_height_and_keeps_scoring_on_actual_colliders(self):
        from robot_tasks.tasks.bench.native_geometry import cup_cavity, verify_stage, catalog
        stage = self.stage('L1-01')
        original = Usd.Stage.Open(str(PROJECT_ROOT/'assets/tuojing'/scenes()['L1-01']['path']))
        root = '/World/Household/Fixture'
        for prim in Usd.PrimRange(stage.GetPrimAtPath(root)):
            if prim.IsA(UsdGeom.Mesh) and prim.GetName() not in {'WhiteLip','Inside'} and not prim.GetName().startswith(('Wall_','Rim_')):
                before = np.asarray(UsdGeom.Mesh(original.GetPrimAtPath(prim.GetPath())).GetPointsAttr().Get())
                after = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
                np.testing.assert_array_equal(before[:, 2], after[:, 2])
        points = np.asarray(UsdGeom.Mesh(stage.GetPrimAtPath(root+'/Visuals/OuterBody')).GetPointsAttr().Get())
        radii = np.linalg.norm(points[:, :2], axis=1)
        self.assertAlmostEqual(float(radii.max()) * 2, .0984, places=6)
        self.assertAlmostEqual(float(radii[np.isclose(points[:,2], .089)].max()) * 2, .088, places=6)
        inside = np.asarray(UsdGeom.Mesh(stage.GetPrimAtPath(root+'/Visuals/Inside')).GetPointsAttr().Get())
        self.assertLess(np.linalg.norm(inside[:, :2], axis=1).min(), 2e-6,
                        'Reshaping the belly must not open a visible hole in the inner floor')
        self.assertAlmostEqual(float(inside[:,2].max()), .085, places=7)
        inner_mouth = np.linalg.norm(inside[np.isclose(inside[:,2],.085),:2],axis=1)
        self.assertAlmostEqual(float(inner_mouth.min()), .0375, places=6)
        self.assertGreater(.044-float(inner_mouth.max()),.006,
                           'The marked white ceramic rim must have physical radial thickness')
        lip = UsdGeom.Mesh(stage.GetPrimAtPath(root+'/Visuals/WhiteLip'))
        lip_points = np.asarray(lip.GetPointsAttr().Get())
        self.assertAlmostEqual(float(lip_points[:,2].max()), .09, places=7)
        self.assertAlmostEqual(float(lip_points[:,2].min()), .085, places=7)
        self.assertGreater(float(lip_points[:,2].max()-inside[:,2].max()), .0049,
                           'The inner lip must sit clearly below the outer crown')
        from pxr import UsdShade
        edge = stage.GetPrimAtPath(root+'/Visuals/WhiteLip/OuterBlackEdge')
        self.assertTrue(UsdShade.MaterialBindingAPI(edge).ComputeBoundMaterial()[0].GetPath().name == 'RimBlackGlaze')
        before = UsdGeom.Xformable(original.GetPrimAtPath(root+'/Visuals')).GetLocalTransformation()
        after = UsdGeom.Xformable(stage.GetPrimAtPath(root+'/Visuals')).GetLocalTransformation()
        np.testing.assert_array_equal(np.asarray(before), np.asarray(after))
        verify_stage(stage, [('video:black_ceramic_mug', root)])
        planes, floor, rim = cup_cavity('video:black_ceramic_mug')
        self.assertAlmostEqual(rim, .085*85/90, places=6)
        self.assertGreater(rim-floor, .07)
        # Central marker-sized cross sections must fit into the hollow cup.
        for z in [.02, .05, .08]:
            for x, y in [(0, .009), (0, -.009), (.009, 0), (-.009, 0)]:
                self.assertTrue(np.all(planes[:, :3] @ [x,y,z] <= planes[:,3]))
        self.assertEqual(catalog()['video:black_ceramic_mug']['runtime_scene_sha256'],
                         scenes()['L1-01']['runtime_sha256'])

    def test_latest_boxes_are_the_only_active_fixed_supports(self):
        for task in ['L2-01','L2-02','L2-04','L2-05','L3-02','L3-04']:
            with self.subTest(task=task):
                stage=self.stage(task)
                self.assertFalse(stage.GetPrimAtPath('/World/ReviewAssets/MeasuredBox'))
                row=next(r for r in scenes()[task]['fixtures'] if r['path'].endswith(('/support_box','/Pedestal')))
                body=stage.GetPrimAtPath(row['path'])
                self.assertEqual(UsdGeom.Imageable(body).ComputeVisibility(),'inherited')
                self.assertTrue(UsdPhysics.RigidBodyAPI(body).GetKinematicEnabledAttr().Get())
                self.assertAlmostEqual(row['bounds'][1][2],.070 if task.startswith('L2-') else .075,places=5)
                self.assertGreater(sum(p.HasAPI(UsdPhysics.CollisionAPI) for p in Usd.PrimRange(body)),0)

    def test_measured_pen_is_bound_to_two_visible_physical_bodies(self):
        stage=self.stage('L2-01')
        self.assertFalse(stage.GetPrimAtPath('/World/ReviewAssets/MeasuredMarker'))
        for name,row in scenes()['L2-01']['tracked'].items():
            body=stage.GetPrimAtPath(row['path'])
            self.assertTrue(body.HasAPI(UsdPhysics.RigidBodyAPI))
            self.assertFalse(UsdPhysics.RigidBodyAPI(body).GetKinematicEnabledAttr().Get())
            self.assertEqual(UsdGeom.Imageable(body).ComputeVisibility(),'inherited')
            self.assertTrue(stage.GetPrimAtPath(row['path']+'/LatestVisuals'))
        barrel=UsdGeom.Cylinder(stage.GetPrimAtPath('/World/L2/marker/Body/Colliders/LatestExterior/Barrel'))
        cache=UsdGeom.XformCache()
        body=stage.GetPrimAtPath('/World/L2/marker/Body')
        relative=np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(body)).T)@np.asarray(cache.GetLocalToWorldTransform(barrel.GetPrim())).T
        end=relative[2,3]+barrel.GetHeightAttr().Get()/2
        cap_min=min(np.asarray(UsdGeom.Mesh(p).GetPointsAttr().Get())[:,2].min() for p in Usd.PrimRange(stage.GetPrimAtPath('/World/L2/marker/Cap/Colliders')) if p.IsA(UsdGeom.Mesh))
        self.assertLessEqual(end,cap_min+1e-7, 'New barrel must not penetrate the cap lip')

    def test_l2_measured_box_has_a_physical_ten_mm_recess(self):
        for task in ('L2-01', 'L2-02', 'L2-04', 'L2-05'):
            with self.subTest(task=task):
                row = scenes()[task]
                support = row['l2_support_measurement']
                fixture = next(r for r in row['fixtures'] if r['path'] == support['fixture'])
                np.testing.assert_allclose(np.ptp(fixture['bounds'], axis=0), [.2, .2, .07], atol=1e-7)
                stage = self.stage(task)
                cache = UsdGeom.XformCache()
                root_inverse = np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(fixture['path']))).T)
                def vertices(path):
                    mesh = UsdGeom.Mesh(stage.GetPrimAtPath(path))
                    transform = root_inverse @ np.asarray(cache.GetLocalToWorldTransform(mesh.GetPrim())).T
                    return np.asarray(mesh.GetPointsAttr().Get()) @ transform[:3, :3].T + transform[:3, 3]
                inset = vertices(fixture['path'] + '/Colliders/InsetFace')
                self.assertAlmostEqual(inset[:, 2].max(), .06, places=7)
                rim = np.concatenate([vertices(str(p.GetPath())) for p in Usd.PrimRange(stage.GetPrimAtPath(fixture['path'] + '/Colliders')) if p.GetName().startswith('Rim_')])
                self.assertAlmostEqual(rim[:, 2].max() - inset[:, 2].max(), .01, places=7)
                shell = vertices(fixture['path'] + '/Visuals/SteppedMoldedShell')
                np.testing.assert_allclose(np.ptp(shell, axis=0), [.2, .2, .07], atol=1e-7)

    def test_l2_supported_items_and_scoring_follow_the_correct_surface(self):
        from test_video_delivery import scene
        for task, expected in (('L2-01', .94), ('L2-02', .95), ('L2-04', .94)):
            spec, _, _ = scene(task)
            self.assertAlmostEqual(spec.parameters['support_z'], expected, places=7)
        books = scenes()['L2-02']['tracked']
        from robot_tasks.robots.transforms import pose_transform
        lower = books['lower_book']
        lo, hi = lower['bounds']
        transform = pose_transform(lower['pose'])
        bottom = np.array([[x, y, lo[2]] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])])
        bottom = bottom @ transform[:3, :3].T + transform[:3, 3]
        self.assertGreater(bottom[:, 2].min(), .944, 'Shaped underside must clear the 0.94 m inset platform')
        self.assertLess(bottom[:, 2].min(), .95)
        # The lower book's fixed bent shape carries the upper books' lateral
        # incline. Its body frame stays upright to avoid fictitious AABB lows.
        upper = pose_transform(books['upper_book']['pose'])
        self.assertGreater(upper[2, 0], .05, 'Image-right edge should be raised')
        self.assertLess(abs(upper[2, 1]), .01, 'Do not tilt the stack mainly fore/aft')
        self.assertTrue(all(not r['kinematic'] for r in books.values()))
        orange = scenes()['L2-04']['orange']
        self.assertAlmostEqual(orange['pose'][2], .940075, places=7)
        self.assertAlmostEqual(scenes()['L2-05']['fixtures'][0]['pose'][2], .95, places=7)

    def test_lower_book_bend_has_matching_dynamic_collision_and_real_rim_clearance(self):
        stage = self.stage('L2-02')
        path = scenes()['L2-02']['tracked']['lower_book']['path']
        self.assertFalse(stage.GetPrimAtPath(path + '/Colliders/Envelope'))
        cache = UsdGeom.XformCache()
        cells = []
        for prim in Usd.PrimRange(stage.GetPrimAtPath(path + '/Colliders')):
            if prim.IsA(UsdGeom.Mesh) and prim.HasAPI(UsdPhysics.CollisionAPI):
                points = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
                matrix = np.asarray(cache.GetLocalToWorldTransform(prim)).T
                cells.append(points[:4] @ matrix[:3, :3].T + matrix[:3, 3])
        self.assertGreater(len(cells), 1)
        bottom = np.concatenate(cells)
        self.assertGreater(bottom[:, 2].min(), .944)
        self.assertLess(bottom[:, 2].min(), .946)
        self.assertGreaterEqual(bottom[:, 2].max(), .9502 - 1e-7)
        self.assertLess(bottom[:, 2].max(), .952, 'Only the raised overhang may exceed the rim')
        self.assertGreater(np.ptp(bottom[:, 2]), .004)
        body = UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(path))
        self.assertFalse(body.GetKinematicEnabledAttr().Get())

    def test_cup_positions_match_video_in_fixed_camera_and_all_twelve_balls_fit(self):
        import json
        from robot_tasks.cameras.geometry import project_world
        from robot_tasks.robots.transforms import pose_transform
        from robot_tasks.tasks.bench.pearl_pouring import cup_contains
        config = json.loads((PROJECT_ROOT / 'configs/task_references/l2_layout_refinement_20261015.json').read_text())['tasks']['L2-03']
        tracked = scenes()['L2-03']['tracked']
        for name, target in config['cup_base_centres_target_px'].items():
            actual = project_world([tracked[name]['pose'][:3]], config['camera_pose_used'], config['intrinsics_used'])[0]
            np.testing.assert_allclose(actual, target, atol=.01)
        stage = self.stage('L2-03')
        loads = [row for name, row in tracked.items() if name.startswith('load_')]
        self.assertEqual(len(loads), 12)
        source_inverse = np.linalg.inv(pose_transform(tracked['source_cup']['pose']))
        centres = []
        for row in loads:
            body = stage.GetPrimAtPath(row['path'])
            self.assertTrue(body.HasAPI(UsdPhysics.RigidBodyAPI))
            self.assertGreaterEqual(body.GetAttribute('physxRigidBody:solverVelocityIterationCount').Get(), 16)
            self.assertFalse(row['kinematic'])
            point = (source_inverse @ pose_transform(row['pose']))[:3, 3]
            self.assertTrue(cup_contains(point, radius=.008, receptacle='video:ripple_paper_cup'))
            centres.append(point)
        distances = np.linalg.norm(np.array(centres)[:, None] - np.array(centres)[None, :], axis=2)
        distances += np.eye(12)
        self.assertGreater(distances.min(), .016)
        glass_assets = [p.GetAttribute('info:mdl:sourceAsset').Get().path for p in stage.Traverse()
                        if p.GetAttribute('info:mdl:sourceAsset') and
                        'OmniGlass.mdl' in p.GetAttribute('info:mdl:sourceAsset').Get().path]
        self.assertTrue(glass_assets)
        self.assertEqual(set(glass_assets), {'OmniGlass.mdl'})

    def test_measured_support_patch_is_idempotent_and_l2_only(self):
        from tools.l2_measured_support import apply
        for task in ('L2-02', 'L3-02'):
            stage = self.stage(task)
            before = stage.GetRootLayer().ExportToString()
            result = apply(stage, task, scenes()[task])
            self.assertEqual(stage.GetRootLayer().ExportToString(), before)
            self.assertEqual(result is None, task == 'L3-02')

    def test_bottle_fixed_in_asset_and_active_scoring(self):
        from test_video_delivery import scene
        stage=self.stage('L3-01');spec,rows,e=scene('L3-01')
        self.assertTrue(UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(scenes()['L3-01']['tracked']['bottle']['path'])).GetKinematicEnabledAttr().Get())
        self.assertTrue(spec.parameters['body_fixed'])
        self.assertIn('fixed',spec.design.instruction)

    def test_whiteboard_scoring_uses_rotated_physical_fixture(self):
        from test_video_delivery import scene
        from robot_tasks.robots.transforms import pose_transform
        spec,_,_=scene('L2-07');row=scenes()['L2-07']['fixtures'][0]
        np.testing.assert_allclose(spec.parameters['board_pose'],row['pose'])
        self.assertGreater(pose_transform(row['pose'])[0,0],.99)

    def test_background_overlay_never_replaces_task_prop_materials(self):
        for task,row in scenes().items():
            overlay=row.get('visual_overlay',{})
            for path in overlay.get('copied_render_prims',[]):
                self.assertFalse(path.startswith(('/World/Household/','/World/L2/','/World/L3Wiping/','/World/ReviewAssets/')),(task,path))
            self.assertFalse(overlay.get('physical_changes'))
