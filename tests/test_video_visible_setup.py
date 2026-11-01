"""Video-derived initial settings must reach the robot, camera and recording."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import hashlib
import copy
import json
import tempfile
import unittest
import numpy as np

from robot_tasks.registry import resolve_task
from robot_tasks.video_delivery import (apply_visual_cameras, apply_visual_background, camera_profile,
    initial_gripper_width, scenes, setup_for, setup_compatible, planning_setup_compatible,
    effective_cameras, video_layout)


class VideoVisibleSetupTests(unittest.TestCase):
    def test_shared_overview_is_locked_to_the_verified_user_image_fit(self):
        from robot_tasks.video_delivery import visual_setup
        from robot_tasks.paths import PROJECT_ROOT
        lock = visual_setup()['overview_reference_image_fit']
        raw = (PROJECT_ROOT/lock['path']).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), lock['sha256'])
        fit = json.loads(raw)
        self.assertEqual(hashlib.sha256((PROJECT_ROOT/lock['image']).read_bytes()).hexdigest(),
                         lock['image_sha256'])
        current = effective_cameras('L3-01')['overview']
        for key in ('pose', 'intrinsics'):
            self.assertEqual(current[key], fit[key])
        self.assertEqual(current['resolution'], fit['native_resolution'])
        self.assertFalse(lock['runtime_image_warp'])
        self.assertLess(fit['native_preview_verification']['direct_no_warp_median_px'], .5)
        self.assertLess(fit['native_preview_verification']['direct_no_warp_p95_px'], 1.5)

    def test_every_task_uses_the_bottle_cap_camera_rig_and_rejects_overrides(self):
        reference = effective_cameras('L3-01')
        for tid in scenes():
            for name, camera in effective_cameras(tid).items():
                for key in ('parent', 'pose', 'intrinsics', 'resolution', 'clipping_m'):
                    self.assertEqual(camera[key], reference[name][key], (tid, name, key))
        original = setup_for
        for name, key in [('overview', 'pose'), ('overview', 'intrinsics'),
                          ('left_wrist', 'pose'), ('right_wrist', 'intrinsics')]:
            changed = copy.deepcopy(setup_for('L1-02'))
            changed['cameras'][name][key][0] += .01
            with patch('robot_tasks.video_delivery.setup_for',
                       side_effect=lambda tid: changed if tid == 'L1-02' else original(tid)):
                with self.assertRaisesRegex(ValueError, 'must match shared camera reference L3-01'):
                    effective_cameras('L1-02')

    def test_historical_evidence_is_reused_only_for_an_identical_complete_task_record(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'current.json'
            record = {'arms': {'left': {'home': [0, 0, 0]}}, 'cameras': {'fx': 500}}
            old = json.dumps({'tasks': {'L1-01': record, 'L1-02': {'old': True}}}).encode()
            sha = hashlib.sha256(old).hexdigest()
            history = path.parent/'video_visual_setup_history'
            history.mkdir()
            (history/(sha+'.json')).write_bytes(old)
            with patch('robot_tasks.video_delivery.VISUAL_SETUP', path), \
                 patch('robot_tasks.video_delivery.setup_digest', return_value='f'*64), \
                 patch('robot_tasks.video_delivery.setup_for', return_value=record):
                self.assertTrue(setup_compatible('L1-01', sha))
                self.assertFalse(setup_compatible('L1-02', sha))
                changed = {**record, 'cameras': {'fx': 600}}
                with patch('robot_tasks.video_delivery.setup_for', return_value=changed):
                    self.assertFalse(setup_compatible('L1-01', sha))
                    self.assertFalse(planning_setup_compatible('L1-01', sha))
                recolored = {**record, 'render_appearance': {'intensity_multiplier': 1.3}}
                with patch('robot_tasks.video_delivery.setup_for', return_value=recolored):
                    self.assertFalse(setup_compatible('L1-01', sha))
                    self.assertTrue(planning_setup_compatible('L1-01', sha))
                moved = {**recolored, 'arms': {'left': {'home': [.1, 0, 0]}}}
                with patch('robot_tasks.video_delivery.setup_for', return_value=moved):
                    self.assertFalse(planning_setup_compatible('L1-01', sha))

    def test_historical_configuration_requires_the_recorded_byte_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'current.json'
            old = b'{"tasks":{"L1-01":{"fx":500}}}'
            sha = hashlib.sha256(old).hexdigest()
            history = path.parent/'video_visual_setup_history'
            history.mkdir()
            archived = history/(sha+'.json')
            archived.write_bytes(old.replace(b'500', b'600'))
            with patch('robot_tasks.video_delivery.VISUAL_SETUP', path), \
                 patch('robot_tasks.video_delivery.setup_digest', return_value='f'*64), \
                 patch('robot_tasks.video_delivery.setup_for', return_value={'fx':600}):
                with self.assertRaisesRegex(ValueError, 'Modified historical'):
                    setup_compatible('L1-01', sha)
                self.assertFalse(setup_compatible('L1-01', '0'*64))
                self.assertFalse(setup_compatible('L1-01', '../untrusted'))

    def test_video_wrist_rotations_and_openings_replace_the_uniform_reset(self):
        for tid in scenes():
            entry, raw = resolve_task(tid)
            config = entry.factory.prepare_config(raw)
            for mount in config.robot.mounts:
                target = setup_for(tid)['arms'][mount.name]
                self.assertEqual(mount.home_positions_rad, tuple(target['home_positions_rad']))
                self.assertAlmostEqual(initial_gripper_width(config, mount.name, .035), target['gripper_width_m'])
        entry, config = resolve_task('L1-02')
        config = entry.factory.prepare_config(config)
        self.assertGreater(config.robot.mounts[0].home_positions_rad[5], 1.3)
        _, closed = resolve_task('L2-05')
        self.assertLess(initial_gripper_width(closed, 'left', .035), .004)
        _, open_ = resolve_task('L1-01')
        self.assertGreater(initial_gripper_width(open_, 'left', .035), .06)

    def test_session_camera_fit_matches_the_bound_camera_and_keeps_source_layer(self):
        try:
            from pxr import Usd, UsdGeom
        except ImportError:
            self.skipTest('Optional usd-core')
        from robot_tasks.paths import PROJECT_ROOT
        from robot_tasks.cameras.profile import load_camera_rig
        from robot_tasks.cameras.isaac_rig import validate_existing_camera
        shared_bounds = None
        for tid, row in scenes().items():
            with self.subTest(task=tid):
                asset = PROJECT_ROOT/'assets/tuojing'/row['runtime_path']
                if not asset.exists():
                    self.skipTest('Optional delivered scene')
                stage = Usd.Stage.Open(str(asset))
                before = stage.GetRootLayer().ExportToString()
                stage.SetEditTarget(stage.GetSessionLayer())
                apply_visual_cameras(stage, tid)
                apply_visual_background(stage, tid)
                self.assertEqual(UsdGeom.Imageable(stage.GetPrimAtPath(
                    '/World/Furniture/ThirdCameraStand')).ComputeVisibility(), 'invisible')
                bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render', 'proxy'])
                framing = []
                for path in ('/World/Furniture/Worktable/Top', '/World/Room/PowerStrip'):
                    box = bounds.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
                    framing.extend([list(box.GetMin()), list(box.GetMax())])
                if shared_bounds is None:
                    shared_bounds = framing
                np.testing.assert_allclose(framing, shared_bounds, atol=1e-8, rtol=0,
                                           err_msg=tid + ' table/background framing differs')
                rig = camera_profile(load_camera_rig(PROJECT_ROOT/'configs/cameras/r5_tuojing_lab.toml'), tid)
                for spec in rig.cameras:
                    parent = '/World' if spec.parent == 'world' else row['robots'][spec.parent]['path']+'/link6'
                    self.assertEqual(validate_existing_camera(stage, spec, parent), spec.existing_prim_path)
                self.assertEqual(before, stage.GetRootLayer().ExportToString())

    def test_recording_layout_uses_the_real_panes_and_robot_perspective(self):
        for tid in scenes():
            layout = video_layout(tid)
            w,h = layout['resolution']
            for roi in layout['rois'].values():
                if roi is None:
                    continue
                x0,y0,x1,y1 = roi
                self.assertTrue(0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h)
        layout = video_layout('L1-01')
        self.assertLess(layout['rois']['right_wrist'][0], layout['rois']['left_wrist'][0])
        self.assertIsNone(video_layout('L1-05')['rois']['left_wrist'])

    def test_rgb_views_are_composed_into_the_matching_real_video_panes(self):
        try:
            import cv2
        except ImportError:
            self.skipTest('Optional OpenCV')
        from robot_tasks.integrationtest.diagnostics.privileged_smoke import Controller
        from robot_tasks.types import RGBFrame
        frames = []
        for name,size,rgb in [('overview',(1280,960),(255,255,255)),
                              ('right_wrist',(1280,720),(255,0,0)),
                              ('left_wrist',(1280,720),(0,0,255))]:
            data = np.full((size[1],size[0],3), rgb, dtype=np.uint8)
            frames.append(RGBFrame(name, .2, *size, data.tobytes()))
        c = Controller.__new__(Controller)
        c.backend = SimpleNamespace(capture_rgb=lambda:frames, time_s=lambda:.2)
        c.config = SimpleNamespace(task_id='L1-01')
        c.cv2, c.label, c.assessment = cv2, 'initial', None
        c.initial_image_saved, c.frames_written = True, 0
        c.writer = SimpleNamespace(write=lambda _:None)
        c.video_layout = video_layout('L1-01')
        canvas = c.capture()
        self.assertEqual(canvas.shape, (1196,960,3))
        for name,bgr in [('overview',(255,255,255)),('right_wrist',(0,0,255)),('left_wrist',(255,0,0))]:
            x0,y0,x1,y1 = c.video_layout['rois'][name]
            self.assertEqual(tuple(canvas[(y0+y1)//2,(x0+x1)//2]), bgr)


if __name__ == '__main__':
    unittest.main()
