from pathlib import Path
import tempfile
import unittest

import numpy as np

from robot_tasks.cameras.profile import intrinsics_from_fov, load_camera_rig
from robot_tasks.config import load_config
from robot_tasks.backends.settings import SimulationSettings
from robot_tasks.robots.transforms import pose_transform


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "configs/cameras/r5_orbbec_provisional.toml"


class CameraProfileTests(unittest.TestCase):
    def test_portable_profile_and_explicit_provenance(self):
        rig = load_camera_rig(PROFILE)
        self.assertEqual(rig.calibration_status, "provisional_uncalibrated")
        self.assertEqual([c.parent for c in rig.cameras], ["world", "left", "right"])
        for camera in rig.cameras:
            self.assertTrue(camera.asset.is_file())
            self.assertEqual(len(camera.metadata()["asset_sha256"]), 64)
        config = load_config(ROOT / "configs/checks/gpt_single_arm_pick_place.toml")
        settings = SimulationSettings.from_config(config)
        self.assertEqual((config.source.parent / settings.camera_profile).resolve(), PROFILE)
        self.assertEqual(tuple(c.name for c in rig.select(config.cameras)), config.cameras)

    def test_camera_order_is_explicit_and_unknown_names_fail(self):
        rig = load_camera_rig(PROFILE)
        self.assertEqual(rig.select(("right_wrist", "overview"))[0].parent, "right")
        with self.assertRaisesRegex(ValueError, "missing"):
            rig.select(("depth",))

    def test_estimated_fov_covers_known_projection(self):
        fx, fy, cx, cy = intrinsics_from_fov(640, 400, 90, 60)
        self.assertAlmostEqual(fx, 320)
        self.assertAlmostEqual(fy * np.tan(np.pi / 6) + cy, 400)
        self.assertAlmostEqual(fx * 1.0 + cx, 640)  # Point (X=Z) at 45 degrees.
        for args in ((0, 400, 90, 60), (640, 400, 180, 60), (640, 400, float("nan"), 60)):
            with self.assertRaises(ValueError):
                intrinsics_from_fov(*args)

    def test_wrist_views_cover_grasp_region(self):
        # Validate usable geometry, rather than one hard-coded look-at point.
        for spec in load_camera_rig(PROFILE).select(("left_wrist", "right_wrist")):
            with self.subTest(camera=spec.name):
                transform = pose_transform(spec.optical_pose_wxyz)
                forward = -transform[:3, 2]
                self.assertGreater(forward[0], 0)
                self.assertLess(forward[2], 0)
                fx, fy, cx, cy = spec.intrinsics
                for point in ((0.14, 0, 0), (0.16, -0.025, 0), (0.16, 0.025, 0)):
                    x, y, z = transform[:3, :3].T @ (np.asarray(point) - transform[:3, 3])
                    self.assertLess(z, -spec.clipping_m[0])  # USD forward is -Z.
                    u, v = cx + fx * x / -z, cy - fy * y / -z
                    self.assertTrue(0 < u < spec.resolution[0])
                    self.assertTrue(0 < v < spec.resolution[1])

    def test_malformed_and_unearned_calibration_are_rejected(self):
        content = PROFILE.read_text(encoding="utf-8")
        for relative in ("../../assets/cameras/gemini_pro.usdc", "../../assets/cameras/gemini305_r5_mounted.usda"):
            content = content.replace(relative, (PROFILE.parent / relative).resolve().as_posix())
        cases = (
            ('calibration_status = "provisional_uncalibrated"', 'calibration_status = "calibrated"', "provisional_uncalibrated"),
            ('parent = "left"', 'parent = "franka"', "parent"),
            ('resolution = [640, 480]', 'resolution = [true, 480]', "resolution"),
            ('intrinsics = [444.508041', 'intrinsics = [-444.508041', "focal"),
            ('clipping_m = [0.005, 20.0]', 'clipping_m = [0.005, 0.001]', "clipping"),
            ('0.6532814824381883, 0.2705980500730985', '2.0, 0.2705980500730985', "normalized"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rig.toml"
            for before, after, error in cases:
                with self.subTest(error=error):
                    path.write_text(content.replace(before, after), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, error):
                        load_camera_rig(path)


if __name__ == "__main__":
    unittest.main()
