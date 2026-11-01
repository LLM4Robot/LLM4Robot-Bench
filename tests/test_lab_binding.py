"""Offline scene-binding checks; no Kit, GPU, private delivery or model calls."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PIL import Image

import numpy as np

from robot_tasks.backends.settings import SimulationSettings
from robot_tasks.cameras.profile import CameraSpec

try:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
except ImportError:
    Usd = None


class LabSettingsTests(unittest.TestCase):
    def test_video_matched_overview_and_unchanged_wrist_and_legacy_cameras(self):
        from robot_tasks.backends.scene_geometry import bench_geometry
        from robot_tasks.cameras.profile import load_camera_rig
        from robot_tasks.registry import resolve_task
        from robot_tasks.robots.transforms import pose_transform

        _, config = resolve_task("L1-01")
        bench = bench_geometry(config)
        rig = load_camera_rig(config.source.parent / config.raw["simulation"]["camera_profile"])
        overview = rig.cameras[0]
        self.assertAlmostEqual(overview.optical_pose_wxyz[2], 1.45)
        self.assertEqual(overview.resolution, (1280, 960))
        self.assertAlmostEqual(overview.optical_pose_wxyz[0],
                               .1956512155)
        self.assertLess(overview.optical_pose_wxyz[1], bench.centre[1] - bench.size[1] / 2)
        forward = -pose_transform(overview.optical_pose_wxyz)[:3, 2]
        np.testing.assert_allclose(forward, [0, np.cos(np.deg2rad(36)), -np.sin(np.deg2rad(36))], atol=1e-9)
        self.assertGreater(forward[1], 0)  # Preserve the delivered heading toward the desk.
        for spec in rig.cameras[1:]:
            t = pose_transform(spec.optical_pose_wxyz)
            np.testing.assert_allclose(-t[:3, 2], [np.sqrt(.5), 0, -np.sqrt(.5)], atol=1e-9)
            np.testing.assert_allclose(t[:3, 0], [0, -1, 0], atol=1e-9)
        # This request only changes the delivered laboratory; legacy scenes retain their framing.
        _, legacy = resolve_task("GPT-PICK-PLACE")
        legacy_bench = bench_geometry(legacy)
        legacy_rig = load_camera_rig(legacy.source.parent / legacy.raw["simulation"]["camera_profile"])
        overview = legacy_rig.cameras[0]
        self.assertAlmostEqual(overview.optical_pose_wxyz[2] - legacy_bench.top, 1.58 - .78)
        self.assertAlmostEqual(overview.optical_pose_wxyz[0] - (legacy_bench.centre[0] + legacy_bench.size[0] / 2), .25)
        np.testing.assert_allclose(-pose_transform(overview.optical_pose_wxyz)[:3, 2],
                                   np.array([-.87, 0, -.72]) / np.linalg.norm([-.87, 0, -.72]), atol=1e-9)
        for spec in legacy_rig.cameras[1:]:
            np.testing.assert_allclose(-pose_transform(spec.optical_pose_wxyz)[:3, 2],
                                       [np.sqrt(.5), 0, -np.sqrt(.5)], atol=1e-9)

    def test_native_lab_requires_camera_binding(self):
        config = SimpleNamespace(source=Path("configs/checks/example.toml"), raw={"simulation": {"environment": "tuojing_lab_v4"}})
        with self.assertRaisesRegex(ValueError, "existing-camera"):
            SimulationSettings.from_config(config)
        config.raw["simulation"]["camera_profile"] = "../cameras/r5_tuojing_lab.toml"
        self.assertEqual(SimulationSettings.from_config(config).environment, "tuojing_lab_v4")

    def test_lab_table_height_reaches_notes_grasp_guard_and_image_projection(self):
        from robot_tasks.backends.scene_geometry import bench_geometry
        from robot_tasks.cameras.visual_aids import annotate
        from robot_tasks.grasp_guard import GraspGuard
        from robot_tasks.policies.operating_notes import operating_notes
        from robot_tasks.registry import resolve_task

        _, config = resolve_task("L1-01")
        bench = bench_geometry(config)
        self.assertAlmostEqual(bench.top, .88)
        self.assertAlmostEqual(GraspGuard(config).table_z_m, .88)
        self.assertAlmostEqual(operating_notes(config)["bench"]["surface_world_z_m"], .88)
        projected = []

        def project(points, *args):
            projected.extend(points)
            return np.full((len(points), 2), np.nan)

        calibration = {"T_world_camera_pose_wxyz": (0, 0, 0, 1, 0, 0, 0),
                       "intrinsics_fx_fy_cx_cy": (1, 1, 0, 0), "distortion": (0,) * 5}
        with patch("robot_tasks.cameras.visual_aids.project_world", side_effect=project):
            annotate(Image.new("RGB", (320, 180)), calibration, {}, bench=bench)
        points = np.asarray(projected)
        self.assertGreater(len(points), 0)
        np.testing.assert_allclose(points[:, 2], .881)
        self.assertTrue(np.all(points[:, 1] > .7))

    def test_private_camera_profile_can_be_parsed_offline_but_no_asset_is_fabricated(self):
        from robot_tasks.cameras.profile import load_camera_rig

        root = Path(__file__).resolve().parents[1]
        with patch.object(Path, "is_file", return_value=False):
            rig = load_camera_rig(root / "configs/cameras/r5_tuojing_lab.toml")
            self.assertTrue(all(c.existing_prim_path and c.metadata()["asset_sha256"] is None for c in rig.cameras))
            with self.assertRaisesRegex(ValueError, "does not exist"):
                load_camera_rig(root / "configs/cameras/r5_orbbec_provisional.toml")


@unittest.skipIf(Usd is None, "Requires optional usd-core; actual Isaac acceptance is separate")
class LabBindingTests(unittest.TestCase):
    def setUp(self):
        self.stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(self.stage, "Z")
        UsdGeom.SetStageMetersPerUnit(self.stage, 1.0)
        self.stage.DefinePrim("/World", "Xform")
        UsdPhysics.Scene.Define(self.stage, "/World/PhysicsScene")

    def camera(self):
        parent = "/World/Robots/R5_Left/link6"
        self.stage.DefinePrim(parent, "Xform")
        camera = UsdGeom.Camera.Define(self.stage, parent + "/Wrist/Camera")
        camera.CreateFocalLengthAttr(20)
        camera.CreateHorizontalApertureAttr(32)
        camera.CreateVerticalApertureAttr(18)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 10))
        spec = CameraSpec("left_wrist", "test pinhole", "left", (1280, 720),
                          (800., 800., 640., 360.), (0.,) * 5,
                          (0., 0., 0., 1., 0., 0., 0.), "test", "test", Path("unused.usda"),
                          (0., 0., 0., 1., 0., 0., 0.), "test", (0.01, 10), str(camera.GetPath()))
        return spec, parent

    def test_camera_rejects_stale_intrinsics_mount_and_parent(self):
        from robot_tasks.cameras.isaac_rig import validate_existing_camera

        spec, parent = self.camera()
        self.assertEqual(validate_existing_camera(self.stage, spec, parent), spec.existing_prim_path)
        with self.assertRaisesRegex(ValueError, "intrinsics"):
            validate_existing_camera(self.stage, replace(spec, intrinsics=(801., 800., 640., 360.)), parent)
        with self.assertRaisesRegex(ValueError, "mount"):
            validate_existing_camera(self.stage, replace(spec, optical_pose_wxyz=(0.01, 0, 0, 1, 0, 0, 0)), parent)
        with self.assertRaisesRegex(ValueError, "descend"):
            validate_existing_camera(self.stage, spec, "/World/Robots/R5_Right/link6")
        with self.assertRaisesRegex(ValueError, "aspect"):
            validate_existing_camera(self.stage, replace(spec, resolution=(640, 480)), parent)

    def test_wrist_override_moves_the_assembly_and_keeps_following_link6(self):
        from robot_tasks.backends.tuojing_lab import apply_camera_mounts
        from robot_tasks.cameras.isaac_rig import validate_existing_camera
        from robot_tasks.robots.transforms import pose_transform

        parent = "/World/Robots/R5_Left/link6"
        link = UsdGeom.Xform.Define(self.stage, parent)
        link.AddTranslateOp().Set(Gf.Vec3d(.2, 1.1, .8))
        assembly = UsdGeom.Xform.Define(self.stage, parent + "/WristSensor")
        assembly.AddTranslateOp().Set(Gf.Vec3d(.04, 0, .06))
        housing = UsdGeom.Cube.Define(self.stage, str(assembly.GetPath()) + "/Sensor/Housing")
        camera = UsdGeom.Camera.Define(self.stage, str(assembly.GetPath()) + "/Sensor/Camera")
        camera.CreateFocalLengthAttr(20)
        camera.CreateHorizontalApertureAttr(32)
        camera.CreateVerticalApertureAttr(18)
        camera.CreateClippingRangeAttr(Gf.Vec2f(.01, 10))
        spec = CameraSpec("left_wrist", "test", "left", (1280, 720), (800., 800., 640., 360.),
                          (0.,) * 5, (.097, 0, .068, .6532814824381883, .2705980500730985,
                                      -.2705980500730985, -.6532814824381883),
                          "test", "test", Path("unused.usda"), (0, 0, 0, 1, 0, 0, 0), "test",
                          (.01, 10), str(camera.GetPath()))
        profile = SimpleNamespace(cameras=(spec,))
        original = self.stage.GetRootLayer().ExportToString()
        apply_camera_mounts(self.stage, profile)
        self.assertEqual(validate_existing_camera(self.stage, spec, parent), str(camera.GetPath()))
        self.assertEqual(original, self.stage.GetRootLayer().ExportToString())
        cache = UsdGeom.XformCache()
        np.testing.assert_allclose(np.array(cache.GetLocalToWorldTransform(housing.GetPrim())),
                                   np.array(cache.GetLocalToWorldTransform(camera.GetPrim())), atol=1e-8)
        # Rotating the wrist changes the world view but not the optical hand-eye transform.
        link.AddRotateZOp().Set(35.)
        self.assertEqual(validate_existing_camera(self.stage, spec, parent), str(camera.GetPath()))
        cache = UsdGeom.XformCache()
        actual = np.array(cache.GetLocalToWorldTransform(camera.GetPrim())).T
        base = np.array(cache.GetLocalToWorldTransform(link.GetPrim())).T
        np.testing.assert_allclose(np.linalg.inv(base) @ actual, pose_transform(spec.optical_pose_wxyz), atol=1e-8)
        apply_camera_mounts(self.stage, profile)
        self.assertEqual(validate_existing_camera(self.stage, spec, parent), str(camera.GetPath()))

    def test_overview_override_keeps_housing_aligned_and_tripod_on_floor(self):
        from robot_tasks.backends.tuojing_lab import apply_camera_mounts

        camera = UsdGeom.Camera.Define(self.stage, "/World/Cameras/ThirdPerson")
        camera.AddTranslateOp().Set(Gf.Vec3d(.2, .52, 1.35))
        stand = UsdGeom.Xform.Define(self.stage, "/World/Furniture/ThirdCameraStand")
        head = UsdGeom.Xform.Define(self.stage, str(stand.GetPath()) + "/Head")
        head.AddTranslateOp().Set(Gf.Vec3d(.2, .52, 1.35))
        riser = UsdGeom.Cylinder.Define(self.stage, str(stand.GetPath()) + "/Riser")
        riser.CreateHeightAttr(.075)
        riser.AddTranslateOp().Set(Gf.Vec3d(.2, .48, 1.2825))
        original = self.stage.GetRootLayer().ExportToString()
        spec = SimpleNamespace(name="overview", parent="world", existing_prim_path=str(camera.GetPath()),
                               optical_pose_wxyz=(.2, .48765, 1.32, .9238795325, .3826834324, 0, 0),
                               pose_source="test", resolution=(1280, 960),
                               intrinsics=(1160., 1160., 640., 480.), intrinsics_source="test")
        profile = SimpleNamespace(cameras=(spec,))
        apply_camera_mounts(self.stage, profile)
        self.assertEqual(original, self.stage.GetRootLayer().ExportToString())
        cache = UsdGeom.XformCache()
        np.testing.assert_allclose(np.array(cache.GetLocalToWorldTransform(head.GetPrim())),
                                   np.array(cache.GetLocalToWorldTransform(camera.GetPrim())), atol=1e-8)
        self.assertAlmostEqual(cache.GetLocalToWorldTransform(stand.GetPrim()).ExtractTranslation()[2], 0)
        self.assertAlmostEqual(riser.GetHeightAttr().Get(), .045)
        focal = camera.GetFocalLengthAttr().Get()
        self.assertAlmostEqual(1280 * focal / camera.GetHorizontalApertureAttr().Get(), 1160., places=3)
        self.assertAlmostEqual(960 * focal / camera.GetVerticalApertureAttr().Get(), 1160., places=3)
        apply_camera_mounts(self.stage, profile)
        self.assertAlmostEqual(riser.GetHeightAttr().Get(), .045)

    def test_session_edits_preserve_delivery_and_planner_uses_actual_table(self):
        from robot_tasks.backends.tuojing_lab import configure_stage
        from robot_tasks.planning.scene import collider_meshes

        for path in ("/World/Furniture/Worktable/Top", "/World/Furniture/R5_Left_Mount/Plate"):
            cube = UsdGeom.Cube.Define(self.stage, path)
            cube.CreateSizeAttr(1.)
            cube.AddScaleOp().Set(Gf.Vec3f(1.7, 0.9, 0.025))
            cube.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.78))
            UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        self.stage.DefinePrim("/World/Props", "Xform")
        original = self.stage.GetRootLayer().ExportToString()
        report = configure_stage(self.stage)
        self.assertEqual(original, self.stage.GetRootLayer().ExportToString())
        self.assertFalse(self.stage.GetPrimAtPath("/World/Props").IsActive())
        backend = SimpleNamespace(world=SimpleNamespace(stage=self.stage), planning_legacy_bench=False,
                                  planning_environment_paths=report["planning_environment_paths"])
        meshes = collider_meshes(backend)
        self.assertEqual(meshes["cuboid"], {})
        self.assertEqual(list(meshes["mesh"]), ["_World_Furniture_Worktable_Top"])
        legacy = collider_meshes(SimpleNamespace(world=SimpleNamespace(stage=self.stage)))
        self.assertIn("bench", legacy["cuboid"])

    def test_wrong_robot_base_rejected_instead_of_overwriting_lab(self):
        from robot_tasks.backends.tuojing_lab import validate_robot_binding

        root = UsdGeom.Xform.Define(self.stage, "/World/Robots/R5_Left")
        root.AddTranslateOp().Set(Gf.Vec3d(0.1, 0.2, 0.79))
        mount = SimpleNamespace(name="left", prim_path=str(root.GetPath()),
                                base_pose_wxyz=(0.1, 0.2, 0.79, 1, 0, 0, 0))
        profile = SimpleNamespace(mounts=(mount,))
        validate_robot_binding(self.stage, profile)
        mount.base_pose_wxyz = (0., 0.42, 0.04, 1, 0, 0, 0)
        with self.assertRaisesRegex(ValueError, "base pose"):
            validate_robot_binding(self.stage, profile)

    def test_duplicate_physics_scene_rejected(self):
        from robot_tasks.backends.tuojing_lab import configure_stage

        UsdPhysics.Scene.Define(self.stage, "/physicsScene")
        with self.assertRaisesRegex(ValueError, "physics scenes"):
            configure_stage(self.stage)


if __name__ == "__main__":
    unittest.main()
