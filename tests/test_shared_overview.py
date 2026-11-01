"""Shared look-at settings reach actual sensor poses in both scene frames."""

import unittest
import tomllib
from pathlib import Path
import numpy as np
from robot_tasks.backends.settings import SimulationSettings
from robot_tasks.cameras.overview import override_overview
from robot_tasks.cameras.profile import load_camera_rig
from robot_tasks.config import load_config
from robot_tasks.robots.transforms import pose_transform

ROOT = Path(__file__).resolve().parents[1]


class SharedOverviewTests(unittest.TestCase):
    def test_all_task_rigs_use_complete_video_framing_and_preserve_wrists(self):
        checked = 0
        for path in (ROOT / "configs/tasks").glob("*.toml"):
            if not tomllib.loads(path.read_text(encoding="utf-8"))["task"].get("simulation_enabled"):
                continue
            cfg = load_config(path)
            with self.subTest(task=cfg.task_id):
                local = tomllib.loads(path.read_text(encoding="utf-8")).get("simulation", {})
                self.assertFalse(set(local) & {"camera_position", "camera_target", "camera_resolution", "camera_intrinsics"})
                settings = SimulationSettings.from_config(cfg)
                rig = load_camera_rig(path.parent / settings.camera_profile)
                updated = override_overview(rig, settings)
                new = updated.cameras[0]
                eye = np.asarray(settings.camera_position)
                direction = np.asarray(settings.camera_target) - eye
                transform = pose_transform(new.optical_pose_wxyz)
                np.testing.assert_allclose(transform[:3, 3], eye)
                np.testing.assert_allclose(
                    -transform[:3, 2], direction / np.linalg.norm(direction), atol=1e-10
                )
                self.assertEqual(rig.cameras[1:], updated.cameras[1:])
                self.assertEqual(new.intrinsics, (1160., 1160., 640., 480.))
                self.assertEqual(new.resolution, (1280, 960))
                if settings.environment == "tuojing_lab_v4":
                    np.testing.assert_allclose(
                        eye, [0.1956512155, 0.55, 1.45]
                    )
                    np.testing.assert_allclose(
                        -transform[:3, 2], [0, np.cos(np.deg2rad(36)), -np.sin(np.deg2rad(36))], atol=1e-9
                    )
                else:
                    self.assertAlmostEqual(eye[0], 1.187650417122201)
                    self.assertAlmostEqual(eye[2], 0.67)
                    self.assertAlmostEqual(settings.camera_target[2], 0)
                checked += 1
        self.assertGreaterEqual(checked, 16)

    def test_legacy_rig_also_uses_shared_optics(self):
        from robot_tasks.registry import resolve_task
        _, cfg = resolve_task("GPT-PICK-PLACE")
        settings = SimulationSettings.from_config(cfg)
        rig = load_camera_rig(cfg.source.parent / settings.camera_profile)
        updated = override_overview(rig, settings)
        self.assertNotEqual(rig.cameras[0].intrinsics, updated.cameras[0].intrinsics)
        self.assertEqual(updated.cameras[0].resolution, (1280, 960))
        self.assertEqual(updated.cameras[0].intrinsics, (1160., 1160., 640., 480.))
        self.assertEqual(rig.cameras[1:], updated.cameras[1:])

    def test_partial_or_invalid_shared_optics_rejected(self):
        from types import SimpleNamespace
        base = {"camera_position": [0, 0, 1], "camera_target": [0, 1, 0],
                "camera_profile": "rig.toml", "camera_resolution": [1280, 960],
                "camera_intrinsics": [1160, 1160, 640, 480]}
        cases = [{k: v for k, v in base.items() if k != "camera_intrinsics"},
                 {**base, "camera_intrinsics": [0, 1160, 640, 480]},
                 {**base, "camera_resolution": [1280, 0]}]
        for data in cases:
            with self.subTest(data=data), self.assertRaises(ValueError):
                SimulationSettings.from_config(SimpleNamespace(source=ROOT / "task.toml", raw={"simulation": data}))

    def test_invalid_look_at_rejected(self):
        from types import SimpleNamespace
        from robot_tasks.cameras.overview import resolve_view

        with self.assertRaisesRegex(ValueError, "must differ"):
            resolve_view(
                SimpleNamespace(camera_position=(0, 0, 0), camera_target=(0, 0, 0)),
                None,
            )


class TaskOverviewLayoutTests(unittest.TestCase):
    def test_right_cup_and_drawer_match_rendered_overview_side_and_arm(self):
        from robot_tasks.registry import resolve_task
        from robot_tasks.tasks.bench.spec import build_spec

        for task in ("L2-03", "L2-05"):
            with self.subTest(task=task):
                entry, cfg = resolve_task(task)
                entry.factory.prepare_config(cfg)
                spec = build_spec(task)
                settings = SimulationSettings.from_config(cfg)
                eye = np.asarray(settings.camera_position)
                forward = np.asarray(settings.camera_target) - eye
                forward /= np.linalg.norm(forward)
                right = np.cross(forward, [0, 0, 1])
                right /= np.linalg.norm(right)

                def screen_x(point):
                    relative = np.asarray(point) - eye
                    return (relative @ right) / (relative @ forward)

                mounts = {m.name: m for m in cfg.robot.mounts}
                self.assertLess(
                    screen_x(mounts["left"].base_pose_wxyz[:3]),
                    screen_x(mounts["right"].base_pose_wxyz[:3]),
                )
                self.assertEqual(cfg.arms, ("right",))
                self.assertEqual(cfg.observation_arms, ("left",))
                primary = spec.design.articles[0].initial[:3]
                if task == "L2-03":
                    receiver = spec.design.articles[1].initial[:3]
                    self.assertGreater(screen_x(primary), screen_x(receiver))
                else:
                    distance = {
                        a: np.linalg.norm(
                            np.asarray(primary[:2]) - m.base_pose_wxyz[:2]
                        )
                        for a, m in mounts.items()
                    }
                    self.assertLess(distance["right"], distance["left"])
