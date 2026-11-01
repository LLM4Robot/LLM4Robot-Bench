"""Reject false success from contact alone, momentary lift or an unreleased object."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import unittest

from robot_tasks.config import load_config
from robot_tasks.tasks.pick_place_metrics import lifted, placed
from robot_tasks.tasks.pick_place_scene import InteractionSettings


class InteractionMetricsTests(unittest.TestCase):
    def setUp(self):
        self.settings = InteractionSettings()
        self.target = [0.38, 0.27, 0.02]

    def samples(self, *, z=0.02, contact=0.0):
        return [{"time_s": time, "object": {"position_m": [0.38, 0.27, z],
                    "linear_velocity_m_s": [0., 0., 0.], "angular_velocity_rad_s": [0., 0., 0.],
                    "finger_normal_force_n": [contact, contact]},
                 "tcp_pose_world_wxyz": [0.38, 0.27, 0.18, 1., 0., 0., 0.], "gripper_width_m": 0.07}
                for time in (0., 0.5, 1.)]

    def test_lift_requires_sustained_object_height_and_bilateral_contact(self):
        self.assertTrue(lifted(self.samples(z=0.12, contact=2.), 0.02, self.settings)["passed"])
        self.assertFalse(lifted(self.samples(z=0.02, contact=2.), 0.02, self.settings)["passed"])
        self.assertFalse(lifted(self.samples(z=0.12), 0.02, self.settings)["passed"])
        self.assertFalse(lifted(self.samples(z=0.12, contact=2.)[:1], 0.02, self.settings)["passed"])
        slipped = self.samples(z=0.12, contact=2.)
        slipped[-1]["object"]["position_m"][2] = 0.02
        self.assertFalse(lifted(slipped, 0.02, self.settings)["passed"])

    def test_placement_requires_release_support_clearance_and_full_stability_window(self):
        baseline = self.samples()
        self.assertTrue(placed(baseline, self.target, self.settings)["passed"])
        for field, value in (("position_m", [0.38, 0.27, 0.12]),
                             ("position_m", [0.38, 0.42, 0.02]),
                             ("linear_velocity_m_s", [0.1, 0., 0.]),
                             ("angular_velocity_rad_s", [0., 0., 1.]),
                             ("finger_normal_force_n", [2., 2.])):
            with self.subTest(field=field, value=value):
                samples = deepcopy(baseline)
                samples[-1]["object"][field] = value
                self.assertFalse(placed(samples, self.target, self.settings)["passed"])
        samples = deepcopy(baseline)
        samples[-1]["gripper_width_m"] = 0.04
        self.assertFalse(placed(samples, self.target, self.settings)["passed"])
        samples = deepcopy(baseline)
        samples[-1]["tcp_pose_world_wxyz"][:3] = self.target
        self.assertFalse(placed(samples, self.target, self.settings)["passed"])
        self.assertFalse(placed(baseline[:1], self.target, self.settings)["passed"])

    def test_scene_config_rejects_ungraspable_geometry_before_starting_isaac(self):
        config = load_config(Path(__file__).resolve().parents[1] / "configs/checks/gpt_single_arm_pick_place.toml")
        InteractionSettings.from_config(config)
        for invalid in ({"cube_size_m": 0.10}, {"friction": float("nan")},
                        {"transfer_y_m": 0.001}, {"open_width_m": -0.1}, {"extra": 1}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                InteractionSettings.from_config(replace(config, raw={**config.raw, "interaction": invalid}))


if __name__ == "__main__":
    unittest.main()
