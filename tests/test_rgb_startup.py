"""RGB readiness must never be worked around by advancing physics or fake pixels."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import numpy as np

from robot_tasks.backends.isaacsim import IsaacSimBackend
from robot_tasks.cameras import RGBNotReady


class RGBStartupTests(unittest.TestCase):
    def setUp(self):
        self.backend = IsaacSimBackend()
        self.backend.world = Mock(current_time=3.0)
        self.backend.settings = SimpleNamespace(width=2, height=2)
        self.backend._rep = Mock()
        self.backend._rgb = Mock()

    def test_empty_first_buffer_is_rendered_again_without_physics_steps(self):
        pixels = np.full((2, 2, 4), 123, dtype=np.uint8)
        self.backend._rgb.get_data.side_effect = [np.array([]), pixels]
        frame, = self.backend.capture_rgb()
        self.assertEqual(frame.data, bytes([123] * 12))
        self.assertEqual(frame.timestamp_s, 3.0)
        self.backend.world.step.assert_not_called()
        self.assertEqual(self.backend.world.render.call_count, 2)

    def test_persistent_empty_buffer_fails_with_bounded_retries(self):
        self.backend._rgb.get_data.return_value = np.array([])
        with self.assertRaises(RGBNotReady):
            self.backend.capture_rgb()
        self.assertEqual(self.backend.world.render.call_count, 8)
        self.backend.world.step.assert_not_called()

    def test_malformed_nonempty_image_is_not_treated_as_warmup(self):
        self.backend._rgb.get_data.return_value = np.zeros((2, 2, 2), dtype=np.uint8)
        with self.assertRaisesRegex(RuntimeError, "invalid RGB dimensions"):
            self.backend.capture_rgb()
        self.assertEqual(self.backend.world.render.call_count, 1)

    def test_clock_change_is_rejected_even_during_warmup(self):
        def bad_render():
            self.backend.world.current_time += 0.1
        self.backend.world.render.side_effect = bad_render
        with self.assertRaisesRegex(RuntimeError, "advanced the simulation clock"):
            self.backend.capture_rgb()
        self.backend._rgb.get_data.assert_not_called()


if __name__ == "__main__":
    unittest.main()
