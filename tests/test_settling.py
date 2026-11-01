"""Stationary contact jitter must not look like continuous macroscopic motion."""

import unittest

import numpy as np

from robot_tasks.robots.settling import SettlingWindow


class SettlingTests(unittest.TestCase):
    def test_small_contact_jitter_settles_but_continuous_motion_does_not(self):
        still, moving = SettlingWindow(), SettlingWindow()
        for t in np.arange(0.0, 0.501, 1 / 240):
            q = np.r_[np.full(6, 0.0002 * np.sin(t * 600)), np.full(2, 0.02)]
            still.add(t, "left", q)
            r = q + np.r_[np.full(6, t * 0.05), np.full(2, t * 0.01)]
            moving.add(t, "left", r)
        self.assertEqual(still.settled(t, "left", q), (True, True))
        self.assertEqual(moving.settled(t, "left", r), (False, False))

    def test_short_stale_or_discontinuous_measurements_are_not_settled(self):
        window = SettlingWindow()
        q = (0.0,) * 6 + (0.02, 0.02)
        window.add(0.0, "left", q)
        self.assertEqual(window.settled(0.1, "left", q), (False, False))
        for t in np.arange(0.01, 0.31, 0.01):
            window.add(t, "left", q)
        self.assertEqual(window.settled(t, "left", q), (True, True))
        self.assertEqual(window.settled(t + 0.1, "left", q), (False, False))
        self.assertEqual(window.settled(t, "left", (0.1,) * 8), (False, False))


if __name__ == "__main__":
    unittest.main()
