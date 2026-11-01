"""Offline pause/resume regressions; no Isaac installation or model requests."""

from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from robot_tasks.backends.isaacsim import IsaacSimBackend


class SimulationClockTests(unittest.TestCase):
    def backend(self, *, paused=False, headless=False):
        backend = IsaacSimBackend()
        backend._physics_dt = .01
        backend.settings = SimpleNamespace(headless=headless)
        backend.app = Mock()
        backend.app.is_running.return_value = True
        backend.world = Mock()
        backend.world.current_time = 0.0
        backend.world.is_playing.return_value = not paused
        backend.world.is_stopped.return_value = False
        backend.before_physics_step = Mock()

        def advance(**kwargs):
            self.assertTrue(backend.world.is_playing())
            backend.world.current_time += backend._physics_dt

        backend.world.step.side_effect = advance
        return backend

    def test_gui_pause_waits_then_executes_full_duration(self):
        backend = self.backend(paused=True)
        renders = 0

        def render():
            nonlocal renders
            renders += 1
            self.assertEqual(backend.time_s(), 0.0)
            backend.before_physics_step.assert_not_called()
            backend.world.step.assert_not_called()
            if renders == 3:
                backend.world.is_playing.return_value = True

        backend.world.render.side_effect = render
        output = StringIO()
        with patch("robot_tasks.backends.isaacsim.time.sleep"), redirect_stdout(output):
            backend.step(.02)
        self.assertAlmostEqual(backend.time_s(), .02)
        self.assertEqual(backend.before_physics_step.call_count, 2)
        self.assertEqual(output.getvalue().count("R5_SIMULATION_PAUSED"), 1)
        self.assertIn("R5_SIMULATION_RESUMED", output.getvalue())

    def test_closing_or_stopping_while_paused_interrupts(self):
        for stop in (False, True):
            backend = self.backend(paused=True)
            if stop:
                backend.world.is_stopped.return_value = True
            else:
                backend.app.is_running.return_value = False
            with self.assertRaises(KeyboardInterrupt):
                backend.step(.02)
            backend.before_physics_step.assert_not_called()
            backend.world.step.assert_not_called()

    def test_headless_pause_cannot_spin_or_fake_time(self):
        backend = self.backend(paused=True, headless=True)
        with self.assertRaisesRegex(RuntimeError, "unexpectedly paused"):
            backend.step(.02)
        self.assertEqual(backend.time_s(), 0.0)
        backend.before_physics_step.assert_not_called()

    def test_playing_clock_failure_is_execution_error(self):
        for timestamp in (0.0, -.1, float("nan")):
            backend = self.backend()
            backend.world.step.side_effect = lambda **kw: setattr(backend.world, "current_time", timestamp)
            with self.assertRaisesRegex(RuntimeError, "Physics step did not advance"):
                backend.step(.02)


if __name__ == "__main__":
    unittest.main()
