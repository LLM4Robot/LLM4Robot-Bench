"""Thread ownership, immutable waiting state and bounded request cancellation."""

import _socket
import socket
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from robot_tasks.backends.isaacsim import IsaacSimBackend
from robot_tasks.policy_worker import PolicyRequestCleanupError, request_action
from robot_tasks.runner import EpisodeLimit
from robot_tasks.types import Action
from robot_tasks.policies.model_config import GPTSettings
from robot_tasks.policies.responses import ResponsesClient


class WaitingPolicy:
    def __init__(self):
        self.release = threading.Event()
        self.exited = threading.Event()
        self.calls = self.cancelled = 0

    def act(self, context):
        self.thread_id = threading.get_ident()
        self.calls += 1
        try:
            if not self.release.wait(3):
                raise TimeoutError("fixture did not receive release")
            return Action((), .1)
        finally:
            self.exited.set()

    def cancel_pending(self):
        self.cancelled += 1
        self.release.set()


class PolicyWorkerTests(unittest.TestCase):
    def backend(self, render):
        backend = IsaacSimBackend()
        backend.app = SimpleNamespace(is_running=lambda: True)
        backend.settings = SimpleNamespace(headless=False)
        backend.world = SimpleNamespace(current_time=12.5, render=render)
        return backend

    def test_model_is_off_thread_and_rendering_stays_on_main_without_physics(self):
        policy = WaitingPolicy()
        main_id, render_ids = threading.get_ident(), []
        def render():
            render_ids.append(threading.get_ident())
            if len(render_ids) == 3:
                policy.release.set()
        backend = self.backend(render)
        context = object()
        action = request_action(policy, context, idle=backend.pump_events, poll_s=.001)
        self.assertEqual(action, Action((), .1))
        self.assertNotEqual(policy.thread_id, main_id)
        self.assertEqual(render_ids, [main_id] * 3)
        self.assertEqual(backend.time_s(), 12.5)
        self.assertEqual(policy.calls, 1)
        self.assertTrue(policy.exited.is_set())

    def test_transport_timeout_is_propagated_once_instead_of_being_polled_as_pending(self):
        policy = SimpleNamespace(act=Mock(side_effect=TimeoutError("provider timeout")))
        with self.assertRaisesRegex(TimeoutError, "provider timeout"):
            request_action(policy, object())
        self.assertEqual(policy.act.call_count, 1)

    def test_window_close_cancels_and_joins_request_before_returning(self):
        policy = WaitingPolicy()
        def closed():
            raise KeyboardInterrupt("window closed")
        with self.assertRaises(KeyboardInterrupt):
            request_action(policy, object(), idle=closed, poll_s=.001)
        self.assertEqual(policy.cancelled, 1)
        self.assertTrue(policy.exited.is_set())

    def test_wall_budget_cancels_request_and_discards_late_action(self):
        policy = WaitingPolicy()
        def budget():
            if policy.calls:
                raise EpisodeLimit("wall_time_limit")
        with self.assertRaisesRegex(EpisodeLimit, "wall_time_limit"):
            request_action(policy, object(), check_budget=budget, poll_s=.001)
        self.assertEqual(policy.calls, 1)
        self.assertEqual(policy.cancelled, 1)
        self.assertTrue(policy.exited.is_set())

    def test_unconfirmed_cleanup_cannot_return_an_action(self):
        policy = WaitingPolicy()
        policy.cancel_pending = lambda: None
        def closed():
            raise KeyboardInterrupt
        try:
            with self.assertRaises(PolicyRequestCleanupError):
                request_action(policy, object(), idle=closed, poll_s=.001, cleanup_timeout_s=.01)
        finally:
            policy.release.set()
            self.assertTrue(policy.exited.wait(1))

    def test_gui_refresh_rejects_simulation_clock_drift(self):
        backend = self.backend(lambda: setattr(backend.world, "current_time", 12.6))
        with self.assertRaisesRegex(RuntimeError, "advanced the simulation clock"):
            backend.pump_events()

    def test_closed_window_is_an_interrupt_and_headless_has_no_rendering(self):
        backend = self.backend(Mock())
        backend.settings.headless = True
        backend.pump_events()
        backend.world.render.assert_not_called()
        backend.app.is_running = lambda: False
        with self.assertRaises(KeyboardInterrupt):
            backend.pump_events()

    def test_http_cancellation_unblocks_socket_without_closing_buffer_on_main_thread(self):
        client = ResponsesClient(GPTSettings(base_url="https://fixture.invalid/v1", model="fixture"))
        sock = Mock()
        client._active_stream = SimpleNamespace(fp=SimpleNamespace(raw=SimpleNamespace(_sock=sock)))
        client.cancel_pending()
        sock.shutdown.assert_called_once_with(socket.SHUT_RDWR)
        self.assertTrue(client._cancel.is_set())
        client.reset_cancellation()
        self.assertFalse(client._cancel.is_set())

    @unittest.skipUnless(
        hasattr(_socket, "socketpair"),
        "Native socketpair required; TCP fallback is forbidden in offline unit tests",
    )
    def test_http_cancellation_releases_a_real_blocked_reader(self):
        client = ResponsesClient(GPTSettings(base_url="https://fixture.invalid/v1", model="fixture"))
        left, right = socket.socketpair()
        reader = left.makefile("rb")
        client._active_stream = SimpleNamespace(fp=reader)
        ready, exited = threading.Event(), threading.Event()
        def act(context):
            ready.set()
            try:
                reader.readline()  # Peer deliberately sends no response.
                return Action((), .1)
            finally:
                exited.set()
        def idle():
            if ready.is_set():
                raise KeyboardInterrupt
        try:
            policy = SimpleNamespace(act=act, cancel_pending=client.cancel_pending)
            with self.assertRaises(KeyboardInterrupt):
                request_action(policy, None, idle=idle, poll_s=.001)
            self.assertTrue(exited.is_set())
        finally:
            right.close()
            reader.close()
            left.close()


if __name__ == "__main__":
    unittest.main()
