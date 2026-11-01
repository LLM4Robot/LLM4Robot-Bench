"""Measured position-window settling, robust to instantaneous contact-velocity noise."""

from collections import deque

import numpy as np


class SettlingWindow:
    def __init__(self, duration_s=0.25):
        self.duration_s = duration_s
        self.history = {}

    def add(self, time_s, arm, positions):
        history = self.history.setdefault(arm, deque())
        history.append((time_s, tuple(positions)))
        # Retain one point at/before the start of the complete time window.
        while len(history) > 2 and history[1][0] <= time_s - self.duration_s:
            history.popleft()

    def settled(self, time_s, arm, positions):
        history = self.history.get(arm, ())
        if (
            not history
            or time_s - history[0][0] < self.duration_s - 1e-9
            or time_s - history[-1][0] > 0.02
        ):
            return False, False
        q = np.asarray([v for _, v in history] + [tuple(positions)])
        if q.shape[1:] != (8,) or not np.isfinite(q).all():
            return False, False
        # Same speed references as the instantaneous check, integrated over .25 s.
        span = np.ptp(q, axis=0)
        return bool(np.all(span[:6] < 0.02 * self.duration_s)), bool(
            np.all(span[6:] < 0.003 * self.duration_s)
        )
