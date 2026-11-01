"""Validate and retime planned joint paths before applying any drive target."""

from dataclasses import dataclass

import numpy as np

from ..robots.control import bounded_measured_joints


@dataclass(frozen=True)
class JointPath:
    positions: tuple[tuple[float, ...], ...]
    times_s: tuple[float, ...]

    @classmethod
    def from_timed(cls, positions, times_s, start, limits):
        """Validate the worker's checked time law without silently retiming it."""
        # Reuse bounds/start validation, but keep the submitted samples/times.
        cls.from_waypoints(positions, start, limits)
        q, times = np.asarray(positions, float), np.asarray(times_s, float)
        if (
            times.shape != (len(q),)
            or not np.isfinite(times).all()
            or times[0] != 0
            or np.any(np.diff(times) <= 0)
        ):
            raise ValueError("Invalid planner trajectory timestamps")
        if len(q) > 1:
            speed = np.abs(np.diff(q, axis=0)) / np.diff(times)[:, None]
            if np.any(speed > np.array([x.velocity for x in limits]) + 1e-7):
                raise ValueError("Timed planner trajectory exceeds joint speed limits")
            if np.max(np.abs(np.diff(q, axis=0))) > 0.01000001:
                raise ValueError("Timed planner trajectory exceeds joint step limit")
        if np.max(np.abs(q[0] - bounded_measured_joints(start, limits))) > 1e-7:
            raise ValueError(
                "Timed path must start at the bounded measured configuration"
            )
        return cls(tuple(tuple(float(x) for x in row) for row in q), tuple(times))

    def stretched(self, duration_s):
        """Slow a trajectory to a common duration; never speed it up."""
        if not np.isfinite(duration_s) or duration_s < self.duration_s:
            raise ValueError("Synchronized duration cannot shorten a path")
        if self.duration_s == 0:
            if duration_s == 0:
                return self
            return JointPath(
                (self.positions[0], self.positions[0]), (0.0, float(duration_s))
            )
        return JointPath(
            self.positions,
            tuple(t * duration_s / self.duration_s for t in self.times_s),
        )

    @classmethod
    def from_waypoints(cls, positions, start, limits, *, max_step_rad=0.01):
        if not np.isfinite(max_step_rad) or max_step_rad <= 0:
            raise ValueError("Maximum joint increment must be finite and positive")
        q = np.asarray(positions, dtype=float)
        if (
            q.ndim != 2
            or q.shape[1] != len(limits)
            or not len(q)
            or not np.isfinite(q).all()
        ):
            raise ValueError("Planner returned an invalid joint trajectory")
        low = np.array([x.lower for x in limits])
        high = np.array([x.upper for x in limits])
        speed = np.array([x.velocity for x in limits])
        if np.any(q < low) or np.any(q > high) or np.any(speed <= 0):
            raise ValueError("Planner trajectory exceeds configured joint limits")
        start = np.asarray(start, dtype=float)
        if (
            start.shape != q.shape[1:]
            or not np.isfinite(start).all()
            or np.max(np.abs(q[0] - start)) > 0.02
        ):
            raise ValueError(
                "Planner path does not start at the measured configuration"
            )
        start = np.asarray(bounded_measured_joints(start, limits))
        # Preserve its polyline; never interpolate straight from start to final IK.
        points = [start]
        times = [0.0]
        for target in q[1:]:
            delta = target - points[-1]
            count = max(1, int(np.ceil(np.max(np.abs(delta)) / max_step_rad)))
            duration = max(float(np.max(np.abs(delta) / speed)), 1e-4)
            origin, t0 = points[-1], times[-1]
            for i in range(1, count + 1):
                points.append(origin + delta * i / count)
                times.append(t0 + duration * i / count)
        return cls(tuple(tuple(float(x) for x in p) for p in points), tuple(times))

    @property
    def duration_s(self):
        return self.times_s[-1]

    def sample(self, time_s):
        q = np.asarray(self.positions)
        return tuple(
            float(np.interp(time_s, self.times_s, q[:, j])) for j in range(q.shape[1])
        )
