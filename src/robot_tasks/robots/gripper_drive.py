"""Symmetric, acceleration-limited simulation drive; no object/contact truth.

Both prismatic fingers receive one common displacement reference. Bounding the
reference's closing lead leaves PD authority to restore symmetry instead of
saturating both motors against an unreachable zero gap. This is software
synchronization, not a claim about the physical R5's mechanical linkage.
"""
from dataclasses import dataclass
import math


@dataclass
class SymmetricGripDrive:
    reference: float
    max_speed: float
    max_acceleration: float
    closing_lead: float
    lower: float
    upper: float
    velocity: float = 0.0
    phase: str = "position"

    def advance(self, goal, measured, dt):
        if not all(math.isfinite(x) for x in (*goal, *measured, dt)) or dt < 0:
            raise ValueError("Nonfinite gripper state or invalid time step")
        if abs(goal[0] - goal[1]) > 1e-6:
            raise ValueError("Symmetric grip drive requires symmetric finger targets")
        # Isaac returns NumPy scalars. Keep the drive state and public targets
        # native Python numbers, including the obstructed/zero-dt branches.
        goal = tuple(float(x) for x in goal)
        measured = tuple(float(x) for x in measured)
        dt = float(dt)
        self.reference, self.velocity = float(self.reference), float(self.velocity)
        target = min(self.upper, max(self.lower, sum(goal) / 2))
        mean = sum(measured) / 2
        bounded = max(target, mean - self.closing_lead)
        bounded = min(self.upper, max(self.lower, bounded))
        self.phase = "bounded_hold" if bounded > target + 1e-5 else "position"
        if dt == 0:
            return (self.reference, self.reference)
        error = bounded - self.reference
        # Braking-distance bound avoids overshoot of a static target.
        speed = min(self.max_speed, math.sqrt(2 * self.max_acceleration * abs(error)))
        desired = math.copysign(speed, error) if error else 0.
        dv = self.max_acceleration * dt
        self.velocity += max(-dv, min(dv, desired - self.velocity))
        delta = self.velocity * dt
        if error * delta >= 0 and abs(delta) >= abs(error):
            self.reference, self.velocity = bounded, 0.
        else:
            self.reference = min(self.upper, max(self.lower, self.reference + delta))
        return (self.reference, self.reference)
