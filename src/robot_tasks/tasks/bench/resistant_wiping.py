"""Contact-distance ink wear for the delivered L2-07 asset only."""
import numpy as np

from .wiping import WipeLedger
from ...robots.transforms import pose_transform


class ResistantWipeLedger(WipeLedger):
    def __init__(self, *args, coverage_required=2.5, motion_deadband_m=0.0005,
                 max_sample_slide_m=0.02, force_deadband_n=0.1, force_half_gain_n=2.0,
                 force_max_gain=1.5, **kwargs):
        super().__init__(*args, **kwargs)
        if not (np.isfinite(coverage_required) and coverage_required > 1):
            raise ValueError("Ink needs more than one contact coverage")
        if not (0 < motion_deadband_m < max_sample_slide_m):
            raise ValueError("Invalid wiping displacement bounds")
        self.coverage_required = coverage_required
        self.motion_deadband_m = motion_deadband_m
        self.max_sample_slide_m = max_sample_slide_m
        self.coverage = np.zeros(len(self.centers))
        if not (np.isfinite(force_deadband_n) and force_deadband_n >= 0
                and np.isfinite(force_half_gain_n) and force_half_gain_n > 0
                and np.isfinite(force_max_gain) and force_max_gain > 0):
            raise ValueError("Invalid normal-force wear parameters")
        self.force_deadband_n = force_deadband_n
        self.force_half_gain_n = force_half_gain_n
        self.force_max_gain = force_max_gain
        self.previous_gain = 0.0
        self.last_normal_force_n = 0.0
        self.last_force_gain = 0.0

    def force_gain(self, normal_force_n):
        """Declared simulation proxy, not a measured real-marker material law."""
        if not np.isfinite(normal_force_n) or normal_force_n <= self.force_deadband_n:
            return 0.0
        ratio = self.force_half_gain_n / (normal_force_n - self.force_deadband_n)
        return float(self.force_max_gain / (1.0 + ratio * ratio))

    @property
    def residue(self):
        return np.clip(1 - self.coverage / self.coverage_required, 0, 1)

    @property
    def opacity(self):
        # A full first pass visibly fades the blue ink; residual ink stays
        # visible until the exact same threshold used for scoring is reached.
        return np.where(self.dirty, np.maximum(0.12, self.residue ** 1.6), 0)

    def update(self, now, eraser_pose, *, pad_contact_n, held):
        if self.last_time is not None and now <= self.last_time:
            return ()
        self.last_time = now
        t = pose_transform(eraser_pose)
        center = t[:3, 3] + self.pad_z * t[:3, 2]
        gain = self.force_gain(pad_contact_n)
        self.last_normal_force_n = float(pad_contact_n) if np.isfinite(pad_contact_n) else 0.0
        self.last_force_gain = gain
        active = (held and gain > 0
                  and np.isfinite(t).all()
                  and float(t[:3, 2] @ self.normal) * self.facing_direction > .5)
        if not active:
            self.previous = None
            self.previous_gain = 0.0
            return ()
        if self.previous is None:
            self.previous = center.copy()
            self.previous_gain = gain
            return ()
        delta = center - self.previous
        planar = delta - self.normal * (delta @ self.normal)
        slide = float(np.linalg.norm(planar))
        # A force spike cannot retroactively credit a low-force segment.
        segment_gain = min(self.previous_gain, gain)
        self.previous_gain = segment_gain
        if slide < self.motion_deadband_m:
            return ()  # Keep anchor: slow real motion accumulates, tiny jitter does not.
        self.previous = center.copy()
        self.previous_gain = gain
        if slide > self.max_sample_slide_m:
            return ()  # Discontinuous relocation is not a wiping stroke.
        local = (self.world - t[:3, 3]) @ t[:3, :3]
        under = ((np.abs(local[:, 0]) <= self.half_extents[0])
                 & (np.abs(local[:, 1]) <= self.half_extents[1])
                 & (np.abs(local[:, 2] - self.pad_z) <= .0015))
        changed = np.flatnonzero(self.dirty & under)
        direction = planar / slide @ t[:3, :3]
        # Normalize by the pad's projected footprint in the stroke direction.
        # One straight traverse contributes at most about one coverage.
        span = 2 * (abs(direction[0]) * self.half_extents[0]
                    + abs(direction[1]) * self.half_extents[1])
        self.coverage[changed] += segment_gain * slide / max(span, self.motion_deadband_m)
        self.dirty = self.residue > 1e-9
        return tuple(map(int, changed))

    def metrics(self):
        return {**super().metrics(),
                "mechanism": "measured normal-force weighted contact sliding with saturating ink wear",
                "ink_remaining_fraction": float(np.average(self.residue, weights=self.weights)),
                "coverage_required": self.coverage_required,
                "pad_normal_force_n": self.last_normal_force_n,
                "force_wear_gain": self.last_force_gain,
                "force_model": {"deadband_n": self.force_deadband_n,
                                "half_gain_above_deadband_n": self.force_half_gain_n,
                                "max_gain": self.force_max_gain,
                                "calibration": "declared simulation proxy; not real-material calibrated"}}
