"""Scoring-only observer contact evidence; never exposed as a policy sensor."""

import math
from dataclasses import replace

from ...types import InvalidSimulationState


class ObserverContactMonitor:
    """Confirmed rigid contact fails; ambiguous soft proximity requires review.

    A short solver impulse is recorded but not treated as sustained assistance.
    Geometry proximity is never mislabeled as a measured force/contact.
    """

    def __init__(self):
        self.since = {}
        self.peak = {}
        self.last_time = None
        self.violation = None

    def update(self, now, evidence, assessment):
        if not math.isfinite(now) or (
            self.last_time is not None and now <= self.last_time
        ):
            raise InvalidSimulationState("Observer evidence time did not advance")
        self.last_time = now
        for arm, row in evidence.items():
            mode = row["mode"]
            value = row["max_force_n"] if mode == "rigid_contact" else row["distance_m"]
            if not math.isfinite(value) or value < 0:
                raise InvalidSimulationState("Invalid observer contact measurement")
            active = value > 0.05 if mode == "rigid_contact" else value <= 0.003
            if mode == "rigid_contact":
                self.peak[arm] = max(self.peak.get(arm, 0.0), value)
            if active:
                self.since.setdefault(arm, now)
                if now - self.since[arm] >= 0.25 - 1e-9:
                    self.violation = {
                        "arm": arm,
                        "mode": mode,
                        "evidence": row,
                        "duration_s": now - self.since[arm],
                    }
            else:
                self.since.pop(arm, None)
        metrics = {
            **assessment.metrics,
            "observer_contact": {
                "evidence": evidence,
                "peak_force_n": dict(self.peak),
                "persistence_s": 0.25,
                "violation": self.violation,
                "geometry_is_contact_proof": False,
            },
        }
        if assessment.status == "invalid":
            return replace(assessment, metrics=metrics)
        if self.violation:
            metrics["process_constraints_satisfied"] = False
            if self.violation["mode"] != "rigid_contact":
                metrics["invalid_reason"] = "observer_contact_requires_review"
                return replace(
                    assessment,
                    status="invalid",
                    reason=(
                        "Observer remained near the soft object; contact cannot be certified "
                        "from bounds alone. Review video; exclude from automatic success rate."
                    ),
                    metrics=metrics,
                )
            metrics["failure_category"] = "observer_role_violation"
            metrics["constraint_violations"] = [
                *metrics.get("constraint_violations", []),
                "Observation arm made sustained contact with task props",
            ]
            return replace(
                assessment,
                status="failed",
                reason=(
                    "Observation arm made sustained measured contact with task props"
                ),
                metrics=metrics,
            )
        return replace(assessment, metrics=metrics)
