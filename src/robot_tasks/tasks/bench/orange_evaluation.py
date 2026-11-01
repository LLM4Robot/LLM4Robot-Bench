"""Score a held lift, return to the original box and release; deformation is diagnostic."""

import math

import numpy as np

from ...types import Assessment, InvalidSimulationState


FINGER_ARM_ORDER = ("left", "right")


def arm_finger_distances(row, arm):
    """Distances remain serialized left/right; select the configured operator."""
    start = 2 * FINGER_ARM_ORDER.index(arm)
    return row["finger_distances_m"][start:start + 2]


def motion_metrics(current, previous, dt):
    """Separate translation from broad shape motion; keep peak jitter diagnostic."""
    velocity = (np.asarray(current) - np.asarray(previous)) / dt
    translation = velocity.mean(axis=0)
    return {
        "speed_m_s": float(np.linalg.norm(velocity, axis=1).max()),
        "centroid_speed_m_s": float(np.linalg.norm(translation)),
        "shape_speed_p95_m_s": float(
            np.quantile(np.linalg.norm(velocity - translation, axis=1), 0.95)
        ),
    }


def deformation_metrics(rest_inverse, current, tetrahedra):
    """Principal stretch is invariant to rigid translation/rotation."""
    p = current[tetrahedra]
    edges = np.stack([p[:, i] - p[:, 0] for i in (1, 2, 3)], axis=2)
    gradient = edges @ rest_inverse
    stretches = np.linalg.svd(gradient, compute_uv=False)
    strain = np.max(np.abs(stretches - 1.0), axis=1)
    return {
        "strain_p95": float(np.quantile(strain, 0.95)),
        "strain_max": float(strain.max()),
        "minimum_volume_ratio": float(np.linalg.det(gradient).min()),
    }


class OrangeEvaluator:
    def __init__(self, spec):
        self.spec = spec
        self.last_time = self.last_assessment = None
        self.lifted = False
        self.initial_centroid_z = None
        self.peak_strain = 0.0
        self.minimum_volume_ratio_seen = 1.0
        self.mesh_inversion_observed = False

    def update(self, now, rows):
        if not math.isfinite(now) or (
            self.last_time is not None and now < self.last_time
        ):
            raise InvalidSimulationState("Invalid or rewound simulation clock")
        # GUI pauses and repeated observations are not new physics samples.
        # Reuse the assessment without inventing elapsed time or task progress.
        if self.last_time == now:
            return self.last_assessment
        self.last_time = now
        p = self.spec.parameters
        row = rows[p.get("soft_object_name", "orange")]
        low, high = np.asarray(row["bounds_min"]), np.asarray(row["bounds_max"])
        checks = [
            *low,
            *high,
            row["speed_m_s"],
            row["centroid_speed_m_s"],
            row["shape_speed_p95_m_s"],
            *row["centroid"],
            *row["collision_bounds_min"],
            *row["collision_bounds_max"],
            row["strain_p95"],
            row["strain_max"],
            row["minimum_volume_ratio"],
            *row["finger_distances_m"],
        ]
        if not np.isfinite(checks).all():
            raise InvalidSimulationState("Nonfinite FEM measurement")
        near = row["finger_distances_m"]
        centroid = np.asarray(row["centroid"])
        if self.initial_centroid_z is None:
            self.initial_centroid_z = float(centroid[2])
        rise = float(centroid[2] - self.initial_centroid_z)
        held = all(d < p["finger_proximity_m"]
                   for d in arm_finger_distances(row, self.spec.arms[0]))
        clear = bool(row["collision_bounds_min"][2] > p["support_z"] + 0.001)
        lifted_now = bool(held and clear and p["lift_m"] - 1e-9 <= rise <= p["max_lift_m"] + 1e-9)
        self.lifted |= lifted_now
        support_xy = np.asarray(p["pedestal_pose"][:2])
        within_box = bool(np.all(np.abs(centroid[:2] - support_xy) <= p["support_half_extents_m"]))
        # The return surface spans the recessed platform and its raised rim.
        on_support = bool(p["support_z"] - 0.003 <= row["collision_bounds_min"][2]
                          <= p.get("support_rim_z", p["table_z"] + 0.100) + 0.003)
        released = all(d >= 0.006 for d in near)
        complete = bool(self.lifted and within_box and on_support and released
                        and row["centroid_speed_m_s"] < 0.05)
        strain = row["strain_p95"]
        self.peak_strain = max(self.peak_strain, strain)
        failure = ""
        if high[2] < p["table_z"] - 0.08:
            failure = "Orange fell below the table"
        self.minimum_volume_ratio_seen = min(
            self.minimum_volume_ratio_seen, row["minimum_volume_ratio"]
        )
        self.mesh_inversion_observed |= row["minimum_volume_ratio"] <= 0
        metrics = {
            "task_version": self.spec.design.version,
            "scoring_protocol": p["scoring_protocol"],
            "mechanism": self.spec.mechanism,
            "limitations": self.spec.limitations,
            "objects": rows,
            "criteria": {
                "lifted_near_left_fingers": self.lifted,
                "held_near_both_left_fingers": held,
                "initial_centroid_z_m": self.initial_centroid_z,
                "centroid_rise_m": rise,
                "required_rise_m": p["lift_m"],
                "clear_of_pedestal": clear,
                "release_required": True,
                "returned_to_original_support": within_box and on_support,
                "released": released,
                "hold_s": 0.0,
                "target_rule": "held 50-200 mm lift, then original box support and release",
                "centroid_speed_m_s": row["centroid_speed_m_s"],
                "shape_speed_p95_m_s": row["shape_speed_p95_m_s"],
                "peak_vertex_speed_m_s": row["speed_m_s"],
                "strain_p95": strain,
                "peak_strain_p95": self.peak_strain,
                "deformation_is_diagnostic_only": True,
                "force_measurement_available": False,
            },
            "stages": {
                "grasped": held,
                "lifted_50mm": self.lifted, "returned_released": complete,
            },
            "goal_complete": complete,
            "process_constraints_satisfied": not bool(failure),
            "simulation_quality": {
                "mesh_inversion_observed": self.mesh_inversion_observed,
                "minimum_volume_ratio_seen": self.minimum_volume_ratio_seen,
                "inversion_is_diagnostic_only": True,
            },
            "quality_warnings": (
                [
                    "Local FEM inversion observed; lift is scored geometrically, "
                    "not as validated material behavior or tissue damage"
                ]
                if self.mesh_inversion_observed
                else []
            ),
        }
        if failure:
            metrics["failure_category"] = "task_failure"
            self.last_assessment = Assessment("failed", failure, metrics)
        elif complete:
            self.last_assessment = Assessment(
                "succeeded",
                "Soft segment lifted 5-20 cm and released on original box",
                metrics,
            )
        else:
            self.last_assessment = Assessment(
                "running",
                "Waiting for a held 5-20 cm lift, return and release on original box",
                metrics,
            )
        return self.last_assessment
