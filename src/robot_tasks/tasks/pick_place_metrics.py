"""Independent measured-state criteria; reaching a commanded TCP is not a grasp."""

import math

from .pick_place_scene import InteractionSettings


def lifted(samples: list[dict], start_z: float, settings: InteractionSettings) -> dict:
    required = settings.lift_threshold_m
    heights = [row["object"]["position_m"][2] - start_z for row in samples]
    bilateral = sum(all(f > settings.contact_threshold_n for f in row["object"]["finger_normal_force_n"]) for row in samples)
    duration = samples[-1]["time_s"] - samples[0]["time_s"] if samples else 0.0
    return {"passed": bool(samples) and min(heights) >= required and bilateral > 0
                and duration >= settings.stability_s - 1e-9,
            "minimum_lift_m": min(heights, default=0.0), "bilateral_contact_samples": bilateral,
            "hold_duration_s": duration}


def placed(samples: list[dict], target, settings: InteractionSettings) -> dict:
    horizontal = [math.dist(row["object"]["position_m"][:2], target[:2]) for row in samples]
    height = [abs(row["object"]["position_m"][2] - settings.cube_size_m / 2) for row in samples]
    speed = [math.sqrt(sum(v*v for v in row["object"]["linear_velocity_m_s"])) for row in samples]
    rotation = [math.sqrt(sum(v*v for v in row["object"]["angular_velocity_rad_s"])) for row in samples]
    contact = [max(row["object"]["finger_normal_force_n"]) for row in samples]
    clearance = [math.dist(row["object"]["position_m"], row["tcp_pose_world_wxyz"][:3]) for row in samples]
    opened = [row["gripper_width_m"] >= settings.open_width_m - settings.gripper_open_tolerance_m for row in samples]
    duration = samples[-1]["time_s"] - samples[0]["time_s"] if samples else 0.0
    return {
        "passed": bool(samples) and duration >= settings.stability_s - 1e-9
            and max(horizontal) <= settings.placement_tolerance_m and max(height) <= settings.placement_height_tolerance_m
            and max(speed) <= settings.placement_speed_m_s and max(rotation) <= settings.placement_angular_speed_rad_s
            and max(contact) < settings.contact_threshold_n
            and min(clearance) > settings.lift_threshold_m and all(opened),
        "stability_duration_s": duration, "max_xy_error_m": max(horizontal, default=math.inf),
        "max_height_error_m": max(height, default=math.inf), "max_speed_m_s": max(speed, default=math.inf),
        "max_angular_speed_rad_s": max(rotation, default=math.inf),
        "max_finger_contact_n": max(contact, default=math.inf),
        "minimum_tcp_clearance_m": min(clearance, default=0.0),
    }
