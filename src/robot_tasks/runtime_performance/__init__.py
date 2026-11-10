"""Reversible single-episode runtime optimizations; no task/protocol overrides."""

import os


def mode():
    value = os.environ.get("GPT_BENCHMARK_RUNTIME_MODE", "optimized")
    if value not in {"off", "timing", "optimized"}:
        raise ValueError("GPT_BENCHMARK_RUNTIME_MODE must be off, timing, or optimized")
    return value


def describe():
    selected = mode()
    return {
        "profile": "single_episode_runtime_v1", "mode": selected,
        "timing": selected != "off",
        "background_video": selected == "optimized",
        "video_queue_frames_per_camera": 4 if selected == "optimized" else 0,
        "joint_read_cache_scope": "one before_physics_step callback" if selected == "optimized" else None,
        "identical_image_encoding_cache": selected == "optimized",
        "camera_render_cache": False,
        "model_history_truncated": False,
        "model_settings_changed": False,
        "physics_and_evaluation_changed": False,
    }
