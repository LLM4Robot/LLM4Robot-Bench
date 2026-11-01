"""Capture actual startup images, physical reset, planner initialization and scoring; no model calls."""
import argparse
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("task")
p.add_argument("--output", required=True, type=Path)
p.add_argument("--gpu", required=True, type=int)
p.add_argument("--reset-only", action="store_true")
p.add_argument("--plan-probe", action="store_true")
p.add_argument("--config", type=Path, help="Explicit initial-state preview configuration")
a = p.parse_args()

# CUDA_VISIBLE_DEVICES selects PhysX/planner GPU 0, while Vulkan enumerates the
# physical cards. This affects device selection only, never poses or cameras.
from robot_tasks.backends.settings import SimulationSettings

original_application_config = SimulationSettings.application_config


def application_config(self):
    settings = original_application_config(self)
    settings.update(active_gpu=a.gpu, physics_gpu=0, multi_gpu=False)
    return settings


SimulationSettings.application_config = application_config

from robot_tasks.integrationtest.diagnostics.scene_smoke import check


def initialize(backend, task, output):
    hook = getattr(task, "initialize_episode", None)
    if hook is not None:
        hook(backend, None, lambda **kwargs: backend.step(task.config.control_dt_s))
    return {"task_initialization_checked": hook is not None, "gpu": a.gpu}


raise SystemExit(
    check(
        a.task,
        a.output,
        prepare=not a.reset_only,
        capture_hook=initialize,
        capture_before_observations=True,
        plan_probe=a.plan_probe,
        config_path=a.config,
    )
)
