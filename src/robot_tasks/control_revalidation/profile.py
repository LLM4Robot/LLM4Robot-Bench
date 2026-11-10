"""Branch-default R5 control adapter around the preserved episode engine.

No imports of Isaac, CUDA or FCL, and no provider calls during inspection.
Effective control changes are recorded separately from the frozen paper tasks.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import tomllib
from uuid import uuid4

from .. import execution as native
from ..config import ExecutionSettings
from ..paths import PROJECT_ROOT
from ..policies.gpt import GPTPolicy
from ..policies.model_config import load_gpt_settings
from .arrival import policy_class
from ..runtime_performance import describe as runtime_description

PROFILE = "r5_precision_arrival_feedback_v2"
WORKER_ROOT = PROJECT_ROOT / "tools/control_revalidation"


def configure(config):
    raw = deepcopy(config.raw)
    raw.setdefault("simulation", {})["robot_gravity_compensation"] = True
    return replace(config, raw=raw)


def describe(spec, config, model_config=None):
    prepared = spec.factory.prepare_config(configure(config))
    original = spec.factory.prepare_config(config)
    configured = config.raw.get("policy", {}).get("model_config")
    path = model_config or (config.source.parent / configured if configured
                            else PROJECT_ROOT / "configs/models/gpt.toml")
    settings = load_gpt_settings(path)
    policy = policy_class()(prepared, settings, client=object())
    baseline = GPTPolicy(original, settings, client=object())
    if (prepared.instruction != original.instruction or policy.instructions != baseline.instructions
            or policy.schema != baseline.schema):
        raise ValueError("Control profile changed task instructions or action schema")
    execution = ExecutionSettings.from_config(prepared)
    files = (sorted(WORKER_ROOT.glob("*.py")) + [WORKER_ROOT / "requirements.txt"]
             + sorted(Path(__file__).parent.glob("*.py"))
             + sorted((Path(__file__).parent.parent / "runtime_performance").glob("*.py")))
    initial = prepared.raw.get("task", {}).get("experimental_initial_state")
    return {
        "profile": PROFILE,
        "task": spec.task_id,
        "robot_gravity_compensation": True,
        "original_robot_gravity_compensation": original.raw.get("simulation", {}).get("robot_gravity_compensation"),
        "native_completion_tolerance_m": prepared.raw["simulation"].get("cartesian_completion_tolerance_m"),
        "planner_position_tolerance_m": 0.0001,
        "planner_endpoint_acceptance_m": 0.00025,
        "arrival_reporting_tolerance_m": 0.0005,
        "stop_on_arrival_error": False,
        "max_calls": settings.max_calls if execution.max_decisions is None else execution.max_decisions,
        "max_wall_time_s": prepared.max_wall_time_s,
        "model_budget_basis": execution.model_budget_basis,
        "experimental_initial_state": initial,
        "instruction_sha256": hashlib.sha256(prepared.instruction.encode()).hexdigest(),
        "control_instructions_sha256": hashlib.sha256(policy.instructions.encode()).hexdigest(),
        "task_control_schema_unchanged": True,
        "frozen_source_config_sha256": hashlib.sha256(config.source.read_bytes()).hexdigest(),
        "profile_source_sha256": {p.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "world_mesh_guard": "solid convex enclosure; concavities may be conservatively rejected",
        "not_a_paper_baseline_run": True,
        "no_successful_grasp_or_task_inferred": True,
        "runtime_performance": runtime_description(),
    }


def planner_python():
    """Reuse the configured cuRobo interpreter without resolving venv symlinks."""
    selected = os.environ.get("R5_PRECISION_PYTHON") or os.environ.get("GPT_BENCHMARK_PLANNER_PYTHON")
    if not selected:
        local = PROJECT_ROOT / "configs/runtime/planner.local.toml"
        if local.is_file():
            selected = tomllib.loads(local.read_text()).get("python")
    if not selected or not Path(selected).is_absolute() or not Path(selected).is_file():
        raise ValueError("Configure the installed cuRobo interpreter via R5_PRECISION_PYTHON or GPT_BENCHMARK_PLANNER_PYTHON")
    if Path(selected).resolve() == (WORKER_ROOT / "planner_python.py").resolve():
        raise ValueError("R5_PRECISION_PYTHON must be the real cuRobo interpreter, not the shim")
    if not os.access(selected, os.X_OK):
        raise ValueError("Configured cuRobo interpreter is not executable")
    return selected


@contextmanager
def activated():
    if not sys.platform.startswith("linux"):
        raise ValueError("Control revalidation currently requires a Linux simulation/planner host")
    python = planner_python()
    keys = ("R5_PRECISION_PYTHON", "GPT_BENCHMARK_PLANNER_PYTHON")
    saved = {key: os.environ.get(key) for key in keys}
    old_policy = native.POLICIES["gpt"]
    old_runner = native.run_episode

    def make_policy(config, model_config):
        policy = native._gpt_policy(config, model_config)
        return policy_class()(config, policy.settings, client=policy.client)

    try:
        os.environ["R5_PRECISION_PYTHON"] = python
        os.environ["GPT_BENCHMARK_PLANNER_PYTHON"] = str(WORKER_ROOT / "planner_python.py")
        native.POLICIES["gpt"] = make_policy
        from ..runtime_performance.runner import run_episode
        native.run_episode = run_episode
        yield
    finally:
        native.POLICIES["gpt"] = old_policy
        native.run_episode = old_runner
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def execute_task(spec, config, output_root, *, policy_name="gpt", model_config=None, **kwargs):
    # Offline lifecycle and caller-supplied policies stay independent of the
    # experimental model adapter (and never need a planner runtime).
    if kwargs.get("dry_run") or policy_name != "gpt":
        return native.execute_task(spec, config, output_root, policy_name=policy_name,
                                   model_config=model_config, **kwargs)
    if config.experiment_id != spec.task_id:
        raise ValueError("Task/config mismatch")
    report = describe(spec, config, model_config)
    report.update(actual_status_and_usage="See the native episode result",
                  shared_wall_deadline_monotonic=os.environ.get("GPT_BENCHMARK_WALL_DEADLINE_MONOTONIC"))
    with activated():
        directory = kwargs.get("run_directory")
        if directory is None:
            directory = Path(output_root) / f"{spec.task_id}_control_{uuid4().hex[:12]}"
            kwargs["run_directory"] = directory
        directory = Path(directory)
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("Control revalidation requires a fresh episode directory")
        # The native recorder requires an empty episode directory, and Isaac
        # shutdown can end this process before execute_task returns. Save a
        # sidecar *before* startup and embed the same evidence in the effective
        # config which the native recorder writes before starting Kit.
        reports = Path(output_root) / "control-profiles"
        reports.mkdir(parents=True, exist_ok=True)
        with (reports / f"{directory.name}.json").open("x") as stream:
            stream.write(json.dumps(report, indent=2) + "\n")
        effective = configure(config)
        effective.raw["control_revalidation"] = report
        return native.execute_task(spec, effective, output_root, policy_name=policy_name,
                                   model_config=model_config, **kwargs)
