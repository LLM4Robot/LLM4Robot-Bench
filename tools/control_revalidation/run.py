#!/usr/bin/env python3
"""Opt-in control profile; check is CPU-only and never calls a model."""
import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from arrival import policy_class


def prepare(task_id):
    from robot_tasks.registry import resolve_task
    spec, config = resolve_task(task_id)
    raw = deepcopy(config.raw)
    # Explicit existing robot-control option, never an asset/scoring rewrite.
    raw.setdefault("simulation", {})["robot_gravity_compensation"] = True
    return spec, replace(config, raw=raw)


def inspect_profile(task_id):
    from robot_tasks.policies.model_config import load_gpt_settings
    from robot_tasks.registry import resolve_task
    spec, config = prepare(task_id)
    prepared = spec.factory.prepare_config(config)
    settings = load_gpt_settings(ROOT / "configs/models/gpt.toml")
    policy = policy_class()(prepared, settings, client=object())
    original_spec, original = resolve_task(task_id)
    original = original_spec.factory.prepare_config(original)
    from robot_tasks.policies.gpt import GPTPolicy
    old_policy = GPTPolicy(original, settings, client=object())
    assert prepared.instruction == original.instruction
    assert policy.instructions == old_policy.instructions and policy.schema == old_policy.schema
    return {
        "profile": "r5_precision_arrival_feedback_v1",
        "task": task_id,
        "simulation_started": False,
        "model_calls": 0,
        "robot_gravity_compensation": True,
        "original_robot_gravity_compensation": original.raw.get("simulation", {}).get("robot_gravity_compensation"),
        "native_completion_tolerance_m": prepared.raw["simulation"].get("cartesian_completion_tolerance_m"),
        "planner_position_tolerance_m": 0.0001,
        "planner_endpoint_acceptance_m": 0.00025,
        "arrival_reporting_tolerance_m": 0.0005,
        "stop_on_arrival_error": False,
        "instruction_sha256": hashlib.sha256(prepared.instruction.encode()).hexdigest(),
        "control_instructions_sha256": hashlib.sha256(policy.instructions.encode()).hexdigest(),
        "task_control_schema_unchanged": True,
        "frozen_source_config_sha256": hashlib.sha256(config.source.read_bytes()).hexdigest(),
        "not_a_paper_baseline_run": True,
        "no_successful_grasp_or_task_inferred": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check", "run"])
    parser.add_argument("--task", default="L3-01")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    report = inspect_profile(args.task)
    if args.action == "check":
        print(json.dumps(report, indent=2))
        return 0
    if args.output is None:
        parser.error("run requires a new --output directory")
    if args.gpu < 0:
        parser.error("--gpu must be nonnegative")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        parser.error("Set CUDA_VISIBLE_DEVICES to the selected physical GPU before starting Isaac")
    if not os.environ.get("R5_PRECISION_PYTHON"):
        parser.error("Set R5_PRECISION_PYTHON to the installed cuRobo Python executable")
    used = int(subprocess.check_output([
        "nvidia-smi", "-i", str(args.gpu), "--query-gpu=memory.used",
        "--format=csv,noheader,nounits"], text=True, timeout=15))
    if used >= 512:
        parser.error("Selected GPU is occupied; no existing process was interrupted")
    args.output.mkdir(parents=True, exist_ok=False)
    run_report = {key: value for key, value in report.items()
                  if key not in {"simulation_started", "model_calls"}}
    run_report.update(mode="explicit_model_run", actual_status_and_usage="See the native episode result")
    (args.output / "control-profile.json").write_text(json.dumps(run_report, indent=2) + "\n")
    os.environ["GPT_BENCHMARK_PLANNER_PYTHON"] = str(Path(__file__).with_name("planner_python.py"))
    from robot_tasks.backends.settings import SimulationSettings
    from robot_tasks.execution import POLICIES, _gpt_policy, execute_task
    original_app = SimulationSettings.application_config

    def application_config(self):
        value = original_app(self)
        value.update(active_gpu=args.gpu, physics_gpu=0, multi_gpu=False)
        return value

    def make_policy(config, model_config):
        native = _gpt_policy(config, model_config)
        return policy_class()(config, native.settings, client=native.client)

    SimulationSettings.application_config = application_config
    POLICIES["control_revalidation"] = make_policy
    try:
        spec, config = prepare(args.task)
        directory, result = execute_task(spec, config, args.output,
                                         policy_name="control_revalidation", model_config=args.model_config)
        print(json.dumps({"output": str(directory), **result}, indent=2))
        return 0 if result["status"] == "finished" else 2
    finally:
        SimulationSettings.application_config = original_app
        POLICIES.pop("control_revalidation", None)


if __name__ == "__main__":
    raise SystemExit(main())
