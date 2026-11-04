#!/usr/bin/env python3
"""Inspect on CPU, or explicitly run the branch control profile under supervision."""
import argparse
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from robot_tasks.control_revalidation.profile import configure, describe, execute_task
from robot_tasks.registry import resolve_task


def bottle_launcher():
    spec = importlib.util.spec_from_file_location("control_bottle_experiment", ROOT / "tools/experiments/bottle_preloosen.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(task_id):
    spec, config = resolve_task(task_id)
    return spec, configure(config)


def inspect_profile(task_id):
    spec, config = resolve_task(task_id)
    return {**describe(spec, config), "simulation_started": False, "model_calls": 0}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check", "run"])
    parser.add_argument("--task", default="L3-01")
    parser.add_argument("--experiment-config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--wall-time-s", type=float)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.experiment_config:
        # Reuse the existing 270-degree launcher, its single engagement context,
        # bridge requirement and process-tree supervisor. Never rebuild those
        # semantics in a second independent launch path.
        bottle = bottle_launcher()
        value = bottle.load_experiment(args.experiment_config)
        if args.task != value["base_task"]:
            parser.error("Experiment task must match --task")
        if args.gpu is not None and args.gpu != value["gpu_index"]:
            parser.error("GPU must match the experiment configuration")
        forwarded = [args.action, "--experiment-config", str(args.experiment_config)]
        for option, value in (("--output", args.output), ("--model-config", args.model_config),
                              ("--max-calls", args.max_calls), ("--wall-time-s", args.wall_time_s)):
            if value is not None:
                forwarded += [option, str(value)]
        if args.worker:
            forwarded.append("--worker")
        return bottle.main(forwarded)
    if args.max_calls is not None and not 1 <= args.max_calls <= 1000:
        parser.error("--max-calls must be in [1, 1000]")
    if args.wall_time_s is not None and (not math.isfinite(args.wall_time_s) or args.wall_time_s < 0):
        parser.error("--wall-time-s must be finite and nonnegative")
    spec, original = resolve_task(args.task)
    config = replace(original, raw=deepcopy(original.raw))
    if args.max_calls is not None:
        config.raw["execution"]["max_decisions"] = args.max_calls
    if args.wall_time_s is not None:
        config.raw["runtime"]["max_wall_time_s"] = args.wall_time_s
        config = replace(config, max_wall_time_s=args.wall_time_s)
    if args.action == "check":
        print(json.dumps({**describe(spec, config, args.model_config), "simulation_started": False, "model_calls": 0}, indent=2))
        return 0
    if args.output is None:
        parser.error("run requires a new --output directory")
    if args.gpu is None or args.gpu < 0:
        parser.error("run requires an explicit nonnegative --gpu")
    if not args.worker:
        from robot_tasks.batch import run_job
        from robot_tasks.runtime_config import load_runtime
        bottle_launcher().require_idle_gpu(args.gpu)
        args.output.mkdir(parents=True, exist_ok=False)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu))
        wall = config.max_wall_time_s
        if wall:
            env["GPT_BENCHMARK_WALL_DEADLINE_MONOTONIC"] = str(time.monotonic() + wall)
        command = [load_runtime().python, str(Path(__file__).resolve()), *argv, "--worker"]
        def record(value):
            (args.output / "worker.json").write_text(json.dumps(value, indent=2) + "\n")
        process = run_job(command, args.output / "process", env, wall + 300 if wall else None, record)
        record(process)
        return process["returncode"] if process["cleanup_complete"] else 2
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.gpu):
        parser.error("Worker CUDA device does not match the supervised selection")
    config.raw.setdefault("simulation", {}).update(active_gpu=args.gpu, physics_gpu=0)
    directory, result = execute_task(spec, config, args.output, model_config=args.model_config)
    print(json.dumps({"output": str(directory), **result}, indent=2))
    return 130 if result["status"] == "cancelled" else (0 if result["status"] == "finished" else 2)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
