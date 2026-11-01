#!/usr/bin/env python3
"""Plan or run the preserved experiment configurations. Planning never calls GPT."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from robot_tasks.config import ExecutionSettings
from robot_tasks.policies.model_config import load_gpt_settings
from robot_tasks.registry import TASKS, resolve_task
from robot_tasks.task_ids import internal_task_id, public_task_id


def plan(task_ids, model_config):
    model = load_gpt_settings(model_config)
    rows = []
    for name in task_ids:
        spec, config = resolve_task(name)
        execution = ExecutionSettings.from_config(config)
        rows.append({
            "task": public_task_id(spec.task_id),
            "record_id": spec.task_id,
            "config": spec.default_config.relative_to(ROOT).as_posix(),
            "seed": config.seed,
            "max_calls": model.max_calls if execution.max_decisions is None else execution.max_decisions,
            "wall_time_s": config.max_wall_time_s,
            "model_budget_basis": execution.model_budget_basis,
            "reasoning_effort": config.raw.get("policy", {}).get("reasoning_effort") or model.reasoning_effort,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--tasks", nargs="+")
    selection.add_argument("--suite", choices=("original", "variant01", "all"), default="original")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true", help="Run Isaac and the configured model")
    action.add_argument("--dry-run", action="store_true", help="Mock batch lifecycle; no model or Isaac")
    parser.add_argument("--model-config", type=Path, default=ROOT / "configs/models/gpt.toml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    ids = args.tasks or [
        key for key, spec in sorted(TASKS.items()) if key.startswith("L")
        and (args.suite == "all" or bool(spec.base_task_id) == (args.suite == "variant01"))]
    if len(set(map(internal_task_id, ids))) != len(ids):
        parser.error("The same task was selected more than once (possibly by an alias).")
    rows = plan(ids, args.model_config.resolve())
    print(json.dumps(rows, indent=2, ensure_ascii=False), flush=True)
    if not (args.execute or args.dry_run):
        return 0
    # Explicit per-task budgets avoid batch defaults overriding the task TOMLs.
    output = (args.output or ROOT / "runs" /
              datetime.now(timezone.utc).strftime("reproduce_%Y%m%dT%H%M%S.%fZ")).resolve()
    output.mkdir(parents=True, exist_ok=False)
    index = {"schema_version": 1, "mode": "dry_run" if args.dry_run else "model",
             "experiments": rows}
    (output / "experiment_index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    failed = False
    for row in rows:
        command = [sys.executable, str(ROOT / "tools/batch_run.py"),
                   "--tasks", row["task"], "--model-config", str(args.model_config.resolve()),
                   "--max-calls", str(row["max_calls"]), "--wall-time-s", str(row["wall_time_s"]),
                   "--seed", str(row["seed"]), "--reasoning-effort", row["reasoning_effort"],
                   "--output", str(output / row["task"])]
        if args.dry_run:
            command.append("--dry-run")
        code = subprocess.call(command, cwd=ROOT)
        row["returncode"] = code
        (output / "experiment_index.json").write_text(
            json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if code in (130, -2):
            return 130
        failed = failed or code != 0
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
