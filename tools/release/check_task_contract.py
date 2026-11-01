#!/usr/bin/env python3
"""Check the frozen task semantics without Isaac, networking, or model calls."""
import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from robot_tasks.backends.settings import SimulationSettings
from robot_tasks.config import ExecutionSettings
from robot_tasks.initial_layout import scene_record
from robot_tasks.policies.gpt import GPTPolicy
from robot_tasks.policies.model_config import load_gpt_settings
from robot_tasks.registry import TASKS, resolve_task
from robot_tasks.tasks.bench.arm_selection import resolve_bench_spec

LOCK = ROOT / "configs/benchmark/task_contract.json"
FLOAT_TOKEN = re.compile(r"(?<![\w.])[-+]?(?:\d+\.\d+(?:[eE][+-]?\d+)?|\d+[eE][+-]?\d+)(?![\w.])")


def numeric_token(value):
    return "0" if abs(value) < 1e-12 else format(value, ".12g")


def normalized(value):
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(k): normalized(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalized(v) for v in value]
    if isinstance(value, Path):
        if value.is_relative_to(ROOT):
            return "<PROJECT>/" + value.relative_to(ROOT).as_posix()
        return value.as_posix()
    if isinstance(value, str):
        result = value.replace(str(ROOT), "<PROJECT>")
        result = result.replace("\\", "/") if "<PROJECT>" in result else result
        # Derived reach bounds and geometry appear inside policy JSON text.
        # Python/NumPy builds differ in the last ~1e-16. This affects the
        # comparison only, never the actual prompt or simulation values.
        return FLOAT_TOKEN.sub(lambda match: numeric_token(float(match[0])), result)
    if hasattr(value, "tolist"):
        return normalized(value.tolist())
    if isinstance(value, float):
        return float(numeric_token(value))
    return value


def digest(value):
    return hashlib.sha256(json.dumps(normalized(value), sort_keys=True,
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def snapshot():
    settings = load_gpt_settings(ROOT / "configs/models/gpt.toml")
    tasks = {}
    for key in sorted(k for k in TASKS if k.startswith("L")):
        spec, config = resolve_task(key)
        prepared = spec.factory.prepare_config(config)
        simulation = asdict(SimulationSettings.from_config(prepared))
        # Platform defaults and generated cache locations are not scene parameters.
        for name in ("asset_directory", "graphics_backend"):
            simulation.pop(name)
        components = {
            "configuration": config.raw,
            "instruction": prepared.instruction,
            "robot": prepared.robot,
            "simulation": simulation,
            "execution": ExecutionSettings.from_config(prepared),
            "system_instructions": GPTPolicy(prepared, settings, client=object()).instructions,
            "initial_layout": scene_record(config.raw["task"]),
            "geometry_and_scoring": (
                resolve_bench_spec(config.raw["task"], config.robot)
                if config.raw["task"]["implementation"] == "household"
                else prepared.raw),
        }
        tasks[key] = {name: digest(value) for name, value in components.items()}
    protected = {}
    for directory in ("src/robot_tasks/tasks", "src/robot_tasks/robots",
                      "src/robot_tasks/backends", "src/robot_tasks/planning",
                      "src/robot_tasks/policies", "src/robot_tasks/cameras",
                      "configs/tasks", "configs/robots", "configs/cameras",
                      "configs/workflows", "configs/task_references"):
        for path in sorted((ROOT / directory).rglob("*")):
            if (path.is_file() and "__pycache__" not in path.parts
                    and path.suffix in {".py", ".json", ".toml"}
                    and ".local." not in path.name):
                protected[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("src/robot_tasks/initial_layout.py", "src/robot_tasks/config.py",
                 "src/robot_tasks/execution.py", "src/robot_tasks/runner.py",
                 "src/robot_tasks/video_appearance.py", "src/robot_tasks/video_delivery.py",
                 "src/robot_tasks/task_ids.py", "src/robot_tasks/task_numbering.py",
                 "src/robot_tasks/asset_layout.py", "configs/benchmark/task_numbering.json",
                 "assets/runtime.lock.json",
                 "configs/models/gpt.toml"):
        protected[name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    return {"schema_version": 2, "numeric_comparison": "12 significant digits; magnitudes below 1e-12 treated as zero; source files exact SHA-256",
            "tasks": tasks, "protected_files": protected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-lock", action="store_true",
                        help="Intentional protocol change only; never use to hide a regression")
    args = parser.parse_args()
    actual = snapshot()
    if args.write_lock:
        LOCK.parent.mkdir(parents=True, exist_ok=True)
        LOCK.write_text(json.dumps(actual, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    expected = json.loads(LOCK.read_text(encoding="utf-8"))
    if expected["schema_version"] != actual["schema_version"]:
        raise SystemExit("Unsupported task contract schema")
    changes = []
    for section in ("tasks", "protected_files"):
        for name in sorted(set(expected[section]) | set(actual[section])):
            if expected[section].get(name) != actual[section].get(name):
                changes.append(f"{section}/{name}")
    if changes:
        raise SystemExit("Task contract changed:\n" + "\n".join(changes))
    print(f"Task contract verified: {len(actual['tasks'])} tasks; "
          f"{len(actual['protected_files'])} protected files; no model calls")


if __name__ == "__main__":
    main()
