"""Assemble a registered benchmark task and delegate its lifecycle to runner."""

from dataclasses import replace
from collections.abc import Callable
from pathlib import Path

from .backends.isaacsim import IsaacSimBackend
from .backends.mock import MockBackend, MockRobot, MockTask, NoopPolicy
from .config import ExecutionSettings, TaskConfig
from .paths import PROJECT_ROOT
from .policies.gpt import GPTPolicy
from .policies.model_config import load_gpt_settings, read_api_key
from .registry import TaskSpec
from .robots.adapters import R5Adapter
from .robots.audit import require_cartesian_sim_ready
from .robots.preparation import preparation_action
from .runner import run_episode


def _gpt_policy(config: TaskConfig, model_config: Path | None):
    configured = config.raw.get("policy", {}).get("model_config")
    path = model_config or (config.source.parent / configured if configured else PROJECT_ROOT / "configs/models/gpt.toml")
    settings = load_gpt_settings(path)
    policy_settings = config.raw.get("policy", {})
    if set(policy_settings) - {"model_config", "reasoning_effort"}:
        raise ValueError("policy only supports model_config and reasoning_effort")
    if "reasoning_effort" in policy_settings:
        settings = replace(settings, reasoning_effort=policy_settings["reasoning_effort"])
    decisions = ExecutionSettings.from_config(config).max_decisions
    if decisions is not None:
        settings = replace(settings, max_calls=decisions)
    if settings.transport == "codex_cli":
        from .policies.codex_bridge import create_codex_client
        client = create_codex_client(settings)
        client.require_chatgpt_login()
        return GPTPolicy(config, settings, client=client)
    read_api_key(settings.api_key_env)
    return GPTPolicy(config, settings)


POLICIES = {"gpt": _gpt_policy}


def execute_task(spec: TaskSpec, config: TaskConfig, output_root: Path, *, policy_name: str = "gpt",
                 model_config: Path | None = None, gui: bool = False, dry_run: bool = False,
                 step_limit: int | None = None, run_directory: Path | None = None,
                 summarize: Callable[[dict], dict] | None = None) -> tuple[Path, dict]:
    if config.experiment_id != spec.task_id:
        raise ValueError("Task/config mismatch")
    prepare = getattr(spec.factory, "prepare_config", None)
    if prepare is not None:
        config = prepare(config)
    if dry_run:
        return run_episode(config, MockBackend(), MockRobot(), MockTask(), NoopPolicy(config.control_dt_s),
                           output_root, dry_run=True, step_limit=step_limit)
    if not spec.implemented:
        raise ValueError(f"{spec.task_id} is not implemented: {spec.evidence}. Use dry-run for the scaffold.")
    if policy_name not in POLICIES:
        raise ValueError(f"Unknown policy: {policy_name}")
    require_cartesian_sim_ready(config.robot)
    ExecutionSettings.from_config(config)
    if gui:
        config = replace(config, raw={**config.raw, "simulation": {
            **config.raw.get("simulation", {}), "headless": False,
        }})
    policy = POLICIES[policy_name](config, model_config)
    return run_episode(config, IsaacSimBackend(), R5Adapter(), spec.factory(), policy, output_root,
                       run_directory=run_directory, record_video=config.raw["recording"].get("save_video", False),
                       preparation=getattr(spec.factory, "preparation_action", preparation_action)(config),
                       summarize=summarize)
