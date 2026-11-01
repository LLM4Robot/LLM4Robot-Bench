"""Command-line parsing and presentation; task/check dispatch lives in registries."""

import argparse
import json
from pathlib import Path

from .execution import POLICIES, execute_task
from .integrationtest.registry import CHECKS
from .integrationtest.simulation_process import run_check_process
from .policies.responses import ModelError
from .registry import TASKS, resolve_task
from .task_ids import public_task_id
from .robots.audit import audit_robot
from .robots.profile import load_robot_profile


def _print_result(directory: Path, result: dict) -> None:
    print(json.dumps({"output": str(directory), **result}, ensure_ascii=False, indent=2))


def _list(args) -> int:
    if args.checks:
        for spec in CHECKS.values():
            print(f"{spec.name} | {spec.description} | {spec.default_config}")
    else:
        for spec in TASKS.values():
            status = "implemented" if spec.implemented else ("real_only" if spec.evidence == "real_only" else "template_only")
            print(f"{public_task_id(spec.task_id)} | {spec.name} | {status} | {spec.default_config} | {spec.evidence}")
    return 0


def _check(args) -> int:
    _, passed = run_check_process(args.name, args.output.resolve(), config_path=args.config,
                                  gui=args.gui, model_config=args.model_config)
    return 0 if passed else 2


def _validate_robot(args) -> int:
    profile = load_robot_profile(args.profile)
    audit = audit_robot(profile)
    report = {"robot": profile.model, "profile": str(profile.source),
              "available_arms": profile.available_arms, **audit.as_dict()}
    serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0 if (audit.ready if args.require_ready else audit.valid) else 2


def _inspect(args) -> int:
    from .workflow import workflow_metadata
    spec, config = resolve_task(args.task, args.config, depth=args.depth)
    if spec.factory is not None:
        prepare = getattr(spec.factory, "prepare_config", None)
        if prepare is not None:
            config = prepare(config)
    audit = audit_robot(config.robot)
    print(json.dumps({
        "task_id": spec.task_id, "public_task_id": public_task_id(spec.task_id),
        "name": spec.name, "config": str(config.source),
        "default_config": str(spec.default_config),
        "configuration_valid": audit.valid, "task_implemented": spec.implemented,
        "robot": config.robot_model, "available_arms": config.robot.available_arms,
        "active_arms": config.arms, "cameras": config.cameras, "robot_audit": audit.as_dict(),
        "command_arms": config.command_arms, "arm_roles": config.arm_roles,
        "workflow_profile": str(config.workflow_source) if config.workflow_source else None,
        "execution": config.raw.get("execution", {}), "workflow": workflow_metadata(config),
        "reasoning_effort_override": config.raw.get("policy", {}).get("reasoning_effort"),
        "observation_modalities": ["rgb", "depth"] if config.include_depth else ["rgb"],
        "save_depth": config.save_depth,
        "asset_path": str(config.asset_path) if config.asset_path else None,
        "asset_exists": config.asset_path.exists() if config.asset_path else False,
        "evidence": spec.evidence,
    }, ensure_ascii=False, indent=2))
    return 0 if audit.valid else 2


def _run(args) -> int:
    spec, config = resolve_task(args.task, args.config, depth=args.depth)
    directory, result = execute_task(
        spec, config, args.output.resolve(), policy_name=args.policy, model_config=args.model_config,
        gui=args.gui, dry_run=args.dry_run, step_limit=args.steps,
    )
    _print_result(directory, result)
    if result["status"] == "cancelled":
        return 130
    return 0 if result["status"] in {"finished", "dry_run_complete"} else 2


def _task_arguments(parser) -> None:
    parser.add_argument("task", nargs="?", help="注册任务 ID，例如 GPT-PICK-PLACE；只传 --config 也可")
    parser.add_argument("--config", type=Path, help="覆盖该任务的默认实验配置")
    parser.add_argument("--depth", action=argparse.BooleanOptionalAction, default=None,
                        help="开启深度采集、模型深度输入/测距和适用的抓取核验；默认关闭，--no-depth 强制关闭")


def _output_argument(parser) -> None:
    parser.add_argument("--output", type=Path, default=None)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="按名称运行 R5 测评任务或工程检查；list 查看实现状态")
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list", help="列出任务、默认配置与实现状态")
    listing.add_argument("--checks", action="store_true", help="列出工程检查")
    listing.set_defaults(handler=_list)

    check = sub.add_parser("check", help="按名称运行工程检查或单回合任务闭环验证")
    check.add_argument("name", choices=tuple(CHECKS))
    check.add_argument("--config", type=Path, help="覆盖检查的默认配置")
    check.add_argument("--model-config", type=Path, help="覆盖使用模型的检查所引用的模型配置")
    check.add_argument("--gui", action="store_true")
    _output_argument(check)
    check.set_defaults(handler=_check)

    validate = sub.add_parser("validate-robot", help="离线校验 R5 资产与配置，不启动仿真")
    validate.add_argument("--profile", type=Path, required=True)
    validate.add_argument("--require-ready", action="store_true")
    validate.add_argument("--output", type=Path)
    validate.set_defaults(handler=_validate_robot)

    inspect = sub.add_parser("inspect", help="按任务名查看配置与待实现状态，不启动仿真")
    _task_arguments(inspect)
    inspect.set_defaults(handler=_inspect)

    dry = sub.add_parser("dry-run", help="按任务名自检占位流程；不运行物理、不调用模型")
    _task_arguments(dry)
    _output_argument(dry)
    dry.add_argument("--steps", type=int, default=3)
    dry.set_defaults(handler=_run, dry_run=True, gui=False, policy="gpt", model_config=None)

    run = sub.add_parser("run", help="按任务名运行真实仿真；未实现任务在启动前报错")
    _task_arguments(run)
    _output_argument(run)
    run.add_argument("--policy", choices=tuple(POLICIES), default="gpt")
    run.add_argument("--model-config", type=Path)
    run.add_argument("--gui", action="store_true")
    run.set_defaults(handler=_run, dry_run=False, steps=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    import signal
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    previous = signal.signal(signal.SIGBREAK, interrupt) if hasattr(signal, "SIGBREAK") else None
    try:
        return _main(argv)
    except KeyboardInterrupt:
        return 130
    finally:
        if hasattr(signal, "SIGBREAK"):
            signal.signal(signal.SIGBREAK, previous)


def _main(argv: list[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    try:
        if hasattr(args, "output") and args.output is None and args.command != "validate-robot":
            from .runtime_config import load_runtime
            args.output = load_runtime().output_dir
        return args.handler(args)
    except (ValueError, KeyError, OSError, TypeError, ModelError) as exc:
        parser.error(f"Configuration/framework error: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
