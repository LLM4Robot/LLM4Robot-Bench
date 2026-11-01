"""Single catalogue used by the CLI, worker and debugger.

Entry points are trusted code registrations, never supplied by TOML. Importing
the catalogue does not import any check implementation or start Isaac Sim.
"""

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

from ..config import TaskConfig, load_config
from ..paths import PROJECT_ROOT
from ..robots.audit import require_cartesian_sim_ready
from ..robots.profile import RobotProfile


@dataclass(frozen=True)
class CheckSpec:
    name: str
    description: str
    task_ids: tuple[str, ...]
    default_config: Path
    entry_point: str
    output_prefix: str
    preflight: Callable[[RobotProfile], object] | None
    debug_config: Path | None = None
    uses_model: bool = False
    uses_simulation: bool = True

    def load_config(self, path: Path | None = None, *, debug: bool = False) -> TaskConfig:
        default = (self.debug_config or self.default_config) if debug else self.default_config
        config = load_config(path if path is not None else default)
        if config.task_id not in self.task_ids:
            raise ValueError(f"Check/config mismatch: {self.name} expects {self.task_ids}, got {config.task_id}")
        return config

    def validate_options(self, *, gui: bool = False, model_config: Path | None = None) -> None:
        if gui and not self.uses_simulation:
            raise ValueError(f"{self.name}: --gui requires a simulation check")
        if model_config is not None and not self.uses_model:
            raise ValueError(f"{self.name}: --model-config is not used by this check")

    def run(self, config_path: Path, output_root: Path, *, gui: bool = False,
            model_config: Path | None = None, run_directory: Path | None = None) -> tuple[Path, dict]:
        self.validate_options(gui=gui, model_config=model_config)
        config = self.load_config(config_path)
        if self.preflight is not None:
            self.preflight(config.robot)
        module, function = self.entry_point.split(":")
        execute = getattr(import_module(f"{__package__}.{module}"), function)
        options = {"gui": gui, "run_directory": run_directory}
        if model_config is not None:
            options["model_path"] = model_config
        return execute(config.source, output_root, **options)


CHECKS = {
    "gpt-pick-place": CheckSpec(
        "gpt-pick-place", "GPT 单臂抓放任务的全链路检查", ("GPT-PICK-PLACE",),
        PROJECT_ROOT / "configs/checks/gpt_single_arm_pick_place.toml",
        "gpt_pick_place_check:run_gpt_pick_place_check", "R5_gpt_pick_place_",
        require_cartesian_sim_ready, uses_model=True,
    ),
}


def get_check(name: str) -> CheckSpec:
    try:
        return CHECKS[name]
    except KeyError:
        raise ValueError(f"Unknown check: {name}; available: {', '.join(CHECKS)}") from None
