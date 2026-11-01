"""Machine paths only; experiment and model settings stay in their existing TOMLs."""
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import tomllib

from .paths import PROJECT_ROOT


@dataclass(frozen=True)
class RuntimeConfig:
    python: str
    cache_dir: Path
    output_dir: Path
    source: Path
    cache_explicit: bool


def load_runtime(local_path: Path | None = None) -> RuntimeConfig:
    selected = local_path or os.environ.get("GPT_BENCHMARK_RUNTIME_CONFIG")
    source = Path(selected).expanduser().resolve() if selected else PROJECT_ROOT / "configs/runtime/local.toml"
    if selected and not source.is_file():
        raise FileNotFoundError(f"Runtime config not found: {source}")
    data = tomllib.loads(source.read_text(encoding="utf-8")) if source.is_file() else {}
    allowed = {"isaacsim": {"python"}, "paths": {"cache_dir", "output_dir"}}
    for section, fields in data.items():
        if section not in allowed or not isinstance(fields, dict) or set(fields) - allowed[section]:
            raise ValueError(f"Unknown runtime settings in {section}; only Python/cache/output paths are supported")
        if any(not isinstance(v, str) or not v.strip() for v in fields.values()):
            raise ValueError(f"Runtime paths in {section} must be non-empty strings")

    def path(value, base):
        value = Path(value).expanduser()
        # Do not dereference interpreter symlinks: venv uses their location.
        return Path(os.path.abspath(base / value))

    python = data.get("isaacsim", {}).get("python", "current")
    base = source.parent
    if "GPT_BENCHMARK_PYTHON" in os.environ:
        python, base = os.environ["GPT_BENCHMARK_PYTHON"], Path.cwd()
        if not python.strip():
            raise ValueError("GPT_BENCHMARK_PYTHON must not be empty")
    python = sys.executable if python == "current" else str(path(python, base))
    paths = data.get("paths", {})
    cache = paths.get("cache_dir", "auto")
    if cache == "auto":
        if sys.platform == "win32":
            configured_cache = os.environ.get("LOCALAPPDATA")
            cache_root = Path(configured_cache) if configured_cache else Path.home() / "AppData/Local"
        else:
            configured_cache = os.environ.get("XDG_CACHE_HOME")
            cache_root = Path(configured_cache) if configured_cache else Path.home() / ".cache"
        cache = cache_root / "GPT_Benchmark"
    else:
        cache = path(cache, source.parent)
    output = path(paths["output_dir"], source.parent) if "output_dir" in paths else PROJECT_ROOT / "runs"
    return RuntimeConfig(python, Path(cache), output, source, "cache_dir" in paths)
