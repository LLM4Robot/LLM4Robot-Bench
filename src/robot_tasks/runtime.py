"""Portable runtime storage with optional Windows storage configuration."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import tempfile

from .paths import PROJECT_ROOT


STORAGE_CONFIG = PROJECT_ROOT / "configs/runtime/windows.json"


def _modern_storage_selected():
    from .runtime_config import load_runtime
    return load_runtime().cache_explicit


def snapshot_runtime_config(output: Path) -> None:
    from .runtime_config import load_runtime
    settings = load_runtime()
    report = {"platform": sys.platform, "python": sys.executable,
              "configured_python": settings.python, "cache_dir": str(settings.cache_dir),
              "output_dir": str(settings.output_dir), "config_source": str(settings.source)}
    legacy = sys.platform == "win32" and STORAGE_CONFIG.is_file() and not settings.cache_explicit
    report["storage_mode"] = "legacy_windows" if legacy else "portable"
    if legacy:
        report["cache_dir"] = None  # Legacy manifest supplies several independent cache paths.
        report["storage_config_source"] = str(STORAGE_CONFIG)
    (output / "runtime_config.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def snapshot_storage_config(output: Path) -> None:
    """Linux runs do not depend on a private Windows drive configuration."""
    target = output / "storage_config.json"
    if sys.platform == "win32" and STORAGE_CONFIG.is_file() and not _modern_storage_selected():
        target.write_bytes(STORAGE_CONFIG.read_bytes())
    else:
        target.write_text(json.dumps({"platform": sys.platform,
                                      "windows_storage_configured": False}, indent=2), encoding="utf-8")


@dataclass(frozen=True)
class RuntimeStorage:
    root: Path
    kit_root: Path
    environment: dict[str, str]
    legacy: bool = True

    def kit_arguments(self) -> list[str]:
        return ["--portable-root", str(self.kit_root)]


def configure_windows_storage() -> RuntimeStorage | None:
    """Configure before importing native libraries/debugpy or starting Kit."""
    if sys.platform != "win32" or not STORAGE_CONFIG.is_file() or _modern_storage_selected():
        from .runtime_config import load_runtime
        root = load_runtime().cache_dir
        kit = root / "kit"
        warp = root / "warp"
        kit.mkdir(parents=True, exist_ok=True)
        warp.mkdir(parents=True, exist_ok=True)
        environment = {"WARP_CACHE_PATH": str(warp)}
        os.environ.update(environment)
        return RuntimeStorage(root, kit, environment, legacy=False)
    data = json.loads(STORAGE_CONFIG.read_text(encoding="utf-8"))
    root = Path(data["root"]).resolve()

    def directory(relative: str) -> Path:
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Runtime path escapes configured root: {relative}")
        path.mkdir(parents=True, exist_ok=True)
        return path

    forbidden = {"HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "PATH", "CODEX_HOME"}
    if data["environment"].keys() & forbidden:
        raise ValueError("Storage configuration must not replace identity or system search paths")
    environment = {name: str(directory(relative)) for name, relative in data["environment"].items()}
    os.environ.update(environment)
    # These may already have been cached by Python before main() was entered.
    tempfile.tempdir = environment["TEMP"]
    sys.pycache_prefix = environment["PYTHONPYCACHEPREFIX"]
    return RuntimeStorage(root, directory(data["kit_root"]), environment)


def storage_report(storage: RuntimeStorage | None, *, kit_started: bool = False) -> dict:
    """Report resolved paths, including junction targets, rather than C: aliases."""
    if storage is None:
        return {"platform": sys.platform, "windows_storage_configured": False}
    paths = {"project": str(PROJECT_ROOT.resolve()), "python": str(Path(sys.executable).resolve()),
             "tempfile": str(Path(tempfile.gettempdir()).resolve()), "kit_root": str(storage.kit_root),
             **{name: str(Path(value).resolve()) for name, value in storage.environment.items()}}
    if not storage.legacy:
        return {"platform": sys.platform, "windows_storage_configured": False,
                "runtime_root": str(storage.root), "paths": paths}
    data = json.loads(STORAGE_CONFIG.read_text(encoding="utf-8"))
    for entry in data["junctions"]:
        alias = Path(os.environ[entry["base"]]) / entry["source"]
        paths[entry["source"]] = str(alias.resolve())
    if kit_started:
        import carb.tokens
        import warp
        tokens = carb.tokens.get_tokens_interface()
        for name in ("cache", "data", "logs"):
            resolved = tokens.resolve("${" + name + "}")
            if "${" in resolved:
                raise RuntimeError(f"Kit storage token was not resolved: {name}")
            paths[f"kit_{name}"] = str(Path(resolved).resolve())
        if warp.config.kernel_cache_dir is not None:
            paths["actual_warp_cache"] = str(Path(warp.config.kernel_cache_dir).resolve())
    configured_drive = storage.root.drive.upper()
    off_drive = {name: value for name, value in paths.items()
                 if Path(value).drive.upper() != configured_drive}
    exceptions = {entry["source"]: {"resolved_path": str((Path(os.environ[entry["base"]]) / entry["source"]).resolve()),
                                    "reason": entry["reason"]}
                  for entry in data.get("shared_system_cache_exceptions", [])}
    return {"platform": sys.platform, "windows_storage_configured": True,
            "runtime_root": str(storage.root), "paths": paths,
            "configured_drive": configured_drive,
            "all_research_paths_on_configured_drive": not off_drive,
            "off_drive_research_paths": off_drive, "shared_system_cache_exceptions": exceptions}


if __name__ == "__main__":
    configured = configure_windows_storage()
    print(json.dumps(storage_report(configured), indent=2, ensure_ascii=False))
