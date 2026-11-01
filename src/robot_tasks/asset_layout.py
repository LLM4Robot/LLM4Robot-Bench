"""Task-local copies of immutable deliveries, with original USD paths intact.

No USD imports are needed to install or hash-check a bundle. The manifest keeps
relative dependency trees together; the optional USD audit verifies closure.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile


def relative_path(value, *, allow_root=False):
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or ".." in path.parts
            or "\\" in value or ":" in value
            or (path == PurePosixPath(".") and not allow_root)):
        raise ValueError(f"Unsafe asset path: {value}")
    return Path(*path.parts)


def read_layout(root):
    data = json.loads((root / "task_assets.json").read_text(encoding="utf-8"))
    if data["schema_version"] != 1:
        raise ValueError("Unsupported task asset layout")
    return data


def current_scene_tasks(root):
    manifest=root/'video_20261015.json'
    return set(json.loads(manifest.read_text(encoding='utf-8'))['tasks']) if manifest.exists() else set()


def verify_files(directory, files):
    """Verify original bytes, and reject links escaping the bundle."""
    for name, expected in files.items():
        path = directory / relative_path(name)
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError(f"Asset escapes its bundle: {path}")
        if path.is_symlink() or not path.is_file() or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Missing or modified asset: {path}")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != expected["sha256"]:
            raise ValueError(f"Modified asset: {path}")
    return len(files)


@dataclass(frozen=True)
class Bundle:
    asset: str
    package: str
    source: Path
    destination: Path
    entry: Path
    files: dict


def bundles(root):
    layout = read_layout(root)
    packages = {p["id"]: p for p in json.loads(
        (root / "source.lock.json").read_text(encoding="utf-8"))["packages"]}
    entries = {e["id"]: e for e in json.loads(
        (root / "catalog.json").read_text(encoding="utf-8"))["entries"]}
    groups = {"common": layout["shared"], **{
        task: data["props"] for task, data in layout["tasks"].items()}}
    for group, props in groups.items():
        directory = layout["tasks"].get(group, {}).get("directory", group)
        if len(relative_path(directory).parts) != 1:
            raise ValueError(f"Expected one directory component: {directory}")
        if len(set(props.values())) != len(props):
            raise ValueError(f"Duplicate asset mapping in {group}")
        for folder, asset in props.items():
            for component in (group, folder):
                if len(relative_path(component).parts) != 1:
                    raise ValueError(f"Expected one directory component: {component}")
            entry, spec = entries[asset], layout["bundles"][asset]
            package = packages[entry["package"]]
            base = relative_path(spec["root"], allow_root=True)
            scopes = [relative_path(s, allow_root=True)
                      for s in spec.get("include", [spec["root"]])]
            if any(not scope.is_relative_to(base) for scope in scopes):
                raise ValueError(f"Bundle scope escapes its root: {asset}")
            files = {}
            for name, evidence in package["files"].items():
                path = relative_path(name)
                if any(path.is_relative_to(scope) for scope in scopes):
                    files[path.relative_to(base).as_posix()] = evidence
            local_entry = relative_path(entry["path"]).relative_to(base)
            if local_entry.as_posix() not in files:
                raise ValueError(f"Bundle entry is not pinned: {asset}")
            yield Bundle(asset, package["id"],
                         root / "packages" / relative_path(package["directory"]) / base,
                         root / "source" / directory / folder, local_entry, files)


def find_bundle(root, group, asset):
    directory = read_layout(root)["tasks"].get(group, {}).get("directory", group)
    for item in bundles(root):
        if item.asset == asset and item.destination.parent.name == directory:
            return item
    raise ValueError(f"Asset {asset} is not registered for {group}")


def install_layout(root, package_ids=None):
    """Make independent copies; never overwrite a changed existing bundle."""
    installed = []
    complete=current_scene_tasks(root)
    groups = {data.get("directory", task): task
              for task, data in read_layout(root)["tasks"].items()}
    for item in bundles(root):
        directory = item.destination.parent.name
        group = groups.get(directory, directory)
        if group in complete or (complete and group=='common'):
            continue  # Current configs load complete scenes; never restore older props over them.
        if package_ids is not None and item.package not in package_ids:
            continue
        verify_files(item.source, item.files)
        if item.destination.exists():
            verify_files(item.destination, item.files)
        else:
            item.destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".asset-", dir=item.destination.parent) as temporary:
                staging = Path(temporary) / "bundle"
                staging.mkdir()
                for name in item.files:
                    target = staging / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item.source / name, target)
                verify_files(staging, item.files)
                staging.rename(item.destination)
        installed.append(item.destination.relative_to(root).as_posix())
    # Procedural props have no delivered USD; say so rather than inventing one.
    for task, data in read_layout(root)["tasks"].items():
        if task in complete:
            continue
        for folder, implementation in data.get("generated", {}).items():
            destination = root / "source" / relative_path(data.get("directory", task)) / relative_path(folder)
            destination.mkdir(parents=True, exist_ok=True)
            readme = destination / "README.md"
            if not readme.exists():
                readme.write_text(
                    f"# {task} {data['name']}\n\n"
                    f"此物品由 `{implementation}` 在运行时生成，没有交付 USD 文件。\n"
                    "尺寸、碰撞与摆放以任务实现和配置为准。\n", encoding="utf-8")
    return installed
