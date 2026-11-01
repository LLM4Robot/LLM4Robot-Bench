"""Install and verify the pinned lab delivery without executing vendor scripts."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1] / "assets" / "tuojing"
sys.path.insert(0, str(ROOT.parents[1] / "src"))
from robot_tasks.asset_layout import bundles, install_layout, current_scene_tasks, verify_files as verify_bundle_files


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relative_path(value):
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or ".." in path.parts
            or "\\" in value or ":" in value or path == PurePosixPath(".")):
        raise ValueError(f"Unsafe asset path: {value}")
    return Path(*path.parts)


def verify_files(directory, package):
    for relative, expected in package["files"].items():
        path = directory / relative_path(relative)
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError(f"Asset escapes its package: {relative}")
        if (not path.is_file() or path.is_symlink()
                or path.stat().st_size != expected["bytes"]
                or sha256(path) != expected["sha256"]):
            raise ValueError(f"Missing or modified delivery file: {path}")
    return len(package["files"])


def install_package(root, archives, package):
    destination = root / "packages" / relative_path(package["directory"])
    previous = root / "source" / relative_path(package["directory"])
    if previous.exists() and not destination.exists():
        verify_files(previous, package)
        destination.parent.mkdir(parents=True, exist_ok=True)
        previous.rename(destination)
    if destination.exists():
        verify_files(destination, package)
        return
    # Different delivery wrappers may contain exactly the same pinned files.
    # Fall back only when an archive is absent, never after a checksum failure.
    wrapper = package
    archive = archives / relative_path(wrapper["archive"])
    if not archive.exists():
        for candidate in package.get("archive_alternatives", []):
            alternative = archives / relative_path(candidate["archive"])
            if alternative.exists():
                wrapper, archive = candidate, alternative
                break
    if sha256(archive) != wrapper["archive_sha256"]:
        raise ValueError(f"Archive checksum mismatch: {archive}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".install-", dir=destination.parent))
    try:
        with zipfile.ZipFile(archive) as source:
            # Some deliveries place assets/ directly at ZIP root. Only an
            # explicit empty root enables this; member safety checks still apply.
            prefix = (relative_path(wrapper["archive_root"])
                      if wrapper["archive_root"] else Path("."))
            members = {}
            for member in source.infolist():
                path = relative_path(member.filename)
                if not path.is_relative_to(prefix):
                    raise ValueError(f"Unexpected archive root: {member.filename}")
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError(f"Archive symlink is not supported: {member.filename}")
                if member.is_dir():
                    continue
                relative = path.relative_to(prefix).as_posix()
                if relative in members:
                    raise ValueError(f"Duplicate archive member: {relative}")
                members[relative] = member
            if set(members) != set(package["files"]):
                raise ValueError("Archive file inventory differs from the source lock")
            for relative, member in members.items():
                target = temporary / relative_path(relative)
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
        verify_files(temporary, package)
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def verify_usd(path, source_root, require_component=False):
    """Static composition/dependency audit. Does not certify physics or rendering."""
    from pxr import Usd, UsdGeom, UsdUtils

    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(path))
    # This shader ships with Isaac; plain usd-core does not register its resolver.
    builtins = {"OmniPBR.mdl", "OmniGlass.mdl", "OmniSurface.mdl"}
    missing = sorted(set(unresolved) - builtins)
    if missing:
        raise ValueError(f"Unresolved USD dependencies in {path.name}: {missing}")
    for dependency in [layer.realPath for layer in layers] + list(assets):
        if dependency in builtins:
            continue
        if not Path(dependency).resolve().is_relative_to(source_root.resolve()):
            raise ValueError(f"USD dependency outside the delivered packages: {dependency}")
    # Inspect authored paths, including texture paths inside binary USD layers.
    for layer in layers:
        for reference in re.findall(r"@([^@]+)@", layer.ExportToString()):
            if (reference.startswith(("/", "\\")) or ":" in reference):
                raise ValueError(f"Nonportable authored USD path: {reference}")
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"Unable to open USD entry: {path}")
    default_prim = stage.GetDefaultPrim()
    up_axis = str(UsdGeom.GetStageUpAxis(stage))
    metres = UsdGeom.GetStageMetersPerUnit(stage)
    if require_component:
        if not default_prim:
            raise ValueError(f"Referenceable asset needs a default prim: {path}")
        if up_axis != "Z" or metres != 1:
            raise ValueError(f"Expected metre units and Z-up: {path}")
    return {"layers": len(layers), "assets": len(assets),
            "isaac_builtin_dependencies": sorted(set(unresolved) & builtins),
            "prims": sum(1 for _ in stage.Traverse()),
            "default_prim": str(default_prim.GetPath()) if default_prim else None,
            "up_axis": up_axis, "meters_per_unit": metres,
            "notes": [] if default_prim else ["No defaultPrim; select an explicit prim when referencing this scene."]}


def select_packages(lock, package_ids=None):
    packages = {item["id"]: item for item in lock["packages"]}
    if package_ids is None:
        return packages
    if not package_ids or len(package_ids) != len(set(package_ids)):
        raise ValueError("Select at least one package, without duplicate IDs")
    unknown = set(package_ids) - packages.keys()
    if unknown:
        raise ValueError(f"Unknown package IDs: {sorted(unknown)}")
    return {key: packages[key] for key in package_ids}


def organize(root, package_ids=None):
    """Migrate complete old deliveries, then populate task-local copies."""
    lock = json.loads((root / "source.lock.json").read_text(encoding="utf-8"))
    packages = select_packages(lock, package_ids)
    for package in packages.values():
        old = root / "source" / relative_path(package["directory"])
        destination = root / "packages" / relative_path(package["directory"])
        if old.exists():
            verify_files(old, package)
            if destination.exists():
                raise ValueError(f"Both old and new deliveries exist; preserve and inspect: {old}, {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            old.rename(destination)
        verify_files(destination, package)
    return install_layout(root, packages)


def verify_complete_scenes(root,usd=False):
    manifest=json.loads((root/'video_20261015.json').read_text(encoding='utf-8'))
    tasks={}
    for tid,row in manifest['tasks'].items():
        files={row['path']:{'bytes':row['bytes'],'sha256':row['sha256']},
               row['runtime_path']:{'bytes':row['runtime_bytes'],'sha256':row['runtime_sha256']}}
        verify_bundle_files(root,files)
        tasks[tid]={'files':len(files),'raw_sha256':row['sha256'],'runtime_sha256':row['runtime_sha256'],
            'usd':verify_usd(root/relative_path(row['runtime_path']),root,require_component=True) if usd else 'not_checked'}
    textures=manifest.get('visual_textures',{})
    if any(row.get('visual_overlay',{}).get('applied') for row in manifest['tasks'].values()) and not textures:
        raise ValueError('Current PR visual textures are not locked')
    verify_bundle_files(root,textures)
    layout_path=root/'task_assets.json'
    legacy=set(json.loads(layout_path.read_text(encoding='utf-8'))['tasks']) if layout_path.exists() else set()
    return {'delivery':manifest['delivery'],'task_assets':tasks,
            'visual_textures_verified':len(manifest.get('visual_textures',{})),
            'not_in_current_delivery':sorted(legacy-tasks.keys()),
            'legacy_task_copies_used':False,'usd_composition_checked':usd,
            'physics_tested':False,'rendering_tested':False,'task_workflow_integrated':False}


def verify(root, usd=False, package_ids=None):
    complete=current_scene_tasks(root)
    if complete and package_ids is None:
        return verify_complete_scenes(root,usd)
    lock = json.loads((root / "source.lock.json").read_text(encoding="utf-8"))
    packages = select_packages(lock, package_ids)
    counts = {key: verify_files(root / "packages" / relative_path(p["directory"]), p)
              for key, p in packages.items()}
    catalog = json.loads((root / "catalog.json").read_text(encoding="utf-8"))
    entries = {}
    for entry in catalog["entries"]:
        if entry["package"] not in packages:
            continue
        package = packages[entry["package"]]
        relative = relative_path(entry["path"])
        if relative.as_posix() not in package["files"]:
            raise ValueError(f"Catalog entry is not pinned: {entry['id']}")
        path = root / "packages" / relative_path(package["directory"]) / relative
        entries[entry["id"]] = verify_usd(
            path, root / "packages", require_component=entry["role"] in
            {"prop", "fragment_template", "environment"}) if usd else "file_verified"
    task_assets = {}
    if (root / "task_assets.json").is_file():
        for item in bundles(root):
            if item.destination.parent.name in complete or (complete and item.destination.parent.name=='common'):
                continue
            if item.package not in packages:
                continue
            count = verify_bundle_files(item.destination, item.files)
            task_assets[item.destination.relative_to(root).as_posix()] = {
                "asset": item.asset, "files": count, "entry": item.entry.as_posix(),
                "usd": verify_usd(item.destination / item.entry, item.destination,
                                  require_component=True) if usd else "not_checked"}
    return {"delivery": lock["delivery"], "verified_files": counts, "entries": entries,
            "task_assets": task_assets,
            "omitted_packages": [p["id"] for p in lock["packages"] if p["id"] not in packages],
            "usd_composition_checked": usd, "physics_tested": False,
            "rendering_tested": False, "task_workflow_integrated": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, help="Install from a directory containing the selected pinned ZIPs")
    parser.add_argument("--packages", nargs="+", help="Only install/verify these package IDs; default: all locked packages")
    parser.add_argument("--usd", action="store_true", help="Also compose USDs; requires pxr (usd-core or Isaac Python)")
    parser.add_argument("--organize", action="store_true", help="Migrate old source/package folders and install task-local copies")
    parser.add_argument("--report", type=Path, help="Optional JSON verification report")
    args = parser.parse_args(argv)
    if args.archives is not None:
        lock = json.loads((ROOT / "source.lock.json").read_text(encoding="utf-8"))
        for package in select_packages(lock, args.packages).values():
            install_package(ROOT, args.archives, package)
        install_layout(ROOT, args.packages)
    if args.organize:
        organize(ROOT, args.packages)
    result = verify(ROOT, usd=args.usd, package_ids=args.packages)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
