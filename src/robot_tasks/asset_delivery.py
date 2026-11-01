"""Portable access to the version-locked, privately delivered Tuojing assets."""

import json

from .asset_layout import find_bundle, relative_path, verify_files
from .paths import PROJECT_ROOT

ROOT = PROJECT_ROOT / "assets/tuojing"


def verified_package(package_id):
    lock = json.loads((ROOT / "source.lock.json").read_text(encoding="utf-8"))
    package = next(p for p in lock["packages"] if p["id"] == package_id)
    directory = ROOT / "packages" / relative_path(package["directory"])
    verify_files(directory, package["files"])
    return directory, package


def verified_shared(name):
    bundle = find_bundle(ROOT, "common", name)
    if not bundle.destination.exists():
        # Standalone laboratory checks can use the immutable source package
        # when its shared copy has not been installed.
        verify_files(bundle.source, bundle.files)
        return bundle.source / bundle.entry
    verify_files(bundle.destination, bundle.files)
    return bundle.destination / bundle.entry


def verified_prop(name, *, task_id=None):
    """Task execution reads its own copy; source-only tooling can omit task_id."""
    catalog = json.loads((ROOT / "catalog.json").read_text(encoding="utf-8"))
    entry = next((e for e in catalog["entries"] if e["id"] == name and e["role"] == "prop"), None)
    if entry is None:
        raise ValueError(f"Unregistered Tuojing prop: {name}")
    if task_id is None:
        directory, package = verified_package(entry["package"])
        path = directory / relative_path(entry["path"])
        evidence = package["files"][entry["path"]]
    else:
        bundle = find_bundle(ROOT, task_id, name)
        verify_files(bundle.destination, bundle.files)
        path = bundle.destination / bundle.entry
        evidence = bundle.files[bundle.entry.as_posix()]
        lock = json.loads((ROOT / "source.lock.json").read_text(encoding="utf-8"))
        package = next(p for p in lock["packages"] if p["id"] == entry["package"])
    return path, {"delivery": package.get("delivery", "2026-10-05"), "prop": name,
                                       "package": entry["package"], "path": entry["path"],
                                       "task_id": task_id,
                                       "installed_path": path.relative_to(ROOT).as_posix(),
                                       "sha256": evidence["sha256"]}
