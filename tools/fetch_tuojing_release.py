"""Download and install the Git-pinned private asset release using GitHub CLI."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

try:
    from .tuojing_assets import ROOT, install_layout, install_package, relative_path, select_packages, sha256, verify
except ImportError:  # Direct script invocation.
    from tuojing_assets import ROOT, install_layout, install_package, relative_path, select_packages, sha256, verify


def release_packages(root, manifest):
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported release manifest version")
    if manifest["repository"] != "TuojingAI/GPT_Benchmark":
        raise ValueError("Unexpected asset repository")
    if not manifest["tag"].startswith("assets-"):
        raise ValueError("Expected a pinned assets-* release tag")
    lock_path = root / "source.lock.json"
    if sha256(lock_path) != manifest["source_lock_sha256"]:
        raise ValueError("Release manifest does not match the source lock")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    packages = select_packages(lock, [p["id"] for p in manifest["packages"]])
    omitted = set(p["id"] for p in lock["packages"]) - packages.keys()
    if omitted != set(manifest["omitted_packages"]):
        raise ValueError("Release coverage does not match the source lock")
    for entry in manifest["packages"]:
        package = packages[entry["id"]]
        name = relative_path(entry["archive"])
        if (len(name.parts) != 1 or entry["archive"] != package["archive"]
                or entry["sha256"] != package["archive_sha256"]
                or not isinstance(entry["bytes"], int) or entry["bytes"] <= 0):
            raise ValueError(f"Release package differs from source lock: {entry['id']}")
    return packages


def verify_archive(path, entry):
    if (path.is_symlink() or not path.is_file() or path.stat().st_size != entry["bytes"]
            or sha256(path) != entry["sha256"]):
        raise ValueError(f"Archive checksum/size mismatch; preserve and inspect this file: {path}")


def download_archive(cache, entry, manifest, gh):
    target = cache / entry["archive"]
    if target.exists() or target.is_symlink():
        verify_archive(target, entry)
        return target
    cache.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".download-", dir=cache) as directory:
        subprocess.run([gh, "release", "download", manifest["tag"], "--repo",
                        manifest["repository"], "--pattern", entry["archive"],
                        "--dir", directory], check=True)
        downloaded = Path(directory) / entry["archive"]
        verify_archive(downloaded, entry)
        # Exclusive creation also protects concurrent downloads/local modifications.
        try:
            output = target.open("xb")
        except FileExistsError:
            verify_archive(target, entry)
        else:
            try:
                with output, downloaded.open("rb") as source:
                    shutil.copyfileobj(source, output)
            except BaseException:
                target.unlink()
                raise
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT.parents[1] / ".cache/tuojing_assets")
    parser.add_argument("--gh", default="gh", help="GitHub CLI executable (authenticate with gh auth login)")
    parser.add_argument("--usd", action="store_true", help="Also check USD composition; requires pxr")
    args = parser.parse_args(argv)
    manifest = json.loads((ROOT / "release.json").read_text(encoding="utf-8"))
    packages = release_packages(ROOT, manifest)
    for entry in manifest["packages"]:
        download_archive(args.cache, entry, manifest, args.gh)
        install_package(ROOT, args.cache, packages[entry["id"]])
    install_layout(ROOT, packages)
    report = verify(ROOT, usd=args.usd, package_ids=list(packages))
    report["release"] = manifest["tag"]
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
