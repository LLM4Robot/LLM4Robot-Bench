#!/usr/bin/env python3
"""Build a source delivery and complete asset delivery; never include Git history."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
TOP_FILES = {
    "README.md", "LICENSE", "NOTICE", "CITATION.bib", "AGENTS.md",
    "pyproject.toml", "environment.yml", ".gitignore", ".gitattributes", ".gitmodules",
}
TOP_DIRS = {"src", "configs", "tests", "tools", "skills", "docs"}
PRIVATE_DIRS = {"__pycache__", ".git", ".venv", "venv", ".conda", ".cache", "node_modules"}


def source_files():
    paths = []
    candidates = [ROOT / name for name in TOP_FILES]
    for directory in sorted(TOP_DIRS | {"docs/release"}):
        candidates.extend((ROOT / directory).rglob("*"))
    for path in candidates:
        rel = path.relative_to(ROOT)
        if not path.is_file() or path.is_symlink():
            continue
        if rel.parts[0] not in TOP_DIRS and rel.as_posix() not in TOP_FILES and not rel.as_posix().startswith("docs/release/"):
            continue
        if (PRIVATE_DIRS.intersection(rel.parts)
                or any(part.endswith((".egg-info", ".dist-info")) for part in rel.parts)
                or path.suffix in {".pyc", ".log", ".tmp", ".pem", ".key", ".p12", ".pfx"}
                or (path.name.startswith(".env") and path.name != ".env.example")
                or path.name in {"auth.json", "credentials.json", "AGENTS.local.md"}
                or ".local." in path.name or rel.as_posix() in
                {"configs/runtime/windows.json", "configs/runtime/local.toml"}):
            continue
        paths.append(rel.as_posix())
    paths.append("assets/runtime.lock.json")
    return sorted(set(paths))


def pack(path, names):
    with tarfile.open(path, "w:gz", compresslevel=6) as bundle:
        for name in names:
            data = (ROOT / name).read_bytes()
            info = tarfile.TarInfo(name)
            info.size, info.mtime = len(data), 0
            info.mode = 0o755 if name.endswith(".sh") else 0o644
            bundle.addfile(info, io.BytesIO(data))
    return {"file": path.name, "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "files": len(names)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    if not (ROOT / "docs/media/paper/figure1.png").is_file():
        parser.error("The complete source delivery includes website media. Run: "
                     "git submodule update --init docs/media")
    subprocess.run([sys.executable, str(ROOT / "tools/verify_runtime_assets.py")], check=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    assets = json.loads((ROOT / "assets/runtime.lock.json").read_text(encoding="utf-8"))["files"]
    source = source_files()
    artifact = {
        "schema_version": 1,
        "source": pack(output / "r5-benchmark-source.tar.gz", source),
        "assets": pack(output / "r5-benchmark-assets.tar.gz", sorted(assets)),
        "asset_lock_sha256": hashlib.sha256((ROOT / "assets/runtime.lock.json").read_bytes()).hexdigest(),
        "source_files": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in source},
    }
    (output / "release.json").write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    for row in (artifact["source"], artifact["assets"]):
        print(f"{row['sha256']}  {row['file']}  ({row['bytes'] / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
