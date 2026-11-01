#!/usr/bin/env python3
"""Install an asset archive after verifying EVERY member against runtime.lock."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def install(archive, root=ROOT):
    expected = json.loads((root / "assets/runtime.lock.json").read_text(encoding="utf-8"))["files"]
    # Stage and verify the whole delivery before replacing any runtime input.
    with tempfile.TemporaryDirectory(prefix="r5-assets-") as tmp, tarfile.open(archive, "r:*") as bundle:
        seen = set()
        for member in bundle:
            if member.name not in expected or member.name in seen or not member.isfile():
                raise ValueError(f"Unexpected or unsafe archive member: {member.name}")
            destination = Path(tmp) / member.name
            if not destination.resolve().is_relative_to(Path(tmp).resolve()):
                raise ValueError("Archive escapes staging directory")
            if member.size != expected[member.name]["bytes"]:
                raise ValueError(f"Wrong asset size: {member.name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with bundle.extractfile(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            sha = hashlib.sha256(destination.read_bytes()).hexdigest()
            if sha != expected[member.name]["sha256"]:
                raise ValueError(f"Wrong asset hash: {member.name}")
            seen.add(member.name)
        if seen != set(expected):
            raise ValueError(f"Missing {len(set(expected) - seen)} assets")
        for name in sorted(seen):
            target = root / name
            if not target.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"Asset destination escapes checkout: {name}")
        for name in sorted(seen):
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(tmp) / name, target)
    return len(seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    print(f"Installed {install(args.archive)} verified assets")


if __name__ == "__main__":
    main()
