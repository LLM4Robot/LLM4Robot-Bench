"""Restore pinned room textures and R5 visual assets without executing vendor code."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def fetch(root, proxy=None):
    root = Path(root).resolve()
    lock = json.loads((root / "source.lock.json").read_text(encoding="utf-8"))
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        if proxy else urllib.request.ProxyHandler()
    )
    for item in lock["files"]:
        target = (root / item["path"]).resolve()
        if not target.is_relative_to(root):
            raise ValueError("Asset path escapes asset directory")
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == item["sha256"]:
            print("verified", item["path"], flush=True)
            continue
        with opener.open(item["url"], timeout=180) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError(f"Appearance asset checksum mismatch: {item['path']}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".download")
        try:
            temporary.write_bytes(data)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        print("downloaded", item["path"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy", help="Optional HTTP(S) proxy URL")
    args = parser.parse_args()
    fetch(Path(__file__).resolve().parents[1] / "assets/appearance", args.proxy)
