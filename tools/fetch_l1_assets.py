"""Download pinned L1 assets; no vendor code is installed or executed."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request


def fetch(root: Path, proxy=None):
    lock = json.loads((root / "source.lock.json").read_text(encoding="utf-8"))
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": proxy, "https": proxy})
        if proxy
        else urllib.request.ProxyHandler()
    )
    for item in lock["files"]:
        for name in ("metadata.json", "object.usdz"):
            relative = item["path"] + "/" + name
            target = (root / "source" / relative).resolve()
            if not target.is_relative_to((root / "source").resolve()):
                raise ValueError("Asset path escapes cache")
            expected = item[name]["sha256"]
            if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
                print("verified", relative, flush=True)
                continue
            url = f"{lock['repository']}/resolve/{lock['revision']}/{lock['prefix']}/{relative}"
            with opener.open(url, timeout=90) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError(f"Asset checksum mismatch: {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + ".download")
            temporary.write_bytes(data)
            temporary.replace(target)
            print("downloaded", relative, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy")
    args = parser.parse_args()
    fetch(Path(__file__).resolve().parents[1] / "assets/l1", args.proxy)
