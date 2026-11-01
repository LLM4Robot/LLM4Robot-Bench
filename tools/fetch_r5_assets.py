"""Fetch the pinned manufacturer description; never execute vendor programs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import urllib.request


ROOT = Path(__file__).resolve().parents[1] / "assets" / "r5"


def blob_hash(data: bytes) -> str:
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy", help="Optional HTTP(S) proxy; otherwise use system settings")
    args = parser.parse_args()
    lock = json.loads((ROOT / "source.lock.json").read_text(encoding="utf-8"))
    repository = lock["repository"].removeprefix("https://github.com/")

    def fetch(item: dict) -> dict:
        path = (ROOT / item["path"]).resolve()
        if not path.is_relative_to(ROOT.resolve()):
            raise ValueError("Asset path escapes the R5 directory")
        if path.is_file() and blob_hash(path.read_bytes()) == item["git_blob_sha1"]:
            return {"path": item["path"], "status": "verified"}
        handler = urllib.request.ProxyHandler(
            {"https": args.proxy, "http": args.proxy}
        ) if args.proxy else urllib.request.ProxyHandler()
        opener = urllib.request.build_opener(handler)
        url = f"https://raw.githubusercontent.com/{repository}/{lock['commit']}/{item['source_path']}"
        with opener.open(url, timeout=30) as response:
            data = response.read()
        if blob_hash(data) != item["git_blob_sha1"]:
            raise ValueError(f"Source hash mismatch: {item['path']}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".download")
        temporary.write_bytes(data)
        temporary.replace(path)
        return {"path": item["path"], "status": "downloaded"}

    failed = False
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [(item, pool.submit(fetch, item)) for item in lock["files"]]
        for item, future in futures:
            try:
                result = future.result()
            except Exception as exc:
                failed = True
                result = {"path": item["path"], "status": "failed", "error": str(exc)}
            print(json.dumps(result), flush=True)
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
