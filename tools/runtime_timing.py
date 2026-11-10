#!/usr/bin/env python3
"""Read single-episode timings; does not start a simulator or contact a model."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Episode directory or its timing.json")
    args = parser.parse_args()
    path = args.path / "timing.json" if args.path.is_dir() else args.path
    value = json.loads(path.read_text(encoding="utf-8"))
    print(f"Mode: {value['profile']['mode']} | Measured elapsed: {value['elapsed_s']:.2f}s")
    print("Phase                          Calls    Inclusive(s)    Exclusive(s)      Max(s)")
    for name, row in sorted(value["phases"].items(), key=lambda item: -item[1]["exclusive_s"]):
        print(f"{name:30} {row['calls']:5d} {row['inclusive_s']:15.3f} {row['exclusive_s']:15.3f} {row['max_s']:11.3f}")
    print("\nCounters:", json.dumps(value["counters"], sort_keys=True))
    print(value["accounting"])
    print(value["scope"])


if __name__ == "__main__":
    main()
