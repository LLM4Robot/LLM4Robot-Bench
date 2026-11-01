#!/usr/bin/env python3
"""Bootstrap the serial benchmark without needing an activated Isaac environment."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from robot_tasks.batch import main

if __name__ == "__main__":
    raise SystemExit(main())
