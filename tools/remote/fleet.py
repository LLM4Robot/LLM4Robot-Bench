#!/usr/bin/env python3
"""Compatibility entrypoint; optional deployment lives in the remote skill."""
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[2]
                      / "skills/r5-remote-deploy/scripts/fleet.py"), run_name="__main__")
