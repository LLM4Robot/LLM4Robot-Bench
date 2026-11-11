#!/usr/bin/env python3
"""Linux PlannerClient interpreter shim; opt-in and no machine paths in Git."""
import os
from pathlib import Path
import sys


def command(argv, python):
    if argv[:3] != ["-u", "-m", "robot_tasks.planning.worker"]:
        raise ValueError("Expected the native PlannerClient worker invocation")
    if not python or not Path(python).is_absolute() or not Path(python).is_file():
        raise ValueError("Set R5_PRECISION_PYTHON to the installed cuRobo Python executable")
    return [python, "-u", str(Path(__file__).with_name("planner_worker.py")), *argv[3:]]


if __name__ == "__main__":
    args = command(sys.argv[1:], os.environ.get("R5_PRECISION_PYTHON"))
    os.execv(args[0], args)
