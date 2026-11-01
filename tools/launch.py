#!/usr/bin/env python3
"""Cross-platform launcher; bootstrap Python 3.11+, simulator Python from runtime config."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from robot_tasks.runtime_config import load_runtime


def run_windows(command, env, shutdown_timeout):
    import signal
    if os.name == "nt":
        import ctypes
        # PowerShell/WSL may start native commands with Ctrl+C ignored. This
        # console flag is inherited separately from Python's signal handler.
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        if not kernel.SetConsoleCtrlHandler(None, False):
            error = ctypes.get_last_error()
            if error != 6:  # ERROR_INVALID_HANDLE: a noninteractive process has no console.
                raise ctypes.WinError(error)
    process = subprocess.Popen(command, env=env, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    try:
        # Windows' indefinite native wait can defer Python's Ctrl+C handler.
        # Return to Python regularly, even while the worker prints nothing.
        while True:
            try:
                return process.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                continue
    except KeyboardInterrupt:
        # Repeated Ctrl+C must not interrupt bounded cleanup halfway through.
        previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            try:
                process.send_signal(signal.CTRL_BREAK_EVENT)
                # Supervisor may need one taskkill timeout plus one wait timeout.
                process.wait(timeout=2 * shutdown_timeout + 2)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   timeout=shutdown_timeout, check=False)
                except (OSError, subprocess.TimeoutExpired):
                    pass
                try:
                    process.wait(timeout=shutdown_timeout)
                except subprocess.TimeoutExpired:
                    print(f"Cancellation could not confirm process-tree exit; supervisor PID={process.pid}", file=sys.stderr)
            return 130
        finally:
            signal.signal(signal.SIGINT, previous)


def main():
    env = dict(os.environ)
    runtime = load_runtime()
    executable = runtime.python
    if not Path(executable).is_file():
        raise ValueError(f"Configured Python does not exist: {executable}")
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    if os.name == "nt":
        prefix = Path(executable).parent
        env["PATH"] = os.pathsep.join(str(prefix / part) for part in ("", "Scripts", "Library/bin")) + os.pathsep + env.get("PATH", "")
    command = [executable, "-m", "robot_tasks", *sys.argv[1:]]
    if os.name != "nt":
        # Replace the bootstrap: SIGINT reaches the actual supervisor, which owns
        # the isolated worker and its bounded cleanup. No parent kills it early.
        os.execve(executable, command, env)
        return 0  # execve only returns in mocked tests.
    return run_windows(command, env, 5)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as exc:
        print(f"Launch error: {exc}", file=sys.stderr)
        raise SystemExit(2)
