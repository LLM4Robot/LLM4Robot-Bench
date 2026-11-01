"""Run a registered check in an isolated process with an optional wall-time watchdog."""

from datetime import datetime, timezone
from contextlib import contextmanager
import json
import os
from pathlib import Path
from queue import Empty, Queue
import signal
import subprocess
import sys
import threading
import time
from uuid import uuid4

from ..paths import PROJECT_ROOT
from ..recording import write_json
from .registry import get_check


@contextmanager
def _uninterrupted_cleanup():
    """A second Ctrl+C must not leave the owned worker half-cleaned."""
    previous = {}
    for name in ("SIGINT", "SIGBREAK"):
        number = getattr(signal, name, None)
        if number is not None:
            previous[number] = signal.signal(number, signal.SIG_IGN)
    try:
        yield
    finally:
        for number, handler in previous.items():
            signal.signal(number, handler)


def _stop_worker(process, *, grace_s=3.0, kill_timeout_s=3.0) -> tuple[bool, bool]:
    """Stop only this isolated worker tree. All waits are bounded.

    Returns (forced, worker_exited). Native Kit/network calls may ignore the
    graceful request; taskkill/process-group kill is the final fallback.
    """
    forced = False
    with _uninterrupted_cleanup():
        try:
            if os.name == "nt":
                process.send_signal(signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(process.pid, signal.SIGINT)
            process.wait(timeout=grace_s)
        except (OSError, subprocess.TimeoutExpired):
            forced = True
        if os.name == "nt":
            if forced and process.poll() is None:
                try:
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   timeout=kill_timeout_s, check=False)
                except (OSError, subprocess.TimeoutExpired):
                    pass
                if process.poll() is None:
                    try:
                        process.kill()
                    except OSError:
                        pass
        else:
            # Kill lingering descendants even if their leader handled SIGINT.
            try:
                os.killpg(process.pid, signal.SIGKILL)
                forced = True
            except ProcessLookupError:
                pass
        try:
            process.wait(timeout=kill_timeout_s)
        except subprocess.TimeoutExpired:
            return forced, False
    return forced, True


def _monitor_worker(process, log, deadline: float | None) -> int:
    """Keep blocking pipe reads off the main thread so Ctrl+C stays responsive."""
    messages = Queue()

    def read_output():
        try:
            # Only this thread owns the buffered read/close lock.
            with process.stdout as stream:
                for line in stream:
                    messages.put(line)
        except Exception as exc:
            messages.put(exc)
        finally:
            messages.put(None)

    reader = threading.Thread(target=read_output, name="r5-worker-output", daemon=True)
    reader.start()
    eof = False
    while True:
        if deadline is not None and time.monotonic() >= deadline:
            raise subprocess.TimeoutExpired("R5 worker", 0)
        try:
            message = messages.get(timeout=0.1) if not eof else None
        except Empty:
            message = ""
        if isinstance(message, Exception):
            raise message
        if message is None:
            eof = True
        elif message:
            log.write(message)
            log.flush()
            print(message, end="", flush=True)
        if eof:
            try:
                return process.wait(timeout=0.1)
            except subprocess.TimeoutExpired:
                pass


def completed_check(output: Path, returncode: int) -> bool:
    """An exit code of zero alone is not evidence that a physical check passed."""
    try:
        result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (isinstance(result, dict) and returncode == 0
            and result.get("status") == "completed" and result.get("passed") is True)


def run_check_process(name: str, output_root: Path, *, config_path: Path | None = None,
                      gui: bool = False, model_config: Path | None = None) -> tuple[Path, bool]:
    spec = get_check(name)
    spec.validate_options(gui=gui, model_config=model_config)
    config = spec.load_config(config_path)
    if spec.preflight is not None:
        spec.preflight(config.robot)
    output = output_root / (spec.output_prefix + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-m", "robot_tasks.integrationtest.simulation_worker", "--config", str(config.source),
               "--run-directory", str(output.resolve()), "--check", spec.name]
    if gui:
        command.append("--gui")
    if model_config is not None:
        command.extend(("--model-config", str(model_config.resolve())))
    environment = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    timed_out = interrupted = forced = False
    cleanup_complete = True
    launch_options = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
                      else {"start_new_session": True})
    with (output / "process.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=PROJECT_ROOT, env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", **launch_options)
        print(f"R5_PROCESS_STARTED worker_pid={process.pid} output={output}", flush=True)
        try:
            deadline = time.monotonic() + config.max_wall_time_s + 60 if config.max_wall_time_s > 0 else None
            returncode = _monitor_worker(process, log, deadline)
        except KeyboardInterrupt:
            interrupted = True
            print(f"R5_PROCESS_STOPPING worker_pid={process.pid} reason=user_interrupt", flush=True)
            forced, cleanup_complete = _stop_worker(process)
            returncode = 130
        except subprocess.TimeoutExpired:
            timed_out = True
            forced, cleanup_complete = _stop_worker(process)
            returncode = 124
        except BaseException:
            _stop_worker(process)
            raise
    from ..runtime import snapshot_runtime_config
    snapshot_runtime_config(output)
    passed = completed_check(output, returncode) and not (timed_out or interrupted)
    write_json(output / "process.json", {"returncode": returncode, "passed": passed,
                                        "timed_out": timed_out, "interrupted": interrupted,
                                        "forced_termination": forced, "cleanup_complete": cleanup_complete,
                                        "worker_returncode": process.returncode, "worker_pid": process.pid})
    if (interrupted or timed_out) and not (output / "result.json").exists():
        write_json(output / "result.json", {
            "task_id": config.task_id, "status": "cancelled" if interrupted else "error",
            "termination_reason": "user_interrupt" if interrupted else "watchdog_timeout",
            "task_evaluated": False, "success": None, "passed": False,
            "error": "Worker stopped before a complete episode result was saved",
        })
    from ..reporting import write_run_report
    write_run_report(output)
    print("R5_PROCESS_RESULT " + json.dumps({"output": str(output), "passed": passed,
                                            "returncode": returncode, "interrupted": interrupted,
                                            "cleanup_complete": cleanup_complete}), flush=True)
    if interrupted:
        raise KeyboardInterrupt
    return output, passed
