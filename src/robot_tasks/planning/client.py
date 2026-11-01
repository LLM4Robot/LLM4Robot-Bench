"""Owned JSON-lines worker; Windows Isaac can use an isolated WSL CUDA environment."""

from dataclasses import asdict
import json
import os
import queue
import subprocess
import sys
import threading
import time
import tomllib

from ..paths import PROJECT_ROOT
from .failures import PlanningFailure

PREFIX = "R5_PLANNER "


def linux_path(path):
    path = str(path).replace("\\", "/")
    if len(path) > 2 and path[1:3] == ":/":
        return "/mnt/" + path[0].lower() + path[2:]
    return path


class PlannerClient:
    def __init__(self, profile, directory, *, log_directory=None, attachments=None):
        local = PROJECT_ROOT / "configs/runtime/planner.local.toml"
        data = tomllib.loads(local.read_text()) if local.is_file() else {}
        if set(data) - {"python", "wsl_distribution"}:
            raise ValueError(
                "Planner machine config accepts only python and wsl_distribution"
            )
        python = os.environ.get(
            "GPT_BENCHMARK_PLANNER_PYTHON", data.get("python", sys.executable)
        )
        bridge = sys.platform == "win32" and python.startswith("/")
        command = (
            [
                "wsl.exe",
                *(
                    ["-d", data["wsl_distribution"]]
                    if data.get("wsl_distribution")
                    else []
                ),
                "--exec",
                python,
            ]
            if bridge
            else [python]
        )
        convert = linux_path if bridge else str
        command += [
            "-u",
            "-m",
            "robot_tasks.planning.worker",
            "--profile",
            convert(profile.source),
            "--directory",
            convert(directory),
        ]
        self.process = None
        self.messages = queue.Queue()
        directory.mkdir(parents=True, exist_ok=True)
        if attachments:
            import hashlib
            payload = json.dumps(attachments, sort_keys=True)
            identity = hashlib.sha256(payload.encode()).hexdigest()[:16]
            attachment_path = directory / ("wrist_" + identity + ".json")
            if not attachment_path.is_file():
                temporary = attachment_path.with_suffix(f".{os.getpid()}.tmp")
                temporary.write_text(payload, encoding="utf-8")
                temporary.replace(attachment_path)
            command += ["--attachments", convert(attachment_path)]
        log_directory = log_directory or directory
        log_directory.mkdir(parents=True, exist_ok=True)
        self.log_path = log_directory / "planner_worker.log"
        self.log = self.log_path.open("a", encoding="utf-8")
        self.sequence = 0
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            cwd=PROJECT_ROOT,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            start_new_session=sys.platform != "win32",
        )
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        try:
            self.metadata = self._receive(600)
            if not self.metadata.get("ready"):
                raise RuntimeError(
                    self.metadata.get("error", "cuRobo worker failed to initialize")
                )
        except BaseException:
            self.close()
            raise

    def _read(self):
        stream = self.process.stdout
        try:
            for line in stream:
                # wsl.exe can append a UTF-16 NUL from its startup diagnostic
                # immediately before the child's first UTF-8 protocol line.
                line = line.lstrip("\x00\r\n")
                if line.startswith(PREFIX):
                    try:
                        self.messages.put(json.loads(line[len(PREFIX) :]))
                    except ValueError:
                        self.messages.put(
                            {"error": "Invalid planner protocol response"}
                        )
                else:
                    self.log.write(line)
                    self.log.flush()
        finally:
            stream.close()
            self.log.close()
            self.messages.put(
                {"error": "Planner worker exited; inspect " + str(self.log_path)}
            )

    def _receive(self, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                return self.messages.get(
                    timeout=min(0.1, max(0.01, deadline - time.monotonic()))
                )
            except queue.Empty:
                pass  # Frequent wakeups keep Ctrl+C responsive, including Windows.
        raise RuntimeError(
            "cuRobo worker response timed out; inspect " + str(self.log_path)
        )

    def plan(self, action, states, scene):
        self.sequence += 1
        request = {
            "id": self.sequence,
            "commands": [asdict(c) for c in action.commands if c.kind != "gripper"],
            "states": {a: asdict(s) for a, s in states.items()},
            "scene": scene,
        }
        self.process.stdin.write(json.dumps(request, allow_nan=False) + "\n")
        self.process.stdin.flush()
        result = self._receive(120)
        if result.get("id") != self.sequence:
            raise RuntimeError(result.get("error", "Planner response ID mismatch"))
        if "error" in result:
            if "planning_failure" in result:
                raise PlanningFailure(result["planning_failure"])
            raise ValueError("cuRobo rejected: " + result["error"])
        return result

    def close(self):
        if self.process is not None:
            if self.process.stdin:
                try:
                    self.process.stdin.close()  # Worker exits on EOF, including WSL bridge.
                except OSError:
                    pass
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=3)
            self.reader.join(timeout=3)
            self.process = None
        # Reader owns the log until EOF; do not race its last diagnostic write.
