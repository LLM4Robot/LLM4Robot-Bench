"""Run Codex natively or through WSL, with an owned, cancellable process tree.

The WSL worker in this file uses only Python 3.10's standard library. It can run
without installing the robot project or Isaac in WSL, and never reads tokens.
"""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import signal
import subprocess
import threading
import sys
import tempfile
import time
from urllib.parse import urlsplit

_WINDOWS = sys.platform == "win32"


def _wsl_path(command, *, timeout_s=10):
    """Bound path conversion even when a WSL descendant retains stdout handles."""
    options = ({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt"
               else {"start_new_session": True})
    # communicate() on Windows can wait forever for inherited pipe handles after
    # killing the timed-out launcher. A file does not require descendant EOF.
    with tempfile.TemporaryFile() as output:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output,
                                   stderr=subprocess.DEVNULL, env=codex_environment(), **options)
        try:
            process.wait(timeout=timeout_s)
        except BaseException:
            try:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   timeout=3, check=False, creationflags=subprocess.CREATE_NO_WINDOW)
                else:
                    _kill_group(process.pid)
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=3)
            raise
        output.seek(0)
        value = output.read(16385).decode("utf-8", errors="replace").strip()
    if process.returncode or len(value) > 16384 or not value.startswith("/") or "\n" in value:
        raise RuntimeError("Cannot translate project paths into WSL")
    return PurePosixPath(value)


class _StdoutLines:
    """Tail a separate read handle; never move the child's stdout file offset."""

    def __init__(self, reader, callback):
        self.reader, self.callback = reader, callback
        self.pending = b""
        self.total = 0

    def poll(self, *, final=False):
        limit = 4 * 1024 * 1024
        chunk = self.reader.read(limit + 1 - self.total)
        self.total += len(chunk)
        if self.total > limit:
            raise RuntimeError("Codex event stream exceeds size limit")
        if self.callback is None:
            return
        self.pending += chunk
        lines = self.pending.split(b"\n")
        self.pending = lines.pop()
        if final and self.pending:
            lines.append(self.pending)
            self.pending = b""
        for line in lines:
            if line.strip():
                self.callback(line)


def codex_environment(environment=None):
    """Keep desktop/network/auth paths, without forwarding API/application keys."""
    source = os.environ if environment is None else environment
    allowed = {"PATH", "HOME", "USER", "LOGNAME", "USERPROFILE", "APPDATA", "LOCALAPPDATA",
               "SYSTEMROOT", "SystemRoot", "COMSPEC", "PATHEXT", "TEMP", "TMP", "TMPDIR",
               "CODEX_HOME", "LANG", "LC_ALL", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME",
               "XDG_RUNTIME_DIR", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
               "http_proxy", "https_proxy", "all_proxy", "no_proxy", "SSL_CERT_FILE", "SSL_CERT_DIR"}
    return {key: value for key, value in source.items() if key in allowed}


def _local_settings():
    import tomllib
    path = Path(__file__).resolve().parents[3] / "configs/runtime/codex.local.toml"
    data = tomllib.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    if set(data) - {"host", "wsl_distribution", "wsl_user", "wsl_python", "wsl_proxy"}:
        raise ValueError("Unknown Codex runtime setting")
    if any(not isinstance(value, str) or not value.strip() or any(c in value for c in "\r\n\x00")
           for value in data.values()):
        raise ValueError("Codex runtime settings must be non-empty single-line strings")
    if data.get("host", "auto") not in {"auto", "native", "wsl"}:
        raise ValueError("Codex host must be auto, native or wsl")
    if "wsl_proxy" in data:
        proxy = urlsplit(data["wsl_proxy"])
        if proxy.scheme not in {"http", "https", "socks5", "socks5h"} or not proxy.hostname or proxy.username or proxy.password:
            raise ValueError("wsl_proxy must be a proxy URL without credentials")
    return data


class CodexProcess:
    def __init__(self, codex_bin):
        self._cancel = threading.Event()
        self._path_roots = {}
        config = _local_settings()
        self.codex_bin = codex_bin
        self.executable = shutil.which(codex_bin)
        preference = config.get("host", "auto")
        self.mode = "wsl" if _WINDOWS and (
            preference == "wsl" or (preference == "auto" and not self.executable)
        ) else "native"
        self.prefix = []
        if self.mode == "native":
            if not self.executable:
                raise RuntimeError("Codex CLI not found; install Codex or configure codex_bin")
            return
        wsl = shutil.which("wsl.exe")
        if not wsl:
            raise RuntimeError("WSL not found; configure a native Codex executable")
        self.prefix = [wsl]
        for field, flag in (("wsl_distribution", "--distribution"), ("wsl_user", "--user")):
            if config.get(field):
                self.prefix.extend([flag, config[field]])
        self.prefix.append("--exec")
        self.python = config.get("wsl_python", "python3")
        self.proxy = config.get("wsl_proxy")
        self.worker = str(self.path(Path(__file__).resolve()))

    def path(self, path):
        if self.mode == "native":
            return path
        windows = PureWindowsPath(str(path))
        anchor = windows.anchor.casefold()
        if windows.is_absolute() and anchor in self._path_roots:
            return self._path_roots[anchor].joinpath(*windows.parts[1:])
        try:
            converted = _wsl_path([*self.prefix, "wslpath", "-a", "-u", str(path)])
        except (OSError, subprocess.TimeoutExpired):
            raise RuntimeError("Cannot translate project paths into WSL") from None
        # Learn the selected distro's actual mount root before Isaac starts;
        # do not assume /mnt, a drive letter, or a particular WSL configuration.
        suffix = windows.parts[1:]
        if windows.is_absolute() and suffix and len(converted.parts) > len(suffix):
            tail = converted.parts[-len(suffix):]
            if tuple(p.casefold() for p in tail) == tuple(p.casefold() for p in suffix):
                self._path_roots[anchor] = PurePosixPath(*converted.parts[:-len(suffix)])
        return converted

    def command(self, arguments, directory, timeout_s):
        if self.mode == "native":
            return [self.executable, *arguments]
        return [*self.prefix, self.python, self.worker, "--wsl-worker",
                "--directory", str(self.path(directory)), "--codex-bin", self.codex_bin,
                "--timeout", str(timeout_s), "--file-output",
                *(["--proxy", self.proxy] if self.proxy else []), "--", *arguments]

    def cancel_pending(self):
        self._cancel.set()

    def reset_cancellation(self):
        self._cancel.clear()

    def run(self, arguments, *, directory, data=b"", timeout_s, on_stdout_line=None):
        if self._cancel.is_set():
            raise KeyboardInterrupt("Codex request cancelled")
        if self.mode == "wsl":
            (directory / "worker_result.json").unlink(missing_ok=True)
        command = self.command(arguments, directory, timeout_s)
        # A finite input file gives the child reliable EOF, even if it starts
        # reading late. Repeated communicate(None) after a short timeout can
        # abandon pending pipe writes on Python 3.10. Keep polling only wait().
        # The temporary input stays in the request directory and is removed
        # on close; never place observation data in a system temporary folder.
        with tempfile.TemporaryFile(mode="w+b", dir=directory) as prompt, \
                (directory / "events.jsonl").open("w+b") as output, \
                (directory / "stderr.log").open("w+b") as errors, \
                (directory / "events.jsonl").open("rb") as reader:
            prompt.write(data)
            prompt.seek(0)
            lines = _StdoutLines(reader, on_stdout_line)
            process = subprocess.Popen(command, stdin=prompt, stdout=output, stderr=errors,
                                       env=codex_environment(), start_new_session=(os.name != "nt"),
                                       # stdin/stdout are files. WSL must not inherit
                                       # Isaac/Kit's console and wait on its host at exit.
                                       creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP |
                                                      subprocess.CREATE_NO_WINDOW) if os.name == "nt" else 0)
            reported_exit = None
            try:
                # WSL enforces the same deadline itself; allow it time to reap
                # Linux descendants before terminating the Windows launcher.
                deadline = time.monotonic() + timeout_s + (3 if self.mode == "wsl" else 0)
                while True:
                    if self._cancel.is_set():
                        raise KeyboardInterrupt("Codex request cancelled")
                    completion = directory / "worker_result.json"
                    if self.mode == "wsl" and completion.is_file():
                        report = json.loads(completion.read_text(encoding="utf-8"))
                        if (type(report.get("returncode")) is not int
                                or report.get("cleanup_complete") is not True):
                            raise RuntimeError("Invalid WSL worker completion record")
                        # The Linux supervisor has reaped Codex and its children.
                        # wsl.exe can linger after that; its exit code is not the
                        # already completed model process's exit code.
                        reported_exit = report["returncode"]
                        try:
                            process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            self.terminate(process, directory)
                        lines.poll(final=True)
                        break
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, timeout_s)
                    try:
                        # Short waits let Python handle Ctrl+C/CTRL_BREAK on
                        # Windows even when the provider has not returned yet.
                        process.wait(timeout=min(.1, remaining))
                        lines.poll(final=True)
                        break
                    except subprocess.TimeoutExpired:
                        lines.poll()
            except BaseException:
                self.terminate(process, directory)
                raise
            output.seek(0)
            stdout = output.read(4 * 1024 * 1024 + 1)
            errors.seek(0)
            stderr = errors.read(8192)
        if len(stdout) > 4 * 1024 * 1024:
            raise RuntimeError("Codex event stream exceeds size limit")
        returncode = process.returncode if reported_exit is None else reported_exit
        if self.mode == "wsl" and returncode == 124:
            raise subprocess.TimeoutExpired(command, timeout_s)
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)

    def terminate(self, process, directory):
        if process.poll() is not None:
            return
        if self.mode == "wsl":
            # taskkill cannot reliably reap Linux descendants. Ask their Linux
            # supervisor to kill its own process group before killing wsl.exe.
            (directory / "cancel").touch()
            try:
                process.wait(timeout=3)
                return
            except subprocess.TimeoutExpired:
                pass
        if os.name == "nt":
            try:
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3,
                               check=False, creationflags=subprocess.CREATE_NO_WINDOW)
            except (OSError, subprocess.TimeoutExpired):
                if process.poll() is None:
                    process.kill()
        else:
            _kill_group(process.pid)
        process.wait(timeout=10)


def _kill_group(pid):
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def wsl_worker(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--wsl-worker", action="store_true", required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--codex-bin", required=True)
    parser.add_argument("--timeout", type=float, required=True)
    parser.add_argument("--proxy")
    parser.add_argument("--file-output", action="store_true")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    executable = shutil.which(args.codex_bin)
    if not executable and args.codex_bin == "codex":
        # Codex's standalone installer uses this location; non-login WSL shells
        # commonly omit it from PATH. Use the selected WSL user's installation.
        candidate = Path.home() / ".local/bin/codex"
        if candidate.is_file() and os.access(candidate, os.X_OK):
            executable = str(candidate)
    if not executable:
        print("Codex CLI not found for the selected WSL user", file=sys.stderr)
        return 127
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    environment = codex_environment()
    if args.proxy:
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            environment[key] = args.proxy
    # Hand the existing input descriptor directly to Codex. This supervisor
    # must not drain/re-send the prompt or keep an extra pipe writer open.
    with ExitStack() as files:
        # Bypass wsl.exe's asynchronous stdout forwarding. Completion is only
        # published after the actual child's output and descendants are closed.
        stdout = files.enter_context((args.directory / "events.jsonl").open("ab")) if args.file_output else None
        stderr = files.enter_context((args.directory / "stderr.log").open("ab")) if args.file_output else None
        process = subprocess.Popen([executable, *arguments], stdin=sys.stdin.buffer,
                                   stdout=stdout, stderr=stderr,
                                   env=environment, start_new_session=True)
        deadline = time.monotonic() + args.timeout
        returncode = None
        try:
            while True:
                if not args.directory.exists() or (args.directory / "cancel").exists():
                    returncode = 130
                    return returncode
                if time.monotonic() >= deadline:
                    returncode = 124
                    return returncode
                try:
                    process.wait(timeout=0.1)
                    returncode = process.returncode
                    return returncode
                except subprocess.TimeoutExpired:
                    pass
        finally:
            _kill_group(process.pid)
            process.wait(timeout=3)
            files.close()
            if returncode is not None and args.directory.exists():
                temporary = args.directory / "worker_result.json.tmp"
                temporary.write_text(json.dumps({"returncode": returncode, "cleanup_complete": True}),
                                     encoding="utf-8")
                temporary.replace(args.directory / "worker_result.json")


if __name__ == "__main__":
    raise SystemExit(wsl_worker())
