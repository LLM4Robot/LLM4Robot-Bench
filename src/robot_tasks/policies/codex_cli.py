"""Use the user's ChatGPT login through Codex for bounded robot decisions.

Codex owns credentials and refresh; this adapter never reads auth.json. Each
invocation receives only disclosed observations/history and returns one JSON
decision. Physics, validation, evaluation and video stay in the existing runner.
CLI usage describes Codex turns, not a promise of one HTTP request per decision.
"""

import base64
import binascii
import json
from pathlib import Path
import re
import subprocess
import tempfile
import time

from ..paths import PROJECT_ROOT
from .codex_process import CodexProcess, codex_environment
from .model_config import GPTSettings
from .reasoning import ReasoningSummaries
from .responses import ModelError, ModelReply, RetryableModelError, _safe_error_field, generation_error


def _temporary_directory():
    root = PROJECT_ROOT / "build/codex"
    root.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(prefix="decision-", dir=root)


def safe_diagnostic(value):
    text = _safe_error_field(value, "") or "No diagnostic available"
    return re.sub(r"\beyJ[A-Za-z0-9_.-]+", "[REDACTED]", text)


def failure_diagnostic(stdout, stderr, returncode):
    """Keep error fields only, never observations, model answers or credentials."""
    errors = []
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        kind = event.get("type")
        if kind == "turn.failed":
            payload = event.get("error", {})
        elif kind == "error":
            payload = event
        elif kind == "item.completed" and isinstance(event.get("item"), dict) and event["item"].get("type") == "error":
            payload = event["item"]
        else:
            continue
        if isinstance(payload, str):
            payload = {"message": payload}
        if isinstance(payload, dict):
            errors.append({"event": kind, **{k: safe_diagnostic(payload[k]) for k in
                           ("code", "type", "message") if isinstance(payload.get(k), (str, int, float))}})
    # stderr is a bounded diagnostic excerpt, not an exact raw transcript.
    return {"exit_code": returncode, "errors": errors[-8:],
            "stderr_excerpt": safe_diagnostic(stderr[-4096:]) if stderr else None}


def codex_failure(diagnostic):
    details = diagnostic["errors"]
    messages = [item.get("message", "") for item in details]
    messages.append(diagnostic.get("stderr_excerpt") or "")
    message = " | ".join(value for value in messages if value)
    code = " ".join(str(item.get("code", "")) for item in details)
    permanent = any(value in message.lower() for value in (
        "usage limit", "usage_limit", "not logged in", "please log in", "authentication",
        "unauthorized", "context window", "context length", "context_length",
        "insufficient_quota", "quota_exceeded", "login required", "refresh token",
        "input_too_large", "input exceeds the maximum length",
    ))
    classified = generation_error({"error": {"message": message, "code": code}}, "codex.failed")
    error_type = ModelError if permanent or not isinstance(classified, RetryableModelError) else RetryableModelError
    return error_type(f"Codex ChatGPT call failed (exit {diagnostic['exit_code']}); no action accepted; "
                      + (message[:2048] or "no diagnostic returned"))


def serialize_observations(messages, directory):
    """Map each attached PNG to an explicit, ordered label in the text prompt."""
    parts, images = [], []
    # Long episodes repeat the identical task instruction in every observation.
    # Share those strings once, keeping every historical observation and action.
    instructions = {}
    for message in messages:
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, list):
            continue
        for item in content:
            if item.get("type") != "input_text":
                continue
            try:
                payload = json.loads(item.get("text", ""))
            except (ValueError, TypeError):
                continue
            if isinstance(payload, dict) and isinstance(payload.get("instruction"), str):
                instruction = payload["instruction"]
                instructions.setdefault(instruction, f"task_{len(instructions) + 1}")
    if instructions:
        parts.append("Shared task instructions: task_instruction_ref in each observation refers to "
                     "the complete instruction below; apply it exactly as if repeated in that record.\n"
                     + json.dumps({ref: text for text, ref in instructions.items()},
                                  ensure_ascii=False, separators=(",", ":")))
    for number, message in enumerate(messages, 1):
        role = message.get("role")
        if role not in {"user", "assistant"}:
            raise ModelError("Unsupported Codex observation role")
        parts.append(f"Record {number}, role={role}:")
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
            continue
        if not isinstance(content, list):
            raise ModelError("Invalid Codex observation content")
        for item in content:
            if item.get("type") == "input_text" and isinstance(item.get("text"), str):
                value = item["text"]
                try:
                    payload = json.loads(value)
                except ValueError:
                    pass
                else:
                    if isinstance(payload, dict) and isinstance(payload.get("instruction"), str) and payload["instruction"] in instructions:
                        payload["task_instruction_ref"] = instructions[payload.pop("instruction")]
                    value = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                parts.append(value)
            elif item.get("type") == "input_image":
                url = item.get("image_url", "")
                prefix = "data:image/png;base64,"
                if not isinstance(url, str) or not url.startswith(prefix):
                    raise ModelError("Codex observations must be inline PNGs")
                try:
                    png = base64.b64decode(url[len(prefix):], validate=True)
                except (ValueError, binascii.Error):
                    raise ModelError("Invalid observation PNG encoding") from None
                if not png.startswith(b"\x89PNG\r\n\x1a\n") or len(png) > 16 * 1024 * 1024:
                    raise ModelError("Invalid or oversized observation PNG")
                path = directory / f"image_{len(images) + 1:03d}.png"
                path.write_bytes(png)
                images.append(path)
                parts.append(f"[Attached image #{len(images)} belongs to the immediately preceding label.]")
            else:
                raise ModelError("Unsupported Codex observation item")
    return "\n".join(parts), images


def parse_events(stdout, *, model, latency_s):
    thread, texts, completed, usage = None, [], False, {}
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            raise ModelError("Malformed Codex event stream; no action accepted") from None
        if not isinstance(event, dict):
            raise ModelError("Malformed Codex event; no action accepted")
        if event.get("type") == "thread.started":
            thread = event.get("thread_id")
        elif event.get("type") == "turn.completed":
            if completed:
                raise ModelError("Multiple Codex turns; no action accepted")
            completed = True
            counts = event.get("usage") or {}
            usage = {k: counts[k] for k in ("input_tokens", "output_tokens") if type(counts.get(k)) is int}
            if type(counts.get("cached_input_tokens")) is int:
                usage["cached_tokens"] = counts["cached_input_tokens"]
            if type(counts.get("reasoning_output_tokens")) is int:
                usage["reasoning_tokens"] = counts["reasoning_output_tokens"]
            if "input_tokens" in usage and "output_tokens" in usage:
                usage["total_tokens"] = usage["input_tokens"] + usage["output_tokens"]
        elif event.get("type") == "turn.failed":
            raise codex_failure(failure_diagnostic(stdout, "", 0))
        elif event.get("type") in {"item.started", "item.updated", "item.completed"}:
            item = event.get("item") or {}
            kind = item.get("type")
            if kind not in {"agent_message", "reasoning", "error"}:
                raise ModelError("Codex attempted a tool or unsupported item; no action accepted")
            if event["type"] == "item.completed" and kind == "agent_message":
                texts.append(item.get("text"))
    if not completed or not isinstance(thread, str) or len(texts) != 1 or not isinstance(texts[0], str):
        if not completed:
            raise codex_failure(failure_diagnostic(stdout, "", 0))
        raise ModelError("Codex did not return one completed decision; no action accepted")
    try:
        document = json.loads(texts[0])
    except ValueError:
        raise ModelError("Codex final decision is not JSON; no action accepted") from None
    if not isinstance(document, dict):
        raise ModelError("Codex final decision must be a JSON object")
    # CLI JSONL doesn't disclose the server-returned model ID. Record the exact
    # configured model honestly; do not claim independent response-ID verification.
    return ModelReply(texts[0], model, thread, usage, latency_s)


class CodexCLIClient:
    provider_metadata = {"authentication": "chatgpt", "model_verification": "explicit_cli_configuration",
                         "response_id_kind": "codex_thread_id",
                         "input_serialization": "compact_json_shared_task_instruction_v1"}

    def __init__(self, settings: GPTSettings):
        self.settings = settings
        self.last_diagnostic = None
        try:
            self.host = CodexProcess(settings.codex_bin)
        except RuntimeError as exc:
            raise ModelError(str(exc)) from None
        self.provider_metadata = {**self.provider_metadata, "codex_host": self.host.mode}

    def cancel_pending(self):
        self.host.cancel_pending()

    def reset_cancellation(self):
        self.host.reset_cancellation()

    def require_chatgpt_login(self):
        try:
            with _temporary_directory() as temporary:
                status = self.host.run(["login", "status"], directory=Path(temporary), timeout_s=15)
        except (OSError, subprocess.TimeoutExpired):
            raise ModelError("Cannot check Codex ChatGPT login; run codex login status") from None
        except RuntimeError as exc:
            raise ModelError(str(exc)) from None
        if status.returncode == 127:
            raise ModelError("Codex CLI not found in WSL; check configs/runtime/codex.local.toml and codex_bin")
        if status.returncode != 0 or b"Logged in using ChatGPT" not in status.stdout + status.stderr:
            raise ModelError("Codex ChatGPT login required; run codex login, then codex login status")

    def generate(self, *, instructions, messages, schema, on_reasoning=None):
        settings = self.settings
        self.last_diagnostic = None
        start = time.monotonic()
        with _temporary_directory() as temporary:
            directory = Path(temporary)
            remote = self.host.path(directory)
            observations, images = serialize_observations(messages, directory)
            schema_file = directory / "action.schema.json"
            schema_file.write_text(json.dumps(schema, allow_nan=False), encoding="utf-8")
            instructions_file = directory / "robot.instructions.txt"
            instructions_file.write_text("You are a robot decision policy, not a coding agent. "
                                         "Use only supplied sensor evidence. Do not call tools or access files. "
                                         "Return one decision matching the output schema.\n" + instructions,
                                         encoding="utf-8")
            prompt = ("You are the robot decision model for a benchmark. Return exactly one JSON object. "
                      "Do not use tools, read files, inspect the repository, or execute commands. "
                      "All permitted evidence is supplied below and in the attached images. "
                      "Image numbers in the records identify the ordered attachments. "
                      "Use the latest user record as the current observation; earlier records are history.\n"
                      + "\nOBSERVATION HISTORY:\n" + observations)
            # Codex turn/start currently rejects more than 1 MiB of input text.
            # Use UTF-8 bytes conservatively; never silently drop old decisions.
            if len(prompt.encode("utf-8")) > 1_048_576:
                self.last_diagnostic = {"exit_code": "local_input_limit", "errors": [{
                    "code": "input_too_large", "message": "Compact full history exceeds Codex input limit"}],
                    "input_chars": len(prompt), "input_utf8_bytes": len(prompt.encode("utf-8")),
                    "stderr_excerpt": None}
                raise ModelError("Codex input_too_large: compact full history exceeds 1048576 UTF-8 bytes; "
                                 "no request sent and no history discarded")
            command = ["exec", "--ignore-user-config", "--ephemeral", "--skip-git-repo-check",
                       "--cd", str(remote), "--sandbox", "read-only", "--model", settings.model,
                       "--json", "--color", "never", "--output-schema", str(remote / schema_file.name)]
            overrides = {"model_provider": "openai", "forced_login_method": "chatgpt",
                         "model_reasoning_effort": settings.reasoning_effort, "web_search": "disabled",
                         "model_reasoning_summary": "auto",
                         "features.shell_tool": False, "features.unified_exec": False,
                         "features.multi_agent": False, "features.apps": False, "features.plugins": False,
                         "features.browser_use": False, "features.computer_use": False,
                         "features.image_generation": False, "features.hooks": False,
                         "features.skip_host_skill_discovery": True, "suppress_unstable_features_warning": True,
                         "features.unbounded_connection_retries": False,
                         "model_instructions_file": str(remote / instructions_file.name)}
            if settings.service_tier is not None:
                # exec ignores user config, so the requested tier must travel
                # with every independent decision, separately from reasoning.
                overrides["service_tier"] = settings.service_tier
                overrides["features.fast_mode"] = True
            for key, value in overrides.items():
                command.extend(["-c", key + "=" + json.dumps(value)])
            for path in images:
                command.extend(["--image", str(remote / path.name)])
            command.append("-")
            summaries = ReasoningSummaries(on_reasoning)
            try:
                result = self.host.run(command, directory=directory, data=prompt.encode("utf-8"),
                                       timeout_s=settings.timeout_s, on_stdout_line=summaries.codex_line)
            except subprocess.TimeoutExpired:
                self.last_diagnostic = failure_diagnostic(
                    (directory / "events.jsonl").read_text(encoding="utf-8", errors="replace")
                    if (directory / "events.jsonl").exists() else "",
                    (directory / "stderr.log").read_text(encoding="utf-8", errors="replace")
                    if (directory / "stderr.log").exists() else "", "timeout")
                raise RetryableModelError("Codex ChatGPT decision timed out; no action accepted") from None
            except OSError:
                raise ModelError("Unable to start Codex CLI") from None
            except RuntimeError as exc:
                raise ModelError(str(exc)) from None
            stdout = result.stdout.decode("utf-8", errors="replace")
            if result.returncode == 130:
                raise KeyboardInterrupt
            if result.returncode:
                self.last_diagnostic = failure_diagnostic(stdout, result.stderr.decode("utf-8", errors="replace"),
                                                          result.returncode)
                raise codex_failure(self.last_diagnostic)
            try:
                return parse_events(stdout, model=settings.model, latency_s=time.monotonic() - start)
            except ModelError:
                self.last_diagnostic = failure_diagnostic(stdout, result.stderr.decode("utf-8", errors="replace"), 0)
                raise
