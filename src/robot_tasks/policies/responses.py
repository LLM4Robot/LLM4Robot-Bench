"""Responses HTTP/SSE transport. No hidden retries, redirects or local model.

Each generate() sends exactly one request. The episode runner owns the fixed
generation retry budget, logs every attempt and counts it against max_calls.
Partial output can never become a robot action.
"""

from dataclasses import dataclass
import json
import re
import socket
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .model_config import GPTSettings, read_api_key
from .reasoning import ReasoningSummaries
from ..types import RetryablePolicyError


class ModelError(RuntimeError):
    """A sanitized provider/protocol error, safe for experiment logs."""


class RetryableModelError(ModelError, RetryablePolicyError):
    """Generation failed, was incomplete or timed out; no output is executable."""


def generation_error(document: dict, event_type: str, secret: str = "") -> ModelError:
    """Preserve bounded scalar reasons; never expose provider output/requests."""
    response = document.get("response")
    response = response if isinstance(response, dict) else document
    payload = response.get("error")
    if isinstance(payload, str):
        payload = {"message": payload}
    payload = payload if isinstance(payload, dict) else response
    details = {"event": event_type}
    for source, key, target in ((response, "id", "response_id"),
                                (payload, "code", "code"), (payload, "type", "type"),
                                (payload, "message", "message")):
        value = _safe_error_field(source.get(key), secret)
        if value is not None:
            details[target] = value
    incomplete = response.get("incomplete_details")
    if isinstance(incomplete, dict):
        value = _safe_error_field(incomplete.get("reason"), secret)
        if value is not None:
            details["incomplete_reason"] = value
    labels = " ".join(details.get(k, "") for k in ("code", "type", "incomplete_reason")).lower()
    # Explicit authorization, quota, refusal and invalid-request failures need
    # a change outside this retry. Unknown gateway generation failures get the
    # user's unlimited retries within episode budgets; this does not assert they are transient.
    permanent = any(code in labels for code in (
        "invalid_api_key", "authentication", "unauthorized", "permission", "access_denied",
        "insufficient_quota", "quota_exceeded", "billing_hard_limit", "subscription_sharing_usage",
        "content_filter", "refusal", "safety_violation", "invalid_request", "context_length_exceeded",
    ))
    message = details.get("message", "").lower()
    permanent |= any(reason in message for reason in (
        "限额已用完", "额度已耗尽", "额度用完", "quota exceeded", "quota exhausted",
        "invalid api key", "incorrect api key",
    ))
    error = ModelError if permanent else RetryableModelError
    return error("Provider reported failed/incomplete generation; provider_error="
                 + json.dumps(details, ensure_ascii=False, allow_nan=False))


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _safe_error_field(value, secret: str, *, limit: int = 512) -> str | None:
    """Only short scalar diagnostics; never dump a request, headers or HTML."""
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return None
    text = str(value).replace(secret, "[REDACTED]") if secret else str(value)
    text = re.sub(r"(?i)\bBearer\s+[^\s\"'<>]+", "Bearer [REDACTED]", text)
    text = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[REDACTED]", text)
    text = " ".join("".join(c if c.isprintable() else " " for c in text).split())
    return text[:limit] if text else None


def http_error_message(error: HTTPError, secret: str) -> str:
    """Keep a bounded, redacted reason from an HTTP rejection before closing it.

    A 429 alone cannot distinguish account quota, token rate limits or a relay's
    capacity. Record what the provider actually says without classifying it.
    """
    details = {}
    try:
        headers = error.headers or {}
        for header, name in (("Retry-After", "retry_after"), ("x-request-id", "request_id")):
            value = _safe_error_field(headers.get(header), secret, limit=128)
            if value is not None:
                details[name] = value
        try:
            raw = error.read(16385)
            if len(raw) > 16384:
                details["body_status"] = "too_large"
            else:
                document = json.loads(raw)
                if isinstance(document, dict):
                    payload = document.get("error", document)
                    if isinstance(payload, str):
                        payload = {"message": payload}
                    if isinstance(payload, dict):
                        for key in ("type", "code", "message"):
                            value = _safe_error_field(payload.get(key), secret)
                            if value is not None:
                                details[key] = value
                    if not any(key in details for key in ("type", "code", "message")):
                        details["body_status"] = "no_error_fields"
                else:
                    details["body_status"] = "unrecognized_json"
        except (ValueError, UnicodeError):
            details["body_status"] = "non_json_or_empty"
        except (OSError, TimeoutError):
            details["body_status"] = "unreadable"
    finally:
        error.close()
    return f"Responses HTTP {error.code}; provider_error=" + json.dumps(details, ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True)
class ModelReply:
    text: str
    model: str
    response_id: str
    usage: dict
    latency_s: float

    def metadata(self) -> dict:
        return {"model": self.model, "response_id": self.response_id,
                "usage": self.usage, "latency_s": self.latency_s}


def completed_response(document: dict, expected_model: str, latency_s: float,
                       *, secret: str = "") -> ModelReply:
    if isinstance(document, dict) and document.get("status") in {"failed", "incomplete"}:
        raise generation_error(document, "response." + document["status"], secret)
    if not isinstance(document, dict) or document.get("status") != "completed":
        raise ModelError("Response did not complete; no action accepted")
    if document.get("model") != expected_model:
        raise ModelError("Returned model ID differs from configured model; no action accepted")
    texts = []
    for item in document.get("output", []):
        if item.get("type") == "reasoning":
            continue
        if item.get("type") != "message" or item.get("role") != "assistant":
            raise ModelError("Unexpected response output type")
        for content in item.get("content", []):
            if content.get("type") == "refusal":
                raise ModelError("Model refused; no action accepted")
            if content.get("type") != "output_text" or not isinstance(content.get("text"), str):
                raise ModelError("Unexpected response content")
            texts.append(content["text"])
    if not texts:
        raise ModelError("Completed response contained no text")
    usage = document.get("usage") or {}
    counts = {key: usage[key] for key in ("input_tokens", "output_tokens", "total_tokens")
              if type(usage.get(key)) is int}
    for parent, child in (("input_tokens_details", "cached_tokens"), ("output_tokens_details", "reasoning_tokens")):
        value = (usage.get(parent) or {}).get(child)
        if type(value) is int:
            counts[child] = value
    return ModelReply("".join(texts), expected_model, str(document.get("id", "")), counts, latency_s)


def read_response(stream, *, expected_model: str, start: float, timeout_s: float,
                  secret: str = "", on_reasoning=None) -> ModelReply:
    """Accept JSON or SSE, requiring the terminal response.completed object."""
    limit = 4 * 1024 * 1024
    summaries = ReasoningSummaries(on_reasoning, secret=secret)
    if "text/event-stream" not in stream.headers.get("Content-Type", ""):
        raw = stream.read(limit + 1)
        if len(raw) > limit:
            raise ModelError("Provider response exceeds size limit")
        document = json.loads(raw)
        reply = completed_response(document, expected_model, time.monotonic() - start, secret=secret)
        summaries.response(document)
        return reply
    total = 0
    data_lines = []
    while True:
        if time.monotonic() - start >= timeout_s:
            raise RetryableModelError("Model stream exceeded request time budget")
        line = stream.readline(limit + 1)
        total += len(line)
        if total > limit:
            raise ModelError("Provider stream exceeds size limit")
        if not line:
            raise ModelError("Model stream ended without a completed response")
        if line.strip():
            if line.startswith(b"data:"):
                data_lines.append(line[5:].strip())
            continue
        if not data_lines:
            continue
        payload = b"\n".join(data_lines)
        data_lines.clear()
        if payload == b"[DONE]":
            raise ModelError("Stream ended without response.completed")
        event = json.loads(payload)
        summaries.response_event(event)
        if event.get("type") == "response.completed":
            return completed_response(event.get("response"), expected_model, time.monotonic() - start, secret=secret)
        if event.get("type") in {"error", "response.failed", "response.incomplete"}:
            raise generation_error(event, event["type"], secret)


class ResponsesClient:
    def __init__(self, settings: GPTSettings):
        self.settings = settings
        self._cancel = threading.Event()
        self._stream_lock = threading.Lock()
        self._active_stream = None

    def reset_cancellation(self):
        self._cancel.clear()

    def cancel_pending(self):
        self._cancel.set()
        with self._stream_lock:
            stream = self._active_stream
        # Shutdown unblocks HTTPResponse.readline without taking its buffered
        # reader lock on the GUI thread. The request thread closes the response.
        sock = getattr(getattr(getattr(stream, "fp", None), "raw", None), "_sock", None)
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def generate(self, *, instructions: str, messages: list[dict], schema: dict,
                 on_reasoning=None) -> ModelReply:
        settings = self.settings
        if self._cancel.is_set():
            raise KeyboardInterrupt("Responses request cancelled")
        secret = read_api_key(settings.api_key_env)
        # Codex-compatible relays may replace top-level instructions. Preserve
        # the action contract in an explicit developer message as well as schema.
        developer = {"role": "developer", "content": [{"type": "input_text", "text": instructions}]}
        payload = {"model": settings.model, "instructions": "Follow the developer's robot action contract.",
                   "input": [developer, *messages],
                   "store": False, "stream": True,
                   "reasoning": {"effort": settings.reasoning_effort, "summary": "auto"}}
        if settings.service_tier is not None:
            payload["service_tier"] = settings.service_tier
        if settings.structured_output:
            payload["text"] = {"format": {"type": "json_schema", "name": "r5_action",
                                         "strict": True, "schema": schema}}
        if settings.max_output_tokens is not None:
            payload["max_output_tokens"] = settings.max_output_tokens
        request = Request(settings.base_url.rstrip("/") + "/responses",
                          data=json.dumps(payload, allow_nan=False).encode("utf-8"),
                          headers={"Authorization": "Bearer " + secret, "Content-Type": "application/json",
                                   "Accept": "text/event-stream", "User-Agent": "R5-GPT-integration/0.1"})
        start = time.monotonic()
        try:
            with build_opener(_NoRedirect()).open(request, timeout=settings.timeout_s) as stream:
                with self._stream_lock:
                    self._active_stream = stream
                try:
                    if self._cancel.is_set():
                        raise KeyboardInterrupt("Responses request cancelled")
                    reply = read_response(stream, expected_model=settings.model, start=start,
                                          timeout_s=settings.timeout_s, secret=secret, on_reasoning=on_reasoning)
                finally:
                    with self._stream_lock:
                        self._active_stream = None
            if self._cancel.is_set():
                raise KeyboardInterrupt("Responses request cancelled")
            if time.monotonic() - start > settings.timeout_s:
                raise RetryableModelError("Model request exceeded time budget; output discarded")
            if secret in reply.text or secret in reply.response_id:
                raise ModelError("Provider output contains a credential; output discarded")
            return reply
        except HTTPError as exc:
            raise ModelError(http_error_message(exc, secret)) from None
        except TimeoutError:
            raise RetryableModelError("Responses request timed out; no action accepted") from None
        except URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise RetryableModelError("Responses request timed out; no action accepted") from None
            raise ModelError("Responses transport failed; no action accepted") from None
        except OSError:
            raise ModelError("Responses transport failed; no action accepted") from None
        except (ValueError, TypeError, AttributeError, KeyError):
            raise ModelError("Malformed Responses payload; no action accepted") from None
