"""Display public reasoning summaries, separate from executable decisions."""

import json
import re


class ReasoningSummaries:
    """Normalize summary events and deduplicate deltas/final snapshots.

    Only public summary fields are read. Raw reasoning, encrypted content,
    assistant action text and arbitrary provider events are never displayed.
    Complete lines are emitted during streaming, with the final line flushed
    at a summary boundary. This also keeps split credentials out of the log.
    """

    def __init__(self, callback=None, *, secret=""):
        self.callback = callback
        self.secret = secret
        self.parts = {}
        self.printed = {}

    def _publish(self, key, text, *, final):
        if not isinstance(text, str):
            return
        previous = self.parts.get(key, "")
        if not text.startswith(previous):
            self.printed[key] = 0
        self.parts[key] = text
        end = len(text) if final else text.rfind("\n") + 1
        start = self.printed.get(key, 0)
        if end <= start:
            return
        self.printed[key] = end
        value = text[start:end]
        if self.secret:
            value = value.replace(self.secret, "[REDACTED]")
        value = re.sub(r"(?i)\bBearer\s+[^\s\"'<>]+", "Bearer [REDACTED]", value)
        value = re.sub(r"\bsk-[A-Za-z0-9_-]+|\beyJ[A-Za-z0-9_.-]+", "[REDACTED]", value)
        for line in value.splitlines():
            line = "".join(c for c in line if c.isprintable() or c == "\t").strip()
            if line and self.callback is not None:
                self.callback(line)

    def _item(self, item, index=0):
        if not isinstance(item, dict) or item.get("type") != "reasoning":
            return
        for number, part in enumerate(item.get("summary") or []):
            if isinstance(part, dict) and part.get("type") == "summary_text":
                self._publish((item.get("id", index), number), part.get("text"), final=True)

    def response(self, document):
        if isinstance(document, dict):
            for index, item in enumerate(document.get("output") or []):
                self._item(item, index)

    def response_event(self, event):
        if not isinstance(event, dict):
            return
        kind = event.get("type")
        key = (event.get("item_id", event.get("output_index", 0)), event.get("summary_index", 0))
        if kind == "response.reasoning_summary_text.delta" and isinstance(event.get("delta"), str):
            self._publish(key, self.parts.get(key, "") + event["delta"], final=False)
        elif kind == "response.reasoning_summary_text.done":
            self._publish(key, event.get("text", self.parts.get(key, "")), final=True)
        elif kind == "response.reasoning_summary_part.done":
            part = event.get("part") or {}
            if isinstance(part, dict) and part.get("type") == "summary_text":
                self._publish(key, part.get("text"), final=True)
        elif kind == "response.output_item.done":
            self._item(event.get("item"), event.get("output_index", 0))
        elif kind == "response.completed":
            self.response(event.get("response"))

    def codex_line(self, line):
        # Display is advisory; the existing strict parser still validates the
        # full event stream before accepting any robot action.
        try:
            event = json.loads(line)
        except (ValueError, UnicodeError):
            return
        if not isinstance(event, dict):
            return
        kind = event.get("type")
        if kind not in {"item.started", "item.updated", "item.completed"}:
            return
        item = event.get("item") or {}
        if isinstance(item, dict) and item.get("type") == "reasoning":
            self._publish(("codex", item.get("id", "summary")), item.get("text"),
                          final=kind == "item.completed")


class DecisionProgress:
    """Per-request terminal output and recorded, public summary lines."""

    def __init__(self, step, exchange):
        self.prefix = f"[request {step}]"
        self.lines = exchange.setdefault("reasoning_summary", [])

    def __enter__(self):
        print(f"{self.prefix} thinking... (waiting for model output)", flush=True)
        return self

    def summary(self, text):
        self.lines.append(text)
        print(f"{self.prefix} thinking (summary): {text}", flush=True)

    def __exit__(self, kind, value, traceback):
        status = "response received" if kind is None else "request interrupted or failed"
        suffix = "" if self.lines else "; no reasoning summary returned"
        print(f"{self.prefix} {status}{suffix}", flush=True)
