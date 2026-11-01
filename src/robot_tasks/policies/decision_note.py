"""Optional decision annotation, separated from executable action decoding."""

from copy import deepcopy
import json

from ..types import InvalidPolicyAction
from .action_codec import _unique_object


MAX_NOTE_LENGTH = 512


def with_decision_note(schema: dict) -> dict:
    result = deepcopy(schema)
    result["required"].append("note")
    result["properties"]["note"] = {
        "type": "string", "minLength": 1, "maxLength": MAX_NOTE_LENGTH,
        "description": "One or two brief sentences describing the visible situation and purpose of this decision.",
    }
    return result


def split_decision_note(text: str) -> tuple[str, str]:
    """Remove only the annotation; the existing decoder validates all other fields."""
    try:
        document = json.loads(text, object_pairs_hook=_unique_object)
        if not isinstance(document, dict) or "note" not in document:
            raise ValueError("Every decision requires a brief note")
        note = document.pop("note")
        if not isinstance(note, str) or not note.strip() or len(note) > MAX_NOTE_LENGTH:
            raise ValueError(f"note must be a nonempty string of at most {MAX_NOTE_LENGTH} characters")
        if any(not c.isprintable() and c not in "\n\r\t" for c in note):
            raise ValueError("note contains unsupported control characters")
        return json.dumps(document, allow_nan=False), note.strip()
    except (ValueError, TypeError) as exc:
        raise InvalidPolicyAction(str(exc)) from exc
