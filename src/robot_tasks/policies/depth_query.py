"""Optional read-only pixel queries on the existing JSON decision interface."""

from copy import deepcopy
import json

from ..cameras.depth import MAX_DEPTH_QUERIES, validate_depth_queries
from ..types import DepthQuery, InvalidPolicyAction
from .action_codec import _unique_object


def with_depth_queries(schema: dict, cameras: tuple[str, ...]) -> dict:
    result = deepcopy(schema)
    result["required"].append("depth_queries")
    result["properties"]["depth_queries"] = {
        "type": "array", "maxItems": MAX_DEPTH_QUERIES, "items": {
            "type": "object", "additionalProperties": False, "required": ["camera", "u", "v"],
            "properties": {"camera": {"type": "string", "enum": list(cameras)},
                           "u": {"type": "integer", "minimum": 0},
                           "v": {"type": "integer", "minimum": 0}}}}
    return result


def split_depth_reply(text: str, image_sizes: dict[str, tuple[int, int]]) -> tuple[str, tuple[DepthQuery, ...]]:
    try:
        document = json.loads(text, object_pairs_hook=_unique_object)
        if not isinstance(document, dict) or set(document) != {"commands", "request_stop", "depth_queries"}:
            raise ValueError("RGB-D decisions require commands, request_stop and depth_queries")
        rows = document.pop("depth_queries")
        if not isinstance(rows, list) or len(rows) > MAX_DEPTH_QUERIES:
            raise ValueError("Invalid depth query count")
        if rows and (document["commands"] != [] or document["request_stop"] is not False):
            raise ValueError("Depth queries cannot accompany motion, waiting or stop; use commands=[] and request_stop=false")
        queries = []
        for row in rows:
            if (not isinstance(row, dict) or set(row) != {"camera", "u", "v"}
                    or not isinstance(row["camera"], str) or row["camera"] not in image_sizes):
                raise ValueError("Depth query requires camera, u and v in a current RGB view")
            queries.append(DepthQuery(row["camera"], row["u"], row["v"], *image_sizes[row["camera"]]))
        queries = tuple(queries)
        validate_depth_queries(queries, tuple(image_sizes))
        return json.dumps(document, allow_nan=False), queries
    except (ValueError, TypeError, KeyError) as exc:
        raise InvalidPolicyAction(str(exc)) from exc
