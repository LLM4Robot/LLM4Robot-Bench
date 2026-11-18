"""Use the R5 System 2 simulation bridge and replay recorded L3-04 actions."""

import argparse
import base64
import json
from pathlib import Path
import re
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener


class McpClient:
    def __init__(self, url, timeout):
        self.url = url
        self.timeout = timeout
        self.opener = build_opener(ProxyHandler({}))
        self.next_id = 1
        self.headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
        }
        result = self.send("initialize", {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "llm4robot-harness-v1", "version": "1.0"},
        })
        self.headers["MCP-Protocol-Version"] = result["protocolVersion"]
        self.send("notifications/initialized", notify=True)

    def send(self, method, params=None, *, notify=False):
        payload = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        if not notify:
            payload["id"] = self.next_id
            self.next_id += 1
        request = Request(self.url, data=json.dumps(payload, allow_nan=False).encode(),
                          headers=self.headers, method="POST")
        with self.opener.open(request, timeout=self.timeout) as response:
            if session_id := response.headers.get("Mcp-Session-Id"):
                self.headers["Mcp-Session-Id"] = session_id
            if notify:
                return None
            reply = json.load(response)
        if reply.get("id") != payload["id"]:
            raise ValueError(f"Unexpected response ID for {method}")
        if "error" in reply:
            raise ValueError(f"{method}: {reply['error']}")
        return reply["result"]


def summarize_call(result, image_dir):
    """Keep structured text and decode the bridge's named JPEG blocks."""
    output = {"is_error": result.get("isError", False)}
    if "structuredContent" in result:
        output["result"] = result["structuredContent"]
    texts = []
    images = {}
    camera = None
    for block in result.get("content", []):
        if block["type"] == "text":
            content = block["text"]
            if content.startswith("camera: "):
                camera = content.removeprefix("camera: ")
            else:
                texts.append(content)
        elif block["type"] == "image" and image_dir is not None:
            if block["mimeType"] != "image/jpeg":
                raise ValueError(f"Unexpected camera image format: {block['mimeType']}")
            camera = camera or f"image_{len(images) + 1}"
            filename = re.sub(r"[^A-Za-z0-9_-]", "_", camera)
            image_dir.mkdir(parents=True, exist_ok=True)
            path = image_dir / f"{filename}.jpg"
            path.write_bytes(base64.b64decode(block["data"], validate=True))
            images[camera] = str(path.resolve())
            camera = None
    if "result" not in output:
        output["text"] = texts
    if images:
        output["images"] = images
    return output


def call_tool(client, name, arguments):
    return client.send("tools/call", {"name": name, "arguments": arguments})


def terminal_job(client, raw):
    # Motion tools can mark a planner rejection isError while still returning a
    # terminal job with a useful structured result.
    job = raw.get("structuredContent", {}).get("job")
    if not job:
        raise ValueError("Motion response did not contain a job. Inspect the simulator before continuing.")
    deadline = time.monotonic() + client.timeout
    while job.get("active"):
        job_id = job["id"]
        if time.monotonic() >= deadline:
            raise ValueError(f"Job {job_id} still active. Inspect it before sending another motion.")
        time.sleep(0.2)
        raw = call_tool(client, "get_job", {"job_id": job_id, "cameras": []})
        job = raw.get("structuredContent", {}).get("job")
        if not job:
            raise ValueError(f"Could not confirm that job {job_id} ended.")
    if not job.get("result"):
        raise ValueError(f"Job {job.get('id')} ended without an action result.")
    return job["result"]


def replay(client, recording):
    actions = recording["actions"]
    allowed_tools = {"cap_move_to_pose", "cap_set_gripper"}
    for index, action in enumerate(actions, 1):
        if action["index"] != index or action["tool"] not in allowed_tools:
            raise ValueError(f"Invalid recorded action {index}.")
        if action["arguments"].get("arm") != "left":
            raise ValueError(f"Action {index} does not target the designated manipulation arm.")
    tools = {tool["name"] for tool in client.send("tools/list")["tools"]}
    missing = (allowed_tools | {"get_job", "r5_task"}) - tools
    if missing:
        raise ValueError(f"Missing simulation tools: {', '.join(sorted(missing))}.")
    task = call_tool(client, "r5_task", {}).get("structuredContent", {}).get("task")
    if task != recording["task_id"]:
        raise ValueError(f"Expected task {recording['task_id']}, received {task!r}.")

    for action in actions:
        result = terminal_job(client, call_tool(client, action["tool"], action["arguments"]))
        status = result.get("status")
        steps = result.get("steps_executed")
        print(json.dumps({
            "index": action["index"],
            "tool": action["tool"],
            "status": status,
            "steps_executed": steps,
            "expected_status": action["expected_status"],
            "recorded_steps": action["steps_executed"],
        }), flush=True)
        if status != action["expected_status"] or (status == "planner_failed" and steps != 0):
            raise ValueError(f"Action {action['index']} differs from the recording. Replay stopped.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Forwarded MCP URL, e.g. http://127.0.0.1:8765/mcp/")
    parser.add_argument("--timeout", type=float, default=120, help="Request timeout in seconds")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List the live MCP tools and input schemas")
    call = commands.add_parser("call", help="Call a live MCP tool")
    call.add_argument("name")
    call.add_argument("arguments", nargs="?", default="{}", help="JSON object")
    call.add_argument("--images", type=Path, help="Directory for returned camera JPEGs")
    replay_parser = commands.add_parser("replay", help="Replay the recorded actions against a freshly reset simulator")
    replay_parser.add_argument("--recording", type=Path,
                               default=Path(__file__).with_name("l3_04_seed0_actions.json"))
    args = parser.parse_args()
    try:
        if args.command == "call":
            arguments = json.loads(args.arguments)
            if not isinstance(arguments, dict):
                raise ValueError("Tool arguments must be a JSON object")
        client = McpClient(args.url, args.timeout)
        if args.command == "list":
            print(json.dumps(client.send("tools/list"), ensure_ascii=False))
            return 0
        if args.command == "replay":
            recording = json.loads(args.recording.read_text(encoding="utf-8"))
            return replay(client, recording)
        result = call_tool(client, args.name, arguments)
        output = summarize_call(result, args.images)
        print(json.dumps(output, ensure_ascii=False))
        return 2 if output["is_error"] else 0
    except HTTPError as error:
        print(f"MCP HTTP {error.code}: {error.read(800).decode(errors='replace')}", file=sys.stderr)
    except (URLError, OSError, KeyError, ValueError) as error:
        print(f"MCP request failed: {error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
