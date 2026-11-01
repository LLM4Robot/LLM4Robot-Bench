"""Debug GPT pick/place in one Windows process from a WSL editor.

The normal check command keeps its subprocess watchdog.
This development entry waits for attachment before opening Isaac Sim.
"""

import argparse
import ipaddress
from pathlib import Path
import sys

from ..paths import PROJECT_ROOT
from .registry import CHECKS, get_check


def main(argv: list[str] | None = None) -> int:
    from ..runtime import configure_windows_storage
    configure_windows_storage()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="默认使用 GPT 抓放任务配置")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "runs/debug")
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5678)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--check", choices=tuple(CHECKS), default="gpt-pick-place")
    parser.add_argument("--model-config", type=Path)
    args = parser.parse_args(argv)
    spec = get_check(args.check)
    try:
        spec.validate_options(gui=args.gui, model_config=args.model_config)
        config = spec.load_config(args.config, debug=True)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if sys.platform != "win32":
        parser.error("Use the Windows isaac51 Python through tools/run_r5_check.ps1 -Debugger")
    address = ipaddress.ip_address(args.listen_host)
    if address.is_unspecified or not (address.is_private or address.is_loopback):
        parser.error("Bind only to loopback or the Windows WSL virtual adapter address")
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")

    print("R5_DEBUG_STARTING", flush=True)
    try:
        import debugpy
    except ImportError:
        parser.error("Install debugpy in the Windows isaac51 environment first")
    debugpy.configure(subProcess=False)
    debugpy.listen((str(address), args.port))
    # VS Code's background task becomes ready only after the socket is listening.
    print(f"R5_DEBUG_READY {address}:{args.port}", flush=True)
    debugpy.wait_for_client()

    # Stops here automatically. F11 on the following call enters the real check.
    debugpy.breakpoint()
    _, result = spec.run(config.source, args.output, gui=args.gui, model_config=args.model_config)
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
