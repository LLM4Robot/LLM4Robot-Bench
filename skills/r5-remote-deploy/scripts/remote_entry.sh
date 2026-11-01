#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
runtime_root="${GPT_BENCHMARK_RUNTIME_ROOT:-$(dirname "$project_root")/runtime_combined}"
bridge_token="$runtime_root/bridge.token"
export PYTHONPATH="$project_root/src"
export PYTHONUNBUFFERED=1
export GPT_BENCHMARK_PYTHON="$runtime_root/bin/isaac-python"
export GPT_BENCHMARK_PLANNER_PYTHON="$runtime_root/bin/planner-python"
export GPT_BENCHMARK_RUNTIME_ROOT="$runtime_root"
if [[ "${GPT_BENCHMARK_USE_LOCAL_CODEX:-1}" == 1 ]]; then
    export GPT_BENCHMARK_CODEX_BRIDGE_URL="${GPT_BENCHMARK_CODEX_BRIDGE_URL:-http://127.0.0.1:18469}"
    export GPT_BENCHMARK_CODEX_BRIDGE_TOKEN_FILE="${GPT_BENCHMARK_CODEX_BRIDGE_TOKEN_FILE:-$bridge_token}"
else
    unset GPT_BENCHMARK_CODEX_BRIDGE_URL GPT_BENCHMARK_CODEX_BRIDGE_TOKEN_FILE
fi
export XDG_CACHE_HOME="$runtime_root/cache/$(hostname)"
export TMPDIR="$runtime_root/tmp/$(hostname)"
export TORCH_EXTENSIONS_DIR="$runtime_root/torch_extensions/$(hostname)"
export GPT_BENCHMARK_PLANNER_CACHE="$runtime_root/cache/$(hostname)/planner"
export GPT_BENCHMARK_ASSET_CACHE="$runtime_root/cache/$(hostname)/generated_assets"
export GPT_BENCHMARK_PLANNER_REFERENCE="$project_root/assets/r5/planner_reference/geometry.json"
export OMNI_KIT_ACCEPT_EULA=YES
export ACCEPT_EULA=Y
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR" "$TORCH_EXTENSIONS_DIR"
cd "$project_root"
exec "$runtime_root/framework/bin/python" "$@"
