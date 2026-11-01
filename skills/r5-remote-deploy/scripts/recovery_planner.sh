#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
runtime_root="${GPT_BENCHMARK_RUNTIME_ROOT:-$(dirname "$project_root")/runtime_combined}"
# Isaac uses explicit global GPU indices; the planner selects the matching UUID.
export CUDA_VISIBLE_DEVICES="${GPT_BENCHMARK_RECOVERY_GPU_UUID:?GPU UUID missing}"
exec "$runtime_root/bin/planner-python" "$@"
