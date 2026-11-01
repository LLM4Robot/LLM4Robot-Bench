#!/usr/bin/env bash
# Compatibility for existing measured-state recovery commands.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$project_root/skills/r5-remote-deploy/scripts/recovery_planner.sh" "$@"
