#!/usr/bin/env bash
# Compatibility for existing SSH commands and runtime wrappers.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
exec bash "$project_root/skills/r5-remote-deploy/scripts/remote_entry.sh" "$@"
