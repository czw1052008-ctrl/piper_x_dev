#!/usr/bin/env bash
# DEPRECATED — use run_pick_system.sh (docs/PICK_CYCLE.md).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "[DEPRECATED] Use: bash scripts/run_pick_system.sh" >&2
exec bash "${ROOT}/scripts/run_pick_system.sh" "$@"
