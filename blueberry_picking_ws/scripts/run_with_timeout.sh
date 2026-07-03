#!/usr/bin/env bash
# Run a command with a hard wall-clock timeout (default 2 minutes).
set -euo pipefail

DEFAULT_TIMEOUT_SEC=120
TIMEOUT_SEC="${1:-$DEFAULT_TIMEOUT_SEC}"
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
  shift
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 [timeout_sec] command [args...]" >&2
  echo "Default timeout: ${DEFAULT_TIMEOUT_SEC}s (2 min)" >&2
  exit 2
fi

echo "[run_with_timeout] limit=${TIMEOUT_SEC}s cmd=$*"
exec timeout --foreground "${TIMEOUT_SEC}" "$@"
