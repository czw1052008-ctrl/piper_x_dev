#!/usr/bin/env bash
# Local bbox annotation (needs GUI / WSLg).
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "${ROOT}/datasets/blueberry/images" "${ROOT}/datasets/blueberry/labels"
exec python3 "${ROOT}/scripts/annotate_blueberry.py" "$@"
