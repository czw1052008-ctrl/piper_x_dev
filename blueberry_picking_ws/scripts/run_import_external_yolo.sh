#!/usr/bin/env bash
# Import Roboflow / other YOLO export into datasets/blueberry/
# Usage:
#   bash scripts/run_import_external_yolo.sh datasets/blueberry/imports/my_set
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "${ROOT}/scripts/import_external_yolo_dataset.py" "$@"
