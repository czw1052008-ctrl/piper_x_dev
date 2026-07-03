#!/usr/bin/env bash
# Auto YOLO bbox pre-label (YOLOE open-vocab) — then review in annotate GUI.
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${FOUNDATIONPOSE_CONDA_ENV:-foundationpose}"
mkdir -p "${ROOT}/datasets/blueberry/images" "${ROOT}/datasets/blueberry/labels"

set +u
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"

export PYTHONPATH="${ROOT}/src/picking_perception${PYTHONPATH:+:${PYTHONPATH}}"
exec python3 "${ROOT}/scripts/prelabel_blueberry.py" "$@"
