#!/usr/bin/env bash
# Download FoundationPose scorer + refiner weights (Google Drive).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FP_ROOT="${FOUNDATIONPOSE_ROOT:-$ROOT/FoundationPose}"
WEIGHTS_DIR="$FP_ROOT/weights"

download_one() {
  local id="$1"
  local out="$2"
  if [[ -f "$out" ]]; then
    echo "exists: $out"
    return 0
  fi
  mkdir -p "$(dirname "$out")"
  echo "downloading $out ..."
  for attempt in 1 2 3; do
    if gdown "$id" -O "$out"; then
      return 0
    fi
    echo "retry $attempt/3 in 15s ..."
    sleep 15
  done
  return 1
}

echo "Target: $WEIGHTS_DIR"
mkdir -p "$WEIGHTS_DIR"

if ! command -v gdown >/dev/null 2>&1; then
  python3 -m pip install --user gdown
  export PATH="${HOME}/.local/bin:${PATH}"
fi

# File IDs from NVlabs FoundationPose Google Drive folder
download_one 1477-st1s1TxXN6oqfM5ZnsQwd8BCzVg1 \
  "$WEIGHTS_DIR/2023-10-28-18-33-37/config.yml"
download_one 1E9FPB5WFIBMLrOJqZLpoVOK4Mjzrrxhv \
  "$WEIGHTS_DIR/2023-10-28-18-33-37/model_best.pth"
download_one 1kQkQG-q_VvLRozv30hyeLB7P_jiEEqiE \
  "$WEIGHTS_DIR/2024-01-11-20-02-45/config.yml"
download_one 1Zdjnkn4EHOI5_k08apofwRgTjWpai4E4 \
  "$WEIGHTS_DIR/2024-01-11-20-02-45/model_best.pth"

for stamp in 2024-01-11-20-02-45 2023-10-28-18-33-37; do
  if [[ ! -f "$WEIGHTS_DIR/$stamp/model_best.pth" ]]; then
    echo "MISSING: $WEIGHTS_DIR/$stamp/model_best.pth"
    echo "Google Drive quota may block automated download; fetch manually from readme link."
    exit 1
  fi
  echo "OK: $stamp"
done

echo "Weights ready under $WEIGHTS_DIR"
