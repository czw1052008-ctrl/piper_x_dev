#!/usr/bin/env bash
# Merge batch4 (images_batch4/) into datasets/blueberry/images + labels (prefix b4_).
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMG="${ROOT}/datasets/blueberry/images"
LAB="${ROOT}/datasets/blueberry/labels"
IMG_B4="${ROOT}/datasets/blueberry/images_batch4"
LAB_B4="${ROOT}/datasets/blueberry/labels_batch4"

if [[ ! -d "${IMG_B4}" ]] || [[ -z "$(ls -A "${IMG_B4}"/*.png 2>/dev/null)" ]]; then
  echo "[merge] ERROR: no images in ${IMG_B4}" >&2
  exit 1
fi

mkdir -p "${IMG}" "${LAB}"

shopt -s nullglob
n4=0
for img in "${IMG_B4}"/*.png; do
  base="$(basename "${img}")"
  stem="${base%.*}"
  if [[ "${base}" == b4_* ]]; then
    new="${base}"
    new_stem="${stem}"
  else
    new="b4_${base}"
    new_stem="b4_${stem}"
  fi
  cp "${img}" "${IMG}/${new}"
  if [[ -f "${LAB_B4}/${stem}.txt" ]]; then
    cp "${LAB_B4}/${stem}.txt" "${LAB}/${new_stem}.txt"
  else
    echo "[merge] WARN: no label for ${base}" >&2
  fi
  n4=$((n4 + 1))
done

echo "[merge] batch4 merged: ${n4} images"
echo "[merge] total images: $(ls "${IMG}"/*.png 2>/dev/null | wc -l)"
echo "[merge] Next: python3 scripts/prepare_yolo_dataset.py --val-count 20"
