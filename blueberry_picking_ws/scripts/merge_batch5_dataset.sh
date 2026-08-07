#!/usr/bin/env bash
# Merge batch5 (after-center wrist) into datasets/blueberry/images + labels (prefix b5_).
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMG="${ROOT}/datasets/blueberry/images"
LAB="${ROOT}/datasets/blueberry/labels"
IMG_B5="${ROOT}/datasets/blueberry/images_batch5"
LAB_B5="${ROOT}/datasets/blueberry/labels_batch5"

if [[ ! -d "${IMG_B5}" ]] || [[ -z "$(ls -A "${IMG_B5}"/*.png 2>/dev/null)" ]]; then
  echo "[merge] ERROR: no images in ${IMG_B5}" >&2
  exit 1
fi

mkdir -p "${IMG}" "${LAB}"

shopt -s nullglob
n5=0
missing=0
for img in "${IMG_B5}"/*.png; do
  base="$(basename "${img}")"
  stem="${base%.*}"
  if [[ "${base}" == b5_* ]]; then
    new="${base}"
    new_stem="${stem}"
  else
    new="b5_${base}"
    new_stem="b5_${stem}"
  fi
  cp "${img}" "${IMG}/${new}"
  if [[ -f "${LAB_B5}/${stem}.txt" ]]; then
    cp "${LAB_B5}/${stem}.txt" "${LAB}/${new_stem}.txt"
  else
    echo "[merge] WARN: no label for ${base}" >&2
    missing=$((missing + 1))
  fi
  n5=$((n5 + 1))
done

echo "[merge] batch5 merged: ${n5} images (missing labels: ${missing})"
echo "[merge] total images: $(ls "${IMG}"/*.png 2>/dev/null | wc -l)"
echo "[merge] Next: python3 scripts/prepare_yolo_dataset.py --val-count 20"
