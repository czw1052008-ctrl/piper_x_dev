#!/usr/bin/env bash
# Merge batch3 (images_batch3/) into datasets/blueberry/images + labels (prefix b3_).
# Run after manual annotation review. Existing b1_/b2_ files in images/ are kept.
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMG="${ROOT}/datasets/blueberry/images"
LAB="${ROOT}/datasets/blueberry/labels"
IMG_B3="${ROOT}/datasets/blueberry/images_batch3"
LAB_B3="${ROOT}/datasets/blueberry/labels_batch3"

if [[ ! -d "${IMG_B3}" ]] || [[ -z "$(ls -A "${IMG_B3}" 2>/dev/null)" ]]; then
  echo "[merge] ERROR: no images in ${IMG_B3}" >&2
  exit 1
fi

mkdir -p "${IMG}" "${LAB}"

shopt -s nullglob
n3=0
for img in "${IMG_B3}"/*; do
  [[ -f "${img}" ]] || continue
  base="$(basename "${img}")"
  stem="${base%.*}"
  ext="${base##*.}"
  if [[ "${base}" == b3_* ]]; then
    new="${base}"
    new_stem="${stem}"
  else
    new="b3_${base}"
    new_stem="b3_${stem}"
  fi
  cp "${img}" "${IMG}/${new}"
  if [[ -f "${LAB_B3}/${stem}.txt" ]]; then
    cp "${LAB_B3}/${stem}.txt" "${LAB}/${new_stem}.txt"
  elif [[ -f "${LAB_B3}/${new_stem}.txt" ]]; then
    cp "${LAB_B3}/${new_stem}.txt" "${LAB}/${new_stem}.txt"
  else
    echo "[merge] WARN: no label for ${base}" >&2
  fi
  n3=$((n3 + 1))
done

echo "[merge] batch3 merged: ${n3} images (prefix b3_)"
echo "[merge] total images: $(ls "${IMG}" | wc -l)"
echo "[merge] Next: python3 scripts/prepare_yolo_dataset.py --val-count 15"
