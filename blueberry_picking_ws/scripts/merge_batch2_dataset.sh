#!/usr/bin/env bash
# Merge batch1 (images/) + batch2 (images_batch2/) without filename collisions.
# Renames batch1 -> b1_frame_XXXX, batch2 -> b2_frame_XXXX in images/ + labels/.
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMG="${ROOT}/datasets/blueberry/images"
LAB="${ROOT}/datasets/blueberry/labels"
IMG_B2="${ROOT}/datasets/blueberry/images_batch2"
LAB_B2="${ROOT}/datasets/blueberry/labels_batch2"
STAGING="${ROOT}/datasets/blueberry/_merge_staging"

rm -rf "${STAGING}"
mkdir -p "${STAGING}/images" "${STAGING}/labels"

shopt -s nullglob
n1=0
for img in "${IMG}"/*.png "${IMG}"/*.jpg "${IMG}"/*.jpeg; do
  [[ -f "${img}" ]] || continue
  base="$(basename "${img}")"
  # Already prefixed — keep as-is
  if [[ "${base}" == b1_* ]] || [[ "${base}" == b2_* ]]; then
    cp "${img}" "${STAGING}/images/${base}"
    stem="${base%.*}"
    [[ -f "${LAB}/${stem}.txt" ]] && cp "${LAB}/${stem}.txt" "${STAGING}/labels/${stem}.txt"
  else
    stem="${base%.*}"
    ext="${base##*.}"
    new="b1_${base}"
    cp "${img}" "${STAGING}/images/${new}"
    if [[ -f "${LAB}/${stem}.txt" ]]; then
      cp "${LAB}/${stem}.txt" "${STAGING}/labels/b1_${stem}.txt"
    fi
  fi
  n1=$((n1 + 1))
done

n2=0
for img in "${IMG_B2}"/*; do
  [[ -f "${img}" ]] || continue
  base="$(basename "${img}")"
  stem="${base%.*}"
  ext="${base##*.}"
  new="b2_${base}"
  cp "${img}" "${STAGING}/images/${new}"
  if [[ -f "${LAB_B2}/${stem}.txt" ]]; then
    cp "${LAB_B2}/${stem}.txt" "${STAGING}/labels/b2_${stem}.txt"
  fi
  n2=$((n2 + 1))
done

rm -f "${IMG}"/* "${LAB}"/*
mv "${STAGING}/images"/* "${IMG}/"
mv "${STAGING}/labels"/* "${LAB}/" 2>/dev/null || true
rmdir "${STAGING}/images" "${STAGING}/labels" "${STAGING}" 2>/dev/null || true

echo "[merge] batch1 files: ${n1} (prefix b1_)"
echo "[merge] batch2 files: ${n2} (prefix b2_)"
echo "[merge] total images: $(ls "${IMG}" | wc -l)"
echo ""
echo "[merge] NOTE: batch1 labels may have been overwritten earlier."
echo "[merge] Re-prelabel batch1 if needed:"
echo "  bash scripts/run_prelabel_blueberry.sh --custom-model \\"
echo "    --model runs/detect/blueberry/weights/best.pt --conf 0.25 --overwrite"
echo "[merge] Then: bash scripts/run_local_yolo_pipeline.sh prepare && ... train"
