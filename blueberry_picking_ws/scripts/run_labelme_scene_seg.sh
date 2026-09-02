#!/usr/bin/env bash
# Edit scene_seg masks in Labelme (berry / branch / rigid / ego).
#
#   bash scripts/run_labelme_scene_seg.sh          # export + open GUI
#   bash scripts/run_labelme_scene_seg.sh --sync # after editing: write YOLO-seg back
#
# Labelme: Ctrl+S 保存；D 下一张；A 上一张；Ctrl+J 切多边形；右侧选 berry/branch/rigid/ego。
set -eo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${ROOT}/datasets/scene_seg"
LABELME_DIR="${DATA}/labelme"
PY="${LABELME_PYTHON:-/usr/bin/python3}"

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  conda deactivate 2>/dev/null || true
fi
export PATH="/usr/bin:/bin:${HOME}/.local/bin:${PATH}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
export DISPLAY="${DISPLAY:-:1}"

SYNC_ONLY=false
LAUNCH_ONLY=false
for arg in "$@"; do
  case "${arg}" in
    --sync) SYNC_ONLY=true ;;
    --launch) LAUNCH_ONLY=true ;;
  esac
done

if ! "${PY}" -c 'import labelme' 2>/dev/null; then
  echo "[labelme] installing labelme for ${PY} ..."
  "${PY}" -m pip install --user -q 'labelme>=5.4' || {
    echo "[labelme] pip install failed" >&2
    exit 1
  }
fi

if [[ "${SYNC_ONLY}" == true ]]; then
  exec "${PY}" "${ROOT}/scripts/labelme_to_yolo_seg.py" --data "${DATA}" --src "${LABELME_DIR}"
fi

"${PY}" "${ROOT}/scripts/yolo_seg_to_labelme.py" --data "${DATA}" --out "${LABELME_DIR}"

if [[ "${LAUNCH_ONLY}" == true ]]; then
  exit 0
fi

echo "[labelme] 打开 ${LABELME_DIR}"
echo "  标签只能用 berry / branch / rigid / ego"
echo "  细枝沿线画；果贴果皮；硬物（盆/桌/电钻/显示器）用 rigid"
echo "  吸杯/夹爪/可见连杆用 ego，不要标成 rigid"
echo "  改完后必须显式写回: bash scripts/run_labelme_scene_seg.sh --sync"
echo "  禁止关窗口自动 sync（会用旧 json 覆盖 annotate_scene_seg 的 mask）"

"${PY}" "${ROOT}/scripts/launch_labelme.py" "${LABELME_DIR}" \
  --labels "${LABELME_DIR}/labels.txt" \
  --validate-label exact
