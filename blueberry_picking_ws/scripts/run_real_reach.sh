#!/usr/bin/env bash
# Bring up real-robot stack + reach perception + reach FSM.
# Docs: docs/REACH_PIPELINE.md
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${ROOT}/config/real_robot.env"
DRY_RUN=false
EXTRA_BRINGUP=()
ALIGN_JUDGE_MODE="heuristic"
ALIGN_VLM_PROVIDER="heuristic"
ALIGN_QA_DIR=""

usage() {
  cat <<EOF
Usage: bash scripts/run_real_reach.sh [--dry-run] [--config PATH]
                                  [--align-judge-mode heuristic|file]
                                  [--align-vlm-provider heuristic]
                                  [--qa-dir PATH]

Starts: arm + Orbbec + fixed cam + global/fine detectors + reach_fsm.
Then wait for /reach/cmd (start / confirm_reset).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=true; shift ;;
    --config) CONFIG="$2"; shift 2 ;;
    --align-judge-mode) ALIGN_JUDGE_MODE="$2"; shift 2 ;;
    --align-vlm-provider) ALIGN_VLM_PROVIDER="$2"; shift 2 ;;
    --qa-dir) ALIGN_QA_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) EXTRA_BRINGUP+=("$1"); shift ;;
  esac
done

if [[ -f "${CONFIG}" ]]; then
  # shellcheck disable=SC1090
  source "${CONFIG}"
fi

bash "${ROOT}/scripts/real_robot_bringup.sh" \
  --config "${CONFIG}" \
  --fixed-cam \
  --reach-perception \
  --no-wait \
  "${EXTRA_BRINGUP[@]+"${EXTRA_BRINGUP[@]}"}"

set +u
# shellcheck disable=SC1090
source "${ROOT}/scripts/setup_env.sh" >/dev/null
if [[ -f "${AGX_ARM_WS:-}/install/setup.bash" ]]; then
  # shellcheck disable=SC1090
  source "${AGX_ARM_WS}/install/setup.bash"
fi
set -u 2>/dev/null || true

EE="${PICK_EE_LINK:-tcp_link}"
CUP="${PICK_CUP_CONTACT_OFFSET:-0.015}"
DRY_FLAG=()
[[ "${DRY_RUN}" == "true" ]] && DRY_FLAG=(--dry-run)
QA_DIR_DEFAULT="${ROOT}/log/real_robot/qa"
QA_DIR="${ALIGN_QA_DIR:-${QA_DIR_DEFAULT}}"
mkdir -p "${QA_DIR}"

if [[ "${ALIGN_JUDGE_MODE}" == "file" ]]; then
  echo "[run_real_reach] starting align_vlm_decider provider=${ALIGN_VLM_PROVIDER} qa_dir=${QA_DIR}"
  python3 "${ROOT}/scripts/align_vlm_decider.py" \
    --qa-dir "${QA_DIR}" \
    --provider "${ALIGN_VLM_PROVIDER}" \
    > "${ROOT}/log/real_robot/align_vlm_decider.log" 2>&1 &
fi

cat <<EOF

================================================================================
  Reach FSM starting.
  ros2 topic echo /reach/status
  ros2 topic pub --once /reach/cmd std_msgs/String "{data: start}"
  # after REACHED:
  ros2 topic pub --once /reach/cmd std_msgs/String "{data: confirm_reset}"
  Stop stack: bash scripts/real_robot_shutdown.sh
================================================================================

EOF

# Ensure arm accepts external trajectories (same as teleop bring-up).
if ! timeout 12 ros2 service call /enable_agx_arm std_srvs/srv/SetBool "{data: true}" 2>&1 | grep -q 'success=True'; then
  echo "[run_real_reach] WARN: enable_agx_arm failed — arm may not move until enabled manually" >&2
fi
timeout 8 ros2 service call /control_enable std_srvs/srv/SetBool "{data: true}" 2>/dev/null || true

exec python3 "${ROOT}/scripts/reach_fsm_node.py" \
  --ee-link "${EE}" \
  --cup-contact-offset "${CUP}" \
  --align-judge-mode "${ALIGN_JUDGE_MODE}" \
  --qa-dir "${QA_DIR}" \
  "${DRY_FLAG[@]}"
