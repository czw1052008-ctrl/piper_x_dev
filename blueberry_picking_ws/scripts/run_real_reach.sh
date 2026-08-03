#!/usr/bin/env bash
# Bring up real-robot stack + reach perception + reach FSM.
# Docs: docs/REACH_PIPELINE.md
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${ROOT}/config/real_robot.env"
DRY_RUN=false
EXTRA_BRINGUP=()

usage() {
  cat <<EOF
Usage: bash scripts/run_real_reach.sh [--dry-run] [--config PATH]

Starts: arm + Orbbec + fixed cam + global/fine detectors + reach_fsm.
Then wait for /reach/cmd (start / confirm_reset).
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=true; shift ;;
    --config) CONFIG="$2"; shift 2 ;;
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
CUP="${PICK_CUP_CONTACT_OFFSET:-0.04}"
DRY_FLAG=()
[[ "${DRY_RUN}" == "true" ]] && DRY_FLAG=(--dry-run)

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

exec python3 "${ROOT}/scripts/reach_fsm_node.py" \
  --ee-link "${EE}" \
  --cup-contact-offset "${CUP}" \
  "${DRY_FLAG[@]}"
