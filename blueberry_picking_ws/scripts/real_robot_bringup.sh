#!/usr/bin/env bash
# One-shot Piper X real-robot bringup: CAN + arm/MoveIt + joint_states relay + Orbbec + optional perception/teleop.
#
# Usage:
#   cp config/real_robot.env.example config/real_robot.env   # once
#   bash scripts/real_robot_bringup.sh                       # stack in background
#   bash scripts/real_robot_bringup.sh --teleop              # stack + keyboard teleop (foreground)
#   bash scripts/real_robot_shutdown.sh                      # stop + disable arm
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${ROOT}/config/real_robot.env"
PIDFILE="${ROOT}/log/real_robot/stack.pids"

ENABLE_CAMERA=true
ENABLE_ARM=true
ENABLE_RELAY=true
ENABLE_PERCEPTION=""
ENABLE_TELEOP=""
USBIP_ATTACH=""
FOREGROUND_WAIT=true
NO_WAIT=false

usage() {
  sed -n '2,12p' "$0"
  echo ""
  echo "Options:"
  echo "  --config PATH       Config file (default: config/real_robot.env)"
  echo "  --teleop            Start link6 keyboard teleop in foreground after stack is up"
  echo "  --perception        Enable fine_detector_node (FoundationPose)"
  echo "  --no-wait           Start stack in background and exit (for run_real_suction_pick.sh)"
  echo "  --no-camera         Skip Orbbec driver"
  echo "  --no-arm            Skip agx_arm (camera + perception only)"
  echo "  --camera-only       Same as --no-arm --perception"
  echo "  --attach-usb        Run usbipd attach for USBIP_CAN_BUSID (Windows PowerShell)"
  echo "  --check-only        Preflight checks, then exit"
  echo "  -h, --help          Show help"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --teleop) ENABLE_TELEOP=true; shift ;;
    --perception) ENABLE_PERCEPTION=true; shift ;;
    --no-wait) NO_WAIT=true; shift ;;
    --no-camera) ENABLE_CAMERA=false; shift ;;
    --no-arm) ENABLE_ARM=false; ENABLE_RELAY=false; shift ;;
    --camera-only) ENABLE_ARM=false; ENABLE_RELAY=false; ENABLE_PERCEPTION=true; shift ;;
    --attach-usb) USBIP_ATTACH=true; shift ;;
    --check-only) FOREGROUND_WAIT=false; CHECK_ONLY=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

# CLI flags must win over config/real_robot.env
_CLI_ENABLE_PERCEPTION="${ENABLE_PERCEPTION}"
_CLI_ENABLE_TELEOP="${ENABLE_TELEOP}"

# Defaults if no config file
PIPER_X_DEV="${HOME}/piper_x_dev"
AGX_ARM_WS="${PIPER_X_DEV}/agx_arm_ros"
ORBBEC_WS="${PIPER_X_DEV}/OrbbecSDK_ROS2_main"
PICKING_WS="${ROOT}"
CAN_INTERFACE=can0
CAN_BITRATE=1000000
USBIP_CAN_BUSID=1-5
USBIP_AUTO_ATTACH=false
ARM_CAN_PORT=can0
ARM_TYPE=piper_x
ARM_EFFECTOR=none
ARM_SPEED_PERCENT=20
ARM_AUTO_CONTROL_GATE=false
ARM_ENABLE_TIMEOUT=15.0
ARM_USE_RVIZ=false
ARM_TCP_OFFSET='[0.0,0.0,0.05,0.0,0.0,0.0]'
ORBBEC_LAUNCH=dabai.launch.py
ORBBEC_USB_PORT=
ORBBEC_CAMERA_NAME=camera_wrist
ORBBEC_DEPTH_REGISTRATION=true
ORBBEC_PUBLISH_TF=false
CAMERA_MOUNT_FRAME=link6
CAMERA_MOUNT_TX=0.0
CAMERA_MOUNT_TY=-0.08
CAMERA_MOUNT_TZ=0.0
CAMERA_MOUNT_RPY=0.0,0.0,0.0
ENABLE_PERCEPTION_CFG=false
FINE_PERCEPTION=foundation_pose
ENABLE_TELEOP_CFG=false
TELEOP_LINEAR_SPEED=0.03
TELEOP_ANGULAR_SPEED=10.0
LOG_DIR="${ROOT}/log/real_robot"

if [[ -f "${CONFIG}" ]]; then
  # shellcheck disable=SC1090
  source "${CONFIG}"
else
  echo "[bringup] WARN: ${CONFIG} not found — using built-in defaults."
  echo "[bringup]       cp config/real_robot.env.example config/real_robot.env"
fi

if [[ -n "${_CLI_ENABLE_PERCEPTION}" ]]; then
  ENABLE_PERCEPTION="${_CLI_ENABLE_PERCEPTION}"
fi
if [[ -n "${_CLI_ENABLE_TELEOP}" ]]; then
  ENABLE_TELEOP="${_CLI_ENABLE_TELEOP}"
fi

[[ "${ENABLE_PERCEPTION}" == "true" ]] && ENABLE_PERCEPTION_CFG=true
[[ "${ENABLE_TELEOP}" == "true" ]] && ENABLE_TELEOP_CFG=true
[[ "${USBIP_ATTACH}" == "true" ]] && USBIP_AUTO_ATTACH=true
[[ "${NO_WAIT}" == "true" ]] && FOREGROUND_WAIT=false

CHECK_ONLY="${CHECK_ONLY:-false}"

_log() { echo "[bringup] $*"; }
_die() { echo "[bringup] ERROR: $*" >&2; exit 1; }

_conda_warn() {
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    _log "WARN: conda active (${CONDA_PREFIX}) — deactivating for ROS nodes"
    conda deactivate 2>/dev/null || true
  fi
}

_source_ros_env() {
  _conda_warn
  export RMW_FASTRTPS_USE_SHM=0
  set +u
  # shellcheck disable=SC1091
  source "${ROOT}/scripts/setup_env.sh" >/dev/null
  if [[ -f "${AGX_ARM_WS}/install/setup.bash" ]]; then
    # shellcheck disable=SC1090
    source "${AGX_ARM_WS}/install/setup.bash"
  fi
  if [[ "${ENABLE_CAMERA}" == "true" && -f "${ORBBEC_WS}/install/setup.bash" ]]; then
    # shellcheck disable=SC1090
    source "${ORBBEC_WS}/install/setup.bash"
  fi
  set -u 2>/dev/null || true
}

_preflight() {
  [[ -f "${AGX_ARM_WS}/install/setup.bash" ]] || _die "agx_arm_ros not built: ${AGX_ARM_WS}/install/setup.bash"
  [[ -f "${ROOT}/install/setup.bash" ]] || _die "blueberry_picking_ws not built. Run: bash scripts/colcon_build.sh"
  if [[ "${ENABLE_CAMERA}" == "true" ]]; then
    [[ -f "${ORBBEC_WS}/install/setup.bash" ]] || _die "OrbbecSDK_ROS2 not built: ${ORBBEC_WS}/install/setup.bash"
  fi
  command -v ros2 >/dev/null || _die "ros2 not in PATH"
}

_usbip_attach() {
  if [[ "${USBIP_AUTO_ATTACH}" != "true" || -z "${USBIP_CAN_BUSID}" ]]; then
    return 0
  fi
  if ! command -v powershell.exe >/dev/null 2>&1; then
    _log "WARN: powershell.exe not found — skip usbipd auto-attach"
    return 0
  fi
  _log "Attaching USB-CAN via usbipd (busid ${USBIP_CAN_BUSID}) ..."
  powershell.exe -NoProfile -Command "usbipd attach --wsl --busid ${USBIP_CAN_BUSID}" \
    >/dev/null 2>&1 || _log "WARN: usbipd attach failed (run manually in Admin PowerShell)"
  sleep 2
}

_setup_can() {
  if [[ "${ENABLE_ARM}" != "true" ]]; then
    return 0
  fi
  _usbip_attach
  if ! ip link show "${CAN_INTERFACE}" &>/dev/null; then
    cat >&2 <<EOF
[bringup] ERROR: ${CAN_INTERFACE} not found.

Windows (Admin PowerShell), once per USB plug:
  usbipd list
  usbipd attach --wsl --busid ${USBIP_CAN_BUSID}

See: pyAgxArm/docs/wsl2_usb_can_guide.md
EOF
    exit 1
  fi
  local state
  state="$(ip -o link show "${CAN_INTERFACE}" | awk '{print $9}')"
  if [[ "${state}" != "UP" ]]; then
    _log "Bringing up ${CAN_INTERFACE} @ ${CAN_BITRATE} bps (sudo) ..."
    sudo ip link set "${CAN_INTERFACE}" down 2>/dev/null || true
    sudo ip link set "${CAN_INTERFACE}" up type can bitrate "${CAN_BITRATE}"
  else
    _log "${CAN_INTERFACE} already UP"
  fi
}

_start_bg() {
  local name="$1"
  local cmd="$2"
  local log_file="${LOG_DIR}/${name}.log"
  mkdir -p "${LOG_DIR}"
  _log "Starting ${name} ..."
  if [[ "${3:-}" == "truncate" ]]; then
    : > "${log_file}"
  fi
  setsid bash -c "${cmd}" >> "${log_file}" 2>&1 &
  local pid=$!
  echo "${name} ${pid}" >> "${PIDFILE}"
  _log "  pid=${pid}  log=${log_file}"
}

_wait_for_action() {
  local action="$1" timeout_sec="${2:-90}"
  local i=0
  _log "Waiting for action ${action} (up to ${timeout_sec}s) ..."
  while (( i < timeout_sec )); do
    if ros2 action list 2>/dev/null | grep -q "${action}"; then
      _log "Ready: action ${action}"
      return 0
    fi
    if (( i > 0 && i % 15 == 0 )); then
      _log "  still waiting for ${action} (${i}s) — check ${LOG_DIR}/arm.log if arm/CAN issue"
    fi
    sleep 1
    i=$((i + 1))
  done
  _die "Timeout waiting for action ${action} (see ${LOG_DIR}/arm.log). CAN/usbipd OK?"
}

_check_orbbec_usb() {
  if ! lsusb 2>/dev/null | grep -q '2bc5:0657'; then
    _die "Orbbec depth USB (0657) not in WSL. Attach both camera USB devices via usbipd."
  fi
  if ! lsusb 2>/dev/null | grep -q '2bc5:0557'; then
    _log "WARN: color USB (0557) not seen — depth-only; color topic may be missing"
  fi
}

_detect_orbbec_usb_port() {
  if [[ -n "${ORBBEC_USB_PORT:-}" ]]; then
    _log "Using ORBBEC_USB_PORT=${ORBBEC_USB_PORT}"
    return 0
  fi
  _log "Auto-detecting Orbbec usb_port ..."
  local detected
  detected="$(ros2 run orbbec_camera list_devices_node 2>&1 | grep -E 'usb port:' | awk '{print $NF}' | head -1 || true)"
  if [[ -z "${detected}" ]]; then
    _log "WARN: list_devices_node returned nothing — launch without usb_port"
    return 0
  fi
  ORBBEC_USB_PORT="${detected}"
  _log "Detected ORBBEC_USB_PORT=${ORBBEC_USB_PORT}"
}

_wait_for_topic() {
  local topic="$1" timeout_sec="${2:-120}" log_hint="${3:-camera.log}"
  local i=0
  _log "Waiting for topic ${topic} (up to ${timeout_sec}s) ..."
  while (( i < timeout_sec )); do
    _check_arm_driver_alive "${i}"
    if timeout 8 ros2 topic echo "${topic}" --once >/dev/null 2>&1; then
      _log "Ready: ${topic}"
      return 0
    fi
    if (( i > 0 && i % 15 == 0 )); then
      _log "  still waiting for ${topic} (${i}s) — see ${LOG_DIR}/${log_hint}"
    fi
    sleep 1
    i=$((i + 1))
  done
  _report_arm_failure "${topic}"
  _die "Timeout waiting for ${topic} (see ${LOG_DIR}/${log_hint})"
}

_check_arm_driver_alive() {
  local elapsed="${1:-0}"
  if [[ "${ENABLE_ARM}" != "true" || ! -f "${PIDFILE}" ]]; then
    return 0
  fi
  # ros2 launch spawns move_group before agx_arm_ctrl; allow bootstrapping.
  if (( elapsed < 25 )); then
    return 0
  fi
  local name pid
  while read -r name pid; do
    [[ "${name}" == "arm" && -n "${pid}" ]] || continue
    if ! kill -0 "${pid}" 2>/dev/null; then
      _report_arm_failure "/feedback/joint_states"
      _die "arm launch exited (pid ${pid}). See ${LOG_DIR}/arm.log"
    fi
  done < "${PIDFILE}"
  if ! pgrep -f 'agx_arm_ctrl_single' >/dev/null 2>&1; then
    _report_arm_failure "/feedback/joint_states"
    _die "agx_arm_ctrl not running after ${elapsed}s. Check CAN: candump can0; see ${LOG_DIR}/arm.log"
  fi
}

_report_arm_failure() {
  local topic="${1:-/feedback/joint_states}"
  echo "" >&2
  echo "[bringup] ERROR: ${topic} not publishing — arm driver likely failed to enable." >&2
  echo "[bringup] Common fixes:" >&2
  echo "  1. Windows Admin: usbipd attach --wsl --busid <CAN-BUSID>" >&2
  echo "  2. sudo ip link set can0 up type can bitrate 1000000" >&2
  echo "  3. Power-cycle arm, then retry" >&2
  echo "  4. tail -40 ${LOG_DIR}/arm.log" >&2
  if [[ -f "${LOG_DIR}/arm.log" ]]; then
    echo "[bringup] --- arm.log (last 15 lines) ---" >&2
    tail -15 "${LOG_DIR}/arm.log" >&2 || true
    if grep -q 'Agx_arm feedback is ready' "${LOG_DIR}/arm.log" 2>/dev/null \
        && grep -q 'process has died' "${LOG_DIR}/arm.log" 2>/dev/null; then
      echo "[bringup] NOTE: log has both success and old crash lines — ignore stale errors above success." >&2
    fi
  fi
  echo "" >&2
}

_wait_for_service() {
  local srv="$1" timeout_sec="${2:-120}"
  local i=0
  _log "Waiting for service /${srv} (up to ${timeout_sec}s) ..."
  while (( i < timeout_sec )); do
    if ros2 service list 2>/dev/null | grep -q "/${srv}"; then
      if timeout 8 ros2 service type "/${srv}" >/dev/null 2>&1; then
        _log "Ready: /${srv}"
        return 0
      fi
    fi
    if (( i > 0 && i % 15 == 0 )); then
      _log "  still waiting for /${srv} (${i}s) — see ${LOG_DIR}/perception.log"
    fi
    sleep 1
    i=$((i + 1))
  done
  _die "Timeout waiting for /${srv} (see ${LOG_DIR}/perception.log)"
}

STACK_RUNNING=false

_cleanup() {
  if [[ "${STACK_RUNNING}" != "true" ]]; then
    return 0
  fi
  _log "Shutting down stack ..."
  bash "${ROOT}/scripts/real_robot_shutdown.sh" || true
}

trap _cleanup INT TERM

main() {
  rm -f "${PIDFILE}"
  mkdir -p "${LOG_DIR}"

  _source_ros_env
  _preflight
  _setup_can

  if [[ "${CHECK_ONLY}" == "true" ]]; then
    _log "Preflight OK."
    exit 0
  fi

  # Stop leftovers from a previous session
  bash "${ROOT}/scripts/real_robot_shutdown.sh" --quiet --no-disable 2>/dev/null || true
  _log "Cleared stale processes (if any)"
  rm -f "${PIDFILE}"
  mkdir -p "${LOG_DIR}"

  local ros_setup="set +u; source /opt/ros/${ROS_DISTRO:-jazzy}/setup.bash; \
source ${AGX_ARM_WS}/install/setup.bash; source ${ROOT}/install/setup.bash"
  if [[ "${ENABLE_CAMERA}" == "true" ]]; then
    ros_setup="${ros_setup}; source ${ORBBEC_WS}/install/setup.bash"
  fi
  ros_setup="${ros_setup}; set -u 2>/dev/null || true"

  if [[ "${ENABLE_ARM}" == "true" ]]; then
    _start_bg arm "${ros_setup}; exec ros2 launch agx_arm_ctrl start_single_agx_arm_moveit.launch.py \
can_port:=${ARM_CAN_PORT} arm_type:=${ARM_TYPE} effector_type:=${ARM_EFFECTOR} \
auto_control_gate:=${ARM_AUTO_CONTROL_GATE} speed_percent:=${ARM_SPEED_PERCENT} \
enable_timeout:=${ARM_ENABLE_TIMEOUT:-15.0} use_rviz:=${ARM_USE_RVIZ} \
tcp_offset:='${ARM_TCP_OFFSET:-[0.0,0.0,0.05,0.0,0.0,0.0]}'" truncate
    _wait_for_action move_action 120
    _log "move_action ready — waiting for agx_arm_ctrl to enable (up to 90s) ..."
    _wait_for_topic /feedback/joint_states 90 arm.log
  fi

  if [[ "${ENABLE_RELAY}" == "true" ]]; then
    _start_bg relay "${ros_setup}; exec ros2 run topic_tools relay /feedback/joint_states /joint_states"
    sleep 1
    _wait_for_topic /joint_states 20 relay.log
  fi

  if [[ "${ENABLE_CAMERA}" == "true" ]]; then
    _check_orbbec_usb
    _detect_orbbec_usb_port
    ORBBEC_PUBLISH_TF="${ORBBEC_PUBLISH_TF:-false}"
    CAMERA_TF_CHILD="${CAMERA_TF_CHILD:-${ORBBEC_CAMERA_NAME}_color_optical_frame}"
    local camera_args="camera_name:=${ORBBEC_CAMERA_NAME} depth_registration:=${ORBBEC_DEPTH_REGISTRATION} publish_tf:=${ORBBEC_PUBLISH_TF}"
    if [[ -n "${ORBBEC_USB_PORT:-}" ]]; then
      camera_args="${camera_args} usb_port:=${ORBBEC_USB_PORT}"
    fi
    _start_bg camera "${ros_setup}; export RMW_FASTRTPS_USE_SHM=0; exec ros2 launch orbbec_camera ${ORBBEC_LAUNCH} ${camera_args}"
    sleep 5
    _wait_for_topic "/${ORBBEC_CAMERA_NAME}/color/image_raw" 120 camera.log
    IFS=',' read -r _r _p _y <<< "${CAMERA_MOUNT_RPY}"
    _cam_parent="${CAMERA_MOUNT_FRAME:-link6}"
    # Lens Z = CS7 +Z; optical axes match link6 — only mount translation (no Orbbec optical rpy)
    _start_bg camera_tf "${ros_setup}; exec ros2 run tf2_ros static_transform_publisher \
--x ${CAMERA_MOUNT_TX} --y ${CAMERA_MOUNT_TY} --z ${CAMERA_MOUNT_TZ} \
--roll ${_r:-0} --pitch ${_p:-0} --yaw ${_y:-0} \
--frame-id ${_cam_parent} --child-frame-id ${CAMERA_TF_CHILD}"
  fi

  if [[ "${ENABLE_PERCEPTION_CFG}" == "true" ]]; then
    _start_bg perception "bash ${ROOT}/scripts/run_fine_detector_node.sh"
    _wait_for_service trigger_fine_detection 180
  fi

  STACK_RUNNING=true

  if [[ "${NO_WAIT}" == "true" ]]; then
    trap - INT TERM
    _log "Stack ready (--no-wait). Logs: ${LOG_DIR}/"
    exit 0
  fi

  cat <<EOF

================================================================================
  Piper X real-robot stack is running.
  Logs: ${LOG_DIR}/
  Stop: bash scripts/real_robot_shutdown.sh

  Verify:
    ros2 topic hz /feedback/joint_states
    ros2 topic hz /${ORBBEC_CAMERA_NAME}/depth/image_raw
    ros2 action list | grep move_action

  Teleop (new terminal):
    bash scripts/real_robot_teleop.sh

  View camera:
    ros2 run rqt_image_view rqt_image_view /${ORBBEC_CAMERA_NAME}/color/image_raw
================================================================================

EOF

  if [[ "${ENABLE_TELEOP_CFG}" == "true" ]]; then
    trap - INT TERM
    _log "Starting keyboard teleop (Esc to quit) ..."
    # shellcheck disable=SC2086
    exec ros2 run picking_bringup link6_teleop_node --ros-args \
      -p linear_speed_m_s:=${TELEOP_LINEAR_SPEED} \
      -p angular_speed_deg_s:=${TELEOP_ANGULAR_SPEED}
  fi

  if [[ "${FOREGROUND_WAIT}" == "true" ]]; then
    _log "Press Ctrl+C to stop the whole stack."
    while true; do
      sleep 3600
    done
  fi
}

main "$@"
