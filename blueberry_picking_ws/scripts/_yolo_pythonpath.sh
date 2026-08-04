#!/usr/bin/env bash
# Shared PYTHONPATH so ultralytics works with system NumPy (agx_arm-safe).
# shellcheck shell=bash

_PY_VER="$(/usr/bin/python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_OVERRIDES="${_ROOT}/third_party/py_overrides"
_SYS_LOCAL="/usr/local/lib/python${_PY_VER}/dist-packages"
_SYS_DIST="/usr/lib/python${_PY_VER}/dist-packages"
_SYS_DIST3="/usr/lib/python3/dist-packages"
_ULTRA_SITE="${HOME}/.local/lib/python${_PY_VER}/site-packages"

export PYTHONNOUSERSITE=1
export MPLBACKEND="${MPLBACKEND:-Agg}"

_PREFIX=""
[[ -d "${_OVERRIDES}" ]] && _PREFIX="${_OVERRIDES}"
for _p in "${_SYS_LOCAL}" "${_SYS_DIST}" "${_SYS_DIST3}"; do
  [[ -d "${_p}" ]] && _PREFIX="${_PREFIX:+${_PREFIX}:}${_p}"
done
if [[ -d "${_ULTRA_SITE}/ultralytics" ]]; then
  export PYTHONPATH="${_PREFIX}:${_ULTRA_SITE}${PYTHONPATH:+:${PYTHONPATH}}"
else
  export PYTHONPATH="${_PREFIX}${PYTHONPATH:+:${PYTHONPATH}}"
fi
