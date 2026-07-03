#!/usr/bin/env bash
# Wrap a node binary for launch prefix=... (gdb backtrace on crash).
set -euo pipefail
exec gdb -batch \
  -ex 'set pagination off' \
  -ex run \
  -ex 'thread apply all bt full' \
  --args "$@"
