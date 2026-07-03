#!/usr/bin/env bash
# Create GitHub repo and push (requires: gh auth login).
set -eo pipefail

WS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_NAME="${1:-blueberry_picking_ws}"
VISIBILITY="${2:-public}"   # public | private

GH="${GH_BIN:-gh}"
if ! command -v "${GH}" >/dev/null 2>&1; then
  if [[ -x /tmp/gh_2.63.2_linux_amd64/bin/gh ]]; then
    GH=/tmp/gh_2.63.2_linux_amd64/bin/gh
  else
    echo "Install GitHub CLI: https://cli.github.com/  then run: gh auth login" >&2
    exit 1
  fi
fi

cd "${WS_ROOT}"

if ! "${GH}" auth status >/dev/null 2>&1; then
  echo "Not logged in. Run: ${GH} auth login" >&2
  exit 1
fi

if git remote get-url origin >/dev/null 2>&1; then
  echo "Remote origin already exists:"
  git remote -v
else
  "${GH}" repo create "${REPO_NAME}" \
    --"${VISIBILITY}" \
    --source=. \
    --remote=origin \
    --description "Piper X blueberry picking — ROS 2 + MoveIt + Gazebo Harmonic" \
    --push
  echo "Done: $(${GH} repo view --json url -q .url)"
  exit 0
fi

git push -u origin main
echo "Pushed to $(git remote get-url origin)"
