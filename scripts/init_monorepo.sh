#!/usr/bin/env bash
# Flatten nested git repos into piper_x_dev monorepo (run once from repo root).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[init] Removing nested .git directories (source files kept) ..."
find "$ROOT" -mindepth 2 -name '.git' -type d -print | while read -r g; do
  echo "  rm -rf $g"
  rm -rf "$g"
done

if [[ ! -d "$ROOT/.git" ]]; then
  echo "[init] git init ..."
  git init -b main
fi

echo "[init] Done. Next: git add . && git commit && git remote add origin <url> && git push -u origin main"
