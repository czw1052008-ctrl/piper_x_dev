#!/usr/bin/env python3
"""Archive pick-cycle failure sessions for offline analysis."""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Optional

SCRIPTS_DIR = Path(__file__).resolve().parent
DEFAULT_FAIL_ROOT = SCRIPTS_DIR.parent / 'log' / 'real_robot' / 'pick_failures'


class PickSessionRecorder:
    def __init__(self, root: Path = DEFAULT_FAIL_ROOT) -> None:
        self._root = root
        self._root.mkdir(parents=True, exist_ok=True)
        self._active: Optional[Path] = None

    def start_fruit(self, *, mission_id: str, fruit_index: int) -> Path:
        stamp = time.strftime('%Y%m%d_%H%M%S')
        d = self._root / f'{mission_id}_fruit{fruit_index:02d}_{stamp}'
        d.mkdir(parents=True, exist_ok=True)
        self._active = d
        return d

    def record_failure(
        self,
        *,
        stage: str,
        reason: str,
        qa_session_dir: Optional[Path] = None,
        extra: Optional[dict] = None,
    ) -> Path:
        if self._active is None:
            stamp = time.strftime('%Y%m%d_%H%M%S')
            self._active = self._root / f'failure_{stamp}'
            self._active.mkdir(parents=True, exist_ok=True)

        payload = {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'stage': stage,
            'reason': reason,
            'extra': extra or {},
        }
        with open(self._active / 'failure.json', 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=True)

        if qa_session_dir and qa_session_dir.is_dir():
            dst = self._active / 'reach_qa'
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(qa_session_dir, dst)

        return self._active

    def record_success(self, *, stage: str, extra: Optional[dict] = None) -> None:
        if self._active is None:
            return
        payload = {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'stage': stage,
            'outcome': 'success',
            'extra': extra or {},
        }
        with open(self._active / 'success.json', 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=True)
