"""Episode-level data collector for ALIGNING behaviour cloning.

Hooks into reach_fsm_node:
  collector.start_episode(plant_xyz)         ← when LOCKING succeeds
  collector.log_step(g_img, w_img, obs, act) ← each _tick_aligning step
  collector.mark_success()                   ← _enter_fine_after_coarse()
  collector.end_episode()                    ← FSM reset to IDLE

Saved layout:
  <save_dir>/episode_NNNNN/
    metadata.json
    steps.json
    step_000_global.jpg
    step_000_wrist.jpg
    ...
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

_LOG = logging.getLogger(__name__)

# Observation keys written to steps.json (subset of build_observation() output).
OBS_KEYS = (
    'joint1_deg', 'joint2_deg', 'joint3_deg', 'joint5_deg',
    'yaw_error_deg', 'pitch_error_rad', 'ee_target_angle_deg',
    'fine_visible', 'fine_du', 'fine_dv',
    'fixed_dx_px', 'fixed_dy_px',
    'horiz_dist', 'plant_yaw_deg',
)


class AlignDataCollector:
    """Collect and persist ALIGNING episodes for BC training."""

    def __init__(
        self,
        save_dir: str,
        teacher: str = 'heuristic',   # 'heuristic' | 'vlm' | 'file'
        jpeg_quality: int = 90,
        enabled: bool = True,
    ) -> None:
        self._save_dir = save_dir
        self._teacher = teacher
        self._jpeg_quality = jpeg_quality
        self._enabled = enabled

        self._ep: Optional[Dict[str, Any]] = None   # active episode
        self._ep_id: int = self._scan_existing_id()

        if enabled:
            os.makedirs(save_dir, exist_ok=True)
            _LOG.info(f'AlignDataCollector: save_dir={save_dir}  next_id={self._ep_id}')

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def start_episode(self, plant_xyz=None) -> None:
        if not self._enabled:
            return
        if self._ep is not None:
            _LOG.warning('AlignDataCollector: start_episode called without end_episode; discarding previous')
            self._ep = None

        ep_id_str = f'{self._ep_id:05d}'
        ep_dir = os.path.join(self._save_dir, f'episode_{ep_id_str}')
        os.makedirs(ep_dir, exist_ok=True)

        self._ep = {
            'ep_id': ep_id_str,
            'ep_dir': ep_dir,
            'plant_xyz': list(plant_xyz) if plant_xyz is not None else None,
            'steps': [],
            'success': False,
            'teacher': self._teacher,
            'start_time': time.time(),
        }
        _LOG.info(f'AlignDataCollector: episode {ep_id_str} started')

    def log_step(
        self,
        global_img: np.ndarray,
        wrist_img: np.ndarray,
        obs: Dict[str, Any],
        action: Dict[str, Any],
    ) -> None:
        if not self._enabled or self._ep is None:
            return

        idx = len(self._ep['steps'])
        ep_dir = self._ep['ep_dir']

        # Save images.
        g_fname = f'step_{idx:03d}_global.jpg'
        w_fname = f'step_{idx:03d}_wrist.jpg'
        self._save_jpg(global_img, os.path.join(ep_dir, g_fname))
        self._save_jpg(wrist_img,  os.path.join(ep_dir, w_fname))

        # Filter obs to known keys, coerce to float.
        obs_filtered = {
            k: float(obs[k]) for k in OBS_KEYS if k in obs
        }

        # Strip large/non-serialisable fields from action.
        action_clean = {
            k: v for k, v in action.items()
            if k in ('action', 'joints_deg', 'delta_deg', 'reason', 'source', 'phase')
        }

        self._ep['steps'].append({
            'step_idx': idx,
            'obs': obs_filtered,
            'action': action_clean,
        })

    def mark_success(self) -> None:
        if self._ep is not None:
            self._ep['success'] = True

    def end_episode(self) -> None:
        if not self._enabled or self._ep is None:
            return

        ep = self._ep
        ep_dir = ep['ep_dir']

        # Write steps.json (images referenced by filename only).
        with open(os.path.join(ep_dir, 'steps.json'), 'w') as f:
            json.dump(ep['steps'], f, indent=2)

        # Write metadata.json.
        meta = {
            'episode_id': ep['ep_id'],
            'success': ep['success'],
            'n_steps': len(ep['steps']),
            'plant_xyz': ep['plant_xyz'],
            'teacher': ep['teacher'],
            'duration_s': round(time.time() - ep['start_time'], 2),
        }
        with open(os.path.join(ep_dir, 'metadata.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        status = 'SUCCESS' if ep['success'] else 'FAIL'
        _LOG.info(
            f'AlignDataCollector: episode {ep["ep_id"]} {status} '
            f'({len(ep["steps"])} steps) → {ep_dir}'
        )
        self._ep_id += 1
        self._ep = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_jpg(self, img: np.ndarray, path: str) -> None:
        bgr = img[:, :, ::-1] if img.ndim == 3 and img.shape[2] == 3 else img
        cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality])

    def _scan_existing_id(self) -> int:
        """Find the next episode ID by scanning existing directories."""
        if not os.path.isdir(self._save_dir):
            return 0
        ids = []
        for name in os.listdir(self._save_dir):
            if name.startswith('episode_'):
                try:
                    ids.append(int(name.split('_')[1]))
                except (IndexError, ValueError):
                    pass
        return (max(ids) + 1) if ids else 0

    @property
    def total_episodes(self) -> int:
        return self._ep_id

    @property
    def active(self) -> bool:
        return self._ep is not None
