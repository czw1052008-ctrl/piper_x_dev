"""PyTorch Dataset for ALIGNING behaviour-cloning episodes.

Loads episodes saved by AlignDataCollector and flattens them to
(global_img, wrist_img, obs_vec, label_joints, is_done) samples.

Only steps from successful episodes are included.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# Must match bc_align_policy.OBS_KEYS exactly.
OBS_KEYS = (
    'joint1_deg', 'joint2_deg', 'joint3_deg', 'joint5_deg',
    'yaw_error_deg', 'pitch_error_rad', 'ee_target_angle_deg',
    'fine_visible', 'fine_du', 'fine_dv',
    'fixed_dx_px', 'fixed_dy_px',
    'horiz_dist', 'plant_yaw_deg',
)

OUTPUT_JOINTS = ('joint1', 'joint2', 'joint3', 'joint5')


class AlignDataset(Dataset):
    """Flat dataset of (obs, action) steps from successful alignment episodes."""

    def __init__(
        self,
        data_dir: str,
        success_only: bool = True,
        norm: Optional[Dict[str, float]] = None,
        clip_processor=None,         # CLIPImageProcessor instance (optional)
        augment: bool = True,
    ) -> None:
        self._data_dir = data_dir
        self._norm = norm or self._default_norm()
        self._clip_proc = clip_processor
        self._augment = augment

        self._samples: List[Dict] = []
        self._load(data_dir, success_only)
        print(f'AlignDataset: {len(self._samples)} steps from {data_dir}')

    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int):
        s = self._samples[idx]
        ep_dir = s['ep_dir']

        global_img = self._load_image(os.path.join(ep_dir, s['global_fname']))
        wrist_img  = self._load_image(os.path.join(ep_dir, s['wrist_fname']))

        if self._clip_proc is not None:
            global_t = self._clip_proc(images=global_img, return_tensors='pt')['pixel_values'][0]
            wrist_t  = self._clip_proc(images=wrist_img,  return_tensors='pt')['pixel_values'][0]
        else:
            # Fallback: resize to 224×224, normalise to [0,1].
            global_t = self._to_tensor(global_img)
            wrist_t  = self._to_tensor(wrist_img)

        obs_vec      = torch.tensor(s['obs_vec'],     dtype=torch.float32)
        label_joints = torch.tensor(s['label_joints'], dtype=torch.float32)
        is_done      = torch.tensor([s['is_done']],    dtype=torch.float32)

        return global_t, wrist_t, obs_vec, label_joints, is_done

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def _load(self, data_dir: str, success_only: bool) -> None:
        for name in sorted(os.listdir(data_dir)):
            if not name.startswith('episode_'):
                continue
            ep_dir = os.path.join(data_dir, name)
            meta_path = os.path.join(ep_dir, 'metadata.json')
            steps_path = os.path.join(ep_dir, 'steps.json')
            if not (os.path.exists(meta_path) and os.path.exists(steps_path)):
                continue

            with open(meta_path) as f:
                meta = json.load(f)
            if success_only and not meta.get('success', False):
                continue

            with open(steps_path) as f:
                steps = json.load(f)

            n = len(steps)
            for i, step in enumerate(steps):
                joints_deg = step['action'].get('joints_deg', {})
                if not joints_deg:
                    continue  # skip steps without explicit joint target

                label = [float(joints_deg.get(j, 0.0)) for j in OUTPUT_JOINTS]
                obs_vec = [
                    float(step['obs'].get(k, 0.0)) / self._norm.get(k, 1.0)
                    for k in OBS_KEYS
                ]

                self._samples.append({
                    'ep_dir': ep_dir,
                    'global_fname': f'step_{i:03d}_global.jpg',
                    'wrist_fname':  f'step_{i:03d}_wrist.jpg',
                    'obs_vec':      obs_vec,
                    'label_joints': label,
                    'is_done': float(i == n - 1),  # last step of successful episode
                })

    def _load_image(self, path: str) -> Image.Image:
        return Image.open(path).convert('RGB')

    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        img = img.resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.tensor(arr).permute(2, 0, 1)   # (3, 224, 224)

    @staticmethod
    def _default_norm() -> Dict[str, float]:
        return {
            'joint1_deg': 90.0, 'joint2_deg': 90.0,
            'joint3_deg': 90.0, 'joint5_deg': 90.0,
            'yaw_error_deg': 90.0, 'pitch_error_rad': 1.57,
            'ee_target_angle_deg': 90.0,
            'fine_visible': 1.0, 'fine_du': 320.0, 'fine_dv': 320.0,
            'fixed_dx_px': 320.0, 'fixed_dy_px': 320.0,
            'horiz_dist': 1.0, 'plant_yaw_deg': 90.0,
        }

    @staticmethod
    def compute_norm(data_dir: str) -> Dict[str, float]:
        """Compute per-key std-dev from data for better normalisation."""
        vals: Dict[str, List[float]] = {k: [] for k in OBS_KEYS}
        for name in sorted(os.listdir(data_dir)):
            ep_dir = os.path.join(data_dir, name)
            steps_path = os.path.join(ep_dir, 'steps.json')
            if not os.path.exists(steps_path):
                continue
            with open(steps_path) as f:
                steps = json.load(f)
            for step in steps:
                for k in OBS_KEYS:
                    if k in step['obs']:
                        vals[k].append(float(step['obs'][k]))
        norm = {}
        for k, v in vals.items():
            std = float(np.std(v)) if v else 1.0
            norm[k] = max(std, 1e-3)
        return norm
