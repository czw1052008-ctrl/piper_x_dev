"""Behaviour-cloning policy for ALIGNING stage.

Architecture:
  CLIP ViT-B/16 (frozen) → image features
  Small MLP fusion        → joint angle targets (4 joints)

Inference wrapper exposes the same interface as align_judge.decide_action().

Usage:
  policy = BCAlignPolicy.load('models/bc_align_policy.pt')
  action = policy.decide(global_img, wrist_img, obs)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

_LOG = logging.getLogger(__name__)

# Observation keys used as input (must match align_data_collector.OBS_KEYS).
OBS_KEYS = (
    'joint1_deg', 'joint2_deg', 'joint3_deg', 'joint5_deg',
    'yaw_error_deg', 'pitch_error_rad', 'ee_target_angle_deg',
    'fine_visible', 'fine_du', 'fine_dv',
    'fixed_dx_px', 'fixed_dy_px',
    'horiz_dist', 'plant_yaw_deg',
)
N_OBS = len(OBS_KEYS)   # 14

# Default normalisation constants (overridden by bc_align_norm.json at load time).
_DEFAULT_NORM = {
    'joint1_deg': 90.0, 'joint2_deg': 90.0,
    'joint3_deg': 90.0, 'joint5_deg': 90.0,
    'yaw_error_deg': 90.0, 'pitch_error_rad': 1.57,
    'ee_target_angle_deg': 90.0,
    'fine_visible': 1.0, 'fine_du': 320.0, 'fine_dv': 320.0,
    'fixed_dx_px': 320.0, 'fixed_dy_px': 320.0,
    'horiz_dist': 1.0, 'plant_yaw_deg': 90.0,
}

# Output joints in order.
OUTPUT_JOINTS = ('joint1', 'joint2', 'joint3', 'joint5')

# Output scale (tanh * scale → bounded degrees).
_OUT_SCALE = torch.tensor([90.0, 60.0, 60.0, 90.0])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BCAlignPolicy(nn.Module):
    """CLIP encoder + MLP head for joint angle regression."""

    def __init__(
        self,
        obs_dim: int = N_OBS,
        img_feat_dim: int = 512,   # CLIP ViT-B/16 pooler output
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()

        # Frozen CLIP vision encoder (loaded separately, not stored in state_dict).
        self._clip: Optional[nn.Module] = None

        # Image projection (one shared linear for both cameras).
        self.img_proj = nn.Linear(img_feat_dim, hidden_dim)

        # Observation encoder.
        self.obs_enc = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Fusion MLP.
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
        )

        # Joint prediction head (tanh * scale → bounded degrees).
        self.joint_head = nn.Linear(128, len(OUTPUT_JOINTS))

        # Done head (logit → sigmoid; >0.8 = coarse_ok).
        self.done_head = nn.Linear(128, 1)

    # ------------------------------------------------------------------

    def forward(
        self,
        global_feat: torch.Tensor,   # (B, img_feat_dim) — pre-computed CLIP features
        wrist_feat: torch.Tensor,    # (B, img_feat_dim)
        obs_vec: torch.Tensor,       # (B, obs_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        g = self.img_proj(global_feat)       # (B, 256)
        w = self.img_proj(wrist_feat)        # (B, 256)
        o = self.obs_enc(obs_vec)            # (B, 256)

        fused = self.fusion(torch.cat([g, w, o], dim=-1))   # (B, 128)

        scale = _OUT_SCALE.to(fused.device)
        joints = torch.tanh(self.joint_head(fused)) * scale  # (B, 4)
        done   = self.done_head(fused)                        # (B, 1)
        return joints, done

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save(self, path: str, norm: Optional[Dict] = None) -> None:
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save(self.state_dict(), path)
        norm_path = path.replace('.pt', '_norm.json')
        with open(norm_path, 'w') as f:
            json.dump(norm or _DEFAULT_NORM, f, indent=2)
        _LOG.info(f'BCAlignPolicy saved → {path}')

    @classmethod
    def load(
        cls,
        path: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ) -> 'BCAlignPolicy':
        model = cls()
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        model = model.to(device)
        norm_path = path.replace('.pt', '_norm.json')
        if os.path.exists(norm_path):
            with open(norm_path) as f:
                model._norm = json.load(f)
        else:
            model._norm = _DEFAULT_NORM
        _LOG.info(f'BCAlignPolicy loaded from {path} on {device}')
        return model


# ---------------------------------------------------------------------------
# Inference wrapper (CLIP encoding + decide() interface)
# ---------------------------------------------------------------------------

class BCAlignPolicyInference:
    """Wraps BCAlignPolicy with CLIP preprocessing for use in reach_fsm_node."""

    def __init__(self, model_path: str) -> None:
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._model = BCAlignPolicy.load(model_path, device=self._device)
        self._norm: Dict[str, float] = getattr(self._model, '_norm', _DEFAULT_NORM)
        self._clip = None
        self._processor = None
        self._load_clip()

    def _load_clip(self) -> None:
        try:
            from transformers import CLIPVisionModel, CLIPImageProcessor  # type: ignore
            self._clip = CLIPVisionModel.from_pretrained(
                'openai/clip-vit-base-patch16'
            ).to(self._device).eval()
            for p in self._clip.parameters():
                p.requires_grad = False
            self._processor = CLIPImageProcessor.from_pretrained(
                'openai/clip-vit-base-patch16'
            )
            _LOG.info('CLIP encoder loaded for BC inference.')
        except Exception as exc:
            _LOG.warning(f'CLIP load failed ({exc}); using zero image features')

    def _encode_image(self, img: np.ndarray) -> torch.Tensor:
        """Returns (1, 512) CLIP pooler feature."""
        if self._clip is None or self._processor is None:
            return torch.zeros(1, 512, device=self._device)
        from PIL import Image as PILImage
        pil = PILImage.fromarray(img)
        inputs = self._processor(images=pil, return_tensors='pt')
        pixel_values = inputs['pixel_values'].to(self._device)
        with torch.no_grad():
            out = self._clip(pixel_values=pixel_values)
        return out.pooler_output   # (1, 512)

    def _obs_to_tensor(self, obs: Dict[str, Any]) -> torch.Tensor:
        vec = [float(obs.get(k, 0.0)) / self._norm.get(k, 1.0) for k in OBS_KEYS]
        return torch.tensor(vec, dtype=torch.float32, device=self._device).unsqueeze(0)

    def decide(
        self,
        global_img: np.ndarray,
        wrist_img: np.ndarray,
        obs: Dict[str, Any],
        phase: str = 'command',
    ) -> Dict[str, Any]:
        """Return action dict compatible with align_judge.decide_action()."""
        from align_judge import align_entry_ready

        # Gate: if heuristic says ready, skip model call.
        ready, _ = align_entry_ready(obs)
        if ready:
            return {'action': 'coarse_ok', 'source': 'bc_gate', 'phase': phase}

        g_feat = self._encode_image(global_img)
        w_feat = self._encode_image(wrist_img)
        obs_vec = self._obs_to_tensor(obs)

        with torch.no_grad():
            joints_t, done_t = self._model(g_feat, w_feat, obs_vec)

        joints = joints_t[0].cpu().numpy()
        done_prob = float(torch.sigmoid(done_t[0, 0]).item())

        if done_prob > 0.80:
            return {'action': 'coarse_ok', 'source': 'bc_done_head', 'phase': phase,
                    'done_prob': done_prob}

        joints_deg = {
            j: float(joints[i])
            for i, j in enumerate(OUTPUT_JOINTS)
        }
        return {
            'action': 'set_joints',
            'joints_deg': joints_deg,
            'source': 'bc_policy',
            'phase': phase,
            'done_prob': done_prob,
        }


# ---------------------------------------------------------------------------
# Standalone smoke test  (python bc_align_policy.py)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    print('BCAlignPolicy smoke test (untrained, random weights)')

    model = BCAlignPolicy()
    g = torch.randn(2, 512)
    w = torch.randn(2, 512)
    o = torch.randn(2, N_OBS)
    joints, done = model(g, w, o)
    assert joints.shape == (2, 4), f'Unexpected joint shape: {joints.shape}'
    assert done.shape   == (2, 1), f'Unexpected done shape: {done.shape}'
    assert (joints.abs() <= 90.1).all(), 'Joints out of tanh*90 range'
    print(f'  forward pass OK  joints={joints[0].tolist()}')
    print('  All assertions passed.')
    sys.exit(0)
