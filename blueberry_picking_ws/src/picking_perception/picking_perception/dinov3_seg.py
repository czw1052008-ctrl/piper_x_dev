"""DINOv3 (or CNN fallback) semantic segmentation: bg + berry + branch + rigid + ego."""

from __future__ import annotations

from typing import Optional, Tuple

import os
import site
import sys

_user_site = site.getusersitepackages()
if _user_site:
    sys.path = [_user_site] + [p for p in sys.path if p != _user_site]

import numpy as np

NUM_CLASSES = 5  # 0=bg, 1=berry, 2=branch, 3=rigid, 4=ego
CLASS_NAMES = ('background', 'berry', 'branch', 'rigid', 'ego')
DEFAULT_MODEL_ID = 'facebook/dinov3-vits16-pretrain-lvd1689m'
_WS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
LOCAL_BACKBONE_DIR = os.path.join(
    _WS_ROOT, 'models', 'dinov3-vits16-pretrain-lvd1689m')


def _try_torch():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    return torch, nn, F


class TinySegNet:
    """Small CNN used when transformers/DINOv3 weights are unavailable."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        torch, nn, F = _try_torch()
        self.torch = torch
        self.F = F

        class _Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.enc = nn.Sequential(
                    nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
                )
                self.head = nn.Conv2d(64, num_classes, 1)

            def forward(self, x):
                h = self.enc(x)
                logits = self.head(h)
                return F.interpolate(
                    logits, size=x.shape[-2:], mode='bilinear', align_corners=False)

        self.net = _Net()
        self.kind = 'tiny'


class DinoV3Seg:
    """Frozen DINOv3 + 1x1 conv head, bilinear upsample to input size."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        model_id: str = DEFAULT_MODEL_ID,
        freeze_backbone: bool = True,
    ) -> None:
        torch, nn, F = _try_torch()
        self.torch = torch
        self.F = F
        self.kind = 'dinov3'
        self.model_id = model_id
        from transformers import AutoModel

        self.backbone = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        hidden = int(getattr(self.backbone.config, 'hidden_size', 384))
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        class _Head(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = nn.Conv2d(hidden, num_classes, 1)

            def forward(self, feat, hw: Tuple[int, int]):
                logits = self.proj(feat)
                return F.interpolate(logits, size=hw, mode='bilinear', align_corners=False)

        self.head = _Head()

        class _BerryHm(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(hidden, 32, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 1, 1),
                )
                nn.init.constant_(self.net[-1].bias, -2.19)

            def forward(self, feat):
                return self.net(feat)

        self.berry_hm = _BerryHm()
        self.has_berry_hm = False
        self.patch = int(getattr(self.backbone.config, 'patch_size', 16))

    def _tokens_to_map(self, last_hidden, b: int, h: int, w: int):
        torch = self.torch
        tokens = last_hidden
        if tokens.dim() == 3:
            n = tokens.shape[1]
            # drop CLS / extra tokens so remaining is a square grid
            gh, gw = h // self.patch, w // self.patch
            expect = gh * gw
            if n >= expect:
                tokens = tokens[:, n - expect:, :]
            gh2 = int(round(tokens.shape[1] ** 0.5))
            tokens = tokens[:, : gh2 * gh2, :]
            feat = tokens.transpose(1, 2).reshape(b, tokens.shape[-1], gh2, gh2)
        else:
            feat = tokens
        return feat

    def forward(self, x):
        b, _, h, w = x.shape
        with self.torch.set_grad_enabled(self.backbone.training):
            out = self.backbone(pixel_values=x)
        hidden = out.last_hidden_state
        feat = self._tokens_to_map(hidden, b, h, w)
        logits = self.head(feat, (h, w))
        hm = self.F.interpolate(
            self.berry_hm(feat), size=(h, w), mode='bilinear', align_corners=False)
        return logits, hm


def resolve_backbone(model_id: str = '') -> str:
    """Prefer the in-repo DINOv3 snapshot; never fall back to DINOv2."""
    if model_id and os.path.isdir(model_id) and os.path.isfile(
            os.path.join(model_id, 'model.safetensors')):
        return model_id
    if os.path.isfile(os.path.join(LOCAL_BACKBONE_DIR, 'model.safetensors')):
        return LOCAL_BACKBONE_DIR
    return (model_id or DEFAULT_MODEL_ID)


def build_seg_model(
    *,
    model_id: str = '',
    num_classes: int = NUM_CLASSES,
    prefer_dino: bool = True,
):
    if prefer_dino:
        mid = resolve_backbone(model_id)
        try:
            m = DinoV3Seg(num_classes=num_classes, model_id=mid)
            print(f'[seg] backbone={mid}')
            return m
        except Exception as exc:
            print(f'[seg] DINOv3 failed ({type(exc).__name__}: {exc}); TinySeg fallback')
    return TinySegNet(num_classes=num_classes)


def as_nn(model) -> object:
    return model.net if isinstance(model, TinySegNet) else _DinoModule(model)


class _DinoModule:
    """nn.Module-like wrapper so train/eval/state_dict work for DinoV3Seg."""

    def __init__(self, inner: DinoV3Seg) -> None:
        self.inner = inner
        self.has_berry_hm = bool(getattr(inner, 'has_berry_hm', False))

    def to(self, *a, **k):
        self.inner.backbone.to(*a, **k)
        self.inner.head.to(*a, **k)
        self.inner.berry_hm.to(*a, **k)
        return self

    def train(self, mode: bool = True):
        self.inner.head.train(mode)
        self.inner.berry_hm.train(mode)
        return self

    def eval(self):
        self.inner.head.eval()
        self.inner.berry_hm.eval()
        self.inner.backbone.eval()
        return self

    def parameters(self):
        import itertools
        return itertools.chain(
            (p for p in self.inner.head.parameters() if p.requires_grad),
            (p for p in self.inner.berry_hm.parameters() if p.requires_grad),
        )

    def state_dict(self):
        return {
            'kind': 'dinov3',
            'head': self.inner.head.state_dict(),
            'berry_hm': self.inner.berry_hm.state_dict(),
            'has_berry_hm': bool(self.has_berry_hm),
        }

    def load_state_dict(self, sd, strict: bool = True):
        if 'head' in sd:
            self.inner.head.load_state_dict(sd['head'], strict=strict)
        if sd.get('has_berry_hm') and 'berry_hm' in sd:
            self.inner.berry_hm.load_state_dict(sd['berry_hm'], strict=strict)
            self.inner.has_berry_hm = True
            self.has_berry_hm = True
        return self

    def __call__(self, x):
        return self.inner.forward(x)


def wrap_for_train(model):
    if isinstance(model, TinySegNet):
        return model.net
    return _DinoModule(model)


def load_checkpoint(path: str, device=None):
    torch, _, _ = _try_torch()
    ckpt = torch.load(path, map_location=device or 'cpu')
    kind = ckpt.get('kind', 'tiny')
    num_classes = int(ckpt.get('num_classes', NUM_CLASSES))
    model_id = resolve_backbone(ckpt.get('model_id', ''))
    if kind == 'dinov3':
        try:
            inner = DinoV3Seg(num_classes=num_classes, model_id=model_id)
            inner.head.load_state_dict(ckpt['head'])
            if ckpt.get('has_berry_hm') and 'berry_hm' in ckpt:
                try:
                    inner.berry_hm.load_state_dict(ckpt['berry_hm'])
                    inner.has_berry_hm = True
                except Exception:
                    inner.has_berry_hm = False
            else:
                inner.has_berry_hm = False
            m = wrap_for_train(inner)
            m.has_berry_hm = bool(inner.has_berry_hm)
            m.eval()
            return m, kind
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f'[dinov3_seg] DINOv3 load failed, not using TinySeg: {exc}', flush=True)
            raise
    net = TinySegNet(num_classes=num_classes).net
    sd = ckpt.get('tiny_state_dict') or ckpt.get('state_dict') or ckpt
    net.load_state_dict(sd)
    net.eval()
    return net, 'tiny'


def unpack_seg(out):
    """(logits, heatmap_or_None) — TinySeg returns logits only."""
    if isinstance(out, (tuple, list)):
        logits = out[0]
        hm = out[1] if len(out) > 1 else None
        return logits, hm
    return out, None


def maps_from_logits(logits, hm, orig_hw, *, use_heatmap: bool):
    """Semantic argmax + instance ids for lift (berry peaks / CC, branch, rigid, ego).

    Berry ids come from heatmap peaks assigned over the full berry mask (no circle
    clip, no watershed-as-skin). Without a heatmap, berry stays connected components.
    """
    import cv2
    import torch
    from picking_perception.z_slice_geometry import instances_for_lift

    h, w = int(orig_hw[0]), int(orig_hw[1])
    pred = logits.argmax(1)[0].detach().cpu().numpy().astype(np.uint8)
    pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
    hm_np = None
    if use_heatmap and hm is not None:
        hm_np = torch.sigmoid(hm[0, 0]).detach().cpu().numpy().astype(np.float32)
        hm_np = cv2.resize(hm_np, (w, h), interpolation=cv2.INTER_LINEAR)
    inst = instances_for_lift(pred, hm_np if use_heatmap else None)
    return pred, inst, hm_np, []


def colorize_semantic(sem: np.ndarray) -> np.ndarray:
    """BGR overlay: berry magenta, branch green, rigid red, ego cyan."""
    h, w = sem.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[sem == 1] = (220, 60, 220)
    out[sem == 2] = (40, 180, 70)
    out[sem == 3] = (40, 40, 220)
    out[sem == 4] = (220, 200, 40)
    return out
