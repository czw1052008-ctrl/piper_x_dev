"""Train the BC alignment policy.

Usage:
    pip install torch torchvision transformers
    python training/train_bc_align.py \
        --data-dir data/align_episodes \
        --output-dir models/ \
        --epochs 100

Expected with 200+ successful episodes:
    val joint RMSE < 2° after 100 epochs
    Inference: 5-15 ms on RTX 3090
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

# Allow importing from scripts/ directory.
_SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'scripts')
sys.path.insert(0, os.path.abspath(_SCRIPTS_DIR))

from align_dataset import AlignDataset          # noqa: E402
from bc_align_policy import BCAlignPolicy, OBS_KEYS, OUTPUT_JOINTS  # noqa: E402


def build_clip_processor():
    try:
        from transformers import CLIPImageProcessor  # type: ignore
        return CLIPImageProcessor.from_pretrained('openai/clip-vit-base-patch16')
    except Exception as exc:
        print(f'[warn] CLIP processor unavailable ({exc}); using simple resize')
        return None


def build_clip_encoder(device: str):
    try:
        from transformers import CLIPVisionModel  # type: ignore
        clip = CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch16')
        clip = clip.to(device).eval()
        for p in clip.parameters():
            p.requires_grad = False
        return clip
    except Exception as exc:
        print(f'[warn] CLIP encoder unavailable ({exc}); using zero image features')
        return None


def encode_images(clip, batch_imgs: torch.Tensor, device: str) -> torch.Tensor:
    """(B, 3, 224, 224) → (B, 512) CLIP pooler features."""
    if clip is None:
        return torch.zeros(batch_imgs.size(0), 512, device=device)
    with torch.no_grad():
        out = clip(pixel_values=batch_imgs.to(device))
    return out.pooler_output


def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Training on {device}')

    # ── Dataset ────────────────────────────────────────────────────────
    clip_proc = build_clip_processor()

    # Optionally compute data-driven normalisation.
    norm_path = os.path.join(args.output_dir, 'bc_align_norm.json')
    if os.path.exists(norm_path):
        with open(norm_path) as f:
            norm = json.load(f)
        print(f'Loaded existing norm from {norm_path}')
    else:
        print('Computing normalisation statistics …')
        norm = AlignDataset.compute_norm(args.data_dir)
        os.makedirs(args.output_dir, exist_ok=True)
        with open(norm_path, 'w') as f:
            json.dump(norm, f, indent=2)
        print(f'Norm saved to {norm_path}')

    dataset = AlignDataset(
        args.data_dir,
        success_only=True,
        norm=norm,
        clip_processor=clip_proc,
        augment=True,
    )

    if len(dataset) < 10:
        print(f'ERROR: only {len(dataset)} samples — need at least 10. Collect more episodes.')
        sys.exit(1)

    val_size   = max(1, int(0.15 * len(dataset)))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size],
                                    generator=torch.Generator().manual_seed(42))
    print(f'Train: {train_size}  Val: {val_size}')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=True)

    # ── CLIP encoder (frozen) ──────────────────────────────────────────
    clip = build_clip_encoder(device)

    # ── Model ─────────────────────────────────────────────────────────
    model = BCAlignPolicy().to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f'Trainable params: {sum(p.numel() for p in trainable):,}')

    # ── Optimiser & scheduler ─────────────────────────────────────────
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val_rmse = float('inf')
    best_path = os.path.join(args.output_dir, 'bc_align_policy_best.pt')
    final_path = os.path.join(args.output_dir, 'bc_align_policy.pt')

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list = []

        for g_img, w_img, obs_vec, label_joints, is_done in train_loader:
            obs_vec     = obs_vec.to(device)
            label_joints = label_joints.to(device)
            is_done     = is_done.to(device)

            g_feat = encode_images(clip, g_img, device)
            w_feat = encode_images(clip, w_img, device)

            pred_joints, pred_done = model(g_feat, w_feat, obs_vec)

            loss_joints = F.mse_loss(pred_joints, label_joints)
            loss_done   = F.binary_cross_entropy_with_logits(pred_done, is_done)
            loss = loss_joints + 0.1 * loss_done

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
            opt.step()

            train_losses.append(loss_joints.item())

        sched.step()

        # ── Validation ────────────────────────────────────────────────
        model.eval()
        val_sq_errs: list = []
        with torch.no_grad():
            for g_img, w_img, obs_vec, label_joints, _ in val_loader:
                obs_vec      = obs_vec.to(device)
                label_joints = label_joints.to(device)
                g_feat = encode_images(clip, g_img, device)
                w_feat = encode_images(clip, w_img, device)
                pred_joints, _ = model(g_feat, w_feat, obs_vec)
                val_sq_errs.extend(
                    ((pred_joints - label_joints) ** 2).mean(dim=1).cpu().tolist()
                )

        val_rmse = math.sqrt(np.mean(val_sq_errs))
        train_rmse = math.sqrt(np.mean(train_losses))

        print(f'Epoch {epoch:3d}/{args.epochs}  '
              f'train_rmse={train_rmse:.3f}°  val_rmse={val_rmse:.3f}°  '
              f'lr={sched.get_last_lr()[0]:.2e}')

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            model.save(best_path, norm=norm)
            print(f'  → best model saved (val_rmse={val_rmse:.3f}°)')

    # Save final.
    model.save(final_path, norm=norm)
    print(f'\nTraining complete. Best val RMSE: {best_val_rmse:.3f}°')
    print(f'Best model: {best_path}')
    print(f'Final model: {final_path}')
    print(f'\nDeploy with: --align-judge-mode bc --bc-model-path {final_path}')


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Train BC alignment policy')
    ap.add_argument('--data-dir',    required=True,
                    help='Directory with episode_NNNNN/ subdirs')
    ap.add_argument('--output-dir',  default='models/',
                    help='Where to save trained model and norm file')
    ap.add_argument('--epochs',      type=int,   default=100)
    ap.add_argument('--batch-size',  type=int,   default=64)
    ap.add_argument('--lr',          type=float, default=1e-4)
    args = ap.parse_args()
    train(args)
