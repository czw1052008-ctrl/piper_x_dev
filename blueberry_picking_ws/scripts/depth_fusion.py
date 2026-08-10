"""Priority-ordered depth estimation for the wrist camera.

Priority:
  1. RGB-D  (new wrist camera, 5 cm – 1 m, reliable when mask is filled)
  2. Depth Anything V2 metric-indoor (HuggingFace, ~100 MB, lazy-loaded)
  3. Caller falls back to existing mono (size-based) estimate

Usage:
    fuser = DepthFuser()
    depth_m, source = fuser.get_depth(rgb_img, raw_depth_m, mask)
    # source in {'rgbd', 'da2', 'unavailable'}
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np

_LOG = logging.getLogger(__name__)

# RGB-D depth validity range for the new wrist camera (metres).
_RGBD_MIN_M = 0.04
_RGBD_MAX_M = 1.10

# Minimum fraction of mask pixels that must carry valid RGB-D depth.
_RGBD_VALID_RATIO = 0.30

# Default DA2 model.  Indoor-Small is ~100 MB and runs on CPU ~50 ms/frame.
_DEFAULT_DA2_MODEL = (
    'depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf'
)


class DepthFuser:
    """Fuse RGB-D and Depth Anything V2 to get the best available depth."""

    def __init__(
        self,
        use_da2: bool = True,
        da2_model: str = _DEFAULT_DA2_MODEL,
    ) -> None:
        self._use_da2 = use_da2
        self._da2_model = da2_model
        self._da2_pipe = None
        self._da2_available: Optional[bool] = None   # None = not yet tried
        self._da2_load_attempted = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_depth(
        self,
        rgb_img: np.ndarray,                  # H×W×3 uint8 RGB
        raw_depth_m: Optional[np.ndarray],    # H×W float32 metres, 0=invalid
        mask: np.ndarray,                     # H×W bool
    ) -> Tuple[Optional[float], str]:
        """Return (depth_metres, source_tag).

        source_tag  ∈  {'rgbd', 'da2', 'unavailable'}
        Returns (None, 'unavailable') when all sources fail.
        """
        # 1 — try RGB-D
        rgbd_result = self._try_rgbd(raw_depth_m, mask)
        if rgbd_result is not None:
            return rgbd_result, 'rgbd'

        # 2 — try Depth Anything V2
        if self._use_da2:
            da2_result = self._try_da2(rgb_img, mask)
            if da2_result is not None:
                return da2_result, 'da2'

        return None, 'unavailable'

    @property
    def da2_available(self) -> bool:
        """True only after a successful DA2 load."""
        return bool(self._da2_available)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_rgbd(
        self,
        raw_depth_m: Optional[np.ndarray],
        mask: np.ndarray,
    ) -> Optional[float]:
        if raw_depth_m is None:
            return None
        in_range = (raw_depth_m > _RGBD_MIN_M) & (raw_depth_m < _RGBD_MAX_M)
        valid = raw_depth_m[mask & in_range]
        mask_area = int(mask.sum())
        if mask_area == 0:
            return None
        if len(valid) / mask_area < _RGBD_VALID_RATIO:
            return None
        return float(np.median(valid))

    def _try_da2(
        self,
        rgb_img: np.ndarray,
        mask: np.ndarray,
    ) -> Optional[float]:
        if not self._ensure_da2_loaded():
            return None
        try:
            from PIL import Image as PILImage
            pil = PILImage.fromarray(rgb_img)
            result = self._da2_pipe(pil)  # type: ignore[misc]
            depth_map = np.array(result['depth'], dtype=np.float32)
            if depth_map.shape != rgb_img.shape[:2]:
                # Resize to match (rare, model outputs same size by default)
                import cv2
                depth_map = cv2.resize(
                    depth_map,
                    (rgb_img.shape[1], rgb_img.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
            vals = depth_map[mask]
            vals = vals[np.isfinite(vals) & (vals > 0)]
            if len(vals) == 0:
                return None
            return float(np.median(vals))
        except Exception as exc:
            _LOG.warning(f'DA2 inference error: {exc}')
            return None

    def _ensure_da2_loaded(self) -> bool:
        if self._da2_available is True:
            return True
        if self._da2_load_attempted:
            return False
        self._da2_load_attempted = True
        try:
            from transformers import pipeline as hf_pipeline  # type: ignore
            _LOG.info(f'Loading Depth Anything V2 model: {self._da2_model}')
            self._da2_pipe = hf_pipeline(
                'depth-estimation',
                model=self._da2_model,
            )
            self._da2_available = True
            _LOG.info('Depth Anything V2 loaded successfully.')
            return True
        except Exception as exc:
            _LOG.warning(
                f'Could not load Depth Anything V2 ({exc}). '
                'Install: pip install transformers accelerate'
            )
            self._da2_available = False
            return False


# ---------------------------------------------------------------------------
# Standalone test  (python depth_fusion.py)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    print('DepthFuser unit test (no model download required)')

    H, W = 480, 640
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    mask = np.zeros((H, W), dtype=bool)
    mask[200:250, 300:340] = True

    # --- Test 1: RGB-D with good coverage ---
    depth_good = np.zeros((H, W), dtype=np.float32)
    depth_good[mask] = 0.15   # 15 cm
    fuser = DepthFuser(use_da2=False)
    val, src = fuser.get_depth(rgb, depth_good, mask)
    assert src == 'rgbd', f'Expected rgbd, got {src}'
    assert abs(val - 0.15) < 1e-4, f'Unexpected depth {val}'
    print(f'  Test 1 passed: source={src}, depth={val:.4f} m')

    # --- Test 2: RGB-D with sparse mask (too many holes) ---
    depth_sparse = np.zeros((H, W), dtype=np.float32)
    depth_sparse[mask] = 0.0   # all zeros = invalid
    val2, src2 = fuser.get_depth(rgb, depth_sparse, mask)
    assert src2 == 'unavailable', f'Expected unavailable, got {src2}'
    print(f'  Test 2 passed: sparse RGB-D → source={src2}')

    # --- Test 3: None raw depth ---
    val3, src3 = fuser.get_depth(rgb, None, mask)
    assert src3 == 'unavailable'
    print(f'  Test 3 passed: None depth → source={src3}')

    print('  All assertions passed.')
    sys.exit(0)
