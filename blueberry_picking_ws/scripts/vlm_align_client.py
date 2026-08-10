"""VLM-backed ALIGNING controller — local inference via Ollama.

Uses a locally-hosted vision-language model (default: qwen2.5-vl:7b) through
Ollama running on the same machine.  No internet or cloud API key required.

Hardware requirement: RTX 3090 / 4090 (24 GB VRAM) → ~1-2 s per call.

Setup (one-time):
    # 1. Install Ollama
    curl -fsSL https://ollama.com/install.sh | sh

    # 2. Pull the model (~16 GB download)
    ollama pull qwen2.5-vl:7b

    # 3. Install Python client
    pip install ollama

Activation:
    Launch reach_fsm_node with --align-judge-mode vlm

Interface (unchanged from previous version):
    client = VLMAlignClient()
    action = client.decide(global_img, wrist_img, obs, phase='command')
    # Returns same schema as align_judge.decide_action()
"""

from __future__ import annotations

import base64
import json
import logging
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np

_LOG = logging.getLogger(__name__)

# Joints the arm can move (joint4 and joint6 locked at 0).
_MOVABLE_JOINTS = ('joint1', 'joint2', 'joint3', 'joint5')

_DEFAULT_MODEL = 'qwen2.5-vl:7b'
_DEFAULT_HOST  = 'http://localhost:11434'

_SYSTEM_PROMPT = """\
You are the ALIGNING controller for a 6-DOF blueberry-picking robot arm.

Two camera images are provided:
- IMAGE 1 (GLOBAL): fixed overhead camera. The target berry is circled in RED.
- IMAGE 2 (WRIST): what the wrist camera currently sees.

Your goal: issue a joint command so the wrist camera centres on the target berry.

AVAILABLE ACTIONS — return exactly one as a JSON object on the last line:

1. Absolute joint targets (for corrections > 5 deg):
   {"action": "set_joints", "joints_deg": {"joint1": X, "joint2": X, "joint3": X, "joint5": X}}

2. Relative delta (for small corrections ≤ 5 deg):
   {"action": "delta_joints", "delta_deg": {"joint1": X, "joint2": X, "joint3": X, "joint5": X}}

3. Alignment done (only when wrist image clearly shows the target berry):
   {"action": "coarse_ok"}

JOINT LIMITS (degrees):
  joint1 (base yaw):    -90 to +90
  joint2 (shoulder):    -40 to +60
  joint3 (elbow):       -60 to +60
  joint5 (wrist pitch): -90 to +90
  joint4 and joint6 are LOCKED — do not include them.

SAFETY RULES:
- No single-step delta larger than 15 degrees on any joint.
- Lean joint2 to -10~-15 deg and joint3 to +10~+20 deg to tilt wrist toward plant.
- Return coarse_ok only when the berry is already centred in the wrist image.

Think briefly, then put the JSON action on its own line at the very end.\
"""


class VLMAlignClient:
    """Local VLM client for ALIGNING decisions, backed by Ollama."""

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        host: str = _DEFAULT_HOST,
        max_retries: int = 1,
        timeout_s: float = 15.0,
        max_img_width: int = 1024,
        jpeg_quality: int = 80,
    ) -> None:
        self._model = model
        self._host = host
        self._max_retries = max_retries
        self._timeout_s = timeout_s
        self._max_img_width = max_img_width
        self._jpeg_quality = jpeg_quality
        self._client = None   # lazy-init

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(
        self,
        global_img: np.ndarray,   # fixed camera RGB (target circled in red)
        wrist_img: np.ndarray,    # wrist camera RGB
        obs: Dict[str, Any],      # align_judge.build_observation() output
        phase: str = 'command',
    ) -> Dict[str, Any]:
        """Return an action dict compatible with align_judge.decide_action()."""
        try:
            return self._call_local(global_img, wrist_img, obs, phase)
        except Exception as exc:
            _LOG.warning(f'VLMAlignClient error ({exc}); falling back to heuristic')
            return self._heuristic_fallback(obs, phase)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _call_local(
        self,
        global_img: np.ndarray,
        wrist_img: np.ndarray,
        obs: Dict[str, Any],
        phase: str,
    ) -> Dict[str, Any]:
        client = self._get_client()

        global_b64 = self._encode_image(global_img)
        wrist_b64  = self._encode_image(wrist_img)

        obs_summary = {
            k: round(v, 3) if isinstance(v, float) else v
            for k, v in obs.items()
            if k in (
                'joint1_deg', 'joint2_deg', 'joint3_deg', 'joint5_deg',
                'yaw_error_deg', 'pitch_error_rad',
                'ee_target_angle_deg', 'fine_visible', 'fine_confidence',
                'fine_du', 'fine_dv', 'fixed_dx_px', 'fixed_dy_px',
                'horiz_dist', 'plant_yaw_deg',
            )
        }

        user_text = (
            f'{_SYSTEM_PROMPT}\n\n'
            f'Phase: {phase}\n'
            f'Current state: {json.dumps(obs_summary)}\n\n'
            f'IMAGE 1 is the global camera (target circled). '
            f'IMAGE 2 is the wrist camera.\n'
            f'Output the JSON action on the last line.'
        )

        # Ollama chat message with multiple images.
        message = {
            'role': 'user',
            'content': user_text,
            'images': [global_b64, wrist_b64],
        }

        last_exc: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                t0 = time.time()
                response = client.chat(
                    model=self._model,
                    messages=[message],
                    options={'temperature': 0.1, 'num_predict': 256},
                )
                elapsed = time.time() - t0
                raw_text = response['message']['content']
                _LOG.info(f'VLM inference {elapsed:.2f}s, response: {raw_text[:120]}')
                return self._parse_response(raw_text, obs, phase)
            except Exception as exc:
                last_exc = exc
                _LOG.warning(f'VLM attempt {attempt+1} failed: {exc}')
                if attempt < self._max_retries:
                    time.sleep(0.5)

        raise RuntimeError(
            f'Ollama call failed after {self._max_retries+1} attempts'
        ) from last_exc

    def _parse_response(
        self,
        text: str,
        obs: Dict[str, Any],
        phase: str,
    ) -> Dict[str, Any]:
        """Scan response bottom-up for a JSON line starting with '{'."""
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        for line in reversed(lines):
            if line.startswith('{'):
                # Strip trailing punctuation that some models add.
                line = line.rstrip('.,;')
                try:
                    parsed = json.loads(line)
                    return self._validate_action(parsed, obs, phase)
                except (json.JSONDecodeError, KeyError, ValueError) as exc:
                    _LOG.debug(f'JSON parse attempt failed ({exc}): {line}')
                    continue
        _LOG.warning(f'No valid JSON found in VLM response: {text[:300]}')
        return self._heuristic_fallback(obs, phase)

    def _validate_action(
        self,
        parsed: Dict[str, Any],
        obs: Dict[str, Any],
        phase: str,
    ) -> Dict[str, Any]:
        action = str(parsed.get('action', '')).strip().lower()
        if action not in ('set_joints', 'delta_joints', 'coarse_ok', 'done'):
            raise ValueError(f'Unknown action: {action!r}')

        if action in ('set_joints', 'delta_joints'):
            key = 'joints_deg' if action == 'set_joints' else 'delta_deg'
            joints = parsed.get(key, {})
            if not joints:
                raise ValueError(f'Missing {key} in action')
            for jname in list(joints.keys()):
                val = float(joints[jname])
                if action == 'delta_joints' and abs(val) > 15.0:
                    raise ValueError(f'Delta too large: {jname}={val:.1f}°')
                joints[jname] = val
            parsed[key] = joints

        parsed['source'] = 'vlm_local'
        parsed.setdefault('phase', phase)
        return parsed

    def _heuristic_fallback(
        self,
        obs: Dict[str, Any],
        phase: str,
    ) -> Dict[str, Any]:
        from align_judge import decide_action
        result = decide_action(obs, phase=phase)
        result['source'] = 'heuristic_fallback'
        return result

    def _encode_image(self, img: np.ndarray) -> str:
        """Resize-if-needed, JPEG-encode, return base64 string."""
        h, w = img.shape[:2]
        if w > self._max_img_width:
            scale = self._max_img_width / w
            new_w = self._max_img_width
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # Convert RGB → BGR for cv2.
        bgr = img[:, :, ::-1].copy() if img.ndim == 3 else img
        ok, buf = cv2.imencode(
            '.jpg', bgr,
            [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
        )
        if not ok:
            raise RuntimeError('JPEG encoding failed')
        return base64.b64encode(buf.tobytes()).decode('ascii')

    def _get_client(self):
        if self._client is None:
            try:
                import ollama  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    'ollama package not found. Install with: pip install ollama'
                ) from exc
            self._client = ollama.Client(host=self._host)
            # Warm-up: verify the model is available.
            try:
                models = self._client.list()
                available = [m['name'] for m in models.get('models', [])]
                if not any(self._model in m for m in available):
                    _LOG.warning(
                        f'Model {self._model!r} not found in Ollama. '
                        f'Run: ollama pull {self._model}\n'
                        f'Available: {available}'
                    )
            except Exception as exc:
                _LOG.warning(f'Could not verify Ollama model list: {exc}')
        return self._client


# ---------------------------------------------------------------------------
# CLI test  (requires Ollama running with the model pulled)
#
#   python vlm_align_client.py --test
#   python vlm_align_client.py --test --model qwen2.5-vl:7b
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('--test', action='store_true')
    ap.add_argument('--model', default=_DEFAULT_MODEL)
    ap.add_argument('--host',  default=_DEFAULT_HOST)
    cli = ap.parse_args()

    if not cli.test:
        print(f'Usage: python vlm_align_client.py --test [--model {_DEFAULT_MODEL}]')
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)

    H, W = 480, 640
    global_img = np.full((H, W, 3), 180, dtype=np.uint8)
    wrist_img  = np.full((H, W, 3),  80, dtype=np.uint8)
    cv2.circle(global_img, (320, 240), 30, (255, 0, 0), 3)   # red circle = target
    cv2.putText(global_img, 'TARGET', (290, 210), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 0, 0), 2)

    obs = {
        'joint1_deg': 0.0, 'joint2_deg': -11.5,
        'joint3_deg': 17.2, 'joint5_deg': -28.6,
        'yaw_error_deg': -12.3, 'pitch_error_rad': 0.08,
        'ee_target_angle_deg': 25.0,
        'fine_visible': False, 'fine_confidence': 0.0,
        'fine_du': 0.0, 'fine_dv': 0.0,
        'fixed_dx_px': -85.0, 'fixed_dy_px': 30.0,
        'horiz_dist': 0.45, 'plant_yaw_deg': -12.0,
    }

    client = VLMAlignClient(model=cli.model, host=cli.host)
    print(f'Querying local Ollama ({cli.host})  model={cli.model} ...')
    action = client.decide(global_img, wrist_img, obs, phase='command')
    print(f'\nAction returned:\n{json.dumps(action, indent=2)}')
    assert 'action' in action, 'No action key in response'
    print('\nVLMAlignClient local test passed.')
