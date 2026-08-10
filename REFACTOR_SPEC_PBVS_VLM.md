# Refactor Spec: PBVS + VLM Reach v2

Branch: `pbvs-vlm-reach-v2`  
Base: `reach-pipeline-ubuntu`

This document is the single source of truth for all architectural changes in this branch.
Any agent reading this file should be able to reproduce the full implementation.

---

## Goals

1. **ALIGNING stage** — replace hardcoded pixel-gain heuristic with VLM + analytical IK.
2. **REFINING stage** — replace the two-step IBVS (center → approach) with continuous
   PBVS driven by a 3D Kalman Filter, with an active-probe trigger when KF uncertainty
   is high, and an approach-direction selector for obstacle avoidance.
3. **Depth pipeline** — prefer real RGB-D depth from the new wrist camera (5 cm – 1 m);
   fall back to Depth Anything V2 (metric) when RGB-D pixels are invalid or sparse.
4. **BC policy** — collect alignment episodes (real robot + VLM-generated), train a
   lightweight behavior-cloning model, and deploy it as the primary ALIGNING decision
   maker (~10 ms inference, replaces both heuristic and VLM at serving time).

Each goal is an independent, backward-compatible addition guarded by a parameter flag.
The old code paths are never deleted — they remain as fallbacks.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│  ALIGNING stage                                                      │
│                                                                      │
│  [Old path] align_judge.decide_action() → pixel-gain heuristic      │
│  [New path] vlm_align_client.decide()  → Claude vision API          │
│              └─ global_img + wrist_img + obs_dict                    │
│              └─ returns set_joints / delta_joints / coarse_ok        │
│  Guard param:  use_vlm_align (default False until validated)         │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  REFINING stage                                                      │
│                                                                      │
│  [Old path] center_oneshot (aim_uv_ik) → range_depth (cup_axis_ik) │
│  [New path] Continuous PBVS loop @ 20–30 Hz                         │
│    1. DepthFuser  → reliable z for KF measurement                   │
│    2. BerryKFTracker → [x,y,z] estimate in base frame               │
│       • predict() each tick                                          │
│       • update() when YOLO fires                                     │
│       • needs_active_probe() → small lateral step → triangulate     │
│    3. ApproachDirSelector → best entry direction from depth image    │
│    4. pbvs_standoff_ik() → arm to standoff point along best dir     │
│    5. At 5 cm: close-eyes FK approach + vacuum contact confirm       │
│  Guard param: use_pbvs (default False until validated)               │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  Depth pipeline (fine_detector_node)                                 │
│                                                                      │
│  [Old] depth_pose_source='mono'  (size-based, ignores RGB-D)        │
│  [New] depth_pose_source='fused'                                     │
│    1. Try RGB-D: median over valid pixels in YOLO mask               │
│       valid ratio threshold: > 30% of mask pixels must have depth   │
│    2. Fallback → Depth Anything V2 metric (indoor-small)             │
│    3. Final fallback → mono (existing size-based estimate)           │
│  Guard param: depth_pose_source='fused' (must set explicitly)        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## New Files

All new files live in `blueberry_picking_ws/scripts/` (same dir as reach_fsm_node.py).

### `scripts/berry_kf_tracker.py`

3D Kalman Filter tracking berry position in `base_link` frame.

```
Class: BerryKFTracker
  State x: [px, py, pz]  (meters, base_link)
  Cov   P: 3×3

Methods:
  initialize(xyz, sigma_init=0.05)
  predict(dt)                        # constant-position, Q = 1e-4*dt*I
  update(xyz_meas, sigma_meas)       # standard KF
  uncertainty() -> float             # trace(P)
  needs_active_probe(thresh) -> bool # uncertainty() > thresh
  position -> Optional[np.ndarray]   # current estimate, None if not init
  position_covariance -> np.ndarray  # current P
  reset()
```

Measurement noise schedule (passed in from caller):
```
sigma_meas = base_sigma / sqrt(valid_depth_ratio + 0.01)
base_sigma:
  'rgbd' source  → 0.005 m   (new wrist camera, reliable)
  'da2'  source  → 0.012 m   (metric DA2, moderate)
  'mono' source  → 0.025 m   (size-based, coarse)
```

Active probe protocol (called from reach_fsm_node._maybe_active_probe()):
```
1. Execute lateral step:  delta_j1 = ±PROBE_STEP_DEG (default 3°)
2. Record (q_before, u_before, v_before) and (q_after, u_after, v_after)
3. Triangulate 3D from two views using camera_K and T_base_cam at each step
4. kf.update(xyz_triangulated, sigma_meas=PROBE_SIGMA=0.003)
5. Return arm to original q
```

### `scripts/depth_fusion.py`

Priority-ordered depth estimation for wrist camera.

```
Class: DepthFuser
  __init__(use_da2=True, da2_model='depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf')
    Lazy-loads DA2 on first use to avoid startup delay.

  get_depth(rgb_img, raw_depth_m, mask) -> Tuple[float, str]
    Returns (depth_meters, source) where source ∈ {'rgbd', 'da2', 'mono', 'unavailable'}

    Step 1 — RGB-D:
      valid = raw_depth_m[mask & (raw_depth_m > 0.04) & (raw_depth_m < 1.1)]
      if len(valid) / mask_area > 0.30:
          return median(valid), 'rgbd'

    Step 2 — Depth Anything V2:
      if self.da2_available:
          metric_map = da2(rgb_img)          # full-image inference
          vals = metric_map[mask]
          return median(vals), 'da2'

    Step 3 — None (caller falls back to mono):
      return None, 'unavailable'

  _load_da2():  # lazy, called on first get_depth() that needs it
    from transformers import pipeline
    self._da2_pipe = pipeline("depth-estimation", model=self._da2_model)
    self.da2_available = True
```

Install dependency (add to requirements or run once):
```bash
pip install transformers accelerate
# model weights auto-downloaded from HuggingFace on first use (~100 MB)
```

### `scripts/approach_dir_selector.py`

Selects the least-obstructed approach direction for PBVS final approach.

```
Function: select_approach_direction(
    depth_img,          # H×W float32, meters (0 = invalid)
    target_uv,          # (u, v) center of target berry in image
    target_depth_m,     # estimated depth of target berry
    camera_K,           # 3×3 intrinsic matrix
    n_candidates=36,    # angular resolution
    cone_radius_px=20,  # lateral offset radius for sampling
    sample_radius_px=8, # radius of sampling circle per candidate
    obstacle_margin_m=0.03,  # pixels closer than target-margin are obstacles
) -> np.ndarray  # unit vector in CAMERA frame

Algorithm:
  For each angle θ in linspace(0, 2π, n_candidates):
    du = cone_radius_px * cos(θ)
    dv = cone_radius_px * sin(θ)
    sample_uvs = circle_pixels(target_u+du, target_v+dv, sample_radius_px)
    depths = depth_img[valid sample_uvs]
    clearance = 1 - fraction(depths < target_depth_m - obstacle_margin_m)
    record (θ, clearance)
  
  best_θ = argmax(clearances)
  # Build 3D direction: forward (0,0,1) + small lateral component
  lateral = (cos(best_θ)/fx, sin(best_θ)/fy, 0)
  direction = normalize([0,0,1] + 0.25 * lateral)
  return direction   # caller transforms to base frame via T_base_cam
```

Fallback: if all clearances < 0.3 (dense cluster), return [0,0,1] (straight ahead).

### `scripts/vlm_align_client.py`

Replaces `align_judge.decide_action()` heuristic with a **local** VLM call via Ollama.
No internet or cloud API key required.

```
Backend: Ollama (http://localhost:11434)
Model:   qwen2.5-vl:7b  (default, ~16 GB download, ~1-2 s/call on RTX 3090)

Setup (one-time):
  curl -fsSL https://ollama.com/install.sh | sh
  ollama pull qwen2.5-vl:7b
  pip install ollama

Class: VLMAlignClient
  __init__(model='qwen2.5-vl:7b', host='http://localhost:11434',
           max_retries=1, timeout_s=15.0)

  decide(global_img, wrist_img, obs, phase='command') -> dict
    # same schema as align_judge.decide_action()

Message format (ollama.Client.chat):
  messages=[{
    'role': 'user',
    'content': system_prompt + obs_json + instructions,
    'images': [global_b64, wrist_b64],   # base64 JPEG, max 1024px wide
  }]
  options={'temperature': 0.1, 'num_predict': 256}

Response parsing:
  Scan response lines bottom-up for first line starting with '{'.
  Validate action schema (set_joints / delta_joints / coarse_ok).
  On any error: fall back to align_judge.decide_action(obs).

Test:
  python vlm_align_client.py --test [--model qwen2.5-vl:7b] [--host http://localhost:11434]
```

---

## Modified Files

### `blueberry_picking_ws/scripts/align_judge.py`

**Change:** add `decide_action_vlm()` that wraps `VLMAlignClient`, called from
`reach_fsm_node._tick_aligning()` when `self._use_vlm_align` is True.

```python
# Add at bottom of align_judge.py:

_vlm_client: Optional['VLMAlignClient'] = None  # module-level singleton

def decide_action_vlm(
    obs: dict,
    global_img: np.ndarray,
    wrist_img: np.ndarray,
    phase: str = 'command',
    **kwargs,
) -> dict:
    """VLM-backed decision; falls back to decide_action() on any error."""
    global _vlm_client
    if _vlm_client is None:
        from vlm_align_client import VLMAlignClient
        _vlm_client = VLMAlignClient()
    try:
        return _vlm_client.decide(global_img, wrist_img, obs, phase=phase)
    except Exception as exc:
        import logging
        logging.getLogger(__name__).warning(f'VLM align failed ({exc}), using heuristic')
        return decide_action(obs, phase=phase, **kwargs)
```

### `blueberry_picking_ws/src/picking_perception/picking_perception/fine_detector_node.py`

**Change:** add `depth_pose_source='fused'` branch in `_det_to_cam_pose()`.

```python
# In FineDetectorNode.__init__, add parameter:
self.declare_parameter('depth_pose_source', 'mono')   # already exists
# (new value 'fused' now also accepted — no other __init__ change needed)

# In _det_to_cam_pose(), after computing z_depth and z_mono, add:
if self._depth_pose_source == 'fused':
    fused_z, fused_src = self._depth_fuser.get_depth(
        rgb_img=self._last_rgb,      # stored each detection cycle
        raw_depth_m=depth,
        mask=det.mask,
    )
    if fused_z is not None:
        z = float(fused_z)
        mode = f'fused_{fused_src}'
    else:
        z = float(z_mono)
        mode = 'mono_fallback'
```

Add `DepthFuser` instantiation in `__init__`:
```python
# After parameter declarations, in __init__:
from depth_fusion import DepthFuser
use_da2 = self.get_parameter('depth_pose_source').value == 'fused'
self._depth_fuser = DepthFuser(use_da2=use_da2)
```

Store last RGB frame in the detection callback so `_depth_fuser` can use it:
```python
# In _cb_color (or wherever rgb msg is processed), add:
self._last_rgb = rgb_img   # np.ndarray H×W×3 uint8
```

### `blueberry_picking_ws/scripts/reach_fsm_node.py`

**Three independent additions — do not modify existing methods.**

#### Addition 1: ALIGNING VLM route

In `_tick_aligning()` around line 757, before the existing `_load_align_decision_from_file` call:

```python
# --- NEW: VLM align route ---
if getattr(self, '_use_vlm_align', False):
    global_img = self._last_fixed_rgb   # already stored
    wrist_img  = self._last_wrist_rgb   # already stored
    if global_img is not None and wrist_img is not None:
        from align_judge import decide_action_vlm
        decision = decide_action_vlm(obs, global_img, wrist_img, phase=phase)
        self._execute_align_decision(decision)
        return True
# --- END NEW ---
```

Add parameter in `__init__` or from CLI arg (already has `--action-judge-source`):
```python
self._use_vlm_align = (self._args.action_judge_source == 'vlm')
```

#### Addition 2: PBVS refining route

Add `use_pbvs` ROS parameter (declare in `__init__`):
```python
self.declare_parameter('use_pbvs', False)
self._use_pbvs = bool(self.get_parameter('use_pbvs').value)
```

Add a new method `_tick_refining_pbvs()` that runs the PBVS loop.
Called from `_start_refining()` if `self._use_pbvs` is True.

Full method signature and logic: see Implementation Notes below.

#### Addition 3: `--action-judge-source` new value

In the argparse section (around line 6480), add `'vlm'` to the choices list:
```python
help='... choices: file | heuristic | vlm'
```

---

## `_tick_refining_pbvs()` — Implementation Notes

This method replaces the center_oneshot → range_depth two-step with a single
continuous loop. It is called each FSM tick when state == REFINING and use_pbvs==True.

```
State machine (internal to PBVS):
  INIT       → first YOLO detection, initialize KF
  SERVO      → continuous: predict KF, update KF if YOLO fires, run standoff IK
  PROBE      → active lateral probe when kf.needs_active_probe()
  APPROACH   → kf.uncertainty() < READY_THRESH and dist < STANDOFF_M*1.2 → straight in
  DONE       → dist < CONTACT_DIST_M, publish /reach/reached

Key constants:
  PBVS_STANDOFF_M     = 0.07   # desired standoff distance (7 cm from cup to berry)
  PBVS_READY_THRESH   = 8e-4   # KF uncertainty (trace P) below this → enter APPROACH
  PBVS_CONTACT_DIST_M = 0.015  # cup-to-berry distance triggering DONE
  PROBE_STEP_DEG      = 3.0    # lateral probe joint1 delta
  PROBE_SIGMA_M       = 0.003  # triangulated observation noise

SERVO tick logic:
  1. berry_msgs = self._last_fine_berries (DetectedBerryArray from /perception/fine/berries)
  2. if berry_msgs not empty:
       xyz_cam = berry_msgs[0].pose.position (x,y,z in camera frame)
       T_base_cam = self._tf_lookup('base_link', camera_frame)
       xyz_base = T_base_cam @ [xyz_cam, 1]
       sigma = depth_source_sigma(berry_msgs[0].mode)  # 'fused_rgbd'→0.005 etc.
       self._pbvs_kf.update(xyz_base[:3], sigma)
  3. self._pbvs_kf.predict(dt)
  4. if self._pbvs_kf.needs_active_probe(PBVS_READY_THRESH * 4):
       self._pbvs_state = 'PROBE'
       return
  5. target_xyz = self._pbvs_kf.position
     approach_dir_cam = select_approach_direction(depth_img, berry_uv, berry_z, K)
     approach_dir_base = (T_base_cam[:3,:3] @ approach_dir_cam)
     standoff_xyz = target_xyz - approach_dir_base * PBVS_STANDOFF_M
  6. q_target = cup_axis_ik(standoff_xyz, approach_dir_base, q_current, robot_model)
     if q_target is None: return  # IK failed, skip tick
  7. self._send_joint_servo_goal(q_target, duration_s=0.15)  # fast short steps

APPROACH tick logic (kf settled, close enough):
  Same as current near_oneshot + range_depth:
  target_xyz = self._pbvs_kf.position  (frozen — kf.predict() only, no more updates)
  Use cup_axis_ik_chunked to walk remaining distance.
  Contact detection: same _cup_berry_dist() < PBVS_CONTACT_DIST_M.
```

---

## Integration Order

Implement in this order to allow incremental testing:

1. **`depth_fusion.py`** — standalone, no ROS dependency. Test with a static depth image.
2. **`berry_kf_tracker.py`** — standalone, pure numpy. Unit test with synthetic trajectory.
3. **`approach_dir_selector.py`** — standalone. Test with a synthetic depth image.
4. **`vlm_align_client.py`** — requires `ANTHROPIC_API_KEY` env var.
5. **`fine_detector_node.py`** changes — set `depth_pose_source:=fused` to activate.
6. **`align_judge.py`** addition — set `--action-judge-source vlm` to activate.
7. **`reach_fsm_node.py`** additions — set `use_pbvs:=true` to activate.

---

## Dependencies

```bash
# Ollama runtime (local VLM inference)
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5-vl:7b          # ~16 GB, one-time download
pip install ollama

# Depth Anything V2 (lazy-loaded, ~100 MB weights auto-downloaded from HuggingFace)
pip install transformers>=4.40 accelerate

# Already present: numpy, opencv-python, rclpy, ultralytics
```

No cloud API key required. Ollama runs entirely on the local GPU.

---

## Testing Checkpoints

| Checkpoint | Command | Expected |
|------------|---------|----------|
| Depth fuser RGB-D path | `python -c "from depth_fusion import DepthFuser; ..."` | Returns float, source='rgbd' |
| Depth fuser DA2 path | Same, with invalid depth | Returns float, source='da2' |
| KF tracker init+update | `python berry_kf_tracker.py` (has `__main__` test) | Position converges |
| Approach dir selector | `python approach_dir_selector.py` (has `__main__` test) | Non-zero direction |
| VLM align (dry run) | `ANTHROPIC_API_KEY=... python vlm_align_client.py --test` | Prints parsed action |
| Full ALIGNING VLM | Launch with `--action-judge-source vlm` | No heuristic fallback in logs |
| Full REFINING PBVS | Launch with `use_pbvs:=true` | KF uncertainty printed each tick |

---

---

## Goal 4: BC Policy Pipeline

### Why BC beats VLM at serving time

| | VLM-7B (Qwen) | Fine-tuned VLM-3B | **BC Policy** |
|---|---|---|---|
| Inference | 1-2 s | 300-500 ms | **5-15 ms** |
| Output type | token generation | token generation | **direct regression** |
| Joint angle precision | ±3-8° | ±2-5° | **±0.5-1.5°** |
| Training data | zero-shot | 100+ episodes | 100+ episodes |
| Deployment | Ollama daemon | Ollama daemon | **single .pt file** |

VLM 用于数据生成阶段（teacher），BC 用于最终部署（student）。

---

### Data Flow

```
Phase 1 — Collection (real robot, heuristic or VLM as teacher)
  reach_fsm_node
    └─ AlignDataCollector.log_step(global_img, wrist_img, obs, action)
    └─ AlignDataCollector.mark_success()   ← called at _enter_fine_after_coarse()
    └─ AlignDataCollector.end_episode()    ← called at IDLE reset
  Saves to: data/align_episodes/episode_NNNNN/

Phase 2 — Synthetic Generation (random starts + VLM teacher)
  generate_align_data.py --n-episodes 500 --teacher vlm
    └─ Moves arm to random joint config (within safe workspace)
    └─ Triggers FSM ALIGNING with --align-judge-mode vlm
    └─ Data collector captures automatically
    └─ Resets and repeats

Phase 3 — Training
  training/train_bc_align.py --data-dir data/align_episodes --output-dir models/
    └─ AlignDataset: loads episodes, flattens to (obs, action) pairs
    └─ BCAlignPolicy: CLIP encoder (frozen) + fusion MLP
    └─ Loss: MSE(predicted_joints, label_joints)
    └─ Saves: models/bc_align_policy.pt + models/bc_align_norm.json

Phase 4 — Deployment
  --align-judge-mode bc --bc-model-path models/bc_align_policy.pt
    └─ align_judge.decide_action_bc() replaces heuristic
    └─ VLM kept as fallback for align_fail_count > 2
```

---

### Episode Data Format

```
data/align_episodes/
  episode_00001/
    metadata.json          # {success, n_steps, plant_xyz, teacher}
    steps.json             # [{obs, action, step_idx}, ...]
    step_000_global.jpg    # 90% JPEG
    step_000_wrist.jpg
    step_001_global.jpg
    ...
```

`steps.json` schema per step:
```json
{
  "step_idx": 0,
  "obs": {
    "joint1_deg": 0.0, "joint2_deg": -11.5, "joint3_deg": 17.2, "joint5_deg": -28.6,
    "yaw_error_deg": -12.3, "pitch_error_rad": 0.08,
    "ee_target_angle_deg": 25.0,
    "fine_visible": 0.0, "fine_du": 0.0, "fine_dv": 0.0,
    "fixed_dx_px": -85.0, "fixed_dy_px": 30.0,
    "horiz_dist": 0.45, "plant_yaw_deg": -12.0
  },
  "action": {
    "action": "set_joints",
    "joints_deg": {"joint1": 12.0, "joint2": -14.0, "joint3": 18.0, "joint5": -30.0},
    "source": "heuristic"
  }
}
```

Only steps from **successful episodes** are used for training.

---

### BC Model Architecture (`scripts/bc_align_policy.py`)

```
Input:
  global_img  (3, 224, 224)   CLIP-preprocessed
  wrist_img   (3, 224, 224)   CLIP-preprocessed
  obs_vec     (14,)            normalised observation

Architecture:
  CLIPVisionModel (openai/clip-vit-base-patch16, frozen)
    → pooler_output (512,) per image

  img_proj: Linear(512 → 256) × 2 images
  obs_enc:  Linear(14 → 128) → LayerNorm → ReLU → Linear(128 → 256)

  fusion:   cat([g_feat, w_feat, obs_feat]) (768,)
            → Linear(768 → 256) → LayerNorm → ReLU
            → Linear(256 → 128) → ReLU

  joint_head: Linear(128 → 4)   outputs [j1,j2,j3,j5] in degrees
  done_head:  Linear(128 → 1)   outputs logit (is alignment ready?)

Total trainable params: ~500K (CLIP encoder frozen, ~86M frozen)
```

Observation normalisation constants (saved to `bc_align_norm.json`):
```
joint*_deg   → / 90.0
yaw_error_deg → / 90.0
pitch_error_rad → / 1.57
ee_target_angle_deg → / 90.0
fine_visible → as-is (0/1)
fine_du/dv   → / 320.0   (half image width)
fixed_dx/dy_px → / 320.0
horiz_dist   → / 1.0     (already metres, roughly 0-1)
plant_yaw_deg → / 90.0
```

---

### Training (`training/train_bc_align.py`)

```bash
pip install torch torchvision transformers
python training/train_bc_align.py \
  --data-dir data/align_episodes \
  --output-dir models/ \
  --epochs 100 \
  --batch-size 64 \
  --lr 1e-4
```

Loss:
```
L = MSE(pred_joints, label_joints)          # primary
  + 0.1 * BCE(pred_done_logit, is_done)     # auxiliary
```

Expected results with 200+ successful episodes:
- val joint RMSE < 2° after 100 epochs
- Inference: 5-15 ms on RTX 3090

---

### Synthetic Data Generation (`scripts/generate_align_data.py`)

```bash
# Requires: reach_fsm_node running in background, data collector enabled
python scripts/generate_align_data.py \
  --n-episodes 500 \
  --teacher vlm \          # or: heuristic
  --joint1-range -60 60 \  # randomise yaw
  --joint2-range -30 10 \
  --joint3-range 0 40 \
  --joint5-range -60 -10
```

Algorithm:
1. Sample random (j1, j2, j3, j5) from specified ranges
2. Move arm to that config via `/control/move_j`
3. Publish `start` command to FSM command topic
4. Wait for FSM state → REFINING (success) or ERROR (fail), timeout 60s
5. Data collector saves episode automatically
6. Send `reset` command, wait for IDLE, repeat

Safety guards:
- Reject configs where EE is closer than 0.15 m to any known obstacle
- Reject configs with joint velocity > 30°/s required
- If FSM enters ERROR 3× in a row: pause and alert operator

---

### Deployment (`align_judge.decide_action_bc`)

```python
# In align_judge.py (already wired, see Modified Files section)
def decide_action_bc(obs, global_img, wrist_img, phase='command') -> dict:
    # 1. Check align_entry_ready first (no model call needed)
    if align_entry_ready(obs, ...):
        return {'action': 'coarse_ok', 'source': 'bc_gate'}

    # 2. Run BC model
    joints, done_logit = _bc_policy.predict(global_img, wrist_img, obs)

    # 3. done_head says ready?
    if torch.sigmoid(done_logit).item() > 0.8:
        return {'action': 'coarse_ok', 'source': 'bc_done_head'}

    return {
        'action': 'set_joints',
        'joints_deg': {'joint1': joints[0], 'joint2': joints[1],
                       'joint3': joints[2], 'joint5': joints[3]},
        'source': 'bc_policy',
    }
```

Activate: `--align-judge-mode bc --bc-model-path models/bc_align_policy.pt`

VLM fallback still active when `align_fail_count > 2`.

---

### New Files for Goal 4

| File | Purpose |
|------|---------|
| `scripts/align_data_collector.py` | Episode logging, hooks into FSM |
| `scripts/bc_align_policy.py` | Model definition + inference wrapper |
| `scripts/generate_align_data.py` | Synthetic data generation (random starts) |
| `training/align_dataset.py` | PyTorch Dataset for alignment episodes |
| `training/train_bc_align.py` | Training loop |

---

## Hard Rules (inherited from SKILL.md)

- Do not touch Gazebo / BT / vibration code.
- Do not use open-loop `build_cup_axis_plan` for approach.
- All new parameters default to the old behavior (False / 'mono') so existing
  launch files continue to work unchanged.
- QA image writes (`_write_align_observation`) must still happen in the VLM path.
- The 5 cm close-eyes approach and vacuum contact confirmation logic is unchanged.
