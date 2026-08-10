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

---

## Hardware Upgrade: DaBai Global + Gemini 305 Wrist (2026-08-10)

Commit: `b02d45d`  Branch: `pbvs-vlm-reach-v2`

### 背景 / Context

升级前存在两个根本性缺陷：

1. **进入方向选择从未运行**：`reach_fsm_node.py` 里 `_last_wrist_depth` 和 `_wrist_camera_K`
   从未被赋值（没有订阅对应话题），`approach_dir_selector` 的判断永远 False，PBVS 一直
   fallback 到 `+X base_link` 方向。
2. **全局相机无深度**：全局相机为 USB v4l2 摄像头，无法提供 3D 信息。

本次升级：
- 全局相机换 **Orbbec DaBai**（RGB-D，SDK v1 `OrbbecSDK_ROS2_main`）
- 腕部相机换 **Orbbec Gemini 305**（RGB-D，新 SDK `OrbbecSDK_ROS2`，需 `uvc_backend:=libuvc`）
- 修复两个 bug，并在 LOCKING 阶段利用全局深度构建 3D 进入方向地图

---

### Bug Fixes

#### Bug 1：腕部深度订阅缺失

`reach_fsm_node.py` 从未订阅 `/camera_wrist/depth/image_raw`，导致 `_last_wrist_depth` 永远
为 None，`approach_dir_selector` 条件永远不满足。

**Fix**：在 `__init__` 订阅块新增：
```python
self.create_subscription(
    Image, '/camera_wrist/depth/image_raw',
    self._on_wrist_depth, qos_profile_sensor_data, callback_group=self._cb_group)

def _on_wrist_depth(self, msg: Image) -> None:
    try:
        d = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
        self._last_wrist_depth = d.astype(np.float32) / 1000.0
    except Exception:
        pass
```

初始化：`self._last_wrist_depth: Optional[np.ndarray] = None`

#### Bug 2：`_wrist_camera_K` 从未以 3×3 矩阵形式赋值

`_on_wrist_info` 只存了 `_wrist_K`（4 元组），但 PBVS 代码引用的是 `_wrist_camera_K`
（3×3 numpy），因此始终为 None。

**Fix**：修改 `_on_wrist_info`，在已有赋值后追加：
```python
self._wrist_camera_K = np.array([
    [float(msg.k[0]), 0.0,             float(msg.k[2])],
    [0.0,             float(msg.k[4]), float(msg.k[5])],
    [0.0,             0.0,             1.0            ],
], dtype=np.float64)
```

初始化：`self._wrist_camera_K: Optional[np.ndarray] = None`

---

### New File: `scripts/berry_approach_mapper.py`

包含两个类：

#### `BerryApproachMapper`

底层几何工具，单帧操作：

```
Class: BerryApproachMapper

  build_local_map(depth_img, K, T_base_cam, berry_xyz_base,
                  radius_m=0.22, depth_min_m=0.05, depth_max_m=4.0)
    → N×3 ndarray (obstacle points in base_link)
    Back-projects valid depth pixels to base_link frame via T_base_cam.
    Keeps only points within radius_m sphere around berry.

  select_approach_direction(point_cloud, berry_xyz, ee_xyz=None,
                            n_candidates=36, lateral_blend=0.25,
                            cylinder_radius_m=0.03, ray_near_m=0.03, ray_far_m=0.25)
    → (3,) unit vector in base_link
    Seeds from current EE→berry direction.
    Casts 36 candidate rays outward from berry, counts point cloud hits
    inside a cylinder (radius 3 cm) between 3–25 cm from berry.
    Returns direction with lowest obstacle count.
    Falls back to EE→berry (or +X base_link) when point cloud is empty.
```

#### `FusedApproachMap`

滚动时间窗多源点云融合，持续接受来自全局相机（DaBai）和腕部相机（Gemini 305）的深度帧：

```
Class: FusedApproachMap(max_age_s=4.0)

  add_frame(depth_img, K, T_base_cam, berry_xyz,
            radius_m=0.22, depth_min_m=0.05, depth_max_m=4.0)
    → int  (added point count, 0 = nothing useful)
    Calls BerryApproachMapper.build_local_map(), appends (N×3, timestamp).
    Auto-prunes frames older than max_age_s.

  get_point_cloud() → N×3 merged array (all non-expired frames)
  get_approach_dir(berry_xyz, ee_xyz=None, **kwargs) → (3,) unit vector
  reset()         — clear all frames
  n_frames (int), n_points (int) — live stats
```

**设计意图**：全局相机在 REFINING 入口提供场景骨架（远视角，无遮挡），腕部相机在
SERVO 期间每 0.5 s 更新一帧（近视角，分辨率高）。两源数据在 base_link 坐标系下
直接合并，方向选择器同时看到两层信息，随手臂靠近动态收敛。

Unit tests embedded in `__main__`:
- Test 1: empty cloud → fallback to EE→berry
- Test 2: obstacle directly ahead → picks lateral direction
- Test 3: `build_local_map` geometry with flat wall
- Test 4: `FusedApproachMap` multi-frame merge + prune

---

### Modified: `scripts/reach_fsm_node.py`

新增三类改动，均不修改已有方法：

#### 新增订阅和状态变量

```python
# 订阅
self.create_subscription(Image, '/camera_fixed/depth/image_raw',
    self._on_fixed_depth, qos_profile_sensor_data, callback_group=self._cb_group)
self.create_subscription(CameraInfo, '/camera_fixed/color/camera_info',
    self._on_fixed_cam_info, qos_profile_sensor_data, callback_group=self._cb_group)
self.create_subscription(Image, '/camera_wrist/depth/image_raw',
    self._on_wrist_depth, qos_profile_sensor_data, callback_group=self._cb_group)

# 状态变量（在 _wrist_K 附近）
self._wrist_camera_K: Optional[np.ndarray] = None
self._last_wrist_depth: Optional[np.ndarray] = None
self._last_fixed_depth: Optional[np.ndarray] = None
self._fixed_camera_K: Optional[np.ndarray] = None
self._fixed_camera_frame: str = 'camera_fixed_color_optical_frame'
self._precomputed_approach_dir: Optional[np.ndarray] = None  # 保留（未使用，供调试）
self._fused_map: Optional[object] = None       # FusedApproachMap（_init_pbvs 创建）
self._pbvs_last_wrist_map_t: float = 0.0
```

#### `_build_approach_map_from_global()`

在 `_enter_fine_after_coarse` 里（`_init_pbvs` 之后）调用一次，将全局相机帧加入 fused map。

```python
def _build_approach_map_from_global(self) -> None:
    # 跳过条件：_fused_map 为 None（非 PBVS / align_only）
    #           或 无全局深度 / 无内参 / 未锁定果实
    # 查找 TF: base_link → camera_fixed_color_optical_frame（_tf_fixed_cam()）
    # fused_map.add_frame(global_depth, K_fixed, T, berry) → n_pts
    # fused_map.get_approach_dir(berry, ee) → self._pbvs_approach_dir_base
    # 日志：'FusedApproachMap: +N pts from global cam (total M), approach_dir=[...]'
```

#### `_init_pbvs()` — 创建 FusedApproachMap

```python
from berry_approach_mapper import FusedApproachMap
self._fused_map = FusedApproachMap(max_age_s=4.0)
self._pbvs_last_wrist_map_t = 0.0
# _pbvs_approach_dir_base 留 None，由随后的 _build_approach_map_from_global 设置
```

#### `_tick_refining_pbvs()` SERVO — 腕部帧时间门控更新

每 0.5 s 将腕部深度帧加入 fused map，重算方向（取代旧的单帧 `approach_dir_selector` 调用）：

```python
if (self._pbvs_state == 'SERVO'
        and depth_img is not None and K_wrist is not None
        and target_xyz is not None
        and getattr(self, '_fused_map', None) is not None):
    now_map = time.time()
    if now_map - self._pbvs_last_wrist_map_t >= 0.5:
        T_base_cam = self._tf_base_cam()
        if T_base_cam is not None:
            n_pts = self._fused_map.add_frame(
                depth_img, K_wrist, T_base_cam,
                target_xyz, radius_m=0.15, depth_min_m=0.10, depth_max_m=1.50)
            self._pbvs_last_wrist_map_t = now_map
            if n_pts > 0:
                new_dir = self._fused_map.get_approach_dir(
                    target_xyz, self._tcp_or_ee_xyz())
                self._pbvs_approach_dir_base = new_dir
```

#### `_enter_fine_after_coarse` 调用顺序（关键）

```
_init_pbvs()                     ← 先建 _fused_map
_build_approach_map_from_global() ← 再喂全局帧
_set_state('REFINING')
```

方向 fallback 链（优先级从高到低）：
1. **FusedApproachMap**：全局 RGB-D（REFINING 入口）+ 腕部 RGB-D（SERVO 每 0.5 s）
2. EE→berry 向量（`_tcp_or_ee_xyz()` 可用时）
3. `+Z base_link`（最后兜底）

---

### Modified: `scripts/depth_fusion.py`

Gemini 305 深度范围调整（原 Orbbec Astra 参数）：

| 参数 | 旧值 | 新值 |
|---|---|---|
| `_RGBD_MIN_M` | 0.04 m | **0.10 m** |
| `_RGBD_MAX_M` | 1.10 m | **1.50 m** |

---

### Modified: `src/picking_perception/config/foundation_pose.yaml`

fine_detector_node 参数更新（Gemini 305 特性 + 已校准直径）：

| 参数 | 旧值 | 新值 | 原因 |
|---|---|---|---|
| `depth_pose_source` | `mono` | **`depth`** | Gemini 305 近距离深度可靠 |
| `depth_min_m` | 0.03 | **0.10** | Gemini 305 最小可靠深度 |
| `depth_max_m` | 1.8 | **1.5** | 限制范围，减少噪声 |
| `berry_diameter_m` | 0.015 | **0.017** | 2026-08-05 用胶带校准 |

---

### Modified: `scripts/real_robot_bringup.sh`

#### 相机 SDK 拆分

```bash
ORBBEC_WS="${PIPER_X_DEV}/OrbbecSDK_ROS2"          # 新 SDK，腕部 Gemini 305
ORBBEC_WS_FIXED="${PIPER_X_DEV}/OrbbecSDK_ROS2_main"  # SDK v1，全局 DaBai
```

`gemini305.launch.py` 仅存在于新 SDK；`dabai.launch.py` 仅存在于旧 SDK。

#### 腕部相机（Gemini 305）

```bash
ORBBEC_LAUNCH=gemini305.launch.py
# 新增参数：
camera_args="... uvc_backend:=libuvc"   # Gemini 305 必须
# 序列号需填写：
ORBBEC_SERIAL=   # 运行 ros2 run orbbec_camera list_devices_node 查询
```

#### 全局相机（DaBai，原 v4l2）

完整替换 `ENABLE_FIXED_CAMERA` 块：
- 使用 `${ORBBEC_WS_FIXED}` source 旧 SDK
- `ros2 launch orbbec_camera dabai.launch.py` + `depth_registration:=true`
- 不加 `uvc_backend`（DaBai 旧 SDK 无此参数）
- 等待 color 和 depth 两个话题就绪
- TF 使用四元数形式发布（`--qx --qy --qz --qw`）
- 新增配置变量：`FIXED_CAM_SERIAL=`、`FIXED_CAM_USB_PORT=`、`FIXED_CAM_QX/Y/Z/W`

---

### Modified: `scripts/_run_refine_pbvs.sh`

fine_detector_node 启动参数新增 `-p depth_pose_source:=depth`，确保覆盖 yaml 默认值：

```bash
nohup /usr/bin/python3 -u \
  install/picking_perception/lib/picking_perception/fine_detector_node \
  --ros-args \
  -p enable_foundation_pose:=false \
  -p publish_hz:=10.0 \
  -p mask_source:=yolo \
  -p depth_pose_source:=depth \          # ← 新增
  --params-file src/picking_perception/config/foundation_pose.yaml \
```

---

### Deployment Checklist（部署前必做）

1. **查询序列号**，填入 `config/real_robot.env`：
   ```bash
   ros2 run orbbec_camera list_devices_node
   # 填写 ORBBEC_SERIAL=<Gemini 305 序列号>
   # 填写 FIXED_CAM_SERIAL=<DaBai 序列号>
   ```

2. **验证话题**：
   ```bash
   ros2 topic hz /camera_wrist/depth/image_raw
   ros2 topic hz /camera_fixed/depth/image_raw
   ros2 topic hz /camera_fixed/color/image_raw
   ```

3. **验证 FusedApproachMap 运行**（FSM 日志应出现）：
   ```
   PBVS loop initialised (FusedApproachMap ready)
   FusedApproachMap: +N pts from global cam (total M), approach_dir=[...]
   FusedApproachMap: +N wrist pts (total M), dir=[...]    ← DEBUG 级别，SERVO 期间
   ```
   不应出现 `FusedApproachMap (global frame) failed`。

4. **重新标定 TF**（换了 DaBai 物理位置后）：
   更新 `src/picking_description/calibration/fixed_camera_to_base.yaml`，
   重新运行 hand-eye 标定。当前先用旧标定文件凑合。

---

## REFINING 完整流程（PBVS 路径，2026-08-10 修订）

Commit: `5a0364e`

### 触发入口：两条路径收敛到同一函数

```
start_refine（调试脚本）             start（完整链路）
       ↓                                   ↓
  _on_cmd:                            LOCKING
  • 若 _locked=None，尝试               • _tick_lock_region()
    全局相机 fallback 锁定               • _build_approach_map_from_global()
  • 记录关节锚点 j1/j5                   → ALIGNING
       ↓                               • _tick_aligning()
  _enter_fine_after_coarse()                ↓
                                    _enter_fine_after_coarse()
```

注意：`start_refine` **直接进 REFINING，不走 LOCKING/ALIGNING**。
`_build_approach_map_from_global()` 现在也在 `_enter_fine_after_coarse` 里调用，
确保调试路径也能预计算进入方向。

---

### `_enter_fine_after_coarse()` 执行顺序

```
1. 保存关节锚点 (j1/j2/j3/j5 anchor)
2. _reset_refine_state(keep_anchors=True)     ← 清除上一轮所有状态
3. _refine_await_tracker_reset = True
4. self._fine = None    ← 丢弃旧 BoT-SORT 缓存，防止 stale berry 被 pin
5. _init_pbvs()         ← 每次进入都重新初始化，多轮 --loops N 安全
   • BerryKFTracker 重置；_pbvs_state = 'INIT'
   • FusedApproachMap(max_age_s=4.0) 重置；_pbvs_last_wrist_map_t = 0.0
   • _pbvs_approach_dir_base = None（由步骤 6 填充）
6. _build_approach_map_from_global()
   • 条件：_fused_map 非 None + _last_fixed_depth 非 None + _locked 非 None
   • TF 查询 base_link → camera_fixed_color_optical_frame
   • fused_map.add_frame(global_depth, K_fixed, T, berry, radius_m=0.22)
   • fused_map.get_approach_dir(berry, ee) → _pbvs_approach_dir_base
   • 日志：'FusedApproachMap: +N pts from global cam (total M), approach_dir=[...]'
7. _set_state('REFINING')
```

---

### `_tick_refining_pbvs()` 主循环

每个 FSM tick 调用一次。每 tick 先采集腕部数据：

```python
berries = self._fine if self._fine_is_fresh() else None
# 取 berries.berries[0]（DetectedBerry），经 TF 转换到 base_link
# DetectedBerry.pose 是 PoseStamped → 位置用 b.pose.pose.position.x/y/z
# 深度来源：b.depth_mode → sigma:
#   rgbd/fused_rgbd → 5mm，da2 → 12mm，mono → 25mm
# 中心像素：b.image_u, b.image_v（非 bbox_xyxy，该字段不在消息里）
```

**PBVS 子状态机：**

```
INIT
  • 等第一帧有效 berry_xyz_base（self._fine = None 时自然等待，等价于 tracker reset）
  • KF.initialize(xyz, sigma_init = berry_sigma × 3)
  → SERVO

SERVO（每 tick）
  • KF.predict(dt)
  • 若有新检测：KF.update(xyz, sigma)
  • 若不确定度 trace(P) > 4×8e-4 且无在途 probe → PROBE
  • 每 0.5 s：fused_map.add_frame(wrist_depth, K_wrist, T_base_wrist,
                                  target_xyz, radius_m=0.15)
              若 n_pts > 0：_pbvs_approach_dir_base 更新为 fused_map.get_approach_dir()
  • approach_dir = _pbvs_approach_dir_base
      若仍 None → fallback EE→berry → fallback +Z
  • standoff_xyz = KF.position - approach_dir × 0.07m
  • cup_axis_ik(standoff_xyz, target_xyz) → q_target
  • send_joint_servo_goal(q_target, 0.15s)
  进入 APPROACH 条件：trace(P) < 8e-4 AND cup_dist < 9.1cm

PROBE（主动三角测量，降低 KF 不确定度）
  • 记录当前关节 q_before
  • joint1 +3°，发 0.4s 轨迹
  • 收集 ≥2 帧 berry_xyz_base，取均值
  • KF.update_triangulated(mean_xyz, sigma=4mm)
  • 回到 q_before（0.4s 轨迹）
  → SERVO

APPROACH（KF 已收敛）
  • KF 冻结：只 predict，不 update
  • standoff_xyz = KF.position（offset=0，直接朝 berry）
  • cup_axis_ik(target_xyz, approach_dir) → q_target
  • send_joint_servo_goal(q_target, 0.15s)
  终止条件：cup_dist < 15mm

DONE
  • 发布 /reach/reached = True
  • FSM → WAIT_CONFIRM
```

**进入方向优先级：**

```
① _precomputed_approach_dir  ← LOCKING 或 _enter_fine_after_coarse 时，
                                全局 RGB-D 点云 + 36 射线投票（最优）
② approach_dir_selector      ← SERVO 首 tick，用腕部深度图在线计算
③ EE → berry                 ← KF.position 和 tcp_or_ee_xyz() 此时均可用，
                                直接计算当前末端到果实的单位向量
④ +Z base_link               ← tcp_pose 彻底不可用时的最后兜底（不应发生）
```

---

### 已修复的 Bug（commit `5a0364e`）

| 位置 | Bug | 影响 | 修复 |
|---|---|---|---|
| `_enter_fine_after_coarse` | `start_refine` 跳过 LOCKING，`_build_approach_map_from_global` 从不运行 | 调试路径永远用 +X | 在 `_enter_fine_after_coarse` 里调用 |
| `_enter_fine_after_coarse` | 多轮 `--loops N` 时 PBVS 状态不重置（`_pbvs_kf` 已存在，`_init_pbvs` 不再调用） | 第二轮 `_pbvs_state='DONE'` 直接跳过 | 每次显式调 `_init_pbvs()` |
| `_tick_refining_pbvs` | `_last_fine_berries`（从未赋值）替代 `self._fine` | `berries` 永远 None，KF 卡死 INIT | 换成 `self._fine if self._fine_is_fresh() else None` |
| `_tick_refining_pbvs` | `b.pose.position.x`（PoseStamped 需双层 `.pose`） | try/except 捕获，`berry_xyz_base` 永远 None | 改为 `b.pose.pose.position.x/y/z` |
| `_tick_refining_pbvs` | `getattr(b, 'mode', ...)` 字段不存在 | 深度 sigma 始终用默认 0.020 | 改为 `b.depth_mode` |
| `_tick_refining_pbvs` | `b.bbox_xyxy` 字段不在 DetectedBerry.msg 里 | approach_dir 在线计算异常 | 改为 `b.image_u, b.image_v` |

---

### 正常成功的日志序列

```
BerryApproachMapper: 312 pts around berry, approach_dir=[0.85 0.02 0.21]
PBVS loop initialised
PBVS INIT → SERVO  berry_base=[0.341 0.008 0.201]
PBVS active probe: step_deg=3.0
PBVS probe triangulated: [0.342 0.007 0.199]  uncertainty→42.3 µm²
PBVS SERVO → APPROACH  dist=8.7 cm
PBVS DONE: cup_dist=11.2 mm
```
