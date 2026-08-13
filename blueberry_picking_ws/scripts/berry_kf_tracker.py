"""3-D Kalman Filter that tracks a single berry's position in base_link frame.

State:  x = [px, py, pz]    (metres, base_link)
Model:  constant-position with small process noise for plant sway.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# Process noise per second (accounts for plant sway, ~1 mm/s²).
_Q_RATE = 1e-4   # m²/s per axis

# Threshold for triggering an active lateral probe.
DEFAULT_PROBE_THRESHOLD = 8e-4   # trace(P) in m²

# χ²(3 dof, 99%) — standard innovation gate for 3-D position updates.
CHI2_3_DOF_99 = 11.345


@dataclass(frozen=True)
class KFUpdateResult:
    """Outcome of a gated KF measurement update."""

    accepted: bool
    reason: str = ''
    mahal_sq: float = 0.0
    innov_norm: float = 0.0


class BerryKFTracker:
    """Lightweight 3-D position KF for a single berry in base_link frame."""

    def __init__(self) -> None:
        self._x: Optional[np.ndarray] = None   # (3,) position estimate
        self._P: Optional[np.ndarray] = None   # (3,3) covariance

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self, xyz: np.ndarray, sigma_init: float = 0.05) -> None:
        """Initialise from a first 3-D observation."""
        self._x = np.array(xyz, dtype=np.float64).flatten()[:3]
        self._P = np.eye(3, dtype=np.float64) * sigma_init ** 2

    def reset(self) -> None:
        self._x = None
        self._P = None

    @property
    def initialized(self) -> bool:
        return self._x is not None

    # ------------------------------------------------------------------
    # KF steps
    # ------------------------------------------------------------------

    def predict(self, dt: float) -> None:
        """Constant-position prediction step."""
        if not self.initialized:
            return
        Q = np.eye(3, dtype=np.float64) * (_Q_RATE * max(dt, 0.0))
        self._P = self._P + Q  # type: ignore[operator]

    def update(self, xyz_meas: np.ndarray, sigma_meas: float) -> None:
        """Standard linear KF update with a 3-D position measurement.

        sigma_meas — 1-sigma measurement noise in metres (isotropic).
        Caller should scale by depth source:
          'rgbd'  → 0.005 m
          'da2'   → 0.012 m
          'mono'  → 0.025 m
        """
        if not self.initialized:
            self.initialize(xyz_meas, sigma_init=sigma_meas * 3)
            return

        z = np.array(xyz_meas, dtype=np.float64).flatten()[:3]
        R = np.eye(3, dtype=np.float64) * sigma_meas ** 2
        H = np.eye(3, dtype=np.float64)

        S = H @ self._P @ H.T + R  # type: ignore[operator]
        K = self._P @ H.T @ np.linalg.inv(S)  # type: ignore[operator]
        innov = z - H @ self._x  # type: ignore[operator]
        self._x = self._x + K @ innov  # type: ignore[operator]
        self._P = (np.eye(3) - K @ H) @ self._P  # type: ignore[operator]

    def gated_update(
        self,
        xyz_meas: np.ndarray,
        sigma_meas: float,
        *,
        physical_max_m: Optional[float] = None,
        chi2_threshold: float = CHI2_3_DOF_99,
    ) -> KFUpdateResult:
        """KF update with Mahalanobis + optional physical innovation gates.

        Rejected measurements leave ``position`` unchanged (predict-only for
        this step).  Returns diagnostics for logging / safety aborts.
        """
        if not self.initialized:
            self.initialize(xyz_meas, sigma_init=sigma_meas * 3)
            return KFUpdateResult(True, reason='init')

        z = np.array(xyz_meas, dtype=np.float64).flatten()[:3]
        H = np.eye(3, dtype=np.float64)
        innov = z - H @ self._x  # type: ignore[operator]
        innov_norm = float(np.linalg.norm(innov))

        if physical_max_m is not None and innov_norm > float(physical_max_m):
            return KFUpdateResult(
                False, reason='physical', innov_norm=innov_norm)

        R = np.eye(3, dtype=np.float64) * sigma_meas ** 2
        S = H @ self._P @ H.T + R  # type: ignore[operator]
        try:
            S_inv = np.linalg.inv(S)
            mahal_sq = float(innov.T @ S_inv @ innov)
        except np.linalg.LinAlgError:
            return KFUpdateResult(
                False, reason='singular_S', innov_norm=innov_norm)

        if mahal_sq > float(chi2_threshold):
            return KFUpdateResult(
                False,
                reason='mahalanobis',
                mahal_sq=mahal_sq,
                innov_norm=innov_norm,
            )

        K = self._P @ H.T @ S_inv  # type: ignore[operator]
        self._x = self._x + K @ innov  # type: ignore[operator]
        self._P = (np.eye(3) - K @ H) @ self._P  # type: ignore[operator]
        return KFUpdateResult(
            True,
            reason='ok',
            mahal_sq=mahal_sq,
            innov_norm=innov_norm,
        )

    def update_triangulated(self, xyz_meas: np.ndarray,
                            sigma_meas: float = 0.003) -> None:
        """Higher-precision update from active lateral-probe triangulation."""
        self.update(xyz_meas, sigma_meas)

    def gated_update_triangulated(
        self,
        xyz_meas: np.ndarray,
        sigma_meas: float = 0.003,
        *,
        physical_max_m: Optional[float] = None,
    ) -> KFUpdateResult:
        """Triangulation update — tighter default sigma, still gated."""
        return self.gated_update(
            xyz_meas,
            sigma_meas,
            physical_max_m=physical_max_m,
            chi2_threshold=CHI2_3_DOF_99,
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def uncertainty(self) -> float:
        """trace(P) — scalar measure of total positional uncertainty (m²)."""
        if self._P is None:
            return float('inf')
        return float(np.trace(self._P))

    def needs_active_probe(self,
                           threshold: float = DEFAULT_PROBE_THRESHOLD) -> bool:
        """Return True when KF is still too uncertain for reliable PBVS."""
        return self.uncertainty() > threshold

    @property
    def position(self) -> Optional[np.ndarray]:
        """Current position estimate in base_link, or None if not initialised."""
        return self._x.copy() if self._x is not None else None

    @property
    def position_covariance(self) -> Optional[np.ndarray]:
        return self._P.copy() if self._P is not None else None

    def sigma_xyz(self) -> Optional[np.ndarray]:
        """Per-axis 1-sigma (m) as (3,) array."""
        if self._P is None:
            return None
        return np.sqrt(np.diag(self._P))


# ---------------------------------------------------------------------------
# Standalone unit test  (python berry_kf_tracker.py)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    print('BerryKFTracker unit test')

    kf = BerryKFTracker()
    assert not kf.initialized

    TRUE_POS = np.array([0.30, 0.05, 0.25])
    rng = np.random.default_rng(0)

    # Simulate 30 noisy observations from two depth sources
    for i in range(30):
        noise = 0.015 if i < 20 else 0.005   # mono then rgbd
        obs = TRUE_POS + rng.normal(0, noise, 3)
        dt = 0.05
        kf.predict(dt)
        kf.update(obs, sigma_meas=noise)

    pos = kf.position
    err = np.linalg.norm(pos - TRUE_POS)
    print(f'  Final position estimate: {pos}')
    print(f'  True position:           {TRUE_POS}')
    print(f'  Error: {err*1000:.1f} mm   uncertainty: {kf.uncertainty()*1e6:.1f} µm²')
    assert err < 0.010, f'KF did not converge: error={err:.4f} m'

    # Active-probe trigger
    kf2 = BerryKFTracker()
    kf2.initialize(TRUE_POS, sigma_init=0.10)
    assert kf2.needs_active_probe(DEFAULT_PROBE_THRESHOLD)
    kf2.update_triangulated(TRUE_POS, sigma_meas=0.003)
    assert not kf2.needs_active_probe(DEFAULT_PROBE_THRESHOLD)

    # Gating rejects a single wild outlier (simulate PBVS drift)
    kf3 = BerryKFTracker()
    kf3.initialize(TRUE_POS, sigma_init=0.03)
    for _ in range(10):
        kf3.predict(0.04)
        kf3.gated_update(TRUE_POS + rng.normal(0, 0.008, 3), 0.008)
    before = kf3.position.copy()
    outlier = TRUE_POS + np.array([0.45, 0.35, -0.20])
    kf3.predict(0.04)
    gate = kf3.gated_update(outlier, 0.025, physical_max_m=0.10)
    assert not gate.accepted, f'outlier should be rejected: {gate}'
    assert np.linalg.norm(kf3.position - before) < 1e-9
    print(f'  Gating rejected outlier: reason={gate.reason} '
          f'mahal={gate.mahal_sq:.1f} |innov|={gate.innov_norm:.3f}m')

    print('  All assertions passed.')
    sys.exit(0)
