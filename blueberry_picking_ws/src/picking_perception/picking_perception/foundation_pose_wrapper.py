"""FoundationPose inference wrapper."""

from __future__ import annotations

import logging
import os
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class FoundationPoseWrapper:
    """Wrap FoundationPose register() for multi-berry detection."""

    SCORE_CKPT = 'weights/2024-01-11-20-02-45'
    REFINE_CKPT = 'weights/2023-10-28-18-33-37'

    def __init__(
        self,
        mesh_path: str,
        foundation_pose_root: str,
        est_refine_iter: int = 5,
        track_refine_iter: int = 2,
        score_threshold: float = 0.3,
    ) -> None:
        self.est_refine_iter = est_refine_iter
        self.track_refine_iter = track_refine_iter
        self.score_threshold = score_threshold
        self._ready = False
        self._fp_root = foundation_pose_root
        self._debug_dir = os.path.join(foundation_pose_root, 'debug', 'novel_pose_debug')
        os.makedirs(self._debug_dir, exist_ok=True)

        if foundation_pose_root not in os.sys.path:
            os.sys.path.insert(0, foundation_pose_root)

        try:
            import trimesh  # noqa: F401
            import cv2  # noqa: F401
            self.mesh = trimesh.load(mesh_path)
            self._init_estimators()
            self._ready = True
            logger.info('FoundationPoseWrapper initialized with mesh: %s', mesh_path)
        except Exception as exc:  # pragma: no cover - GPU/weights may be missing
            logger.warning('FoundationPose unavailable: %s', exc)
            self._ready = False

    def _init_estimators(self) -> None:
        import nvdiffrast.torch as dr
        from estimater import FoundationPose
        from learning.training.predict_pose_refine import PoseRefinePredictor
        from learning.training.predict_score import ScorePredictor

        self._FoundationPose = FoundationPose
        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()

    @property
    def ready(self) -> bool:
        return self._ready

    def _make_estimator(self):
        return self._FoundationPose(
            model_pts=np.array(self.mesh.vertices),
            model_normals=np.array(self.mesh.vertex_normals),
            mesh=self.mesh,
            scorer=self.scorer,
            refiner=self.refiner,
            glctx=self.glctx,
            debug=0,
            debug_dir=self._debug_dir,
        )

    def detect_all(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        k: np.ndarray,
        mask: np.ndarray,
    ) -> List[Tuple[np.ndarray, float]]:
        if not self._ready:
            return []

        import cv2

        num_labels, labels = cv2.connectedComponents(mask)
        results: List[Tuple[np.ndarray, float]] = []

        for label_id in range(1, num_labels):
            component_mask = (labels == label_id).astype(np.uint8)
            if component_mask.sum() < 200:
                continue
            est = self._make_estimator()
            try:
                pose = est.register(
                    K=k,
                    rgb=rgb,
                    depth=depth,
                    ob_mask=component_mask,
                    iteration=self.est_refine_iter,
                )
                score = float(getattr(est, 'pose_last_score', 0.5))
                if score >= self.score_threshold:
                    results.append((pose, score))
            except Exception as exc:
                logger.debug('register failed for label %d: %s', label_id, exc)
        return results
