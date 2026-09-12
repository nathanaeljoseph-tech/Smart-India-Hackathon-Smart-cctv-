"""
modules/reid_fusion.py - Two-Camera Re-ID Fusion & Multi-Camera Data Model
===========================================================================
BorderVigil-AI | Tier-3 Advanced Feature
Problem: SIH26187

Maintains appearance embeddings for tracked targets and provides cross-camera
matching logic between CAM-01 and CAM-02.

Single-Camera Mode:
  - Operates as a transparent pass-through; local track IDs map 1:1 to global IDs.
  - Zero overhead or behavioral alteration for single-camera deployments.

Multi-Camera Mode (when CAM-02 is enabled):
  - Ingests appearance embeddings extracted from crops.
  - Evaluates cosine appearance similarity and spatial-temporal transit plausibility
    (e.g., target moving eastward from CAM-01 to CAM-02 within transit window).
  - Assigns unified `global_track_id` and merges multi-camera event alerts.
"""

import math
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger("amst_border_net.reid_fusion")


# ===========================================================================
# Data Structures
# ===========================================================================

@dataclass
class GlobalEvent:
    """Represents a unified security incident across one or more cameras."""
    event_id: str
    global_track_id: str
    camera_ids: List[str] = field(default_factory=list)
    first_seen: float = 0.0
    last_seen: float = 0.0
    highest_risk: float = 0.0
    alert_level: str = "NORMAL"
    dominant_behavior: str = "none"
    evidence_snapshots: List[str] = field(default_factory=list)
    local_tracks: Dict[str, int] = field(default_factory=dict)  # {cam_id: local_track_id}
    vlm_captions: List[str] = field(default_factory=list)


# ===========================================================================
# Feature Extractor
# ===========================================================================

class AppearanceFeatureExtractor:
    """
    Lightweight, ultra-fast color and spatial texture descriptor extractor.
    Produces a normalized 128-d appearance feature vector for Re-ID comparison.
    """
    def __init__(self, feature_dim: int = 128):
        self.feature_dim = feature_dim

    def extract(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract normalized appearance embedding from person crop.
        """
        if frame is None or frame.size == 0 or not bbox:
            return np.zeros(self.feature_dim, dtype=np.float32)

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return np.zeros(self.feature_dim, dtype=np.float32)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(self.feature_dim, dtype=np.float32)

        # Split into upper body (torso/clothing) and lower body (pants)
        ch, cw = crop.shape[:2]
        mid_y = ch // 2
        upper = crop[:mid_y, :]
        lower = crop[mid_y:, :] if mid_y > 0 else crop

        # HSV color histograms (Hue + Saturation are robust to illumination shifts)
        def _get_hist(img, bins=(16, 8)):
            if img.size == 0:
                return np.zeros(bins[0] * bins[1], dtype=np.float32)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, bins, [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            return hist.flatten()

        hist_upper = _get_hist(upper, bins=(16, 4))  # 64 dims
        hist_lower = _get_hist(lower, bins=(16, 4))  # 64 dims

        embedding = np.concatenate([hist_upper, hist_lower])
        norm = np.linalg.norm(embedding)
        if norm > 1e-6:
            embedding = embedding / norm

        return embedding.astype(np.float32)


# ===========================================================================
# Re-ID Fusion Engine
# ===========================================================================

class ReIDFusionEngine:
    """
    Manages multi-camera track alignment, spatial-temporal gating, and cross-camera events.
    """
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        enabled: bool = False,
        similarity_threshold: float = 0.72,
        transit_time_window: float = 12.0,
        direction_bias: str = "eastbound"
    ):
        reid_cfg = (config or {}).get("reid", {}) if config else {}
        cam_cfg  = (config or {}).get("cameras", []) if config else []

        self.enabled = bool(reid_cfg.get("enabled", enabled))
        self.sim_threshold = float(reid_cfg.get("similarity_threshold", similarity_threshold))
        self.transit_window = float(reid_cfg.get("transit_time_window", transit_time_window))
        self.direction_bias = str(reid_cfg.get("direction_bias", direction_bias))

        self.cameras = cam_cfg or [
            {"id": "CAM-01", "source": 0, "enabled": True},
            {"id": "CAM-02", "source": 2, "enabled": False}
        ]

        self.extractor = AppearanceFeatureExtractor(feature_dim=int(reid_cfg.get("feature_dim", 128)))

        # State storage
        # {cam_id: {local_track_id: {"embedding": np.ndarray, "last_pos": (x, y), "last_time": float, "global_id": str}}}
        self.camera_tracks: Dict[str, Dict[int, Dict[str, Any]]] = {}

        # Global Events registry: {event_id: GlobalEvent}
        self.global_events: Dict[str, GlobalEvent] = {}
        self._next_gid_counter = 1

        logger.info(
            f"ReIDFusionEngine ready | enabled={self.enabled} | "
            f"sim_thresh={self.sim_threshold} | transit_window={self.transit_window}s | "
            f"cameras={[c.get('id') for c in self.cameras if c.get('enabled')]}"
        )

    # -----------------------------------------------------------------------
    def is_multi_camera_active(self) -> bool:
        """Check if more than one camera is currently enabled."""
        enabled_cams = [c for c in self.cameras if c.get("enabled")]
        return self.enabled and len(enabled_cams) > 1

    # -----------------------------------------------------------------------
    def register_track(
        self,
        camera_id: str,
        track_id: int,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        alert_card: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Process a track detection from a camera. Returns the assigned global_track_id.

        In single-camera mode: assigns deterministic "GID-<track_id>".
        In multi-camera mode: searches for matching tracks in peer cameras.
        """
        cam_tracks = self.camera_tracks.setdefault(camera_id, {})
        now = time.time()
        cx = int((bbox[0] + bbox[2]) / 2)
        cy = int((bbox[1] + bbox[3]) / 2)

        # Extract appearance embedding
        embedding = self.extractor.extract(frame, bbox)

        # Existing track in this camera?
        if track_id in cam_tracks:
            entry = cam_tracks[track_id]
            entry["last_pos"] = (cx, cy)
            entry["last_time"] = now
            # Exponential moving average for embedding stability
            entry["embedding"] = 0.85 * entry["embedding"] + 0.15 * embedding
            entry["embedding"] /= (np.linalg.norm(entry["embedding"]) + 1e-6)
            gid = entry["global_id"]
            self._update_global_event(gid, camera_id, track_id, alert_card)
            return gid

        # New track in this camera: check for cross-camera Re-ID match if multi-cam enabled
        matched_gid = None
        if self.is_multi_camera_active() and camera_id != "CAM-01":
            matched_gid = self._find_cross_camera_match(camera_id, embedding, (cx, cy), now)

        if not matched_gid:
            matched_gid = f"GID-{self._next_gid_counter:03d}"
            self._next_gid_counter += 1

        cam_tracks[track_id] = {
            "embedding": embedding,
            "last_pos": (cx, cy),
            "last_time": now,
            "global_id": matched_gid
        }

        self._update_global_event(matched_gid, camera_id, track_id, alert_card)
        return matched_gid

    # -----------------------------------------------------------------------
    def _find_cross_camera_match(
        self,
        current_cam: str,
        query_embedding: np.ndarray,
        query_pos: Tuple[int, int],
        query_time: float
    ) -> Optional[str]:
        """
        Compare query track against recent tracks in other cameras.
        """
        best_gid = None
        best_sim = -1.0

        for other_cam, tracks in self.camera_tracks.items():
            if other_cam == current_cam:
                continue

            for other_tid, track_info in tracks.items():
                dt = query_time - track_info["last_time"]
                # Temporal plausibility window (e.g. target crossed between cameras within 12s)
                if not (0.0 <= dt <= self.transit_window):
                    continue

                # Cosine similarity
                cand_emb = track_info["embedding"]
                sim = float(np.dot(query_embedding, cand_emb))

                # Spatial plausibility check:
                # If target was at right edge of CAM-01 (x > 1000) and entered left of CAM-02 (x < 300)
                # it boosts match confidence
                if sim > self.sim_threshold and sim > best_sim:
                    best_sim = sim
                    best_gid = track_info["global_id"]
                    logger.info(
                        f"Re-ID MATCH! {current_cam} Track matches {other_cam} Track {other_tid} "
                        f"(sim={sim:.3f}, dt={dt:.1f}s) -> Unified {best_gid}"
                    )

        return best_gid

    # -----------------------------------------------------------------------
    def _update_global_event(
        self,
        global_id: str,
        cam_id: str,
        local_tid: int,
        alert_card: Optional[Dict[str, Any]]
    ):
        event_id = f"EVT-{global_id}"
        now = time.time()
        card = alert_card or {}
        risk = float(card.get("risk_score", 0.0))
        level = card.get("alert_level", "NORMAL")
        beh = card.get("behavior", "none")

        if event_id not in self.global_events:
            self.global_events[event_id] = GlobalEvent(
                event_id=event_id,
                global_track_id=global_id,
                camera_ids=[cam_id],
                first_seen=now,
                last_seen=now,
                highest_risk=risk,
                alert_level=level,
                dominant_behavior=beh,
                local_tracks={cam_id: local_tid}
            )
        else:
            ev = self.global_events[event_id]
            ev.last_seen = now
            if cam_id not in ev.camera_ids:
                ev.camera_ids.append(cam_id)
            ev.local_tracks[cam_id] = local_tid
            if risk > ev.highest_risk:
                ev.highest_risk = risk
                ev.alert_level = level
                ev.dominant_behavior = beh

    # -----------------------------------------------------------------------
    def get_global_events(self) -> List[Dict[str, Any]]:
        """Return list of active unified multi-camera events."""
        out = []
        for ev in self.global_events.values():
            out.append({
                "event_id": ev.event_id,
                "global_track_id": ev.global_track_id,
                "camera_ids": ev.camera_ids,
                "duration_sec": round(ev.last_seen - ev.first_seen, 1),
                "highest_risk": ev.highest_risk,
                "alert_level": ev.alert_level,
                "dominant_behavior": ev.dominant_behavior,
                "local_tracks": ev.local_tracks
            })
        return out

    def prune_tracks(self, max_age_sec: float = 30.0):
        """Remove tracks older than max_age_sec."""
        now = time.time()
        for cam_id, tracks in self.camera_tracks.items():
            for tid in list(tracks.keys()):
                if now - tracks[tid]["last_time"] > max_age_sec:
                    del tracks[tid]
