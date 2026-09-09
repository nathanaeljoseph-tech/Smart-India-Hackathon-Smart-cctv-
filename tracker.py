"""
tracker.py - Lightweight Person Tracking Module (DEV 1)
========================================================
Smart India Hackathon 2026 | Problem: SIH26187
Team Role: DEV 1 - Camera + YOLO Detection + Tracking

This module provides:
  - ByteTrackWrapper class : Interface to ByteTrack via Ultralytics
  - FallbackIOUTracker     : Backup tracker (pure Python, no extra deps)

Tracker hierarchy:
  1. PREFERRED  : ByteTrack via Ultralytics built-in (most accurate)
  2. FALLBACK   : Our own IOU-based tracker (works offline, simpler)

WHY two trackers?
  Hackathon environments can have limited internet/install time.
  The fallback ensures the system ALWAYS runs, even if ByteTrack
  installation fails.

How tracking works (briefly):
  - Detection gives us boxes per frame (no memory of previous frames)
  - Tracking assigns consistent IDs across frames by matching boxes
    between the current and previous frame using IoU overlap
  - ByteTrack is state-of-the-art: it uses Kalman filtering to predict
    where a person SHOULD be even if briefly occluded (hidden)

Author : DEV 1
Date   : 2026-09-05
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional

# ---------------------------------------------------------------------------
# EMA smoothing factor for centroid trajectories
# ---------------------------------------------------------------------------
# alpha = weight given to the NEW centroid measurement each frame.
# (1 - alpha) = weight retained from the historical EMA.
#
# Tuning guide:
#   alpha = 0.3  ΓåÆ very smooth, 3-4 frames to fully settle (less jitter)
#   alpha = 0.4  ΓåÆ balanced, ~2-3 frames to settle  ΓåÉ DEFAULT
#   alpha = 0.6  ΓåÆ fast response, minimal smoothing
#
# Close-range people (large boxes) cause rapid centroid jumps ΓåÆ use lower alpha.
# The RiskEngine uses the smoothed centroid for behaviour classification.
_EMA_ALPHA: float = 0.40


# ---------------------------------------------------------------------------
# ByteTrack Wrapper (Primary - via Ultralytics)
# ---------------------------------------------------------------------------

class ByteTrackWrapper:
    """
    Wraps ByteTrack tracker provided by Ultralytics.

    WHY Ultralytics ByteTrack?
        Ultralytics YOLO has ByteTrack built-in as a tracker option.
        This means no separate installation needed for ByteTrack - it
        comes with the ultralytics package. Clean and simple.

    ByteTrack algorithm summary:
        1. Each detection is a "high confidence" or "low confidence" box
        2. High confidence boxes are matched to existing tracks via IoU
        3. Low confidence boxes are used to rescue "lost" tracks
        4. New tracks are created for unmatched high-confidence boxes
        5. Kalman filter predicts position of each track between frames

    Track lifecycle:
        new ΓåÆ confirmed ΓåÆ lost ΓåÆ removed
        A track becomes confirmed after N consecutive detections.
        A lost track is kept for M frames before being removed.

    Attributes:
        tracker_type (str)    : "bytetrack" or "botsort" (both built into ultralytics)
        track_history (dict)  : Stores bbox history per track_id for analysis
        logger                : Logger instance
    """

    def __init__(self, tracker_type: str = "bytetrack", max_lost_frames: int = 30):
        """
        Args:
            tracker_type    : "bytetrack" (recommended) or "botsort"
            max_lost_frames : Frames to keep a track alive when person
                              is not detected (handles brief occlusion).
        """
        self.tracker_type    = tracker_type
        self.max_lost_frames = max_lost_frames
        self.logger          = logging.getLogger("amst_border_net.tracker")

        # Store recent bboxes per track for analysis by other modules
        # Format: {track_id: [bbox1, bbox2, ...]} (last 30 bboxes)
        self.track_history: Dict[int, List[List[int]]] = {}

        self.logger.info(
            f"ByteTrackWrapper initialized | type={tracker_type} | "
            f"max_lost={max_lost_frames}"
        )

    def update(
        self,
        detections: List[Dict[str, Any]],
        frame: np.ndarray,
        yolo_results
    ) -> List[Dict[str, Any]]:
        """
        Update tracker with new detections from the current frame.

        WHY yolo_results is passed directly?
            Ultralytics ByteTrack works directly on YOLO Results objects
            (it calls result.boxes internally). This avoids re-parsing.

        Args:
            detections  : List of detection dicts from YOLODetector.
                          (Used only if yolo_results is None - fallback mode)
            frame       : Current BGR frame (needed by some trackers).
            yolo_results: Raw YOLO Results object from model inference.

        Returns:
            List of track dicts:
            [
                {
                    "track_id"  : int,
                    "bbox"      : [x1, y1, x2, y2],
                    "confidence": float,
                    "class"     : "person"
                },
                ...
            ]
        """
        tracks = []

        # Use Ultralytics' built-in tracking on the YOLO results object
        # This modifies the Results object in-place to add track IDs
        try:
            if yolo_results is not None and len(yolo_results) > 0:
                result = yolo_results[0]   # First (only) image result

                # Check if result has boxes with tracking IDs
                if (result.boxes is not None
                        and result.boxes.id is not None
                        and len(result.boxes.id) > 0):

                    # Extract tracked boxes
                    for box in result.boxes:
                        # box.id contains the track ID (assigned by tracker)
                        if box.id is None:
                            continue

                        track_id = int(box.id[0])
                        class_id = int(box.cls[0])

                        # Only keep person class
                        if class_id != 0:
                            continue

                        confidence = float(box.conf[0])
                        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]

                        track = {
                            "track_id"  : track_id,
                            "bbox"      : [x1, y1, x2, y2],
                            "confidence": round(confidence, 4),
                            "class"     : "person",
                        }
                        tracks.append(track)

                        # Update history for this track
                        self._update_history(track_id, [x1, y1, x2, y2])

        except Exception as e:
            self.logger.warning(
                f"ByteTrack update failed: {e}. "
                "Detections will be returned without track IDs."
            )

        self.logger.debug(f"Tracker updated: {len(tracks)} active track(s).")
        return tracks

    def _update_history(self, track_id: int, bbox: List[int]) -> None:
        """
        Record the bbox for a track, keeping last 30 positions.
        Useful for visualizing movement trails (future enhancement).
        """
        if track_id not in self.track_history:
            self.track_history[track_id] = []

        self.track_history[track_id].append(bbox)

        # Keep only the last 30 positions
        if len(self.track_history[track_id]) > 30:
            self.track_history[track_id].pop(0)

    def get_tracks(self) -> Dict[int, List[List[int]]]:
        """
        Return the full track history for all active tracks.

        Returns:
            Dict mapping track_id ΓåÆ list of recent bboxes.
        """
        return self.track_history

    def reset(self) -> None:
        """Clear all track history. Useful when restarting the stream."""
        self.track_history.clear()
        self.logger.info("Track history cleared.")


# ---------------------------------------------------------------------------
# Fallback IOU Tracker (Pure Python - no extra dependencies)
# ---------------------------------------------------------------------------

class FallbackIOUTracker:
    """
    A simple IOU-based tracker that works without any extra libraries.

    WHY this exists:
        If ByteTrack fails (import error, version mismatch), this
        tracker ensures the system still assigns consistent IDs.

    Algorithm:
        1. For each detection in the current frame, compute IoU with
           all tracks from the previous frame.
        2. Match using a greedy algorithm (highest IoU first).
        3. Matched detections keep the same track ID.
        4. Unmatched detections get new IDs.
        5. Tracks not matched for `max_lost_frames` frames are deleted.

    Limitations vs ByteTrack:
        - No Kalman filtering (no prediction during occlusion)
        - More ID switches when people cross paths
        - Simpler, but sufficient for hackathon demo

    Attributes:
        next_id (int)          : Next available track ID (auto-increments)
        active_tracks (dict)   : Current tracks: {id: {bbox, lost_frames, ...}}
        iou_threshold (float)  : Minimum overlap to consider a match
        max_lost_frames (int)  : Frames before track is permanently removed
    """

    def __init__(self, iou_threshold: float = 0.25, max_lost_frames: int = 25):
        """
        Args:
            iou_threshold   : Minimum IoU to match a detection to a track.
                              Default 0.25 (tuned down from 0.30 for large,
                              close-range bounding boxes that shift significantly
                              between frames ΓÇö lower threshold = more matches).
            max_lost_frames : Frames to keep a track without detection.
                              Default 25 (tuned up from 15 to keep track IDs
                              alive longer during brief occlusion by large boxes).
        """
        self.next_id         = 1                 # Track IDs start at 1
        self.active_tracks   = {}                # id ΓåÆ track_info dict
        self.iou_threshold   = iou_threshold
        self.max_lost_frames = max_lost_frames
        self.logger          = logging.getLogger("amst_border_net.tracker.fallback")

        self.logger.info(
            f"FallbackIOUTracker initialized | iou_thresh={iou_threshold} | "
            f"max_lost={max_lost_frames}"
        )

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Update tracks with new detections and return current tracks.

        Args:
            detections: List of detection dicts from YOLODetector.

        Returns:
            List of track dicts with track_id assigned.
        """
        if not detections:
            # No detections: age all existing tracks
            self._age_tracks()
            return []

        # ---- Step 1: Compute IoU matrix ----
        # Rows = active tracks, Cols = new detections
        track_ids   = list(self.active_tracks.keys())
        track_bboxes = [self.active_tracks[tid]["bbox"] for tid in track_ids]
        det_bboxes  = [d["bbox"] for d in detections]

        # Build IoU matrix: iou_matrix[i][j] = IoU(track_i, detection_j)
        iou_matrix = self._compute_iou_matrix(track_bboxes, det_bboxes)

        # ---- Step 2: Greedy matching ----
        matched_track_ids, matched_det_idxs = set(), set()
        assignments = {}  # track_idx ΓåÆ det_idx

        if iou_matrix.size > 0:
            # Process matches in descending IoU order (best first)
            # np.argmax on flattened matrix gives us best pair
            flat_indices = np.argsort(iou_matrix.flatten())[::-1]

            for flat_idx in flat_indices:
                t_idx = flat_idx // len(det_bboxes)
                d_idx = flat_idx % len(det_bboxes)

                # Stop if IoU is too low (no more valid matches)
                if iou_matrix[t_idx, d_idx] < self.iou_threshold:
                    break

                # Only match each track and detection ONCE
                if t_idx not in matched_track_ids and d_idx not in matched_det_idxs:
                    assignments[t_idx] = d_idx
                    matched_track_ids.add(t_idx)
                    matched_det_idxs.add(d_idx)

        # ---- Step 3: Update matched tracks ----
        tracks_output = []

        for t_idx, d_idx in assignments.items():
            tid = track_ids[t_idx]
            det = detections[d_idx]

            # Update EMA centroid for this matched track
            raw_cx = int((det["bbox"][0] + det["bbox"][2]) / 2)
            raw_cy = int((det["bbox"][1] + det["bbox"][3]) / 2)
            prev_ema_cx = self.active_tracks[tid].get("ema_cx", float(raw_cx))
            prev_ema_cy = self.active_tracks[tid].get("ema_cy", float(raw_cy))
            new_ema_cx  = _EMA_ALPHA * raw_cx + (1.0 - _EMA_ALPHA) * prev_ema_cx
            new_ema_cy  = _EMA_ALPHA * raw_cy + (1.0 - _EMA_ALPHA) * prev_ema_cy

            self.active_tracks[tid]["bbox"]        = det["bbox"]
            self.active_tracks[tid]["confidence"]  = det["confidence"]
            self.active_tracks[tid]["lost_frames"] = 0  # Reset lost counter
            self.active_tracks[tid]["ema_cx"]      = new_ema_cx
            self.active_tracks[tid]["ema_cy"]      = new_ema_cy
            self.active_tracks[tid]["track_age"]   = \
                self.active_tracks[tid].get("track_age", 0) + 1

            tracks_output.append({
                "track_id"     : tid,
                "bbox"         : det["bbox"],
                "confidence"   : det["confidence"],
                "class"        : "person",
                "track_age"    : self.active_tracks[tid]["track_age"],
                "ema_centroid" : [int(round(new_ema_cx)), int(round(new_ema_cy))],
            })

        # ---- Step 4: Create new tracks for unmatched detections ----
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_det_idxs:
                # This detection has no matching track ΓåÆ new person
                new_id = self.next_id
                self.next_id += 1
                raw_cx = int((det["bbox"][0] + det["bbox"][2]) / 2)
                raw_cy = int((det["bbox"][1] + det["bbox"][3]) / 2)
                self.active_tracks[new_id] = {
                    "bbox"        : det["bbox"],
                    "confidence"  : det["confidence"],
                    "lost_frames" : 0,
                    "track_age"   : 1,
                    "ema_cx"      : float(raw_cx),
                    "ema_cy"      : float(raw_cy),
                }
                tracks_output.append({
                    "track_id"     : new_id,
                    "bbox"         : det["bbox"],
                    "confidence"   : det["confidence"],
                    "class"        : "person",
                    "track_age"    : 1,
                    "ema_centroid" : [raw_cx, raw_cy],
                })

        # ---- Step 5: Age unmatched tracks ----
        for t_idx, tid in enumerate(track_ids):
            if t_idx not in matched_track_ids:
                self.active_tracks[tid]["lost_frames"] += 1

        # ---- Step 6: Remove tracks lost too long ----
        to_remove = [
            tid for tid, t in self.active_tracks.items()
            if t["lost_frames"] > self.max_lost_frames
        ]
        for tid in to_remove:
            del self.active_tracks[tid]
            self.logger.debug(f"Track {tid} removed (lost for too long).")

        self.logger.debug(
            f"FallbackTracker: {len(tracks_output)} active tracks, "
            f"{len(to_remove)} removed."
        )
        return tracks_output

    def _age_tracks(self) -> None:
        """Increment lost_frames for all tracks when no detections are present."""
        for tid in list(self.active_tracks.keys()):
            self.active_tracks[tid]["lost_frames"] += 1
            if self.active_tracks[tid]["lost_frames"] > self.max_lost_frames:
                del self.active_tracks[tid]

    def _compute_iou_matrix(
        self,
        tracks: List[List[int]],
        dets: List[List[int]]
    ) -> np.ndarray:
        """
        Compute pairwise IoU between all track bboxes and all detection bboxes.

        Returns:
            numpy array of shape (len(tracks), len(dets))
        """
        if not tracks or not dets:
            return np.zeros((len(tracks), len(dets)))

        matrix = np.zeros((len(tracks), len(dets)), dtype=np.float32)
        for i, t_box in enumerate(tracks):
            for j, d_box in enumerate(dets):
                matrix[i, j] = self._iou(t_box, d_box)
        return matrix

    @staticmethod
    def _iou(box1: List[int], box2: List[int]) -> float:
        """Compute IoU between two [x1,y1,x2,y2] boxes."""
        ix1 = max(box1[0], box2[0])
        iy1 = max(box1[1], box2[1])
        ix2 = min(box1[2], box2[2])
        iy2 = min(box1[3], box2[3])

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        inter  = (ix2 - ix1) * (iy2 - iy1)
        area1  = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2  = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union  = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def reset(self) -> None:
        """Reset tracker to initial state."""
        self.active_tracks.clear()
        self.next_id = 1
        self.logger.info("FallbackIOUTracker reset.")
