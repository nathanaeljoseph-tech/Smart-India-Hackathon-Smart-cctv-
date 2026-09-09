"""
data_exporter.py - Detection & Tracking Data Export Module (DEV 1)
===================================================================
Smart India Hackathon 2026 | Problem: SIH26187
Team Role: DEV 1 - Camera + YOLO Detection + Tracking

This module handles all data output:
  - Formats detection + tracking data into a structured dict
  - Buffers frames and saves to JSON (per-file or batched)
  - Provides a get_current_data() method for other modules (DEV 2, 3, etc.)
    to consume live data without reading files

WHY structured export?
  Other team members (DEV 2 = Event Detection, DEV 3 = Alerts) need
  the detection data in a predictable format so they can build their
  modules independently and integrate cleanly.

Output format per frame (Tier-1 enriched):
  {
    "frame_id"  : 123,
    "timestamp" : "2026-09-05 10:30:45.123",
    "detections": [
      {
        "track_id"    : 1,
        "bbox"        : [x1, y1, x2, y2],
        "centroid"    : [cx, cy],
        "ema_centroid": [ecx, ecy],
        "confidence"  : 0.89,
        "class"       : "person",
        "track_age"   : 24,
        "is_too_close": false,
        "alert_type"  : "loitering",
        "severity"    : "medium",
        "zone_id"     : "perimeter",
        "behavior"    : "loiter",
        "risk_score"  : 62.0,
        "alert_level" : "HIGH",
        "reasoning"   : "Loitering in restricted zone 'perimeter' for 14s",
        "dwell_sec"   : 14.2
      }
    ]
  }

Author : DEV 1
Date   : 2026-09-05
"""

import json
import logging
import os
from datetime import datetime
from typing import List, Dict, Any, Optional


class DataExporter:
    """
    Collects, formats, and exports detection + tracking data.

    Usage:
        exporter = DataExporter(output_dir="output", batch_size=30)
        # Inside main loop:
        exporter.add_frame(frame_id, detections, tracks)
        # At end:
        exporter.flush()   # Save any remaining buffered data

    Attributes:
        output_dir   (str)  : Directory where JSON files are saved.
        batch_size   (int)  : Number of frames to buffer before saving.
        save_json    (bool) : Whether to save JSON files at all.
        buffer       (list) : In-memory buffer of frame data dicts.
        current_data (dict) : The most recent frame's data (for live consumption).
        file_counter (int)  : Counter to make output filenames unique.
    """

    def __init__(
        self,
        output_dir: str = "output",
        batch_size: int = 30,
        save_json: bool = True
    ):
        """
        Args:
            output_dir : Directory to save JSON output files.
                         Created automatically if it doesn't exist.
            batch_size : Save to JSON after this many frames.
                         Larger = fewer disk writes = faster.
                         Smaller = less data lost if the app crashes.
            save_json  : If False, data is only kept in memory (for demos
                         where disk I/O would slow things down).
        """
        self.output_dir   = output_dir
        self.batch_size   = batch_size
        self.save_json    = save_json
        self.buffer       : List[Dict[str, Any]] = []
        self.current_data : Optional[Dict[str, Any]] = None
        self.file_counter = 0

        self.logger = logging.getLogger("amst_border_net.exporter")

        # Create output directory if it doesn't exist
        if self.save_json:
            os.makedirs(self.output_dir, exist_ok=True)
            self.logger.info(
                f"DataExporter ready | output_dir={output_dir} | "
                f"batch_size={batch_size}"
            )
        else:
            self.logger.info(
                "DataExporter ready (JSON saving DISABLED, in-memory only)."
            )

    def format_data(
        self,
        frame_id: int,
        detections: List[Dict[str, Any]],
        tracks: List[Dict[str, Any]],
        alert_cards: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Build the standard structured data dict for a single frame.

        WHY merge detections, tracks, and alert cards?
            Detections give us : bbox, confidence, class, is_too_close
            Tracks give us     : track_id, ema_centroid, track_age
            Alert cards give us: behavior, risk_score, alert_level, reasoning,
                                 dwell_sec, alert_type, severity, zone_id

            Merging them puts all fields in one record ΓÇö the canonical format
            that DEV 2 / DEV 3 consume downstream.

        Args:
            frame_id    : Sequential frame number (from main loop counter).
            detections  : List of detection dicts from YOLODetector.
            tracks      : List of track dicts from FallbackIOUTracker.
            alert_cards : Optional dict {track_id: AlertCard} from RiskEngine.
                          If None, alert card fields default to safe values.

        Returns:
            Structured frame dict in the standard format.
        """
        # Get current timestamp with milliseconds
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # Build a lookup: track_id ΓåÆ track info (for quick access)
        track_lookup: Dict[int, Dict[str, Any]] = {
            t["track_id"]: t for t in tracks
        }
        cards = alert_cards or {}

        # Build the detections list by merging detection + track + alert info
        formatted_detections = []

        if tracks:
            # PRIMARY: Use tracks as the source of truth (they have IDs)
            for t in tracks:
                x1, y1, x2, y2 = t["bbox"]
                cx = int((x1 + x2) / 2)   # Centroid X
                cy = int((y1 + y2) / 2)   # Centroid Y

                # EMA centroid from tracker (smoother than raw bbox centre)
                ema_c = t.get("ema_centroid", [cx, cy])

                # Pull alert card for this track (default to safe/empty values)
                card = cards.get(t["track_id"], {})

                record = {
                    # --- Core tracking fields ---
                    "track_id"    : t["track_id"],
                    "bbox"        : [x1, y1, x2, y2],
                    "centroid"    : [cx, cy],
                    "ema_centroid": ema_c,
                    "confidence"  : t.get("confidence", 0.0),
                    "class"       : t.get("class", "person"),
                    "track_age"   : t.get("track_age",  0),
                    "is_too_close": t.get("is_too_close", False),
                    # --- Boundary Engine fields ---
                    "alert_type"  : card.get("alert_type",  t.get("alert_type",  "none")),
                    "severity"    : card.get("severity",    t.get("severity",    "none")),
                    "zone_id"     : card.get("zone_id",     t.get("zone_id",      None)),
                    # --- Risk Engine / Tier-1 enrichment fields ---
                    "behavior"    : card.get("behavior",    "none"),
                    "risk_score"  : card.get("risk_score",  0.0),
                    "alert_level" : card.get("alert_level", "NORMAL"),
                    "reasoning"   : card.get("reasoning",  "Monitoring ΓÇö no immediate threat"),
                    "dwell_sec"   : card.get("dwell_sec",   0.0),
                }
                formatted_detections.append(record)

        else:
            # FALLBACK: No tracking data, just use raw detections
            # Track ID is set to -1 to indicate "no tracking"
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                record = {
                    "track_id"    : -1,       # -1 = tracking not available
                    "bbox"        : [x1, y1, x2, y2],
                    "centroid"    : [cx, cy],
                    "ema_centroid": [cx, cy],
                    "confidence"  : det.get("confidence", 0.0),
                    "class"       : det.get("class", "person"),
                    "track_age"   : 0,
                    "is_too_close": det.get("is_too_close", False),
                    "alert_type"  : "none",
                    "severity"    : "none",
                    "zone_id"     : None,
                    "behavior"    : "none",
                    "risk_score"  : 0.0,
                    "alert_level" : "NORMAL",
                    "reasoning"   : "Monitoring ΓÇö no immediate threat",
                    "dwell_sec"   : 0.0,
                }
                formatted_detections.append(record)

        frame_data = {
            "frame_id"  : frame_id,
            "timestamp" : timestamp,
            "detections": formatted_detections,
        }

        return frame_data

    def add_frame(
        self,
        frame_id: int,
        detections: List[Dict[str, Any]],
        tracks: List[Dict[str, Any]],
        alert_cards: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Format and buffer data for one frame. Saves to disk if buffer is full.

        This is the main method called in the main loop on every processed frame.

        Args:
            frame_id    : Sequential frame number.
            detections  : Detection dicts from YOLODetector.
            tracks      : Track dicts from FallbackIOUTracker.
            alert_cards : Optional {track_id: AlertCard} from RiskEngine.

        Returns:
            The formatted frame data dict (also available via get_current_data()).
        """
        # Format the data for this frame
        frame_data = self.format_data(frame_id, detections, tracks, alert_cards)

        # Store as the most recent data (for live access by other modules)
        self.current_data = frame_data

        # Add to buffer
        self.buffer.append(frame_data)

        # Log at debug level (only visible with --verbose)
        self.logger.debug(
            f"Frame {frame_id}: {len(frame_data['detections'])} person(s) tracked."
        )

        # Save to disk if buffer is full
        if self.save_json and len(self.buffer) >= self.batch_size:
            self.flush()

        return frame_data

    def get_current_data(self) -> Optional[Dict[str, Any]]:
        """
        Return the most recently formatted frame data.

        WHY: DEV 2 (Event Detection) can call this to get live detection
        data without reading from disk. This is the "shared memory"
        interface between modules.

        Returns:
            The most recent frame data dict, or None if no frames processed.
        """
        return self.current_data

    def flush(self) -> Optional[str]:
        """
        Save all buffered frame data to a JSON file on disk.

        WHY batched saving (not per-frame)?
            Disk I/O is slow. Writing per-frame at 15 FPS means 15 file
            writes per second, which adds latency. Batching every 30 frames
            reduces disk writes to ~0.5/second.

        Returns:
            Path to the saved JSON file, or None if nothing was saved.
        """
        if not self.buffer or not self.save_json:
            return None

        # Generate a timestamped filename to avoid overwriting
        self.file_counter += 1
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detections_{timestamp_str}_batch{self.file_counter:04d}.json"
        filepath = os.path.join(self.output_dir, filename)

        # Write the buffer as a JSON array
        output = {
            "metadata": {
                "project"    : "AMST Border-Net",
                "hackathon"  : "SIH 2026",
                "problem_id" : "SIH26187",
                "batch_num"  : self.file_counter,
                "frame_count": len(self.buffer),
                "saved_at"   : datetime.now().isoformat(),
            },
            "frames": self.buffer
        }

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)

            self.logger.info(
                f"Saved {len(self.buffer)} frames to: {filepath}"
            )

            # Clear buffer after successful save
            self.buffer.clear()

            return filepath

        except IOError as e:
            self.logger.error(f"Failed to save JSON: {e}")
            return None

    def save_to_json(self, filepath: str) -> bool:
        """
        Save current buffer to a specific file path (one-shot).

        WHY: Useful for end-of-session save (called when user presses 'q').
        Also used by other team members to trigger a save manually.

        Args:
            filepath: Absolute or relative path to save the JSON file.

        Returns:
            True if saved successfully, False on error.
        """
        if not self.buffer:
            self.logger.warning("No data in buffer to save.")
            return False

        output = {
            "metadata": {
                "project"    : "AMST Border-Net",
                "hackathon"  : "SIH 2026",
                "problem_id" : "SIH26187",
                "frame_count": len(self.buffer),
                "saved_at"   : datetime.now().isoformat(),
            },
            "frames": self.buffer
        }

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Session data saved to: {filepath}")
            return True
        except IOError as e:
            self.logger.error(f"save_to_json failed: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Return summary statistics for the current session.

        Useful for showing a "session summary" when the user quits.

        Returns:
            Dict with total_frames, total_detections, unique_track_ids.
        """
        total_frames     = self.file_counter * self.batch_size + len(self.buffer)
        total_detections = 0
        unique_ids       = set()

        for frame in self.buffer:
            total_detections += len(frame["detections"])
            for det in frame["detections"]:
                if det["track_id"] != -1:
                    unique_ids.add(det["track_id"])

        return {
            "total_frames"    : total_frames,
            "buffered_frames" : len(self.buffer),
            "total_detections": total_detections,
            "unique_track_ids": len(unique_ids),
            "saved_batches"   : self.file_counter,
        }
