"""
modules/tamper_detector.py  -  Camera Tamper & Visibility Monitor
==================================================================
AMST Border-Net | SIH26187 | DEV 1

Detects four camera fault conditions in real-time:

  1. COVERED   - Camera lens physically blocked (hand, cloth, tape)
                 Signal: frame pixel std-dev drops below threshold.

  2. DARK/OFF  - Camera switched off, lens cap on, or complete darkness.
                 Signal: mean pixel brightness below threshold.

  3. FROZEN    - Feed is stuck / hardware hang (same frame repeating).
                 Signal: frame hash unchanged for N consecutive frames.

  4. FEED LOST - cap.read() returned ret=False (disconnected / end of stream).
                 Handled in main.py; draw_tamper_overlay() called directly.

Usage in main.py:
    tamper = CameraTamperDetector()

    # After cap.read():
    status = tamper.check(frame)          # returns TamperStatus
    if status.is_fault:
        draw_tamper_overlay(frame, status)
        cv2.imshow(window_name, frame)
        continue                          # skip YOLO + tracking this frame
"""

import cv2
import numpy as np
import logging
import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("amst_border_net.tamper_detector")

# ---------------------------------------------------------------------------
# Thresholds (tunable)
# ---------------------------------------------------------------------------
_COVERED_STD_THRESHOLD  = 8.0    # Grayscale std-dev below this => covered
_DARK_MEAN_THRESHOLD    = 12.0   # Mean brightness below this => dark/off
_FREEZE_FRAMES          = 8      # Hash must repeat this many times => frozen
_CONSEC_FRAMES_TO_ALERT = 4      # Faults must persist N frames before alerting
                                  # (prevents false alarms on single bad frames)

# ---------------------------------------------------------------------------
# Status dataclass
# ---------------------------------------------------------------------------
@dataclass
class TamperStatus:
    """Result returned by CameraTamperDetector.check() each frame."""
    is_fault   : bool   = False
    fault_type : str    = "ok"       # "ok" | "covered" | "dark" | "frozen" | "lost"
    message    : str    = ""         # Human-readable alert message
    severity   : str    = "ok"       # "ok" | "warning" | "critical"
    std_dev    : float  = 0.0
    mean_brightness: float = 0.0
    consec_count: int  = 0           # How many consecutive fault frames


# ---------------------------------------------------------------------------
# Detector Class
# ---------------------------------------------------------------------------
class CameraTamperDetector:
    """
    Lightweight, frame-by-frame camera tamper and occlusion detector.

    Call check(frame) once per frame. It returns a TamperStatus describing
    any detected fault. On clean frames it returns TamperStatus(is_fault=False).

    Design notes:
    - Uses grayscale analysis only (very fast, < 0.1ms per frame)
    - Requires N consecutive fault frames before raising an alert
      (prevents false positives on single motion blur / bright flashes)
    - Hash-based freeze detection uses MD5 on a 64x64 downscaled frame
      (fast enough to run every frame at 30 FPS)
    """

    def __init__(
        self,
        covered_std_threshold: float = _COVERED_STD_THRESHOLD,
        dark_mean_threshold   : float = _DARK_MEAN_THRESHOLD,
        freeze_frames         : int   = _FREEZE_FRAMES,
        consec_to_alert       : int   = _CONSEC_FRAMES_TO_ALERT,
    ):
        self.covered_std  = covered_std_threshold
        self.dark_mean    = dark_mean_threshold
        self.freeze_n     = freeze_frames
        self.consec_n     = consec_to_alert

        self._hash_history: list = []       # Rolling list of frame hashes
        self._consec_fault : int = 0        # Consecutive fault frame counter
        self._last_fault_type: str = "ok"
        self._last_logged_time: float = 0.0

        logger.info(
            f"CameraTamperDetector ready | "
            f"covered_std<{covered_std_threshold} | "
            f"dark_mean<{dark_mean_threshold} | "
            f"freeze_after={freeze_frames}frames | "
            f"alert_after={consec_to_alert}frames"
        )

    # ------------------------------------------------------------------
    def check(self, frame: np.ndarray) -> TamperStatus:
        """
        Analyse one frame and return a TamperStatus.

        Args:
            frame: BGR numpy array (current video frame).

        Returns:
            TamperStatus with is_fault=True if a problem is detected.
        """
        if frame is None or frame.size == 0:
            return self._make_status("lost", 0.0, 0.0)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        std  = float(np.std(gray))
        mean = float(np.mean(gray))

        # ---- 1. Covered (low texture, any brightness) ----
        if std < self.covered_std:
            # Distinguish covered (dark) vs covered (bright - white cloth/flash)
            if mean < self.dark_mean:
                return self._make_status("dark", std, mean)
            return self._make_status("covered", std, mean)

        # ---- 2. Dark / camera off ----
        if mean < self.dark_mean:
            return self._make_status("dark", std, mean)

        # ---- 3. Frozen feed ----
        frame_hash = self._hash_frame(gray)
        self._hash_history.append(frame_hash)
        if len(self._hash_history) > self.freeze_n + 2:
            self._hash_history.pop(0)

        if (len(self._hash_history) >= self.freeze_n and
                len(set(self._hash_history[-self.freeze_n:])) == 1):
            return self._make_status("frozen", std, mean)

        # ---- 4. All clear ----
        self._consec_fault = 0
        self._last_fault_type = "ok"
        return TamperStatus(is_fault=False, fault_type="ok",
                            std_dev=std, mean_brightness=mean)

    # ------------------------------------------------------------------
    def check_feed_lost(self) -> TamperStatus:
        """Call this when cap.read() returns ret=False."""
        return self._make_status("lost", 0.0, 0.0, force=True)

    # ------------------------------------------------------------------
    def reset(self):
        """Clear state ? call after reconnecting the camera."""
        self._hash_history.clear()
        self._consec_fault = 0
        self._last_fault_type = "ok"
        logger.info("CameraTamperDetector state reset.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_status(
        self, fault_type: str, std: float, mean: float, force: bool = False
    ) -> TamperStatus:
        """Increment consecutive counter; return status only after N frames."""
        if fault_type == self._last_fault_type:
            self._consec_fault += 1
        else:
            self._consec_fault = 1
            self._last_fault_type = fault_type

        # Only raise alert after N consecutive fault frames
        if not force and self._consec_fault < self.consec_n:
            return TamperStatus(is_fault=False, fault_type="ok",
                                std_dev=std, mean_brightness=mean,
                                consec_count=self._consec_fault)

        msg, severity = _FAULT_META.get(fault_type, ("Unknown camera fault.", "warning"))

        # Throttle repeated log messages (log once every 5 seconds)
        now = time.monotonic()
        if now - self._last_logged_time > 5.0:
            if severity == "critical":
                logger.error(f"CAMERA FAULT [{fault_type.upper()}] | {msg} "
                             f"| std={std:.1f} mean={mean:.1f} "
                             f"| consec={self._consec_fault}")
            else:
                logger.warning(f"CAMERA FAULT [{fault_type.upper()}] | {msg} "
                               f"| std={std:.1f} mean={mean:.1f} "
                               f"| consec={self._consec_fault}")
            self._last_logged_time = now

        return TamperStatus(
            is_fault      = True,
            fault_type    = fault_type,
            message       = msg,
            severity      = severity,
            std_dev       = std,
            mean_brightness= mean,
            consec_count  = self._consec_fault,
        )

    @staticmethod
    def _hash_frame(gray: np.ndarray) -> str:
        """Fast perceptual hash: downsample to 64x64 and MD5."""
        small = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
        return hashlib.md5(small.tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# Fault metadata
# ---------------------------------------------------------------------------
_FAULT_META = {
    "covered": (
        "CAMERA BLOCKED ? Lens covered or obstructed",
        "critical",
    ),
    "dark": (
        "NO VISIBILITY ? Camera OFF or lens cap on",
        "critical",
    ),
    "frozen": (
        "FEED FROZEN ? Camera hardware fault or stream stuck",
        "warning",
    ),
    "lost": (
        "FEED LOST ? Camera disconnected",
        "critical",
    ),
}


# ---------------------------------------------------------------------------
# Overlay Drawing
# ---------------------------------------------------------------------------

# Colour scheme per fault type  (BGR)
_OVERLAY_COLORS = {
    "covered" : (0,   0,   220),   # Deep red
    "dark"    : (30,  30,  180),   # Dark red-orange
    "frozen"  : (0,   140, 200),   # Amber
    "lost"    : (0,   0,   180),   # Red
}

_BLINK_RATE = 18   # frames; icon blinks every N frames


def draw_tamper_overlay(
    frame    : np.ndarray,
    status   : TamperStatus,
    frame_id : int = 0,
) -> None:
    """
    Draw a full-screen tamper alert overlay on the frame (in-place).

    Layout:
      - Semi-transparent dark red tint over entire frame
      - Thick red border
      - Blinking warning icon  ?
      - Large fault type line  (e.g. "CAMERA BLOCKED")
      - Smaller detail line    (e.g. "Lens covered or obstructed")
      - Diagnostic bar         (std / mean / consecutive frames)

    Args:
        frame    : BGR frame (modified in-place).
        status   : TamperStatus from CameraTamperDetector.check().
        frame_id : Used for blink animation.
    """
    h, w = frame.shape[:2]
    color = _OVERLAY_COLORS.get(status.fault_type, (0, 0, 200))

    # ---- 1. Semi-transparent tint ----
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (10, 10, 60), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # ---- 2. Thick border ----
    blink_on = (frame_id // _BLINK_RATE) % 2 == 0
    border_color = color if blink_on else (60, 60, 60)
    cv2.rectangle(frame, (4, 4), (w - 4, h - 4), border_color, 6)

    # ---- 3. Warning icon (blinks) ----
    icon = "  WARNING  " if blink_on else "           "
    cx   = w // 2
    cy   = h // 2

    font      = cv2.FONT_HERSHEY_DUPLEX
    font_icon = cv2.FONT_HERSHEY_SIMPLEX

    # Icon background pill
    (iw, ih), _ = cv2.getTextSize(icon, font, 1.2, 3)
    cv2.rectangle(frame,
                  (cx - iw // 2 - 16, cy - 100),
                  (cx + iw // 2 + 16, cy - 100 + ih + 20),
                  color if blink_on else (40, 40, 40), -1)
    cv2.putText(frame, icon,
                (cx - iw // 2, cy - 100 + ih + 4),
                font, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

    # ---- 4. Main alert line ----
    main_lines = {
        "covered" : "CAMERA BLOCKED",
        "dark"    : "NO VISIBILITY",
        "frozen"  : "FEED FROZEN",
        "lost"    : "FEED LOST",
    }
    main_text = main_lines.get(status.fault_type, "CAMERA FAULT")
    (mw, mh), _ = cv2.getTextSize(main_text, font, 2.0, 4)
    cv2.putText(frame, main_text,
                (cx - mw // 2, cy - 10),
                font, 2.0, color, 4, cv2.LINE_AA)
    cv2.putText(frame, main_text,
                (cx - mw // 2, cy - 10),
                font, 2.0, (255, 255, 255), 1, cv2.LINE_AA)

    # ---- 5. Detail line ----
    detail = status.message
    (dw, dh), _ = cv2.getTextSize(detail, font_icon, 0.7, 2)
    cv2.putText(frame, detail,
                (cx - dw // 2, cy + 35),
                font_icon, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

    # ---- 6. Diagnostic bar ----
    diag = (f"std={status.std_dev:.1f}  "
            f"mean={status.mean_brightness:.1f}  "
            f"consec={status.consec_count}f")
    (dgw, _), _ = cv2.getTextSize(diag, font_icon, 0.45, 1)
    cv2.putText(frame, diag,
                (cx - dgw // 2, cy + 65),
                font_icon, 0.45, (130, 130, 130), 1, cv2.LINE_AA)

    # ---- 7. Timestamp ----
    from datetime import datetime
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, ts,
                (12, h - 14),
                font_icon, 0.5, (160, 160, 160), 1, cv2.LINE_AA)