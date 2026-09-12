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


# ===========================================================================
# Tier-3 Feature: Fence-Tamper Detection (Frame-Diff Energy)
# ===========================================================================

@dataclass
class FenceTamperStatus:
    """Result returned by FenceTamperDetector.update() each frame."""
    is_tamper          : bool  = False
    tamper_type        : str   = "none"   # "none" | "spike" | "persistent"
    energy             : float = 0.0      # Current mean diff energy
    baseline_energy    : float = 0.0
    spike_detected     : bool  = False
    persistent_detected: bool  = False
    risk_score         : float = 0.0      # 0.0 - 100.0 scale
    alert_level        : str   = "NORMAL" # "NORMAL" | "HIGH" | "CRITICAL"
    message            : str   = ""
    duration_sec       : float = 0.0
    evidence_snapshot  : Optional[np.ndarray] = None


class FenceTamperDetector:
    """
    Monitors a designated fence or barrier ROI for physical disturbance, wire cutting,
    or structural shaking using frame-difference energy aggregation.

    Detects:
      1. Energy Spike without tracked object: sudden physical impact / fence shake.
      2. Persistent High Energy: ongoing cutting, climbing, or mechanical tampering.
    """
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        roi_polygon: Optional[List[List[int]]] = None,
        energy_spike_threshold: float = 30.0,
        persistent_threshold: float = 18.0,
        persistent_duration_sec: float = 2.0,
        tamper_base_risk: float = 85.0,
    ):
        ft_cfg = (config or {}).get("fence_tamper", {}) if config else {}
        self.enabled = bool(ft_cfg.get("enabled", True))
        self.spike_thresh = float(ft_cfg.get("energy_spike_threshold", energy_spike_threshold))
        self.pers_thresh = float(ft_cfg.get("persistent_threshold", persistent_threshold))
        self.pers_duration = float(ft_cfg.get("persistent_duration_sec", persistent_duration_sec))
        self.base_risk = float(ft_cfg.get("tamper_base_risk", tamper_base_risk))
        self.min_motion_area = int(ft_cfg.get("min_motion_area", 1200))

        raw_poly = ft_cfg.get("roi_polygon") or roi_polygon or [
            [0, 520], [1280, 520], [1280, 600], [0, 600]
        ]
        self.raw_polygon = np.array(raw_poly, dtype=np.int32)
        self.polygon = self.raw_polygon.copy()

        self._mask: Optional[np.ndarray] = None
        self._mask_area: int = 1
        self._prev_roi_gray: Optional[np.ndarray] = None
        self._energy_history: List[float] = []
        self._pers_start_time: Optional[float] = None
        self._pers_active_sec: float = 0.0

        logger.info(
            f"FenceTamperDetector ready | enabled={self.enabled} | "
            f"spike_thresh={self.spike_thresh} | pers_thresh={self.pers_thresh} | "
            f"pers_dur={self.pers_duration}s | risk={self.base_risk}"
        )

    # -----------------------------------------------------------------------
    def scale_to_frame(self, frame_w: int, frame_h: int, ref_w: int = 1280, ref_h: int = 720):
        """Scale ROI polygon to match camera resolution."""
        sx = frame_w / float(ref_w)
        sy = frame_h / float(ref_h)
        scaled = []
        for pt in self.raw_polygon:
            scaled.append([int(pt[0] * sx), int(pt[1] * sy)])
        self.polygon = np.array(scaled, dtype=np.int32)
        self._mask = None  # Force mask rebuild

    # -----------------------------------------------------------------------
    def _build_mask(self, h: int, w: int):
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [self.polygon], 255)
        self._mask = mask
        self._mask_area = max(1, int(np.count_nonzero(mask)))

    # -----------------------------------------------------------------------
    def update(
        self,
        frame: np.ndarray,
        tracked_objects: Optional[List[Dict[str, Any]]] = None,
        fps: float = 15.0
    ) -> FenceTamperStatus:
        """
        Process frame and evaluate fence energy metrics.

        Args:
            frame: Current BGR video frame.
            tracked_objects: List of current track dicts with 'bbox'.
            fps: Frame rate for timing.
        """
        if not self.enabled or frame is None or frame.size == 0:
            return FenceTamperStatus()

        h, w = frame.shape[:2]
        if self._mask is None or self._mask.shape != (h, w):
            self._build_mask(h, w)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply slight Gaussian blur to suppress camera sensor noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        if self._prev_roi_gray is None or self._prev_roi_gray.shape != blurred.shape:
            self._prev_roi_gray = blurred
            return FenceTamperStatus()

        # Compute absolute difference inside ROI
        diff = cv2.absdiff(blurred, self._prev_roi_gray)
        self._prev_roi_gray = blurred

        masked_diff = cv2.bitwise_and(diff, diff, mask=self._mask)
        mean_energy = float(np.sum(masked_diff)) / float(self._mask_area)

        self._energy_history.append(mean_energy)
        if len(self._energy_history) > 30:
            self._energy_history.pop(0)

        # Check if a tracked person/vehicle overlaps the fence ROI
        has_overlapping_track = False
        rx1, ry1 = int(np.min(self.polygon[:, 0])), int(np.min(self.polygon[:, 1]))
        rx2, ry2 = int(np.max(self.polygon[:, 0])), int(np.max(self.polygon[:, 1]))

        if tracked_objects:
            for obj in tracked_objects:
                bx1, by1, bx2, by2 = obj.get("bbox", [0, 0, 0, 0])
                # Overlap test between bbox and fence bounding box
                if not (bx2 < rx1 or bx1 > rx2 or by2 < ry1 or by1 > ry2):
                    has_overlapping_track = True
                    break

        now = time.time()
        spike = (mean_energy >= self.spike_thresh) and not has_overlapping_track
        persistent = False

        if mean_energy >= self.pers_thresh and not has_overlapping_track:
            if self._pers_start_time is None:
                self._pers_start_time = now
            self._pers_active_sec = now - self._pers_start_time
            if self._pers_active_sec >= self.pers_duration:
                persistent = True
        else:
            self._pers_start_time = None
            self._pers_active_sec = 0.0

        is_tamper = spike or persistent
        if not is_tamper:
            return FenceTamperStatus(
                is_tamper=False,
                energy=round(mean_energy, 2),
                alert_level="NORMAL"
            )

        tamper_type = "spike" if spike else "persistent"
        severity = "CRITICAL" if spike else "HIGH"
        risk_val = self.base_risk if spike else (self.base_risk - 10.0)

        msg = (
            f"FENCE TAMPER [{tamper_type.upper()}]: Frame-diff energy {mean_energy:.1f} "
            f"(threshold: {self.spike_thresh if spike else self.pers_thresh:.1f}) "
            f"without tracked object — potential perimeter barrier breach or wire vibration."
        )

        logger.warning(msg)

        return FenceTamperStatus(
            is_tamper=True,
            tamper_type=tamper_type,
            energy=round(mean_energy, 2),
            spike_detected=spike,
            persistent_detected=persistent,
            risk_score=round(risk_val, 1),
            alert_level=severity,
            message=msg,
            duration_sec=round(self._pers_active_sec, 1),
            evidence_snapshot=frame.copy()
        )

    def reset(self):
        self._prev_roi_gray = None
        self._energy_history.clear()
        self._pers_start_time = None
        self._pers_active_sec = 0.0


# ---------------------------------------------------------------------------
# Fence Tamper Overlay
# ---------------------------------------------------------------------------

def draw_fence_tamper_overlay(
    frame: np.ndarray,
    status: FenceTamperStatus,
    polygon: np.ndarray,
    frame_id: int = 0
) -> None:
    """Draw fence ROI and tamper warning banner on frame."""
    if polygon is None or len(polygon) < 3:
        return

    # Draw fence ROI polygon (flashing red if tamper, subtle purple if normal)
    if status.is_tamper:
        is_blink = (frame_id // 6) % 2 == 0
        poly_col = (0, 0, 255) if is_blink else (0, 140, 255)
        thickness = 3
    else:
        poly_col = (180, 100, 220)
        thickness = 1

    cv2.polylines(frame, [polygon], isClosed=True, color=poly_col, thickness=thickness)

    # Fence ROI label
    fx, fy = int(polygon[0][0]), int(polygon[0][1])
    fence_lbl = f" FENCE-ROI (Energy: {status.energy:.1f}) "
    cv2.putText(frame, fence_lbl, (fx + 10, fy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, poly_col, 1, cv2.LINE_AA)

    if status.is_tamper:
        h, w = frame.shape[:2]
        banner_bg = (0, 0, 200)
        cv2.rectangle(frame, (0, 36), (w, 68), banner_bg, -1)
        alert_txt = f" [!] CRITICAL: FENCE TAMPER / PHYSICAL DISTURBANCE DETECTED ({status.tamper_type.upper()}) | Risk: {status.risk_score:.0f} "
        cv2.putText(frame, alert_txt, (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)