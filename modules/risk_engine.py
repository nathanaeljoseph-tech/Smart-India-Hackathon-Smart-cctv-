"""
risk_engine.py - Progressive Risk Scoring & Behaviour Classification (DEV 1)
=============================================================================
Smart India Hackathon 2026 | Problem: SIH26187
AMST Border-Net | DEV 1 module

This module is the NEW stateful engine that sits between the tracker output
and the JSON exporter.  It handles everything that requires memory across
frames:

  1. Too-close detection   ΓÇö flags and filters persons that are less than
                             ~2 ft from the camera (huge bounding boxes).
  2. EMA centroid smoother ΓÇö exponential moving average on (cx, cy) per track
                             to reduce jitter in centroid trajectories.
  3. Behaviour classifier  ΓÇö determines APPROACH / LOITER / ACCESS / NONE
                             based on zone position, trajectory, and dwell time.
  4. Progressive risk scoreΓÇö 0ΓÇô100 score combining behaviour, zone type, dwell
                             time, and an optional night multiplier.
                             Hysteresis prevents flickering; decay reduces score
                             smoothly when the person retreats.
  5. Alert card generator  ΓÇö builds a rich alert dict per track for HUD display
                             and JSON export.

Design philosophy:
  - This module is STATEFUL (per-track dicts); boundary_engine.py stays
    STATELESS (pure geometry functions).  They are complementary.
  - All tuneable parameters are defined as module-level constants so they
    can be changed in one place without touching logic.
  - Graceful degradation: if boundary_engine is unavailable the risk engine
    returns empty cards rather than crashing.

Public API:
    RiskEngine(config)              constructor ΓÇö accepts a config dict
    engine.update(tracks, ...)      main call per frame ΓåÆ Dict[int, AlertCard]
    engine.reset()                  clear all per-track state

Tuning guide (all constants below):
    TOO_CLOSE_HEIGHT_RATIO  ΓÇö raise to let more very-close detections through;
                               lower to filter them more aggressively.
    TOO_CLOSE_CONF_THRESH   ΓÇö raise if very-close people produce noisy false
                               positives; lower if valid close people are lost.
    EMA_ALPHA               ΓÇö 0.3 = very smooth (more latency); 0.7 = fast
                               response (more jitter). 0.4 is the sweet spot.
    APPROACH_BAND_RATIO     ΓÇö fraction of frame height treated as approach band.
                               0.20 = bottom fifth of the frame.
    MIN_TRACK_LEN           ΓÇö frames before behaviour is assigned.  Prevents
                               flickering on freshly-created tracks.
    RISK_DECAY_RATE         ΓÇö risk points shed per SECOND when behaviour=none.
    HYSTERESIS_BAND         ΓÇö maximum risk DROP per frame.  Prevents the score
                               from bouncing between frames when the detector
                               flickers.

Author : DEV 1
Date   : 2026-09-09
"""

import math
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("amst_border_net.risk_engine")

# ---------------------------------------------------------------------------
# Tuneable constants (override via boundary_config.yaml ΓåÆ RiskEngine(config))
# ---------------------------------------------------------------------------

# --- Too-close filtering ---
# If a person's bbox height exceeds this fraction of the frame height they are
# considered "very close to the camera" (< ~2 ft / 15-30 cm) and trigger a proximity breach alert.
# Typical value: 0.70 (covers ΓëÑ 70 % of frame height)
TOO_CLOSE_HEIGHT_RATIO: float = 0.70

# Minimum confidence required for a very-close detection to be kept.
# Lowered to 0.35 to catch macro close-ups and camera obstructions.
TOO_CLOSE_CONF_THRESH: float = 0.35

# --- EMA centroid smoothing ---
# alpha = weight given to the NEW measurement.
# (1 - alpha) = weight given to the historical average.
# Lower ΓåÆ smoother trajectory, more latency.
EMA_ALPHA: float = 0.40

# --- Behaviour classification ---
# Fraction of frame HEIGHT from the bottom that constitutes the "approach band".
# e.g. 0.20 ΓåÆ bottom 20 % of the frame.
APPROACH_BAND_RATIO: float = 0.20

# Minimum track age (frames) before behaviour is assigned.
# Avoids jitter on brand-new detections.
MIN_TRACK_LEN: int = 12

# Number of frames used to compute motion direction (approach direction check).
APPROACH_DIRECTION_FRAMES: int = 6

# Minimum upward pixel delta (cy getting smaller = person moving toward camera)
# over the last APPROACH_DIRECTION_FRAMES to confirm "approaching".
# Set to 0 to disable direction check (presence in band is enough).
APPROACH_MIN_UPWARD_PX: float = 4.0

# --- Progressive risk score ---
# Base risk contributions per behaviour
RISK_BASE: Dict[str, float] = {
    "approach" : 20.0,
    "access"   : 40.0,
    "loiter"   : 55.0,
    "none"     : 0.0,
}

# Zone-type multiplier applied to the base risk value
ZONE_MULTIPLIER: Dict[str, float] = {
    "restricted": 1.0,
    "monitor"   : 0.65,
}

# Loiter bonus: extra risk per second of continuous loitering (on top of base)
LOITER_BONUS_PER_SEC: float = 2.5

# Night mode multiplier (applied to final risk score when --night flag is set)
NIGHT_MULTIPLIER: float = 1.30

# Max risk score decay per SECOND (subtracted when behaviour == "none")
RISK_DECAY_RATE: float = 8.0

# Hysteresis: maximum DROP in risk score per frame.
# Prevents rapid flickering when detections are unstable.
HYSTERESIS_BAND: float = 5.0

# --- Alert levels (risk thresholds) ---
ALERT_THRESHOLDS = [
    (75, "CRITICAL"),
    (50, "HIGH"),
    (25, "SUSPICIOUS"),
    (0,  "NORMAL"),
]


# ===========================================================================
# Helpers
# ===========================================================================

def _risk_to_alert_level(risk: float) -> str:
    """Map a 0ΓÇô100 risk score to an alert level string."""
    for threshold, level in ALERT_THRESHOLDS:
        if risk >= threshold:
            return level
    return "NORMAL"


def _build_reasoning(behavior: str, alert_type: str, dwell_sec: float,
                     zone_id: Optional[str], risk: float) -> str:
    """
    Build a human-readable reasoning string for the alert card.

    Examples:
        "Loitering in restricted zone for 32s"
        "Approaching restricted area"
        "Access to restricted zone detected (14s dwell)"
    """
    zone_str = f" zone '{zone_id}'" if zone_id else " zone"

    if behavior == "loiter":
        return f"Loitering in restricted{zone_str} for {dwell_sec:.0f}s"
    if behavior == "approach":
        return f"Approaching restricted area ΓÇö risk {risk:.0f}/100"
    if behavior == "access":
        return f"Access to restricted{zone_str} detected ({dwell_sec:.0f}s dwell)"
    if alert_type == "intrusion":
        return f"Intrusion in restricted{zone_str}"
    if alert_type == "loitering":
        return f"Loitering in{zone_str} for {dwell_sec:.0f}s"
    return "Monitoring ΓÇö no immediate threat"


# ===========================================================================
# Too-close filtering
# ===========================================================================

def filter_too_close(
    detections: List[Dict[str, Any]],
    frame_h: int,
    height_ratio: float = TOO_CLOSE_HEIGHT_RATIO,
    conf_threshold: float = TOO_CLOSE_CONF_THRESH,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Separate detections into normal (trackable) and very-close (special handling).

    HOW IT WORKS:
        A person 1-2 ft from the camera will have a bounding box whose height
        covers most of the frame.  We compute:
            ratio = (y2 - y1) / frame_height
        If ratio > height_ratio: the detection is flagged as "very close".
        Very-close detections are only kept if confidence >= conf_threshold
        (stricter gate because close-up detections are often noisier).

    WHY SEPARATE instead of discard:
        We still draw very-close people on the HUD (with a distinct yellow
        dotted box) so the operator sees them.  We just exclude them from
        the tracker to prevent ID flickering and history contamination.

    Args:
        detections    : List of detection dicts from YOLODetector.detect().
        frame_h       : Frame height in pixels.
        height_ratio  : Bbox-height / frame-height threshold.  Default 0.85.
        conf_threshold: Minimum confidence for a very-close detection to be
                        kept in the very-close list (for HUD display).

    Returns:
        (normal_dets, very_close_dets)
            normal_dets     : Detections safe to pass to the tracker.
            very_close_dets : Detections flagged as too-close (for HUD only).
    """
    if frame_h <= 0:
        return detections, []

    normal: List[Dict[str, Any]] = []
    very_close: List[Dict[str, Any]] = []

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        bbox_h = y2 - y1
        ratio = bbox_h / frame_h

        if ratio >= height_ratio or det.get("is_too_close", False):
            # Mark the detection as close-proximity breach
            det = dict(det)  # shallow copy ΓÇö don't mutate original
            det["is_too_close"]     = True
            det["bbox_height_ratio"] = round(ratio, 3)
            # Keep if confidence meets the gate
            if det.get("confidence", 0.0) >= conf_threshold:
                very_close.append(det)
        else:
            det = dict(det)
            det["is_too_close"]      = False
            det["bbox_height_ratio"] = round(ratio, 3)
            normal.append(det)

    if very_close:
        logger.debug(
            f"filter_too_close: {len(very_close)} very-close detection(s) "
            f"detected as proximity breach (height_ratio>={height_ratio:.2f})"
        )

    return normal, very_close


# ===========================================================================
# Zone position classifier
# ===========================================================================

def classify_zone_position(
    centroid: Tuple[int, int],
    frame_h: int,
    approach_ratio: float = APPROACH_BAND_RATIO,
) -> str:
    """
    Determine where in the frame the centroid sits.

    Spatial model (top ΓåÆ bottom):
        "safe"     ΓåÆ top (1 - approach_ratio) of the frame  (distant / far away)
        "interior" ΓåÆ overlap with restricted zone polygon (main access area)
        "approach" ΓåÆ bottom approach_ratio of the frame (person walking toward cam)

    NOTE: The boundary between "interior" and "approach" is purely geometric
    (y-coordinate based).  The actual polygon check is done in boundary_engine.
    This function is a fast pre-check used by the behaviour classifier.

    Args:
        centroid      : (cx, cy) in pixel coordinates.
        frame_h       : Frame height in pixels.
        approach_ratio: Fraction of frame height for the approach band.

    Returns:
        "approach" | "interior" | "safe"
    """
    _, cy = centroid
    approach_y_start = int(frame_h * (1.0 - approach_ratio))

    if cy >= approach_y_start:
        return "approach"
    return "interior"


# ===========================================================================
# Behaviour classifier
# ===========================================================================

def classify_behavior(
    centroid: Tuple[int, int],
    history: List[Dict[str, Any]],
    frame_h: int,
    in_restricted_zone: bool,
    dwell_sec: float,
    min_loiter_sec: float,
    approach_ratio: float = APPROACH_BAND_RATIO,
    min_track_len: int = MIN_TRACK_LEN,
) -> str:
    """
    Classify current behaviour for one tracked person.

    Rules (evaluated top-down; first match wins):

    1. NONE ΓÇö track is too young (age < min_track_len).
    2. LOITER ΓÇö person has been continuously inside the restricted zone for
                ΓëÑ min_loiter_sec seconds.
    3. ACCESS ΓÇö person is inside the restricted zone (not approach band),
                track age ΓëÑ min_track_len.
    4. APPROACH ΓÇö person is in the approach band AND moving upward (cy
                  decreasing over last APPROACH_DIRECTION_FRAMES frames).
    5. NONE ΓÇö otherwise.

    Args:
        centroid          : (cx, cy) ΓÇö current EMA-smoothed centroid.
        history           : Ordered list (oldestΓåÆnewest) of centroid history
                            dicts {\"xy\": (cx, cy), \"frame_id\": int}.
        frame_h           : Frame height in pixels.
        in_restricted_zone: Whether the centroid is inside any restricted polygon
                            (result from boundary_engine.check_intrusion).
        dwell_sec         : Continuous dwell time inside the zone (seconds).
        min_loiter_sec    : Seconds of dwell to trigger loitering.
        approach_ratio    : Fraction of frame height for approach band.
        min_track_len     : Minimum track age (frames) before behaviour assigned.

    Returns:
        "loiter" | "access" | "approach" | "none"
    """
    track_age = len(history)

    # Rule 1: track too young
    if track_age < min_track_len:
        return "none"

    # Rule 2: loitering (continuous dwell ΓëÑ threshold)
    if in_restricted_zone and dwell_sec >= min_loiter_sec:
        return "loiter"

    # Rule 3: access (inside zone, not approach band)
    zone_pos = classify_zone_position(centroid, frame_h, approach_ratio)
    if in_restricted_zone and zone_pos == "interior":
        return "access"

    # Rule 4: approach (in approach band + moving toward camera / upward)
    if zone_pos == "approach":
        # Check direction: is the person's cy decreasing (moving toward cam)?
        n = min(APPROACH_DIRECTION_FRAMES, len(history))
        if n >= 2:
            ys = [h["xy"][1] for h in history[-n:]]
            # Positive delta_y means person moved down (away from camera).
            # Negative delta_y means person moved up (toward camera).
            delta_y = ys[-1] - ys[0]   # negative = approaching
            if delta_y <= APPROACH_MIN_UPWARD_PX:  # approaching or stationary
                return "approach"

    return "none"


# ===========================================================================
# Risk Engine (stateful, per-track)
# ===========================================================================

class RiskEngine:
    """
    Progressive risk scorer with hysteresis and decay.

    Maintains per-track state:
        ema_cx, ema_cy      : EMA-smoothed centroid
        risk_score          : Current 0ΓÇô100 risk value
        track_age           : Number of frames this track has been seen
        dwell_frames        : Frames continuously inside the zone
        last_alert_card     : Most recent alert card dict

    Usage:
        engine = RiskEngine(config)
        # inside main loop:
        alert_cards = engine.update(tracks, frame_h, frame_w,
                                    boundary_zones, fps, frame_id, night_mode)
        # on tracker reset:
        engine.reset()
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config : Optional dict loaded from boundary_config.yaml.
                     Keys recognised:
                       approach_band_ratio, min_track_len_for_behavior,
                       min_loiter_seconds, too_close_height_ratio,
                       too_close_conf_threshold,
                       risk.base_approach, risk.base_access, risk.base_loiter,
                       risk.decay_rate, risk.hysteresis_band, risk.night_multiplier
        """
        cfg = config or {}
        risk_cfg = cfg.get("risk", {}) or {}

        # Thresholds (can be overridden from yaml)
        self.approach_ratio    = float(cfg.get("approach_band_ratio",          APPROACH_BAND_RATIO))
        self.min_track_len     = int(cfg.get("min_track_len_for_behavior",     MIN_TRACK_LEN))
        self.min_loiter_sec    = float(cfg.get("min_loiter_seconds",           5.0))
        self.too_close_ratio   = float(cfg.get("too_close_height_ratio",       TOO_CLOSE_HEIGHT_RATIO))
        self.too_close_conf    = float(cfg.get("too_close_conf_threshold",     TOO_CLOSE_CONF_THRESH))
        self.proximity_risk    = float(cfg.get("proximity_risk_score",         98.0))
        self.proximity_enabled = bool(cfg.get("proximity_threat_enabled",       True))

        # Risk base values
        self.risk_base = {
            "approach" : float(risk_cfg.get("base_approach",  RISK_BASE["approach"])),
            "access"   : float(risk_cfg.get("base_access",    RISK_BASE["access"])),
            "loiter"   : float(risk_cfg.get("base_loiter",    RISK_BASE["loiter"])),
            "none"     : 0.0,
        }
        self.decay_rate      = float(risk_cfg.get("decay_rate",        RISK_DECAY_RATE))
        self.hysteresis_band = float(risk_cfg.get("hysteresis_band",   HYSTERESIS_BAND))
        self.night_mult      = float(risk_cfg.get("night_multiplier",  NIGHT_MULTIPLIER))

        # Per-track state dicts
        # {track_id: {"ema_cx": float, "ema_cy": float, "risk": float,
        #             "track_age": int, "dwell_frames": int, "behavior": str}}
        self._state: Dict[int, Dict[str, Any]] = {}

        logger.info(
            f"RiskEngine ready | approach_ratio={self.approach_ratio} | "
            f"min_track_len={self.min_track_len} | min_loiter={self.min_loiter_sec}s | "
            f"decay={self.decay_rate}/s | hysteresis={self.hysteresis_band}/frame"
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def update(
        self,
        tracks: List[Dict[str, Any]],
        frame_h: int,
        frame_w: int,
        boundary_zones: List[Dict[str, Any]],
        fps: float,
        frame_id: int,
        night_mode: bool = False,
        centroid_histories: Optional[Dict[int, List[Dict]]] = None,
        very_close_dets: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[Any, Dict[str, Any]]:
        """
        Main update call ΓÇö called once per processed frame.

        Args:
            tracks             : List of track dicts from FallbackIOUTracker.
                                 Each dict must contain: track_id, bbox, confidence, class.
                                 Optionally: ema_centroid, track_age (added by tracker).
            frame_h            : Frame height in pixels.
            frame_w            : Frame width in pixels.
            boundary_zones     : List of zone config dicts (from yaml).
            fps                : Current measured FPS (used for dwell ΓåÆ seconds).
            frame_id           : Current processed frame index.
            night_mode         : If True, apply night multiplier to risk score.
            centroid_histories : Per-track centroid history dicts from main.py
                                 {track_id: [{\"xy\": (cx,cy), \"frame_id\": int}, ...]}.

        Returns:
            Dict mapping track_id ΓåÆ alert card dict.
            Alert card keys:
                track_id, bbox, centroid, ema_centroid, confidence, track_age,
                is_too_close, behavior, risk_score, alert_level, reasoning,
                dwell_sec, alert_type, severity, zone_id
        """
        safe_fps = fps if fps and fps > 0 else 15.0
        alert_cards: Dict[int, Dict[str, Any]] = {}

        active_ids = {t["track_id"] for t in tracks}

        # ---- Age out state for tracks that have disappeared ----
        for old_id in list(self._state.keys()):
            if old_id not in active_ids:
                del self._state[old_id]

        for t in tracks:
            tid  = t["track_id"]
            bbox = t["bbox"]
            x1, y1, x2, y2 = bbox

            # ---- 1. Update EMA centroid ----
            raw_cx = int((x1 + x2) / 2)
            raw_cy = int((y1 + y2) / 2)

            state = self._state.setdefault(tid, {
                "ema_cx"      : float(raw_cx),
                "ema_cy"      : float(raw_cy),
                "risk"        : 0.0,
                "track_age"   : 0,
                "dwell_frames": 0,
                "behavior"    : "none",
                "last_zone_id": None,
            })

            state["ema_cx"] = EMA_ALPHA * raw_cx + (1.0 - EMA_ALPHA) * state["ema_cx"]
            state["ema_cy"] = EMA_ALPHA * raw_cy + (1.0 - EMA_ALPHA) * state["ema_cy"]
            state["track_age"] += 1

            ema_cx = int(round(state["ema_cx"]))
            ema_cy = int(round(state["ema_cy"]))
            centroid = (ema_cx, ema_cy)

            # ---- 2. Zone checks (restricted/monitor) ----
            # Determine if the smoothed centroid is inside any restricted zone
            in_restricted = False
            alert_type_be  = "none"  # from boundary_engine style logic
            severity_be    = "none"
            zone_id        = state.get("last_zone_id")

            for zone in boundary_zones:
                z_type  = zone.get("type", "restricted").lower()
                polygon = zone.get("polygon", [])
                if len(polygon) < 3:
                    continue

                # Point-in-polygon check
                try:
                    from modules.boundary_engine import point_in_polygon
                    inside = point_in_polygon(ema_cx, ema_cy, polygon)
                except ImportError:
                    inside = False

                if inside:
                    if z_type == "restricted":
                        in_restricted = True
                        alert_type_be = "intrusion"
                        severity_be   = "high"
                        zone_id       = zone.get("id", "zone")
                        break  # restricted beats monitor
                    elif alert_type_be == "none":
                        alert_type_be = "presence"
                        zone_id       = zone.get("id", "zone")

            state["last_zone_id"] = zone_id

            # ---- 3. Dwell time tracking ----
            if in_restricted:
                state["dwell_frames"] += 1
            else:
                state["dwell_frames"] = 0

            dwell_sec = state["dwell_frames"] / safe_fps

            # ---- 4. Behaviour classification ----
            history = (centroid_histories or {}).get(tid, [])
            behavior = classify_behavior(
                centroid          = centroid,
                history           = history,
                frame_h           = frame_h,
                in_restricted_zone= in_restricted,
                dwell_sec         = dwell_sec,
                min_loiter_sec    = self.min_loiter_sec,
                approach_ratio    = self.approach_ratio,
                min_track_len     = self.min_track_len,
            )
            state["behavior"] = behavior

            # ---- 5. Risk scoring ----
            target_risk = self._compute_target_risk(
                behavior, in_restricted, dwell_sec, night_mode, zone_id, boundary_zones
            )
            prev_risk = state["risk"]
            state["risk"] = self._apply_risk_dynamics(
                prev_risk, target_risk, behavior, safe_fps
            )
            risk_score  = round(min(100.0, max(0.0, state["risk"])), 1)
            alert_level = _risk_to_alert_level(risk_score)

            # Upgrade boundary_engine intrusion severity if risk is critical
            if alert_type_be == "intrusion" and risk_score >= 75:
                severity_be = "high"
            elif behavior == "loiter":
                alert_type_be = "loitering"
                severity_be   = "medium"

            # ---- 6. Build alert card ----
            reasoning = _build_reasoning(
                behavior, alert_type_be, dwell_sec, zone_id, risk_score
            )

            card: Dict[str, Any] = {
                # Identity
                "track_id"   : tid,
                "bbox"       : bbox,
                "centroid"   : [raw_cx, raw_cy],
                "ema_centroid": [ema_cx, ema_cy],
                "confidence" : t.get("confidence", 0.0),
                "track_age"  : state["track_age"],
                "is_too_close": t.get("is_too_close", False),
                # Behaviour & risk
                "behavior"   : behavior,
                "risk_score" : risk_score,
                "alert_level": alert_level,
                "reasoning"  : reasoning,
                "dwell_sec"  : round(dwell_sec, 1),
                # Boundary engine compat fields
                "alert_type" : alert_type_be,
                "severity"   : severity_be,
                "zone_id"    : zone_id,
            }

            alert_cards[tid] = card

            # Log non-normal alerts
            if alert_level != "NORMAL":
                logger.info(
                    f"RISK | track_id={tid} | behavior={behavior.upper()} | "
                    f"risk={risk_score:.0f} | level={alert_level} | "
                    f"dwell={dwell_sec:.1f}s | zone={zone_id} | {reasoning}"
                )

        # Process any very-close detections (macro close-ups e.g. 15-30cm / camera tampering)
        if very_close_dets and self.proximity_enabled:
            for idx, vc_det in enumerate(very_close_dets):
                vc_id = f"PROX-{idx+1}"
                vx1, vy1, vx2, vy2 = vc_det["bbox"]
                vcx = int((vx1 + vx2) / 2)
                vcy = int((vy1 + vy2) / 2)
                prox_card: Dict[str, Any] = {
                    "track_id"    : vc_id,
                    "bbox"        : vc_det["bbox"],
                    "centroid"    : [vcx, vcy],
                    "ema_centroid": [vcx, vcy],
                    "confidence"  : vc_det.get("confidence", 0.90),
                    "track_age"   : 1,
                    "is_too_close": True,
                    "behavior"    : "PROXIMITY BREACH",
                    "risk_score"  : self.proximity_risk,
                    "alert_level" : "CRITICAL",
                    "reasoning"   : "Subject <30cm from sensor (tampering/breach threat)",
                    "dwell_sec"   : 0.0,
                    "alert_type"  : "intrusion",
                    "severity"    : "high",
                    "zone_id"     : "proximity",
                }
                alert_cards[vc_id] = prox_card
                logger.warning(
                    f"PROXIMITY THREAT | track_id={vc_id} | RISK={self.proximity_risk} | "
                    f"CRITICAL | Subject <30cm from camera!"
                )

        return alert_cards

    # -----------------------------------------------------------------------
    # Risk dynamics
    # -----------------------------------------------------------------------

    def _compute_target_risk(
        self,
        behavior: str,
        in_restricted: bool,
        dwell_sec: float,
        night_mode: bool,
        zone_id: Optional[str],
        boundary_zones: List[Dict[str, Any]],
    ) -> float:
        """
        Compute the DESIRED risk score given current behaviour and context.
        The actual score may differ due to hysteresis and decay.
        """
        base = self.risk_base.get(behavior, 0.0)

        # Zone multiplier (restricted zones get full weight)
        zone_mult = 1.0
        if zone_id:
            for zone in boundary_zones:
                if zone.get("id") == zone_id:
                    z_type    = zone.get("type", "restricted").lower()
                    zone_mult = ZONE_MULTIPLIER.get(z_type, 1.0)
                    break

        target = base * zone_mult

        # Loiter bonus: each second of dwell beyond the base adds more risk
        if behavior == "loiter" and dwell_sec > 0:
            target += dwell_sec * LOITER_BONUS_PER_SEC

        # Night multiplier
        if night_mode and target > 0:
            target *= self.night_mult

        return min(100.0, target)

    def _apply_risk_dynamics(
        self,
        prev_risk: float,
        target_risk: float,
        behavior: str,
        fps: float,
    ) -> float:
        """
        Apply hysteresis (limits drop per frame) and decay (when idle).

        Hysteresis: risk can only FALL by at most HYSTERESIS_BAND per frame.
        This prevents rapid flickering when a detection briefly disappears.

        Decay: when behaviour is "none", subtract DECAY_RATE/fps per frame.
        Risk never decays below 0.
        """
        if behavior == "none":
            # Decay mode: slowly bring risk down
            decay_per_frame = self.decay_rate / fps
            new_risk = max(0.0, prev_risk - decay_per_frame)
            # Still apply hysteresis cap on how fast it can drop
        else:
            # Drive toward target; cap the drop by hysteresis_band
            if target_risk >= prev_risk:
                # Rising risk: allow immediate increase
                new_risk = target_risk
            else:
                # Falling risk (behaviour improved): apply hysteresis
                max_drop = self.hysteresis_band
                new_risk = max(target_risk, prev_risk - max_drop)

        return new_risk

    # -----------------------------------------------------------------------
    # State management
    # -----------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all per-track state. Call when tracker is reset."""
        self._state.clear()
        logger.info("RiskEngine state cleared.")

    def get_state(self) -> Dict[int, Dict[str, Any]]:
        """Return a copy of the current per-track state (read-only)."""
        return dict(self._state)
