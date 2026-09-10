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

# Grace period: zone-exit frames allowed before dwell counter resets
DWELL_RESET_GRACE: int = 20

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

# --- Animal class names for suppression ---
ANIMAL_CLASSES = {
    "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe",
    "teddy bear", "animal"
}

# Context penalty when unexpected object or time violation detected
CONTEXT_PENALTY_WEIGHT: float = 30.0

# Base risk for abandoned object alerts
ABANDONED_BASE_RISK: float = 75.0


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

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None
    ):
        """
        Args:
            config : Optional dict loaded from boundary_config.yaml.
            config_path : Optional path to boundary_config.yaml.
        """
        cfg = config or {}
        if not cfg and config_path:
            import os
            import yaml
            if os.path.isfile(config_path):
                try:
                    with open(config_path, "r") as f:
                        cfg = yaml.safe_load(f) or {}
                except Exception as e:
                    logger.warning(f"Failed to load risk config from {config_path}: {e}")

        risk_cfg = cfg.get("risk", {}) or {}

        # Thresholds (can be overridden from yaml)
        self.approach_ratio    = float(cfg.get("approach_band_ratio",          APPROACH_BAND_RATIO))
        self.min_track_len     = int(cfg.get("min_track_len_for_behavior",     MIN_TRACK_LEN))
        self.min_loiter_sec    = float(cfg.get("min_loiter_seconds",           5.0))
        self.too_close_ratio   = float(cfg.get("too_close_height_ratio",       TOO_CLOSE_HEIGHT_RATIO))
        self.too_close_conf    = float(cfg.get("too_close_conf_threshold",     TOO_CLOSE_CONF_THRESH))
        self.proximity_risk    = float(cfg.get("proximity_risk_score",         98.0))
        self.proximity_enabled = bool(cfg.get("proximity_threat_enabled",       True))

        # Tier-2 parameters
        self.animal_suppression_enabled = bool(cfg.get("animal_suppression_enabled", True))
        self.animal_suppressed_risk     = float(cfg.get("animal_suppressed_risk", 0.0))
        self.context_penalty_weight     = float(cfg.get("context_penalty_weight", CONTEXT_PENALTY_WEIGHT))
        self.abandoned_risk_score       = float(cfg.get("abandoned_risk_score", ABANDONED_BASE_RISK))

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
            f"decay={self.decay_rate}/s | hysteresis={self.hysteresis_band}/frame | "
            f"animal_suppression={self.animal_suppression_enabled} | "
            f"context_penalty={self.context_penalty_weight}"
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def update(
        self,
        tracks: List[Dict[str, Any]],
        frame_h: int = 720,
        frame_w: int = 1280,
        boundary_zones: Optional[List[Dict[str, Any]]] = None,
        fps: float = 15.0,
        frame_id: int = 0,
        night_mode: bool = False,
        centroid_histories: Optional[Dict[int, List[Dict]]] = None,
        very_close_dets: Optional[List[Dict[str, Any]]] = None,
        current_time: Optional[Any] = None,
        abandoned_events: Optional[List[Dict[str, Any]]] = None,
        animal_suppression: Optional[bool] = None,
        zone_context_mgr: Optional[Any] = None,
        boundary_alerts: Optional[Any] = None,
        abandoned_alerts: Optional[Any] = None,
        is_night_mode: Optional[bool] = None,
        **kwargs,
    ) -> Dict[Any, Dict[str, Any]]:
        """
        Main update call — called once per processed frame.

        Args:
            tracks             : List of track dicts.
            frame_h            : Frame height in pixels.
            frame_w            : Frame width in pixels.
            boundary_zones     : List of zone config dicts (from yaml).
            fps                : Current measured FPS.
            frame_id           : Current processed frame index.
            night_mode         : If True, apply night multiplier.
            centroid_histories : Per-track centroid history dicts.
            very_close_dets    : Close-up detections.
            current_time       : Optional datetime or time string for zone context.
            abandoned_events   : List of active abandoned object events.
            animal_suppression : Optional override for animal suppression.
            zone_context_mgr   : Optional ZoneContextMemory instance.
        """
        if is_night_mode is not None:
            night_mode = bool(is_night_mode)
        if abandoned_alerts is not None and abandoned_events is None:
            abandoned_events = abandoned_alerts
        if boundary_zones is None:
            boundary_zones = []
        safe_fps = fps if fps and fps > 0 else 15.0
        alert_cards: Dict[int, Dict[str, Any]] = {}

        active_ids = {t["track_id"] for t in tracks}

        # ---- Age out state for tracks that have disappeared ----
        for old_id in list(self._state.keys()):
            if old_id not in active_ids:
                del self._state[old_id]

        suppress_animals = self.animal_suppression_enabled if animal_suppression is None else animal_suppression

        for t in tracks:
            tid  = t["track_id"]
            bbox = t["bbox"]
            x1, y1, x2, y2 = bbox
            track_cls = str(t.get("class", "person")).lower()
            is_person = bool(t.get("is_person", False) or (track_cls == "person"))
            is_animal = bool(t.get("is_animal", False) or (track_cls in ANIMAL_CLASSES))
            is_vehicle = bool(t.get("is_vehicle", False) or (track_cls in ("car", "truck", "bus", "motorcycle", "bicycle", "van")))

            # ---- 1. Update EMA centroid ----
            raw_cx = int((x1 + x2) / 2)
            raw_cy = int((y1 + y2) / 2)

            state = self._state.setdefault(tid, {
                "ema_cx"      : float(raw_cx),
                "ema_cy"      : float(raw_cy),
                "risk"        : 0.0,
                "track_age"   : 0,
                "dwell_frames": 0,
                "out_frames"  : 0,
                "behavior"    : "none",
                "last_zone_id": None,
            })

            state["ema_cx"] = EMA_ALPHA * raw_cx + (1.0 - EMA_ALPHA) * state["ema_cx"]
            state["ema_cy"] = EMA_ALPHA * raw_cy + (1.0 - EMA_ALPHA) * state["ema_cy"]
            state["track_age"] += 1

            ema_cx = int(round(state["ema_cx"]))
            ema_cy = int(round(state["ema_cy"]))
            centroid = (ema_cx, ema_cy)

            # ---- 2. Zone checks (restricted/monitor/secure/no_parking) ----
            in_restricted = False
            alert_type_be  = "none"
            severity_be    = "none"
            zone_id        = state.get("last_zone_id")
            matched_zone   = None

            if zone_context_mgr is not None:
                z_ctx = zone_context_mgr.get_zone_at_point(ema_cx, ema_cy, current_time)
                if z_ctx is not None:
                    matched_zone = z_ctx.get("raw_zone", z_ctx)
                    effective_type = z_ctx.get("effective_type", "monitor").lower()
                    zone_id = z_ctx.get("zone_id")
                    can_intrude = is_person or (is_animal and not suppress_animals)
                    if effective_type in ("restricted", "secure") and can_intrude:
                        in_restricted = True
                        alert_type_be = "intrusion"
                        severity_be = "high"
                    elif alert_type_be == "none":
                        alert_type_be = "presence" if can_intrude else "none"

            if matched_zone is None:
                for zone in boundary_zones:
                    z_type  = zone.get("type", "restricted").lower()
                    polygon = zone.get("polygon", [])
                    if len(polygon) < 3:
                        continue

                    try:
                        from modules.boundary_engine import point_in_polygon
                        inside = point_in_polygon(ema_cx, ema_cy, polygon)
                    except ImportError:
                        inside = False

                    if inside:
                        matched_zone = zone
                        can_intrude = is_person or (is_animal and not suppress_animals)
                        if z_type in ("restricted", "secure") and can_intrude:
                            in_restricted = True
                            alert_type_be = "intrusion"
                            severity_be   = "high"
                            zone_id       = zone.get("id", "zone")
                            break
                        elif alert_type_be == "none":
                            alert_type_be = "presence" if can_intrude else "none"
                            zone_id       = zone.get("id", "zone")

            state["last_zone_id"] = zone_id

            # ---- 3. Dwell time tracking ----
            in_zone = in_restricted or (alert_type_be == "presence")
            if in_zone:
                state["dwell_frames"] += 1
                state["out_frames"]    = 0
            else:
                state["out_frames"] = state.get("out_frames", 0) + 1
                if state["out_frames"] >= DWELL_RESET_GRACE:
                    state["dwell_frames"] = 0

            dwell_sec = state["dwell_frames"] / safe_fps

            # ---- 4. Context & Target Classification Check ----
            context_violation = False
            context_reason = ""

            if is_animal and suppress_animals:
                # ANIMAL SUPPRESSION: Animal detected and suppression is active
                behavior = "animal_presence"
                state["behavior"] = behavior
                target_risk = self.animal_suppressed_risk
                state["risk"] = self.animal_suppressed_risk
                risk_score = round(self.animal_suppressed_risk, 1)
                alert_level = "NORMAL"
                reasoning = f"Animal detected ({track_cls}) - risk suppressed"
                alert_type_be = "none"
                severity_be = "none"

            elif not is_person and not (is_animal and not suppress_animals):
                # NON-PERSON HANDLING: Generic objects, toys, or vehicles
                # Non-person tracks do not trigger intrusion alerts
                alert_type_be = "none"
                severity_be = "none"
                if is_vehicle:
                    behavior = "vehicle_presence"
                    state["behavior"] = behavior
                    # Check zone context for vehicle
                    if matched_zone is not None:
                        try:
                            from modules.boundary_engine import is_object_allowed
                            is_ok, reason = is_object_allowed(t, matched_zone, current_time)
                            if not is_ok:
                                context_violation = True
                                context_reason = reason
                        except Exception as ce:
                            logger.debug(f"Vehicle context check error: {ce}")

                    # Vehicles receive context penalty if unauthorized, but no human intrusion base risk
                    target_risk = self.context_penalty_weight if context_violation else 0.0
                    state["risk"] = target_risk
                    risk_score = round(min(100.0, max(0.0, state["risk"])), 1)
                    alert_level = _risk_to_alert_level(risk_score)
                    reasoning = f"Vehicle detected ({track_cls})"
                    if context_violation and context_reason:
                        reasoning = f"{reasoning} [{context_reason}]"
                else:
                    # Generic objects / toys / clutter - suppress risk completely
                    behavior = "object_presence"
                    state["behavior"] = behavior
                    target_risk = 0.0
                    state["risk"] = 0.0
                    risk_score = 0.0
                    alert_level = "NORMAL"
                    reasoning = f"Tracked object ({track_cls}) - risk suppressed"

            else:
                # PERSON HANDLING: Intrusion, loiter, approach behaviour classification
                if matched_zone is not None:
                    try:
                        from modules.boundary_engine import is_object_allowed
                        is_ok, reason = is_object_allowed(t, matched_zone, current_time)
                        if not is_ok:
                            context_violation = True
                            context_reason = reason
                    except Exception as ce:
                        logger.debug(f"Context check error: {ce}")

                # Standard behaviour classification
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

                # Risk scoring with context penalty & night multiplier
                target_risk = self._compute_target_risk(
                    behavior,
                    in_restricted,
                    dwell_sec,
                    night_mode,
                    zone_id,
                    boundary_zones,
                    context_penalty=self.context_penalty_weight if context_violation else 0.0,
                )
                prev_risk = state["risk"]
                state["risk"] = self._apply_risk_dynamics(
                    prev_risk, target_risk, behavior, safe_fps
                )
                risk_score  = round(min(100.0, max(0.0, state["risk"])), 1)
                alert_level = _risk_to_alert_level(risk_score)

                if alert_type_be == "intrusion" and risk_score >= 75:
                    severity_be = "high"
                elif behavior == "loiter":
                    alert_type_be = "loitering"
                    severity_be   = "medium"

                reasoning = _build_reasoning(
                    behavior, alert_type_be, dwell_sec, zone_id, risk_score
                )
                if context_violation and context_reason:
                    reasoning = f"{reasoning} [{context_reason}]"

            # Build alert card
            card: Dict[str, Any] = {
                # Identity
                "track_id"         : tid,
                "bbox"             : bbox,
                "centroid"         : [raw_cx, raw_cy],
                "ema_centroid"     : [ema_cx, ema_cy],
                "confidence"       : t.get("confidence", 0.0),
                "track_age"        : state["track_age"],
                "is_too_close"     : t.get("is_too_close", False),
                "class"            : track_cls,
                "is_animal"        : is_animal,
                # Context info
                "context_violation": context_violation,
                "context_reason"   : context_reason,
                # Behaviour & risk
                "behavior"         : behavior,
                "risk_score"       : risk_score,
                "alert_level"      : alert_level,
                "reasoning"        : reasoning,
                "dwell_sec"        : round(dwell_sec, 1),
                # Boundary engine compat fields
                "alert_type"       : alert_type_be,
                "severity"         : severity_be,
                "zone_id"          : zone_id,
            }

            alert_cards[tid] = card

            if alert_level != "NORMAL" and not (is_animal and suppress_animals):
                logger.info(
                    f"RISK | track_id={tid} | behavior={behavior.upper()} | "
                    f"risk={risk_score:.0f} | level={alert_level} | "
                    f"dwell={dwell_sec:.1f}s | zone={zone_id} | {reasoning}"
                )

        # Process any very-close detections (macro close-ups e.g. 15-30cm / camera tampering)
        if very_close_dets and self.proximity_enabled:
            has_animal_in_scene = any(t.get("is_animal") for t in tracks)
            for idx, vc_det in enumerate(very_close_dets):
                # Suppress proximity breach if detection is an animal or animal in scene
                if vc_det.get("is_animal") or (has_animal_in_scene and suppress_animals and not vc_det.get("is_person")):
                    continue
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

        # Process abandoned object alerts (Tier-2 Multi-Category)
        if abandoned_events:
            for ab_ev in abandoned_events:
                ab_id = f"ABANDONED-{ab_ev['object_id']}"
                ab_cx, ab_cy = ab_ev["centroid"]
                category = ab_ev.get("category", "bag")
                cls_name = ab_ev.get("class", "object")
                ab_zone_id = ab_ev.get("zone_id")
                ab_zone_name = ab_ev.get("zone_name", "Zone")
                effective_zone_type = str(ab_ev.get("zone_type", "none")).lower()

                if not ab_zone_id:
                    for zone in boundary_zones:
                        poly = zone.get("polygon", [])
                        if len(poly) >= 3:
                            try:
                                from modules.boundary_engine import point_in_polygon
                                if point_in_polygon(ab_cx, ab_cy, poly):
                                    ab_zone_id = zone.get("id")
                                    ab_zone_name = zone.get("name", ab_zone_id)
                                    effective_zone_type = str(zone.get("type", "restricted")).lower()
                                    break
                            except Exception:
                                pass

                is_critical_zone = effective_zone_type in ("restricted", "secure", "no_parking")

                # Category-based base risk:
                # - Bags/Boxes in restricted/secure zones: 85.0 (CRITICAL/HIGH)
                # - Vehicles in restricted/no_parking zones: 80.0 (HIGH/CRITICAL)
                # - Small devices in secure zones: 60.0 (HIGH/SUSPICIOUS)
                # - Outside critical zones: baseline proportionate risk
                if category == "bag":
                    base_ab_risk = 85.0 if is_critical_zone else 65.0
                elif category == "vehicle":
                    base_ab_risk = 80.0 if is_critical_zone else 50.0
                elif category == "device":
                    base_ab_risk = 60.0 if is_critical_zone else 40.0
                else:
                    base_ab_risk = self.abandoned_risk_score

                is_night_event = bool(night_mode or ab_ev.get("is_night", False))
                if is_night_event:
                    base_ab_risk *= self.night_mult

                ab_risk = round(min(100.0, max(10.0, base_ab_risk)), 1)
                ab_level = _risk_to_alert_level(ab_risk)

                alert_cards[ab_id] = {
                    "track_id"     : ab_id,
                    "object_id"    : ab_ev["object_id"],
                    "bbox"         : ab_ev["bbox"],
                    "centroid"     : [ab_cx, ab_cy],
                    "ema_centroid" : [ab_cx, ab_cy],
                    "confidence"   : ab_ev.get("confidence", 0.85),
                    "track_age"    : int(ab_ev.get("stationary_time", 10.0) * safe_fps),
                    "is_too_close" : False,
                    "is_abandoned" : True,
                    "category"     : category,
                    "class"        : cls_name,
                    "behavior"     : "ABANDONED_OBJECT",
                    "risk_score"   : ab_risk,
                    "alert_level"  : ab_level,
                    "reasoning"    : ab_ev.get("reasoning", f"Unattended {category} ({cls_name})"),
                    "dwell_sec"    : ab_ev.get("stationary_time", 0.0),
                    "alert_type"   : "intrusion" if is_critical_zone else "presence",
                    "severity"     : "high" if ab_risk >= 75 else "medium",
                    "zone_id"      : ab_zone_id,
                    "zone_name"    : ab_zone_name,
                    "zone_type"    : effective_zone_type,
                }

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
        context_penalty: float = 0.0,
    ) -> float:
        """
        Compute the DESIRED risk score given current behaviour and context.
        The actual score may differ due to hysteresis and decay.
        """
        base = self.risk_base.get(behavior, 0.0)
        if base == 0.0 and in_restricted:
            base = self.risk_base.get("access", 40.0)

        # Zone multiplier and zone-specific night multiplier
        zone_mult = 1.0
        zone_night_mult = self.night_mult

        if zone_id:
            for zone in boundary_zones:
                if zone.get("id") == zone_id:
                    z_type    = zone.get("type", "restricted").lower()
                    zone_mult = ZONE_MULTIPLIER.get(z_type, 1.0)
                    if "night_multiplier" in zone:
                        zone_night_mult = float(zone["night_multiplier"])
                    break

        target = base * zone_mult + context_penalty

        # Loiter bonus: each second of dwell beyond the base adds more risk
        if behavior == "loiter" and dwell_sec > 0:
            target += dwell_sec * LOITER_BONUS_PER_SEC

        # Night multiplier
        if night_mode and target > 0:
            target *= zone_night_mult

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

        Decay: when behaviour is "none" and target_risk is 0, subtract DECAY_RATE/fps per frame.
        Risk never decays below 0.
        """
        if (behavior == "none" or behavior == "animal_presence") and target_risk <= 0.0:
            # Decay mode: slowly bring risk down
            decay_per_frame = self.decay_rate / fps
            new_risk = max(0.0, prev_risk - decay_per_frame)
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
