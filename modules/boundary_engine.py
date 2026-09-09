"""
boundary_engine.py - Restricted Zone Intrusion & Loitering Detection
=====================================================================
Smart India Hackathon 2026 | Problem: SIH26187
AMST Border-Net | DEV 1 module

This module provides pure-Python geometry functions for determining whether
a tracked person has entered (intrusion) or dwelled (loitering) inside a
configured restricted zone polygon.

Design goals:
  - Zero extra dependencies beyond OpenCV + NumPy (already installed).
  - Pure functions ΓÇö no state, no side effects ΓÇö easy to test in isolation.
  - Fully backward-compatible: main.py skips all logic gracefully if no
    config file exists.

Public API:
    point_in_polygon(x, y, polygon)                              -> bool
    check_intrusion(centroid, polygon)                           -> bool
    check_loitering(track_history, polygon, min_sec, fps)        -> bool
    compute_alert(centroid, track_history, zone, fps)            -> (str, str)
    classify_zone_position(centroid, frame_h, approach_ratio)    -> str
    draw_zones(frame, zones, alerts)                             -> np.ndarray
    draw_alert_box(frame, track, alert_type, severity)           -> None
    draw_legend(frame, has_zones)                                -> None
    scale_zones_to_frame(zones, w, h, ref_res)                   -> list

Author : DEV 1
Date   : 2026-09-08
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict, Any, Optional

logger = logging.getLogger("amst_border_net.boundary_engine")


# ---------------------------------------------------------------------------
# Alert / zone colour palette (BGR for OpenCV)
# ---------------------------------------------------------------------------
COLOR_INTRUSION = (0,   0,   255)   # Bright red  ΓÇö severity: high
COLOR_LOITERING = (0, 128,   255)   # Orange      ΓÇö severity: medium
COLOR_NORMAL    = (0, 255,   255)   # Cyan        ΓÇö no alert

# Zone border colours by type (idle state)
_COLOR_RESTRICTED_IDLE = (0,  60, 200)   # Dark red   ΓÇö restricted zone, no active alert
_COLOR_MONITOR_IDLE    = (0, 180, 220)   # Amber/gold ΓÇö monitor zone, no active alert

# Zone fill colours (active alert state)
_COLOR_FILL_INTRUSION  = (0,   0, 200)   # Bright red fill
_COLOR_FILL_LOITERING  = (0,  80, 200)   # Orange-red fill
_COLOR_FILL_RESTRICTED = (0,  40, 160)   # Muted dark-red fill (idle restricted)
_COLOR_FILL_MONITOR    = (0, 100, 140)   # Muted amber fill (idle monitor)

# Zone overlay fill opacity (0 = transparent, 1 = solid)
_ZONE_FILL_ALPHA = 0.18


# ===========================================================================
# Core geometry
# ===========================================================================

def point_in_polygon(
    x: int,
    y: int,
    polygon: List[Tuple[int, int]]
) -> bool:
    """
    Determine whether the point (x, y) lies inside the given polygon using
    the ray-casting algorithm.

    Works correctly for:
      - Convex and concave polygons
      - Any number of vertices (ΓëÑ 3)
      - Points on the edge (treated as inside)

    Args:
        x       : Pixel x-coordinate of the test point.
        y       : Pixel y-coordinate of the test point.
        polygon : Ordered list of (px, py) vertex tuples (pixel coordinates).
                  Vertices should be listed in either clockwise or
                  counter-clockwise order; the algorithm handles both.

    Returns:
        True if the point is inside (or on the boundary of) the polygon.

    Example:
        poly = [(100,100), (400,100), (400,400), (100,400)]
        point_in_polygon(250, 250, poly)   # ΓåÆ True
        point_in_polygon(50,  50,  poly)   # ΓåÆ False
    """
    n = len(polygon)
    if n < 3:
        return False

    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        # Check whether the ray from (x, y) going right crosses edge (i, j)
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside


def check_intrusion(
    centroid: Tuple[int, int],
    polygon:  List[Tuple[int, int]]
) -> bool:
    """
    Return True if the centroid is currently inside the restricted polygon.

    This is a real-time (single-frame) check ΓÇö it does not require history.

    Args:
        centroid : (cx, cy) pixel coordinate of the tracked person's centroid.
        polygon  : Restricted zone polygon as list of (x, y) tuples.

    Returns:
        True if the centroid is inside the polygon.
    """
    return point_in_polygon(centroid[0], centroid[1], polygon)


def check_loitering(
    track_history:    List[Dict[str, Any]],
    region_polygon:   List[Tuple[int, int]],
    min_duration_sec: float,
    fps:              float
) -> bool:
    """
    Return True if the track has been *continuously* inside the region
    polygon for at least min_duration_sec.

    The check is *continuous* (person must stay inside without leaving).
    A single frame outside the zone resets the consecutive counter.
    This is stricter than cumulative dwell time, which is appropriate for
    a fenced border perimeter where momentary presence is already a breach.

    Args:
        track_history    : Ordered list (oldest ΓåÆ newest) of history entries.
                           Each entry is a dict: {"xy": (cx, cy), "frame_id": int}
        region_polygon   : Restricted zone polygon (list of (x, y) tuples).
        min_duration_sec : Minimum continuous dwell time to trigger loitering.
        fps              : Current measured FPS, used to convert framesΓåÆseconds.
                           Use a fallback of 15.0 if FPS is zero/unknown.

    Returns:
        True if the track has been continuously inside for ΓëÑ min_duration_sec.

    Note:
        Requires at least (min_duration_sec ├ù fps) history entries.
        Short histories will never trigger loitering even if all frames are inside.
    """
    if fps <= 0:
        fps = 15.0
    if not track_history:
        return False

    min_frames = max(1, int(min_duration_sec * fps))

    # Walk backwards from the newest entry, counting consecutive inside frames.
    consecutive = 0
    for entry in reversed(track_history):
        cx, cy = entry["xy"]
        if point_in_polygon(cx, cy, region_polygon):
            consecutive += 1
        else:
            break   # Chain broken ΓÇö not continuously inside zone

    return consecutive >= min_frames


# ===========================================================================
# Border-focused alert computation (zone-type-aware)
# ===========================================================================

def compute_alert(
    track_centroid: Tuple[int, int],
    track_history:  List[Dict[str, Any]],
    zone:           Dict[str, Any],
    fps:            float,
) -> Tuple[str, str]:
    """
    Decide the alert type and severity for one (track, zone) pair.

    Respects zone type:

    type == "restricted":
        - Centroid inside polygon ΓåÆ ("intrusion", "high")
        - Centroid outside BUT continuous dwell >= min_loiter_seconds
          ΓåÆ ("loitering", "medium")  [person may have just stepped out]
        - Neither ΓåÆ ("none", "none")

    type == "monitor" (or any other value):
        - Presence alone is NOT an offence; no intrusion alert is raised.
        - Continuous dwell >= min_loiter_seconds ΓåÆ ("loitering", "medium")
        - Otherwise ΓåÆ ("none", "none")

    Default zone type (if absent): "restricted" ΓÇö fail safe for border use.

    Args:
        track_centroid : (cx, cy) pixel coordinate of the tracked person.
        track_history  : Ordered list of history dicts {"xy": (cx,cy), "frame_id": int}.
        zone           : Zone config dict with keys id, type, polygon, min_loiter_seconds.
        fps            : Current measured FPS (used to convert seconds ΓåÆ frames).

    Returns:
        (alert_type, severity)
            alert_type : "intrusion" | "loitering" | "none"
            severity   : "high"      | "medium"     | "none"
    """
    if fps <= 0:
        fps = 15.0

    zone_type = zone.get("type", "restricted").lower()
    polygon   = zone.get("polygon", [])
    min_sec   = zone.get("min_loiter_seconds", 5)

    if len(polygon) < 3:
        return ("none", "none")

    inside = check_intrusion(track_centroid, polygon)

    # ΓöÇΓöÇ restricted zone: immediate intrusion on centroid entry ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇ
    if zone_type == "restricted":
        if inside:
            return ("intrusion", "high")
        # Check loitering (person may have just stepped out after long dwell)
        if check_loitering(track_history, polygon, min_sec, fps):
            return ("loitering", "medium")
        return ("none", "none")

    # ΓöÇΓöÇ monitor zone: presence alone is fine; only loitering fires alert ΓöÇΓöÇ
    if check_loitering(track_history, polygon, min_sec, fps):
        return ("loitering", "medium")
    return ("none", "none")


# ===========================================================================
# Drawing helpers
# ===========================================================================

def draw_zones(
    frame:  np.ndarray,
    zones:  List[Dict[str, Any]],
    alerts: Dict[int, Dict[str, Any]]
) -> np.ndarray:
    """
    Draw zone polygon overlays on the frame.

    Each zone is drawn as:
      - A semi-transparent filled polygon (colour depends on alert state).
      - A solid polyline border (2px).
      - The zone ID label at the polygon centroid.

    Zone colours:
      - Bright red  : at least one tracked person is currently intruding.
      - Muted red   : zone is configured but no active intrusion.

    Args:
        frame  : BGR frame (modified in-place).
        zones  : List of zone config dicts (from boundary_config.yaml).
        alerts : {track_id: {"alert_type": str, "zone_id": str|None}}
                 Used to determine which zones are actively triggered.

    Returns:
        The annotated frame (same object as input, modified in-place).
    """
    # Collect zone IDs with active alerts
    intruded_zone_ids = {
        v["zone_id"]
        for v in alerts.values()
        if v.get("alert_type") == "intrusion" and v.get("zone_id")
    }
    loitering_zone_ids = {
        v["zone_id"]
        for v in alerts.values()
        if v.get("alert_type") == "loitering" and v.get("zone_id")
    }

    # We blend on a copy to achieve semi-transparency
    overlay = frame.copy()

    for zone in zones:
        raw_poly  = zone["polygon"]
        zone_type = zone.get("type", "restricted").lower()
        # Accept both list-of-lists and list-of-tuples
        pts = np.array([[int(p[0]), int(p[1])] for p in raw_poly], dtype=np.int32)
        zid = zone.get("id", "zone")

        # Choose zone colour based on alert state and zone type
        if zid in intruded_zone_ids:
            z_fill   = _COLOR_FILL_INTRUSION    # Bright red fill
            z_border = COLOR_INTRUSION
        elif zid in loitering_zone_ids:
            z_fill   = _COLOR_FILL_LOITERING    # Orange fill
            z_border = COLOR_LOITERING
        elif zone_type == "monitor":
            z_fill   = _COLOR_FILL_MONITOR      # Muted amber fill
            z_border = _COLOR_MONITOR_IDLE
        else:  # restricted idle
            z_fill   = _COLOR_FILL_RESTRICTED   # Muted dark-red fill
            z_border = _COLOR_RESTRICTED_IDLE

        # Fill overlay (drawn on the copy)
        cv2.fillPoly(overlay, [pts], z_fill)

        # Solid border on the original frame
        cv2.polylines(frame, [pts], isClosed=True, color=z_border, thickness=2)

        # Zone ID label ΓÇö placed at the polygon centroid
        cx = int(pts[:, 0].mean())
        cy = int(pts[:, 1].mean())

        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(zid, font, 0.52, 1)
        # Small dark background for the text
        cv2.rectangle(
            frame,
            (cx - tw // 2 - 4, cy - th - 4),
            (cx + tw // 2 + 4, cy + 4),
            (0, 0, 0), -1
        )
        cv2.putText(
            frame, zid,
            (cx - tw // 2, cy),
            font, 0.52,
            (255, 255, 255), 1, cv2.LINE_AA
        )

    # Blend the filled overlay onto the original frame
    cv2.addWeighted(overlay, _ZONE_FILL_ALPHA, frame, 1.0 - _ZONE_FILL_ALPHA, 0, frame)

    return frame


def draw_alert_box(
    frame:      np.ndarray,
    track:      Dict[str, Any],
    alert_type: str,
    severity:   str = "high",
) -> None:
    """
    Draw an alert-coloured bounding box and label over a tracked person.

    Called *after* detector.draw_detections() so the alert box overrides
    the default cyan person box.

    Color selection is driven by *severity*, not alert_type, so callers
    can pass any combination:
        severity "high"   ΓåÆ red box   (intrusion in restricted zone)
        severity "medium" ΓåÆ orange box (loitering in any zone)

    Args:
        frame      : BGR frame (modified in-place).
        track      : Track dict with keys "bbox" and "track_id".
        alert_type : "intrusion" or "loitering".
        severity   : "high" | "medium" (default "high").
    """
    if alert_type not in ("intrusion", "loitering"):
        return

    x1, y1, x2, y2 = [int(v) for v in track["bbox"]]
    tid = track["track_id"]

    if severity == "high" or alert_type == "intrusion":
        color = COLOR_INTRUSION
        label = f"INTRUSION - ID:{tid}"
    else:
        color = COLOR_LOITERING
        label = f"LOITERING - ID:{tid}"

    # Thick alert box (4px) to visually dominate the normal box underneath
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)

    # Flashing / solid corners (corner marks for extra visibility)
    corner_len = 18
    thick_c = 5
    # Top-left
    cv2.line(frame, (x1, y1), (x1 + corner_len, y1), color, thick_c)
    cv2.line(frame, (x1, y1), (x1, y1 + corner_len), color, thick_c)
    # Top-right
    cv2.line(frame, (x2, y1), (x2 - corner_len, y1), color, thick_c)
    cv2.line(frame, (x2, y1), (x2, y1 + corner_len), color, thick_c)
    # Bottom-left
    cv2.line(frame, (x1, y2), (x1 + corner_len, y2), color, thick_c)
    cv2.line(frame, (x1, y2), (x1, y2 - corner_len), color, thick_c)
    # Bottom-right
    cv2.line(frame, (x2, y2), (x2 - corner_len, y2), color, thick_c)
    cv2.line(frame, (x2, y2), (x2, y2 - corner_len), color, thick_c)

    # Label: coloured background + white text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.62
    font_thick = 2
    (tw, th), bl = cv2.getTextSize(label, font, font_scale, font_thick)

    label_x1 = x1
    label_y1 = max(0, y1 - th - bl - 8)
    label_x2 = x1 + tw + 10
    label_y2 = y1

    cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), color, -1)
    cv2.putText(
        frame, label,
        (label_x1 + 5, label_y2 - bl - 2),
        font, font_scale,
        (255, 255, 255),   # White text on coloured background
        font_thick, cv2.LINE_AA
    )


def draw_legend(frame: np.ndarray, has_zones: bool) -> None:
    """
    Draw a small legend in the bottom-left corner of the frame.

    Only drawn when at least one zone is configured (has_zones=True).

    Legend rows (bottom to top):
      ≡ƒƒÑ Intrusion   (red)
      ≡ƒƒº Loitering   (orange)
      ≡ƒƒª Normal      (cyan)

    Args:
        frame     : BGR frame (modified in-place).
        has_zones : If False, legend is skipped entirely.
    """
    if not has_zones:
        return

    h = frame.shape[0]
    items = [
        ("Intrusion ΓÇô Restricted zone entry",  COLOR_INTRUSION),
        ("Loitering ΓÇô Extended dwell time",    COLOR_LOITERING),
        ("Normal    ΓÇô No alert",               COLOR_NORMAL),
    ]

    # Title
    title = "ALERT LEGEND"
    font = cv2.FONT_HERSHEY_SIMPLEX
    (ttw, tth), _ = cv2.getTextSize(title, font, 0.42, 1)

    pad_x = 10
    row_h = 22
    box_w = ttw + 24
    total_h = tth + 8 + len(items) * row_h + 8

    # Background panel
    panel_y1 = h - total_h - 10
    panel_y2 = h - 10
    cv2.rectangle(
        frame,
        (pad_x - 4, panel_y1 - 4),
        (pad_x + box_w, panel_y2 + 4),
        (20, 20, 20), -1
    )
    cv2.rectangle(
        frame,
        (pad_x - 4, panel_y1 - 4),
        (pad_x + box_w, panel_y2 + 4),
        (80, 80, 80), 1
    )

    # Title text
    cv2.putText(
        frame, title,
        (pad_x, panel_y1 + tth),
        font, 0.42,
        (200, 200, 200), 1, cv2.LINE_AA
    )

    # Colour rows
    y = panel_y1 + tth + 8
    for label, color in items:
        swatch_y = y
        cv2.rectangle(frame, (pad_x, swatch_y), (pad_x + 14, swatch_y + 14), color, -1)
        cv2.putText(
            frame, label,
            (pad_x + 18, swatch_y + 12),
            font, 0.44,
            (210, 210, 210), 1, cv2.LINE_AA
        )
        y += row_h


# ===========================================================================
# Resolution Auto-Scaling
# ===========================================================================

def scale_zones_to_frame(
    zones: List[Dict[str, Any]],
    frame_width: int,
    frame_height: int,
    reference_resolution: Optional[List[int]] = None
) -> List[Dict[str, Any]]:
    """
    Scale zone polygons to match the active video or webcam frame resolution.

    Handles:
      1. Reference resolution scaling: If reference_resolution is provided (e.g. [1280, 720]),
         vertices are mapped proportionally:
             x' = round(x * frame_width / ref_w)
             y' = round(y * frame_height / ref_h)
      2. Normalized coordinates: If all coordinates in a polygon are in [0.0, 1.0],
         they are multiplied by frame_width and frame_height.
      3. Boundary clamping: Coordinates are clamped to [0, frame_width - 1] and [0, frame_height - 1].

    Args:
        zones: List of zone dictionaries, each containing a 'polygon' key.
        frame_width: Active video frame width in pixels.
        frame_height: Active video frame height in pixels.
        reference_resolution: Optional [ref_w, ref_h] list/tuple of reference dimensions.

    Returns:
        New list of zone dictionaries with scaled polygon coordinates as (x, y) tuples.
    """
    if frame_width <= 0 or frame_height <= 0 or not zones:
        return zones

    ref_w, ref_h = None, None
    if reference_resolution and len(reference_resolution) == 2:
        try:
            ref_w = float(reference_resolution[0])
            ref_h = float(reference_resolution[1])
        except (ValueError, TypeError):
            ref_w, ref_h = None, None

    scaled_zones = []
    for z in zones:
        poly = z.get("polygon", [])
        if not poly:
            scaled_zones.append(dict(z))
            continue

        # Check if coordinates are normalized [0.0, 1.0]
        is_normalized = (
            all(0.0 <= float(p[0]) <= 1.0 and 0.0 <= float(p[1]) <= 1.0 for p in poly)
            and any(float(p[0]) > 0.0 for p in poly)
        )

        scaled_poly: List[Tuple[int, int]] = []
        for pt in poly:
            px, py = float(pt[0]), float(pt[1])
            if is_normalized:
                sx = int(round(px * frame_width))
                sy = int(round(py * frame_height))
            elif ref_w and ref_h and ref_w > 0 and ref_h > 0:
                sx = int(round(px * (frame_width / ref_w)))
                sy = int(round(py * (frame_height / ref_h)))
            else:
                sx = int(round(px))
                sy = int(round(py))

            # Clamp within valid frame boundaries
            sx = max(0, min(sx, frame_width - 1))
            sy = max(0, min(sy, frame_height - 1))
            scaled_poly.append((sx, sy))

        new_z = dict(z)
        new_z["polygon"] = scaled_poly
        scaled_zones.append(new_z)

    return scaled_zones


# ===========================================================================
# Zone Position Helper  (used by RiskEngine for behaviour classification)
# ===========================================================================

def classify_zone_position(
    centroid: Tuple[int, int],
    frame_h: int,
    approach_ratio: float = 0.20,
) -> str:
    """
    Fast geometric pre-check: where in the frame does the centroid sit?

    Spatial model (top ΓåÆ bottom):
        "interior"  ΓåÆ upper (1 - approach_ratio) of the frame.
                       Person is deep inside the surveillance zone.
        "approach"  ΓåÆ bottom approach_ratio fraction of the frame.
                       Person is walking toward the camera / entering the scene.

    This is a *frame-coordinate* check only ΓÇö it does not consult zone
    polygons.  The polygon check is done separately in check_intrusion().
    This function is intentionally simple and O(1).

    Args:
        centroid       : (cx, cy) pixel coordinate of the tracked centroid.
        frame_h        : Frame height in pixels.
        approach_ratio : Fraction of frame height for the approach band.
                         Default 0.20 (bottom 20 %).  Must be in (0, 1).

    Returns:
        "approach" if cy >= frame_h * (1 - approach_ratio),
        "interior" otherwise.

    Example:
        # Frame is 720px tall; approach band is bottom 20% (y >= 576)
        classify_zone_position((320, 600), 720, 0.20)  # -> "approach"
        classify_zone_position((320, 300), 720, 0.20)  # -> "interior"
    """
    if frame_h <= 0:
        return "interior"

    approach_y_start = int(frame_h * (1.0 - max(0.0, min(1.0, approach_ratio))))
    _, cy = centroid

    if cy >= approach_y_start:
        return "approach"
    return "interior"
