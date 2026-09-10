"""
zone_context.py - Zone Context Memory with Time-of-Day & Expected-Object Rules (Tier-2 DEV 1)
=============================================================================================
Smart India Hackathon 2026 | Problem: SIH26187
AMST Border-Net | DEV 1 module

Stores and queries contextual memory for all zones in the surveillance field:
  1. Zone ID, human-readable name, polygon in frame coordinates.
  2. Base zone type: restricted, secure, no_parking, allowed, parking, monitor.
  3. Expected/allowed object classes (e.g. person, vehicle, bag, device).
  4. Time-of-day rules (e.g. night/after-hours window):
       - Overrides zone type to a more restrictive level (e.g. parking -> no_parking).
       - Adjusts expected objects (e.g. after-hours: only security personnel).
  5. Point-in-polygon spatial queries returning the effective zone type, expected objects,
     and after-hours override status.
"""

import cv2
import numpy as np
import math
import logging
from datetime import datetime, time as dtime
from typing import List, Tuple, Dict, Any, Optional, Union

logger = logging.getLogger("amst_border_net.zone_context")

# Category class expansions
CLASS_EXPANSIONS: Dict[str, Tuple[str, ...]] = {
    "vehicle": ("car", "truck", "bus", "motorcycle", "bicycle", "van"),
    "bag": ("backpack", "handbag", "suitcase", "umbrella", "bag", "box"),
    "device": ("remote", "cell phone", "laptop", "tablet", "keyboard", "mouse", "tv"),
    "animal": ("bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "teddy bear"),
    "person": ("person",),
}


def parse_time_component(val: Union[int, str, dtime, datetime]) -> dtime:
    """Convert hour int (0-23) or string/time into datetime.time object."""
    if isinstance(val, int):
        h = max(0, min(23, val))
        return dtime(h, 0)
    if isinstance(val, datetime):
        return val.time()
    if isinstance(val, dtime):
        return val
    s = str(val).strip()
    for fmt in ("%H:%M", "%H:%M:%S", "%H"):
        try:
            return datetime.strptime(s, fmt).time()
        except ValueError:
            pass
    return dtime(12, 0)


def is_hour_in_window(
    current_time_or_hour: Optional[Union[int, str, dtime, datetime]],
    window_str: str,
) -> bool:
    """
    Check if a time/hour falls inside a window string 'HH:MM-HH:MM' or 'HH-HH'.
    Supports overnight wrapping (e.g. '22:00-06:00').
    """
    if not window_str or "-" not in window_str:
        return False

    if current_time_or_hour is None:
        now_t = datetime.now().time()
    elif isinstance(current_time_or_hour, int):
        now_t = dtime(max(0, min(23, current_time_or_hour)), 0)
    else:
        now_t = parse_time_component(current_time_or_hour)

    parts = window_str.split("-", 1)
    t_start = parse_time_component(parts[0].strip())
    t_end = parse_time_component(parts[1].strip())

    if t_start <= t_end:
        return t_start <= now_t <= t_end
    else:
        # Overnight e.g. 22:00 - 06:00
        return now_t >= t_start or now_t <= t_end


def point_in_polygon(x: int, y: int, polygon: List[Union[Tuple[int, int], List[int]]]) -> bool:
    """
    Ray-casting algorithm to test if point (x, y) is inside polygon.
    Pure Python with zero external dependencies.
    """
    n = len(polygon)
    if n < 3:
        return False

    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        x_inters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    else:
                        x_inters = p1x
                    if p1x == p2x or x <= x_inters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


class ZoneContextMemory:
    """
    Manages zone contextual rules, time-of-day dynamics, and expected objects.
    """

    DEFAULT_SAMPLE_ZONES = [
        {
            "id": "perimeter",
            "name": "Perimeter Wall",
            "base_type": "restricted",
            "polygon": [(0, 0), (1280, 0), (1280, 576), (0, 576)],
            "expected_objects": ["person"],
            "time_rules": [
                {
                    "time_window": "22:00-06:00",
                    "override_type": "secure",
                    "override_expected_objects": [],
                    "description": "Overnight strict lockdown"
                }
            ],
            "night_multiplier": 1.40,
        },
        {
            "id": "gate_approach",
            "name": "Gate Approach",
            "base_type": "monitor",
            "polygon": [(400, 450), (880, 450), (880, 650), (400, 650)],
            "expected_objects": ["person", "vehicle"],
            "time_rules": [
                {
                    "time_window": "20:00-06:00",
                    "override_type": "restricted",
                    "override_expected_objects": ["person"],
                    "description": "Gate closed after-hours"
                }
            ],
            "night_multiplier": 1.30,
        },
        {
            "id": "parking_area",
            "name": "Authorized Parking Area",
            "base_type": "parking",
            "polygon": [(50, 450), (380, 450), (380, 700), (50, 700)],
            "expected_objects": ["vehicle", "person"],
            "time_rules": [
                {
                    "time_window": "21:00-05:00",
                    "override_type": "no_parking",
                    "override_expected_objects": [],
                    "description": "No overnight unattended parking"
                }
            ],
            "night_multiplier": 1.25,
        },
        {
            "id": "restricted_corridor",
            "name": "Restricted Access Corridor",
            "base_type": "secure",
            "polygon": [(900, 300), (1250, 300), (1250, 550), (900, 550)],
            "expected_objects": ["person"],
            "time_rules": [
                {
                    "time_window": "19:00-07:00",
                    "override_type": "restricted",
                    "override_expected_objects": [],
                    "description": "Secure corridor locked down at night"
                }
            ],
            "night_multiplier": 1.45,
        },
        {
            "id": "approach_band",
            "name": "Outer Approach Band",
            "base_type": "monitor",
            "polygon": [(0, 576), (1280, 576), (1280, 720), (0, 720)],
            "expected_objects": ["person", "vehicle", "animal"],
            "time_rules": [],
            "night_multiplier": 1.15,
        },
    ]

    def __init__(
        self,
        zones: Optional[List[Dict[str, Any]]] = None,
        config_path: Optional[str] = None
    ):
        """Initialize Zone Context Memory with zones list, YAML config path, or sample defaults."""
        raw_zones = zones
        if raw_zones is None and config_path:
            import os
            import yaml
            if os.path.isfile(config_path):
                try:
                    with open(config_path, "r") as f:
                        cfg = yaml.safe_load(f) or {}
                    raw_zones = cfg.get("zones")
                except Exception as e:
                    logger.warning(f"Failed to load zones from {config_path}: {e}")

        if not raw_zones:
            raw_zones = self.DEFAULT_SAMPLE_ZONES

        self.zones: List[Dict[str, Any]] = []

        for z in raw_zones:
            zc = dict(z)
            raw_poly = zc.get("polygon", [])
            zc["polygon"] = np.array(raw_poly, dtype=np.int32)
            zc["base_type"] = str(zc.get("base_type") or zc.get("type", "monitor")).lower()
            zc["type"] = zc["base_type"]
            zc["name"] = zc.get("name") or zc.get("id", "Zone").replace("_", " ").title()

            # Ensure expected objects are lowercase strings
            exp_objs = zc.get("expected_objects") or zc.get("allowed_objects") or ["person"]
            zc["expected_objects"] = [str(o).lower() for o in exp_objs]
            zc["allowed_objects"] = zc["expected_objects"]

            zc["time_rules"] = zc.get("time_rules", [])
            zc["night_multiplier"] = float(zc.get("night_multiplier", 1.30))
            self.zones.append(zc)

        logger.info(f"ZoneContextMemory initialized with {len(self.zones)} zone(s).")

    def get_all_zones(self) -> List[Dict[str, Any]]:
        """Return list of all configured zones."""
        return list(self.zones)

    def get_zone_by_id(self, zone_id: str) -> Optional[Dict[str, Any]]:
        """Lookup zone definition by string ID."""
        for z in self.zones:
            if z.get("id") == zone_id:
                return z
        return None

    def scale_zones_to_frame(
        self,
        frame_w: int,
        frame_h: int,
        ref_res: Optional[Tuple[int, int]] = (1280, 720),
    ) -> None:
        """Scale all zone polygons from reference resolution to frame dimensions."""
        if frame_w <= 0 or frame_h <= 0:
            return
        self.frame_resolution = (frame_w, frame_h)
        if not ref_res or len(ref_res) != 2:
            ref_w, ref_h = 1280, 720
        else:
            ref_w, ref_h = ref_res

        if frame_w == ref_w and frame_h == ref_h:
            return

        sx = frame_w / ref_w
        sy = frame_h / ref_h

        for z in self.zones:
            scaled = []
            for pt in z["polygon"]:
                px, py = pt[0], pt[1]
                nx = max(0, min(frame_w - 1, int(round(px * sx))))
                ny = max(0, min(frame_h - 1, int(round(py * sy))))
                scaled.append((nx, ny))
            z["polygon"] = np.array(scaled, dtype=np.int32)

        logger.info(f"ZoneContextMemory: Scaled {len(self.zones)} zones to {frame_w}x{frame_h}.")

    def get_zone_at_point(
        self,
        x: Union[int, Tuple[int, int], List[int]],
        y: Optional[int] = None,
        current_time_or_hour: Optional[Union[int, str, dtime, datetime]] = None,
        current_time: Optional[Union[int, str, dtime, datetime]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Find which zone contains (x, y) and evaluate time-based rules.
        Supports passing a (x, y) tuple/list as first argument.
        """
        if isinstance(x, (tuple, list)):
            px, py = int(x[0]), int(x[1])
            t = y if current_time_or_hour is None else current_time_or_hour
        else:
            px, py = int(x), int(y if y is not None else 0)
            t = current_time_or_hour

        eval_time = current_time if current_time is not None else t

        matching_zones = []
        for zone in self.zones:
            poly = zone.get("polygon", [])
            if len(poly) < 3:
                continue

            if point_in_polygon(px, py, poly):
                area = cv2.contourArea(np.array(poly, dtype=np.int32))
                matching_zones.append((area, zone))

        if matching_zones:
            matching_zones.sort(key=lambda item: item[0])
            best_zone = matching_zones[0][1]
            return self.evaluate_zone_context(best_zone, eval_time)

        return None

    def evaluate_zone_context(
        self,
        zone: Union[str, Dict[str, Any]],
        current_time_or_hour: Optional[Union[int, str, dtime, datetime]] = None,
        current_time: Optional[Union[int, str, dtime, datetime]] = None,
    ) -> Dict[str, Any]:
        """
        Apply time-based rules to a zone and compute its effective state.
        Supports passing zone_id (str) or zone dict.
        """
        eval_time = current_time if current_time is not None else current_time_or_hour

        target_zone: Dict[str, Any]
        if isinstance(zone, str):
            found = self.get_zone_by_id(zone)
            if found:
                target_zone = found
            else:
                target_zone = {"id": zone, "name": zone, "base_type": "monitor", "type": "monitor", "polygon": []}
        else:
            target_zone = zone

        base_type = target_zone.get("base_type") or target_zone.get("type", "monitor")
        expected_objs = list(target_zone.get("expected_objects", []))
        time_rules = target_zone.get("time_rules", [])

        effective_type = base_type
        effective_expected = expected_objs
        override_active = False
        active_rule_desc = ""

        for rule in time_rules:
            window = rule.get("time_window") or rule.get("window")
            if window and is_hour_in_window(eval_time, window):
                override_active = True
                if "override_type" in rule:
                    effective_type = rule["override_type"].lower()
                if "override_expected_objects" in rule:
                    effective_expected = [str(o).lower() for o in rule["override_expected_objects"]]
                active_rule_desc = rule.get("description", "Time-based override active")
                break

        return {
            "id": target_zone.get("id", "zone"),
            "zone_id": target_zone.get("id", "zone"),
            "name": target_zone.get("name") or target_zone.get("id", "Zone"),
            "zone_name": target_zone.get("name") or target_zone.get("id", "Zone"),
            "polygon": target_zone.get("polygon", []),
            "base_type": base_type,
            "effective_type": effective_type,
            "zone_type": effective_type,
            "expected_objects": effective_expected,
            "is_override_active": override_active,
            "is_after_hours": override_active,
            "rule_description": active_rule_desc,
            "night_multiplier": float(target_zone.get("night_multiplier", 1.30)),
            "raw_zone": target_zone,
        }

    def is_object_expected(
        self,
        zone_or_class: str,
        expected_or_class: Optional[Union[List[str], str]] = None,
        current_time: Optional[Union[int, str, dtime, datetime]] = None,
    ) -> bool:
        """
        Check if an object class matches expected objects.
        Supports both:
          1. is_object_expected(zone_id, class_name, current_time=...)
          2. is_object_expected(class_name, expected_objects_list)
        """
        if isinstance(expected_or_class, (list, tuple)):
            # Signature: (object_class, expected_objects)
            cls_lower = str(zone_or_class).strip().lower()
            expected_objects = expected_or_class
        else:
            # Signature: (zone_id, object_class, current_time)
            zone_id = str(zone_or_class).strip()
            cls_lower = str(expected_or_class or "").strip().lower()
            zone_eval = self.evaluate_zone_context(zone_id, current_time=current_time)
            expected_objects = zone_eval.get("expected_objects", [])

        if not expected_objects:
            return False

        for expected in expected_objects:
            exp_lower = str(expected).strip().lower()
            if exp_lower == cls_lower:
                return True
            expanded = CLASS_EXPANSIONS.get(exp_lower, ())
            if cls_lower in expanded:
                return True

        return False

    def get_all_zones(self) -> List[Dict[str, Any]]:
        """Return list of all configured zone dictionaries."""
        return self.zones
