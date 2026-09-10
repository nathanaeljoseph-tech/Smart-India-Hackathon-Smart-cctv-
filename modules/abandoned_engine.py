"""
abandoned_engine.py - Multi-Category Abandoned Object Engine (Tier-2 DEV 1)
============================================================================
Smart India Hackathon 2026 | Problem: SIH26187
AMST Border-Net | DEV 1 module

Detects unattended / abandoned objects across three broad categories:
  1. Bags: backpack, handbag, suitcase, umbrella, bag, box
  2. Vehicles: car, truck, bus, motorcycle, bicycle, van
  3. Small devices: remote, cell phone, laptop, tablet, keyboard, mouse, tv

Key Capabilities:
  - Multi-category classification with category-specific dwell thresholds.
  - Integration with ZoneContextMemory:
      * Faster alerts in restricted/secure/no_parking zones.
      * Normal parking in designated parking areas (unless night override triggers no_parking).
      * Stricter dwell time threshold multiplier at night/after-hours.
  - Strict Human-Only Ownership:
      * Only human tracks (is_person=True, is_animal=False) can own an object.
      * Animals are strictly ignored and never prevent an abandonment alert.
      * Continuous attendance tracking prevents false alerts when owner stays nearby.
"""

import math
import logging
from typing import List, Dict, Any, Optional, Tuple, Union

logger = logging.getLogger("amst_border_net.abandoned_engine")

# Categories definition
CATEGORY_MAP: Dict[str, Tuple[str, ...]] = {
    "bag": (
        "backpack", "handbag", "suitcase", "umbrella", "bag", "box"
    ),
    "vehicle": (
        "car", "truck", "bus", "motorcycle", "bicycle", "van"
    ),
    "device": (
        "remote", "cell phone", "laptop", "tablet", "keyboard", "mouse", "tv"
    ),
}

# Inverted lookup for fast matching: class_name -> category
CLASS_TO_CATEGORY: Dict[str, str] = {}
for cat, classes in CATEGORY_MAP.items():
    for c in classes:
        CLASS_TO_CATEGORY[c] = cat

# COCO ID fallback mapping
COCO_CLASS_ID_TO_CATEGORY: Dict[int, str] = {
    # Bags
    24: "bag",      # backpack
    25: "bag",      # umbrella
    26: "bag",      # handbag
    28: "bag",      # suitcase
    # Vehicles
    1: "vehicle",   # bicycle
    2: "vehicle",   # car
    3: "vehicle",   # motorcycle
    5: "vehicle",   # bus
    7: "vehicle",   # truck
    # Devices
    62: "device",   # tv
    63: "device",   # laptop
    64: "device",   # mouse
    65: "device",   # remote
    66: "device",   # keyboard
    67: "device",   # cell phone
}


def categorize_object(class_name_or_id: Union[str, int]) -> Optional[str]:
    """
    Categorize an object class or class ID into 'bag', 'vehicle', 'device', or None.
    """
    if isinstance(class_name_or_id, int):
        return COCO_CLASS_ID_TO_CATEGORY.get(class_name_or_id)

    name_clean = str(class_name_or_id).strip().lower()
    cat = CLASS_TO_CATEGORY.get(name_clean)
    if cat:
        return cat

    # Partial / substring match fallback
    if any(k in name_clean for k in ("bag", "pack", "case", "box", "luggage")):
        return "bag"
    if any(k in name_clean for k in ("car", "truck", "bus", "bike", "motorcycle", "vehicle", "van")):
        return "vehicle"
    if any(k in name_clean for k in ("phone", "laptop", "tablet", "remote", "device")):
        return "device"

    return None


def _bbox_iou(box1: List[int], box2: List[int]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def _euclidean_dist(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """Euclidean distance between two 2D points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


class AbandonedObjectEngine:
    """
    Multi-category stateful analyzer for abandoned luggage, vehicles, and devices.
    """

    def __init__(
        self,
        stationary_threshold: float = 10.0,             # Base seconds for bags
        stationary_threshold_vehicle: float = 15.0,     # Base seconds for vehicles
        stationary_threshold_device: float = 12.0,      # Base seconds for devices
        owner_distance_threshold: float = 120.0,        # Pixels separation for departed owner
        owner_absent_threshold: float = 5.0,            # Seconds owner absent before triggering
        max_movement_px: float = 28.0,                  # Jitter tolerance for stationary state
        iou_match_threshold: float = 0.25,              # IoU to match object across frames
        max_lost_seconds: float = 3.5,                  # Seconds before dropping track on occlusion
        night_threshold_multiplier: float = 0.70,       # Stricter dwell time at night
        config_path: Optional[str] = None,
    ):
        if config_path:
            import os
            import yaml
            if os.path.isfile(config_path):
                try:
                    with open(config_path, "r") as f:
                        cfg = yaml.safe_load(f) or {}
                    acfg = cfg.get("abandoned", {})
                    stationary_threshold = acfg.get("stationary_threshold", stationary_threshold)
                    stationary_threshold_vehicle = acfg.get("stationary_threshold_vehicle", stationary_threshold_vehicle)
                    stationary_threshold_device = acfg.get("stationary_threshold_device", stationary_threshold_device)
                    owner_distance_threshold = acfg.get("owner_distance_threshold", owner_distance_threshold)
                    owner_absent_threshold = acfg.get("owner_absent_threshold", owner_absent_threshold)
                    night_threshold_multiplier = acfg.get("night_threshold_multiplier", night_threshold_multiplier)
                except Exception as e:
                    logger.warning(f"Failed to load abandoned config from {config_path}: {e}")

        self.stationary_threshold_bag = float(stationary_threshold)
        self.stationary_threshold_vehicle = float(stationary_threshold_vehicle)
        self.stationary_threshold_device = float(stationary_threshold_device)
        # Keep backward-compatible attribute
        self.stationary_threshold = self.stationary_threshold_bag

        self.owner_distance_threshold = float(owner_distance_threshold)
        self.owner_absent_threshold = float(owner_absent_threshold)
        self.max_movement_px = float(max_movement_px)
        self.iou_match_threshold = float(iou_match_threshold)
        self.max_lost_seconds = float(max_lost_seconds)
        self.night_threshold_multiplier = float(night_threshold_multiplier)

        self.next_object_id = 1
        self.tracked_objects: Dict[int, Dict[str, Any]] = {}

        logger.info(
            f"AbandonedObjectEngine ready | bag_thresh={self.stationary_threshold_bag}s | "
            f"vehicle_thresh={self.stationary_threshold_vehicle}s | "
            f"device_thresh={self.stationary_threshold_device}s | "
            f"owner_dist={self.owner_distance_threshold}px"
        )

    def categorize_object(self, class_name_or_id: Union[str, int]) -> Optional[str]:
        """Categorize object into 'bag', 'vehicle', 'device', or None."""
        return categorize_object(class_name_or_id)

    def reset(self) -> None:
        """Reset all tracked objects."""
        self.tracked_objects.clear()
        self.next_object_id = 1
        logger.info("AbandonedObjectEngine reset.")

    def update(
        self,
        detections: Optional[List[Dict[str, Any]]] = None,
        person_tracks: Optional[List[Dict[str, Any]]] = None,
        fps: Optional[float] = None,
        zone_context_mgr: Optional[Any] = None,
        current_time: Optional[Any] = None,
        is_night: bool = False,
        object_tracks: Optional[List[Dict[str, Any]]] = None,
        frame_dt: Optional[float] = None,
        is_night_mode: Optional[bool] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Update object tracks and detect abandoned objects.
        Supports both YOLODetector outputs and custom track dicts.
        """
        if frame_dt is not None and frame_dt > 0:
            dt = float(frame_dt)
        else:
            safe_fps = fps if fps and fps > 0 else 15.0
            dt = 1.0 / safe_fps

        if is_night_mode is not None:
            is_night = bool(is_night_mode)

        raw_dets = detections if detections is not None else (object_tracks or [])
        raw_persons = person_tracks or []

        # 1. STRICT HUMAN-ONLY OWNER FILTERING:
        # Animals must NEVER count as owners.
        valid_human_tracks = [
            pt for pt in raw_persons
            if pt.get("is_person") and not pt.get("is_animal")
        ]

        active_person_lookup: Dict[int, Tuple[int, int]] = {}
        for pt in valid_human_tracks:
            box = pt.get("bbox") or pt.get("box") or [0, 0, 0, 0]
            pc = pt.get("ema_centroid") or pt.get("center") or [
                (box[0] + box[2]) // 2,
                (box[1] + box[3]) // 2,
            ]
            tid = pt.get("track_id", pt.get("id", 0))
            active_person_lookup[tid] = (int(pc[0]), int(pc[1]))

        # 2. Filter candidate objects for the 3 categories
        obj_candidates: List[Dict[str, Any]] = []
        for d in raw_dets:
            # Exclude humans and animals
            if d.get("is_person") or d.get("is_animal"):
                continue

            cls_name = str(d.get("class") or d.get("class_name") or "").lower()
            category = categorize_object(cls_name) or categorize_object(d.get("class_id", -1))
            if category is not None:
                d_copy = dict(d)
                d_copy["category"] = category
                d_copy["class"] = cls_name
                d_copy["class_name"] = cls_name
                if "bbox" not in d_copy and "box" in d_copy:
                    d_copy["bbox"] = d_copy["box"]
                if "center" in d_copy and "centroid" not in d_copy:
                    d_copy["centroid"] = d_copy["center"]
                if "confidence" in d_copy and "conf" not in d_copy:
                    d_copy["conf"] = d_copy["confidence"]
                obj_candidates.append(d_copy)

        # 3. Match candidate detections with existing tracked objects
        matched_obj_ids = set()
        matched_det_indices = set()

        for det_idx, det in enumerate(obj_candidates):
            det_box = det["bbox"]
            det_cx = (det_box[0] + det_box[2]) // 2
            det_cy = (det_box[1] + det_box[3]) // 2
            det_centroid = (det_cx, det_cy)
            det_cat = det["category"]

            best_match_id = None
            best_score = 0.0

            for obj_id, track in self.tracked_objects.items():
                if obj_id in matched_obj_ids:
                    continue
                # Only match same category
                if track.get("category") != det_cat:
                    continue

                iou = _bbox_iou(det_box, track["bbox"])
                dist = _euclidean_dist(det_centroid, track["centroid"])

                # Vehicles have larger tolerance for distance
                tol = self.max_movement_px * (1.6 if det_cat == "vehicle" else 1.0)

                if iou >= self.iou_match_threshold or dist < tol:
                    score = iou * 100.0 + (100.0 - min(dist, 100.0))
                    if score > best_score:
                        best_score = score
                        best_match_id = obj_id

            if best_match_id is not None:
                matched_obj_ids.add(best_match_id)
                matched_det_indices.add(det_idx)
                track = self.tracked_objects[best_match_id]

                # Check stationary status from anchor centroid
                dist_from_anchor = _euclidean_dist(det_centroid, track["anchor_centroid"])
                tol = self.max_movement_px * (1.5 if det_cat == "vehicle" else 1.0)

                if dist_from_anchor <= tol:
                    track["stationary_time"] += dt
                else:
                    track["anchor_centroid"] = det_centroid
                    track["stationary_time"] = 0.0

                track["bbox"] = det_box
                track["centroid"] = det_centroid
                track["confidence"] = det.get("confidence", 0.8)
                track["lost_time"] = 0.0
                track["total_age"] += dt

        # 4. Create new tracks for unmatched detections
        for det_idx, det in enumerate(obj_candidates):
            if det_idx in matched_det_indices:
                continue

            det_box = det["bbox"]
            det_cx = (det_box[0] + det_box[2]) // 2
            det_cy = (det_box[1] + det_box[3]) // 2
            det_centroid = (det_cx, det_cy)
            det_cat = det["category"]

            new_id = self.next_object_id
            self.next_object_id += 1

            # Find nearest human owner at inception
            initial_owner_id = None
            min_dist = float("inf")
            for pid, pc in active_person_lookup.items():
                d = _euclidean_dist(det_centroid, pc)
                if d < min_dist and d <= self.owner_distance_threshold:
                    min_dist = d
                    initial_owner_id = pid

            orig_tid = det.get("track_id")
            self.tracked_objects[new_id] = {
                "object_id": new_id,
                "orig_track_id": orig_tid,
                "category": det_cat,
                "class": det.get("class", "object"),
                "bbox": det_box,
                "centroid": det_centroid,
                "anchor_centroid": det_centroid,
                "confidence": det.get("confidence", 0.8),
                "stationary_time": 0.0,
                "owner_id": initial_owner_id,
                "owner_distance": min_dist if initial_owner_id is not None else 9999.0,
                "owner_in_frame": (initial_owner_id is not None),
                "owner_absent_time": 0.0 if initial_owner_id is not None else self.owner_absent_threshold,
                "lost_time": 0.0,
                "total_age": 0.0,
                "is_abandoned": False,
            }

        # 5. Age and prune lost tracks
        to_remove = []
        for obj_id, track in self.tracked_objects.items():
            if obj_id not in matched_obj_ids:
                track["lost_time"] += dt
                if track["lost_time"] > self.max_lost_seconds:
                    to_remove.append(obj_id)

        for obj_id in to_remove:
            del self.tracked_objects[obj_id]

        # 6. Evaluate Owner Separation, Zone Context & Abandonment
        abandoned_events: List[Dict[str, Any]] = []

        for obj_id, track in self.tracked_objects.items():
            if track["lost_time"] > 0:
                continue

            owner_id = track.get("owner_id")
            centroid = track["centroid"]
            cx, cy = centroid
            category = track.get("category", "bag")

            # Update owner distance using valid human tracks only
            if owner_id is None:
                min_dist = float("inf")
                closest_pid = None
                for pid, pc in active_person_lookup.items():
                    d = _euclidean_dist(centroid, pc)
                    if d < min_dist:
                        min_dist = d
                        closest_pid = pid

                if closest_pid is not None and min_dist <= self.owner_distance_threshold:
                    track["owner_id"] = closest_pid
                    owner_id = closest_pid
                    track["owner_distance"] = min_dist
                    track["owner_in_frame"] = True
                    track["owner_absent_time"] = 0.0
                else:
                    track["owner_distance"] = min_dist if min_dist != float("inf") else 9999.0
                    track["owner_in_frame"] = False
                    track["owner_absent_time"] += dt
            else:
                if owner_id in active_person_lookup:
                    owner_pos = active_person_lookup[owner_id]
                    dist = _euclidean_dist(centroid, owner_pos)
                    track["owner_distance"] = dist
                    track["owner_in_frame"] = True

                    if dist <= self.owner_distance_threshold:
                        track["owner_absent_time"] = 0.0
                    else:
                        track["owner_absent_time"] += dt
                else:
                    track["owner_in_frame"] = False
                    track["owner_distance"] = 9999.0
                    track["owner_absent_time"] += dt

            # 7. Compute Dynamic Zone-Aware & Night Dwell Thresholds
            base_thresh = self.stationary_threshold_bag
            if category == "vehicle":
                base_thresh = self.stationary_threshold_vehicle
            elif category == "device":
                base_thresh = self.stationary_threshold_device

            effective_zone_type = "none"
            zone_id = None
            zone_name = "Unassigned Zone"
            is_override_active = False

            if zone_context_mgr is not None:
                z_ctx = zone_context_mgr.get_zone_at_point(cx, cy, current_time)
                if z_ctx is not None:
                    zone_id = z_ctx.get("zone_id")
                    zone_name = z_ctx.get("zone_name", zone_id)
                    effective_zone_type = z_ctx.get("effective_type", "monitor").lower()
                    is_override_active = z_ctx.get("is_override_active", False)

            # Zone-type multiplier adjustments:
            zone_thresh_mult = 1.0
            if effective_zone_type in ("restricted", "secure"):
                # Stricter in restricted/secure zones
                zone_thresh_mult = 0.70
            elif effective_zone_type == "no_parking":
                # Vehicles in no_parking zone get immediate strict threshold
                zone_thresh_mult = 0.50 if category == "vehicle" else 0.70
            elif effective_zone_type == "parking":
                if category == "vehicle":
                    # Legitimate parking area during authorized hours does not alert
                    zone_thresh_mult = 999.0
                else:
                    zone_thresh_mult = 1.20
            elif effective_zone_type in ("monitor", "allowed"):
                zone_thresh_mult = 1.20

            # Night multiplier adjustment (faster dwell requirement at night)
            night_flag = bool(is_night or is_override_active)
            night_mult = self.night_threshold_multiplier if night_flag else 1.0

            effective_threshold = max(2.0, base_thresh * zone_thresh_mult * night_mult)

            # 8. Check Abandoned Conditions
            is_stationary = (track["stationary_time"] >= effective_threshold)
            owner_away = (
                (track["owner_distance"] > self.owner_distance_threshold)
                or (not track["owner_in_frame"])
            )
            absent_long_enough = (track["owner_absent_time"] >= self.owner_absent_threshold)

            # If vehicle in authorized parking, suppress abandonment
            if effective_zone_type == "parking" and category == "vehicle" and not night_flag:
                is_stationary = False

            if is_stationary and owner_away and absent_long_enough:
                track["is_abandoned"] = True
                owner_desc = "left frame" if not track["owner_in_frame"] else f"{track['owner_distance']:.0f}px away"
                night_str = " [NIGHT]" if night_flag else ""

                reasoning = (
                    f"Abandoned {category} ({track['class']}) stationary {track['stationary_time']:.0f}s "
                    f"in {effective_zone_type} zone '{zone_name}' (owner {owner_desc}){night_str}"
                )

                orig_tid = track.get("orig_track_id")
                event_tid = orig_tid if orig_tid is not None else f"obj_{obj_id}"

                event = {
                    "event_type": "ABANDONED_OBJECT",
                    "object_id": obj_id,
                    "track_id": event_tid,
                    "category": category,
                    "class": track["class"],
                    "class_name": track["class"],
                    "bbox": track["bbox"],
                    "box": track["bbox"],
                    "centroid": track["centroid"],
                    "center": track["centroid"],
                    "stationary_time": round(track["stationary_time"], 1),
                    "dwell_time": round(track["stationary_time"], 1),
                    "dwell_sec": round(track["stationary_time"], 1),
                    "threshold": round(effective_threshold, 1),
                    "owner_id": track["owner_id"],
                    "owner_track_id": track["owner_id"],
                    "owner_distance": round(track["owner_distance"], 1),
                    "owner_in_frame": track["owner_in_frame"],
                    "owner_absent_time": round(track["owner_absent_time"], 1),
                    "zone_id": zone_id,
                    "zone_name": zone_name,
                    "zone_type": effective_zone_type,
                    "is_night": night_flag,
                    "reasoning": reasoning,
                    "confidence": track.get("confidence", 0.85),
                }
                abandoned_events.append(event)
            else:
                track["is_abandoned"] = False

        return abandoned_events

    def get_tracked_objects(self) -> List[Dict[str, Any]]:
        """Return list of all actively tracked objects."""
        return [
            dict(track)
            for track in self.tracked_objects.values()
            if track["lost_time"] == 0
        ]
