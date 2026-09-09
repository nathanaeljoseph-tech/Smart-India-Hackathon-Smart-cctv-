"""
detector.py - YOLO11 Multi-Class Object Detection Module (DEV 1)
================================================================
Smart India Hackathon 2026 | Problem: SIH26187
Team Role: DEV 1 - Camera + YOLO Detection + Tracking

UPDATED: Now detects ALL 80 COCO classes with proper labels.
People (class 0) are highlighted as the PRIMARY target.
All other objects are labeled with their correct COCO names.

Design:
  - People  ΓåÆ Cyan box, bold label, "PERSON" prefix
  - VehiclesΓåÆ Yellow box (car, truck, bus, motorcycle)
  - Bags    ΓåÆ Orange box (backpack, handbag, suitcase)
  - Animals ΓåÆ Purple box
  - Other   ΓåÆ Gray box

Author : DEV 1
Date   : 2026-09-05
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple

try:
    import torch
except ImportError:
    torch = None

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError(
        "Ultralytics is not installed. Run: pip install ultralytics"
    )

# ---------------------------------------------------------------------------
# COCO Class Names (all 80 classes)
# Index = COCO class ID, value = human-readable label
# ---------------------------------------------------------------------------
COCO_CLASSES = {
    0: "person",        1: "bicycle",       2: "car",
    3: "motorcycle",    4: "airplane",      5: "bus",
    6: "train",         7: "truck",         8: "boat",
    9: "traffic light", 10: "fire hydrant", 11: "stop sign",
    12: "parking meter",13: "bench",        14: "bird",
    15: "cat",          16: "dog",          17: "horse",
    18: "sheep",        19: "cow",          20: "elephant",
    21: "bear",         22: "zebra",        23: "giraffe",
    24: "backpack",     25: "umbrella",     26: "handbag",
    27: "tie",          28: "suitcase",     29: "frisbee",
    30: "skis",         31: "snowboard",    32: "sports ball",
    33: "kite",         34: "baseball bat", 35: "baseball glove",
    36: "skateboard",   37: "surfboard",    38: "tennis racket",
    39: "bottle",       40: "wine glass",   41: "cup",
    42: "fork",         43: "knife",        44: "spoon",
    45: "bowl",         46: "banana",       47: "apple",
    48: "sandwich",     49: "orange",       50: "broccoli",
    51: "carrot",       52: "hot dog",      53: "pizza",
    54: "donut",        55: "cake",         56: "chair",
    57: "couch",        58: "potted plant", 59: "bed",
    60: "dining table", 61: "toilet",       62: "tv",
    63: "laptop",       64: "mouse",        65: "remote",
    66: "keyboard",     67: "cell phone",   68: "microwave",
    69: "oven",         70: "toaster",      71: "sink",
    72: "refrigerator", 73: "book",         74: "clock",
    75: "vase",         76: "scissors",     77: "teddy bear",
    78: "hair drier",   79: "toothbrush",
}

# ---------------------------------------------------------------------------
# Category Groups ΓåÆ Box Colors (BGR format for OpenCV)
# People are CYAN (primary target), others use category-specific colors
# ---------------------------------------------------------------------------
CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {}

# Helper: assign colors by category
_PERSON    = (0, 255, 255)      # Cyan   ΓÇö primary surveillance target
_VEHICLE   = (0, 220, 255)      # Yellow ΓÇö vehicles
_ANIMAL    = (180, 0, 255)      # Purple ΓÇö animals
_BAG       = (0, 140, 255)      # Orange ΓÇö bags/luggage (security relevant)
_WEAPON    = (0, 0, 255)        # Red    ΓÇö weapons/knives (security relevant)
_FOOD      = (0, 200, 80)       # Green  ΓÇö food items
_SPORTS    = (255, 100, 0)      # Blue   ΓÇö sports equipment
_FURNITURE = (150, 150, 150)    # Gray   ΓÇö furniture
_TECH      = (255, 200, 0)      # Teal   ΓÇö electronics
_DEFAULT   = (200, 200, 200)    # Light gray ΓÇö everything else

_CATEGORY_MAP = {
    # Person (primary)
    0: _PERSON,
    # Vehicles
    1: _VEHICLE, 2: _VEHICLE, 3: _VEHICLE, 4: _VEHICLE,
    5: _VEHICLE, 6: _VEHICLE, 7: _VEHICLE, 8: _VEHICLE,
    # Animals
    14: _ANIMAL, 15: _ANIMAL, 16: _ANIMAL, 17: _ANIMAL,
    18: _ANIMAL, 19: _ANIMAL, 20: _ANIMAL, 21: _ANIMAL,
    22: _ANIMAL, 23: _ANIMAL,
    # Bags / Luggage (security relevant)
    24: _BAG, 26: _BAG, 28: _BAG, 25: _BAG,
    # Weapons / Dangerous items
    43: _WEAPON, 44: _WEAPON, 76: _WEAPON,  # knife, spoon, scissors
    # Food
    46: _FOOD, 47: _FOOD, 48: _FOOD, 49: _FOOD, 50: _FOOD,
    51: _FOOD, 52: _FOOD, 53: _FOOD, 54: _FOOD, 55: _FOOD,
    39: _FOOD, 40: _FOOD, 41: _FOOD, 45: _FOOD,
    # Sports
    29: _SPORTS, 30: _SPORTS, 31: _SPORTS, 32: _SPORTS,
    33: _SPORTS, 34: _SPORTS, 35: _SPORTS, 36: _SPORTS,
    37: _SPORTS, 38: _SPORTS,
    # Furniture
    13: _FURNITURE, 56: _FURNITURE, 57: _FURNITURE,
    59: _FURNITURE, 60: _FURNITURE,
    # Electronics / Tech
    62: _TECH, 63: _TECH, 64: _TECH, 65: _TECH,
    66: _TECH, 67: _TECH,
}

# Build full class color map (fallback to _DEFAULT)
for cid in COCO_CLASSES:
    CLASS_COLORS[cid] = _CATEGORY_MAP.get(cid, _DEFAULT)

# Person class constant
PERSON_CLASS_ID = 0


class YOLODetector:
    """
    Wraps YOLO11 for multi-class object detection.

    - Detects ALL 80 COCO classes by default (configurable via target_classes)
    - People (class 0) are highlighted as PRIMARY target
    - All other objects labeled with correct COCO names
    - Each category has a distinct box color for easy visual parsing

    Attributes:
        model_path      : Path/name of YOLO weights file
        conf_threshold  : Min confidence to accept a detection
        input_size      : Inference image size in pixels
        target_classes  : List of class IDs to detect, or None = all 80
        model           : Loaded YOLO instance
    """

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        conf_threshold: float = 0.35,
        input_size: int = 640,
        target_classes: Optional[List[int]] = None,  # None = detect all classes
        device: Optional[Any] = "auto",
    ):
        """
        Args:
            model_path      : YOLO weights file. Auto-downloaded if not found.
            conf_threshold  : Min confidence (0.0ΓÇô1.0). Default 0.35.
            input_size      : Inference size. 640 = standard, 416 = faster.
            target_classes  : List of COCO class IDs to detect.
                              None means detect ALL 80 classes.
                              Example: [0, 2, 5] = person + car + bus only.
            device          : Device for inference ('auto', 'cuda', 0, 'cpu'). Default 'auto'.
        """
        self.model_path     = model_path
        self.conf_threshold = conf_threshold
        self.input_size     = input_size
        # None ΓåÆ detect all 80 classes; list ΓåÆ only those class IDs
        self.target_classes = target_classes
        self.model: Optional[YOLO] = None

        # Determine compute device
        if device == "auto" or device is None:
            if torch is not None and torch.cuda.is_available():
                self.device = 0
            else:
                self.device = "cpu"
        else:
            self.device = device
        self._tracker_type = "bytetrack.yaml"  # set via set_tracker_type()
        self._ema_centroids = {}
        self._track_ages = {}  # {track_id: (cx, cy)} for ByteTrack EMA

        gpu_info = ""
        if self.device != "cpu" and torch is not None and torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name(self.device if isinstance(self.device, int) else 0)
                gpu_info = f" ({gpu_name})"
            except Exception:
                gpu_info = f" (CUDA:{self.device})"

        self.logger = logging.getLogger("amst_border_net.detector")
        self.logger.info(
            f"YOLODetector created | model={model_path} | device={self.device}{gpu_info} | "
            f"conf={conf_threshold} | input_size={input_size} | "
            f"classes={'ALL 80' if target_classes is None else target_classes}"
        )

        # Fast Face Cascade for macro close-ups (e.g. 15-30cm from camera)
        self.face_cascade = None
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                self.face_cascade = None
            else:
                self.logger.info("Face Cascade loaded for macro proximity fallback.")
        except Exception as fe:
            self.logger.debug(f"Face cascade initialization failed: {fe}")

    def load_model(self) -> None:
        """
        Load YOLO11 model weights into memory and run warm-up inference.
        Auto-downloads model if not found locally (requires internet on first run).
        """
        self.logger.info(f"Loading YOLO model: {self.model_path} ...")
        self.logger.info(
            "First run will auto-download the model (~6MB). Please wait..."
        )
        try:
            self.model = YOLO(self.model_path)
            self.logger.info(f"Model loaded. Running warm-up inference on device: {self.device}...")
            self._warmup()
            self.logger.info(
                f"YOLODetector ready. "
                f"Detecting: {'ALL 80 COCO classes' if self.target_classes is None else [COCO_CLASSES.get(c, str(c)) for c in self.target_classes]}"
            )
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            raise

    def _warmup(self) -> None:
        """Run one blank-frame inference to trigger JIT compilation."""
        blank = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        self.model(blank, verbose=False, device=self.device)
        self.logger.debug("Warm-up done.")

    def set_tracker_type(self, tracker_type: str) -> None:
        """Set the Ultralytics tracker config to use (bytetrack or botsort)."""
        mapping = {
            "bytetrack": "bytetrack.yaml",
            "botsort"  : "botsort.yaml",
        }
        self._tracker_type = mapping.get(tracker_type, "bytetrack.yaml")
        self.logger.info(f"ByteTrack tracker set to: {self._tracker_type}")


    def reset_tracker(self) -> None:
        """Reset the internal ByteTrack tracker, EMA centroids, and track ages."""
        self._ema_centroids.clear()
        self._track_ages.clear()
        try:
            if hasattr(self.model, "predictor") and self.model.predictor is not None:
                if hasattr(self.model.predictor, "trackers"):
                    for t in self.model.predictor.trackers:
                        if hasattr(t, "reset"):
                            t.reset()
        except Exception as e:
            self.logger.warning(f"Tracker reset warning: {e}")

    def track(self, frame: np.ndarray):
        """
        Run YOLO + ByteTrack in a single model.track() call.

        Returns (detections, tracks) where:
          - detections : same format as detect() ? all classes, full HUD info
          - tracks     : person-only list with stable ByteTrack IDs + EMA centroids
        """
        if self.model is None:
            self.logger.error("track() called before load_model()!")
            return [], []
        if frame is None or frame.size == 0:
            return [], []

        EMA_ALPHA = 0.35  # smoother than FallbackIOUTracker for ByteTrack

        try:
            results = self.model.track(
                frame,
                imgsz   = self.input_size,
                conf    = self.conf_threshold,
                iou     = 0.5,
                persist = True,               # keeps Kalman state across frames
                tracker = self._tracker_type,
                device  = self.device,
                verbose = False,
            )

            # All-class detections for HUD (reuse existing parse + close-range logic)
            detections = self._parse_results(results, frame.shape)
            h, w = frame.shape[:2]
            people_found = any(d.get("is_person") for d in detections)
            for d in detections:
                if not d.get("is_person"):
                    x1, y1, x2, y2 = d["bbox"]
                    bh = (y2 - y1) / max(1, h)
                    ar = d.get("bbox_area_ratio", 0.0)
                    if (bh >= 0.40 or ar >= 0.15) and d.get("class") in (
                            "tie", "backpack", "suitcase", "umbrella"):
                        d["is_person"] = True; d["class"] = "person"
                        d["class_id"] = 0; people_found = True

            # Person tracks with ByteTrack IDs
            tracks = self._parse_tracks(results, frame.shape)

            # Update EMA centroids per track
            new_ema = {}
            for t in tracks:
                tid = t["track_id"]
                cx, cy = t["_raw_cx"], t["_raw_cy"]
                if tid in self._ema_centroids:
                    ex, ey = self._ema_centroids[tid]
                    ex = EMA_ALPHA * cx + (1 - EMA_ALPHA) * ex
                    ey = EMA_ALPHA * cy + (1 - EMA_ALPHA) * ey
                else:
                    ex, ey = float(cx), float(cy)
                new_ema[tid] = (ex, ey)
                t["ema_centroid"] = [int(ex), int(ey)]
                self._track_ages[tid] = self._track_ages.get(tid, 0) + 1
                t["track_age"] = self._track_ages[tid]
            self._ema_centroids = new_ema

            # Remove helper keys
            for t in tracks:
                t.pop("_raw_cx", None); t.pop("_raw_cy", None)

            return detections, tracks

        except Exception as e:
            self.logger.error(f"track() error: {e}. Falling back to detect().")
            return self.detect(frame), []

    def _parse_tracks(self, results, shape) -> list:
        """Extract ByteTrack person tracks from model.track() results."""
        tracks = []
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        if result.boxes.id is None:
            return []   # no tracks assigned yet (first 1-2 frames)

        h, w = shape[:2]
        for box in result.boxes:
            if int(box.cls[0]) != 0:   # person only
                continue
            if box.id is None:
                continue
            track_id = int(box.id[0])
            conf     = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            x1 = max(0, min(x1, w - 1)); y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1)); y2 = max(0, min(y2, h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            tracks.append({
                "track_id"    : track_id,
                "bbox"        : [x1, y1, x2, y2],
                "confidence"  : conf,
                "class"       : "person",
                "class_id"    : 0,
                "is_person"   : True,
                "track_age"   : 1,
                "ema_centroid": [cx, cy],   # updated after EMA calc
                "_raw_cx"     : cx,
                "_raw_cy"     : cy,
            })
        return tracks

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run YOLO inference and return ALL detected objects with proper labels.

        Each detected object becomes a dict:
        {
            "bbox"      : [x1, y1, x2, y2],
            "confidence": 0.87,
            "class"     : "person",        ΓåÉ always correct COCO label
            "class_id"  : 0,               ΓåÉ COCO class index
            "is_person" : True,            ΓåÉ True only for class 0
            "color"     : (0, 255, 255),   ΓåÉ BGR color for drawing
        }

        Args:
            frame: BGR image from OpenCV (numpy HxWx3).

        Returns:
            List of detection dicts. Empty list if nothing detected.
        """
        if self.model is None:
            self.logger.error("detect() called before load_model()!")
            return []

        if frame is None or frame.size == 0:
            self.logger.warning("Empty frame received. Skipping.")
            return []

        try:
            # Run YOLO inference
            # classes=None ΓåÆ detect all 80 COCO classes
            # classes=[0,2,...] ΓåÆ detect only those specific class IDs
            results = self.model(
                frame,
                imgsz   = self.input_size,
                conf    = self.conf_threshold,
                iou     = 0.5,               # NMS overlap threshold
                classes = self.target_classes,  # None = all classes
                device  = self.device,
                verbose = False,
            )

            detections = self._parse_results(results, frame.shape)

            # --- Check for Macro Close-ups / Face Fallback ---
            # When someone is 15-30cm away, YOLO might detect 0 people or misclassify as "tie"
            h, w = frame.shape[:2]
            people_found = any(d.get("is_person") for d in detections)

            # 1. Upgrade large misclassified clothing objects (e.g. tie, backpack)
            for d in detections:
                if not d.get("is_person"):
                    x1, y1, x2, y2 = d["bbox"]
                    box_h_ratio = (y2 - y1) / max(1, h)
                    area_ratio = d.get("bbox_area_ratio", 0.0)
                    if (box_h_ratio >= 0.40 or area_ratio >= 0.15) and d.get("class") in ("tie", "backpack", "suitcase", "umbrella"):
                        d["is_person"] = True
                        d["class"] = "person"
                        d["class_id"] = PERSON_CLASS_ID
                        d["color"] = CLASS_COLORS.get(PERSON_CLASS_ID, (0, 255, 255))
                        d["is_too_close"] = (box_h_ratio >= 0.65)
                        people_found = True

            # 2. If still no person detected, run lightweight Haar face cascade for close proximity (<50cm)
            if not people_found and self.face_cascade is not None:
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60)
                    )
                    for (fx, fy, fw, fh) in faces:
                        face_ratio_h = fh / max(1, h)
                        face_ratio_w = fw / max(1, w)
                        # If face is reasonably sized (close-up)
                        if face_ratio_h >= 0.12 or face_ratio_w >= 0.12:
                            px1 = max(0, int(fx - fw * 0.4))
                            py1 = max(0, int(fy - fh * 0.2))
                            px2 = min(w - 1, int(fx + fw * 1.4))
                            py2 = min(h - 1, int(fy + fh * 2.5))

                            is_close = ((py2 - py1) / max(1, h) >= 0.65) or (face_ratio_h >= 0.22)

                            detections.append({
                                "bbox"           : [px1, py1, px2, py2],
                                "confidence"     : 0.89,
                                "class"          : "person",
                                "class_id"       : PERSON_CLASS_ID,
                                "is_person"      : True,
                                "color"          : CLASS_COLORS.get(PERSON_CLASS_ID, (0, 255, 255)),
                                "is_too_close"   : is_close,
                                "bbox_area_ratio": round(((px2 - px1) * (py2 - py1)) / max(1, h * w), 4),
                            })
                except Exception as fe:
                    self.logger.debug(f"Face fallback check failed: {fe}")

            self.logger.debug(
                f"Detected {len(detections)} object(s): "
                + ", ".join([d["class"] for d in detections])
            )
            return detections

        except Exception as e:
            self.logger.error(f"Inference error: {e}")
            return []

    # Keep detect_people() as an alias for backward compatibility
    # (called by tracker.py fallback path)
    def detect_people(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect ONLY people (class 0). Alias kept for backward compatibility.
        Returns same format as detect() but filtered to class 0 only.
        """
        all_dets = self.detect(frame)
        return [d for d in all_dets if d["class_id"] == PERSON_CLASS_ID]

    def _parse_results(
        self,
        results,
        original_shape: Tuple[int, int, int]
    ) -> List[Dict[str, Any]]:
        """
        Convert raw YOLO Results object ΓåÆ list of clean dicts.

        Args:
            results        : YOLO Results list (one per image).
            original_shape : (H, W, C) of original frame for bounds clamping.

        Returns:
            List of detection dicts with all fields populated.
        """
        detections = []
        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return []

        h, w = original_shape[:2]

        for box in result.boxes:
            class_id   = int(box.cls[0])
            confidence = float(box.conf[0])

            # Get the human-readable label from our COCO map
            # If class_id is somehow unknown, fall back to "unknown_<id>"
            label = COCO_CLASSES.get(class_id, f"unknown_{class_id}")

            # Get bbox and clamp to frame boundaries
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            # Skip degenerate boxes (zero area)
            if x2 <= x1 or y2 <= y1:
                continue

            # Get the color for this class
            color = CLASS_COLORS.get(class_id, _DEFAULT)

            detection = {
                "bbox"           : [x1, y1, x2, y2],
                "confidence"     : round(confidence, 4),
                "class"          : label,          # e.g. "person", "car", "scissors"
                "class_id"       : class_id,       # COCO integer ID
                "is_person"      : (class_id == PERSON_CLASS_ID),
                "color"          : color,          # BGR tuple for drawing
                "is_too_close"   : False,          # set by filter_too_close()
                "bbox_area_ratio": round(((x2-x1)*(y2-y1)) / max(1, h*w), 4),
            }
            detections.append(detection)

        return detections

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        tracks: Optional[List[Dict[str, Any]]] = None,
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on the frame.

        Visual design:
          - PEOPLE  : Thick (3px) cyan box + "PERSON | ID:X | 0.89"
          - OTHERS  : Thin (2px) category-colored box + "scissors | 0.74"
          - Centroid: Red dot on people only (for tracking visualization)

        Args:
            frame      : BGR image. Modified in-place and returned.
            detections : List of dicts from detect().
            tracks     : Optional track list from FallbackIOUTracker.
                         Used to show track IDs on people.

        Returns:
            The annotated frame.
        """
        # Build a lookup from bbox ΓåÆ track_id (for people only)
        # We match by IoU overlap since tracked bbox may differ slightly
        track_id_map: Dict[tuple, int] = {}
        if tracks:
            for t in tracks:
                key = tuple(t["bbox"])
                track_id_map[key] = t["track_id"]

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            class_id  = det["class_id"]
            label     = det["class"]
            conf      = det["confidence"]
            is_person = det["is_person"]
            color     = det["color"]

            # --- Box thickness: people get thicker border to stand out ---
            thickness = 3 if is_person else 2

            # --- Draw bounding box ---
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # --- Build label text ---
            if is_person and tracks:
                # Find track ID for this person
                tid = self._find_track_id(det["bbox"], tracks)
                if tid is not None:
                    text = f"PERSON | ID:{tid} | {conf:.2f}"
                else:
                    text = f"PERSON | {conf:.2f}"
            else:
                # All other objects: just show their name + confidence
                text = f"{label} | {conf:.2f}"

            # --- Draw label background rectangle ---
            font       = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6 if is_person else 0.5
            font_thick = 2   if is_person else 1

            (tw, th), baseline = cv2.getTextSize(text, font, font_scale, font_thick)
            label_y1 = max(0, y1 - th - baseline - 4)
            label_y2 = y1
            label_x2 = x1 + tw + 6

            cv2.rectangle(frame, (x1, label_y1), (label_x2, label_y2), color, -1)

            # --- Draw label text (black for contrast) ---
            cv2.putText(
                frame, text,
                (x1 + 3, y1 - baseline - 2),
                font, font_scale,
                (0, 0, 0),      # Black text on colored background
                font_thick,
                cv2.LINE_AA
            )

            # --- Draw centroid dot on people only ---
            if is_person:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)  # Red dot

            # --- Draw CRITICAL CLOSE PROXIMITY indicator (Bold Red Alert) ---
            if det.get("is_too_close", False):
                _c = (0, 0, 255)  # Bright RED in BGR
                cv2.rectangle(frame, (x1, y1), (x2, y2), _c, 3)
                seg = 30
                cv2.line(frame, (x1, y1), (x1+seg, y1), _c, 6)
                cv2.line(frame, (x1, y1), (x1, y1+seg), _c, 6)
                cv2.line(frame, (x2, y1), (x2-seg, y1), _c, 6)
                cv2.line(frame, (x2, y1), (x2, y1+seg), _c, 6)
                cv2.line(frame, (x1, y2), (x1+seg, y2), _c, 6)
                cv2.line(frame, (x1, y2), (x1, y2-seg), _c, 6)
                cv2.line(frame, (x2, y2), (x2-seg, y2), _c, 6)
                cv2.line(frame, (x2, y2), (x2, y2-seg), _c, 6)
                vc_text = " [!] CRITICAL: PROXIMITY BREACH (<30cm) "
                (tw, th), _ = cv2.getTextSize(vc_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw + 6, max(th + 10, y1)), _c, -1)
                cv2.putText(
                    frame, vc_text, (x1 + 3, max(th + 2, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
                )

        return frame

    def _find_track_id(
        self,
        bbox: List[int],
        tracks: List[Dict[str, Any]],
        min_iou: float = 0.3
    ) -> Optional[int]:
        """
        Find the track ID for a detection by matching bboxes with IoU.
        Returns the track_id of the best-matching track, or None.
        """
        best_iou = min_iou
        best_id  = None
        for t in tracks:
            iou = self._iou(bbox, t["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_id  = t["track_id"]
        return best_id

    @staticmethod
    def _iou(box1: List[int], box2: List[int]) -> float:
        """Compute Intersection-over-Union between two [x1,y1,x2,y2] boxes."""
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

    @staticmethod
    def filter_too_close(
        detections: List[Dict[str, Any]],
        frame_h: int,
        height_ratio: float = 0.70,
        conf_threshold: float = 0.35,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Split detections into normal (trackable) and very-close (immediate proximity breach).
        """
        if frame_h <= 0:
            return detections, []

        normal: List[Dict[str, Any]] = []
        very_close: List[Dict[str, Any]] = []

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            bbox_h = y2 - y1
            ratio  = bbox_h / frame_h

            if ratio >= height_ratio or det.get("is_too_close", False):
                d = dict(det)
                d["is_too_close"]      = True
                d["bbox_height_ratio"] = round(ratio, 3)
                if d.get("confidence", 0.0) >= conf_threshold:
                    very_close.append(d)
            else:
                d = dict(det)
                d["is_too_close"]      = False
                d["bbox_height_ratio"] = round(ratio, 3)
                normal.append(d)

        return normal, very_close
