"""
test_tier2_features.py - Automated Unit & Integration Tests for Tier-2 Enhancements
==================================================================================
Tests all 4 Tier-2 features:
  1. Animal-vs-Human Suppression Classifier
  2. Zone Context Memory (Time-of-Day / Expected-Object Config & Multiplier)
  3. Abandoned-Object Detection Logic
  4. Day/Night Preprocessing (CLAHE + Illumination Classifier)
"""

import unittest
import numpy as np
import cv2
from datetime import datetime, time as dtime

# Import modules under test
from modules.illumination import IlluminationManager, get_illumination_state, enhance_low_light
from modules.abandoned_engine import AbandonedObjectEngine, _bbox_iou
from modules.boundary_engine import is_time_in_windows, get_allowed_objects, is_object_allowed, point_in_polygon
from modules.risk_engine import RiskEngine, ANIMAL_CLASSES
from data_exporter import DataExporter


class TestIlluminationModule(unittest.TestCase):
    """Test Day/Night Illumination Preprocessing, Hysteresis, and CLAHE."""

    def test_brightness_computation(self):
        mgr = IlluminationManager(low_thresh=50.0, high_thresh=70.0)
        # Create dark frame (brightness ~20)
        dark_frame = np.full((100, 100, 3), 20, dtype=np.uint8)
        # Create bright frame (brightness ~180)
        bright_frame = np.full((100, 100, 3), 180, dtype=np.uint8)

        self.assertAlmostEqual(mgr.compute_brightness(dark_frame), 20.0, delta=1.0)
        self.assertAlmostEqual(mgr.compute_brightness(bright_frame), 180.0, delta=1.0)

    def test_hysteresis_state_transitions(self):
        mgr = IlluminationManager(low_thresh=50.0, high_thresh=70.0, initial_night=False)

        # Starts as DAY
        self.assertFalse(mgr.is_night)

        # Frame with brightness 60 (between 50 and 70): should STAY DAY
        frame_mid = np.full((50, 50, 3), 60, dtype=np.uint8)
        state = mgr.update(frame_mid)
        self.assertFalse(state["is_night"])
        self.assertEqual(state["mode"], "DAY")

        # Frame with brightness 40 (< 50): should transition to NIGHT
        frame_dark = np.full((50, 50, 3), 40, dtype=np.uint8)
        state = mgr.update(frame_dark)
        self.assertTrue(state["is_night"])
        self.assertEqual(state["mode"], "NIGHT")

        # Frame with brightness 60: should STAY NIGHT due to hysteresis
        state = mgr.update(frame_mid)
        self.assertTrue(state["is_night"])

        # Frame with brightness 80 (> 70): should transition back to DAY
        frame_bright = np.full((50, 50, 3), 80, dtype=np.uint8)
        state = mgr.update(frame_bright)
        self.assertFalse(state["is_night"])
        self.assertEqual(state["mode"], "DAY")

    def test_clahe_enhancement(self):
        mgr = IlluminationManager()
        dark_frame = np.full((100, 100, 3), 30, dtype=np.uint8)
        enhanced = mgr.apply_clahe(dark_frame)

        self.assertEqual(enhanced.shape, dark_frame.shape)
        self.assertEqual(enhanced.dtype, np.uint8)
        # CLAHE on low-light frame should produce valid BGR image
        self.assertTrue(isinstance(enhanced, np.ndarray))


class TestZoneContextMemory(unittest.TestCase):
    """Test Zone Context Memory: Time-of-Day windows, allowed objects, context penalties."""

    def test_time_windows(self):
        # Standard day window
        windows = ["06:00-18:00"]
        self.assertTrue(is_time_in_windows(dtime(10, 0), windows))
        self.assertFalse(is_time_in_windows(dtime(20, 0), windows))

        # Overnight window e.g. 22:00 to 04:00
        night_windows = ["22:00-04:00"]
        self.assertTrue(is_time_in_windows(dtime(23, 30), night_windows))
        self.assertTrue(is_time_in_windows(dtime(2, 0), night_windows))
        self.assertFalse(is_time_in_windows(dtime(12, 0), night_windows))

    def test_object_allowed_in_zone(self):
        zone = {
            "id": "restricted_perimeter",
            "type": "restricted",
            "allowed_objects": ["person"],
            "time_windows": ["06:00-22:00"],
        }

        # Person during allowed time: Allowed
        ok, reason = is_object_allowed({"class": "person"}, zone, current_time=dtime(12, 0))
        self.assertTrue(ok)

        # Vehicle in person-only zone: Disallowed (unexpected object)
        ok, reason = is_object_allowed({"class": "car"}, zone, current_time=dtime(12, 0))
        self.assertFalse(ok)
        self.assertIn("Unexpected object", reason)

        # Person outside allowed time window: Disallowed (restricted time window)
        ok, reason = is_object_allowed({"class": "person"}, zone, current_time=dtime(23, 30))
        self.assertFalse(ok)
        self.assertIn("Restricted time window", reason)

    def test_context_penalty_in_risk_engine(self):
        risk_eng = RiskEngine(config={
            "context_penalty_weight": 35.0,
            "risk": {"base_access": 40.0}
        })
        boundary_zones = [{
            "id": "perimeter",
            "type": "restricted",
            "polygon": [[0, 0], [500, 0], [500, 500], [0, 500]],
            "allowed_objects": ["person"],
            "time_windows": ["06:00-18:00"],
        }]

        # Track with a vehicle (unexpected in perimeter)
        vehicle_track = [{
            "track_id": 10,
            "bbox": [100, 100, 200, 200],
            "confidence": 0.90,
            "class": "car",
            "is_person": False,
            "is_animal": False,
        }]

        cards = risk_eng.update(
            tracks=vehicle_track,
            frame_h=720,
            frame_w=1280,
            boundary_zones=boundary_zones,
            fps=15.0,
            frame_id=1,
            current_time=dtime(12, 0),
        )

        card = cards[10]
        self.assertTrue(card["context_violation"])
        self.assertIn("Unexpected object", card["context_reason"])
        # Risk should have context penalty added
        self.assertGreaterEqual(card["risk_score"], 35.0)


class TestAnimalSuppression(unittest.TestCase):
    """Test Animal-vs-Human Suppression logic."""

    def test_animal_risk_suppressed(self):
        risk_eng = RiskEngine(config={
            "animal_suppression_enabled": True,
            "animal_suppressed_risk": 0.0,
        })
        boundary_zones = [{
            "id": "perimeter",
            "type": "restricted",
            "polygon": [[0, 0], [500, 0], [500, 500], [0, 500]],
        }]

        dog_track = [{
            "track_id": 5,
            "bbox": [100, 100, 180, 180],
            "confidence": 0.88,
            "class": "dog",
            "is_person": False,
            "is_animal": True,
        }]

        cards = risk_eng.update(
            tracks=dog_track,
            frame_h=720,
            frame_w=1280,
            boundary_zones=boundary_zones,
            fps=15.0,
            frame_id=1,
        )

        card = cards[5]
        self.assertTrue(card["is_animal"])
        self.assertEqual(card["alert_level"], "NORMAL")
        self.assertEqual(card["risk_score"], 0.0)
        self.assertIn("risk suppressed", card["reasoning"])

    def test_animal_suppression_disabled_flag(self):
        risk_eng = RiskEngine(config={
            "animal_suppression_enabled": False,
        })
        boundary_zones = [{
            "id": "perimeter",
            "type": "restricted",
            "polygon": [[0, 0], [500, 0], [500, 500], [0, 500]],
        }]

        dog_track = [{
            "track_id": 5,
            "bbox": [100, 100, 180, 180],
            "confidence": 0.88,
            "class": "dog",
            "is_person": False,
            "is_animal": True,
        }]

        # With suppression explicitly False
        cards = risk_eng.update(
            tracks=dog_track,
            frame_h=720,
            frame_w=1280,
            boundary_zones=boundary_zones,
            fps=15.0,
            frame_id=1,
            animal_suppression=False,
        )

        card = cards[5]
        # When suppression is disabled, intrusion in restricted zone triggers higher alert
        self.assertGreater(card["risk_score"], 0.0)


class TestAbandonedObjectEngine(unittest.TestCase):
    """Test Abandoned and Unattended Object Detection Engine."""

    def test_abandoned_object_workflow(self):
        engine = AbandonedObjectEngine(
            stationary_threshold=2.0,       # 2 seconds for quick test
            owner_distance_threshold=100.0,
            owner_absent_threshold=1.0,     # 1 second owner absent
        )

        fps = 10.0
        # Frame 1: Person at (200, 200) places backpack at (220, 200)
        bag_det = [{
            "bbox": [210, 190, 230, 210],
            "confidence": 0.90,
            "class": "backpack",
            "class_id": 24,
            "is_person": False,
            "is_animal": False,
        }]
        person_track_nearby = [{
            "track_id": 1,
            "bbox": [190, 180, 210, 220],
            "ema_centroid": [200, 200],
            "is_person": True,
        }]

        # Update 1 second: owner present near bag
        for _ in range(10):
            events = engine.update(bag_det, person_track_nearby, fps=fps)
            self.assertEqual(len(events), 0)

        # Owner walks away and leaves frame: person_tracks empty
        # Run for 2.5 seconds: object stationary for >2.0s, owner absent >1.0s
        triggered_events = []
        for _ in range(25):
            triggered_events = engine.update(bag_det, [], fps=fps)

        self.assertGreaterEqual(len(triggered_events), 1)
        ev = triggered_events[0]
        self.assertEqual(ev["event_type"], "ABANDONED_OBJECT")
        self.assertEqual(ev["class"], "backpack")
        self.assertFalse(ev["owner_in_frame"])
        self.assertGreaterEqual(ev["stationary_time"], 2.0)
        self.assertIn("owner left frame", ev["reasoning"])


class TestDataExporterTier2(unittest.TestCase):
    """Test JSON export carries Tier-2 fields."""

    def test_format_data_carries_tier2_fields(self):
        exporter = DataExporter(save_json=False)

        tracks = [{
            "track_id": 1,
            "bbox": [10, 10, 50, 50],
            "class": "dog",
            "is_animal": True,
            "is_person": False,
            "confidence": 0.85,
        }]
        cards = {
            1: {
                "track_id": 1,
                "is_animal": True,
                "is_abandoned": False,
                "context_violation": False,
                "alert_level": "NORMAL",
                "risk_score": 0.0,
                "reasoning": "Animal detected (dog) - risk suppressed",
            },
            "ABANDONED-1": {
                "track_id": "ABANDONED-1",
                "bbox": [200, 200, 250, 250],
                "centroid": [225, 225],
                "class": "backpack",
                "is_abandoned": True,
                "alert_level": "HIGH",
                "risk_score": 75.0,
                "reasoning": "Object stationary for 15s, owner left frame",
            }
        }

        data = exporter.format_data(frame_id=1, detections=[], tracks=tracks, alert_cards=cards)
        dets = data["detections"]

        # Track 1 check
        t1 = next(d for d in dets if d["track_id"] == 1)
        self.assertTrue(t1["is_animal"])
        self.assertFalse(t1["is_abandoned"])

        # Abandoned card check
        ab1 = next(d for d in dets if d["track_id"] == "ABANDONED-1")
        self.assertTrue(ab1["is_abandoned"])
        self.assertEqual(ab1["alert_level"], "HIGH")


class TestPersonFocusAndClutterSuppression(unittest.TestCase):
    """Test that intrusion events only fire for persons and toy cars/clutter are suppressed."""

    def test_generic_object_in_restricted_zone_suppressed(self):
        risk_eng = RiskEngine()
        boundary_zones = [{
            "id": "perimeter",
            "type": "restricted",
            "polygon": [[0, 0], [600, 0], [600, 600], [0, 600]],
        }]

        # Clutter / Toy RC car detected as generic object
        toy_track = [{
            "track_id": 99,
            "bbox": [100, 100, 150, 150],
            "confidence": 0.75,
            "class": "remote",
            "is_person": False,
            "is_animal": False,
            "is_vehicle": False,
        }]

        cards = risk_eng.update(
            tracks=toy_track,
            frame_h=720,
            frame_w=1280,
            boundary_zones=boundary_zones,
            fps=15.0,
            frame_id=1,
        )

        card = cards[99]
        self.assertEqual(card["alert_type"], "none")
        self.assertEqual(card["risk_score"], 0.0)
        self.assertEqual(card["alert_level"], "NORMAL")
        self.assertIn("risk suppressed", card["reasoning"])

    def test_vehicle_in_restricted_zone_no_intrusion_alert(self):
        risk_eng = RiskEngine()
        boundary_zones = [{
            "id": "perimeter",
            "type": "restricted",
            "polygon": [[0, 0], [600, 0], [600, 600], [0, 600]],
        }]

        # Real vehicle or toy detected as car
        car_track = [{
            "track_id": 100,
            "bbox": [100, 100, 200, 200],
            "confidence": 0.85,
            "class": "car",
            "is_person": False,
            "is_animal": False,
            "is_vehicle": True,
        }]

        cards = risk_eng.update(
            tracks=car_track,
            frame_h=720,
            frame_w=1280,
            boundary_zones=boundary_zones,
            fps=15.0,
            frame_id=1,
        )

        card = cards[100]
        # Vehicles do NOT directly trigger intrusion alerts
        self.assertEqual(card["alert_type"], "none")
        self.assertNotEqual(card["alert_type"], "intrusion")

    def test_yaml_config_vehicle_gate_removed(self):
        import yaml
        import os
        cfg_path = os.path.join(os.path.dirname(__file__), "config", "boundary_config.yaml")
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        zone_ids = [z["id"] for z in cfg.get("zones", [])]
        self.assertNotIn("vehicle_gate", zone_ids)
        self.assertIn("perimeter", zone_ids)
        self.assertIn("approach_band", zone_ids)


if __name__ == "__main__":
    unittest.main()
