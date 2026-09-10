import unittest
from datetime import time
import numpy as np

from modules.zone_context import ZoneContextMemory
from modules.abandoned_engine import AbandonedObjectEngine
from modules.risk_engine import RiskEngine


class TestZoneContextMemory(unittest.TestCase):
    def setUp(self):
        self.zone_mgr = ZoneContextMemory(config_path="config/boundary_config.yaml")
        self.sample_zone_mgr = ZoneContextMemory()  # Default sample demonstration zones

    def test_zone_initialization(self):
        # boundary_config.yaml maintains clean 2-tier spatial model (perimeter + approach_band)
        zones = self.zone_mgr.get_all_zones()
        self.assertEqual(len(zones), 2)
        zone_ids = [z["id"] for z in zones]
        self.assertIn("perimeter", zone_ids)
        self.assertIn("approach_band", zone_ids)

        # DEFAULT_SAMPLE_ZONES contains all 5 sample demonstration zones
        sample_zones = self.sample_zone_mgr.get_all_zones()
        self.assertGreaterEqual(len(sample_zones), 5)

    def test_scale_zones(self):
        self.zone_mgr.scale_zones_to_frame(1280, 720)
        self.assertEqual(self.zone_mgr.frame_resolution, (1280, 720))
        p_zone = self.zone_mgr.get_zone_by_id("perimeter")
        self.assertIsNotNone(p_zone)
        pts = p_zone["polygon"]
        self.assertGreater(np.max(pts[:, 0]), 1.0)
        self.assertGreater(np.max(pts[:, 1]), 1.0)

    def test_point_query(self):
        self.zone_mgr.scale_zones_to_frame(1280, 720)
        # Point inside perimeter restricted zone: (640, 200) during daytime (14:00)
        zone = self.zone_mgr.get_zone_at_point((640, 200), current_time=time(14, 0))
        self.assertIsNotNone(zone)
        self.assertEqual(zone["id"], "perimeter")
        self.assertEqual(zone["base_type"], "restricted")
        self.assertEqual(zone["effective_type"], "restricted")

        # Point inside approach band: (640, 650)
        app_zone = self.zone_mgr.get_zone_at_point((640, 650))
        self.assertIsNotNone(app_zone)
        self.assertEqual(app_zone["id"], "approach_band")
        self.assertEqual(app_zone["effective_type"], "monitor")

    def test_time_of_day_override(self):
        # Daylight time: 14:00 (2 PM) - perimeter is restricted
        day_eval = self.zone_mgr.evaluate_zone_context("perimeter", current_time=time(14, 0))
        self.assertEqual(day_eval["zone_type"], "restricted")
        self.assertFalse(day_eval["is_after_hours"])

        # Night time: 23:00 (11 PM) - after-hours rule flips perimeter to secure lockdown
        night_eval = self.zone_mgr.evaluate_zone_context("perimeter", current_time=time(23, 0))
        self.assertEqual(night_eval["zone_type"], "secure")
        self.assertTrue(night_eval["is_after_hours"])

    def test_expected_object(self):
        # In perimeter: person is expected, car/backpack is unexpected
        self.assertTrue(self.zone_mgr.is_object_expected("perimeter", "person", current_time=time(14, 0)))
        self.assertFalse(self.zone_mgr.is_object_expected("perimeter", "car", current_time=time(14, 0)))
        self.assertFalse(self.zone_mgr.is_object_expected("perimeter", "backpack", current_time=time(14, 0)))

        # In sample parking_area during day: vehicle is expected
        self.assertTrue(self.sample_zone_mgr.is_object_expected("parking_area", "car", current_time=time(14, 0)))
        self.assertFalse(self.sample_zone_mgr.is_object_expected("parking_area", "backpack", current_time=time(14, 0)))


class TestAbandonedObjectEngine(unittest.TestCase):
    def setUp(self):
        self.engine = AbandonedObjectEngine(config_path="config/boundary_config.yaml")
        self.zone_mgr = ZoneContextMemory(config_path="config/boundary_config.yaml")
        self.zone_mgr.scale_zones_to_frame(1280, 720)

    def test_categorize_object(self):
        self.assertEqual(self.engine.categorize_object("backpack"), "bag")
        self.assertEqual(self.engine.categorize_object("handbag"), "bag")
        self.assertEqual(self.engine.categorize_object("suitcase"), "bag")
        self.assertEqual(self.engine.categorize_object("car"), "vehicle")
        self.assertEqual(self.engine.categorize_object("truck"), "vehicle")
        self.assertEqual(self.engine.categorize_object("cell phone"), "device")
        self.assertEqual(self.engine.categorize_object("laptop"), "device")
        self.assertIsNone(self.engine.categorize_object("person"))
        self.assertIsNone(self.engine.categorize_object("dog"))

    def test_stationary_dwell_and_abandonment(self):
        # Backpack placed stationary at (640, 360) inside perimeter zone
        obj_track = {
            "track_id": 101,
            "class_name": "backpack",
            "box": [620, 340, 660, 380],
            "center": (640, 360),
            "confidence": 0.85
        }

        # Feed 30 frames at dt=0.5s -> 15.0 seconds
        for _ in range(30):
            alerts = self.engine.update(
                object_tracks=[obj_track],
                person_tracks=[],
                frame_dt=0.5,
                zone_context_mgr=self.zone_mgr,
                current_time=time(14, 0),
                is_night_mode=False
            )

        self.assertEqual(len(alerts), 1)
        alert = alerts[0]
        self.assertEqual(alert["track_id"], 101)
        self.assertEqual(alert["category"], "bag")
        self.assertGreaterEqual(alert["dwell_time"], 10.0)

    def test_human_ownership_prevents_abandonment(self):
        # Backpack at (640, 360)
        obj_track = {
            "track_id": 102,
            "class_name": "backpack",
            "box": [620, 340, 660, 380],
            "center": (640, 360),
            "confidence": 0.85
        }
        # Human standing right next to backpack at (650, 360)
        human_track = {
            "track_id": 1,
            "class_name": "person",
            "is_person": True,
            "is_animal": False,
            "center": (650, 360),
            "box": [630, 300, 670, 420]
        }

        # Update 30 times
        alerts = []
        for _ in range(30):
            alerts = self.engine.update(
                object_tracks=[obj_track],
                person_tracks=[human_track],
                frame_dt=0.5,
                zone_context_mgr=self.zone_mgr,
                current_time=time(14, 0),
                is_night_mode=False
            )

        # Because a human is nearby, it is attended, so NO abandoned alert
        self.assertEqual(len(alerts), 0)

    def test_animal_does_not_prevent_abandonment(self):
        # Backpack at (640, 360)
        obj_track = {
            "track_id": 103,
            "class_name": "backpack",
            "box": [620, 340, 660, 380],
            "center": (640, 360),
            "confidence": 0.85
        }
        # Animal (e.g. dog / cattle) right next to backpack at (645, 360)
        animal_track = {
            "track_id": 99,
            "class_name": "dog",
            "is_person": False,
            "is_animal": True,
            "center": (645, 360),
            "box": [630, 340, 660, 380]
        }

        alerts = []
        for _ in range(30):
            alerts = self.engine.update(
                object_tracks=[obj_track],
                person_tracks=[animal_track],  # Animal passed to engine
                frame_dt=0.5,
                zone_context_mgr=self.zone_mgr,
                current_time=time(14, 0),
                is_night_mode=False
            )

        # Animal must NOT be treated as human owner! Abandonment MUST trigger.
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["track_id"], 103)
        self.assertIsNone(alerts[0]["owner_track_id"])

    def test_night_multiplier_faster_abandonment(self):
        # Device at night (multiplier 0.7x) triggers faster than day
        obj_track = {
            "track_id": 104,
            "class_name": "cell phone",
            "box": [635, 355, 645, 365],
            "center": (640, 360),
            "confidence": 0.80
        }

        # Device in perimeter zone (restricted): base threshold = 15.0s, zone mult = 0.6x, night mult = 0.7x -> 6.3s
        # 14 frames at dt=0.5s = 7.0s
        for _ in range(14):
            alerts = self.engine.update(
                object_tracks=[obj_track],
                person_tracks=[],
                frame_dt=0.5,
                zone_context_mgr=self.zone_mgr,
                current_time=time(23, 0),
                is_night_mode=True
            )

        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["track_id"], 104)
        self.assertEqual(alerts[0]["category"], "device")


class TestRiskEngineAbandonedIntegration(unittest.TestCase):
    def setUp(self):
        self.risk_engine = RiskEngine(config_path="config/boundary_config.yaml")
        self.zone_mgr = ZoneContextMemory(config_path="config/boundary_config.yaml")
        self.zone_mgr.scale_zones_to_frame(1280, 720)

    def test_abandoned_bag_risk_score(self):
        abandoned_alerts = [{
            "object_id": 201,
            "track_id": "obj_201",
            "class": "backpack",
            "class_name": "backpack",
            "category": "bag",
            "bbox": [600, 300, 650, 350],
            "centroid": [625, 325],
            "stationary_time": 25.0,
            "dwell_sec": 25.0,
            "dwell_time": 25.0,
            "threshold": 10.0,
            "is_night": False,
            "zone_id": "perimeter",
            "zone_type": "restricted",
            "zone_name": "Perimeter Wall",
            "owner_id": None,
            "owner_track_id": None,
            "confidence": 0.85,
        }]

        cards = self.risk_engine.update(
            tracks=[],
            boundary_zones=[],
            abandoned_events=abandoned_alerts,
            zone_context_mgr=self.zone_mgr,
            current_time=time(14, 0),
            night_mode=False
        )

        card = cards.get("ABANDONED-201") or cards.get("obj_201")
        self.assertIsNotNone(card)
        self.assertGreaterEqual(card["risk_score"], 80.0)
        self.assertTrue(card["is_abandoned"])
        self.assertEqual(card["category"], "bag")
        self.assertIn(card["alert_level"], ("CRITICAL", "HIGH"))

    def test_animal_suppression_remains_intact(self):
        # Verify that animal track still gets suppressed to 0 risk in risk engine
        animal_track = {
            "track_id": 50,
            "class": "dog",
            "is_animal": True,
            "is_person": False,
            "bbox": [100, 100, 150, 150],
            "confidence": 0.88,
        }
        boundary_zones = [{
            "id": "perimeter",
            "type": "restricted",
            "polygon": [[0, 0], [500, 0], [500, 500], [0, 500]],
        }]

        cards = self.risk_engine.update(
            tracks=[animal_track],
            boundary_zones=boundary_zones,
            fps=15.0,
            frame_id=1,
            zone_context_mgr=self.zone_mgr,
            current_time=time(14, 0),
            night_mode=False
        )

        self.assertIn(50, cards)
        card = cards[50]
        self.assertEqual(card["risk_score"], 0.0)
        self.assertTrue(card["is_animal"])
        self.assertEqual(card["alert_level"], "NORMAL")
        self.assertIn("risk suppressed", card["reasoning"])


if __name__ == "__main__":
    unittest.main()
