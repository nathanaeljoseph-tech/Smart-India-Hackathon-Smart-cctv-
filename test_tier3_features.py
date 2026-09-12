"""
test_tier3_features.py - Comprehensive Unit & Integration Test Suite for Tier-3
================================================================================
BorderVigil-AI | Tier-3 Verification
Problem: SIH26187

Verifies:
  1. 1D TCN + Attention Temporal Behavior Model (kinematics, forward pass, fusion)
  2. Qwen2-VL VLM Client (triggering, cooldown, evidence caption generation)
  3. Llama 3.1 8B Instruct LLM Client (incident summaries, operator queries)
  4. Two-Camera Re-ID Fusion (appearance embeddings, single-cam pass-through, cross-cam match)
  5. Fence-Tamper Detector (frame-diff energy, spike detection, persistent disturbance)
  6. VMS REST API & Webhook Dispatcher
"""

import unittest
import numpy as np
import cv2
import time
from datetime import datetime

from modules.temporal_behavior import TemporalBehaviorModel, TCNAttentionBehaviorNet
from modules.vlm_client import VLMClient
from modules.llm_client import LLMClient
from modules.reid_fusion import ReIDFusionEngine, AppearanceFeatureExtractor
from modules.tamper_detector import FenceTamperDetector, FenceTamperStatus
from modules.vms_api import VMSStateManager, state_manager
from modules.risk_engine import RiskEngine


class TestTemporalBehaviorModel(unittest.TestCase):
    def setUp(self):
        self.tcn_model = TemporalBehaviorModel(
            history_len=16,
            alpha_rule=0.5,
            beta_tcn=0.5,
            boundary_y_ref=576.0
        )

    def test_feature_extraction_shape(self):
        feat = self.tcn_model.extract_frame_features(
            centroid=(640, 600),
            prev_centroid=(640, 620),
            frame_w=1280,
            frame_h=720,
            in_restricted_zone=False,
            dwell_sec=2.0
        )
        self.assertEqual(len(feat), 8)
        # Check normalized ranges
        self.assertTrue(0.0 <= feat[0] <= 1.0)
        self.assertTrue(0.0 <= feat[1] <= 1.0)

    def test_approach_motion_detection(self):
        # Target starts at y=680 and moves upward towards border (y=576) over 12 frames
        res = None
        for i in range(12):
            cy = 680 - i * 8
            res = self.tcn_model.update_track(
                track_id=1,
                centroid=(640, cy),
                in_restricted_zone=False,
                dwell_sec=0.5
            )
        self.assertIsNotNone(res)
        self.assertGreater(res["approach_score"], 0.3)
        self.assertIn(res["dominant_behavior"], ["approach", "cross"])

    def test_loitering_motion_detection(self):
        # Target stays stationary inside perimeter (y=300) for 10 seconds
        res = None
        for i in range(15):
            res = self.tcn_model.update_track(
                track_id=2,
                centroid=(500, 300),
                in_restricted_zone=True,
                dwell_sec=10.0
            )
        self.assertIsNotNone(res)
        self.assertGreater(res["loiter_score"], 0.4)
        self.assertGreater(res["tcn_likelihood"], 50.0)

    def test_score_fusion(self):
        rule_risk = 40.0
        tcn_data = {"tcn_likelihood": 80.0}
        fused, reason = self.tcn_model.fuse_scores(rule_risk, tcn_data)
        # 0.5 * 40 + 0.5 * 80 = 60.0
        self.assertAlmostEqual(fused, 60.0, delta=1.0)
        self.assertIn("Fused", reason)


class TestVLMClient(unittest.TestCase):
    def setUp(self):
        self.client = VLMClient(
            min_risk_trigger=60.0,
            cooldown_seconds=5.0
        )

    def test_trigger_threshold(self):
        # Risk 40 -> should not trigger
        self.assertFalse(self.client.should_trigger(40.0, track_id=10))
        # Risk 75 -> should trigger
        self.assertTrue(self.client.should_trigger(75.0, track_id=10))

    def test_cooldown(self):
        # First trigger succeeds
        self.assertTrue(self.client.should_trigger(80.0, track_id=11))
        # Generate caption (updates timestamp)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        caption = self.client.generate_caption(frame, bbox=(100, 100, 200, 300), context={"track_id": 11, "risk_score": 80})
        self.assertIsInstance(caption, str)
        self.assertGreater(len(caption), 10)

        # Immediate second trigger on same track must be blocked by cooldown
        self.assertFalse(self.client.should_trigger(85.0, track_id=11))

    def test_caption_content(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        ctx = {
            "track_id": 12,
            "camera_id": "CAM-01",
            "zone_id": "perimeter",
            "behavior": "loiter",
            "risk_score": 78.0,
            "is_night": True,
            "class": "person",
            "dwell_sec": 14.0
        }
        caption = self.client.generate_caption(frame, bbox=(50, 50, 150, 250), context=ctx)
        self.assertIn("loitering", caption.lower())
        self.assertIn("night", caption.lower())


class TestLLMClient(unittest.TestCase):
    def setUp(self):
        self.client = LLMClient(summary_interval_sec=2.0)

    def test_incident_summary(self):
        sample_alerts = [
            {
                "camera_id": "CAM-01",
                "track_id": 101,
                "zone_id": "perimeter",
                "risk_score": 85.0,
                "behavior": "intrusion",
                "vlm_caption": "Unknown person crossing restricted fence at night.",
                "timestamp": "23:15:00"
            }
        ]
        summary = self.client.generate_incident_summary(sample_alerts, time_window_str="Last 5 Minutes")
        self.assertIsInstance(summary, str)
        self.assertIn("SUMMARY", summary.upper())
        self.assertIn("ACTIONS", summary.upper())

    def test_operator_query_filtering(self):
        history = [
            {"track_id": 1, "zone_id": "approach_band", "behavior": "approach", "risk_score": 30.0},
            {"track_id": 2, "zone_id": "perimeter", "behavior": "intrusion", "risk_score": 88.0, "reasoning": "Breach at Gate-2"},
            {"track_id": 3, "zone_id": "parking", "behavior": "vehicle_presence", "risk_score": 20.0},
        ]
        result = self.client.answer_operator_query("Show me high-risk events in perimeter", history)
        self.assertIsInstance(result, dict)
        self.assertGreater(result["matched_count"], 0)
        self.assertEqual(result["matched_events"][0]["track_id"], 2)

    def test_operator_headcount_query(self):
        history = [
            {"track_id": 101, "zone_id": "perimeter", "behavior": "approach", "risk_score": 85.0},
            {"track_id": 102, "zone_id": "perimeter", "behavior": "intrusion", "risk_score": 90.0},
        ]
        live_ctx = {
            "active_camera_id": "CAM-01",
            "active_tracks": [{"track_id": 101, "class": "person", "zone_id": "perimeter"}],
            "system_metrics": {"status": "OPERATIONAL"}
        }
        res = self.client.answer_operator_query("How many people are there right now?", history, live_context=live_ctx)
        self.assertIn("answer", res)
        # Should mention active people on camera
        self.assertIn("1 person", res["answer"].lower())
        self.assertIn("CAM-01", res["answer"])


class TestReIDFusionEngine(unittest.TestCase):
    def setUp(self):
        self.reid = ReIDFusionEngine(enabled=True, similarity_threshold=0.70)
        self.single_cam_reid = ReIDFusionEngine(enabled=False)

    def test_feature_extractor(self):
        ext = AppearanceFeatureExtractor(feature_dim=128)
        img = np.ones((200, 100, 3), dtype=np.uint8) * 120
        emb = ext.extract(img, bbox=(0, 0, 100, 200))
        self.assertEqual(len(emb), 128)
        self.assertAlmostEqual(np.linalg.norm(emb), 1.0, places=3)

    def test_single_camera_dormant(self):
        # In single camera mode, local track ID maps deterministically
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 80
        gid = self.single_cam_reid.register_track("CAM-01", 5, frame, (100, 100, 150, 250))
        self.assertTrue(gid.startswith("GID-"))

    def test_two_camera_cross_match_simulation(self):
        # Enable multi-camera
        self.reid.cameras = [
            {"id": "CAM-01", "source": 0, "enabled": True},
            {"id": "CAM-02", "source": 2, "enabled": True}
        ]
        frame = np.ones((720, 1280, 3), dtype=np.uint8) * 150

        # Target appears in CAM-01
        gid_cam1 = self.reid.register_track("CAM-01", track_id=42, frame=frame, bbox=(1050, 100, 1150, 300))

        # Same target appears in CAM-02 within 2 seconds
        gid_cam2 = self.reid.register_track("CAM-02", track_id=1, frame=frame, bbox=(50, 100, 150, 300))

        # Because appearance is identical and transit plausibility holds, they match to same global ID
        self.assertEqual(gid_cam1, gid_cam2)

        events = self.reid.get_global_events()
        self.assertEqual(len(events), 1)
        self.assertIn("CAM-01", events[0]["camera_ids"])
        self.assertIn("CAM-02", events[0]["camera_ids"])


class TestFenceTamperDetector(unittest.TestCase):
    def setUp(self):
        self.detector = FenceTamperDetector(
            energy_spike_threshold=25.0,
            tamper_base_risk=85.0
        )
        self.detector.scale_to_frame(640, 480, ref_w=1280, ref_h=720)

    def test_clean_static_scene(self):
        frame1 = np.ones((480, 640, 3), dtype=np.uint8) * 100
        frame2 = np.ones((480, 640, 3), dtype=np.uint8) * 100
        self.detector.update(frame1)
        status = self.detector.update(frame2)
        self.assertFalse(status.is_tamper)
        self.assertEqual(status.alert_level, "NORMAL")

    def test_energy_spike_tamper_alert(self):
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        # Create sudden heavy disturbance in fence ROI
        frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2[340:420, :] = 255  # Intense brightness shift in ROI

        self.detector.update(frame1)
        status = self.detector.update(frame2, tracked_objects=[])

        self.assertTrue(status.is_tamper)
        self.assertEqual(status.tamper_type, "spike")
        self.assertEqual(status.alert_level, "CRITICAL")
        self.assertGreaterEqual(status.risk_score, 80.0)

    def test_person_crossing_suppresses_tamper_false_alarm(self):
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2[340:420, 200:300] = 200

        self.detector.update(frame1)
        # Tracked person overlapping the disturbed region
        person_track = {"bbox": [180, 320, 320, 450], "is_person": True}
        status = self.detector.update(frame2, tracked_objects=[person_track])

        # Legitimate person track present -> NOT classified as anonymous fence tamper
        self.assertFalse(status.is_tamper)


class TestVMSAPI(unittest.TestCase):
    def setUp(self):
        self.mgr = VMSStateManager(api_key="test-key")

    def test_webhook_registration_and_query(self):
        sub = self.mgr.register_webhook("http://127.0.0.1:9999/callback", description="Unit Test", min_risk=65.0)
        self.assertEqual(sub["callback_url"], "http://127.0.0.1:9999/callback")
        self.assertEqual(len(self.mgr.subscribers), 1)

    def test_event_ingestion_and_filtering(self):
        self.mgr.add_event({
            "camera_id": "CAM-01",
            "track_id": 99,
            "risk_score": 82.0,
            "alert_level": "CRITICAL",
            "behavior": "intrusion",
            "zone_id": "perimeter"
        })
        evs = self.mgr.get_events(camera_id="CAM-01", min_risk=70.0)
        self.assertEqual(len(evs), 1)
        self.assertEqual(evs[0]["track_id"], 99)

        # Min risk 90 should filter it out
        evs_high = self.mgr.get_events(min_risk=90.0)
        self.assertEqual(len(evs_high), 0)

    def test_mjpeg_synthetic_stream_rendering(self):
        # Should generate valid JPEG bytes even when no camera has pushed frames
        jpeg_bytes = self.mgr.get_latest_jpeg()
        self.assertIsInstance(jpeg_bytes, bytes)
        self.assertGreater(len(jpeg_bytes), 1000)
        # JPEG SOI marker check
        self.assertEqual(jpeg_bytes[:2], b'\xff\xd8')

    def test_live_frame_caching(self):
        dummy_frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
        self.mgr.set_latest_frame(dummy_frame, tracks=[{"track_id": 101}])
        jpeg = self.mgr.get_latest_jpeg()
        self.assertEqual(jpeg[:2], b'\xff\xd8')
        metrics = self.mgr.get_system_metrics()
        self.assertEqual(metrics["status"], "OPERATIONAL")
        self.assertIn("threat_level", metrics)

    def test_camera_selection_and_rendering(self):
        cams = self.mgr.get_cameras()
        self.assertEqual(len(cams), 1)
        self.assertEqual(cams[0]["id"], "CAM-01")
        self.assertEqual(cams[0]["name"], "Camera 1 (Real-Time Monitoring Feed)")
        self.assertTrue(cams[0]["is_active"])

        # Select CAM-01
        ok = self.mgr.select_camera("CAM-01")
        self.assertTrue(ok)
        self.assertEqual(self.mgr.active_camera_id, "CAM-01")

        # Renders valid standby JPEG feed
        standby_jpeg = self.mgr.get_latest_jpeg()
        self.assertEqual(standby_jpeg[:2], b'\xff\xd8')

        # Invalid camera
        bad = self.mgr.select_camera("CAM-99")
        self.assertFalse(bad)


class TestTCNTelemetryAndVLMImmediate(unittest.TestCase):
    def test_tcn_telemetry_retrieval(self):
        tcn = TemporalBehaviorModel(history_len=16)
        # Feed track points
        for cy in [650, 620, 590, 560, 530]:
            tcn.update_track(track_id=42, centroid=(600, cy), in_restricted_zone=True, dwell_sec=2.0)
        telem = tcn.get_track_telemetry(42)
        self.assertIsNotNone(telem)
        self.assertIn("dominant_behavior", telem)
        self.assertIn("attention_weights", telem)
        self.assertIn("speed_px", telem)
        self.assertIn("trajectory", telem)

    def test_vlm_immediate_trigger(self):
        client = VLMClient()
        dummy_frame = np.ones((200, 200, 3), dtype=np.uint8) * 100
        res = client.generate_caption_immediate(
            frame=dummy_frame,
            bbox=(50, 50, 150, 150),
            context={"track_id": 99, "zone_id": "perimeter", "behavior": "approach", "risk_score": 85.0}
        )
        self.assertIsInstance(res, dict)
        self.assertIn("caption", res)
        self.assertIn("snapshot", res)
        self.assertTrue(res["snapshot"].startswith("data:image/jpeg;base64,"))


if __name__ == "__main__":
    unittest.main()
