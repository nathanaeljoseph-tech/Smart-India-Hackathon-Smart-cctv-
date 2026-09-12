"""
main.py - Primary Entry Point for AMST Border-Net (DEV 1)
==========================================================
Smart India Hackathon 2026 | Problem: SIH26187
Team Role: DEV 1 - Camera + YOLO Detection + Tracking

AMST Border-Net: AI-based Intelligent Video Analytics Platform
for Border Surveillance using existing CCTV Infrastructure.

This is the main script that ties everything together:
  1. Parse command-line arguments
  2. Open video source (webcam / file / RTSP)
  3. Load YOLO11n model (CPU only)
  4. Initialize ByteTrack tracker
  5. Main loop: read ?????? detect ?????? track ?????? draw ?????? display ?????? export
  6. Graceful shutdown on 'q' or Ctrl+C

--- HOW TO RUN ---
  Webcam        : python main.py
  Video file    : python main.py --source video.mp4
  Different cam : python main.py --source 1
  Verbose logs  : python main.py --verbose
  Skip frames   : python main.py --skip-frames 1
  No JSON save  : python main.py --no-save

--- KEYBOARD CONTROLS ---
  q : Quit the application
  p : Pause / Resume video
  s : Screenshot (save current frame as PNG)
  r : Reset tracker (clear all track IDs)

Author : DEV 1
Date   : 2026-09-05
"""

import argparse
import cv2
import logging
import sys
import os
import time
from datetime import datetime
from typing import Optional, Dict, List

# --- Import our custom modules ---
# These are the DEV 1 modules in the same directory
from utils         import FPSCounter, setup_logging, open_video_source, \
                          warn_low_fps, draw_fps_overlay
from detector      import YOLODetector
from tracker       import ByteTrackWrapper, FallbackIOUTracker
from data_exporter import DataExporter

# --- Boundary Engine (Hook 1: import) ---
try:
    import yaml as _yaml
    _YAML_AVAILABLE = True
except ImportError:
    _yaml = None
    _YAML_AVAILABLE = False

try:
    from modules.boundary_engine import (
        compute_alert,
        draw_zones,
        draw_alert_box,
        draw_legend,
        scale_zones_to_frame,
    )
    _BOUNDARY_ENGINE_AVAILABLE = True
except ImportError:
    _BOUNDARY_ENGINE_AVAILABLE = False

# --- Risk Engine (Tier-1: import) ---
try:
    from modules.risk_engine import RiskEngine, filter_too_close
    _RISK_ENGINE_AVAILABLE = True
except ImportError:
    _RISK_ENGINE_AVAILABLE = False
    logger_pre = logging.getLogger("amst_border_net")
    logger_pre.warning("modules/risk_engine.py not found ?????? Tier-1 risk scoring disabled.")

# --- Camera Tamper / Visibility Monitor ---
try:
    from modules.tamper_detector import CameraTamperDetector, draw_tamper_overlay
    _TAMPER_DETECTOR_AVAILABLE = True
except ImportError:
    _TAMPER_DETECTOR_AVAILABLE = False

# --- Tier-2 Modules: Illumination & Abandoned Objects ---
try:
    from modules.illumination import IlluminationManager, get_illumination_state
    _ILLUMINATION_AVAILABLE = True
except ImportError:
    _ILLUMINATION_AVAILABLE = False

try:
    from modules.abandoned_engine import AbandonedObjectEngine, _bbox_iou
    _ABANDONED_ENGINE_AVAILABLE = True
except ImportError:
    _ABANDONED_ENGINE_AVAILABLE = False
    def _bbox_iou(b1, b2): return 0.0

try:
    from modules.zone_context import ZoneContextMemory
    _ZONE_CONTEXT_AVAILABLE = True
except ImportError:
    _ZONE_CONTEXT_AVAILABLE = False

# --- Tier-3 Modules: TCN Temporal, VLM, LLM, Re-ID, Fence Tamper, VMS API ---
try:
    from modules.temporal_behavior import TemporalBehaviorModel
    _TEMPORAL_MODEL_AVAILABLE = True
except ImportError:
    _TEMPORAL_MODEL_AVAILABLE = False

try:
    from modules.vlm_client import VLMClient
    _VLM_AVAILABLE = True
except ImportError:
    _VLM_AVAILABLE = False

try:
    from modules.llm_client import LLMClient
    _LLM_AVAILABLE = True
except ImportError:
    _LLM_AVAILABLE = False

try:
    from modules.reid_fusion import ReIDFusionEngine
    _REID_AVAILABLE = True
except ImportError:
    _REID_AVAILABLE = False

try:
    from modules.tamper_detector import FenceTamperDetector, draw_fence_tamper_overlay, FenceTamperStatus
    _FENCE_TAMPER_AVAILABLE = True
except ImportError:
    _FENCE_TAMPER_AVAILABLE = False

try:
    from modules.vms_api import VMSServerThread, state_manager
    _VMS_API_AVAILABLE = True
except ImportError:
    _VMS_API_AVAILABLE = False



# ===========================================================================
# Argument Parser
# ===========================================================================

def parse_args() -> argparse.Namespace:
    """
    Define and parse all command-line arguments.

    WHY command-line args?
        Hardcoding values (like model path, confidence) in the code means
        you need to edit and restart the script for every change. CLI args
        let you tweak settings without touching the code.

    Returns:
        argparse.Namespace with all parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "AMST Border-Net | DEV 1 | SIH 2026\n"
            "AI-based people detection & tracking using YOLO11 + ByteTrack\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main.py                          # Webcam 0\n"
            "  python main.py --source video.mp4       # Local video\n"
            "  python main.py --source 1               # Webcam 1\n"
            "  python main.py --verbose --skip-frames 1\n"
        )
    )

    # --- Input source ---
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help=(
            "Video source: '0' for webcam, camera index (e.g., '1'), "
            "path to video file (.mp4/.avi), or RTSP URL. "
            "Default: '0' (default webcam)."
        )
    )

    # --- YOLO model ---
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help=(
            "YOLO model weights file. Use 'yolo11n.pt' for fastest CPU speed. "
            "'yolo11s.pt' is slightly slower but more accurate. "
            "Default: yolo11n.pt (auto-downloaded on first run)."
        )
    )

    # --- Confidence threshold ---
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.35,
        help=(
            "Minimum confidence score (0.0-1.0) to accept a detection. "
            "Lower = more detections but more false positives. "
            "Higher = fewer detections but more accurate. "
            "Default: 0.35 (optimized for animals, screens, and border monitoring)"
        )
    )

    # --- Frame skipping ---
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=0,
        help=(
            "Process every Nth+1 frame. 0 = process every frame. "
            "1 = process every 2nd frame (doubles effective FPS budget). "
            "Use this if FPS drops below 8 on your CPU. "
            "Default: 0"
        )
    )

    # --- Input resize ---
    parser.add_argument(
        "--input-size",
        type=int,
        default=640,
        help=(
            "Resize frames to this size (pixels) before YOLO inference. "
            "Smaller = faster but less accurate. "
            "Recommended: 416 for speed, 640 for accuracy. "
            "Default: 640"
        )
    )

    # --- Tracker type ---
    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack",
        choices=["bytetrack", "botsort", "fallback"],
        help=(
            "Tracker algorithm. 'bytetrack' = best (default). "
            "'botsort' = alternative in Ultralytics. "
            "'fallback' = pure Python IOU tracker (no extra deps). "
            "Default: bytetrack"
        )
    )

    # --- Output directory ---
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save JSON export files. Default: output/"
    )

    # --- Batch size for JSON save ---
    parser.add_argument(
        "--batch-size",
        type=int,
        default=30,
        help="Number of frames to buffer before saving JSON. Default: 30"
    )

    # --- Disable JSON saving ---
    parser.add_argument(
        "--no-save",
        action="store_true",
        default=False,
        help="Disable JSON export (good for pure demo mode to max FPS)."
    )

    # --- Verbose logging ---
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging (very detailed output)."
    )

    # --- Display window size ---
    parser.add_argument(
        "--window-scale",
        type=float,
        default=1.0,
        help=(
            "Scale the display window. 0.5 = half size, 1.0 = original. "
            "Useful for small screens. Default: 1.0"
        )
    )

    # --- Compute device ---
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help=(
            "Device for YOLO inference: 'auto' (GPU if available, else CPU), "
            "'0' / 'cuda' for NVIDIA GPU, or 'cpu'. Default: 'auto'"
        )
    )

    # --- Night mode ---
    parser.add_argument(
        "--night",
        action="store_true",
        default=False,
        help=(
            "Enable manual night-time mode. Forces night risk multiplier and CLAHE."
        )
    )

    # --- Tier-2: Animal Suppression ---
    parser.add_argument(
        "--enable-animal-suppression",
        dest="enable_animal_suppression",
        action="store_true",
        default=True,
        help="Enable suppression of false alarms from animals (default: True)."
    )
    parser.add_argument(
        "--no-animal-suppression",
        dest="enable_animal_suppression",
        action="store_false",
        help="Disable animal suppression (treat animals with standard risk)."
    )

    # --- Tier-2: Abandoned Object Detection ---
    parser.add_argument(
        "--enable-abandoned-object",
        dest="enable_abandoned_object",
        action="store_true",
        default=True,
        help="Enable abandoned object and unattended luggage detection (default: True)."
    )
    parser.add_argument(
        "--no-abandoned-object",
        dest="enable_abandoned_object",
        action="store_false",
        help="Disable abandoned object detection."
    )
    parser.add_argument(
        "--abandoned-stationary-threshold",
        type=float,
        default=10.0,
        help="Seconds an object must remain stationary to be considered abandoned. Default: 10.0"
    )
    parser.add_argument(
        "--abandoned-owner-distance",
        type=float,
        default=120.0,
        help="Distance in pixels between owner and object before considered separated. Default: 120.0"
    )
    parser.add_argument(
        "--abandoned-owner-absent-threshold",
        type=float,
        default=5.0,
        help="Seconds owner must be absent before triggering abandoned alert. Default: 5.0"
    )

    # --- Tier-2: Day/Night Preprocessing (CLAHE + Illumination) ---
    parser.add_argument(
        "--enable-illumination-auto",
        dest="enable_illumination_auto",
        action="store_true",
        default=True,
        help="Automatically detect day vs night and apply CLAHE enhancement (default: True)."
    )
    parser.add_argument(
        "--no-auto-night",
        dest="enable_illumination_auto",
        action="store_false",
        help="Disable automatic day/night detection."
    )
    parser.add_argument(
        "--night-low-thresh",
        type=float,
        default=50.0,
        help="Luminance threshold below which state switches to NIGHT. Default: 50.0"
    )
    parser.add_argument(
        "--night-high-thresh",
        type=float,
        default=70.0,
        help="Luminance threshold above which state switches to DAY. Default: 70.0"
    )
    parser.add_argument(
        "--clahe-clip",
        type=float,
        default=2.0,
        help="Contrast limit for CLAHE preprocessing. Default: 2.0"
    )

    # --- Tier-3: TCN Temporal Behavior Model ---
    parser.add_argument(
        "--enable-tcn",
        dest="enable_tcn",
        action="store_true",
        default=True,
        help="Enable 1D TCN + Attention Temporal Behavior Model (default: True)."
    )
    parser.add_argument(
        "--no-tcn",
        dest="enable_tcn",
        action="store_false",
        help="Disable TCN Temporal Behavior Model."
    )

    # --- Tier-3: VLM Qwen2-VL ---
    parser.add_argument(
        "--enable-vlm",
        dest="enable_vlm",
        action="store_true",
        default=True,
        help="Enable Qwen2-VL VLM for high-risk incident captioning (default: True)."
    )
    parser.add_argument(
        "--no-vlm",
        dest="enable_vlm",
        action="store_false",
        help="Disable VLM incident captioning."
    )

    # --- Tier-3: LLM Llama 3.1 ---
    parser.add_argument(
        "--enable-llm",
        dest="enable_llm",
        action="store_true",
        default=True,
        help="Enable Llama 3.1 8B LLM for summaries and operator chat (default: True)."
    )
    parser.add_argument(
        "--no-llm",
        dest="enable_llm",
        action="store_false",
        help="Disable LLM summarizer and query engine."
    )

    # --- Tier-3: Re-ID Multi-Camera ---
    parser.add_argument(
        "--enable-reid",
        dest="enable_reid",
        action="store_true",
        default=False,
        help="Enable two-camera Re-ID fusion (default: False, single-camera mode)."
    )

    # --- Tier-3: Fence Tamper Detection ---
    parser.add_argument(
        "--enable-fence-tamper",
        dest="enable_fence_tamper",
        action="store_true",
        default=True,
        help="Enable fence frame-diff energy tamper detector (default: True)."
    )
    parser.add_argument(
        "--no-fence-tamper",
        dest="enable_fence_tamper",
        action="store_false",
        help="Disable fence tamper detector."
    )

    # --- Tier-3: VMS REST API Server ---
    parser.add_argument(
        "--vms-port",
        type=int,
        default=8080,
        help="Port for VMS REST API & Dashboard. Default: 8080"
    )
    parser.add_argument(
        "--no-vms",
        action="store_true",
        default=False,
        help="Disable VMS REST API server."
    )

    return parser.parse_args()


# ===========================================================================
# Main Application
# ===========================================================================

def main():
    """
    Main entry point. Orchestrates the entire detection + tracking pipeline.

    Flow:
        parse args ?????? setup logging ?????? open video ?????? load model ?????? init tracker
        ?????? init exporter ?????? main loop ?????? shutdown
    """

    # --- Step 1: Parse arguments ---
    args = parse_args()

    # --- Step 2: Setup logging ---
    # Must be done before any other module logs anything
    logger = setup_logging(verbose=args.verbose)
    logger.info("=" * 60)
    logger.info("  AMST Border-Net | SIH 2026 | DEV 1")
    logger.info("  AI People Detection & Tracking")
    logger.info("=" * 60)
    logger.info(f"Arguments: {vars(args)}")

    # --- Boundary Engine (Hook 1: load config) ---
    # Gracefully skip if config file is missing, yaml is not installed, or
    # the boundary engine module is unavailable ?????? no errors, no alerts.
    boundary_zones: List[Dict] = []
    _ref_res   = None
    _risk_cfg  : Dict = {}   # Full config dict forwarded to RiskEngine
    _BOUNDARY_CFG = os.path.join(os.path.dirname(__file__), "config", "boundary_config.yaml")

    if _BOUNDARY_ENGINE_AVAILABLE and _YAML_AVAILABLE:
        try:
            with open(_BOUNDARY_CFG, "r", encoding="utf-8") as _f:
                _cfg = _yaml.safe_load(_f) or {}
            _ref_res  = _cfg.get("reference_resolution")
            _risk_cfg = _cfg   # Pass the full config to RiskEngine
            _raw_zones = _cfg.get("zones", []) or []
            for _z in _raw_zones:
                # Convert polygon lists ?????? tuples for geometry functions
                _z["polygon"] = [tuple(p) for p in _z.get("polygon", [])]
            boundary_zones = [z for z in _raw_zones if len(z.get("polygon", [])) >= 3]
            if boundary_zones:
                logger.info(f"Boundary Engine: {len(boundary_zones)} zone(s) loaded from {_BOUNDARY_CFG}")
            else:
                logger.info("Boundary Engine: config found but no valid zones ?????? running without alerts.")
        except FileNotFoundError:
            logger.info("Boundary Engine: config/boundary_config.yaml not found ?????? running without zones.")
        except Exception as _be_err:
            logger.warning(f"Boundary Engine: config load failed ({_be_err}) ?????? running without zones.")
    elif not _BOUNDARY_ENGINE_AVAILABLE:
        logger.info("Boundary Engine: modules/boundary_engine.py not found ?????? skipping.")
    else:
        logger.info("Boundary Engine: PyYAML not installed (pip install pyyaml) ?????? skipping.")

    # --- Step 3: Open video source ---
    cap = open_video_source(args.source, logger)

    # Get frame dimensions for later use
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # --- Initialize Zone Context Memory (Tier-2) and Auto-Scale Zones ---
    zone_context_mgr: Optional[Any] = None
    if _ZONE_CONTEXT_AVAILABLE:
        zone_context_mgr = ZoneContextMemory(boundary_zones)
        if frame_width > 0 and frame_height > 0:
            zone_context_mgr.scale_zones_to_frame(frame_width, frame_height, ref_res=_ref_res)
        boundary_zones = zone_context_mgr.get_all_zones()
        logger.info(f"ZoneContextMemory: Tier-2 memory active with {len(boundary_zones)} sample/configured zones.")
    elif _BOUNDARY_ENGINE_AVAILABLE and boundary_zones and frame_width > 0 and frame_height > 0:
        boundary_zones = scale_zones_to_frame(
            boundary_zones,
            frame_width=frame_width,
            frame_height=frame_height,
            reference_resolution=_ref_res
        )
        ref_str = f" from reference {_ref_res[0]}x{_ref_res[1]}" if (_ref_res and len(_ref_res) == 2) else ""
        logger.info(f"Boundary Engine: Auto-scaled {len(boundary_zones)} zone(s) to match frame ({frame_width}x{frame_height}){ref_str}")

    # --- Step 4: Initialize YOLO Detector ---
    detector = YOLODetector(
        model_path     = args.model,
        conf_threshold = args.conf_threshold,
        input_size     = args.input_size,
        device         = args.device,
    )

    try:
        detector.load_model()
    except Exception as e:
        logger.error(f"Could not load YOLO model: {e}")
        logger.error("Exiting. Please check your installation.")
        cap.release()
        sys.exit(1)

    # --- Step 5: Initialize Tracker ---
    #
    # DESIGN DECISION (important for hackathon reliability):
    #   We use model.predict() + FallbackIOUTracker instead of model.track().
    #
    # WHY NOT model.track()?
    #   model.track(tracker="bytetrack") requires a bytetrack.yaml config file
    #   to be present in the Ultralytics package. If it's missing or the version
    #   doesn't match, model.track() silently falls back to detecting ALL 80 COCO
    #   classes ?????? ignoring our classes=[0] filter. This caused pens, cups, and
    #   other objects to be detected and labeled on screen.
    #
    # WHY model.predict() + FallbackIOUTracker?
    #   model.predict(classes=[0]) is GUARANTEED to only detect people.
    #   It never silently changes behavior. FallbackIOUTracker then assigns
    #   consistent IDs across frames. This combination is 100% reliable.
    use_bytetrack = (args.tracker in ("bytetrack", "botsort"))
    if use_bytetrack:
        detector.set_tracker_type(args.tracker)
        tracker = FallbackIOUTracker(
            iou_threshold   = 0.25,
            max_lost_frames = 25,
        )
        logger.info(f"Tracker: {args.tracker.upper()} via model.track() (Kalman + ByteTrack pipeline active).")
    else:
        tracker = FallbackIOUTracker(
            iou_threshold   = 0.25,
            max_lost_frames = 25,
        )
        logger.info("Tracker: FallbackIOUTracker (pure Python IOU tracker).")

    # --- Step 5b: Initialize Risk Engine ---
    risk_engine: Optional[Any] = None
    if _RISK_ENGINE_AVAILABLE:
        risk_engine = RiskEngine(config=_risk_cfg)
        logger.info(
            f"Risk Engine: Tier-1 enabled | night_mode={args.night} | "
            f"approach_ratio={_risk_cfg.get('approach_band_ratio', 0.20)}"
        )
    else:
        logger.info("Risk Engine: unavailable — running without Tier-1 risk scoring.")

    # --- Step 5c: Initialize Tier-2 Illumination Manager ---
    illum_cfg = _risk_cfg.get("illumination", {}) if isinstance(_risk_cfg, dict) else {}
    low_th = float(args.night_low_thresh if args.night_low_thresh != 50.0 else illum_cfg.get("low_thresh", 50.0))
    high_th = float(args.night_high_thresh if args.night_high_thresh != 70.0 else illum_cfg.get("high_thresh", 70.0))
    clip_lim = float(args.clahe_clip if args.clahe_clip != 2.0 else illum_cfg.get("clahe_clip", 2.0))

    illumination_mgr: Optional[Any] = None
    if _ILLUMINATION_AVAILABLE:
        illumination_mgr = IlluminationManager(
            low_thresh=low_th,
            high_thresh=high_th,
            clahe_clip=clip_lim,
            initial_night=args.night,
        )
        logger.info(
            f"IlluminationManager: Tier-2 enabled | auto={args.enable_illumination_auto} | "
            f"low={low_th} | high={high_th} | clahe_clip={clip_lim}"
        )

    # --- Step 5d: Initialize Tier-2 Multi-Category Abandoned Object Engine ---
    ab_cfg = _risk_cfg.get("abandoned", {}) if isinstance(_risk_cfg, dict) else {}
    stat_th = float(args.abandoned_stationary_threshold if args.abandoned_stationary_threshold != 10.0 else ab_cfg.get("stationary_threshold", 10.0))
    stat_veh = float(ab_cfg.get("stationary_threshold_vehicle", 15.0))
    stat_dev = float(ab_cfg.get("stationary_threshold_device", 12.0))
    dist_th = float(args.abandoned_owner_distance if args.abandoned_owner_distance != 120.0 else ab_cfg.get("owner_distance_threshold", 120.0))
    abs_th = float(args.abandoned_owner_absent_threshold if args.abandoned_owner_absent_threshold != 5.0 else ab_cfg.get("owner_absent_threshold", 5.0))
    night_th_mult = float(ab_cfg.get("night_threshold_multiplier", 0.70))

    abandoned_engine: Optional[Any] = None
    if _ABANDONED_ENGINE_AVAILABLE and args.enable_abandoned_object:
        abandoned_engine = AbandonedObjectEngine(
            stationary_threshold=stat_th,
            stationary_threshold_vehicle=stat_veh,
            stationary_threshold_device=stat_dev,
            owner_distance_threshold=dist_th,
            owner_absent_threshold=abs_th,
            night_threshold_multiplier=night_th_mult,
        )
        logger.info(
            f"AbandonedObjectEngine: Tier-2 Multi-Category enabled | bag={stat_th}s | "
            f"vehicle={stat_veh}s | device={stat_dev}s | owner_dist={dist_th}px"
        )

    # --- Step 5e: Initialize Tier-2 Zone Context Memory ---
    zone_context_mgr: Optional[Any] = None
    if _ZONE_CONTEXT_AVAILABLE:
        cfg_zones = _risk_cfg.get("zones", []) if isinstance(_risk_cfg, dict) else []
        zone_context_mgr = ZoneContextMemory(zones=cfg_zones)
        zone_context_mgr.scale_zones_to_frame(frame_width, frame_height)
        logger.info(f"ZoneContextMemory: Tier-2 loaded with {len(zone_context_mgr.zones)} configured zone(s).")

    # --- Step 5f: Initialize Tier-3 Temporal Behavior Model ---
    temporal_behavior_mgr: Optional[Any] = None
    if _TEMPORAL_MODEL_AVAILABLE and args.enable_tcn:
        temporal_behavior_mgr = TemporalBehaviorModel(
            config=_risk_cfg,
            boundary_y_ref=576.0 * (frame_height / 720.0),
        )
        logger.info("TemporalBehaviorModel: Tier-3 TCN+Attention enabled.")

    # --- Step 5g: Initialize Tier-3 VLM Client ---
    vlm_client: Optional[Any] = None
    if _VLM_AVAILABLE and args.enable_vlm:
        vlm_client = VLMClient(config=_risk_cfg)
        logger.info("VLMClient: Tier-3 Qwen2-VL enabled.")

    # --- Step 5h: Initialize Tier-3 LLM Client ---
    llm_client: Optional[Any] = None
    if _LLM_AVAILABLE and args.enable_llm:
        llm_client = LLMClient(config=_risk_cfg)
        logger.info("LLMClient: Tier-3 Llama 3.1 8B enabled.")

    # --- Step 5i: Initialize Tier-3 Re-ID Fusion Engine ---
    reid_engine: Optional[Any] = None
    if _REID_AVAILABLE:
        reid_engine = ReIDFusionEngine(config=_risk_cfg, enabled=args.enable_reid)
        logger.info(f"ReIDFusionEngine: Tier-3 ready | multi_camera_active={reid_engine.is_multi_camera_active()}")

    # --- Step 5j: Initialize Tier-3 Fence-Tamper Detector ---
    fence_tamper: Optional[Any] = None
    if _FENCE_TAMPER_AVAILABLE and args.enable_fence_tamper:
        fence_tamper = FenceTamperDetector(config=_risk_cfg)
        fence_tamper.scale_to_frame(frame_width, frame_height)
        logger.info("FenceTamperDetector: Tier-3 Frame-Diff Energy monitor enabled.")

    # --- Step 5k: Initialize Tier-3 VMS REST API Server & Webhook Dispatcher ---
    vms_server: Optional[Any] = None
    if _VMS_API_AVAILABLE and not args.no_vms:
        state_manager.llm_client = llm_client
        state_manager.vlm_client = vlm_client
        state_manager.reid_engine = reid_engine
        state_manager.temporal_behavior_mgr = temporal_behavior_mgr
        vms_server = VMSServerThread(port=args.vms_port)
        vms_server.start()
        logger.info(f"VMS API Gateway: Tier-3 server live on port {args.vms_port} (Dashboard: /dashboard)")

    # --- Step 6: Initialize Data Exporter ---
    exporter = DataExporter(
        output_dir = args.output_dir,
        batch_size = args.batch_size,
        save_json  = not args.no_save,
    )

    # --- Step 7: Initialize FPS Counter ---
    fps_counter = FPSCounter(window_size=30)

    # --- Step 8: Create Display Window ---
    window_name = "AMST Border-Net | DEV 1 | Press Q to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(
        window_name,
        int(frame_width  * args.window_scale),
        int(frame_height * args.window_scale)
    )

    # --- State Variables ---
    frame_id              = 0     # Total frames read from source
    proc_id               = 0     # Frames actually processed (after skipping)
    is_paused             = False # Pause flag
    last_tracks           = []    # Last known tracks (used for skipped frames)
    last_dets             = []    # Last known detections
    last_very_close_dets  = []    # Last known very-close detections (HUD only)

    # --- Tier-2 State Variables ---
    is_night_active       = args.night
    current_brightness    = 100.0
    last_abandoned_events = []

    # FPS warning throttle: only warn once every 5 seconds
    last_fps_warn_time = 0.0

    # --- Boundary Engine (Hook 2: state vars) ---
    # track_histories: per-track centroid history for loitering detection.
    # track_alerts: current alert status per track ID.
    # HISTORY_MAX_LEN: cap at ~10 seconds of history at 15 fps.
    track_histories : Dict[int, List[Dict]] = {}
    track_alerts    : Dict[int, Dict]       = {}
    HISTORY_MAX_LEN = 150

    # --- Tier-1: alert cards dict (populated by RiskEngine each frame) ---
    alert_cards: Dict[int, Dict] = {}

    # --- Tier-1: too-close filter thresholds (from yaml config) ---
    _too_close_ratio = float(_risk_cfg.get("too_close_height_ratio",  0.85))
    _too_close_conf  = float(_risk_cfg.get("too_close_conf_threshold", 0.60))

    if args.night:
        logger.info("Night mode ACTIVE ?????? risk multiplier applied to all scores.")

    # --- Camera Tamper / Visibility Monitor ---
    tamper_detector = CameraTamperDetector() if _TAMPER_DETECTOR_AVAILABLE else None
    if tamper_detector:
        logger.info("CameraTamperDetector active.")

    logger.info("Main loop starting. Press 'q' to quit, 'p' to pause.")
    logger.info("-" * 60)

    # ===========================================================================
    # Main Loop
    # ===========================================================================
    try:
        while True:

            # --- Pause handling ---
            if is_paused:
                key = cv2.waitKey(30) & 0xFF
                if key == ord('p'):
                    is_paused = False
                    logger.info("Resumed.")
                elif key == ord('q'):
                    break
                continue

            # --- Read frame ---
            ret, frame = cap.read()

            if not ret:
                logger.warning(
                    "Could not read frame from source. "
                    "End of video or camera disconnected."
                )
                # For webcam: try to reconnect briefly before giving up
                if isinstance(args.source, str) and args.source.isdigit():
                    if tamper_detector is not None:
                        import numpy as _np
                        _blk = _np.zeros((frame_height, frame_width, 3), dtype="uint8")
                        draw_tamper_overlay(_blk, tamper_detector.check_feed_lost(), frame_id)
                        cv2.imshow(window_name, _blk)
                        cv2.waitKey(1)
                    logger.info("Waiting 1 second then retrying...")
                    time.sleep(1)
                    continue
                else:
                    logger.info("End of video file reached. Exiting.")
                    break

            frame_id += 1

            # --- Camera Tamper / Visibility Check ---
            if tamper_detector is not None:
                _ts = tamper_detector.check(frame)
                if _ts.is_fault:
                    draw_tamper_overlay(frame, _ts, frame_id)
                    cv2.imshow(window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q") or key == 27:
                        logger.info("Quit requested during tamper alert.")
                        break
                    continue

            # --- FPS tick (count every read frame for true FPS) ---
            fps_counter.tick()
            current_fps = fps_counter.get()

            # --- Skip frame logic ---
            # If skip_frames=1, process frame 1, skip 2, process 3, skip 4...
            # This halves the processing load on CPU-heavy frames
            should_process = (frame_id % (args.skip_frames + 1) == 1) \
                             if args.skip_frames > 0 else True

            if should_process:
                proc_id += 1

                # --- Tier-2: Day/Night Illumination Preprocessing (only on processed frames) ---
                if illumination_mgr is not None:
                    enhanced_frame, illum_state = illumination_mgr.process_frame(
                        frame,
                        force_night=True if args.night else None,
                        apply_enhancement=args.enable_illumination_auto,
                    )
                    is_night_active = illum_state["is_night"]
                    current_brightness = illum_state["brightness"]
                    det_input_frame = enhanced_frame if is_night_active else frame
                else:
                    det_input_frame = frame
                    is_night_active = args.night

                # ---------------------------------------------------------------
                # STEP A & C: DETECT & TRACK (ByteTrack or Fallback)
                # ---------------------------------------------------------------
                if use_bytetrack:
                    dets, tracks = detector.track(det_input_frame)
                    people_raw  = [d for d in dets if d.get("is_person")]
                    other_dets  = [d for d in dets if not d.get("is_person")]

                    if _RISK_ENGINE_AVAILABLE and frame_height > 0:
                        people_dets, very_close_dets = filter_too_close(
                            people_raw,
                            frame_h        = frame_height,
                            height_ratio   = _too_close_ratio,
                            conf_threshold = _too_close_conf,
                        )
                    else:
                        people_dets     = people_raw
                        very_close_dets = []
                else:
                    dets = detector.detect(det_input_frame)
                    people_raw  = [d for d in dets if d.get("is_person")]
                    other_dets  = [d for d in dets if not d.get("is_person")]

                    if _RISK_ENGINE_AVAILABLE and frame_height > 0:
                        people_dets, very_close_dets = filter_too_close(
                            people_raw,
                            frame_h        = frame_height,
                            height_ratio   = _too_close_ratio,
                            conf_threshold = _too_close_conf,
                        )
                    else:
                        people_dets     = people_raw
                        very_close_dets = []

                    # Fallback tracker matches people and animals
                    animal_dets = [d for d in other_dets if d.get("is_animal")]
                    tracks = tracker.update(people_dets + animal_dets)

                # ---------------------------------------------------------------
                # TIER-2: ABANDONED OBJECT DETECTION
                # ---------------------------------------------------------------
                abandoned_events = []
                if abandoned_engine is not None:
                    safe_fps = current_fps if current_fps and current_fps > 0 else 15.0
                    person_tracks = [t for t in tracks if t.get("is_person") and not t.get("is_animal")]
                    abandoned_events = abandoned_engine.update(
                        dets,
                        person_tracks,
                        fps=safe_fps,
                        zone_context_mgr=zone_context_mgr,
                        current_time=datetime.now().time(),
                        is_night=is_night_active,
                    )
                last_abandoned_events = abandoned_events

                # ---------------------------------------------------------------
                # TIER-3: FENCE-TAMPER DETECTION (Frame-Diff Energy)
                # ---------------------------------------------------------------
                fence_tamper_status = None
                if fence_tamper is not None:
                    safe_fps = current_fps if current_fps and current_fps > 0 else 15.0
                    fence_tamper_status = fence_tamper.update(frame, tracks, fps=safe_fps)

                # ---------------------------------------------------------------
                # TIER-3: RE-ID FEATURE EXTRACTION & TRACK REGISTRATION
                # ---------------------------------------------------------------
                if reid_engine is not None:
                    for t in tracks:
                        gid = reid_engine.register_track("CAM-01", t["track_id"], frame, t["bbox"])
                        t["global_track_id"] = gid

                last_tracks          = tracks
                last_dets            = dets
                last_very_close_dets = very_close_dets

                # ---------------------------------------------------------------
                # BOUNDARY ENGINE (Hook 3): update histories & compute alerts
                # AND RISK ENGINE (Tier-1): behaviour + risk scoring
                # ---------------------------------------------------------------
                # The boundary engine populates track_alerts for zone overlays.
                # The risk engine produces richer alert cards with risk scores,
                # behaviour labels, and reasoning strings.
                if boundary_zones and _BOUNDARY_ENGINE_AVAILABLE:
                    track_alerts.clear()
                    safe_fps = current_fps if current_fps and current_fps > 0 else 15.0

                    for t in tracks:
                        tid = t["track_id"]

                        # Intrusion and loitering alerts are strictly for PERSON tracks
                        is_person = bool(t.get("is_person", False) or str(t.get("class", "")).lower() == "person")
                        if not is_person:
                            track_alerts[tid] = {
                                "alert_type": "none",
                                "severity":   "none",
                                "zone_id":    None,
                            }
                            continue

                        cx  = int((t["bbox"][0] + t["bbox"][2]) / 2)
                        cy  = int((t["bbox"][1] + t["bbox"][3]) / 2)

                        # Use EMA centroid if available (smoother)
                        if "ema_centroid" in t:
                            cx, cy = t["ema_centroid"]

                        # Maintain capped centroid history
                        hist = track_histories.setdefault(tid, [])
                        hist.append({"xy": (cx, cy), "frame_id": proc_id})
                        if len(hist) > HISTORY_MAX_LEN:
                            hist.pop(0)

                        alert_type = "none"
                        severity   = "none"
                        zone_id    = None

                        for zone in boundary_zones:
                            a_type, a_sev = compute_alert(
                                (cx, cy), hist, zone, safe_fps
                            )
                            if a_type == "none":
                                continue
                            if a_type == "intrusion":
                                alert_type = a_type
                                severity   = a_sev
                                zone_id    = zone.get("id", "zone")
                                break
                            if a_type == "loitering" and alert_type != "intrusion":
                                alert_type = a_type
                                severity   = a_sev
                                zone_id    = zone.get("id", "zone")

                        track_alerts[tid] = {
                            "alert_type": alert_type,
                            "severity":   severity,
                            "zone_id":    zone_id,
                        }

                    # Prune histories for tracks that have disappeared
                    active_ids = {t["track_id"] for t in tracks}
                    for _old_id in list(track_histories.keys()):
                        if _old_id not in active_ids:
                            del track_histories[_old_id]

                # ---------------------------------------------------------------
                # TIER-1 & TIER-2 RISK ENGINE: behaviour + progressive risk scoring
                # ---------------------------------------------------------------
                if risk_engine is not None:
                    alert_cards = risk_engine.update(
                        tracks                 = tracks,
                        frame_h                = frame_height,
                        frame_w                = frame_width,
                        boundary_zones         = boundary_zones,
                        fps                    = current_fps if current_fps and current_fps > 0 else 15.0,
                        frame_id               = proc_id,
                        night_mode             = is_night_active,
                        centroid_histories     = track_histories,
                        very_close_dets        = very_close_dets,
                        current_time           = datetime.now().time(),
                        abandoned_events       = abandoned_events,
                        animal_suppression     = args.enable_animal_suppression,
                        zone_context_mgr       = zone_context_mgr,
                        temporal_behavior_mgr  = temporal_behavior_mgr,
                        fence_tamper_status    = fence_tamper_status,
                    )
                    # Mirror risk engine results into track_alerts for boundary
                    # engine drawing functions (backward compatible)
                    for tid, card in alert_cards.items():
                        c_type = card.get("alert_type", "none")
                        if c_type in ("intrusion", "loitering"):
                            track_alerts[tid] = {
                                "alert_type": c_type,
                                "severity"  : card.get("severity", "high"),
                                "zone_id"   : card.get("zone_id", "proximity"),
                            }

                # ---------------------------------------------------------------
                # TIER-3: VLM CAPTIONING & VMS WEBHOOK DISPATCH
                # ---------------------------------------------------------------
                if alert_cards:
                    for tid, card in alert_cards.items():
                        c_risk = float(card.get("risk_score", 0.0))
                        # Trigger VLM on high risk events
                        if vlm_client is not None and vlm_client.should_trigger(c_risk, tid):
                            c_bbox = card.get("bbox")
                            c_ctx = {
                                "track_id"  : tid,
                                "camera_id" : "CAM-01",
                                "zone_id"   : card.get("zone_id", "perimeter"),
                                "behavior"  : card.get("behavior", "intrusion"),
                                "risk_score": c_risk,
                                "is_night"  : is_night_active,
                                "class"     : card.get("class", "person"),
                            }
                            vlm_cap = vlm_client.generate_caption(frame, c_bbox, c_ctx)
                            card["vlm_caption"] = vlm_cap

                        # Relay high-risk alerts to VMS state manager / webhooks
                        if _VMS_API_AVAILABLE and c_risk >= 50.0:
                            snap_uri = vlm_client.get_track_snapshot(tid) if vlm_client is not None else None
                            ev_data = {
                                "camera_id"      : "CAM-01",
                                "track_id"       : tid,
                                "global_track_id": card.get("global_track_id", f"GID-{tid:03d}" if isinstance(tid, int) else str(tid)),
                                "risk_score"     : c_risk,
                                "alert_level"    : card.get("alert_level", "HIGH"),
                                "behavior"       : card.get("behavior", "unknown"),
                                "zone_id"        : card.get("zone_id", "perimeter"),
                                "vlm_caption"    : card.get("vlm_caption"),
                                "reasoning"      : card.get("reasoning"),
                                "snapshot"       : snap_uri,
                                "timestamp"      : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                            state_manager.add_event(ev_data)
                            if vms_server is not None and getattr(vms_server, "is_external", False):
                                vms_server.relay_alert(ev_data)

                # Check periodic LLM incident summary
                if llm_client is not None and llm_client.should_generate_summary():
                    recent_evs = state_manager.get_events(limit=15) if _VMS_API_AVAILABLE else []
                    summary_txt = llm_client.generate_incident_summary(recent_evs)
                    logger.info(f"\n[PERIODIC SECURITY BRIEF]\n{summary_txt}\n")

                # ---------------------------------------------------------------
                # BOUNDARY ENGINE (Hook 4): annotate track dicts before export
                # ---------------------------------------------------------------
                # Inject alert_type + severity + zone_id into each track dict.
                # If the risk engine is active, its cards take priority.
                for t in tracks:
                    tid  = t["track_id"]
                    card = alert_cards.get(tid, {})
                    info = track_alerts.get(
                        tid,
                        {"alert_type": "none", "severity": "none", "zone_id": None}
                    )
                    t["alert_type"] = card.get("alert_type", info["alert_type"])
                    t["severity"]   = card.get("severity",   info["severity"])
                    t["zone_id"]    = card.get("zone_id",    info["zone_id"])

                # ---------------------------------------------------------------
                # EXPORT DATA
                # ---------------------------------------------------------------
                all_people = list(people_dets) + list(very_close_dets)
                frame_data = exporter.add_frame(
                    proc_id, all_people, last_tracks,
                    alert_cards=alert_cards if risk_engine is not None else None
                )

                # --- Console log summary ---
                person_tracks = [t for t in last_tracks if t.get("is_person")]
                animal_tracks = [t for t in last_tracks if t.get("is_animal")]
                person_count = len(all_people)
                animal_count = len([d for d in dets if d.get("is_animal")])
                other_count  = max(0, len(other_dets) - animal_count)

                if dets or very_close_dets:
                    p_parts = [
                        f"ID:{t['track_id']}(conf={t['confidence']:.2f})"
                        for t in person_tracks
                    ]
                    if very_close_dets:
                        p_parts.append("PROXIMITY BREACH (<30cm) [CRITICAL]")
                    person_summary = ", ".join(p_parts) if p_parts else "no IDs yet"

                    a_parts = [
                        f"ID:{t['track_id']}({t.get('class', 'animal')},conf={t['confidence']:.2f})"
                        for t in animal_tracks
                    ]
                    animal_summary = ", ".join(a_parts) if a_parts else ("present" if animal_count > 0 else "")

                    non_animal_dets = [d for d in other_dets if not d.get("is_animal")]
                    other_summary = ", ".join(
                        set(d["class"] for d in non_animal_dets)
                    ) if non_animal_dets else ""

                    log_msg = (
                        f"Frame {frame_id:5d} | proc#{proc_id:4d} | "
                        f"FPS:{current_fps:5.1f} | "
                        f"People:{person_count} [{person_summary}]"
                    )
                    if animal_count > 0 or animal_tracks:
                        log_msg += f" | Animals:{max(animal_count, len(animal_tracks))} [{animal_summary}]"
                    if other_count > 0:
                        log_msg += f" | Objects:{other_count} [{other_summary}]"

                    logger.info(log_msg)

                    if last_abandoned_events:
                        for ab_ev in last_abandoned_events:
                            ab_cat = ab_ev.get("category", "object").upper()
                            ab_cls = ab_ev.get("class", "object")
                            ab_tid = ab_ev.get("track_id", f"obj_{ab_ev['object_id']}")
                            ab_dw = ab_ev.get("stationary_time", 0.0)
                            ab_zn = ab_ev.get("zone_name") or ab_ev.get("zone_id", "zone")
                            ab_zt = ab_ev.get("zone_type", "none")
                            logger.warning(
                                f"[ABANDONED {ab_cat}] Track: {ab_tid} ({ab_cls}) | "
                                f"Dwell: {ab_dw:.0f}s | Zone: {ab_zn} ({ab_zt})"
                            )
                elif frame_id % 30 == 0:
                    logger.info(
                        f"Frame {frame_id:5d} | proc#{proc_id:4d} | "
                        f"FPS:{current_fps:5.1f} | Scene empty."
                    )

                # Verbose: full JSON dump of this frame's data
                if args.verbose and dets:
                    import json
                    logger.debug(
                        f"Frame data:\n{json.dumps(frame_data, indent=2)}"
                    )

            else:
                # Skipped frame: reuse last known tracks/dets for drawing
                dets             = last_dets
                tracks           = last_tracks
                very_close_dets  = last_very_close_dets

            # ---------------------------------------------------------------
            # DRAW
            # ---------------------------------------------------------------
            # Draw bounding boxes and labels using the detector's draw method.
            # Merge very_close_dets into dets for drawing (yellow corner markers).
            draw_dets = list(dets) + list(very_close_dets)

            # Mark abandoned detections if any
            if last_abandoned_events:
                ab_boxes = [ab["bbox"] for ab in last_abandoned_events]
                for d in draw_dets:
                    for ab_box in ab_boxes:
                        if _bbox_iou(d["bbox"], ab_box) > 0.4:
                            d["is_abandoned"] = True

            frame = detector.draw_detections(frame, draw_dets, tracks)

            # --- Draw Abandoned Objects (Tier-2) ---
            for ab_ev in last_abandoned_events:
                ax1, ay1, ax2, ay2 = ab_ev["bbox"]
                ab_cat = ab_ev.get("category", "bag").upper()
                ab_cls = ab_ev.get("class", "object").upper()
                ab_zn = ab_ev.get("zone_name") or ab_ev.get("zone_id", "Zone")
                ab_tid = ab_ev.get("track_id", f"obj_{ab_ev['object_id']}")
                ab_col = (0, 69, 255) if (proc_id // 5) % 2 == 0 else (0, 140, 255)
                cv2.rectangle(frame, (ax1, ay1), (ax2, ay2), ab_col, 3)
                ab_lbl = f"[ABANDONED {ab_cat}: {ab_cls}] {ab_tid} ({ab_ev['stationary_time']:.0f}s) [{ab_zn}]"
                (atw, ath), _ = cv2.getTextSize(ab_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 2)
                cv2.rectangle(frame, (ax1, max(0, ay1 - ath - 8)), (ax1 + atw + 6, ay1), ab_col, -1)
                cv2.putText(frame, ab_lbl, (ax1 + 3, ay1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 2, cv2.LINE_AA)

            # --- Boundary Engine (Hook 5): zone overlays + alert boxes ---
            if boundary_zones and _BOUNDARY_ENGINE_AVAILABLE:
                # 1. Semi-transparent zone polygons (drawn below alert boxes)
                draw_zones(frame, boundary_zones, track_alerts)

                # 2. Alert boxes override the default cyan person box
                for t in (last_tracks if not should_process else tracks):
                    info = track_alerts.get(
                        t["track_id"],
                        {"alert_type": "none", "severity": "none"}
                    )
                    if info.get("alert_type", "none") != "none":
                        draw_alert_box(
                            frame, t,
                            info["alert_type"],
                            info.get("severity", "high")
                        )

                # 3. Legend in bottom-left
                draw_legend(frame, has_zones=True)

            # --- Draw Fence-Tamper Overlay (Tier-3) ---
            if fence_tamper is not None and fence_tamper_status is not None:
                draw_fence_tamper_overlay(frame, fence_tamper_status, fence_tamper.polygon, proc_id)

            # Draw FPS overlay on top-left
            draw_fps_overlay(
                frame,
                current_fps,
                extra_info=f"{'GPU' if args.device not in ('cpu','CPU') else 'CPU'} | {args.tracker.upper()} | proc:{proc_id}"
            )

            # --- Night Mode Badge (Tier-2) ---
            if is_night_active:
                night_badge = f" NIGHT MODE [CLAHE ACTIVE] (Bri: {current_brightness:.0f}) "
                (nbw, nbh), _ = cv2.getTextSize(night_badge, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 2)
                nx = max(10, frame_width - nbw - 300)
                cv2.rectangle(frame, (nx - 4, 10), (nx + nbw + 4, 34), (40, 15, 90), -1)
                cv2.rectangle(frame, (nx - 4, 10), (nx + nbw + 4, 34), (180, 50, 255), 1)
                cv2.putText(frame, night_badge, (nx, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2, cv2.LINE_AA)

            # --- Proximity Breach Flashing Banner ---
            has_prox = bool(very_close_dets) or any(c.get("is_too_close") for c in alert_cards.values())
            if has_prox:
                banner_text = " [!] CRITICAL: CAMERA PROXIMITY BREACH / TAMPERING DETECTED (<30cm) "
                banner_h = 32
                banner_bg = (0, 0, 220) if (proc_id // 6) % 2 == 0 else (0, 0, 140)
                cv2.rectangle(frame, (0, 0), (frame_width, banner_h), banner_bg, -1)
                (btw, bth), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
                bx = max(10, (frame_width - btw) // 2)
                cv2.putText(frame, banner_text, (bx, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

            # ---------------------------------------------------------------
            # ALERT CARD PANELS (Tier-1 & Tier-2) — right side of frame
            # ---------------------------------------------------------------
            if alert_cards:
                _panel_x  = max(5, frame_width - 260)
                _panel_y  = 75
                _card_h   = 82
                _card_gap = 6
                _card_w   = 250

                _level_colors = {
                    "CRITICAL"  : (0,   0,   220),   # Red
                    "HIGH"      : (0,  90,   220),   # Orange-red
                    "SUSPICIOUS": (0, 180,   220),   # Amber
                    "NORMAL"    : (60,  60,   60),   # Dark gray
                }

                _cards_to_show = [
                    (tid, card) for tid, card in alert_cards.items()
                    if card.get("alert_level", "NORMAL") != "NORMAL" or card.get("behavior") == "animal_presence"
                ]

                for card_idx, (tid, card) in enumerate(_cards_to_show[:4]):  # max 4 cards
                    cy1 = _panel_y + card_idx * (_card_h + _card_gap)
                    cy2 = cy1 + _card_h

                    level  = card.get("alert_level",  "NORMAL")
                    behav  = card.get("behavior",     "none").upper()
                    risk   = card.get("risk_score",   0.0)
                    reason = card.get("reasoning",    "")
                    is_animal_supp = (card.get("behavior") == "animal_presence" or card.get("is_animal", False))
                    if is_animal_supp:
                        col = (200, 40, 180)  # Purple/magenta for animal suppression
                    else:
                        col = _level_colors.get(level, (60, 60, 60))

                    # Card background
                    overlay_c = frame.copy()
                    cv2.rectangle(overlay_c, (_panel_x-4, cy1), (_panel_x + _card_w, cy2), (15, 15, 15), -1)
                    cv2.addWeighted(overlay_c, 0.75, frame, 0.25, 0, frame)

                    # Coloured left border bar
                    cv2.rectangle(frame, (_panel_x-4, cy1), (_panel_x, cy2), col, -1)
                    cv2.rectangle(frame, (_panel_x-4, cy1), (_panel_x + _card_w, cy2), col, 1)

                    # Header: ID + LEVEL badge
                    if is_animal_supp:
                        header = f"ID:{tid}  ANIMAL SUPPRESSED"
                    else:
                        header = f"ID:{tid}  {level}"
                    cv2.putText(frame, header, (_panel_x + 4, cy1 + 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 2, cv2.LINE_AA)

                    # Behaviour / Context label
                    if is_animal_supp:
                        animal_name = card.get("class", "animal").upper()
                        behav_lbl = f"Animal: {animal_name} (Risk: 0)"
                    elif card.get("context_violation"):
                        behav_lbl = f"{behav} [CONTEXT]"
                    elif card.get("is_abandoned"):
                        cat = card.get("category", "object").upper()
                        cls_name = card.get("class", "").upper()
                        behav_lbl = f"ABANDONED {cat}: {cls_name}" if cls_name else f"ABANDONED {cat}"
                    else:
                        behav_lbl = f"Behavior: {behav}"
                    cv2.putText(frame, behav_lbl, (_panel_x + 4, cy1 + 33),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1, cv2.LINE_AA)

                    # Reasoning (truncated)
                    if card.get("is_abandoned"):
                        z_info = card.get("zone_name") or card.get("zone_id", "Zone")
                        reason_short = f"Zone: {z_info} | {card.get('dwell_sec', 0):.0f}s dwell"
                    else:
                        reason_short = reason[:35] + "…" if len(reason) > 35 else reason
                    cv2.putText(frame, reason_short, (_panel_x + 4, cy1 + 49),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 180), 1, cv2.LINE_AA)

                    # Risk score progress bar
                    bar_x1  = _panel_x + 4
                    bar_x2  = _panel_x + _card_w - 4
                    bar_y   = cy1 + 62
                    bar_fill = int((bar_x2 - bar_x1) * min(risk, 100.0) / 100.0)
                    cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y + 8), (50, 50, 50), -1)
                    cv2.rectangle(frame, (bar_x1, bar_y), (bar_x1 + bar_fill, bar_y + 8), col, -1)
                    cv2.putText(frame, f"Risk: {risk:.0f}/100",
                                (bar_x1, bar_y + 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 180, 180), 1, cv2.LINE_AA)

                    # Mini risk bar overlaid on the actual bounding box (if visible)
                    bbox_t = card.get("bbox")
                    if bbox_t and len(bbox_t) == 4:
                        bx1, by1, bx2, by2 = [int(v) for v in bbox_t]
                        rb_y   = max(0, by1 - 12)
                        rb_len = bx2 - bx1
                        rb_fill = int(rb_len * min(risk, 100.0) / 100.0)
                        cv2.rectangle(frame, (bx1, rb_y), (bx2, rb_y + 6), (40, 40, 40), -1)
                        cv2.rectangle(frame, (bx1, rb_y), (bx1 + rb_fill, rb_y + 6), col, -1)

            # Draw counts on top-right: People, Animals, Objects, Abandoned
            people_count_now = len([d for d in dets if d.get("is_person")])
            animal_count_now = len([d for d in dets if d.get("is_animal")])
            total_count_now  = len(dets)
            other_count_now  = max(0, total_count_now - people_count_now - animal_count_now)
            ab_count_now     = len(last_abandoned_events)

            count_line1 = f"People: {people_count_now} | Animals: {animal_count_now}"
            count_line2 = f"Objects: {other_count_now} | Abandoned: {ab_count_now}"

            (tw1, th1), _ = cv2.getTextSize(count_line1, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
            (tw2, th2), _ = cv2.getTextSize(count_line2, cv2.FONT_HERSHEY_SIMPLEX, 0.54, 1)
            box_w = max(tw1, tw2) + 16

            cv2.rectangle(frame,
                          (frame_width - box_w - 5, 5),
                          (frame_width - 5, 65), (0, 0, 0), -1)
            cv2.putText(
                frame, count_line1,
                (frame_width - box_w, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                (0, 255, 255) if people_count_now else (180, 180, 180),
                2, cv2.LINE_AA
            )
            cv2.putText(
                frame, count_line2,
                (frame_width - box_w, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.54,
                (0, 140, 255) if ab_count_now else (200, 200, 100),
                1, cv2.LINE_AA
            )

            # Draw pause indicator
            if is_paused:
                cv2.putText(
                    frame, "PAUSED", (frame_width // 2 - 60, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA
                )

            # ---------------------------------------------------------------
            # DISPLAY & VMS WEB STREAM BRIDGE
            # ---------------------------------------------------------------
            cv2.imshow(window_name, frame)
            if _VMS_API_AVAILABLE:
                state_manager.set_latest_frame(frame, tracks=last_tracks)
                if vms_server is not None and getattr(vms_server, "is_external", False):
                    vms_server.relay_frame(frame, tracks=last_tracks)

            # ---------------------------------------------------------------
            # FPS Warning
            # ---------------------------------------------------------------
            now = time.time()
            if now - last_fps_warn_time > 5.0:  # Check every 5 seconds
                warn_low_fps(current_fps, threshold=8.0, logger=logger)
                last_fps_warn_time = now

            # ---------------------------------------------------------------
            # KEY HANDLING
            # ---------------------------------------------------------------
            key = cv2.waitKey(1) & 0xFF   # waitKey(1) = non-blocking, 1ms wait

            if key == ord('q') or key == 27:   # 'q' or Escape
                logger.info("Quit requested by user.")
                break

            elif key == ord('p'):               # Pause/Resume
                is_paused = True
                logger.info("Paused. Press 'p' to resume.")

            elif key == ord('s'):               # Screenshot
                ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"screenshot_{ts}.png"
                cv2.imwrite(fname, frame)
                logger.info(f"Screenshot saved: {fname}")

            elif key == ord('r'):               # Reset tracker
                tracker.reset()
                if use_bytetrack and hasattr(detector, 'reset_tracker'):
                    detector.reset_tracker()
                if risk_engine is not None:
                    risk_engine.reset()
                track_histories.clear()
                track_alerts.clear()
                alert_cards.clear()
                logger.info("Tracker + Risk Engine reset. Track IDs restarted from 1.")

    except KeyboardInterrupt:
        # Ctrl+C handling: exit cleanly instead of showing a stack trace
        logger.info("\nKeyboard interrupt received. Shutting down...")

    finally:
        # -----------------------------------------------------------------------
        # SHUTDOWN - Always runs, even if an exception occurred
        # -----------------------------------------------------------------------
        logger.info("Performing clean shutdown...")

        # Save any remaining buffered data
        if not args.no_save and exporter.buffer:
            ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
            fin_path = os.path.join(args.output_dir, f"final_session_{ts}.json")
            exporter.save_to_json(fin_path)

        # Print session summary
        stats = exporter.get_stats()
        logger.info("--- Session Summary ---")
        logger.info(f"  Total frames read   : {frame_id}")
        logger.info(f"  Frames processed    : {proc_id}")
        logger.info(f"  Total detections    : {stats['total_detections']}")
        logger.info(f"  Unique track IDs    : {stats['unique_track_ids']}")
        logger.info(f"  JSON batches saved  : {stats['saved_batches']}")
        logger.info("-----------------------")

        # Release video capture
        if cap.isOpened():
            cap.release()
            logger.info("Camera/video released.")

        # Stop VMS server thread if active
        if vms_server is not None:
            vms_server.stop()
            logger.info("VMS REST API server stopped.")

        # Destroy all OpenCV windows
        cv2.destroyAllWindows()
        logger.info("All windows closed. Goodbye!")


# ===========================================================================
# Script Entry Point
# ===========================================================================

if __name__ == "__main__":
    """
    Python executes this block when you run: python main.py
    The 'if __name__ == "__main__"' guard ensures main() is only
    called when this file is run directly, not when imported.
    """
    main()
