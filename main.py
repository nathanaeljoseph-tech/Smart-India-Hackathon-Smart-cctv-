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
  5. Main loop: read ΓåÆ detect ΓåÆ track ΓåÆ draw ΓåÆ display ΓåÆ export
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
    logger_pre.warning("modules/risk_engine.py not found ΓÇö Tier-1 risk scoring disabled.")


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
        default=0.45,
        help=(
            "Minimum confidence score (0.0-1.0) to accept a detection. "
            "Lower = more detections but more false positives. "
            "Higher = fewer detections but more accurate. "
            "Default: 0.45  (raised from 0.35 to reduce false positives on objects)"
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
            "Enable night-time mode. Applies a risk multiplier (├ù1.30 by default) "
            "to all risk scores, raising alert levels faster in low-visibility conditions."
        )
    )

    return parser.parse_args()


# ===========================================================================
# Main Application
# ===========================================================================

def main():
    """
    Main entry point. Orchestrates the entire detection + tracking pipeline.

    Flow:
        parse args ΓåÆ setup logging ΓåÆ open video ΓåÆ load model ΓåÆ init tracker
        ΓåÆ init exporter ΓåÆ main loop ΓåÆ shutdown
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
    # the boundary engine module is unavailable ΓÇö no errors, no alerts.
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
                # Convert polygon lists ΓåÆ tuples for geometry functions
                _z["polygon"] = [tuple(p) for p in _z.get("polygon", [])]
            boundary_zones = [z for z in _raw_zones if len(z.get("polygon", [])) >= 3]
            if boundary_zones:
                logger.info(f"Boundary Engine: {len(boundary_zones)} zone(s) loaded from {_BOUNDARY_CFG}")
            else:
                logger.info("Boundary Engine: config found but no valid zones ΓÇö running without alerts.")
        except FileNotFoundError:
            logger.info("Boundary Engine: config/boundary_config.yaml not found ΓÇö running without zones.")
        except Exception as _be_err:
            logger.warning(f"Boundary Engine: config load failed ({_be_err}) ΓÇö running without zones.")
    elif not _BOUNDARY_ENGINE_AVAILABLE:
        logger.info("Boundary Engine: modules/boundary_engine.py not found ΓÇö skipping.")
    else:
        logger.info("Boundary Engine: PyYAML not installed (pip install pyyaml) ΓÇö skipping.")

    # --- Step 3: Open video source ---
    cap = open_video_source(args.source, logger)

    # Get frame dimensions for later use
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Auto-scale boundary zones to match the actual stream resolution
    if _BOUNDARY_ENGINE_AVAILABLE and boundary_zones and frame_width > 0 and frame_height > 0:
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
    #   classes ΓÇö ignoring our classes=[0] filter. This caused pens, cups, and
    #   other objects to be detected and labeled on screen.
    #
    # WHY model.predict() + FallbackIOUTracker?
    #   model.predict(classes=[0]) is GUARANTEED to only detect people.
    #   It never silently changes behavior. FallbackIOUTracker then assigns
    #   consistent IDs across frames. This combination is 100% reliable.
    tracker = FallbackIOUTracker(
        iou_threshold   = 0.25,   # Lower threshold for large, close-range boxes
        max_lost_frames = 25,     # Keep IDs alive longer during brief occlusion
    )
    logger.info("Tracker: FallbackIOUTracker (predict+track pipeline ΓÇö people only).")

    # --- Step 5b: Initialize Risk Engine ---
    risk_engine: Optional[Any] = None
    if _RISK_ENGINE_AVAILABLE:
        risk_engine = RiskEngine(config=_risk_cfg)
        logger.info(
            f"Risk Engine: Tier-1 enabled | night_mode={args.night} | "
            f"approach_ratio={_risk_cfg.get('approach_band_ratio', 0.20)}"
        )
    else:
        logger.info("Risk Engine: unavailable ΓÇö running without Tier-1 risk scoring.")

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
        logger.info("Night mode ACTIVE ΓÇö risk multiplier applied to all scores.")

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
                    logger.info("Waiting 1 second then retrying...")
                    time.sleep(1)
                    continue
                else:
                    logger.info("End of video file reached. Exiting.")
                    break

            frame_id += 1

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

                # ---------------------------------------------------------------
                # STEP A: DETECT ΓÇö run YOLO on ALL 80 COCO classes
                # ---------------------------------------------------------------
                # detector.detect() uses classes=None ΓåÆ detects everything.
                # It returns a rich dict for each object with:
                #   class, class_id, bbox, confidence, is_person, color
                # This gives us correct labels for every object in the scene.
                dets = detector.detect(frame)

                # ---------------------------------------------------------------
                # STEP B: SPLIT ΓÇö people vs. other objects
                # Also apply too-close filtering before passing to tracker.
                # ---------------------------------------------------------------
                people_raw  = [d for d in dets if d["is_person"]]
                other_dets  = [d for d in dets if not d["is_person"]]

                # Too-close filtering:
                #   normal_dets  ΓåÆ safe to track (bbox height < threshold)
                #   very_close_dets ΓåÆ HUD-only (bbox almost fills frame height)
                if _RISK_ENGINE_AVAILABLE and frame_height > 0:
                    people_dets, very_close_dets = filter_too_close(
                        people_raw,
                        frame_h      = frame_height,
                        height_ratio = _too_close_ratio,
                        conf_threshold = _too_close_conf,
                    )
                else:
                    people_dets     = people_raw
                    very_close_dets = []

                # ---------------------------------------------------------------
                # STEP C: TRACK ΓÇö assign persistent IDs to normal-range people
                # ---------------------------------------------------------------
                tracks = tracker.update(people_dets)

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
                # TIER-1 RISK ENGINE: behaviour + progressive risk scoring
                # ---------------------------------------------------------------
                if risk_engine is not None:
                    alert_cards = risk_engine.update(
                        tracks             = tracks,
                        frame_h            = frame_height,
                        frame_w            = frame_width,
                        boundary_zones     = boundary_zones,
                        fps                = current_fps if current_fps and current_fps > 0 else 15.0,
                        frame_id           = proc_id,
                        night_mode         = args.night,
                        centroid_histories = track_histories,
                        very_close_dets    = very_close_dets,
                    )
                    # Mirror risk engine results into track_alerts for boundary
                    # engine drawing functions (backward compatible)
                    for tid, card in alert_cards.items():
                        track_alerts[tid] = {
                            "alert_type": card.get("alert_type", "intrusion"),
                            "severity"  : card.get("severity", "high"),
                            "zone_id"   : card.get("zone_id", "proximity"),
                        }

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
                person_count = len(all_people)
                other_count  = len(other_dets)

                if dets or very_close_dets:
                    parts = [
                        f"ID:{t['track_id']}(conf={t['confidence']:.2f})"
                        for t in last_tracks
                    ]
                    if very_close_dets:
                        parts.append("PROXIMITY BREACH (<30cm) [CRITICAL]")
                    person_summary = ", ".join(parts) if parts else "no IDs yet"
                    other_summary = ", ".join(
                        set(d["class"] for d in other_dets)
                    ) if other_dets else ""

                    logger.info(
                        f"Frame {frame_id:5d} | proc#{proc_id:4d} | "
                        f"FPS:{current_fps:5.1f} | "
                        f"People:{person_count} [{person_summary}] | "
                        f"Objects:{other_count} [{other_summary}]"
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
            frame = detector.draw_detections(frame, draw_dets, tracks)

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

            # Draw FPS overlay on top-left
            draw_fps_overlay(
                frame,
                current_fps,
                extra_info=f"{'GPU' if args.device not in ('cpu','CPU') else 'CPU'} | {args.tracker.upper()} | proc:{proc_id}"
            )

            # --- Proximity Breach Flashing Banner ---
            has_prox = bool(very_close_dets) or any(c.get("is_too_close") for c in alert_cards.values())
            if has_prox:
                banner_text = " [!] CRITICAL: CAMERA PROXIMITY BREACH / TAMPERING DETECTED (<30cm) "
                banner_h = 32
                # Pulsing red background
                banner_bg = (0, 0, 220) if (proc_id // 6) % 2 == 0 else (0, 0, 140)
                cv2.rectangle(frame, (0, 0), (frame_width, banner_h), banner_bg, -1)
                (btw, bth), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
                bx = max(10, (frame_width - btw) // 2)
                cv2.putText(frame, banner_text, (bx, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

            # ---------------------------------------------------------------
            # ALERT CARD PANELS (Tier-1) ΓÇö right side of frame
            # ---------------------------------------------------------------
            # Draw a compact alert card for each non-NORMAL track.
            # Each card shows: ID | BEHAVIOR | ALERT_LEVEL | risk bar | reasoning
            if alert_cards:
                _panel_x  = max(5, frame_width - 260)
                _panel_y  = 75
                _card_h   = 80
                _card_gap = 6
                _card_w   = 250

                _level_colors = {
                    "CRITICAL"  : (0,   0,   220),   # Red
                    "HIGH"      : (0,  90,   220),   # Orange-red
                    "SUSPICIOUS": (0, 180,   220),   # Amber
                    "NORMAL"    : (60,  60,   60),   # Dark gray
                }

                non_normal = [
                    (tid, card) for tid, card in alert_cards.items()
                    if card.get("alert_level", "NORMAL") != "NORMAL"
                ]

                for card_idx, (tid, card) in enumerate(non_normal[:4]):  # max 4 cards
                    cy1 = _panel_y + card_idx * (_card_h + _card_gap)
                    cy2 = cy1 + _card_h

                    level  = card.get("alert_level",  "NORMAL")
                    behav  = card.get("behavior",     "none").upper()
                    risk   = card.get("risk_score",   0.0)
                    reason = card.get("reasoning",    "")
                    col    = _level_colors.get(level, (60, 60, 60))

                    # Card background
                    overlay_c = frame.copy()
                    cv2.rectangle(overlay_c, (_panel_x-4, cy1), (_panel_x + _card_w, cy2), (15, 15, 15), -1)
                    cv2.addWeighted(overlay_c, 0.75, frame, 0.25, 0, frame)

                    # Coloured left border bar
                    cv2.rectangle(frame, (_panel_x-4, cy1), (_panel_x, cy2), col, -1)

                    # Card border
                    cv2.rectangle(frame, (_panel_x-4, cy1), (_panel_x + _card_w, cy2), col, 1)

                    # Header: ID + LEVEL badge
                    header = f"ID:{tid}  {level}"
                    cv2.putText(frame, header, (_panel_x + 4, cy1 + 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 2, cv2.LINE_AA)

                    # Behaviour label
                    cv2.putText(frame, f"Behavior: {behav}", (_panel_x + 4, cy1 + 33),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

                    # Reasoning (truncated)
                    reason_short = reason[:36] + "ΓÇª" if len(reason) > 36 else reason
                    cv2.putText(frame, reason_short, (_panel_x + 4, cy1 + 49),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

                    # Risk score progress bar
                    bar_x1  = _panel_x + 4
                    bar_x2  = _panel_x + _card_w - 4
                    bar_y   = cy1 + 62
                    bar_fill = int((bar_x2 - bar_x1) * min(risk, 100.0) / 100.0)
                    cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y + 10), (50, 50, 50), -1)
                    cv2.rectangle(frame, (bar_x1, bar_y), (bar_x1 + bar_fill, bar_y + 10), col, -1)
                    cv2.putText(frame, f"Risk: {risk:.0f}/100",
                                (bar_x1, bar_y + 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

                    # Mini risk bar overlaid on the actual bounding box (if visible)
                    bbox_t = card.get("bbox")
                    if bbox_t and len(bbox_t) == 4:
                        bx1, by1, bx2, by2 = [int(v) for v in bbox_t]
                        # Draw risk bar just above the bounding box
                        rb_y   = max(0, by1 - 12)
                        rb_len = bx2 - bx1
                        rb_fill = int(rb_len * min(risk, 100.0) / 100.0)
                        cv2.rectangle(frame, (bx1, rb_y), (bx2, rb_y + 6), (40, 40, 40), -1)
                        cv2.rectangle(frame, (bx1, rb_y), (bx1 + rb_fill, rb_y + 6), col, -1)

            # Draw person count + object count on top-right
            people_count_now = len([d for d in dets if d.get("is_person")])
            total_count_now  = len(dets)
            other_count_now  = total_count_now - people_count_now

            count_line1 = f"People: {people_count_now}"
            count_line2 = f"Objects: {other_count_now}"

            (tw1, th1), _ = cv2.getTextSize(count_line1, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            (tw2, th2), _ = cv2.getTextSize(count_line2, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            box_w = max(tw1, tw2) + 14

            cv2.rectangle(frame,
                          (frame_width - box_w - 5, 5),
                          (frame_width - 5, 65), (0, 0, 0), -1)
            cv2.putText(
                frame, count_line1,
                (frame_width - box_w, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 255) if people_count_now else (180, 180, 180),
                2, cv2.LINE_AA
            )
            cv2.putText(
                frame, count_line2,
                (frame_width - box_w, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (200, 200, 100) if other_count_now else (180, 180, 180),
                1, cv2.LINE_AA
            )

            # Draw pause indicator
            if is_paused:
                cv2.putText(
                    frame, "PAUSED", (frame_width // 2 - 60, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA
                )

            # ---------------------------------------------------------------
            # DISPLAY
            # ---------------------------------------------------------------
            cv2.imshow(window_name, frame)

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
