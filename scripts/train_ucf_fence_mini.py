"""
scripts/train_ucf_fence_mini.py
================================
AMST Border-Net | SIH 2026 | DEV 1
UCF Fence Mini-Dataset Evaluation Script

Iterates over all frame-sequence clips in the UCF Fence mini-dataset,
runs the existing detection + tracking pipeline on each, and saves a
summary JSON with per-clip statistics.

Usage:
  python scripts/train_ucf_fence_mini.py
  python scripts/train_ucf_fence_mini.py --headless
  python scripts/train_ucf_fence_mini.py --headless --max-videos 6
  python scripts/train_ucf_fence_mini.py --headless --class fence_climbing
  python scripts/train_ucf_fence_mini.py --headless --max-videos 2 --save-video

Notes:
  - Does NOT modify main.py, detector.py, tracker.py, data_exporter.py, or utils.py
  - The dataset is stored as image-frame sequences (not raw video files).
    Each clip folder contains sorted .jpg/.png frames that are fed in order
    as a synthetic video stream to the detection + tracking pipeline.
  - Annotated output videos are written via cv2.VideoWriter (15 fps by default).
"""

import argparse
import cv2
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so we can import our modules
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from detector import YOLODetector, COCO_CLASSES
from tracker import FallbackIOUTracker

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ucf_fence_eval")


# ---------------------------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="UCF Fence Mini-Dataset Evaluation | AMST Border-Net | SIH 2026",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/train_ucf_fence_mini.py --headless\n"
            "  python scripts/train_ucf_fence_mini.py --headless --max-videos 6\n"
            "  python scripts/train_ucf_fence_mini.py --headless --class fence_climbing\n"
            "  python scripts/train_ucf_fence_mini.py --headless --save-video\n"
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "config" / "ucf_fence_mini.yaml"),
        help="Path to ucf_fence_mini.yaml config file.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying any OpenCV windows (for server/CI use).",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        metavar="N",
        help="Limit the total number of clip-videos processed across all classes.",
    )
    parser.add_argument(
        "--class",
        dest="only_class",
        type=str,
        default=None,
        metavar="CLASS_NAME",
        help="Process only clips from this class (e.g. fence_climbing, fighting).",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Write annotated output videos (.mp4) to output/ucf_fence_mini/videos/.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override YOLO model path from config.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# ---------------------------------------------------------------------------
# Frame loader ΓÇö sorts frames by numeric index extracted from filename
# ---------------------------------------------------------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def load_frames_sorted(clip_dir: str):
    """
    Return sorted list of absolute image paths in clip_dir.
    Sorts by the numeric suffix in the filename so frames are in temporal order.
    """
    p = Path(clip_dir)
    frames = [f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS]

    def frame_num(path: Path) -> int:
        stem = path.stem  # e.g. "Fence Climbing 1_frame_000445"
        # Try _frame_NNNNNN pattern
        if "_frame_" in stem:
            try:
                return int(stem.split("_frame_")[-1])
            except ValueError:
                pass
        # Try _x264_NNNNNN pattern
        if "_x264_" in stem:
            try:
                return int(stem.split("_x264_")[-1])
            except ValueError:
                pass
        # Fallback: last numeric token
        tokens = stem.replace("_", " ").split()
        for t in reversed(tokens):
            if t.isdigit():
                return int(t)
        return 0

    frames.sort(key=frame_num)
    return [str(f) for f in frames]


# ---------------------------------------------------------------------------
# Draw helpers (inline, no dependency on main.py)
# ---------------------------------------------------------------------------

def draw_detections(frame, tracks, person_count: int, frame_idx: int, fps: float):
    """Draw track boxes and overlay HUD on frame."""
    h, w = frame.shape[:2]

    for t in tracks:
        x1, y1, x2, y2 = t["bbox"]
        tid = t["track_id"]
        conf = t["confidence"]
        label = f"ID:{tid} ({conf:.2f})"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, label, (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    # HUD overlay
    hud = f"FPS:{fps:.1f}  People:{person_count}  Frame:{frame_idx}"
    cv2.putText(frame, hud, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
    return frame


# ---------------------------------------------------------------------------
# Process a single clip
# ---------------------------------------------------------------------------

def process_clip(
    class_name: str,
    clip_name: str,
    clip_dir: str,
    detector: YOLODetector,
    cfg: dict,
    headless: bool,
    save_video: bool,
    output_videos_dir: str,
) -> dict:
    """
    Run detection + tracking on a single frame-sequence clip.

    Mirrors the exact pipeline from main.py:
      detect() ΓåÆ filter people ΓåÆ FallbackIOUTracker.update(people_dets)

    Returns a stats dict:
      {
        class, clip, total_frames, processed_frames, duration_sec,
        unique_track_ids, avg_persons_per_frame, max_persons_in_frame,
        processing_time_sec, effective_fps
      }
    """
    frame_paths = load_frames_sorted(clip_dir)
    total_frames = len(frame_paths)

    if total_frames == 0:
        logger.warning(f"  [{class_name}/{clip_name}] No frames found in {clip_dir}")
        return {}

    logger.info(f"  [{class_name}/{clip_name}] {total_frames} frames ΓåÆ running pipeline...")

    # Fresh FallbackIOUTracker per clip ΓÇö mirrors main.py exactly
    tracker = FallbackIOUTracker(
        iou_threshold=cfg["tracking"].get("iou_threshold", 0.3),
        max_lost_frames=cfg["tracking"].get("max_age", 20),
    )

    # Video writer setup
    video_writer = None
    if save_video:
        os.makedirs(output_videos_dir, exist_ok=True)
        out_path = os.path.join(output_videos_dir, f"{class_name}_{clip_name}.mp4")
        sample = cv2.imread(frame_paths[0])
        if sample is not None:
            fh, fw = sample.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps_out = cfg.get("output", {}).get("output_fps", 15)
            video_writer = cv2.VideoWriter(out_path, fourcc, fps_out, (fw, fh))

    # Per-frame stats
    all_track_ids = set()
    persons_per_frame = []
    t_start = time.perf_counter()

    for frame_idx, frame_path in enumerate(frame_paths):
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        # STEP A: Detect ΓÇö same as main.py: detector.detect(frame)
        detections = detector.detect(frame)

        # STEP B: Split ΓÇö people vs others (same as main.py)
        people_dets = [d for d in detections if d.get("is_person")]

        # STEP C: Track ΓÇö FallbackIOUTracker.update(people_dets) ΓÇö same signature as main.py
        tracks = tracker.update(people_dets)

        person_count = len(tracks) if tracks else len(people_dets)
        persons_per_frame.append(person_count)
        for t in tracks:
            all_track_ids.add(t["track_id"])

        # Optional display
        if not headless:
            elapsed = time.perf_counter() - t_start
            fps_now = (frame_idx + 1) / max(elapsed, 1e-6)
            vis = draw_detections(frame.copy(), tracks, person_count, frame_idx, fps_now)
            cv2.imshow(f"UCF Eval | {class_name}/{clip_name}", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("User quit during display.")
                break

        if video_writer is not None:
            elapsed = time.perf_counter() - t_start
            fps_now = (frame_idx + 1) / max(elapsed, 1e-6)
            vis = draw_detections(frame.copy(), tracks, person_count, frame_idx, fps_now)
            video_writer.write(vis)

    processing_time = time.perf_counter() - t_start

    if video_writer is not None:
        video_writer.release()
    if not headless:
        cv2.destroyAllWindows()

    avg_persons = (sum(persons_per_frame) / len(persons_per_frame)) if persons_per_frame else 0.0
    max_persons = max(persons_per_frame) if persons_per_frame else 0

    stats = {
        "class": class_name,
        "clip": clip_name,
        "clip_dir": clip_dir,
        "total_frames": total_frames,
        "processed_frames": len(persons_per_frame),
        "duration_sec": round(total_frames / 30.0, 2),   # estimated at 30fps
        "unique_track_ids": len(all_track_ids),
        "avg_persons_per_frame": round(avg_persons, 3),
        "max_persons_in_frame": max_persons,
        "processing_time_sec": round(processing_time, 2),
        "effective_fps": round(len(persons_per_frame) / max(processing_time, 1e-6), 2),
    }

    logger.info(
        f"    ΓåÆ {stats['processed_frames']} frames | "
        f"tracks={stats['unique_track_ids']} | "
        f"avg_people={stats['avg_persons_per_frame']:.2f} | "
        f"max_people={stats['max_persons_in_frame']} | "
        f"fps={stats['effective_fps']:.1f}"
    )
    return stats



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load config
    logger.info(f"Loading config: {args.config}")
    cfg = load_config(args.config)

    dataset_root = cfg["dataset_root"]
    output_root  = cfg.get("output", {}).get("root", "output/ucf_fence_mini")
    summary_path = cfg.get("output", {}).get("summary_json", os.path.join(output_root, "summary.json"))
    videos_dir   = cfg.get("output", {}).get("videos_dir", os.path.join(output_root, "videos"))

    os.makedirs(output_root, exist_ok=True)

    # Build detector
    det_cfg = cfg.get("detection", {})
    model_path = args.model or det_cfg.get("model", "yolo11n.pt")
    image_size = det_cfg.get("image_size", 512)
    conf_thr   = det_cfg.get("conf_threshold", 0.35)
    class_ids  = det_cfg.get("class_ids", None)  # None = all 80 classes

    logger.info(f"Initialising YOLODetector | model={model_path} | size={image_size}")
    detector = YOLODetector(
        model_path=model_path,
        conf_threshold=conf_thr,
        input_size=image_size,
        target_classes=class_ids,
    )
    detector.load_model()

    # Collect clips to process
    class_configs = cfg.get("classes", [])
    all_clips = []
    for cls_cfg in class_configs:
        cls_name = cls_cfg["name"]
        if args.only_class and cls_name != args.only_class:
            continue
        for clip_name in cls_cfg.get("clips", []):
            clip_dir = os.path.join(dataset_root, cls_name, clip_name)
            if not os.path.isdir(clip_dir):
                logger.warning(f"Clip directory not found, skipping: {clip_dir}")
                continue
            all_clips.append((cls_name, clip_name, clip_dir))

    if args.max_videos is not None:
        all_clips = all_clips[: args.max_videos]

    logger.info(f"Processing {len(all_clips)} clip(s)...")

    results = []
    session_start = datetime.now().isoformat()

    for i, (cls_name, clip_name, clip_dir) in enumerate(all_clips):
        logger.info(f"[{i+1}/{len(all_clips)}] {cls_name} / {clip_name}")
        stats = process_clip(
            class_name=cls_name,
            clip_name=clip_name,
            clip_dir=clip_dir,
            detector=detector,
            cfg=cfg,
            headless=args.headless,
            save_video=args.save_video,
            output_videos_dir=videos_dir,
        )
        if stats:
            results.append(stats)

    # Build summary
    total_frames = sum(r.get("processed_frames", 0) for r in results)
    total_time   = sum(r.get("processing_time_sec", 0.0) for r in results)
    summary = {
        "session_start": session_start,
        "session_end": datetime.now().isoformat(),
        "config": args.config,
        "model": model_path,
        "clips_processed": len(results),
        "total_frames_processed": total_frames,
        "total_processing_time_sec": round(total_time, 2),
        "overall_fps": round(total_frames / max(total_time, 1e-6), 2),
        "clips": results,
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"  UCF Fence Eval Complete")
    logger.info(f"  Clips processed : {len(results)}")
    logger.info(f"  Total frames    : {total_frames}")
    logger.info(f"  Total time      : {total_time:.1f}s")
    logger.info(f"  Summary saved   : {summary_path}")
    if args.save_video:
        logger.info(f"  Videos saved    : {videos_dir}")
    logger.info(f"{'='*60}\n")


if __name__ == "__main__":
    main()
