<<<<<<< HEAD
# AMST Border-Net | DEV 1 Module

> **Smart India Hackathon 2026**  
> Problem: **SIH26187** ΓÇö AI-Based Intelligent Video Analytics Platform for Border Surveillance  
> Module: **Camera + YOLO Detection + Tracking** (DEV 1)

---

## ≡ƒôï Project Overview

This module is DEV 1's responsibility in the 5-member SIH team. It provides:
- **Live video ingestion** (webcam, video file, RTSP stream)
- **YOLO11n people detection** (CPU-only, no NVIDIA GPU required)
- **ByteTrack / IOU tracking** (persistent track IDs across frames)
- **Structured JSON data export** (for DEV 2 ΓÇö Event Detection to consume)
- **Real-time display** with bounding boxes, track IDs, FPS counter

---

## ≡ƒûÑ∩╕Å Hardware Requirements

| Component | Required | Your System |
|-----------|----------|-------------|
| CPU | Any modern CPU | Intel i5-13420H Γ£à |
| GPU | **NOT required** | Intel UHD iGPU (CPU-only mode) Γ£à |
| RAM | 8GB minimum | 16GB Γ£à |
| OS | Windows / Linux / macOS | Windows Γ£à |
| Python | 3.8+ | 3.10 recommended |

---

## ≡ƒÜÇ Quick Start

### Step 1 ΓÇö Navigate to project directory
```cmd
cd D:\amst_border_net
```

### Step 2 ΓÇö (Recommended) Create a virtual environment
```cmd
python -m venv venv
venv\Scripts\activate
```

### Step 3 ΓÇö Install dependencies
```cmd
pip install -r requirements.txt
```
> ΓÅ│ This may take 3ΓÇô10 minutes. PyTorch (CPU) + Ultralytics are large packages.

### Step 4 ΓÇö Run with webcam
```cmd
python main.py
```
> On first run: `yolo11n.pt` (~6MB) is auto-downloaded. Requires internet.

---

## ≡ƒÄ« Usage Examples

```cmd
# Default: Webcam at index 0
python main.py

# Use a different webcam
python main.py --source 1

# Run on a video file
python main.py --source path\to\video.mp4

# Enable verbose debug logging
python main.py --verbose

# Skip frames for better FPS on slow systems
python main.py --skip-frames 1

# Use fallback IOU tracker (no additional dependencies)
python main.py --tracker fallback

# Adjust confidence threshold
python main.py --conf-threshold 0.4

# Demo mode (no JSON saving, maximum FPS)
python main.py --no-save

# Full example with all options
python main.py --source video.mp4 --skip-frames 1 --conf-threshold 0.35 --verbose
```

---

## Γî¿∩╕Å Keyboard Controls

| Key | Action |
|-----|--------|
| `q` or `Esc` | Quit application |
| `p` | Pause / Resume video |
| `s` | Save screenshot (PNG) |
| `r` | Reset tracker (clear all track IDs) |

---

## ≡ƒôü File Structure

```
amst_border_net/
Γöé
Γö£ΓöÇΓöÇ main.py           # Entry point ΓÇö orchestrates everything
Γö£ΓöÇΓöÇ detector.py       # YOLODetector class ΓÇö YOLO11 inference
Γö£ΓöÇΓöÇ tracker.py        # ByteTrackWrapper + FallbackIOUTracker
Γö£ΓöÇΓöÇ data_exporter.py  # DataExporter ΓÇö formats & saves JSON
Γö£ΓöÇΓöÇ utils.py          # FPSCounter, logging, video source helpers
Γöé
Γö£ΓöÇΓöÇ requirements.txt  # pip dependencies (CPU-only)
Γö£ΓöÇΓöÇ README.md         # This file
Γöé
Γö£ΓöÇΓöÇ output/           # Auto-created ΓÇö JSON export files go here
ΓööΓöÇΓöÇ yolo11n.pt        # Auto-downloaded on first run (~6MB)
```

---

## ≡ƒôè Data Output Format

Every processed frame produces structured data in this format:

```json
{
  "frame_id": 123,
  "timestamp": "2026-09-05 10:30:45.123",
  "detections": [
    {
      "track_id": 1,
      "bbox": [120, 80, 240, 360],
      "centroid": [180, 220],
      "confidence": 0.89,
      "class": "person"
    },
    {
      "track_id": 2,
      "bbox": [400, 100, 520, 380],
      "centroid": [460, 240],
      "confidence": 0.76,
      "class": "person"
    }
  ]
}
```

**Field descriptions:**

| Field | Type | Description |
|-------|------|-------------|
| `frame_id` | int | Sequential frame number (resets per session) |
| `timestamp` | str | Wall-clock time when frame was processed |
| `track_id` | int | Persistent ID for this person (consistent across frames) |
| `bbox` | list[int] | `[x1, y1, x2, y2]` ΓÇö top-left and bottom-right pixels |
| `centroid` | list[int] | `[cx, cy]` ΓÇö center of the bounding box |
| `confidence` | float | YOLO detection confidence (0.0ΓÇô1.0) |
| `class` | str | Always `"person"` (only class detected) |

---

## ΓÜí Performance Guide

### Expected FPS on Intel i5-13420H (CPU-only)

| Setting | Expected FPS |
|---------|-------------|
| `yolo11n.pt`, input=640, skip=0 | 8ΓÇô12 FPS |
| `yolo11n.pt`, input=640, skip=1 | 12ΓÇô18 FPS (display) |
| `yolo11n.pt`, input=416, skip=0 | 12ΓÇô18 FPS |
| `yolo11n.pt`, input=416, skip=1 | 18ΓÇô25 FPS (display) |

### Speed Tips

1. **Use `--skip-frames 1`** ΓÇö Processes every 2nd frame. Visual appears ~2├ù faster.
2. **Use `--input-size 416`** ΓÇö Smaller input = faster inference. Trade-off: slightly lower detection accuracy.
3. **Use `--no-save`** ΓÇö Disables JSON disk writes, removing I/O overhead.
4. **Close other apps** ΓÇö More RAM for model inference.
5. **Use a video file instead of webcam** ΓÇö More consistent frame timing.

---

## ≡ƒöî Integration with Team Members

### For DEV 2 (Event Detection Module)

Import and use `DataExporter.get_current_data()` to get live detection data:

```python
# In DEV 2's code:
from data_exporter import DataExporter

# DataExporter instance is shared (passed from main.py)
frame_data = exporter.get_current_data()

if frame_data and frame_data["detections"]:
    for person in frame_data["detections"]:
        track_id  = person["track_id"]
        centroid  = person["centroid"]
        # ... your event detection logic here
```

OR read from the JSON files in `output/` directory.

### For DEV 3 (Alert Module)

Consume the JSON files in `output/` for alert triggering logic.
Files are saved in batches every 30 frames (configurable via `--batch-size`).

---

## ≡ƒ¢á∩╕Å Troubleshooting

### "Camera not found" / "Could not open video source"
```
Possible fixes:
  - Unplug and replug your webcam
  - Try --source 1 (different camera index)
  - Check if another app (Teams, Zoom) is using the camera
  - On Windows: check Camera Privacy Settings
```

### Very low FPS (below 5)
```
Try:
  python main.py --skip-frames 1 --input-size 416
  
Also:
  - Close background apps (Chrome, Teams, etc.)
  - Make sure power plan is set to "High Performance" in Windows
```

### "ultralytics not found" / "yolo11n.pt not found"
```
Run:
  pip install ultralytics
  
The model auto-downloads on first run. If download fails:
  - Check internet connection
  - Try: from ultralytics import YOLO; YOLO("yolo11n.pt")
```

### Too many false detections (ghosts)
```
Increase confidence threshold:
  python main.py --conf-threshold 0.5
```

### IDs changing too frequently
```
Use ByteTrack (default) instead of fallback:
  python main.py --tracker bytetrack
  
Or adjust FPS: higher FPS = more stable tracking.
```

---

## ≡ƒÅù∩╕Å Architecture Notes

```
ΓöîΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÉ
Γöé                      main.py                            Γöé
Γöé  CLI args ΓåÆ Video Loop ΓåÆ Orchestration ΓåÆ Display        Γöé
ΓööΓöÇΓöÇΓöÇΓöÇΓöÇΓö¼ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓö¼ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓö¼ΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÿ
      Γöé           Γöé              Γöé
      Γû╝           Γû╝              Γû╝
ΓöîΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÉ ΓöîΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÉ ΓöîΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÉ
Γöédetector.pyΓöé Γöétracker.pyΓöé Γöédata_exporter Γöé
Γöé           Γöé Γöé          Γöé Γöé.py           Γöé
ΓöéYOLO11n    Γöé ΓöéByteTrack Γöé ΓöéJSON output   Γöé
Γöéinference  Γöé ΓöéIOU track Γöé ΓöéLive data API Γöé
ΓööΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÿ ΓööΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÿ ΓööΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÿ
      Γöé                          Γöé
      ΓööΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÇΓöÿ
                    Γöé
              utils.py
         (FPS, logging, video)
```

---

## ≡ƒÄ¼ UCF Fence Mini-Dataset

A curated subset of the [UCF-Crime with Fence Climbing](https://www.kaggle.com/datasets/) dataset
for border/perimeter security evaluation.

### Dataset Location

```
data/ucf_fence_mini/
  fence_climbing/   ΓåÉ Fence Climbing 1, Fence Climbing 3, Suspicious4
  fighting/         ΓåÉ Fighting041, Fighting004, Fighting008
  stealing/         ΓåÉ Stealing025, Stealing077, Stealing086
  robbery/          ΓåÉ Robbery014, Robbery026
  shooting/         ΓåÉ Shooting032, Shooting006
```

- **Source**: `archive.zip` (4.1 GB, Kaggle) ΓÇö extracted frame sequences (`.jpg`/`.png`)
- **Total clips**: 13 across 5 classes | **Total frames**: 24,311
- **Config**: [`config/ucf_fence_mini.yaml`](config/ucf_fence_mini.yaml)
- **Full docs**: [`docs/ucf_fence_mini.md`](docs/ucf_fence_mini.md)

### How to Run

```bash
# Process all 13 clips headlessly
python scripts/train_ucf_fence_mini.py --headless

# Limit to 6 clips (quick smoke test)
python scripts/train_ucf_fence_mini.py --headless --max-videos 6

# Process only fence_climbing class
python scripts/train_ucf_fence_mini.py --headless --class fence_climbing

# Save annotated output videos
python scripts/train_ucf_fence_mini.py --headless --save-video
```

### Output

```
output/ucf_fence_mini/
  summary.json          ΓåÉ per-clip stats (tracks, avg people, duration, fps)
  videos/               ΓåÉ annotated .mp4 files (only when --save-video passed)
    fence_climbing_Fence_Climbing_1.mp4
    fighting_Fighting041.mp4
    ...
```

**`summary.json` structure:**
```json
{
  "session_start": "2026-09-07T23:45:00",
  "clips_processed": 13,
  "total_frames_processed": 24311,
  "overall_fps": 18.5,
  "clips": [
    {
      "class": "fence_climbing",
      "clip": "Fence_Climbing_1",
      "total_frames": 773,
      "unique_track_ids": 2,
      "avg_persons_per_frame": 0.84,
      "max_persons_in_frame": 2,
      "processing_time_sec": 41.7,
      "effective_fps": 18.5
    },
    ...
  ]
}
```

---

## ≡ƒô¥ Author Notes

- This is a **hackathon prototype** ΓÇö clarity over perfection
- All GPU-related code is intentionally excluded (CPU-only)
- The fallback tracker (FallbackIOUTracker) works without any internet or extra installs
- JSON export format is designed to be easily consumed by other team modules (DEV 2, DEV 3)
- Comments in code explain **WHY** decisions were made, not just **WHAT** the code does

---

*AMST Border-Net | DEV 1 | Smart India Hackathon 2026*
=======
# Smart-India-Hackathon-Smart-cctv-
>>>>>>> f28fae7a2b92918772fee7382788d33538b36f0e
