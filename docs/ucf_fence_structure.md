# UCF-Crime with Fence Climbing Dataset Structure

## Overview
- **Source Archive**: `C:\Users\Lenovo\SIH\archive.zip` (4.10 GB)
- **Extracted Root**: `UCF-Crime with Fence Climbing/` (with `Train/` and `Test/` partitions)
- **Format**: Video sequences stored as extracted high-resolution image frames (`.jpg` / `.png`) corresponding to each video recording.
- **Total Images**: 267,248 frames across 1,273 unique video recordings.

---

## Dataset Classes and Video Summary

| Class Name | Total Videos | Total Frames | Example Video Sequences (with Sample Frames) |
| :--- | :---: | :---: | :--- |
| **`Fence_Climbing`** | 12 | 11,213 | `Fence Climbing 1` (`Fence Climbing 1_frame_000445.jpg`)<br>`Fence Climbing 2` (`Fence Climbing 2_frame_003432.jpg`)<br>`Fence Climbing 3` (`Fence Climbing 3_frame_001572.jpg`) |
| **`Fighting`** | 51 | 26,231 | `Fighting002_x264` (`Fighting002_x264_0.png`)<br>`Fighting046_x264` (`Fighting046_x264_0.png`)<br>`Fighting 1` (`Fighting 1_frame_000161.jpg`) |
| **`Robbery`** | 151 | 42,479 | `Robbery001_x264` (`Robbery001_x264_0.png`)<br>`Robbery002_x264` (`Robbery002_x264_0.png`)<br>`Robbery005_x264` (`Robbery005_x264_0.png`) |
| **`Stealing`** | 100 | 46,786 | `Stealing002_x264` (`Stealing002_x264_0.png`)<br>`Stealing010_x264` (`Stealing010_x264_0.png`)<br>`Stealing019_x264` (`Stealing019_x264_0.png`) |
| **`Shooting`** | 52 | 15,279 | `Shooting001_x264` (`Shooting001_x264_0.png`)<br>`Shooting003_x264` (`Shooting003_x264_0.png`)<br>`Shooting005_x264` (`Shooting005_x264_0.png`) |
| **`Normal_Videos`** | 947 | 125,260 | `Normal_Videos003_x264` (`Normal_Videos003_x264_0.png`)<br>`Normal_Videos006_x264` (`Normal_Videos006_x264_0.png`)<br>`Normal_Videos010_x264` (`Normal_Videos010_x264_0.png`) |

---

## Split Breakdown

### 1. Test Split
- **Fence_Climbing**: 3 videos (1,087 frames)
- **Fighting**: 6 videos (1,547 frames)
- **Robbery**: 6 videos (986 frames)
- **Stealing**: 5 videos (1,984 frames)
- **Shooting**: 25 videos (8,139 frames)
- **Normal_Videos**: 150 videos (19,952 frames)

### 2. Train Split
- **Fence_Climbing**: 10 videos (10,126 frames)
- **Fighting**: 45 videos (24,684 frames)
- **Robbery**: 145 videos (41,493 frames)
- **Stealing**: 95 videos (44,802 frames)
- **Shooting**: 27 videos (7,140 frames)
- **Normal_Videos**: 797 videos (105,308 frames)

---

## Relevance to Border & Perimeter Intrusion Security

1. **`Fence_Climbing` (Priority: CRITICAL / TOP MATCH)**:
   - **Relevance**: Directly captures physical perimeter breach scenarios where intruders scale, hurdle, or tamper with security fencing and boundaries.
   - **Key Behaviors**: Suspects approaching fence lines, mounting barbed wire or mesh fencing, and dropping onto restricted property.
   - **Application**: Ground truth for perimeter intrusion alerts, tripwires, and fence-line virtual zones.

2. **`Fighting` / Physical Confrontation (Priority: HIGH)**:
   - **Relevance**: Models border patrol skirmishes, confrontations with security personnel, human trafficking disputes, or aggressive intrusion teams.
   - **Key Behaviors**: Multiple interacting humans engaged in violent motion, rapid bounding box shifts, and physical struggle.
   - **Application**: Suspicious behavior escalation, anomaly detection, multi-person tracking stress testing.

3. **`Stealing` / Trespassing & Prowling (Priority: HIGH)**:
   - **Relevance**: Closely mirrors unauthorized perimeter entry, facility trespassing, compound infiltration, and smuggling retrieval.
   - **Key Behaviors**: Covert movement near perimeter structures, stealth approaches, crouching, and rapid egress.
   - **Application**: Loitering detection, perimeter crawl monitoring, unauthorized access verification.

4. **`Robbery` / Armed Incursion (Priority: MEDIUM-HIGH)**:
   - **Relevance**: High-threat incursion scenarios involving forceful entry into guarded checkpoints or border outposts.
   - **Key Behaviors**: Fast-moving assailants, vehicle ingress, and confrontation with checkpoint guards.
