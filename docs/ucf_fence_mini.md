# UCF Fence Mini-Dataset

## Overview

A curated subset of the **UCF-Crime with Fence Climbing** dataset, selected for
border/perimeter security demonstration with AMST Border-Net.

- **Source archive**: `C:\Users\Lenovo\SIH\archive.zip` (4.1 GB)
- **Mini-dataset root**: `C:\Users\Lenovo\SIH\data\ucf_fence_mini\`
- **Format**: Image-frame sequences (`.jpg` / `.png`), fed in sorted order as synthetic video
- **Total clips**: 13 across 5 classes
- **Config**: [`config/ucf_fence_mini.yaml`](../config/ucf_fence_mini.yaml)

---

## Selected Classes & Clips

### 1. `fence_climbing` ΓÇö Perimeter Breach (PRIMARY)

| Clip Folder | Frames | Duration | Rationale |
|---|---|---|---|
| `Fence_Climbing_1` | 773 | ~25.8s | Clear daytime fence-scaling sequence; cleanest shot of a single intruder mounting a chain-link fence |
| `Fence_Climbing_3` | 1500 | ~50.0s | Longer sustained climbing sequence; good for multi-frame tracking continuity |
| `Suspicious4` | 3648 | ~121.6s | Longest Fence_Climbing clip; extended perimeter loitering before breach ΓÇö ideal for lingering-person alerts |

---

### 2. `fighting` ΓÇö Physical Confrontation / Patrol Skirmish

| Clip Folder | Frames | Duration | Rationale |
|---|---|---|---|
| `Fighting041` | 3165 | ~105.5s | Longest fighting clip; dense multi-person scene, tracking stress test |
| `Fighting004` | 1678 | ~55.9s | Clear two-person confrontation, useful for bounding-box overlap testing |
| `Fighting008` | 1314 | ~43.8s | Compact fight sequence; rapid motion and occlusion testing |

---

### 3. `stealing` ΓÇö Covert Movement / Compound Infiltration

| Clip Folder | Frames | Duration | Rationale |
|---|---|---|---|
| `Stealing025` | 2182 | ~72.7s | Longest stealing clip; extended sneaking/loitering ΓÇö mirrors unauthorized perimeter approach |
| `Stealing077` | 1919 | ~64.0s | Indoor scene; tests detection under challenging lighting |
| `Stealing086` | 1815 | ~60.5s | Crouching/low-profile movement ΓÇö tests detection of non-upright persons |

---

### 4. `robbery` ΓÇö Armed Incursion / Gun-Threat

| Clip Folder | Frames | Duration | Rationale |
|---|---|---|---|
| `Robbery014` | 1528 | ~50.9s | Armed confrontation with clear person detections; models checkpoint gun threat |
| `Robbery026` | 1312 | ~43.7s | Multi-person robbery scene; secondary aggressor for multi-track ID assignment |

---

### 5. `shooting` ΓÇö Gunfire / High-Threat Armed Intrusion

| Clip Folder | Frames | Duration | Rationale |
|---|---|---|---|
| `Shooting032` | 2169 | ~72.3s | Longest shooting clip (Test split); outdoor scene, armed intruder in open terrain |
| `Shooting006` | 1308 | ~43.6s | Train split; fast-motion scene tests detection under motion blur |

---

## Directory Structure

```
data/ucf_fence_mini/
  fence_climbing/
    Fence_Climbing_1/     (773 frames)
    Fence_Climbing_3/     (1500 frames)
    Suspicious4/          (3648 frames)
  fighting/
    Fighting041/          (3165 frames)
    Fighting004/          (1678 frames)
    Fighting008/          (1314 frames)
  stealing/
    Stealing025/          (2182 frames)
    Stealing077/          (1919 frames)
    Stealing086/          (1815 frames)
  robbery/
    Robbery014/           (1528 frames)
    Robbery026/           (1312 frames)
  shooting/
    Shooting032/          (2169 frames)
    Shooting006/          (1308 frames)
```
