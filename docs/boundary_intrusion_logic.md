# Boundary Intrusion Logic ΓÇö AMST Border-Net
## SIH26187 | DEV 1 | 2026-09-08

---

## Overview

The Boundary Engine determines whether a tracked person has **intruded** into a restricted area or is **loitering** in a monitored area. All logic is pure geometry ΓÇö no biometric identification is used in this prototype.

---

## Zone Types

The system supports two zone types, configured in `config/boundary_config.yaml`.

### `restricted` ΓÇö Hard Perimeter Zone

A zone where **no person should ever be present**.

| Event | Alert | Severity | Colour |
|---|---|---|---|
| Centroid enters polygon | `intrusion` | `high` | ≡ƒö┤ Red |
| Continuous dwell ΓëÑ `min_loiter_seconds` | `loitering` | `medium` | ≡ƒƒá Orange |

- Intrusion fires the moment the tracked centroid crosses the polygon boundary.
- Loitering fires after continuous unbroken presence inside the zone.
- If a person is detected inside, **intrusion takes priority** over loitering in visualisation and exports.

**Use for:** Border fence lines, exclusion zones, restricted infrastructure areas.

---

### `monitor` ΓÇö Observation Zone

A zone where personnel **legitimately pass through** (e.g., gates, checkpoints).

| Event | Alert | Severity | Colour |
|---|---|---|---|
| Centroid enters polygon | *no alert* | ΓÇö | ≡ƒö╡ Cyan (normal) |
| Continuous dwell ΓëÑ `min_loiter_seconds` | `loitering` | `medium` | ≡ƒƒá Orange |

- Presence alone is **logged** (visible in JSON exports) but does **not** raise an intrusion alert.
- Loitering fires after continuous unbroken presence exceeds `min_loiter_seconds`.

**Use for:** Gate entry zones, checkpoint areas, access corridors.

---

## Alert Priority

When a person is inside multiple zones simultaneously:

1. **`intrusion` (high)** in any `restricted` zone ΓÇö highest priority, stops evaluation.
2. **`loitering` (medium)** in any zone ΓÇö only if no intrusion was found.

---

## Severity and Colour Mapping

| Severity | Meaning | Box Colour | JSON value |
|---|---|---|---|
| `high` | Intrusion in restricted zone | Red `(0, 0, 255)` | `"high"` |
| `medium` | Loitering in any zone | Orange `(0, 128, 255)` | `"medium"` |
| `none` | No alert | Cyan `(0, 255, 255)` | `"none"` |

Zone borders also reflect state:
- **Restricted idle** ΓåÆ dark red border
- **Monitor idle** ΓåÆ amber border  
- **Active intrusion** ΓåÆ bright red border
- **Active loitering** ΓåÆ orange border

---

## JSON Export Fields

Each tracked person record in the exported JSON batches includes:

```json
{
  "track_id": 3,
  "bbox": [x1, y1, x2, y2],
  "centroid": [cx, cy],
  "confidence": 0.91,
  "class": "person",
  "alert_type": "intrusion",
  "severity": "high",
  "zone_id": "border_fence"
}
```

Fields default to `"none"` / `null` when no boundary config is loaded.

---

## Loitering Detection

Loitering is measured as **continuous, unbroken dwell time** inside a zone polygon.

- The system keeps a rolling centroid history per track (capped at `HISTORY_MAX_LEN` frames).
- On each processed frame, it counts how many recent frames (newest ΓåÆ oldest) have the centroid continuously inside the polygon, stopping at the first frame outside.
- If `consecutive_inside_frames ΓëÑ (min_loiter_seconds ├ù FPS)`, loitering fires.
- A single frame outside the zone **resets the count** (strict continuous requirement).

> This is intentionally strict for a border perimeter context: momentary departure and re-entry counts as a fresh intrusion, not a continuation of the same loitering event.

---

## Prototype Assumptions & Authorization (Future Work)

> **This prototype treats ALL persons detected inside a restricted zone as potential intruders.**

In a real deployment, some persons (security personnel, maintenance workers) may be authorized to enter restricted zones. This prototype does not distinguish them.

The following authorization mechanisms are **planned future work**:

| Feature | Status |
|---|---|
| Access badge / RFID integration | ≡ƒö£ Planned |
| Uniform / vest colour detection | ≡ƒö£ Planned |
| Face recognition against allowlist | ≡ƒö£ Planned |
| Biometric identification | ≡ƒö£ Planned |

When authorization is added, the `compute_alert()` function in `modules/boundary_engine.py` will accept an additional `authorized: bool` flag. If `authorized=True`, `restricted` zones will behave like `monitor` zones for that individual ΓÇö logging presence without raising intrusion alerts.

---

## Files

| File | Purpose |
|---|---|
| `config/boundary_config.yaml` | Zone definitions (type, polygon, loiter threshold) |
| `modules/boundary_engine.py` | Geometry engine, `compute_alert()`, drawing helpers |
| `main.py` | Pipeline integration (Hooks 3 / 4 / 5) |
| `data_exporter.py` | JSON export with `alert_type`, `severity`, `zone_id` |
| `scripts/pick_zone_points.py` | Interactive zone polygon picker tool |
