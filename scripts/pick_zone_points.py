"""
pick_zone_points.py ΓÇö Interactive Zone Coordinate Picker
=========================================================
SIH26187 | AMST Border-Net

Opens the first frame of a video (or a live webcam) and lets you click
polygon vertices for restricted zones.  When you finish a zone, the
coordinates are printed and optionally written back to
config/boundary_config.yaml.

HOW TO USE:
    # Pick points from a video file:
    python scripts/pick_zone_points.py --source data/demo_videos/border_intrusion.mp4

    # Pick points from webcam:
    python scripts/pick_zone_points.py --source 0

    # Just print frame resolution (no window):
    python scripts/pick_zone_points.py --source video.mp4 --info-only

CONTROLS (when window is open):
    Left-click   : Add a vertex to the current polygon.
    ENTER        : Finish the current polygon and start a new one.
    BACKSPACE    : Remove the last vertex added.
    ESC / q      : Quit and save collected zones to boundary_config.yaml.
    r            : Reset all collected zones and start over.

Author : DEV 1
Date   : 2026-09-08
"""

import argparse
import cv2
import os
import sys
import yaml
from typing import List, Tuple

# Colours for interactive drawing
_VERTEX_COLOR  = (0, 255, 0)     # Green dots
_LINE_COLOR    = (0, 220, 255)   # Yellow lines
_COMPLETE_COLOR= (0, 100, 255)   # Orange completed polygon
_TEXT_COLOR    = (255, 255, 255) # White text


def _mouse_callback(event, x, y, flags, param):
    """Append clicked point to param['current_poly'] list."""
    if event == cv2.EVENT_LBUTTONDOWN:
        param["current_poly"].append((x, y))


def _draw_state(canvas, zones_done, current_poly):
    """Render all completed zones + in-progress polygon on canvas."""
    frame = canvas.copy()

    # Draw completed zones
    for idx, pts in enumerate(zones_done):
        import numpy as np
        arr = __import__("numpy").array(pts, dtype=__import__("numpy").int32)
        cv2.polylines(frame, [arr], isClosed=True, color=_COMPLETE_COLOR, thickness=2)
        cx, cy = int(arr[:, 0].mean()), int(arr[:, 1].mean())
        cv2.putText(frame, f"zone_{idx+1}", (cx - 30, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, _COMPLETE_COLOR, 1, cv2.LINE_AA)

    # Draw in-progress polygon
    for i, pt in enumerate(current_poly):
        cv2.circle(frame, pt, 5, _VERTEX_COLOR, -1)
        cv2.putText(frame, str(i + 1), (pt[0] + 6, pt[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _VERTEX_COLOR, 1)
        if i > 0:
            cv2.line(frame, current_poly[i - 1], pt, _LINE_COLOR, 2)
    if len(current_poly) >= 3:
        cv2.line(frame, current_poly[-1], current_poly[0], _LINE_COLOR, 1)

    # HUD instructions
    instructions = [
        "Left-click: add vertex",
        "ENTER: finish zone (auto-named)",
        "BACKSPACE: undo last vertex",
        "r: clear zones & redraw",
        "c: reset to full-frame & quit",
        "ESC / q: save & quit",
    ]
    y0 = 20
    for line in instructions:
        cv2.putText(frame, line, (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _TEXT_COLOR, 1, cv2.LINE_AA)
        y0 += 18

    # Point count
    cv2.putText(frame,
                f"Vertices: {len(current_poly)}  |  Zones done: {len(zones_done)}",
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, _TEXT_COLOR, 1, cv2.LINE_AA)

    return frame


def _write_config(zones_done: List[List[Tuple[int, int]]],
                  zone_ids:   List[str],
                  cfg_path:   str,
                  frame_w:    int,
                  frame_h:    int) -> None:
    """Write collected zones to boundary_config.yaml."""
    # Load existing config if present
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}

    # If a reference_resolution already exists, scale newly picked points to match it;
    # otherwise, set the reference resolution to the current frame size.
    ref_res = cfg.get("reference_resolution")
    if ref_res and len(ref_res) == 2 and ref_res[0] > 0 and ref_res[1] > 0:
        ref_w, ref_h = ref_res[0], ref_res[1]
    else:
        ref_w, ref_h = frame_w, frame_h
        cfg["reference_resolution"] = [ref_w, ref_h]

    existing = cfg.get("zones", []) or []

    # When the user draws custom zones, automatically remove the default
    # full_frame_restricted zone so it doesn't overlap with their selections.
    existing = [z for z in existing if z.get("id") != "full_frame_restricted"]

    for pts, zid in zip(zones_done, zone_ids):
        # Scale coordinates into reference resolution space if different
        if (frame_w, frame_h) != (ref_w, ref_h):
            norm_pts = [
                [int(round(x * ref_w / frame_w)), int(round(y * ref_h / frame_h))]
                for x, y in pts
            ]
        else:
            norm_pts = [[int(x), int(y)] for x, y in pts]

        entry = {
            "id":                 zid,
            "type":               "restricted",
            "polygon":            norm_pts,
            "min_loiter_seconds": 5,
        }
        # Replace if ID already exists, else append
        replaced = False
        for i, z in enumerate(existing):
            if z["id"] == zid:
                existing[i] = entry
                replaced = True
                break
        if not replaced:
            existing.append(entry)

    cfg["zones"] = existing

    os.makedirs(os.path.dirname(os.path.abspath(cfg_path)), exist_ok=True)
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"\n[OK] Zones written to: {cfg_path}")


def _reset_to_full_frame(cfg_path: str) -> None:
    """Remove all custom zones and restore the single full-frame restricted zone."""
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}

    cfg["reference_resolution"] = [1280, 720]
    cfg["zones"] = [{
        "id":                 "full_frame_restricted",
        "type":               "restricted",
        "polygon":            [[0, 0], [1280, 0], [1280, 720], [0, 720]],
        "min_loiter_seconds": 5,
    }]

    os.makedirs(os.path.dirname(os.path.abspath(cfg_path)), exist_ok=True)
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    print(f"[OK] Reset: full_frame_restricted zone restored in {cfg_path}")



def main():
    parser = argparse.ArgumentParser(
        description="Interactive zone polygon picker for AMST Border-Net."
    )
    parser.add_argument(
        "--source", type=str, default="0",
        help="Video file path or camera index (default: 0)"
    )
    parser.add_argument(
        "--config", type=str,
        default=os.path.join(os.path.dirname(__file__), "..", "config", "boundary_config.yaml"),
        help="Path to boundary_config.yaml to write results to."
    )
    parser.add_argument(
        "--info-only", action="store_true",
        help="Just print frame resolution and exit (no window)."
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Remove all custom zones and restore the full-frame restricted zone, then exit."
    )
    args = parser.parse_args()

    cfg_path = os.path.normpath(args.config)

    # Handle --reset before opening any camera
    if args.reset:
        _reset_to_full_frame(cfg_path)
        print("Done. Run 'python main.py' to apply.")
        return

    # Open video source
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"ERROR: Cannot open source: {args.source}", file=sys.stderr)
        sys.exit(1)

    ret, first_frame = cap.read()
    cap.release()
    if not ret:
        print("ERROR: Could not read a frame from source.", file=sys.stderr)
        sys.exit(1)

    h, w = first_frame.shape[:2]
    print(f"Frame resolution: {w} \u00d7 {h}")

    if args.info_only:
        return

    # --- Interactive picking ---
    state = {"current_poly": []}
    zones_done: List[List[Tuple[int, int]]] = []
    zone_ids:   List[str] = []

    win = "AMST | Zone Picker ΓÇö press ESC/q to save"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(w, 1280), min(h, 720))
    cv2.setMouseCallback(win, _mouse_callback, state)

    print("\nWindow open. Click polygon vertices on the frame.")
    print("Press ENTER to finish each zone, ESC/q to save & quit.")
    print("Press 'r' to clear all zones | 'c' to reset to full-frame zone & quit.\n")

    while True:
        display = _draw_state(first_frame, zones_done, state["current_poly"])
        cv2.imshow(win, display)
        key = cv2.waitKey(20) & 0xFF

        if key in (27, ord('q')):   # ESC or q ΓåÆ finish & save
            # Auto-close any in-progress polygon with >= 3 points
            if len(state["current_poly"]) >= 3:
                idx = len(zones_done) + 1
                zones_done.append(list(state["current_poly"]))
                zone_ids.append(f"zone_{idx}")
                print(f"  Auto-closed zone_{idx} with {len(state['current_poly'])} vertices.")
            break

        elif key == ord('c'):       # c ΓåÆ reset to full-frame zone & quit
            cv2.destroyAllWindows()
            _reset_to_full_frame(cfg_path)
            print("Done. Run 'python main.py' to apply.")
            return

        elif key == 13:             # ENTER ΓåÆ finish polygon
            if len(state["current_poly"]) < 3:
                print("  Need at least 3 vertices to close a polygon ΓÇö keep clicking.")
            else:
                idx = len(zones_done) + 1
                zid = f"zone_{idx}"          # Auto-named; rename in the YAML afterwards
                zones_done.append(list(state["current_poly"]))
                zone_ids.append(zid)
                print(f"  [OK] Zone '{zid}' saved with {len(state['current_poly'])} vertices.")
                print(f"       Tip: rename '{zid}' in config/boundary_config.yaml if needed.")
                print(f"       Draw another zone or press ESC/q to save & quit.\n")
                state["current_poly"] = []

        elif key == 8:              # BACKSPACE ΓåÆ undo last vertex
            if state["current_poly"]:
                state["current_poly"].pop()

        elif key == ord('r'):       # r ΓåÆ clear all zones drawn so far
            zones_done.clear()
            zone_ids.clear()
            state["current_poly"] = []
            print("  Reset. All zones cleared. Keep drawing or press ESC to quit.\n")

    cv2.destroyAllWindows()

    if zones_done:
        print(f"\nCollected {len(zones_done)} zone(s):")
        for zid, pts in zip(zone_ids, zones_done):
            print(f"  {zid}: {pts}")
        _write_config(zones_done, zone_ids, args.config, w, h)
    else:
        print("No zones collected ΓÇö nothing written.")


if __name__ == "__main__":
    main()
