"""
utils.py - Utility Module for AMST Border-Net (DEV 1)
======================================================
Smart India Hackathon 2026 | Problem: SIH26187
Team Role: DEV 1 - Camera + YOLO Detection + Tracking

This module provides:
  - FPSCounter     : Real-time FPS measurement
  - setup_logging  : Configures logging with optional verbose mode
  - open_video_source : Opens webcam, video file, or RTSP stream
  - Performance warning helpers

Author : DEV 1
Date   : 2026-09-05
"""

import cv2
import time
import logging
import sys
from collections import deque


# ---------------------------------------------------------------------------
# FPS Counter
# ---------------------------------------------------------------------------

class FPSCounter:
    """
    Measures Frames Per Second (FPS) using a rolling average over a
    configurable window. A rolling average is smoother than measuring
    instantaneous FPS.

    Usage:
        fps_counter = FPSCounter(window_size=30)
        while True:
            fps_counter.tick()        # call once per frame
            fps = fps_counter.get()   # get current FPS
    """

    def __init__(self, window_size: int = 30):
        """
        Args:
            window_size: Number of recent frames to average over.
                         Larger = smoother but slower to react to changes.
        """
        # deque automatically discards old entries when maxlen is reached
        self.timestamps = deque(maxlen=window_size)
        self.window_size = window_size

    def tick(self):
        """Record the timestamp of the current frame."""
        self.timestamps.append(time.perf_counter())

    def get(self) -> float:
        """
        Return the current average FPS.
        Returns 0.0 if fewer than 2 frames have been recorded yet.
        """
        if len(self.timestamps) < 2:
            return 0.0

        # FPS = (number of frames - 1) / (time elapsed across those frames)
        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed <= 0:
            return 0.0

        return (len(self.timestamps) - 1) / elapsed

    def reset(self):
        """Clear all recorded timestamps."""
        self.timestamps.clear()


# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool = False) -> logging.Logger:
    """
    Configure the root logger for the entire application.

    Two modes:
      - Normal  : Shows INFO and above (WARNING, ERROR, CRITICAL)
      - Verbose : Shows DEBUG and above (everything)

    The log format includes timestamp, log level, and the message.
    This makes it easy to trace what happened and when.

    Args:
        verbose: If True, enable DEBUG-level logging.

    Returns:
        A configured Logger instance for the calling module.
    """
    # Choose log level based on verbose flag
    log_level = logging.DEBUG if verbose else logging.INFO

    # Build a human-readable format string
    log_format = (
        "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)-20s | %(message)s"
    )
    date_format = "%Y-%m-%d %H:%M:%S"

    # Apply configuration to the ROOT logger so all child loggers inherit it
    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(sys.stdout)  # Print to console
        ]
    )

    logger = logging.getLogger("amst_border_net")
    logger.info(
        f"Logging initialized. Level: {'DEBUG (verbose)' if verbose else 'INFO'}"
    )
    return logger


# ---------------------------------------------------------------------------
# Video Source Handler
# ---------------------------------------------------------------------------

def open_video_source(source: str | int, logger: logging.Logger = None) -> cv2.VideoCapture:
    """
    Open a video source (webcam, video file, or RTSP stream).

    WHY this function exists:
        OpenCV's cv2.VideoCapture accepts integers (webcam index) or strings
        (file paths, RTSP URLs). We handle type conversion and error checking
        in one place so main.py stays clean.

    Supported sources:
        - "0" or 0        ΓåÆ Webcam at index 0
        - "1", "2", ...   ΓåÆ Other webcam indices
        - "video.mp4"     ΓåÆ Local MP4 file
        - "path/file.avi" ΓåÆ Local AVI file
        - "rtsp://..."    ΓåÆ RTSP IP camera stream (future use)

    Args:
        source  : Camera index (int), file path (str), or RTSP URL (str).
        logger  : Optional logger. If None, uses print().

    Returns:
        cv2.VideoCapture object (already opened and verified).

    Raises:
        SystemExit: If the source cannot be opened (no point continuing).
    """
    log = logger or logging.getLogger("amst_border_net.utils")

    # Convert string integers like "0", "1" to actual int for OpenCV
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    log.info(f"Attempting to open video source: {source}")

    # Create the VideoCapture object
    cap = cv2.VideoCapture(source)

    # Check if it opened successfully
    if not cap.isOpened():
        log.error(
            f"FATAL: Could not open video source '{source}'.\n"
            "  Possible causes:\n"
            "    - Webcam not connected or in use by another app\n"
            "    - Wrong camera index (try --source 1)\n"
            "    - Video file path is wrong\n"
            "    - RTSP URL is unreachable\n"
        )
        sys.exit(1)  # Exit cleanly instead of crashing with a stack trace

    # Log source properties for debugging
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    log.info(f"Video source opened successfully.")
    log.info(f"  Source     : {source}")
    log.info(f"  Resolution : {width}x{height}")
    log.info(f"  Native FPS : {fps:.1f}")

    return cap


# ---------------------------------------------------------------------------
# Performance Warning
# ---------------------------------------------------------------------------

def warn_low_fps(fps: float, threshold: float = 8.0, logger: logging.Logger = None):
    """
    Print a warning if FPS drops below the acceptable threshold.

    WHY: On CPU-only machines, FPS can drop if the frame is complex or the
    system is under load. This helps DEV 1 know when to enable optimizations
    like --skip-frames.

    Args:
        fps       : Current measured FPS.
        threshold : Minimum acceptable FPS. Default = 8.0
        logger    : Logger instance.
    """
    log = logger or logging.getLogger("amst_border_net.utils")

    if 0 < fps < threshold:
        log.warning(
            f"Low FPS detected: {fps:.1f} FPS "
            f"(below threshold {threshold:.1f}). "
            "Try: --skip-frames 1 to process every 2nd frame."
        )


# ---------------------------------------------------------------------------
# Draw Overlay Text
# ---------------------------------------------------------------------------

def draw_fps_overlay(frame, fps: float, extra_info: str = "") -> None:
    """
    Draw the FPS counter on the top-left corner of the video frame.

    WHY: A live FPS counter helps us quickly see if performance is acceptable
    during the hackathon demo. This avoids the need for separate profiling.

    Args:
        frame      : OpenCV BGR image (numpy array), modified in-place.
        fps        : Current FPS value to display.
        extra_info : Optional string (e.g., "CPU Mode") shown on second line.
    """
    # Choose color based on FPS - green is good, yellow is ok, red is bad
    if fps >= 12:
        color = (0, 255, 0)      # Green - good performance
    elif fps >= 8:
        color = (0, 200, 200)    # Yellow - acceptable
    else:
        color = (0, 0, 255)      # Red - performance issue

    # Draw semi-transparent background rectangle for readability
    overlay_rect = frame.copy()
    cv2.rectangle(overlay_rect, (5, 5), (220, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay_rect, 0.5, frame, 0.5, 0, frame)

    # Draw FPS text
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),                    # Position (x, y)
        cv2.FONT_HERSHEY_SIMPLEX,    # Font type
        0.8,                         # Font scale
        color,                       # Color (BGR)
        2,                           # Thickness
        cv2.LINE_AA                  # Anti-aliased for smoother text
    )

    # Draw extra info (e.g., inference backend)
    if extra_info:
        cv2.putText(
            frame,
            extra_info,
            (10, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),  # Light gray
            1,
            cv2.LINE_AA
        )
