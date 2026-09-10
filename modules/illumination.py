"""
illumination.py - Day/Night Preprocessing & Illumination Classifier (Tier-2 DEV 1)
================================================================================
Smart India Hackathon 2026 | Problem: SIH26187
AMST Border-Net | DEV 1 module

This module automatically determines ambient scene brightness, detects DAY vs NIGHT
conditions using hysteresis to prevent rapid flickering, applies Contrast Limited
Adaptive Histogram Equalization (CLAHE) on the luminance channel when night is detected,
and provides illumination state for risk scaling and HUD badges.

Design:
  - Brightness computed from grayscale luminance mean.
  - Hysteresis:
      brightness < low_thresh  -> switches to NIGHT
      brightness > high_thresh -> switches to DAY
      between low and high     -> retains previous state
  - CLAHE enhancement:
      Converts BGR to LAB color space, applies CLAHE to L (luminance) channel,
      and converts back to BGR. Preserves color balance while enhancing dark details.
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger("amst_border_net.illumination")


class IlluminationManager:
    """
    Stateful day/night detector with hysteresis and CLAHE frame enhancement.
    """

    def __init__(
        self,
        low_thresh: float = 50.0,
        high_thresh: float = 70.0,
        clahe_clip: float = 2.0,
        clahe_grid: Tuple[int, int] = (8, 8),
        initial_night: bool = False,
    ):
        """
        Args:
            low_thresh   : Brightness below which state transitions to NIGHT.
            high_thresh  : Brightness above which state transitions to DAY.
            clahe_clip   : Contrast limit for CLAHE.
            clahe_grid   : Tile grid size for CLAHE (e.g. (8, 8)).
            initial_night: Initial guess for night mode.
        """
        self.low_thresh = float(low_thresh)
        self.high_thresh = float(high_thresh)
        self.clahe_clip = float(clahe_clip)
        self.clahe_grid = tuple(clahe_grid)
        self.is_night = bool(initial_night)
        self.last_brightness = 100.0

        self._clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip,
            tileGridSize=self.clahe_grid
        )

        logger.info(
            f"IlluminationManager initialized | low={self.low_thresh} | "
            f"high={self.high_thresh} | clahe_clip={self.clahe_clip} | "
            f"grid={self.clahe_grid}"
        )

    def compute_brightness(self, frame: np.ndarray) -> float:
        """Calculate mean luminance brightness from BGR frame (fast sub-sampled)."""
        if frame is None or frame.size == 0:
            return 0.0
        # Fast subsampling (every 8th pixel) gives identical mean luminance with ~64x speedup
        sample = frame[::8, ::8]
        gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    def update(self, frame: np.ndarray, force_night: Optional[bool] = None) -> Dict[str, Any]:
        """
        Compute illumination state with hysteresis.

        Args:
            frame      : Current BGR image from OpenCV.
            force_night: If boolean provided, manual override takes precedence.

        Returns:
            Dict:
                "is_night": bool,
                "brightness": float,
                "mode": "NIGHT" | "DAY"
        """
        brightness = self.compute_brightness(frame)
        self.last_brightness = brightness

        if force_night is not None:
            self.is_night = bool(force_night)
        else:
            if self.is_night:
                # If currently night, must rise above high_thresh to become day
                if brightness > self.high_thresh:
                    self.is_night = False
                    logger.info(
                        f"Illumination state switched to DAY (brightness={brightness:.1f} > {self.high_thresh})"
                    )
            else:
                # If currently day, must fall below low_thresh to become night
                if brightness < self.low_thresh:
                    self.is_night = True
                    logger.info(
                        f"Illumination state switched to NIGHT (brightness={brightness:.1f} < {self.low_thresh})"
                    )

        return {
            "is_night": self.is_night,
            "brightness": round(brightness, 2),
            "mode": "NIGHT" if self.is_night else "DAY",
        }

    def apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """
        Enhance low-light frame using CLAHE on LAB luminance channel.

        Args:
            frame: Input BGR frame.

        Returns:
            Enhanced BGR frame.
        """
        if frame is None or frame.size == 0:
            return frame

        try:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            enhanced_l = self._clahe.apply(l_channel)
            merged = cv2.merge((enhanced_l, a_channel, b_channel))
            return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        except Exception as e:
            logger.warning(f"Failed to apply CLAHE: {e}")
            return frame

    def process_frame(
        self, frame: np.ndarray, force_night: Optional[bool] = None, apply_enhancement: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Convenience function: computes illumination state and applies CLAHE if night.

        Returns:
            (processed_frame, illumination_state)
        """
        state = self.update(frame, force_night=force_night)
        processed = frame
        if state["is_night"] and apply_enhancement:
            processed = self.apply_clahe(frame)
        return processed, state


# Global singleton helper for simple stateless or shared usage
_default_manager: Optional[IlluminationManager] = None


def get_illumination_state(
    frame: np.ndarray,
    low_thresh: float = 50.0,
    high_thresh: float = 70.0,
    force_night: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Functional interface required by Tier-2 specifications.
    Returns {"is_night": bool, "brightness": float, "mode": str}.
    """
    global _default_manager
    if _default_manager is None:
        _default_manager = IlluminationManager(low_thresh=low_thresh, high_thresh=high_thresh)
    return _default_manager.update(frame, force_night=force_night)


def enhance_low_light(frame: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Apply standalone CLAHE enhancement on luminance channel."""
    mgr = IlluminationManager(clahe_clip=clip_limit)
    return mgr.apply_clahe(frame)
