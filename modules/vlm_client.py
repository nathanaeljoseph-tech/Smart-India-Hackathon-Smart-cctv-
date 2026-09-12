"""
modules/vlm_client.py - Qwen2-VL Vision-Language Model Client
=============================================================
BorderVigil-AI | Tier-3 Advanced Feature
Problem: SIH26187

Connects to a locally hosted Qwen2-VL model (2B or 7B) via Ollama or vLLM.
Triggered exclusively for high-risk security events (risk >= 60.0) or
critical proximity/tamper breaches.

Produces 1-2 sentence operational descriptions:
  "Unknown person crossing restricted fence at night, moving from left to right."
Includes robust local fallback generator when offline.
"""

import base64
import cv2
import json
import logging
import time
from typing import Dict, Any, Optional, Tuple
import numpy as np

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

logger = logging.getLogger("amst_border_net.vlm_client")


class VLMClient:
    """
    Client for Qwen2-VL inference over local HTTP endpoints (Ollama / vLLM).
    """
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        endpoint: str = "http://localhost:11434",
        model: str = "qwen2-vl:7b",
        provider: str = "ollama",
        min_risk_trigger: float = 60.0,
        cooldown_seconds: float = 15.0,
        timeout_seconds: float = 4.0
    ):
        vlm_cfg = (config or {}).get("vlm", {}) if config else {}
        self.enabled = bool(vlm_cfg.get("enabled", True))
        self.endpoint = str(vlm_cfg.get("endpoint", endpoint)).rstrip("/")
        self.model = str(vlm_cfg.get("model", model))
        self.provider = str(vlm_cfg.get("provider", provider)).lower()
        self.min_risk = float(vlm_cfg.get("min_risk_trigger", min_risk_trigger))
        self.cooldown_sec = float(vlm_cfg.get("cooldown_seconds", cooldown_seconds))
        self.timeout = float(vlm_cfg.get("timeout_seconds", timeout_seconds))

        # Per-track cooldown tracking: {track_id: timestamp_last_captioned}
        self._last_caption_time: Dict[Any, float] = {}
        self._caption_cache: Dict[Any, str] = {}
        self._snapshot_cache: Dict[Any, str] = {}
        self._last_caption_meta: Dict[Any, Dict[str, Any]] = {}
        self._endpoint_online: Optional[bool] = None
        self._last_endpoint_check: float = 0.0

        logger.info(
            f"VLMClient ready | enabled={self.enabled} | provider={self.provider} | "
            f"model={self.model} | endpoint={self.endpoint} | min_risk={self.min_risk}"
        )

    # -----------------------------------------------------------------------
    def is_endpoint_online(self) -> bool:
        """Fast non-blocking check if local Ollama/vLLM port is listening."""
        now = time.time()
        if self._endpoint_online is not None and (now - self._last_endpoint_check) < 30.0:
            return self._endpoint_online

        self._last_endpoint_check = now
        try:
            import socket
            from urllib.parse import urlparse
            p = urlparse(self.endpoint)
            host = p.hostname or "localhost"
            port = p.port or 11434
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.15)
                res = s.connect_ex((host, port))
                self._endpoint_online = (res == 0)
        except Exception:
            self._endpoint_online = False
        return self._endpoint_online

    # -----------------------------------------------------------------------
    def should_trigger(self, risk_score: float, track_id: Any) -> bool:
        """Check if an event meets risk threshold and cooldown criteria."""
        if not self.enabled or risk_score < self.min_risk:
            return False

        now = time.time()
        last_t = self._last_caption_time.get(track_id, 0.0)
        return (now - last_t) >= self.cooldown_sec

    # -----------------------------------------------------------------------
    def get_track_snapshot(self, track_id: Any) -> Optional[str]:
        """Retrieve latest base64 image data-URI snapshot for track."""
        return self._snapshot_cache.get(track_id)

    # -----------------------------------------------------------------------
    def generate_caption(
        self,
        frame: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate natural language caption for high-risk security event.
        Runs asynchronously to guarantee high video FPS (>20 FPS) without stalling.
        """
        import threading
        ctx = context or {}
        track_id = ctx.get("track_id", "unknown")
        now = time.time()
        self._last_caption_time[track_id] = now

        # Extract and cache target crop snapshot
        if frame is not None and frame.size > 0:
            crop = self._extract_crop(frame, bbox)
            b64_img = self._encode_image_b64(crop)
            if b64_img:
                self._snapshot_cache[track_id] = f"data:image/jpeg;base64,{b64_img}"
        else:
            b64_img = ""

        # Immediate fast heuristic caption
        fallback = self._generate_rule_caption(ctx)

        if not self.is_endpoint_online():
            cap = self._caption_cache.get(track_id, fallback)
            self._last_caption_meta[track_id] = {
                "caption": cap, "source": "heuristic_fallback", "timestamp": now
            }
            return cap

        # If endpoint is online, run model query in background thread
        def _worker():
            try:
                crop = self._extract_crop(frame, bbox)
                b64_data = self._encode_image_b64(crop)
                prompt = self._build_prompt(ctx)
                if self.provider == "ollama":
                    caption = self._call_ollama(b64_data, prompt)
                else:
                    caption = self._call_vllm(b64_data, prompt)
                if caption:
                    self._caption_cache[track_id] = caption
                    self._last_caption_meta[track_id] = {
                        "caption": caption, "source": self.model, "timestamp": time.time()
                    }
                    logger.info(f"VLM Caption [Track {track_id}]: \"{caption}\"")
            except Exception as e:
                logger.debug(f"Async VLM call failed: {e}")

        threading.Thread(target=_worker, daemon=True).start()
        return self._caption_cache.get(track_id, fallback)

    # -----------------------------------------------------------------------
    def generate_caption_immediate(
        self,
        frame: Optional[np.ndarray],
        bbox: Optional[Tuple[int, int, int, int]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Synchronous / on-demand VLM query for dashboard operator triggers.
        """
        ctx = context or {}
        track_id = ctx.get("track_id", "manual")
        b64_uri = ""

        if frame is not None and frame.size > 0:
            crop = self._extract_crop(frame, bbox)
            b64_data = self._encode_image_b64(crop)
            if b64_data:
                b64_uri = f"data:image/jpeg;base64,{b64_data}"
                self._snapshot_cache[track_id] = b64_uri
        else:
            b64_data = ""

        fallback = self._generate_rule_caption(ctx)
        caption = fallback
        used_model = "Rule Heuristic Prior (Offline)"

        if self.is_endpoint_online() and b64_data:
            try:
                prompt = self._build_prompt(ctx)
                res_cap = self._call_ollama(b64_data, prompt) if self.provider == "ollama" else self._call_vllm(b64_data, prompt)
                if res_cap:
                    caption = res_cap
                    used_model = self.model
            except Exception as e:
                logger.debug(f"Immediate VLM call failed: {e}")

        self._caption_cache[track_id] = caption
        return {
            "caption": caption,
            "track_id": track_id,
            "model": used_model,
            "provider": self.provider,
            "snapshot": b64_uri,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }


    # -----------------------------------------------------------------------
    def _extract_crop(self, frame: np.ndarray, bbox: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
        if frame is None or frame.size == 0:
            return np.zeros((128, 128, 3), dtype=np.uint8)
        if not bbox:
            return frame

        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        # Add 15% margin for visual context
        pad_w = int((x2 - x1) * 0.15)
        pad_h = int((y2 - y1) * 0.15)
        cx1 = max(0, x1 - pad_w)
        cy1 = max(0, y1 - pad_h)
        cx2 = min(w, x2 + pad_w)
        cy2 = min(h, y2 + pad_h)

        if cx2 > cx1 and cy2 > cy1:
            return frame[cy1:cy2, cx1:cx2]
        return frame

    @staticmethod
    def _encode_image_b64(crop: np.ndarray, max_dim: int = 448) -> str:
        h, w = crop.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            crop = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        success, enc = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not success:
            return ""
        return base64.b64encode(enc).decode("utf-8")

    # -----------------------------------------------------------------------
    def _build_prompt(self, ctx: Dict[str, Any]) -> str:
        cam_id = ctx.get("camera_id", "CAM-01")
        zone_id = ctx.get("zone_id", "perimeter")
        behavior = ctx.get("behavior", "intrusion")
        risk = ctx.get("risk_score", 75.0)
        is_night = ctx.get("is_night", False)
        cls_name = ctx.get("class", "person")

        return (
            f"You are a border security AI visual analyst. Describe this security event in 1 or 2 concise, factual sentences. "
            f"Camera: {cam_id}, Zone: {zone_id}, Target: {cls_name}, Detected behavior: {behavior}, Risk: {risk:.0f}/100, "
            f"Lighting: {'Night/IR' if is_night else 'Daylight'}. State who or what is doing what and where."
        )

    # -----------------------------------------------------------------------
    def _call_ollama(self, b64_img: str, prompt: str) -> Optional[str]:
        url = f"{self.endpoint}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [b64_img] if b64_img else [],
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 64}
        }
        res = requests.post(url, json=payload, timeout=self.timeout)
        if res.status_code == 200:
            data = res.json()
            caption = data.get("response", "").strip()
            return caption if caption else None
        return None

    # -----------------------------------------------------------------------
    def _call_vllm(self, b64_img: str, prompt: str) -> Optional[str]:
        url = f"{self.endpoint}/v1/chat/completions"
        content = [{"type": "text", "text": prompt}]
        if b64_img:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
            })
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 64,
            "temperature": 0.2
        }
        res = requests.post(url, json=payload, timeout=self.timeout)
        if res.status_code == 200:
            data = res.json()
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "").strip()
        return None

    # -----------------------------------------------------------------------
    def _generate_rule_caption(self, ctx: Dict[str, Any]) -> str:
        """
        High-fidelity deterministic caption generator when local LLM server is offline.
        """
        cls_name = ctx.get("class", "person")
        behavior = ctx.get("behavior", "intrusion")
        zone_id = ctx.get("zone_id", "perimeter")
        dwell = ctx.get("dwell_sec", 0.0)
        risk = ctx.get("risk_score", 75.0)
        is_night = ctx.get("is_night", False)
        time_str = "at night" if is_night else "during daytime"

        if behavior == "loiter":
            return f"Individual loitering inside restricted {zone_id} for {dwell:.0f}s {time_str}, exhibiting persistent dwelling and pacing behavior along security perimeter (Risk: {risk:.0f})."
        elif behavior == "approach":
            return f"Target approaching boundary {zone_id} from perimeter access corridor {time_str}, advancing rapidly towards security fence line (Risk: {risk:.0f})."
        elif behavior == "tamper":
            return f"Critical physical perimeter disturbance detected along {zone_id} barrier {time_str}; vibration energy spike indicates climbing or fence tampering."
        elif behavior == "abandoned":
            cat = ctx.get("category", "unattended item")
            return f"Stationary unattended {cat} detected in {zone_id} with no owner present {time_str}; potential drop-off hazard under continuous observation."
        else:
            return f"Unknown {cls_name} breached restricted {zone_id} {time_str}, confirmed boundary intrusion crossing line into restricted zone (Risk: {risk:.0f})."
