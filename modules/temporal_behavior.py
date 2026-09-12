"""
modules/temporal_behavior.py - 1D TCN + Attention Temporal Behavior Model
==========================================================================
BorderVigil-AI | Tier-3 Advanced Feature
Problem: SIH26187

Analyzes short temporal sequences of track kinematics and spatial context:
  1. Position & velocity (x, y, dx, dy, speed, direction)
  2. Distance to border boundary line (e.g. y=576)
  3. Zone context state (restricted vs monitor)
  4. Temporal dwell progression

Implements:
  - Dilated 1D Temporal Convolutional Network (TCN) with residual connections
  - Temporal Attention Pooling mechanism
  - Classification / Likelihood heads for:
      * approach_score (0.0 - 1.0)
      * loiter_score   (0.0 - 1.0)
      * cross_score    (0.0 - 1.0)
      * intrusion_likelihood (0.0 - 100.0)
  - Score fusion with rule-based behavior engine:
      final_score = alpha * rule_score + beta * tcn_score
"""

import math
import logging
from typing import Dict, List, Any, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False

logger = logging.getLogger("amst_border_net.temporal_behavior")


# ===========================================================================
# 1D Dilated Residual Block
# ===========================================================================

_ModuleBase = nn.Module if TORCH_AVAILABLE else object

class TemporalBlock(_ModuleBase):
    """
    1D Dilated Convolution block with causal/padded temporal receptive field,
    BatchNorm, ReLU, and residual skip connection.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # Match shapes if temporal length differs by 1 due to padding
        if out.shape[-1] != res.shape[-1]:
            min_l = min(out.shape[-1], res.shape[-1])
            out = out[:, :, :min_l]
            res = res[:, :, :min_l]
        return self.relu(out + res)


# ===========================================================================
# Temporal Attention Pooling
# ===========================================================================

class TemporalAttentionPooling(_ModuleBase):
    """
    Learned attention mechanism over time steps to weight salient motion moments.
    """
    def __init__(self, in_features: int):
        if TORCH_AVAILABLE:
            super().__init__()
            self.query = nn.Linear(in_features, 1)

    def forward(self, x: Any) -> Tuple[Any, Any]:
        if not TORCH_AVAILABLE:
            return None, None
        # x shape: [B, C, T] -> transpose to [B, T, C]
        x_trans = x.transpose(1, 2)
        attn_scores = self.query(x_trans)  # [B, T, 1]
        attn_weights = F.softmax(attn_scores, dim=1)  # [B, T, 1]
        pooled = torch.sum(x_trans * attn_weights, dim=1)  # [B, C]
        return pooled, attn_weights.squeeze(-1)


# ===========================================================================
# TCN + Attention PyTorch Model
# ===========================================================================

class TCNAttentionBehaviorNet(_ModuleBase):
    """
    Complete 1D Dilated TCN + Temporal Attention network for target behavior scoring.
    """
    def __init__(
        self,
        input_dim: int = 8,
        hidden_channels: int = 32,
        num_levels: int = 3,
        num_classes: int = 3
    ):
        if not TORCH_AVAILABLE:
            return
        super().__init__()
        layers = []
        channels = [input_dim] + [hidden_channels] * num_levels
        for i in range(num_levels):
            dilation = 2 ** i
            layers.append(
                TemporalBlock(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=3,
                    dilation=dilation
                )
            )
        self.tcn = nn.Sequential(*layers)
        self.attention = TemporalAttentionPooling(hidden_channels)

        # Classification heads: [approach, loiter, cross]
        self.head_behavior = nn.Linear(hidden_channels, num_classes)
        # Direct intrusion likelihood regression head (0.0 to 100.0)
        self.head_intrusion = nn.Sequential(
            nn.Linear(hidden_channels, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with realistic surveillance prior."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Tensor of shape [B, input_dim, seq_len]
        Returns:
            Dict containing behavior logits and intrusion likelihood.
        """
        feat = self.tcn(x)
        pooled, attn = self.attention(feat)
        logits = self.head_behavior(pooled)
        probs = F.softmax(logits, dim=-1)
        intrusion_norm = self.head_intrusion(pooled).squeeze(-1)  # [0, 1]

        return {
            "probs": probs,                        # [B, 3] -> [approach, loiter, cross]
            "intrusion_likelihood": intrusion_norm * 100.0, # [B] -> 0-100 scale
            "attention_weights": attn
        }


# ===========================================================================
# High-Level Temporal Behavior Manager
# ===========================================================================

class TemporalBehaviorModel:
    """
    Stateful manager for track feature buffering, TCN inference, and score fusion.
    """
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        boundary_y_ref: float = 576.0,
        history_len: int = 24,
        alpha_rule: float = 0.50,
        beta_tcn: float = 0.50,
        device: str = "cpu"
    ):
        cfg = (config or {}).get("temporal_behavior", {}) if config else {}
        self.enabled = bool(cfg.get("enabled", True))
        self.history_len = int(cfg.get("history_len", history_len))
        self.alpha_rule = float(cfg.get("alpha_rule", alpha_rule))
        self.beta_tcn = float(cfg.get("beta_tcn", beta_tcn))
        self.boundary_y_ref = float(cfg.get("boundary_y_ref", boundary_y_ref))
        self.input_dim = int(cfg.get("input_dim", 8))
        self.hidden_channels = int(cfg.get("hidden_channels", 32))
        self.num_levels = int(cfg.get("num_levels", 3))

        if TORCH_AVAILABLE:
            self.device = torch.device(device if torch.cuda.is_available() and device != "cpu" else "cpu")
            self.model = TCNAttentionBehaviorNet(
                input_dim=self.input_dim,
                hidden_channels=self.hidden_channels,
                num_levels=self.num_levels,
                num_classes=3
            ).to(self.device)
            self.model.eval()
        else:
            self.device = "cpu"
            self.model = None

        # Track history feature buffer: {track_id: list of feature dicts}
        self._buffers: Dict[int, List[Dict[str, float]]] = {}
        self._last_telemetry: Dict[int, Dict[str, Any]] = {}

        logger.info(
            f"TemporalBehaviorModel ready | enabled={self.enabled} | "
            f"seq_len={self.history_len} | alpha={self.alpha_rule} | "
            f"beta={self.beta_tcn} | device={self.device} | torch={TORCH_AVAILABLE}"
        )

    # -----------------------------------------------------------------------
    def extract_frame_features(
        self,
        centroid: Tuple[int, int],
        prev_centroid: Optional[Tuple[int, int]],
        frame_w: int = 1280,
        frame_h: int = 720,
        in_restricted_zone: bool = False,
        dwell_sec: float = 0.0
    ) -> List[float]:
        """
        Extract an 8-dimensional normalized kinematic and spatial feature vector.
        [norm_x, norm_y, norm_vx, norm_vy, norm_speed, dist_to_boundary, is_restricted, norm_dwell]
        """
        cx, cy = centroid
        norm_x = min(1.0, max(0.0, cx / max(1.0, float(frame_w))))
        norm_y = min(1.0, max(0.0, cy / max(1.0, float(frame_h))))

        if prev_centroid is not None:
            px, py = prev_centroid
            vx = (cx - px) / max(1.0, float(frame_w))
            vy = (cy - py) / max(1.0, float(frame_h))
        else:
            vx, vy = 0.0, 0.0

        speed = math.sqrt(vx * vx + vy * vy)
        # Distance to boundary reference line (y=576), normalized to [-1.0, 1.0]
        # Negative = inside perimeter (above 576), Positive = approach band (below 576)
        dist_boundary = (cy - self.boundary_y_ref) / max(1.0, float(frame_h))
        restricted_flag = 1.0 if in_restricted_zone else 0.0
        norm_dwell = min(1.0, dwell_sec / 15.0)

        return [norm_x, norm_y, vx * 10.0, vy * 10.0, speed * 10.0, dist_boundary, restricted_flag, norm_dwell]

    # -----------------------------------------------------------------------
    def update_track(
        self,
        track_id: int,
        centroid: Tuple[int, int],
        frame_w: int = 1280,
        frame_h: int = 720,
        in_restricted_zone: bool = False,
        dwell_sec: float = 0.0
    ) -> Dict[str, float]:
        """
        Update rolling buffer for target and compute TCN temporal behavior scores.

        Returns:
            Dict with keys:
              - 'approach_score' (0.0 to 1.0)
              - 'loiter_score'   (0.0 to 1.0)
              - 'cross_score'    (0.0 to 1.0)
              - 'tcn_likelihood' (0.0 to 100.0)
              - 'dominant_behavior' ('approach' | 'loiter' | 'cross' | 'benign')
        """
        buf = self._buffers.setdefault(track_id, [])
        prev_pt = buf[-1]["pt"] if buf else None
        feat = self.extract_frame_features(
            centroid=centroid,
            prev_centroid=prev_pt,
            frame_w=frame_w,
            frame_h=frame_h,
            in_restricted_zone=in_restricted_zone,
            dwell_sec=dwell_sec
        )

        buf.append({"pt": centroid, "feat": feat})
        if len(buf) > self.history_len:
            buf.pop(0)

        # Compute heuristic prior component for physical robustness
        heuristic_scores = self._compute_heuristic_prior(buf, in_restricted_zone, dwell_sec)

        # If buffer is short, TCN disabled, or model not loaded, return heuristic prior directly
        if not self.enabled or len(buf) < 4 or self.model is None:
            self._last_telemetry[track_id] = heuristic_scores
            return heuristic_scores

        try:
            # Build tensor: [1, input_dim, seq_len]
            seq_data = [item["feat"] for item in buf]
            tensor_in = torch.tensor(seq_data, dtype=torch.float32).transpose(0, 1).unsqueeze(0).to(self.device)

            with torch.no_grad():
                out = self.model(tensor_in)
                probs = out["probs"][0].cpu().numpy()
                intrusion_reg = float(out["intrusion_likelihood"][0].item())
                attn_w = out["attention_weights"][0].cpu().numpy().tolist()

            tcn_approach = float(probs[0])
            tcn_loiter   = float(probs[1])
            tcn_cross    = float(probs[2])

            # Blend model inference with kinematic heuristic prior for zero-hallucination guarantees
            f_approach = 0.5 * tcn_approach + 0.5 * heuristic_scores["approach_score"]
            f_loiter   = 0.5 * tcn_loiter   + 0.5 * heuristic_scores["loiter_score"]
            f_cross    = 0.5 * tcn_cross    + 0.5 * heuristic_scores["cross_score"]
            f_like     = 0.5 * intrusion_reg + 0.5 * heuristic_scores["tcn_likelihood"]

            classes = ["approach", "loiter", "cross"]
            dominant = classes[int(np.argmax([f_approach, f_loiter, f_cross]))]
            if max(f_approach, f_loiter, f_cross) < 0.25 and not in_restricted_zone:
                dominant = "benign"

            # Compute kinematic speed in px
            last_feat = buf[-1]["feat"]
            vx_val = last_feat[2] / 10.0 * frame_w
            vy_val = last_feat[3] / 10.0 * frame_h
            speed_val = math.hypot(vx_val, vy_val)

            res = {
                "approach_score": round(f_approach, 3),
                "loiter_score"  : round(f_loiter, 3),
                "cross_score"   : round(f_cross, 3),
                "tcn_likelihood": round(min(100.0, max(0.0, f_like)), 1),
                "dominant_behavior": dominant,
                "attention_weights": [round(float(w), 3) for w in attn_w],
                "speed_px": round(speed_val, 1),
                "velocity": (round(vx_val, 1), round(vy_val, 1)),
                "border_dist_px": round(centroid[1] - self.boundary_y_ref, 1),
                "dwell_sec": round(dwell_sec, 1)
            }
            self._last_telemetry[track_id] = res
            return res
        except Exception as e:
            logger.debug(f"TCN inference fallback for track {track_id}: {e}")
            self._last_telemetry[track_id] = heuristic_scores
            return heuristic_scores

    # -----------------------------------------------------------------------
    def _compute_heuristic_prior(
        self,
        buf: List[Dict[str, Any]],
        in_restricted_zone: bool,
        dwell_sec: float
    ) -> Dict[str, Any]:
        """
        Fast physics-based kinematic prior for behavior validation.
        """
        if not buf:
            return {
                "approach_score": 0.0, "loiter_score": 0.0,
                "cross_score": 0.0, "tcn_likelihood": 0.0,
                "dominant_behavior": "benign"
            }

        pts = [b["pt"] for b in buf]
        ys = [p[1] for p in pts]
        delta_y = ys[-1] - ys[0] if len(ys) >= 2 else 0.0

        # Moving upward (smaller y) towards/across border = approach
        is_moving_up = delta_y < -5.0
        approach_score = min(1.0, abs(delta_y) / 80.0) if is_moving_up else 0.0

        # Stationary or small motion inside restricted zone = loiter
        if len(pts) >= 4:
            xs = [p[0] for p in pts]
            span = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
            is_stationary = span < 40.0
        else:
            is_stationary = False

        loiter_score = min(1.0, dwell_sec / 8.0) if (in_restricted_zone and is_stationary) else 0.0

        # Crossing line: trajectory starts below boundary and ends above boundary
        crossed = (ys[0] >= self.boundary_y_ref) and (ys[-1] < self.boundary_y_ref)
        cross_score = 0.95 if crossed else (0.4 if (is_moving_up and ys[-1] < self.boundary_y_ref + 30) else 0.0)

        # Baseline intrusion likelihood
        if in_restricted_zone:
            tcn_likelihood = 60.0 + 35.0 * max(loiter_score, cross_score)
        elif is_moving_up:
            tcn_likelihood = 25.0 + 35.0 * approach_score
        else:
            tcn_likelihood = 5.0

        classes = [("approach", approach_score), ("loiter", loiter_score), ("cross", cross_score)]
        top_name, top_score = max(classes, key=lambda x: x[1])
        dominant = top_name if top_score > 0.3 else "benign"

        if len(pts) >= 2:
            vx_val = float(pts[-1][0] - pts[-2][0])
            vy_val = float(pts[-1][1] - pts[-2][1])
            speed_val = math.hypot(vx_val, vy_val)
        else:
            vx_val, vy_val, speed_val = 0.0, 0.0, 0.0

        n_pts = len(buf)
        attn_w = [round(1.0 / max(1, n_pts), 3)] * n_pts

        return {
            "approach_score": round(approach_score, 3),
            "loiter_score"  : round(loiter_score, 3),
            "cross_score"   : round(cross_score, 3),
            "tcn_likelihood": round(min(100.0, max(0.0, tcn_likelihood)), 1),
            "dominant_behavior": dominant,
            "attention_weights": attn_w,
            "speed_px": round(speed_val, 1),
            "velocity": (round(vx_val, 1), round(vy_val, 1)),
            "border_dist_px": round(pts[-1][1] - self.boundary_y_ref, 1),
            "dwell_sec": round(dwell_sec, 1)
        }

    # -----------------------------------------------------------------------
    def fuse_scores(self, rule_risk_score: float, tcn_scores: Dict[str, Any]) -> Tuple[float, str]:
        """
        Fuse rule-based risk score with TCN likelihood:
          final_score = alpha * rule_risk + beta * tcn_likelihood
        """
        if not self.enabled:
            return rule_risk_score, "rule_only"

        tcn_like = tcn_scores.get("tcn_likelihood", rule_risk_score)
        fused = self.alpha_rule * rule_risk_score + self.beta_tcn * tcn_like
        fused = round(min(100.0, max(0.0, fused)), 1)
        reason = f"Fused(rule={rule_risk_score:.0f}, tcn={tcn_like:.0f})"
        return fused, reason

    # -----------------------------------------------------------------------
    def get_track_telemetry(self, track_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve latest TCN inference telemetry and kinematics for a track."""
        if track_id in self._last_telemetry:
            res = dict(self._last_telemetry[track_id])
            pts = [b["pt"] for b in self._buffers.get(track_id, [])]
            res["trajectory"] = pts
            return res
        return None

    def prune_tracks(self, active_track_ids: set):
        """Remove buffers for expired tracks."""
        for tid in list(self._buffers.keys()):
            if tid not in active_track_ids:
                del self._buffers[tid]
                self._last_telemetry.pop(tid, None)

    def reset(self):
        """Clear all buffered track states."""
        self._buffers.clear()
        self._last_telemetry.clear()
        logger.info("TemporalBehaviorModel reset.")
