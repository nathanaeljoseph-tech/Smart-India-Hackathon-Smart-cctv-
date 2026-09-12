"""
modules/llm_client.py - Llama 3.1 8B Instruct LLM Client
=========================================================
BorderVigil-AI | Tier-3 Advanced Feature
Problem: SIH26187

Provides operational intelligence via a locally hosted Llama 3.1 8B model:
  1. Incident Summaries: Periodically aggregates high-risk alerts, VLM captions,
     and zone states into an executive tactical brief with operator actions.
  2. Operator Queries (Chat): Parses natural-language security queries,
     filters relevant events from alert logs, and formats structured answers.
"""

import json
import logging
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

logger = logging.getLogger("amst_border_net.llm_client")


class LLMClient:
    """
    Client for Llama 3.1 8B Instruct running locally on Ollama or vLLM.
    """
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        endpoint: str = "http://localhost:11434",
        model: str = "llama3.1:8b",
        provider: str = "ollama",
        summary_interval_sec: float = 300.0,
        timeout_seconds: float = 5.0
    ):
        llm_cfg = (config or {}).get("llm", {}) if config else {}
        self.enabled = bool(llm_cfg.get("enabled", True))
        self.endpoint = str(llm_cfg.get("endpoint", endpoint)).rstrip("/")
        self.model = str(llm_cfg.get("model", model))
        self.provider = str(llm_cfg.get("provider", provider)).lower()
        self.summary_interval = float(llm_cfg.get("summary_interval_sec", summary_interval_sec))
        self.timeout = float(llm_cfg.get("timeout_seconds", timeout_seconds))

        self._last_summary_time = time.time()
        self._last_summary_text = "System initialized. No incidents recorded in initial observation window."
        self._endpoint_online: Optional[bool] = None
        self._last_endpoint_check: float = 0.0

        logger.info(
            f"LLMClient ready | enabled={self.enabled} | provider={self.provider} | "
            f"model={self.model} | endpoint={self.endpoint} | summary_interval={self.summary_interval}s"
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


    # =======================================================================
    # Use Case A: Incident Summaries
    # =======================================================================

    def should_generate_summary(self) -> bool:
        """Check if periodic summary interval has elapsed."""
        if not self.enabled:
            return False
        return (time.time() - self._last_summary_time) >= self.summary_interval

    def generate_incident_summary(self, recent_alerts: List[Dict[str, Any]], time_window_str: str = "Last 10 Minutes") -> str:
        """
        Generate executive operational summary of security events.
        """
        self._last_summary_time = time.time()
        if not recent_alerts:
            self._last_summary_text = f"[{time_window_str} Security Summary] All sectors quiet. Zero perimeter breaches or abnormal behaviors detected."
            return self._last_summary_text

        # Format alerts into structured text for prompt
        alert_lines = []
        for idx, a in enumerate(recent_alerts[:15], 1):
            cam = a.get("camera_id", "CAM-01")
            zone = a.get("zone_id", "perimeter")
            beh = a.get("behavior", "unknown")
            risk = a.get("risk_score", 0.0)
            caption = a.get("vlm_caption") or a.get("reasoning", "")
            ts = a.get("timestamp", datetime.now().strftime("%H:%M:%S"))
            alert_lines.append(f"{idx}. [{ts}] Cam:{cam} | Zone:{zone} | Risk:{risk:.0f} | Behavior:{beh} | Caption:{caption}")

        alerts_block = "\n".join(alert_lines)

        prompt = (
            f"You are a Senior Border Security Operations Commander. Analyze the following security incidents recorded in the {time_window_str}:\n"
            f"{alerts_block}\n\n"
            f"Provide a crisp 3-part operational report in standard military/security format:\n"
            f"1. SITUATION SUMMARY (1-2 sentences on general threat level)\n"
            f"2. KEY INCIDENT HIGHLIGHTS (bullet points on highest-risk breaches)\n"
            f"3. RECOMMENDED OPERATOR ACTIONS (immediate patrol dispatch, PTZ verification, or sensor check)"
        )

        response = None
        if self.is_endpoint_online():
            response = self._call_llm(prompt)

        if not response:
            # Instant deterministic offline fallback (0.0ms)
            response = self._generate_fallback_summary(recent_alerts, time_window_str)

        self._last_summary_text = response
        logger.info(f"LLM Incident Summary generated for {len(recent_alerts)} alerts.")
        return response

    # =======================================================================
    # Use Case B: Operator Queries (Chat / Natural Language Search)
    # =======================================================================

    def answer_operator_query(
        self,
        query: str,
        alert_history: List[Dict[str, Any]],
        live_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process operator question, filter matching alerts, and format response.
        Supports headcount ('how many people are there'), zone checks, and behavior queries.

        Returns:
            Dict with:
              - 'answer': str (natural language reply)
              - 'matched_events': list of alert dicts
              - 'matched_count': int
        """
        q_lower = query.lower()
        live_ctx = live_context or {}
        active_tracks = live_ctx.get("active_tracks", [])
        active_camera = live_ctx.get("active_camera_id", "CAM-01")

        # Headcount & Target Count intent detection
        is_count_query = any(k in q_lower for k in [
            "how many people", "number of people", "count of people", "how many person",
            "how many targets", "how many individuals", "headcount", "how many tracks",
            "how many in", "who is there", "how many detected", "is anyone there"
        ])

        # Step 1: Rule-based & semantic metadata filtering
        matched = self._filter_alerts_by_query(query, alert_history)

        # Step 2: Assemble rich multi-source context
        active_people_tracks = [t for t in active_tracks if t.get("class", "person") == "person" or "track_id" in t]
        active_count = len(active_people_tracks)
        active_tids = [str(t.get("track_id", "?")) for t in active_people_tracks]
        unique_history_ids = list(dict.fromkeys([str(a.get("track_id")) for a in alert_history if a.get("track_id") is not None]))

        # Zone-specific counts
        zone_target = None
        for z in ["perimeter", "approach", "fence", "gate", "red"]:
            if z in q_lower:
                zone_target = "perimeter" if z in ["perimeter", "fence", "red"] else "approach_band"
                break

        zone_active_count = sum(1 for t in active_people_tracks if zone_target and zone_target in str(t.get("zone_id", "")).lower())
        zone_event_count = sum(1 for a in alert_history if zone_target and zone_target in str(a.get("zone_id", "")).lower())

        context_str = (
            f"--- LIVE SURVEILLANCE TELEMETRY ---\n"
            f"Active Camera View: {active_camera}\n"
            f"Live People Currently Tracked: {active_count} (Track IDs: {', '.join(active_tids) if active_tids else 'None'})\n"
            f"Total Unique Historical Suspects Logged: {len(unique_history_ids)} (IDs: {', '.join(unique_history_ids[:8]) if unique_history_ids else 'None'})\n"
        )
        if zone_target:
            context_str += f"Target Zone '{zone_target.upper()}': {zone_active_count} active target(s), {zone_event_count} historical alerts.\n"

        context_str += f"\n--- SECURITY INCIDENT DATABASE (Matched: {len(matched)}) ---\n"
        for idx, m in enumerate(matched[:6], 1):
            context_str += (
                f"Event #{idx}: Track {m.get('track_id')} in {m.get('zone_id', 'zone')} at {m.get('timestamp', 'recent')} "
                f"with Risk {m.get('risk_score', 0):.0f} ({m.get('behavior', 'motion')}). "
                f"Notes: {m.get('vlm_caption') or m.get('reasoning', 'No caption')}\n"
            )

        prompt = (
            f"You are BorderVigil-AI Command Assistant. Answer the operator's query accurately based on the live surveillance telemetry and security records:\n\n"
            f"Operator Query: \"{query}\"\n\n"
            f"{context_str}\n\n"
            f"Instructions:\n"
            f"- If the operator asks about headcount or number of people, state clearly how many people are currently active on feed and how many have been logged overall.\n"
            f"- Give a crisp, authoritative response with precise numbers and track IDs.\n"
            f"- Follow up with a tactical recommendation."
        )

        llm_reply = self._call_llm(prompt)
        if not llm_reply:
            if is_count_query:
                # Deterministic high-precision tactical headcount response
                if active_count > 0:
                    active_desc = f"**{active_count} person(s)** currently in view on {active_camera} (Active Track IDs: #{', #'.join(active_tids)})."
                else:
                    active_desc = f"**0 active persons** detected right now on {active_camera}."

                if unique_history_ids:
                    hist_desc = f"Across recorded incidents, **{len(unique_history_ids)} unique suspect(s)** have been logged (Track IDs: #{', #'.join(unique_history_ids[:6])})."
                else:
                    hist_desc = "No security incident breach tracks have been recorded yet."

                zone_info = ""
                if zone_target:
                    zone_info = f"\n• **Sector ({zone_target.upper()})**: {zone_active_count} active person(s) currently inside, {zone_event_count} incident event(s) recorded."

                llm_reply = (
                    f"**Personnel & Target Headcount Report:**\n\n"
                    f"• **Live Camera Feed**: {active_desc}\n"
                    f"• **Incident Database**: {hist_desc}{zone_info}\n\n"
                    f"**Tactical Status**: All monitored sectors under automated YOLO11 + ByteTrack and TCN temporal surveillance. "
                    f"{'Perimeter breach alert active; maintain visual focus.' if active_count > 0 or len(matched) > 0 else 'Perimeter secure.'}"
                )
            elif not matched:
                llm_reply = f"No security events found matching '{query}'. All recorded perimeter sectors and time windows on {active_camera} returned clear."
            else:
                top_risk = max([float(m.get("risk_score", 0)) for m in matched])
                severity_tag = "CRITICAL" if top_risk >= 75 else ("HIGH" if top_risk >= 50 else "MODERATE")
                bullet_lines = []
                for m in matched[:4]:
                    t_id = m.get("track_id", "N/A")
                    z_id = m.get("zone_id", "perimeter")
                    b_type = m.get("behavior", "motion")
                    r_val = float(m.get("risk_score", 0))
                    cap = m.get("vlm_caption") or m.get("reasoning", "Motion detected")
                    bullet_lines.append(f"  • Track #{t_id} [{z_id} | Risk: {r_val:.0f} | {b_type.upper()}]: \"{cap}\"")
                
                bullets = "\n".join(bullet_lines)
                llm_reply = (
                    f"Retrieved {len(matched)} event(s) matching your query. Highest severity recorded: {severity_tag} (Risk: {top_risk:.0f}).\n\n"
                    f"Incident Highlights:\n{bullets}\n\n"
                    f"Tactical Recommendation: Verify live camera feed on flagged sector; dispatch ground patrol if targets persist near barrier wire."
                )

        return {
            "query": query,
            "answer": llm_reply,
            "matched_events": matched,
            "matched_count": len(matched)
        }

    # -----------------------------------------------------------------------
    def _filter_alerts_by_query(self, query: str, history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter alerts based on keywords, track IDs, zones, risk levels, and timestamps."""
        q = query.lower()
        results = []

        # Track ID extraction (e.g. "track 2", "track #101", "#42")
        track_match = re.search(r'\btrack\s*#?(\d+)\b', q) or re.search(r'#(\d+)\b', q)
        target_track_id = int(track_match.group(1)) if track_match else None

        # Keywords extraction
        high_risk_only = any(w in q for w in ["high", "critical", "danger", "breach", "severe"])
        loiter_only = any(w in q for w in ["loiter", "dwell", "waiting", "lingering"])
        approach_only = any(w in q for w in ["approach", "approaching", "coming"])
        tamper_only = any(w in q for w in ["tamper", "fence", "cut", "shake", "sensor"])
        abandoned_only = any(w in q for w in ["abandoned", "bag", "backpack", "unattended", "package"])

        # Extract zone hints
        zone_hint = None
        for z_name in ["perimeter", "approach_band", "gate", "fence", "parking"]:
            if z_name in q:
                zone_hint = z_name
                break

        for a in history:
            risk = float(a.get("risk_score", 0.0))
            beh = str(a.get("behavior", "")).lower()
            zone = str(a.get("zone_id", "")).lower()
            tid = a.get("track_id")

            if target_track_id is not None and str(tid) != str(target_track_id):
                continue
            if high_risk_only and risk < 50.0:
                continue
            if loiter_only and "loiter" not in beh:
                continue
            if approach_only and "approach" not in beh:
                continue
            if tamper_only and ("tamper" not in beh and "tamper" not in zone):
                continue
            if abandoned_only and "abandoned" not in beh:
                continue
            if zone_hint and zone_hint not in zone:
                continue

            results.append(a)

        # Return most recent matching alerts
        return results[::-1] if results else history[-5:][::-1]

    # -----------------------------------------------------------------------
    def _call_llm(self, prompt: str) -> Optional[str]:
        if not _REQUESTS_AVAILABLE:
            return None
        try:
            if self.provider == "ollama":
                url = f"{self.endpoint}/api/generate"
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 256}
                }
                res = requests.post(url, json=payload, timeout=self.timeout)
                if res.status_code == 200:
                    return res.json().get("response", "").strip()
            else:
                url = f"{self.endpoint}/v1/chat/completions"
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 256,
                    "temperature": 0.3
                }
                res = requests.post(url, json=payload, timeout=self.timeout)
                if res.status_code == 200:
                    choices = res.json().get("choices", [])
                    if choices:
                        return choices[0].get("message", {}).get("content", "").strip()
        except Exception as e:
            logger.debug(f"LLM call to {self.endpoint} failed ({e}). Using robust local generator.")
        return None

    # -----------------------------------------------------------------------
    def _generate_fallback_summary(self, alerts: List[Dict[str, Any]], window_str: str) -> str:
        """Deterministic fallback summary for offline demos."""
        high_risk = [a for a in alerts if float(a.get("risk_score", 0)) >= 60.0]
        loitering = [a for a in alerts if a.get("behavior") == "loiter"]
        tampers = [a for a in alerts if "tamper" in str(a.get("behavior", "")).lower()]

        lines = [
            f"--- SECURITY OPERATIONAL BRIEF ({window_str}) ---",
            f"1. SITUATION SUMMARY: Surveillance active across perimeter sectors. Processed {len(alerts)} alert events, including {len(high_risk)} high-threat security triggers.",
            f"2. KEY INCIDENT HIGHLIGHTS:"
        ]
        if high_risk:
            top = high_risk[0]
            lines.append(f"  * CRITICAL: Track {top.get('track_id')} in {top.get('zone_id')} reached Risk {top.get('risk_score', 0):.0f} ({top.get('behavior')}). {top.get('vlm_caption', '')}")
        if tampers:
            lines.append(f"  * TAMPER: Perimeter vibration/disturbance detected on fence line barrier.")
        if loitering:
            lines.append(f"  * LOITER: Persistent dwelling detected inside restricted perimeter zone.")
        if not high_risk and not tampers and not loitering:
            lines.append(f"  * Normal traffic and approach-band monitoring; zero perimeter breaches confirmed.")

        lines.append("3. RECOMMENDED OPERATOR ACTIONS: Verify live feed for flagged tracks; keep patrol team on alert.")
        return "\n".join(lines)

    def get_last_summary(self) -> str:
        return self._last_summary_text
