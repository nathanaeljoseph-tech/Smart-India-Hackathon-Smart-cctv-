"""
scripts/demo_webhook_receiver.py - Demo External VMS Webhook Receiver Client
=============================================================================
BorderVigil-AI | Tier-3 Demo Client

Demonstrates integration with third-party VMS / Command Centers:
  1. Starts a lightweight HTTP server on port 9090.
  2. Registers http://127.0.0.1:9090/webhook with BorderVigil-AI VMS API.
  3. Receives and displays high-risk security alert JSON payloads in real time.
"""

import sys
import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.request

VMS_API_URL = "http://127.0.0.1:8080/api/v1/webhooks"
MY_HOST = "127.0.0.1"
MY_PORT = 9090
CALLBACK_URL = f"http://{MY_HOST}:{MY_PORT}/webhook"


class WebhookHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)

        try:
            alert = json.loads(post_data.decode("utf-8"))
            print("\n" + "="*65)
            print(">>> [VMS CLIENT RECEIVED HIGH-RISK WEBHOOK ALERT] <<<")
            print(f"Timestamp   : {alert.get('timestamp')}")
            print(f"Event ID    : {alert.get('id', 'N/A')}")
            print(f"Camera      : {alert.get('camera_id', 'CAM-01')}")
            print(f"Track ID    : {alert.get('track_id')} (Global: {alert.get('global_track_id', 'N/A')})")
            print(f"Risk Score  : {alert.get('risk_score', 0):.0f} [{alert.get('alert_level')}]")
            print(f"Behavior    : {alert.get('behavior')} in zone: {alert.get('zone_id')}")
            if alert.get("vlm_caption"):
                print(f"VLM Caption : \"{alert.get('vlm_caption')}\"")
            print("="*65 + "\n")
        except Exception as e:
            print(f"Error parsing webhook payload: {e}")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"received"}')

    def log_message(self, format, *args):
        pass  # Suppress default HTTP server access logs


def register_with_bordervigil():
    print(f"Attempting to register webhook ({CALLBACK_URL}) with BorderVigil-AI at {VMS_API_URL}...")
    payload = json.dumps({
        "callback_url": CALLBACK_URL,
        "description": "Demonstration Command Center Receiver",
        "min_risk": 70.0
    }).encode("utf-8")

    req = urllib.request.Request(
        VMS_API_URL,
        data=payload,
        headers={"Content-Type": "application/json", "X-API-Key": "bordervigil-secret-key"}
    )
    try:
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print(f"Registration successful! Response: {data}")
            return True
    except Exception as e:
        print(f"Registration warning: BorderVigil API not yet reachable at {VMS_API_URL} ({e}).")
        print("Start BorderVigil-AI (python main.py), then this receiver will catch alerts.")
        return False


def main():
    print("="*65)
    print("BorderVigil-AI - External VMS Webhook Receiver Client")
    print("="*65)

    server = HTTPServer((MY_HOST, MY_PORT), WebhookHandler)
    print(f"Listening for alert webhooks at {CALLBACK_URL}...")

    # Register in background / attempt
    register_with_bordervigil()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down webhook receiver.")
        server.server_close()


if __name__ == "__main__":
    main()
