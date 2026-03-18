import os
import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from typing import Dict, Any, Optional

# ─── Global State ────────────────────────────────────────────────────
# Shared in-memory state accessible by the HTTP handler thread.
_GLOBAL_PAYLOAD = {
    "meta": {"max_steps": 0, "total_epochs": 0},
    "hardware": {},
    "hyperparameters": {},
    "logs": [],
    "phase": "idle",           # idle | training | completed | error
    "elapsed_seconds": 0,
    "eta_seconds": None,
}

# SSE subscribers: list of threading.Event objects to notify on new data
_SSE_SUBSCRIBERS = []
_SSE_LOCK = threading.Lock()


def _notify_subscribers():
    """Wake up all SSE subscriber threads so they push the latest payload."""
    with _SSE_LOCK:
        for event in _SSE_SUBSCRIBERS:
            event.set()


# ─── HTTP Request Handler ────────────────────────────────────────────
class DashboardRequestHandler(BaseHTTPRequestHandler):
    """Serves the dashboard HTML, JSON metrics, health check, and SSE stream."""

    template_path = "templates/dashboard.html"

    def do_GET(self):
        try:
            if self.path == "/":
                self._serve_html()
            elif self.path == "/api/metrics":
                self._serve_metrics()
            elif self.path == "/api/health":
                self._serve_health()
            elif self.path == "/api/stream":
                self._serve_sse()
            else:
                self.send_response(404)
                self.end_headers()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def _serve_html(self):
        self.send_response(200)
        self.send_header("Content-type", "text/html; charset=utf-8")
        self.end_headers()
        if os.path.exists(self.template_path):
            with open(self.template_path, "rb") as f:
                self.wfile.write(f.read())
        else:
            self.wfile.write(
                b"<h1>Gaslamp Dashboard</h1>"
                b"<p>Error: templates/dashboard.html not found locally.</p>"
            )

    def _serve_metrics(self):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        try:
            self.wfile.write(json.dumps(_GLOBAL_PAYLOAD).encode("utf-8"))
        except Exception as e:
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

    def _serve_health(self):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "healthy"}).encode("utf-8"))

    def _serve_sse(self):
        """Server-Sent Events endpoint. Holds the connection open and pushes updates."""
        self.send_response(200)
        self.send_header("Content-type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        # Tell the browser to retry every 3 seconds if the connection drops
        self.wfile.write(b"retry: 3000\n\n")
        self.wfile.flush()

        # Register this connection as a subscriber
        notify_event = threading.Event()
        with _SSE_LOCK:
            _SSE_SUBSCRIBERS.append(notify_event)

        try:
            # Send the current full state immediately on connect
            self._send_sse_event(notify_event)

            # Then wait for updates
            while True:
                # Block until new data arrives (or timeout for heartbeat)
                got_data = notify_event.wait(timeout=15)
                if got_data:
                    notify_event.clear()
                    self._send_sse_event(notify_event)
                else:
                    # Heartbeat to keep connection alive
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()

        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass
        finally:
            with _SSE_LOCK:
                if notify_event in _SSE_SUBSCRIBERS:
                    _SSE_SUBSCRIBERS.remove(notify_event)

    def _send_sse_event(self, notify_event):
        """Serialize the current payload as an SSE event."""
        payload = _GLOBAL_PAYLOAD
        step = 0
        logs = payload.get("logs", [])
        if logs:
            last = logs[-1]
            step = last.get("step", 0)

        data = json.dumps(payload)
        msg = f"id: {step}\nevent: progress\ndata: {data}\n\n"
        self.wfile.write(msg.encode("utf-8"))
        self.wfile.flush()

    def log_message(self, format, *args):
        # Suppress HTTP access logs
        pass


# ─── Trainer Callback ────────────────────────────────────────────────
class GaslampDashboardCallback(TrainerCallback):
    """
    HuggingFace TrainerCallback that spawns a local HTTP server with:
      - GET /           → Dashboard HTML
      - GET /api/metrics → Full JSON payload (for reconnection recovery)
      - GET /api/stream  → SSE live stream (instant push updates)
      - GET /api/health  → Health check
    """

    def __init__(self, port: int = 8080):
        self.port = port
        self.server_thread = None
        self.httpd = None
        self.is_running = False
        self._train_start_time = None

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Starts the background HTTP daemon server."""
        global _GLOBAL_PAYLOAD
        if state.is_world_process_zero and not self.is_running:
            self._train_start_time = time.time()
            _GLOBAL_PAYLOAD["phase"] = "training"
            self._start_server()

    def _start_server(self):
        """Spawns the local web server on a background daemon thread."""
        try:
            self.httpd = HTTPServer(("localhost", self.port), DashboardRequestHandler)
            self.server_thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
            self.server_thread.start()
            self.is_running = True

            # Print a prominent, clickable log in the user's terminal
            time.sleep(0.5)
            print("\n" + "=" * 60)
            print(f"🚀 Gaslamp Live Training Dashboard: http://localhost:{self.port}/")
            print("=" * 60 + "\n")
        except OSError as e:
            if e.errno == 48:  # Address already in use
                print(f"⚠️ [Gaslamp] Port {self.port} is busy. Dashboard might already be running.")
            else:
                print(f"⚠️ [Gaslamp] Failed to start web server: {e}")
        except Exception as e:
            print(f"⚠️ [Gaslamp] Failed to start web server: {e}")

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Marks training as completed."""
        global _GLOBAL_PAYLOAD
        if state.is_world_process_zero:
            _GLOBAL_PAYLOAD["phase"] = "completed"
            if self._train_start_time:
                _GLOBAL_PAYLOAD["elapsed_seconds"] = round(time.time() - self._train_start_time, 1)
            _GLOBAL_PAYLOAD["eta_seconds"] = 0
            _notify_subscribers()

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Dict[str, float] = None, **kwargs):
        """Called whenever `trainer.log()` is invoked. Updates payload + notifies SSE subscribers."""
        global _GLOBAL_PAYLOAD

        if state.is_world_process_zero:
            # 1. Hardware stats
            hw_info = {"device": "Unknown", "peak_vram_mb": 0}
            try:
                import torch
                if torch.cuda.is_available():
                    hw_info["device"] = torch.cuda.get_device_name(0)
                    hw_info["peak_vram_mb"] = int(torch.cuda.max_memory_allocated() / (1024 * 1024))
            except Exception:
                pass

            # 2. Hyperparameters
            hp_info = {}
            try:
                hp_info = {
                    "learning_rate": args.learning_rate,
                    "train_batch_size": args.train_batch_size,
                    "gradient_accumulation": args.gradient_accumulation_steps,
                    "optimizer": getattr(args, "optim", "Unknown"),
                    "seed": getattr(args, "seed", None),
                }
            except Exception:
                pass

            # 3. ETA calculation
            elapsed = 0
            eta = None
            if self._train_start_time and state.max_steps > 0:
                elapsed = round(time.time() - self._train_start_time, 1)
                current_step = state.global_step if hasattr(state, 'global_step') else 0
                if current_step > 0:
                    secs_per_step = elapsed / current_step
                    remaining_steps = state.max_steps - current_step
                    eta = round(secs_per_step * remaining_steps, 1)

            # 4. Overwrite the global payload (no disk I/O)
            _GLOBAL_PAYLOAD = {
                "meta": {
                    "max_steps": state.max_steps,
                    "total_epochs": getattr(args, "num_train_epochs", 0),
                },
                "hardware": hw_info,
                "hyperparameters": hp_info,
                "logs": state.log_history,
                "phase": "training",
                "elapsed_seconds": elapsed,
                "eta_seconds": eta,
            }

            # 5. Push to all SSE subscribers instantly
            _notify_subscribers()
