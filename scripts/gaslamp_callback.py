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
    "meta": {"max_steps": 0, "total_epochs": 0, "task_type": "sft"},
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

    Args:
        port      : HTTP port for the dashboard (default 8080).
        task_type : One of "sft", "dpo", "grpo", "vision" — controls which
                    panels the dashboard renders.
    """

    def __init__(self, port: int = 8080, task_type: str = "sft"):
        self.port = port
        self.task_type = task_type.lower()
        self.server_thread = None
        self.httpd = None
        self.is_running = False
        self._train_start_time = None
        # Memory baseline — captured once before training starts
        self._baseline_vram_mb = 0
        self._total_vram_mb = 0
        self._train_runtime_seconds = None

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Starts the background HTTP daemon server."""
        global _GLOBAL_PAYLOAD
        if state.is_world_process_zero and not self.is_running:
            self._train_start_time = time.time()
            # Snapshot GPU memory *before* training begins (model load baseline)
            try:
                import torch
                if torch.cuda.is_available():
                    self._baseline_vram_mb = int(torch.cuda.max_memory_reserved() / (1024 * 1024))
                    props = torch.cuda.get_device_properties(0)
                    self._total_vram_mb = int(props.total_memory / (1024 * 1024))
            except Exception:
                pass
            _GLOBAL_PAYLOAD["phase"] = "training"
            _GLOBAL_PAYLOAD["meta"]["task_type"] = self.task_type
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
            elapsed = 0
            if self._train_start_time:
                elapsed = round(time.time() - self._train_start_time, 1)
            _GLOBAL_PAYLOAD["phase"] = "completed"
            _GLOBAL_PAYLOAD["elapsed_seconds"] = elapsed
            _GLOBAL_PAYLOAD["eta_seconds"] = 0
            # Final memory summary (mirrors unsloth-studio Colab cell 12)
            try:
                import torch
                if torch.cuda.is_available():
                    peak_mb   = int(torch.cuda.max_memory_reserved() / (1024 * 1024))
                    lora_mb   = peak_mb - self._baseline_vram_mb
                    total_mb  = self._total_vram_mb or 1
                    _GLOBAL_PAYLOAD.setdefault("hardware", {})
                    _GLOBAL_PAYLOAD["hardware"].update({
                        "peak_vram_mb":      peak_mb,
                        "baseline_vram_mb":  self._baseline_vram_mb,
                        "lora_vram_mb":      max(lora_mb, 0),
                        "total_vram_mb":     self._total_vram_mb,
                        "vram_pct":          round(peak_mb / total_mb * 100, 1),
                        "lora_vram_pct":     round(max(lora_mb, 0) / total_mb * 100, 1),
                    })
            except Exception:
                pass
            _GLOBAL_PAYLOAD["train_runtime_seconds"] = elapsed
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
                elif torch.backends.mps.is_available():
                    hw_info["device"] = "Apple MPS"
                    hw_info["peak_vram_mb"] = int(torch.mps.driver_allocated_memory() / (1024 * 1024))
            except Exception:
                pass

            # CPU RAM (optional, guarded)
            try:
                import psutil
                hw_info["cpu_ram_mb"] = int(psutil.Process().memory_info().rss / (1024 * 1024))
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

            # 3b. Epoch
            current_epoch = round(getattr(state, 'epoch', 0) or 0, 2)

            # 4. Build enriched log entry from current logs dict
            log_entry = {}
            if logs:
                step = state.global_step if hasattr(state, 'global_step') else 0
                log_entry["step"] = step

                # Standard metrics (all task types)
                for key in ("loss", "eval_loss", "learning_rate", "grad_norm",
                            "train_samples_per_second", "train_steps_per_second"):
                    if key in logs:
                        log_entry[key] = logs[key]

                # Tokens/sec — derive from samples*seq_len or use direct key
                if "train_samples_per_second" in logs:
                    # Approximate: multiply by max_seq_length if available
                    seq_len = getattr(args, "max_seq_length", None) or getattr(args, "max_length", None)
                    if seq_len:
                        log_entry["tokens_per_sec"] = round(logs["train_samples_per_second"] * seq_len, 1)
                if "tokens_per_sec" in logs:
                    log_entry["tokens_per_sec"] = logs["tokens_per_sec"]

                # DPO-specific
                for key in ("rewards/chosen", "rewards/rejected",
                            "rewards/accuracies", "rewards/margins",
                            "logps/chosen", "logps/rejected",
                            "kl", "kl_divergence"):
                    if key in logs:
                        # Normalise key names for the frontend
                        clean = key.replace("/", "_")
                        log_entry[clean] = logs[key]

                # GRPO-specific
                for key in ("reward", "reward_std", "kl", "kl_divergence",
                            "completion_length", "policy_loss", "value_loss"):
                    if key in logs:
                        log_entry[key] = logs[key]

            # 5. Overwrite the global payload (no disk I/O)
            existing_logs = _GLOBAL_PAYLOAD.get("logs", [])
            # Merge into existing step entry if present, otherwise append
            step_key = log_entry.get("step", -1)
            merged = False
            for entry in existing_logs:
                if entry.get("step") == step_key:
                    entry.update(log_entry)
                    merged = True
                    break
            if not merged and log_entry:
                existing_logs.append(log_entry)

            # Live memory breakdown (unsloth-studio style)
            hw_memory_extras = {}
            try:
                import torch
                if torch.cuda.is_available():
                    peak_mb  = int(torch.cuda.max_memory_reserved() / (1024 * 1024))
                    lora_mb  = peak_mb - self._baseline_vram_mb
                    total_mb = self._total_vram_mb or 1
                    hw_info.update({
                        "baseline_vram_mb": self._baseline_vram_mb,
                        "lora_vram_mb":     max(lora_mb, 0),
                        "total_vram_mb":    self._total_vram_mb,
                        "vram_pct":         round(peak_mb / total_mb * 100, 1),
                        "lora_vram_pct":    round(max(lora_mb, 0) / total_mb * 100, 1),
                    })
            except Exception:
                pass

            _GLOBAL_PAYLOAD = {
                "meta": {
                    "max_steps":    state.max_steps,
                    "total_epochs": getattr(args, "num_train_epochs", 0),
                    "task_type":    self.task_type,
                    "current_epoch": current_epoch,
                },
                "hardware": hw_info,
                "hyperparameters": hp_info,
                "logs": existing_logs,
                "phase": "training",
                "elapsed_seconds": elapsed,
                "eta_seconds": eta,
            }

            # 6. Push to all SSE subscribers instantly
            _notify_subscribers()
