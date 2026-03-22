"""
mlx_gaslamp_dashboard.py — Live training dashboard for mlx-tune on Apple Silicon.

mlx-tune's SFTTrainer has no callbacks API, so this module intercepts stdout to
parse training metrics and serve them via the same HTTP dashboard used on the
NVIDIA/TRL path.

Usage:
    from mlx_gaslamp_dashboard import MlxGaslampDashboard

    with MlxGaslampDashboard(iters=ITERS, task_type="sft", hyperparams={"learning_rate": LR, ...}):
        trainer.train()

Dashboard → http://localhost:8080/
Terminal  → .venv/bin/python scripts/terminal_dashboard.py
"""

import os
import re
import sys
import json
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

# Shared payload updated by the stdout parser and read by the HTTP handler
_GLOBAL_PAYLOAD = {
    "meta": {"max_steps": 0, "total_epochs": 1, "task_type": "sft"},
    "hardware": {"device": "Apple MPS", "peak_vram_mb": 0},
    "hyperparameters": {},
    "logs": [],
    "phase": "idle",
    "elapsed_seconds": 0,
    "eta_seconds": None,
}

# SSE subscribers
_SSE_SUBSCRIBERS = []
_SSE_LOCK = threading.Lock()


def _notify_subscribers():
    """Wake up all SSE subscriber threads so they push the latest payload."""
    with _SSE_LOCK:
        for event in _SSE_SUBSCRIBERS:
            event.set()


# mlx-tune stdout patterns
# "Iter 10: Train loss 1.839, Learning Rate 1.990e-04, It/sec 1.889, Tokens/sec 423.247, Trained Tokens 2241, Peak mem 1.725 GB"
_TRAIN_RE = re.compile(
    r"Iter\s+(\d+):\s+Train loss\s+([\d.]+),\s+Learning Rate\s+([\d.e+\-]+)"
    r".*?It/sec\s+([\d.]+).*?Tokens/sec\s+([\d.]+).*?Trained Tokens\s+(\d+)"
    r".*?Peak mem\s+([\d.]+)\s+GB"
)
# "Iter 100: Val loss 1.352, Val took 3.885s"
_VAL_RE = re.compile(r"Iter\s+(\d+):\s+Val loss\s+([\d.]+)")


class _TeeWriter:
    """Writes to the original stdout and calls on_line for each complete line."""

    def __init__(self, original, on_line):
        self._original = original
        self._on_line = on_line
        self._buf = ""

    def write(self, text):
        self._original.write(text)
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self._on_line(line)

    def flush(self):
        self._original.flush()

    def __getattr__(self, name):
        return getattr(self._original, name)


class _DashboardHandler(BaseHTTPRequestHandler):
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
            self.wfile.write(b"<p>Error: templates/dashboard.html not found.</p>")

    def _serve_metrics(self):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(_GLOBAL_PAYLOAD).encode())

    def _serve_health(self):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "healthy"}).encode())

    def _serve_sse(self):
        """Server-Sent Events — pushes live updates to the browser without page refresh."""
        self.send_response(200)
        self.send_header("Content-type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        self.wfile.write(b"retry: 3000\n\n")
        self.wfile.flush()

        notify_event = threading.Event()
        with _SSE_LOCK:
            _SSE_SUBSCRIBERS.append(notify_event)

        try:
            # Send full state immediately on connect
            self._send_sse_event()

            while True:
                got_data = notify_event.wait(timeout=15)
                if got_data:
                    notify_event.clear()
                    self._send_sse_event()
                else:
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass
        finally:
            with _SSE_LOCK:
                if notify_event in _SSE_SUBSCRIBERS:
                    _SSE_SUBSCRIBERS.remove(notify_event)

    def _send_sse_event(self):
        payload = _GLOBAL_PAYLOAD
        logs = payload.get("logs", [])
        step = logs[-1].get("step", 0) if logs else 0
        data = json.dumps(payload)
        msg = f"id: {step}\nevent: progress\ndata: {data}\n\n"
        self.wfile.write(msg.encode("utf-8"))
        self.wfile.flush()

    def log_message(self, format, *args):
        pass  # suppress HTTP access logs


class MlxGaslampDashboard:
    """
    Context manager that wraps mlx-tune's trainer.train() to provide a live
    Gaslamp dashboard at http://localhost:{port}/.

    Args:
        iters       : Total training iterations (drives the progress bar).
        port        : HTTP port for the dashboard server (default 8080).
        task_type   : One of "sft", "dpo", "grpo", "vision" — controls which
                      panels the dashboard renders.
        hyperparams : Dict shown in the Hyperparameters panel, e.g.
                      {"learning_rate": LR, "batch_size": BATCH_SIZE, "lora_rank": R}
    """

    def __init__(self, iters: int, port: int = 8080, task_type: str = "sft", hyperparams: dict = None):
        self.iters = iters
        self.port = port
        self.task_type = task_type.lower()
        self._original_stdout = None
        self._httpd = None
        self._start_time = None
        self._baseline_vram_mb = 0

        global _GLOBAL_PAYLOAD
        _GLOBAL_PAYLOAD = {
            "meta": {"max_steps": iters, "total_epochs": 1, "task_type": self.task_type, "current_epoch": 0},
            "hardware": {"device": "Apple MPS", "peak_vram_mb": 0, "baseline_vram_mb": 0, "total_vram_mb": 0, "lora_vram_mb": 0, "vram_pct": 0, "lora_vram_pct": 0},
            "hyperparameters": hyperparams or {},
            "logs": [],
            "phase": "idle",
            "elapsed_seconds": 0,
            "eta_seconds": None,
            "train_runtime_seconds": None,
        }

    def __enter__(self):
        self._start_server()
        self._start_time = time.time()
        # For Apple Silicon we don't have a CUDA reserved-memory API,
        # so we treat the current peak as the baseline before training.
        try:
            import psutil
            self._baseline_vram_mb = int(psutil.Process().memory_info().rss / (1024 * 1024))
        except Exception:
            self._baseline_vram_mb = 0
        _GLOBAL_PAYLOAD["phase"] = "training"
        self._original_stdout = sys.stdout
        sys.stdout = _TeeWriter(sys.stdout, self._parse_line)
        return self

    def __exit__(self, *_):
        sys.stdout = self._original_stdout
        elapsed = round(time.time() - self._start_time, 1) if self._start_time else 0
        _GLOBAL_PAYLOAD["phase"] = "completed"
        _GLOBAL_PAYLOAD["elapsed_seconds"] = elapsed
        _GLOBAL_PAYLOAD["eta_seconds"] = 0
        _GLOBAL_PAYLOAD["train_runtime_seconds"] = elapsed
        _notify_subscribers()

    def _start_server(self):
        try:
            self._httpd = HTTPServer(("localhost", self.port), _DashboardHandler)
            t = threading.Thread(target=self._httpd.serve_forever, daemon=True)
            t.start()
            time.sleep(0.3)
            print(f"\n{'='*60}")
            print(f"🚀 Gaslamp Live Training Dashboard: http://localhost:{self.port}/")
            print(f"{'='*60}\n")
        except OSError as e:
            if e.errno == 48:
                print(f"[Gaslamp] Port {self.port} already in use — dashboard may already be running.")
            else:
                print(f"[Gaslamp] Could not start dashboard server: {e}")

    def _parse_line(self, line: str):
        global _GLOBAL_PAYLOAD

        m = _TRAIN_RE.search(line)
        if m:
            step, loss, lr, its, tps, tokens, peak_gb = m.groups()
            step    = int(step)
            peak_mb = int(float(peak_gb) * 1024)
            _GLOBAL_PAYLOAD["hardware"]["peak_vram_mb"] = peak_mb
            # LoRA memory delta: peak − baseline (Apple MPS uses system RAM)
            lora_mb = max(peak_mb - self._baseline_vram_mb, 0)
            _GLOBAL_PAYLOAD["hardware"]["lora_vram_mb"] = lora_mb

            # ETA / elapsed
            elapsed = 0
            eta = None
            if self._start_time:
                elapsed = round(time.time() - self._start_time, 1)
                if step > 0:
                    secs_per_step = elapsed / step
                    remaining = self.iters - step
                    eta = round(secs_per_step * remaining, 1)
            _GLOBAL_PAYLOAD["elapsed_seconds"] = elapsed
            _GLOBAL_PAYLOAD["eta_seconds"] = eta
            # Approximate epoch from step count
            _GLOBAL_PAYLOAD["meta"]["current_epoch"] = round(step / max(self.iters, 1), 2)

            train_entry = {
                "step":           step,
                "loss":           float(loss),
                "learning_rate":  float(lr),
                "it_per_sec":     float(its),
                "tokens_per_sec": float(tps),
                "trained_tokens": int(tokens),
            }
            # Merge into existing placeholder entry (e.g. val loss arrived first)
            logs = _GLOBAL_PAYLOAD["logs"]
            for entry in logs:
                if entry.get("step") == step and "loss" not in entry:
                    entry.update(train_entry)
                    _notify_subscribers()
                    return
            logs.append(train_entry)
            _notify_subscribers()
            return

        m = _VAL_RE.search(line)
        if m:
            step, val_loss = m.groups()
            step = int(step)
            logs = _GLOBAL_PAYLOAD["logs"]
            # Attach to any existing entry with the same step
            for entry in logs:
                if entry.get("step") == step:
                    entry["eval_loss"] = float(val_loss)
                    _notify_subscribers()
                    return
            # No matching entry yet — add placeholder; the train parser will merge
            logs.append({"step": step, "eval_loss": float(val_loss)})
            _notify_subscribers()
