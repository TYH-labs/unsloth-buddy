"""
mlx_gaslamp_dashboard.py — Live training dashboard for mlx-tune on Apple Silicon.

mlx-tune's SFTTrainer has no callbacks API, so this module intercepts stdout to
parse training metrics and serve them via the same HTTP dashboard used on the
NVIDIA/TRL path.

Usage:
    from mlx_gaslamp_dashboard import MlxGaslampDashboard

    with MlxGaslampDashboard(iters=ITERS, hyperparams={"learning_rate": LR, ...}):
        trainer.train()

Dashboard → http://localhost:8080/
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
    "meta": {"max_steps": 0, "total_epochs": 0},
    "hardware": {"device": "Apple MPS", "peak_vram_mb": 0},
    "hyperparameters": {},
    "logs": [],
}

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
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            if os.path.exists(self.template_path):
                with open(self.template_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.wfile.write(b"<p>Error: templates/dashboard.html not found.</p>")
        elif self.path == "/api/metrics":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(_GLOBAL_PAYLOAD).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # suppress HTTP access logs


class MlxGaslampDashboard:
    """
    Context manager that wraps mlx-tune's trainer.train() to provide a live
    Gaslamp dashboard at http://localhost:{port}/.

    Args:
        iters       : Total training iterations (drives the progress bar).
        port        : HTTP port for the dashboard server (default 8080).
        hyperparams : Dict shown in the Hyperparameters panel, e.g.
                      {"learning_rate": LR, "batch_size": BATCH_SIZE, "lora_rank": R}
    """

    def __init__(self, iters: int, port: int = 8080, hyperparams: dict = None):
        self.iters = iters
        self.port = port
        self._original_stdout = None
        self._httpd = None

        global _GLOBAL_PAYLOAD
        _GLOBAL_PAYLOAD = {
            "meta": {"max_steps": iters, "total_epochs": 1},
            "hardware": {"device": "Apple MPS", "peak_vram_mb": 0},
            "hyperparameters": hyperparams or {},
            "logs": [],
        }

    def __enter__(self):
        self._start_server()
        self._original_stdout = sys.stdout
        sys.stdout = _TeeWriter(sys.stdout, self._parse_line)
        return self

    def __exit__(self, *_):
        sys.stdout = self._original_stdout

    def _start_server(self):
        try:
            self._httpd = HTTPServer(("localhost", self.port), _DashboardHandler)
            t = threading.Thread(target=self._httpd.serve_forever, daemon=True)
            t.start()
            time.sleep(0.3)
            print(f"\n{'='*60}")
            print(f"Gaslamp Live Training Dashboard: http://localhost:{self.port}/")
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
            _GLOBAL_PAYLOAD["hardware"]["peak_vram_mb"] = int(float(peak_gb) * 1024)
            train_entry = {
                "step":           int(step),
                "loss":           float(loss),
                "learning_rate":  float(lr),
                "it_per_sec":     float(its),
                "tokens_per_sec": float(tps),
                "trained_tokens": int(tokens),
            }
            # Merge into existing placeholder entry (e.g. val loss arrived first)
            logs = _GLOBAL_PAYLOAD["logs"]
            for entry in logs:
                if entry.get("step") == int(step) and "loss" not in entry:
                    entry.update(train_entry)
                    return
            logs.append(train_entry)
            return

        m = _VAL_RE.search(line)
        if m:
            step, val_loss = m.groups()
            logs = _GLOBAL_PAYLOAD["logs"]
            # Attach to any existing entry with the same step (train may arrive after val)
            for entry in logs:
                if entry.get("step") == int(step):
                    entry["eval_loss"] = float(val_loss)
                    return
            # No matching entry yet — add a placeholder; the train parser will
            # find and merge into it when that line arrives
            logs.append({"step": int(step), "eval_loss": float(val_loss)})
