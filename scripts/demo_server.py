"""
demo_server.py — serves dashboard.html with rich mock data so you can see
every panel without running a real training job.

Usage:
    python scripts/demo_server.py [--task grpo|dpo|sft|vision] [--port 8080] [--hardware nvidia|mps]
"""
import json
import math
import argparse
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

# ── Hardware presets ─────────────────────────────────────────────────
# peak_vram_mb  = highest driver-allocated memory during training
# baseline_vram_mb = memory after model load, before first step
# lora_vram_mb  = peak - baseline (activations + optimizer + generations)
# total_vram_mb = device capacity (GPU VRAM or MPS recommended_max_memory)
_HARDWARE_PRESETS = {
    "nvidia": {
        # 7B model in QLoRA 4-bit on A100-40GB, batch=2×4, GRPO with 8 generations
        "device":           "NVIDIA A100-SXM4-40GB",
        "total_vram_mb":    40_960,
        "baseline_vram_mb":  7_200,   # 7B QLoRA 4-bit ~7 GB
        "peak_vram_mb":     28_450,   # activations + optimizer + 8 GRPO generations
        "lora_vram_mb":     21_250,   # peak - baseline
        "vram_pct":          69.5,
        "lora_vram_pct":     51.8,
        "cpu_ram_mb":       18_432,
        "tokens_per_sec_base": 390.0,
    },
    "mps": {
        # 1B model in float16 on Apple Silicon, batch=1×4, GRPO with 4 generations
        "device":           "Apple Silicon (MPS)",
        "total_vram_mb":    18_432,   # 18 GB unified memory (recommended_max_memory)
        "baseline_vram_mb":  2_048,   # 1B float16 ~2 GB
        "peak_vram_mb":      7_800,   # activations + AdamW states + 4 GRPO generations
        "lora_vram_mb":      5_752,   # peak - baseline
        "vram_pct":           42.3,
        "lora_vram_pct":      31.2,
        "cpu_ram_mb":         8_192,
        "tokens_per_sec_base": 7.0,   # MPS ~7 tok/s for 1B vs A100's 390
    },
}

# ── Mock data ───────────────────────────────────────────────────────
def make_payload(task_type="grpo", hardware="nvidia"):
    hw = _HARDWARE_PRESETS.get(hardware, _HARDWARE_PRESETS["nvidia"])
    tok_base = hw["tokens_per_sec_base"]

    steps = list(range(5, 305, 5))
    logs = []
    for i, s in enumerate(steps):
        t = i / len(steps)
        entry = {
            "step": s,
            "loss": max(0.05, 2.4 * math.exp(-2.5 * t) + 0.08 * math.sin(i * 0.7)),
            "learning_rate": 2e-4 * (1 - t * 0.85),
            "grad_norm": 0.8 + 0.4 * math.exp(-t) + 0.05 * math.sin(i * 1.3),
            "tokens_per_sec": tok_base + tok_base * 0.07 * math.sin(i * 0.4),
        }
        if i % 10 == 9:
            entry["eval_loss"] = entry["loss"] * 1.08 + 0.03 * math.sin(i)
        if task_type in ("grpo",):
            entry["reward"]     = -1.2 + 2.0 * t + 0.15 * math.sin(i * 0.6)
            entry["reward_std"] = max(0.05, 0.6 * (1 - t * 0.5))
            entry["kl_divergence"] = max(0.001, 0.3 * math.exp(-t * 2) + 0.01 * abs(math.sin(i)))
        if task_type in ("dpo",):
            entry["rewards_chosen"]   =  0.5 + 1.5 * t + 0.1 * math.sin(i * 0.5)
            entry["rewards_rejected"] = -0.5 - 0.8 * t + 0.1 * math.sin(i * 0.5 + 1)
            entry["kl_divergence"]    = max(0.001, 0.25 * math.exp(-t * 1.5))
        logs.append(entry)

    hw_payload = {k: v for k, v in hw.items() if k != "tokens_per_sec_base"}

    return {
        "meta": {
            "max_steps": 300,
            "total_epochs": 3,
            "task_type": task_type,
            "current_epoch": 2.87,
        },
        "hardware": hw_payload,
        "hyperparameters": {
            "learning_rate":          2e-4 if hardware == "nvidia" else 5e-6,
            "per_device_train_batch_size": 2 if hardware == "nvidia" else 1,
            "gradient_accumulation":  4,
            "optimizer":              "adamw_8bit" if hardware == "nvidia" else "adamw_torch",
            "seed":                   3407,
        },
        "logs": logs,
        "phase": "training",
        "elapsed_seconds": 1847,
        "eta_seconds":     312,
        "train_runtime_seconds": None,
    }


_PAYLOAD = None
_SSE_SUBSCRIBERS = []
_SSE_LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    template_path = "templates/dashboard.html"

    def do_GET(self):
        try:
            if self.path == "/":
                with open(self.template_path, "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/metrics":
                body = json.dumps(_PAYLOAD).encode()
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/api/stream":
                self.send_response(200)
                self.send_header("Content-type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                data = json.dumps(_PAYLOAD)
                self.wfile.write(f"event: progress\ndata: {data}\n\n".encode())
                self.wfile.flush()
                # keep alive
                import time
                while True:
                    time.sleep(10)
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
            elif self.path == "/api/health":
                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status":"healthy"}')
            else:
                self.send_response(404)
                self.end_headers()
        except Exception:
            pass

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",     default="grpo", choices=["sft","dpo","grpo","vision"])
    parser.add_argument("--hardware", default="nvidia", choices=["nvidia","mps"])
    parser.add_argument("--port",     type=int, default=8080)
    args = parser.parse_args()

    _PAYLOAD = make_payload(args.task, args.hardware)
    httpd = HTTPServer(("localhost", args.port), Handler)
    print(f"🚀  Demo dashboard at  http://localhost:{args.port}/  (task={args.task}, hardware={args.hardware})")
    print("    Press Ctrl+C to stop.")
    httpd.serve_forever()
