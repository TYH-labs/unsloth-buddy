import os
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl
from typing import Dict, Any, Optional
import time

# Global reference holding the active payload so the HTTP handler can access it
_GLOBAL_PAYLOAD = {
    "meta": {"max_steps": 0, "total_epochs": 0},
    "hardware": {},
    "hyperparameters": {},
    "logs": []
}

class DashboardRequestHandler(BaseHTTPRequestHandler):
    """Serves the dashboard HTML and the live JSON metrics payload."""
    
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
                self.wfile.write(b"<h1>Gaslamp Dashboard</h1><p>Error: templates/dashboard.html not found locally.</p>")
                
        elif self.path == "/api/metrics":
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            # CORS just in case they open it from elsewhere
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            
            try:
                response = json.dumps(_GLOBAL_PAYLOAD).encode("utf-8")
                self.wfile.write(response)
            except Exception as e:
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()
            
    def log_message(self, format, *args):
        # Suppress standard HTTP server access logs so we don't spam the training terminal
        pass


class GaslampDashboardCallback(TrainerCallback):
    """
    A custom Hugging Face TrainerCallback that generates a real-time, 
    auto-refreshing HTML dashboard using the Gaslamp template.
    
    Spawns a local background web server to stream real-time training metrics 
    to a HTML dashboard securely on `localhost:8080`.
    """
    
    def __init__(self, port: int = 8080):
        self.port = port
        self.server_thread = None
        self.httpd = None
        self.is_running = False

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Starts the background HTTP daemon server."""
        if state.is_world_process_zero and not self.is_running:
            self._start_server()

    def _start_server(self):
        """Spawns the local web server on a background daemon thread."""
        try:
            self.httpd = HTTPServer(("localhost", self.port), DashboardRequestHandler)
            self.server_thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
            self.server_thread.start()
            self.is_running = True
            
            # Print a prominent, clickable log in the user's terminal
            time.sleep(1) # tiny delay to ensure it prints after initial spam
            print("\n" + "="*60)
            print(f"🚀 Gaslamp Live Training Dashboard: http://localhost:{self.port}/")
            print("="*60 + "\n")
        except OSError as e:
            if e.errno == 48: # Address already in use
                print(f"⚠️ [Gaslamp Callback] Port {self.port} is busy. The dashboard might already be running.")
            else:
                print(f"⚠️ [Gaslamp Callback] Failed to start local web server: {e}")
        except Exception as e:
            print(f"⚠️ [Gaslamp Callback] Failed to start local web server: {e}")

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Gracefully shuts down the background server if needed."""
        if state.is_world_process_zero and self.is_running and self.httpd:
            # We don't strictly *have* to shutdown since it's a daemon, but it's cleaner
            # (Though keeping it alive lets the user review the final charts until the script exits)
            pass

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Dict[str, float] = None, **kwargs):
        """Called whenever `trainer.log()` is invoked. Updates in-memory dictionary."""
        global _GLOBAL_PAYLOAD
        
        if state.is_world_process_zero:
            # 1. Gather hardware stats
            hw_info = {"device": "Unknown", "peak_vram_mb": 0}
            try:
                import torch
                if torch.cuda.is_available():
                    hw_info["device"] = torch.cuda.get_device_name(0)
                    hw_info["peak_vram_mb"] = int(torch.cuda.max_memory_allocated() / (1024*1024))
            except:
                pass

            # 2. Extract Hyperparameters
            hp_info = {}
            try:
                hp_info = {
                    "learning_rate": args.learning_rate,
                    "train_batch_size": args.train_batch_size,
                    "gradient_accumulation": args.gradient_accumulation_steps,
                    "optimizer": getattr(args, "optim", "Unknown"),
                    "seed": getattr(args, "seed", None),
                }
            except:
                pass

            # 3. Overwrite the global payload instantly (no disk I/O)
            _GLOBAL_PAYLOAD = {
                "meta": {
                    "max_steps": state.max_steps,
                    "total_epochs": getattr(args, "num_train_epochs", 0),
                },
                "hardware": hw_info,
                "hyperparameters": hp_info,
                "logs": state.log_history,
            }
