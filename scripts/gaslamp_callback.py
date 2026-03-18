import json
import os
import shutil
from typing import Dict, Any, Optional
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

class GaslampDashboardCallback(TrainerCallback):
    """
    A custom Hugging Face TrainerCallback that generates a real-time, 
    auto-refreshing HTML dashboard using the Gaslamp template.
    
    It injects the training log history dynamically bypassing CORS constraints.
    """
    
    def __init__(self, template_path: str = "dashboard.html", output_dir: str = "outputs"):
        """
        Args:
            template_path: Path to the Gaslamp dashboard.html template (usually copied to project root).
            output_dir: Where to save the rendered output (e.g. outputs/training_dashboard.html)
        """
        self.template_path = template_path
        self.output_dir = output_dir
        self.output_file = os.path.join(self.output_dir, "training_dashboard.html")
        
        # Ensure output dir exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Fallback template if the user didn't copy it
        self.fallback_template = """<!DOCTYPE html><html><head><meta http-equiv="refresh" content="5"></head><body>
        <h1>Gaslamp Training Logs (Fallback)</h1>
        <p>Could not find the beautiful template at {self.template_path}. Ensure it is copied to the project directory.</p>
        <script>window.TRAINING_DATA = __METRICS_DATA__; console.log(window.TRAINING_DATA);</script>
        </body></html>"""

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Dict[str, float] = None, **kwargs):
        """
        Called whenever `trainer.log()` is invoked (depends on logging_steps).
        """
        if state.is_world_process_zero:
            self._render_dashboard(args, state)

    def _render_dashboard(self, args: TrainingArguments, state: TrainerState):
        """Bundles system/run metadata with logs and renders the HTML."""
        if not os.path.exists(self.template_path):
            print(f"⚠️ [Gaslamp Callback] Template not found at {self.template_path}. Using plain fallback.")
            html_content = self.fallback_template
        else:
            with open(self.template_path, "r", encoding="utf-8") as f:
                html_content = f.read()

        # 1. Gather hardware stats
        hardware_info = {"device": "Unknown", "peak_vram_mb": 0}
        try:
            import torch
            if torch.cuda.is_available():
                hardware_info["device"] = torch.cuda.get_device_name(0)
                # peak memory allocated in MB
                hardware_info["peak_vram_mb"] = int(torch.cuda.max_memory_allocated() / (1024*1024))
        except Exception:
            pass

        # 2. Extract key Hyperparameters
        try:
            hyperparams = {
                "learning_rate": args.learning_rate,
                "train_batch_size": args.train_batch_size,
                "gradient_accumulation": args.gradient_accumulation_steps,
                "optimizer": args.optim,
                "seed": args.seed,
            }
        except:
            hyperparams = {}

        # 3. Create the payload
        payload = {
            "meta": {
                "max_steps": state.max_steps,
                "total_epochs": getattr(args, "num_train_epochs", 0),
            },
            "hardware": hardware_info,
            "hyperparameters": hyperparams,
            "logs": state.log_history,
        }

        # Convert to a clean JSON string
        try:
            json_data = json.dumps(payload, indent=2)
        except Exception as e:
            print(f"⚠️ [Gaslamp Callback] Failed to serialize logs: {e}")
            json_data = "[]"
            
        # Inject the data into the exact placeholder
        target = "__METRICS_DATA__"
        if target in html_content:
            rendered_html = html_content.replace(target, json_data)
        else:
            # Failsafe if the template was modified manually
            rendered_html = html_content + f"\n<script>window.TRAINING_DATA = {json_data};</script>"

        # Write atomically (write to temp file, then rename) to prevent the auto-refreshing browser from reading a half-written file
        temp_file = self.output_file + ".tmp"
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(rendered_html)
            
        os.replace(temp_file, self.output_file)
