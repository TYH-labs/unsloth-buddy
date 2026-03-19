"""
colab_training.py — Code templates for training on Google Colab via colab-mcp.

Each constant / function returns a Python code string to pass to the
`execute_code` MCP tool.  The training cell runs the trainer in a background
thread so execute_code returns immediately; use POLL_CELL to track progress.

Typical call sequence:
    1. execute_code(SETUP_CELL)          # install unsloth, check GPU
    2. execute_code(VERIFY_CELL)         # smoke-test imports + VRAM
    3. execute_code(get_training_cell(...))   # start training in background
    4. loop: execute_code(POLL_CELL)     # monitor until done: true
    5. execute_code(FINAL_CELL)          # fetch final metrics + adapter path
"""

# ── Step 1: Install & GPU check ───────────────────────────────────────────────
SETUP_CELL = """
import subprocess, sys, json

print("Installing Unsloth...")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "unsloth", "-q"],
    check=True, capture_output=True
)

import torch
assert torch.cuda.is_available(), (
    "FAIL: No GPU — go to Runtime → Change runtime type → GPU"
)

gpu_name = torch.cuda.get_device_name(0)
vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9

print(json.dumps({
    "gpu":     gpu_name,
    "vram_gb": round(vram_gb, 1),
    "cuda":    torch.version.cuda,
}))
print("SETUP_OK")
"""

# ── Step 2: Verify imports & VRAM ─────────────────────────────────────────────
VERIFY_CELL = """
import json, torch
from unsloth import FastLanguageModel
import unsloth, trl, transformers, datasets as ds_lib

vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
assert vram_gb >= 6, f"FAIL: Only {vram_gb:.1f} GB VRAM — need at least 6 GB"

result = {
    "unsloth":       unsloth.__version__,
    "trl":           trl.__version__,
    "transformers":  transformers.__version__,
    "datasets":      ds_lib.__version__,
    "gpu":           torch.cuda.get_device_name(0),
    "vram_gb":       round(vram_gb, 1),
}
print(json.dumps(result))
print("VERIFY_OK")
"""

# ── Step 3: Training cell (run in background thread) ─────────────────────────
def get_training_cell(
    model_name:     str,
    hf_dataset_id:  str,
    dataset_split:  str  = "train",
    dataset_field:  str  = "text",       # "text" for pre-formatted, "messages" for chat
    lora_rank:      int  = 16,
    lora_alpha:     int  = 16,
    max_seq_length: int  = 2048,
    max_steps:      int  = 200,
    batch_size:     int  = 2,
    grad_accum:     int  = 4,
    learning_rate:  float = 2e-4,
    output_dir:     str  = "/content/outputs",
) -> str:
    """
    Returns a Python code string that:
      - Loads the model + LoRA
      - Loads the dataset from HuggingFace Hub
      - Defines _colab_metrics (list) and _colab_training_done (bool) globals
      - Attaches ColabMetricsCallback to the TRL SFTTrainer
      - Starts trainer.train() in a daemon background thread
      - Prints TRAINING_STARTED: <json meta> when the thread is launched
    """
    return f"""
import json, threading, torch
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

# ── Shared state (read by POLL_CELL) ─────────────────────────────────────────
_colab_metrics       = []
_colab_training_done = False
_colab_error         = None

# ── Metrics callback ──────────────────────────────────────────────────────────
class ColabMetricsCallback(TrainerCallback):
    def on_log(self, args: TrainingArguments, state: TrainerState,
               control: TrainerControl, logs=None, **kwargs):
        global _colab_metrics
        if logs and state.is_world_process_zero:
            entry = {{**{{k: v for k, v in logs.items() if isinstance(v, (int, float))}},
                      "step": state.global_step}}
            _colab_metrics.append(entry)

# ── Load model ────────────────────────────────────────────────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = {model_name!r},
    max_seq_length = {max_seq_length},
    load_in_4bit   = True,
)
model = FastLanguageModel.get_peft_model(
    model,
    r                          = {lora_rank},
    lora_alpha                 = {lora_alpha},
    target_modules             = ["q_proj","k_proj","v_proj","o_proj",
                                  "gate_proj","up_proj","down_proj"],
    lora_dropout               = 0,
    bias                       = "none",
    use_gradient_checkpointing = "unsloth",
    random_state               = 3407,
)

# ── Dataset ───────────────────────────────────────────────────────────────────
dataset = load_dataset({hf_dataset_id!r}, split={dataset_split!r})

# ── Trainer ───────────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model            = model,
    processing_class = tokenizer,
    train_dataset    = dataset,
    {"dataset_text_field = " + repr(dataset_field) + "," if dataset_field != "messages" else "# messages format — chat template applied automatically"}
    args = SFTConfig(
        per_device_train_batch_size = {batch_size},
        gradient_accumulation_steps = {grad_accum},
        max_steps                   = {max_steps},
        learning_rate               = {learning_rate},
        fp16                        = not torch.cuda.is_bf16_supported(),
        bf16                        = torch.cuda.is_bf16_supported(),
        logging_steps               = 1,
        optim                       = "adamw_8bit",
        output_dir                  = {output_dir!r},
        report_to                   = "none",
    ),
    callbacks = [ColabMetricsCallback()],
)

# ── Start training in background ──────────────────────────────────────────────
def _run():
    global _colab_training_done, _colab_error
    try:
        trainer.train()
    except Exception as e:
        _colab_error = str(e)
    finally:
        _colab_training_done = True

_thread = threading.Thread(target=_run, daemon=True)
_thread.start()

print("TRAINING_STARTED: " + json.dumps({{
    "model":       {model_name!r},
    "dataset":     {hf_dataset_id!r},
    "max_steps":   {max_steps},
    "output_dir":  {output_dir!r},
}}))
"""

# ── Step 4: Poll training progress ───────────────────────────────────────────
POLL_CELL = """
import json
snapshot = {
    "done":    _colab_training_done,
    "error":   _colab_error,
    "n_logs":  len(_colab_metrics),
    "recent":  _colab_metrics[-5:] if _colab_metrics else [],
}
if _colab_metrics:
    latest = _colab_metrics[-1]
    snapshot["latest_loss"] = latest.get("loss")
    snapshot["latest_step"] = latest.get("step")
print("POLL: " + json.dumps(snapshot))
"""

# ── Step 5: Final metrics + adapter location ──────────────────────────────────
FINAL_CELL = """
import json, os, glob

adapter_files = glob.glob("/content/outputs/**/*.safetensors", recursive=True)
summary = {
    "total_steps":    len(_colab_metrics),
    "final_loss":     _colab_metrics[-1].get("loss") if _colab_metrics else None,
    "all_metrics":    _colab_metrics,
    "adapter_files":  adapter_files,
    "error":          _colab_error,
}
print("FINAL: " + json.dumps(summary))
"""

# ── Colab MCP installation instructions (printed to user) ────────────────────
INSTALL_INSTRUCTIONS = """
To use Google Colab for training, install colab-mcp in Claude Code:

1. Install Python 3.13 (colab-mcp requires it; keeps your training venv intact):
     uv python install 3.13

2. Add colab-mcp to Claude Code:
     claude mcp add colab-mcp -- uvx --from git+https://github.com/googlecolab/colab-mcp --python 3.13 colab-mcp

3. Open ~/.claude.json, find the colab-mcp entry under your project's
   mcpServers, and make sure it looks like:
     "colab-mcp": {
       "command": "uvx",
       "args": ["--from", "git+https://github.com/googlecolab/colab-mcp",
                "--python", "3.13", "colab-mcp"],
       "timeout": 30000
     }

   Note: do NOT add --enable-runtime — proxy mode is correct and
   --enable-runtime requires a Google OAuth config not publicly available.

3. Restart Claude Code.

4. Open a new Colab notebook at https://colab.research.google.com
   and connect to a GPU runtime (Runtime → Change runtime type → T4 GPU).

5. Confirm the MCP tools are available — you should see:
   - execute_code
   - open_colab_browser_connection

   Note: "Failed to connect" before opening a Colab notebook is normal.
   The tools become active once a runtime is connected (Step 4).
"""
