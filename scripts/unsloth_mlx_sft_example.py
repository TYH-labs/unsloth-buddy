"""
unsloth_mlx_sft_example.py — SFT fine-tuning on Apple Silicon via mlx-tune.

IMPORTANT: mlx-tune has its own SFTTrainer (NOT trl's). The model and tokenizer
are MLX-native objects and are incompatible with TRL's PyTorch-based trainer.

Key API differences from the NVIDIA/Unsloth path:
  - Import SFTTrainer from mlx_tune, not trl
  - SFTTrainer takes iters/adapter_path instead of SFTConfig
  - train_dataset is a list of dicts, not a HuggingFace Dataset object
  - tokenizer is passed as-is (mlx-tune's TokenizerWrapper, not unwrapped)
"""

import json
from mlx_tune import FastLanguageModel, SFTTrainer
from mlx_gaslamp_dashboard import MlxGaslampDashboard

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME     = "mlx-community/Llama-3.2-1B-Instruct-4bit"  # or any HF model
MAX_SEQ_LENGTH = 2048
LORA_RANK      = 16
LORA_ALPHA     = 16
ITERS          = 100
BATCH_SIZE     = 2
LEARNING_RATE  = 2e-4

# ── Load model ────────────────────────────────────────────────────────────────
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    load_in_4bit   = True,
)

# ── Apply LoRA ────────────────────────────────────────────────────────────────
model = FastLanguageModel.get_peft_model(
    model,
    r                          = LORA_RANK,
    target_modules             = ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"],
    lora_alpha                 = LORA_ALPHA,
    lora_dropout               = 0,       # must be 0 for mlx-tune optimization
    bias                       = "none",  # must be "none"
    use_gradient_checkpointing = "unsloth",
    random_state               = 3407,
)

# ── Dataset ───────────────────────────────────────────────────────────────────
# mlx-tune SFTTrainer expects a list of dicts with a "messages" key.
# Load from JSONL:
train_dataset = []
with open("data/train.jsonl") as f:
    for line in f:
        train_dataset.append(json.loads(line))

# Expected format:
# {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

# ── Train ─────────────────────────────────────────────────────────────────────
# NOTE: Use mlx_tune.SFTTrainer — NOT trl.SFTTrainer.
# Parameters are completely different from the TRL API.
trainer = SFTTrainer(
    model                       = model,
    tokenizer                   = tokenizer,   # pass TokenizerWrapper directly
    train_dataset               = train_dataset,
    max_seq_length              = MAX_SEQ_LENGTH,
    learning_rate               = LEARNING_RATE,
    per_device_train_batch_size = BATCH_SIZE,
    output_dir                  = "outputs",
    adapter_path                = "adapters",  # relative to output_dir → saves to outputs/adapters/
    iters                       = ITERS,
    # NOTE: mlx-tune SFTTrainer has no callbacks parameter — use MlxGaslampDashboard
    # context manager around trainer.train() instead (see below).
)

with MlxGaslampDashboard(iters=ITERS, hyperparams={
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "lora_rank": LORA_RANK,
    "lora_alpha": LORA_ALPHA,
}):
    trainer.train()

# ── Save ──────────────────────────────────────────────────────────────────────
# Adapters are already saved to {output_dir}/{adapter_path} during training.
# For inference, load with:
#   model, tokenizer = FastLanguageModel.from_pretrained(..., adapter_path="outputs/adapters")
print("Done. Adapters saved to outputs/adapters/")
