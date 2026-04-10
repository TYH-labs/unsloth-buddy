"""
MLX Vision Fine-Tuning Example for Apple Silicon
Fine-tune a vision language model (VLM) natively on Mac using mlx-tune.
Supports: Qwen2.5-VL, Gemma 3, Llama 3.2 Vision, Pixtral, etc.
"""
import os
from mlx_tune import (
    FastVisionModel,
    VLMSFTTrainer,
    VLMSFTConfig,
    UnslothVisionDataCollator,
)
from datasets import load_dataset
from mlx_gaslamp_dashboard import MlxGaslampDashboard

# ============================================================
# 1. Load Model
# ============================================================
max_seq_length = 2048

# Note: Always prefer mlx-community pre-quantized 4-bit models for Mac.
model, processor = FastVisionModel.from_pretrained(
    model_name="mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)

# ============================================================
# 2. Apply LoRA
# ============================================================
# mlx-tune handles LoRA setup natively
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,      # Fine-tune vision encoder
    finetune_language_layers=True,     # Fine-tune language decoder
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# Required for mlx-tune VLM — switches model from inference → training mode
FastVisionModel.for_training(model)

# ============================================================
# 3. Prepare Dataset
# ============================================================
# Vision datasets should have a "messages" column with image content.
# Example format:
# {"messages": [
#     {"role": "user", "content": [
#         {"type": "image", "image": "https://..."},
#         {"type": "text", "text": "Describe this image."}
#     ]},
#     {"role": "assistant", "content": [
#         {"type": "text", "text": "The image shows..."}
#     ]}
# ]}

dataset = load_dataset("your-dataset-here", split="train")

# If your dataset needs reformatting, ensure you map it to the 'messages' format 
# and optionally construct an HTRVisionCollator if images remain outside messages dicts.

# ============================================================
# 4. Train
# ============================================================
args = VLMSFTConfig(
    per_device_train_batch_size=1,  # VLM forces batch_size=1 due to variable image sizes
    gradient_accumulation_steps=8,
    warmup_steps=20,
    max_steps=300,
    learning_rate=5e-5,
    logging_steps=10,
    optim="adam",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    max_length=max_seq_length,
    output_dir="outputs-vision",
    seed=3407,
    remove_unused_columns=False,  # REQUIRED for vision
    train_on_completions=True,    # computes loss cleanly just on assistant outputs
)

trainer = VLMSFTTrainer(
    model=model,
    tokenizer=processor,
    data_collator=UnslothVisionDataCollator(model, processor),
    train_dataset=dataset,
    args=args,
)

# ============================================================
# 5. Train with Gaslamp Dashboard
# ============================================================
# Web dashboard available at http://localhost:8080/
with MlxGaslampDashboard(
    iters=args.max_steps,
    task_type="vision",
    hyperparams={
        "model":         "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
        "learning_rate": args.learning_rate,
        "lora_rank":     16,
        "lora_alpha":    16,
        "max_steps":     args.max_steps,
        "grad_accum":    args.gradient_accumulation_steps,
        "max_length":    args.max_length,
    },
):
    trainer.train()

# ============================================================
# 6. Save & Export
# ============================================================
# Save LoRA adapters (Note: use model.save_pretrained in mlx-tune, not trainer.save_model)
os.makedirs("outputs-vision/adapters", exist_ok=True)
model.save_pretrained("outputs-vision/adapters")

print("\\nAdapters saved to outputs-vision/adapters/")

# Export to GGUF format:
# from mlx_vlm.utils import save_pretrained_gguf
# save_pretrained_gguf("vision-model-gguf", model, processor, quantization_method="q4_k_m")
