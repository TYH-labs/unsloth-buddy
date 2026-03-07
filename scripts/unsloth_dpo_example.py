"""
Unsloth DPO (Direct Preference Optimization) Example
Fine-tune a language model using human preference data with Unsloth + TRL DPOTrainer.
Dataset requires: prompt, chosen, rejected columns.
"""
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig

# ============================================================
# 1. Load Model
# ============================================================
max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.1-8B-Instruct-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)

# ============================================================
# 2. Apply LoRA
# ============================================================
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,              # MUST be 0 for Unsloth
    bias="none",                 # MUST be "none" for Unsloth
    use_gradient_checkpointing="unsloth",  # Saves ~30% VRAM
    random_state=3407,
    max_seq_length=max_seq_length,
)

# ============================================================
# 3. Prepare Dataset
# ============================================================
# DPO datasets MUST have: prompt, chosen, rejected
# Example:
# {"prompt": "Explain quantum computing",
#  "chosen": "Quantum computing uses qubits...",
#  "rejected": "Computers are fast machines..."}

dataset = load_dataset("your-dpo-dataset", split="train")

# Validate format
assert "prompt" in dataset.column_names, "Dataset must have 'prompt' column"
assert "chosen" in dataset.column_names, "Dataset must have 'chosen' column"
assert "rejected" in dataset.column_names, "Dataset must have 'rejected' column"
print(f"Dataset loaded: {len(dataset)} samples")
print(f"Sample: {dataset[0]}")

# If your dataset needs reformatting:
# def format_dpo_sample(sample):
#     return {
#         "prompt": sample["question"],
#         "chosen": sample["good_answer"],
#         "rejected": sample["bad_answer"],
#     }
# dataset = dataset.map(format_dpo_sample)

# ============================================================
# 4. Train
# ============================================================
trainer = DPOTrainer(
    model=model,
    ref_model=None,               # Unsloth handles reference model internally
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=DPOConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_ratio=0.1,
        num_train_epochs=3,
        learning_rate=5e-6,
        logging_steps=1,
        optim="adamw_8bit",
        max_length=1024,
        max_prompt_length=512,
        output_dir="outputs-dpo",
        seed=3407,
    ),
)

trainer.train()

# ============================================================
# 5. Save & Export
# ============================================================
# Save LoRA adapters
model.save_pretrained("dpo-lora-model")
tokenizer.save_pretrained("dpo-lora-model")

# Export to GGUF for Ollama/LM Studio:
# model.save_pretrained_gguf("dpo-model", tokenizer, quantization_method="q4_k_m")

# Merge to 16-bit for vLLM:
# model.save_pretrained_merged("dpo-model", tokenizer, save_method="merged_16bit")

# Push to Hub:
# model.push_to_hub("your-username/dpo-model", token="hf_...")
# tokenizer.push_to_hub("your-username/dpo-model", token="hf_...")
