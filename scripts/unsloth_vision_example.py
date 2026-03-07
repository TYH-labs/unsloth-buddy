"""
Unsloth Vision Fine-Tuning Example
Fine-tune a vision language model (VLM) using Unsloth + TRL SFTTrainer.
Supports: Qwen2.5-VL, Gemma 3, Llama 3.2 Vision, Pixtral, etc.
"""
from unsloth import FastVisionModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# ============================================================
# 1. Load Model
# ============================================================
max_seq_length = 2048

model, tokenizer = FastVisionModel.from_pretrained(
    model_name="unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
)

# ============================================================
# 2. Apply LoRA
# ============================================================
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,      # Fine-tune vision encoder
    finetune_language_layers=True,     # Fine-tune language decoder
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
    target_modules="all-linear",
    modules_to_save=["lm_head", "embed_tokens"],
)

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

# If your dataset needs reformatting:
# def format_vision_sample(sample):
#     return {
#         "messages": [
#             {"role": "user", "content": [
#                 {"type": "image", "image": sample["image_url"]},
#                 {"type": "text", "text": sample["question"]},
#             ]},
#             {"role": "assistant", "content": [
#                 {"type": "text", "text": sample["answer"]},
#             ]},
#         ]
#     }
# dataset = dataset.map(format_vision_sample)

# ============================================================
# 4. Train
# ============================================================
from unsloth import UnslothVisionDataCollator

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    train_dataset=dataset,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=30,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        max_seq_length=max_seq_length,
        output_dir="outputs-vision",
        seed=3407,
        remove_unused_columns=False,  # REQUIRED for vision
        dataset_num_proc=4,
    ),
)

trainer.train()

# ============================================================
# 5. Save & Export
# ============================================================
# Save LoRA adapters
model.save_pretrained("vision-lora-model")
tokenizer.save_pretrained("vision-lora-model")

# Export to GGUF for Ollama/LM Studio:
# model.save_pretrained_gguf("vision-model", tokenizer, quantization_method="q4_k_m")

# Merge to 16-bit for vLLM:
# model.save_pretrained_merged("vision-model", tokenizer, save_method="merged_16bit")

# Push to Hub:
# model.push_to_hub("your-username/vision-model", token="hf_...")
# tokenizer.push_to_hub("your-username/vision-model", token="hf_...")
