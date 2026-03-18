"""
unsloth_sft_example.py

Reference template for Supervised Fine-Tuning (SFT) using Unsloth.
This script demonstrates the optimal configuration for fast, memory-efficient training.
"""

import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
import os

try:
    from gaslamp_callback import GaslampDashboardCallback
    callbacks = [GaslampDashboardCallback()]
except ImportError:
    callbacks = []

def main():
    # 1. Configuration
    max_seq_length = 2048 # Supports RoPE Scaling internally
    dtype = None # None for auto detection (Float16 for Tesla T4, Bfloat16 for Ampere+)
    load_in_4bit = True # Use 4bit quantization to reduce memory usage

    # 2. Load Model & Tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        # Always prefer unsloth's 4-bit optimized models!
        model_name = "unsloth/llama-3-8b-bnb-4bit",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    # 3. Patch Model with Unsloth's Optimized LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0, # MUST be 0
        bias = "none",    # MUST be "none"
        use_gradient_checkpointing = "unsloth", # CRITICAL: "unsloth" uses 30% less VRAM and fits 2x larger batch sizes!
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    # 4. Data Preparation formatting
    # Note: Replace this formatting logic with whatever matches the dataset
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
    
    EOS_TOKEN = tokenizer.eos_token
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs       = examples["input"]
        outputs      = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            # Must add EOS_TOKEN, otherwise generation goes on forever.
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return { "text" : texts, }
    
    dataset = load_dataset("yahma/alpaca-cleaned", split = "train")
    dataset = dataset.map(formatting_prompts_func, batched = True)

    # 5. Training setup
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Can make training 5x faster for short sequences.
        args = SFTConfig(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4, # Effective batch size = 2 * 4 = 8
            warmup_steps = 5,
            max_steps = 60, # Set num_train_epochs = 1 for full training runs
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to="none", # Change to "wandb" or "trackio" as needed
        ),
        callbacks = callbacks,
    )

    # 6. Execute Training
    print("Starting training...")
    trainer_stats = trainer.train()
    print(f"Training completed. Stats: {trainer_stats}")

    # 7. Save Model Adapters
    print("Saving LoRA adapters...")
    model.save_pretrained("lora_model")
    tokenizer.save_pretrained("lora_model")
    print("Adapters saved to 'lora_model' folder.")

    # Optional: GGUF Export
    # print("Exporting to GGUF Q4_K_M...")
    # model.save_pretrained_gguf("model_gguf", tokenizer, quantization_method = "q4_k_m")
    # print("GGUF saved.")

if __name__ == "__main__":
    main()
