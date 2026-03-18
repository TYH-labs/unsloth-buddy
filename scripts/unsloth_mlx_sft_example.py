import os
from mlx_tune import FastLanguageModel
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset
from transformers import TrainingArguments

# [MAC SPECIFIC]: This example uses mlx-tune instead of unsloth.
# It runs natively on Apple Silicon via MLX using the exact same API.

def train_sft():
    # 1. Load Model & Tokenizer
    # We use a 4bit quantized model for speed and low RAM footprint.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "mlx-community/Llama-3.2-1B-Instruct-4bit",
        max_seq_length = 2048,
        load_in_4bit = True,
    )

    # 2. Add LoRA Adapters
    # This keeps most weights frozen and only trains (~2%) new weights.
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    # 3. Prepare Dataset
    # Here we are just using Alpaca to show how Standard Format works
    dataset = load_dataset("yahma/alpaca-cleaned", split="train[:1000]")

    def format_prompts(examples):
        texts = []
        for instruction, input, output in zip(examples["instruction"], examples["input"], examples["output"]):
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}" + tokenizer.eos_token
            texts.append(prompt)
        return { "text" : texts }
        
    dataset = dataset.map(format_prompts, batched=True)

    # 4. Train Model
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = 2048,
        dataset_num_proc = 2,
        args = SFTConfig(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 60,
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported() if hasattr(torch, "cuda") else True,
            bf16 = torch.cuda.is_bf16_supported() if hasattr(torch, "cuda") else False,
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
        ),
    )
    
    print("Starting MLX training...")
    trainer_stats = trainer.train()

    # 5. Save the trained adapters
    print("Saving model adapters...")
    model.save_pretrained("lora_model") 
    tokenizer.save_pretrained("lora_model")

if __name__ == "__main__":
    train_sft()
