"""
unsloth_grpo_example.py

Reference template for Reinforcement Learning (GRPO) using Unsloth.
GRPO allows reasoning/math formulation training with surprisingly low VRAM requirements.
"""

import re
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel, PatchFastRL
import trl

# Important: Patch FastRL MUST be called before importing GRPOTrainer
PatchFastRL("GRPO", FastLanguageModel)
from trl import GRPOTrainer, GRPOConfig

def extract_xml_answer(text: str) -> str:
    ans = text.split("<answer>")[-1]
    if "</answer>" in ans:
        ans = ans.split("</answer>")[0]
    return ans.strip()

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# ----- Reward Functions -----
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    # Simple strict exact match approach
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if completion has basic XML structure."""
    responses = [completion[0]["content"] for completion in completions]
    return [0.5 if "<reasoning>" in r and "</reasoning>" in r and "<answer>" in r and "</answer>" in r else 0.0 for r in responses]

def main():
    max_seq_length = 1024
    load_in_4bit = True

    # 1. Load Model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen2.5-1.5B-Instruct", # Example small reasoning model
        max_seq_length = max_seq_length,
        load_in_4bit = load_in_4bit,
        fast_inference = True, # Enable vLLM optimized inference for generation
        max_lora_rank = 16,
        gpu_memory_utilization = 0.6, # Reduce VLLM usage
    )

    # 2. Add LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        use_gradient_checkpointing = "unsloth", # CRITICAL for memory
        random_state = 3407,
    )

    # 3. Load & Prepare Dataset
    # GRPO requires a 'prompt' and 'answer' column generally.
    system_prompt = "You are a helpful reasoning assistant. Provide your step-by-step thinking in <reasoning> tags, and your final answer in <answer> tags."
    
    def prep_dataset(example):
        answer = extract_hash_answer(example["answer"])
        return {
            "prompt": [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': example['question']}
            ],
            "answer": answer
        }

    dataset = load_dataset("openai/gsm8k", "main", split="train[:500]") # Subset for demo
    dataset = dataset.map(prep_dataset, remove_columns=dataset.column_names)

    # 4. Configure GRPO
    # Note: GRPO generates num_generations answers for the same prompt, then scores them.
    training_args = GRPOConfig(
        use_vllm = True, # Use vLLM for the rollout generation phase! Massive speedup.
        learning_rate = 5e-6,
        adam_beta1 = 0.9,
        adam_beta2 = 0.99,
        weight_decay = 0.1,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        optim = "adamw_8bit",
        logging_steps = 1,
        bf16 = torch.cuda.is_bf16_supported(),
        fp16 = not torch.cuda.is_bf16_supported(),
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        num_generations = 4, # Generate 4 rollouts per prompt
        max_prompt_length = 256,
        max_completion_length = 400,
        max_steps = 50, # Demo
        save_steps = 50,
        max_grad_norm = 0.1,
        report_to = "none",
        output_dir = "outputs",
    )

    # 5. Execute Training
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            correctness_reward_func,
            int_reward_func, 
            strict_format_reward_func, 
            soft_format_reward_func
        ],
        args = training_args,
        train_dataset = dataset,
    )
    
    print("Starting GRPO training...")
    trainer.train()

    # 6. Save
    print("Saving GRPO LoRA weights...")
    model.save_pretrained("grpo_lora_model")
    tokenizer.save_pretrained("grpo_lora_model")
    print("Done!")

if __name__ == "__main__":
    main()
