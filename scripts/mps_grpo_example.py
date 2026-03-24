"""
mps_grpo_example.py — GRPO fine-tuning on Apple Silicon via TRL + PyTorch MPS.

Use this template when training on an Apple Silicon Mac (M1/M2/M3/M4).
For NVIDIA GPUs, use unsloth_grpo_example.py instead.

Why not mlx-tune?
  mlx-tune (mlx_lm.lora) supports SFT only. GRPO with custom reward functions
  requires TRL's GRPOTrainer, which runs on PyTorch MPS — no Unsloth, no vLLM.

Key differences from the NVIDIA template (unsloth_grpo_example.py):
  - No Unsloth: use AutoModelForCausalLM + peft.LoraConfig instead of
    FastLanguageModel / PatchFastRL
  - No vLLM: remove fast_inference=True, use_vllm=True, gpu_memory_utilization
  - No bitsandbytes: load_in_4bit is CUDA-only; load in torch.float16 instead
  - Optimizer: paged_adamw_8bit is CUDA-only; use adamw_torch
  - Dashboard: GaslampDashboardCallback (TRL callback) not MlxGaslampDashboard
  - PYTORCH_ENABLE_MPS_FALLBACK=1 required for ops not yet on Metal

Hardware requirements:
  - Apple Silicon Mac with ≥16 GB unified memory (24 GB+ recommended for 3B+ models)
  - Python ≤ 3.12 (MPS support requirement)
  - PyTorch ≥ 2.1 with MPS backend enabled

Supported models (float16, no quantization):
  - 1B models: ~2 GB  — fits any M-series Mac
  - 3B models: ~6 GB  — fits any M-series Mac
  - 7B models: ~14 GB — requires 24 GB+ unified memory
"""

import os
import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from trl import GRPOConfig, GRPOTrainer
from gaslamp_callback import GaslampDashboardCallback

# Enable MPS fallback for ops not yet implemented on Metal
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Edit these for your project. All other code should not need changes.

MODEL_NAME      = "google/gemma-3-1b-it"   # any HF model; Gemma/Qwen/Llama work well
MAX_SEQ_LENGTH  = 1024                      # increase for longer reasoning traces
LORA_RANK       = 16                        # 8–32; higher = more expressive, more memory
LORA_ALPHA      = 16                        # keep equal to LORA_RANK (standard scaling)
MAX_STEPS       = 250                       # GRPO needs ≥150 steps before reward improves
LEARNING_RATE   = 5e-6                      # conservative; raise to 1e-5 if loss stalls
NUM_GENERATIONS = 4                         # completions per prompt; reduce to 2 if OOM
BATCH_SIZE      = 1                         # keep at 1 for MPS; accumulate gradients instead
GRAD_ACCUM      = 1                         # increase for smoother training (costs memory)
MAX_GRAD_NORM   = 0.1
OUTPUT_DIR      = "outputs"

# ── SYSTEM PROMPT & FORMAT ────────────────────────────────────────────────────
# Default: chain-of-thought math reasoning with XML tags.
# Replace SYSTEM_PROMPT and the reward functions below for other tasks.

SYSTEM_PROMPT = """Respond in the following format:

<think>
...
</think>

<answer>
...
</answer>"""

# ── DATA ──────────────────────────────────────────────────────────────────────
# Default: GSM8K grade-school math (openai/gsm8k).
# Replace get_dataset() for other tasks — GRPOTrainer requires a "prompt" column
# containing a list of chat messages (OpenAI format).

def extract_xml_answer(text: str) -> str:
    """Extract the content inside <answer>...</answer> tags."""
    if "<answer>" not in text or "</answer>" not in text:
        return ""
    return text.split("<answer>")[-1].split("</answer>")[0].strip()

def extract_hash_answer(text: str):
    """Extract the numeric answer after #### in GSM8K-style annotations."""
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def get_dataset(split: str = "train") -> Dataset:
    """Load and format GSM8K for GRPO. Returns a dataset with 'prompt' and 'answer' columns."""
    data = load_dataset("openai/gsm8k", "main")[split]
    data = data.map(lambda x: {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": x["question"]},
        ],
        "answer": extract_hash_answer(x["answer"]),
    })
    return data

# ── REWARD FUNCTIONS ──────────────────────────────────────────────────────────
# GRPO learns by comparing groups of completions. Define reward functions that
# return a float per completion. Combine multiple rewards for better signal.
#
# Signature: f(prompts, completions, **kwargs) -> list[float]
# - completions: list of list of dicts — each is a chat message list
# - prompts: list of prompt message lists (same length as completions)
# - kwargs: passes any extra dataset columns (e.g. "answer" if in the dataset)

def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    """2.0 if the extracted answer matches the ground truth, else 0.0."""
    responses = [c[0]["content"] for c in completions]
    extracted = [extract_xml_answer(r) for r in responses]
    return [2.0 if r == a else 0.0 for r, a in zip(extracted, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    """0.5 if the answer is a pure integer string."""
    responses = [c[0]["content"] for c in completions]
    extracted = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """0.5 if the completion exactly follows <think>\\n...\\n</think>\\n\\n<answer>\\n...\\n</answer>."""
    pattern   = r"^\n?<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>\n?$"
    responses = [c[0]["content"] for c in completions]
    return [0.5 if re.match(pattern, r, re.DOTALL) else 0.0 for r in responses]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """0.5 if both <think> and <answer> tag pairs appear anywhere in the completion."""
    pattern   = r"<think>.*?</think>\s*<answer>.*?</answer>"
    responses = [c[0]["content"] for c in completions]
    return [0.5 if re.search(pattern, r, re.DOTALL) else 0.0 for r in responses]

def count_xml(text: str) -> float:
    """Reward correct XML tag counts; penalise content after the closing answer tag."""
    score = 0.0
    if text.count("<think>")   == 1: score += 0.125
    if text.count("</think>")  == 1: score += 0.125
    if text.count("<answer>")  == 1:
        score += 0.125
        score -= len(text.split("</answer>")[-1]) * 0.001
    if text.count("</answer>") == 1:
        score += 0.125
        score -= (len(text.split("</answer>")[-1]) - 1) * 0.001
    return score

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    return [count_xml(c[0]["content"]) for c in completions]

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[train] Device      : {device}")
    print(f"[train] Model       : {MODEL_NAME}")
    print(f"[train] LoRA rank   : {LORA_RANK}  |  steps: {MAX_STEPS}  |  generations: {NUM_GENERATIONS}")

    # 1. Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load model in float16 on MPS
    # Note: bitsandbytes 4-bit quantization (load_in_4bit) is CUDA-only.
    # float16 on MPS gives a good quality/memory tradeoff.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        device_map=device,
    )

    # 3. Apply LoRA via PEFT (replaces Unsloth's FastLanguageModel.get_peft_model)
    peft_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,      # 0 is standard for fine-tuning at this scale
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Load dataset
    print("[train] Loading dataset...")
    dataset = get_dataset()

    # 5. Configure GRPO trainer
    # Note: use_vllm / paged_adamw_8bit are CUDA-only — do not set them here.
    warmup_steps = round(MAX_STEPS * 0.1)
    training_args = GRPOConfig(
        learning_rate=LEARNING_RATE,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_steps=warmup_steps,          # warmup_ratio deprecated in TRL ≥0.15
        lr_scheduler_type="cosine",
        optim="adamw_torch",            # paged_adamw_8bit / adamw_8bit are CUDA-only
        logging_steps=1,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_generations=NUM_GENERATIONS,
        generation_batch_size=NUM_GENERATIONS,  # must be divisible by num_generations (TRL ≥0.15)
        max_completion_length=MAX_SEQ_LENGTH,   # max_prompt_length removed — not in TRL ≥0.15
        max_steps=MAX_STEPS,
        save_steps=MAX_STEPS,           # save once at end
        max_grad_norm=MAX_GRAD_NORM,
        report_to="none",
        output_dir=OUTPUT_DIR,
        seed=3407,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ],
        args=training_args,
        train_dataset=dataset,
        callbacks=[GaslampDashboardCallback(task_type="grpo")],
    )

    # 6. Train
    print("[train] Starting GRPO training...")
    print("[train] Dashboard → http://localhost:8080/")
    print("[train] Rewards typically don't improve until step 150–200. Be patient.")
    trainer.train()

    # 7. Save LoRA adapters
    adapter_path = f"{OUTPUT_DIR}/adapters"
    print(f"\n[train] Saving LoRA adapters → {adapter_path}/")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print("[train] Done.")

if __name__ == "__main__":
    main()
