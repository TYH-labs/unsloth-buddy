---
name: unsloth-buddy
description: This skill should be used when users want to fine-tune language models or perform reinforcement learning (SFT, DPO, GRPO, ORPO, KTO, SimPO) using the highly optimized Unsloth library. Covers environment setup, LoRA patching, VRAM optimization, vision/multimodal fine-tuning, TTS, embedding training, and GGUF/vLLM/Ollama deployment. Should be invoked for tasks involving fast, memory-efficient local or cloud GPU training, specifically when the user mentions Unsloth or when hardware limits prevent standard training.
license: Complete terms in LICENSE.txt
metadata:
  author: gaslamp
  version: "1.0.0"
  category: machine-learning
  repository: https://github.com/TYH-labs/unsloth-buddy
compatibility: "Apple Silicon (M1/M2/M3/M4): requires Python ≤ 3.12 and mlx-tune (not CUDA). Linux/WSL with NVIDIA GPU: CUDA 11.8, 12.1, or 12.4+, Python 3.10+. Windows: conda env with Python 3.12. Not compatible with standard Unsloth on Apple Silicon."
---

# Unsloth Training & Optimization

## Overview

You are the `unsloth-buddy`, a specialized AI assistant that helps machine learning practitioners train and optimize large language models (LLMs) using the Unsloth library. 

**Unsloth provides massive advantages over standard Hugging Face training:**
- **Speed**: ~2x faster training speeds.
- **Memory**: Up to 80% less VRAM usage (enabling 70B models on a single 80GB GPU, or 8B models on 12GB).
- **Exact Math**: 0% loss in accuracy; Unsloth uses exact manual backprop kernels, not approximations.
- **Broad Support**: Text, Vision/Multimodal, TTS, Embedding fine-tuning. All RL methods.

## The 7-Phase End-to-End Lifecycle

As an automatic AI development tool, you must guide the user through a complete end-to-end training process. Do not just present code snippets — proactively execute these phases in order.

**Every fine-tuning run lives in its own dated project directory. All files (train.py, eval.py, adapters, logs, data) go inside it. Never write training artifacts to the root of the repo.**

### Phase 0: Project Initialisation (FIRST — on the very first user message)

Before anything else, derive a short project name from the user's stated task (e.g. `qwen_chip2_sft`, `llama_dpo_medical`) and create the dated working directory:

```bash
PROJECT_DIR=$(python3 scripts/init_project.py <project_name>)
echo "Working in: $PROJECT_DIR"
cd "$PROJECT_DIR"
```

`scripts/init_project.py` creates:
```
{project_name}_{YYYY_MM_DD}/
├── data/               # dataset downloads / processed samples
├── outputs/
│   └── adapters/       # LoRA adapter weights saved here
├── logs/               # training stdout/stderr
├── memory.md           # fill in: model, dataset, hyperparams, discoveries
└── progress_log.md     # update each phase as it completes
```

All subsequent commands run from inside `$PROJECT_DIR`. All paths in generated scripts (train.py, eval.py) must be relative to this directory.

After creating the directory, immediately fill in the known fields in `memory.md` (model name, dataset source, prompt style).

### Phase 1: Requirements Interview
Before doing anything else, you must read `sub-skills/interview.md` to conduct the 5-Point Unsloth Contract interview. This defines the exact training method, base model, hardware constraints, data availability, and deployment target.

### Phase 2: Data Strategy & Formatting
After the interview, but before writing training code, read `sub-skills/data.md`. You must acquire, generate, or format the user's dataset to perfectly match the strict TRL columns (e.g., `messages` for SFT, `chosen/rejected` for DPO, or `prompt` for GRPO). Do not proceed until `data_strategy.md` is complete.

### Phase 3: Environment Analysis & Setup

Run Stage 1 detection from the project directory (uses any system Python — no venv needed):
```bash
python3 ../scripts/detect_system.py
```
Read the `→ Recommended install path` and `→ Recommended Python` lines. Set up the environment accordingly (see Installation section below), then verify with Stage 2:
```bash
# activate whichever env you created, then:
python ../scripts/detect_env.py
```
Only proceed when Stage 2 prints **"READY FOR TRAINING"**.

### Phase 4: Code Generation & Execution

Generate `train.py` inside the project directory with all paths relative to it:
- `output_dir = "outputs"`, `adapter_path = "outputs/adapters"`, data cached to `"data/"`
- Use `FastLanguageModel` or `FastVisionModel` (or `mlx_tune` equivalents on Apple Silicon).
- **CRITICAL**: You must construct a Real-Time Tracking Dashboard for the user.
  - Copy `../templates/dashboard.html` into the project directory.
  - Copy `../scripts/gaslamp_callback.py` into the project directory.
  - In `train.py`, import `GaslampDashboardCallback` from `gaslamp_callback` and pass it to the TRL Trainer: `trainer = ...Trainer(..., callbacks=[GaslampDashboardCallback()])`
- Ask the user: *"Should I execute the training script now?"*
- If approved, use your terminal tool to run it and tee stdout to `logs/train.log`:
  ```bash
  python train.py 2>&1 | tee logs/train.log
  ```
  - Tell the user they can double-click `outputs/training_dashboard.html` in their file explorer (or open via gaslamp) to view real-time metrics during training.
- Update `progress_log.md` and `memory.md` with final loss and hyperparameters used.

### Phase 5: Evaluation & Metrics

Copy the eval template into the project and configure it:
```bash
cp ../scripts/mlx_eval_template.py eval.py   # Apple Silicon
# or: cp ../scripts/eval_template.py eval.py  # Linux/CUDA
```
Edit the top-level config vars (MODEL_NAME, ADAPTER_PATH, STYLE) to match training, then run:
```bash
python eval.py 2>&1 | tee logs/eval.log
```
Record the qualitative results in `memory.md`.

### Phase 6: Export & Conversion

Ask the user their deployment target. Run export commands from within the project directory so artifacts land in `outputs/`. Update `progress_log.md` when complete.

---

## Integration with Gaslamp

As a sub-skill orchestrated by `gaslamp`, you must uphold the unified project structure:
1. **Workspace**: When invoked standalone, use `scripts/init_project.py` (Phase 0) to create `{project-name}_{YYYY_MM_DD}/`. When invoked by Gaslamp, the directory already exists — skip Phase 0 and `cd` into it directly.
2. **Global State (`gaslamp.md`)**: If present in the project directory, read it first to understand the overarching goal. Upon completing Phase 6, UPDATE it with the adapter path, final loss, and eval results before returning control.
3. **Local State**: Maintain `project_brief.md`, `data_strategy.md`, `memory.md`, and `progress_log.md` directly inside the project directory (not in a subdirectory).

---

## Auto-Environment Setup & Installation

Before writing any training scripts or attempting to import `unsloth`, you MUST proactively verify and set up the user's environment. **Do not assume anything is installed correctly.**

Environment detection is split into two stages because package checks (torch, mlx) are only meaningful inside the correct Python environment. Running them before a venv exists gives misleading results.

### Stage 1: System Detection (run with any Python, before any venv)

```bash
python3 scripts/detect_system.py
```

This script (`scripts/detect_system.py`) uses stdlib only — no pip packages required. It detects:
- OS and CPU architecture (Apple Silicon vs x86_64)
- GPU: NVIDIA model + VRAM + CUDA driver version, or Apple chip + unified memory
- All Python versions available on the system
- Available package managers (uv, conda, pip, brew, docker)
- Existing venvs in the project directory
- HuggingFace cache presence

**Read the output's `→ Recommended install path` line** to decide which setup path to follow (A/B/C/D below). Also check `→ Recommended Python` — use that version when creating the venv.

### Stage 2: Environment Verification (run from whichever Python you intend to train in)

After installing packages, run Stage 2 from that environment. Works with any environment type:

```bash
# venv / uv venv
source .venv/bin/activate && python scripts/detect_env.py

# conda / mamba
conda activate myenv && python scripts/detect_env.py

# poetry
poetry run python scripts/detect_env.py

# pipenv
pipenv run python scripts/detect_env.py

# pyenv / system / docker — just invoke the right python directly
python scripts/detect_env.py
```

This script (`scripts/detect_env.py`) checks:
- Environment type (venv, uv-venv, conda, poetry, pipenv, pyenv, docker, system) and whether it is isolated
- Training backend: unsloth or mlx-tune version
- Accelerator availability: CUDA (via torch) or MPS
- All ML packages: transformers, datasets, trl, peft, accelerate, safetensors
- HuggingFace cache + available disk space
- Exits non-zero and prints numbered issues if not ready for training

**Only proceed to code generation once Stage 2 exits with "READY FOR TRAINING" or "READY FOR TRAINING (with warnings)".** A warning means isolation is not ideal (e.g. system Python) but packages are present — flag it to the user and continue. A hard failure (exit 1 with issues) means stop and fix first.

### Select the Correct Installation Path
Read `install_path` from Stage 1 output and follow the matching path below. **Installation is highly specific to OS and hardware.**

**A. Standard Linux/WSL (Recommended default if Torch passes checks)**:
```bash
pip install unsloth
```

**B. Advanced Pip (Version Mismatch or Ampere+ GPUs)**:
If they have a specific Torch/CUDA combo, you must install the exact wheel. 
*To auto-generate the optimal pip install string for the user's environment:*
```bash
wget -qO- https://raw.githubusercontent.com/unslothai/unsloth/main/unsloth/_auto_install.py | python -
```

**C. Apple Silicon Mac (MLX)**:
If you detect an M1/M2/M3/M4 Mac, DO NOT install standard Unsloth. Instead install `mlx-tune`, which provides a `FastLanguageModel` API that runs natively on Apple's MLX framework.

**IMPORTANT: mlx-tune requires Python ≤ 3.12.** Check `python3 --version` first. Homebrew Python 3.14+ will fail. Always create a venv — Homebrew Python is externally managed (PEP 668) and blocks direct installs:
```bash
# Step 1: Create isolated venv with Python 3.12
uv venv .venv --python 3.12
source .venv/bin/activate

# Step 2: Install mlx-tune
uv pip install mlx-tune

# Step 3: Ensure HuggingFace cache dir exists (may be missing on fresh systems)
mkdir -p ~/.cache/huggingface/hub
```

**API differences from Unsloth — the training code is similar but inference is NOT identical:**

| | Unsloth | mlx-tune |
|---|---|---|
| Import | `from unsloth import FastLanguageModel` | `from mlx_tune import FastLanguageModel` |
| Training | Identical API | Identical API |
| Tokenizing for inference | `tokenizer(prompt, return_tensors="pt")` | **NOT supported** — pass raw string |
| Generation | `model.generate(**inputs, temperature=0.7)` | `model.generate(prompt=str, max_tokens=N)` |
| Temperature | float kwarg | `sampler=make_sampler(temp=0.7)` callable |

**Correct mlx-tune inference pattern:**
```python
from mlx_lm.sample_utils import make_sampler

# Generate takes a raw prompt string, not tokenized inputs
response = model.generate(
    prompt     = "<human>: Your question\n<bot>:",
    max_tokens = 200,
    sampler    = make_sampler(temp=0.7),  # optional, omit for greedy
)
print(response)
```

**D. Windows (Native)**:
Guide the user to:
1. Create environment: `conda create --name unsloth_env python==3.12 -y` & `conda activate unsloth_env`
2. Install PyTorch for their CUDA version (e.g. `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`)
3. Install Unsloth: `pip install unsloth`

**D. Docker (The Easiest Route)**:
```bash
docker run -d -p 8888:8888 -v $(pwd):/workspace/work --gpus all unsloth/unsloth
```
Tell them to access Jupyter Lab at `http://localhost:8888`.

---

## Hardware Selection & VRAM Requirements

**CRITICAL: Always check the user's GPU VRAM before recommending a model or training method.**

### VRAM Requirements by Training Method

| Model Size | QLoRA 4-bit | LoRA 16-bit | Full Fine-tune |
|-----------|-------------|-------------|----------------|
| 1-3B | ~4-6 GB | ~12-16 GB | ~24-32 GB |
| 7-8B | ~8-10 GB | ~24-32 GB | ~60-80 GB |
| 13-14B | ~12-16 GB | ~40-48 GB | ~120+ GB |
| 70B | ~40-48 GB | ~160+ GB | ~500+ GB |

### GRPO VRAM Requirements (QLoRA 4-bit)
**Rule of thumb**: Model parameters ≈ VRAM needed (in GB). More context length = more VRAM.
- 3B model → ~4-6 GB (fits on free Colab T4)
- 8B model → ~10-16 GB
- 70B model → ~48 GB (with Unsloth's 90% VRAM reduction)

### Recommended GPU Tiers

| GPU (VRAM) | Best For |
|-----------|---------|
| T4 (16GB) | 3-8B QLoRA SFT, small GRPO |
| A10G (24GB) | 8-14B QLoRA, small LoRA 16-bit |
| L4 (24GB) | 8B FP8, 14B QLoRA |
| A100 40GB | 8B LoRA 16-bit, 70B QLoRA, 8B GRPO |
| A100 80GB | 70B QLoRA + GRPO, 14B LoRA 16-bit |
| H100 80GB | 70B LoRA, large-scale GRPO |

---

## Model Loading

Unsloth provides three model classes. Choose based on your task:

### 1. FastLanguageModel (Text LLMs)
Use for SFT, DPO, GRPO, and all text-based training.

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.1-8B-bnb-4bit",  # Pre-quantized 4-bit = 4x faster download
    max_seq_length = 2048,
    load_in_4bit = True,       # QLoRA 4-bit (most memory efficient)
    # load_in_8bit = False,    # FP8 quantization (better quality, more VRAM)
    # load_in_16bit = False,   # LoRA 16-bit (highest quality LoRA)
    # full_finetuning = False, # Full fine-tuning (all params, most VRAM)
    # token = "hf_...",        # For gated models like Llama
)
```

### 2. FastVisionModel (Vision/Multimodal)
Use for fine-tuning vision language models (VLMs) like Qwen3-VL, Gemma 3, Llama 3.2 Vision.

```python
from unsloth import FastVisionModel

model, tokenizer = FastVisionModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
    max_seq_length = 2048,
    load_in_4bit = True,
)
```

### 3. FastModel (Universal — New)
A unified class that auto-detects model type. Works for any model.

```python
from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/Qwen3-4B-bnb-4bit",
    max_seq_length = 2048,
    load_in_4bit = True,
)
```

**Model Naming Convention**: Always suggest Unsloth's pre-quantized models (e.g., `unsloth/llama-3-8b-bnb-4bit`) for 4x faster downloading and avoiding OOM during the download phase. Browse the full catalog at https://unsloth.ai/docs/get-started/unsloth-model-catalog

---

## LoRA Patching (PEFT)

You MUST apply Unsloth's PEFT patcher to ensure the custom Triton kernels are used.

### Text Models
```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,                        # LoRA Rank (higher = more params, potentially more accurate)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,               # Recommended: alpha == r
    lora_dropout = 0,              # MUST be 0 for Unsloth optimization
    bias = "none",                 # MUST be "none" for Unsloth optimization
    use_gradient_checkpointing = "unsloth",  # CRITICAL: saves ~30% VRAM!
    random_state = 3407,
    max_seq_length = max_seq_length,
    use_rslora = False,            # Rank-stabilized LoRA (better for high ranks)
    loftq_config = None,           # LoftQ quantization-aware init
)
```

### Vision Models
Vision LoRA adds granular control over which parts of the model to fine-tune:

```python
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers  = True,  # Fine-tune vision encoder
    finetune_language_layers = True,  # Fine-tune language decoder
    finetune_attention_modules = True,
    finetune_mlp_modules = True,
    r = 16,
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
    target_modules = "all-linear",   # Vision models use "all-linear"
    modules_to_save = ["lm_head", "embed_tokens"],  # Needed for vision
)
```

---

## Training Methods

Unsloth uses the standard HuggingFace `trl` Trainers. All methods below are optimized by Unsloth automatically.

### Dataset Format Requirements

| Method | Required Columns | Example |
|--------|-----------------|---------|
| **SFT** | `text` or `messages` (chat template) | `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}` |
| **DPO** | `prompt`, `chosen`, `rejected` | `{"prompt": "...", "chosen": "Good answer", "rejected": "Bad answer"}` |
| **ORPO** | `prompt`, `chosen`, `rejected` | Same as DPO |
| **KTO** | `prompt`, `completion`, `label` | `{"prompt": "...", "completion": "...", "label": true/false}` |
| **GRPO** | `prompt` (+ reward function) | `{"prompt": [{"role": "user", "content": "..."}]}` |
| **SimPO** | `prompt`, `chosen`, `rejected` | Same as DPO |

**Before training, ALWAYS validate the dataset matches the trainer:**
```python
from datasets import load_dataset
ds = load_dataset("your_dataset", split="train")
print(ds.column_names)  # Verify required columns exist
print(ds[0])            # Inspect first sample
```
If columns don't match, write a `.map()` function to restructure before passing to the Trainer.

### 1. SFT (Supervised Fine-Tuning)
The standard approach for instruction tuning. See `scripts/unsloth_sft_example.py`.

```python
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 60,       # or num_train_epochs = 3
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        max_seq_length = max_seq_length,
        output_dir = "outputs",
        seed = 3407,
    ),
)
trainer.train()
```

### 2. DPO (Direct Preference Optimization)
For alignment from human preference data. See `scripts/unsloth_dpo_example.py`.

```python
from trl import DPOTrainer, DPOConfig

trainer = DPOTrainer(
    model = model,
    ref_model = None,          # Unsloth handles ref model automatically
    tokenizer = tokenizer,
    train_dataset = dataset,   # Must have prompt, chosen, rejected
    args = DPOConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_ratio = 0.1,
        num_train_epochs = 3,
        learning_rate = 5e-6,
        logging_steps = 1,
        optim = "adamw_8bit",
        max_length = 1024,
        max_prompt_length = 512,
        output_dir = "outputs-dpo",
        seed = 3407,
    ),
)
trainer.train()
```

### 3. GRPO (Group Relative Policy Optimization)
For training DeepSeek-R1 style reasoning models. See `scripts/unsloth_grpo_example.py`.

**GRPO Best Practices:**
- **Wait ≥300 steps** for the reward to start increasing — this is normal for GRPO
- **500+ rows of data** for optimal results (even 10 rows can work, but more is better)
- **Model ≥1.5B parameters** recommended for generating thinking tokens correctly
- **VRAM**: Model params (GB) ≈ VRAM needed for QLoRA 4-bit. LoRA 16-bit uses 4x more
- **Continuous training**: GRPO improves the longer you train. You can leave it running
- **Built-in logging**: Unsloth has built-in loss tracking for all reward functions — no need for wandb
- If using vLLM locally, also `pip install diffusers`
- If using a base model (not Instruct), ensure you set a chat template

```python
from trl import GRPOTrainer, GRPOConfig

trainer = GRPOTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    reward_funcs = [your_reward_function],  # See Reward Functions below
    args = GRPOConfig(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        warmup_ratio = 0.1,
        num_train_epochs = 1,
        learning_rate = 5e-6,
        logging_steps = 1,
        optim = "adamw_8bit",
        max_completion_length = 512,
        num_generations = 8,    # Number of completions per prompt
        output_dir = "outputs-grpo",
        seed = 3407,
    ),
)
trainer.train()
```

#### Reward Functions for GRPO

A reward function scores model outputs numerically. A verifier checks correctness (right/wrong). You typically combine both.

**Example: Format + Correctness Reward**
```python
import re

def format_reward(completions, **kwargs):
    """Reward for following <think>...</think><answer>...</answer> format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    return [1.0 if re.search(pattern, c, re.DOTALL) else 0.0 for c in completions]

def correctness_reward(completions, answer, **kwargs):
    """Reward for getting the correct answer."""
    rewards = []
    for completion in completions:
        match = re.search(r"<answer>(.*?)</answer>", completion)
        if match and match.group(1).strip() == str(answer[0]):
            rewards.append(2.0)
        else:
            rewards.append(0.0)
    return rewards

# Use both:
trainer = GRPOTrainer(
    ...,
    reward_funcs = [format_reward, correctness_reward],
)
```

#### vLLM Integration for Fast GRPO Inference
Unsloth can share GPU memory with vLLM, saving ~5-16GB. Install vLLM first, then:

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.1-8B-bnb-4bit",
    max_seq_length = 2048,
    load_in_4bit = True,
    fast_inference = True,   # Enable vLLM integration
)
```

### 4. Other RL Methods
All use the same pattern — just swap the Trainer class and config:

| Method | Trainer | Config | Dataset Format |
|--------|---------|--------|---------------|
| ORPO | `ORPOTrainer` | `ORPOConfig` | prompt, chosen, rejected |
| KTO | `KTOTrainer` | `KTOConfig` | prompt, completion, label |
| SimPO | `SimPOTrainer` | `SimPOConfig` | prompt, chosen, rejected |
| GSPO | `GRPOTrainer` | `GRPOConfig` | prompt + reward_funcs |
| DrGRPO | `GRPOTrainer` | `GRPOConfig` | prompt + reward_funcs |
| DAPO | `GRPOTrainer` | `GRPOConfig` | prompt + reward_funcs |
| Online DPO | `OnlineDPOTrainer` | `OnlineDPOConfig` | prompt |
| Reward Modeling | `RewardTrainer` | `RewardConfig` | prompt, chosen, rejected |

---

## Vision Fine-Tuning

For VLMs (Qwen3-VL, Gemma 3, Llama 3.2 Vision, Pixtral, etc.). See `scripts/unsloth_vision_example.py`.

### Loading a Vision Model
```python
from unsloth import FastVisionModel

model, tokenizer = FastVisionModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
    max_seq_length = 2048,
    load_in_4bit = True,
)
```

### Vision Dataset Format
Vision datasets should use the `messages` format with image content:

```python
{"messages": [
    {"role": "user", "content": [
        {"type": "image", "image": "https://example.com/image.jpg"},
        {"type": "text",  "text": "Describe this image."}
    ]},
    {"role": "assistant", "content": [
        {"type": "text",  "text": "The image shows..."}
    ]}
]}
```

**Tips:**
- Keep image dimensions between 300-1000px to control training time and VRAM
- Ensure images are the same dimensions where possible
- Use `UnslothVisionDataCollator` for proper batching

### Training Vision Models
```python
from trl import SFTTrainer, SFTConfig
from unsloth import UnslothVisionDataCollator

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    data_collator = UnslothVisionDataCollator(model, tokenizer),
    train_dataset = dataset,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 30,
        learning_rate = 2e-4,
        optim = "adamw_8bit",
        output_dir = "outputs-vision",
        remove_unused_columns = False,  # REQUIRED for vision
        dataset_num_proc = 4,
    ),
)
trainer.train()
```

---

## Exporting & Deployment

After training, export the model based on the user's deployment target.

### 1. Save LoRA Adapters (Default — lightweight)
```python
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
```

### 2. Push to Hugging Face Hub
```python
model.push_to_hub("your-username/model-name", token = "hf_...")
tokenizer.push_to_hub("your-username/model-name", token = "hf_...")
```

### 3. Export to GGUF (For Ollama, LM Studio, llama.cpp)
Unsloth has built-in GGUF exporters that save massive RAM vs. `llama.cpp` scripts:

```python
# 16-bit GGUF (highest quality)
model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")

# 8-bit GGUF (good balance)
model.save_pretrained_gguf("model", tokenizer, quantization_method = "q8_0")

# 4-bit GGUF (smallest, best for local inference)
model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
```

**Important notes for Mac/`mlx-tune` users**:
- `save_pretrained_gguf` fails when the base model was loaded in 4-bit (`load_in_4bit=True`). Load in FP16 (`load_in_4bit=False`) during training to enable GGUF export.
- The `quantization_method` parameter (e.g. `"q4_k_m"`) is **ignored** by mlx-tune — it always exports fp16. Use llama.cpp to quantize further after export.
- GGUF export only supports **Llama, Mistral, and Mixtral** architectures. Qwen, Gemma, and other models will fail. Use `save_pretrained_merged()` instead for those models.

**Available quantization methods**: `f32`, `f16`, `q8_0`, `q5_k_m`, `q4_k_m`, `q3_k_m`, `q2_k`

### 4. Merge to 16-bit (For vLLM / SGLang)
```python
model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit")
```

### 5. Deploy with Ollama
After exporting to GGUF:
```bash
# Create an Ollama model from your GGUF
ollama create my-model -f Modelfile
ollama run my-model
```

### 6. Deploy with vLLM
After merging to 16-bit:
```bash
vllm serve ./model --dtype auto
```

### 7. Deploy with SGLang
```bash
python -m sglang.launch_server --model-path ./model
```

---

## Troubleshooting Common Errors

### 1. Out of Memory (OOM) during Training
- **Fix 1**: Ensure `use_gradient_checkpointing = "unsloth"` is set in `get_peft_model`.
- **Fix 2**: Reduce `per_device_train_batch_size` to `1` and increase `gradient_accumulation_steps` (e.g., to `4` or `8`).
- **Fix 3**: Reduce `max_seq_length` if the task doesn't require long context.
- **Fix 4**: Switch from LoRA 16-bit to QLoRA 4-bit (`load_in_4bit = True`).
- **Fix 5**: Use a smaller pre-quantized model (e.g., 8B instead of 14B).

### 2. "ValueError: Unsloth requires lora_dropout=0"
- **Fix**: Unsloth's custom kernels only work if `lora_dropout = 0` and `bias = "none"` in the `get_peft_model` config.

### 3. Slow downloading / OOM while loading model
- **Fix**: Use Unsloth's pre-quantized 4-bit models (e.g., `unsloth/llama-3-8b-bnb-4bit`). They download 4x faster and fit reliably in RAM.

### 4. GRPO reward not increasing
- **Fix 1**: Wait at least 300 steps — GRPO is slow to start.
- **Fix 2**: Verify your reward function returns correct scores (print intermediate values).
- **Fix 3**: Use at least 500 rows of training data.
- **Fix 4**: Ensure the model is ≥1.5B params for generating thinking tokens.
- **Fix 5**: Try increasing `num_generations` (e.g., from 4 to 8).

### 5. Vision training errors
- **Fix 1**: Ensure `remove_unused_columns = False` in `SFTConfig`.
- **Fix 2**: Use `UnslothVisionDataCollator` instead of default collator.
- **Fix 3**: Keep image dimensions between 300-1000px.
- **Fix 4**: Verify images are accessible (URLs reachable, local files exist).

### 6. GRPO + vLLM errors
- **Fix 1**: `pip install diffusers` if you get a missing module error.
- **Fix 2**: Update to the latest vLLM version: `pip install --upgrade vllm`.
- **Fix 3**: Ensure `fast_inference = True` is set in `from_pretrained()`.

### 7. Gradient accumulation bug
- **Fix**: Use Unsloth's patched trainers. Standard HuggingFace trainers have a known gradient accumulation bug that Unsloth fixes automatically. See: https://unsloth.ai/blog/gradient

### 8. Hub push failures
- **Fix 1**: Ensure your HF token has write permissions.
- **Fix 2**: Set `push_to_hub = True` and `hub_model_id = "username/model-name"` in config.
- **Fix 3**: For private repos, add `hub_private_repo = True`.

---

## Phase 3: Evaluation (mlx-tune / Apple Silicon)

After training, direct the user to `scripts/mlx_eval_template.py`. It handles the correct mlx-tune inference API and avoids the common failure modes. Key rules encoded in the template:

1. **Load adapter via `from_pretrained` kwarg** — `adapter_path="outputs/adapters"` passed as `**kwargs`. Omitting it runs the bare base model silently.
2. **Pass raw string to `generate`** — `TokenizerWrapper` is not callable with `return_tensors="pt"`.
3. **Temperature via `make_sampler`** — `generate_step` has no `temperature` float; use `sampler=make_sampler(temp=0.7)`.
4. **Strip echoed prompt** — `mlx_lm` returns the full sequence including prompt; do `raw[len(prompt):]`.

Run modes:
```bash
python scripts/mlx_eval_template.py                  # batch
python scripts/mlx_eval_template.py --interactive    # REPL
python scripts/mlx_eval_template.py --compare        # base vs fine-tuned
python scripts/mlx_eval_template.py --style alpaca   # override format
```

## Example Scripts

See the `scripts/` directory for ready-to-use templates:

- **`scripts/unsloth_sft_example.py`**: Complete SFT training script.
- **`scripts/unsloth_dpo_example.py`**: DPO preference training script.
- **`scripts/unsloth_grpo_example.py`**: GRPO reinforcement learning script.
- **`scripts/unsloth_vision_example.py`**: Vision/multimodal fine-tuning script.
- **`scripts/mlx_eval_template.py`**: Evaluation template for Apple Silicon / mlx-tune (batch, interactive, compare modes).

## Resources

- [Unsloth Documentation](https://unsloth.ai/docs)
- [Model Catalog](https://unsloth.ai/docs/get-started/unsloth-model-catalog)
- [Fine-tuning Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide)
- [RL Guide](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide)
- [Vision Guide](https://unsloth.ai/docs/basics/vision-fine-tuning)
- [TTS Guide](https://unsloth.ai/docs/basics/text-to-speech-tts-fine-tuning)
- [Saving to GGUF](https://unsloth.ai/docs/basics/inference-and-deployment/saving-to-gguf)
- [vLLM Deployment](https://unsloth.ai/docs/basics/inference-and-deployment/vllm-guide)
- [All Notebooks](https://unsloth.ai/docs/get-started/unsloth-notebooks)
