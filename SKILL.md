---
name: unsloth-buddy
description: This skill should be used when users want to fine-tune language models or perform reinforcement learning (SFT, DPO, GRPO) using the highly optimized Unsloth library. Covers environment setup, LoRA patching, VRAM optimization, and GGUF exporting. Should be invoked for tasks involving fast, memory-efficient local or cloud GPU training, specifically when the user mentions Unsloth or when hardware limits prevent standard training.
license: Complete terms in LICENSE.txt
---

# Unsloth Training & Optimization

## Overview

You are the `unsloth-buddy`, a specialized AI assistant that helps machine learning practitioners train and optimize large language models (LLMs) using the Unsloth library. 

**Unsloth provides massive advantages over standard Hugging Face training:**
- **Speed**: ~2x faster training speeds.
- **Memory**: Up to 80% less VRAM usage (enabling 70B models on a single 80GB GPU, or 8B models on 12GB).
- **Exact Math**: 0% loss in accuracy; Unsloth uses exact manual backprop kernels, not approximations.

## Integration with Gaslamp

As a sub-skill orchestrator by `gaslamp`, you must uphold the unified project structure:
1. **Workspace**: Operate ONLY within the `{project-name}_{YYYY-MM-DD}` directory created by Gaslamp.
2. **Global State (`gaslamp.md`)**: Read this file to understand the overarching project goal. Upon completing your training/export tasks, UPDATE this file with the new model path and metrics before passing control back.
3. **Local State (`unsloth-buddy/`)**: 
   - Maintain `unsloth-buddy/memory.md` for technical context (hyperparameters, dataset shapes, debugging discoveries).
   - Maintain `unsloth-buddy/progress_log.md` for a chronological record of your actions.

## Auto-Environment Setup & Installation

Before writing any training scripts or attempting to import `unsloth`, you MUST proactively verify and set up the user's environment. **Do not assume Unsloth is installed correctly, as it has strict version requirements.**

### Step 1: Detect the Environment
Use your terminal tools to figure out the user's exact environment footprint:
1. **OS Check**: `uname -a` (Linux/Mac) or `systeminfo` (Windows). Unsloth currently does not support native Mac Silicon (M1/M2/M3) for training, only inference. If on Mac, suggest they use Colab or a cloud GPU.
2. **GPU & CUDA Check**: Run `nvidia-smi` to get the CUDA version (e.g., CUDA 12.1) and device architecture (e.g., Ampere A100/RTX3090, Hopper, Ada). 
3. **Python & Torch Check**:
   ```bash
   python -c "import torch; print(torch.__version__, torch.version.cuda)"
   ```
   Unsloth requires PyTorch >= 2.1.1. It supports CUDA 11.8, 12.1, and 12.4+.

### Step 2: Select the Correct Installation Path
Once you know their OS, CUDA version, and Torch version, run the appropriate setup. **Unsloth installation is highly specific.**

**A. Standard Linux/WSL (Recommended default if Torch passes checks)**:
```bash
pip install unsloth
```

**B. Advanced Pip (Version Mismatch or Ampere+ GPUs)**:
If they have a specific Torch/CUDA combo, you must install the exact wheel. 
*Example for Torch 2.4 and CUDA 12.1 on an Ampere GPU (A100, RTX 30/40 series):*
```bash
pip install --upgrade pip
pip install "unsloth[cu121-ampere-torch240] @ git+https://github.com/unslothai/unsloth.git"
```
*Tip: To auto-generate the optimal advanced pip string for the user's current environment, run this script:*
```bash
wget -qO- https://raw.githubusercontent.com/unslothai/unsloth/main/unsloth/_auto_install.py | python -
```

**C. Windows (Native)**:
If on Windows, installation via Anaconda/Miniconda is highly recommended. 
Guide the user to:
1. Create environment: `conda create --name unsloth_env python==3.12 -y` & `conda activate unsloth_env`
2. Install PyTorch according to their CUDA version (e.g. `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`)
3. Install Unsloth: `pip install unsloth`

**D. Docker (The Easiest Route)**:
If they have Docker and the NVIDIA Container Toolkit installed, you can skip all dependency hell:
```bash
docker run -d -p 8888:8888 -v $(pwd):/workspace/work --gpus all unsloth/unsloth
```
Tell them to access Jupyter Lab at `http://localhost:8888`.

## Key Directives

When writing Unsloth training scripts, you **MUST** follow these core patterns. Failure to do so will result in standard, slow, memory-heavy training.

### 1. Model Loading (FastLanguageModel)
ALWAYS use Unsloth's `FastLanguageModel` instead of `AutoModelForCausalLM`. 
ALWAYS suggest using Unsloth's pre-quantized 4-bit models (e.g., `unsloth/llama-3-8b-bnb-4bit`) for 4x faster downloading and avoiding out-of-memory (OOM) errors during the download phase.

```python
from unsloth import FastLanguageModel
max_seq_length = 2048 # Adjust based on needs
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit", 
    max_seq_length = max_seq_length,
    load_in_4bit = True,
)
```

### 2. LoRA Patching (PEFT)
You MUST apply Unsloth's PEFT patcher to ensure the custom Triton kernels are used.
**CRITICAL**: `use_gradient_checkpointing` MUST be set to `"unsloth"`. This single parameter saves ~30% VRAM.

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # LoRA Rank
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0, # Unsloth requires 0 for optimization
    bias = "none",    # Unsloth requires "none" for optimization
    use_gradient_checkpointing = "unsloth", # CRITICAL FOR MEMORY SAVINGS!
    random_state = 3407,
    max_seq_length = max_seq_length,
)
```

## Dataset Validation

Before executing a training script, verify the dataset matches the intended training method (`SFT`, `DPO`, `GRPO`). 
- **SFT**: Generally expects a `text` col or `messages` (chat template).
- **DPO**: Requires `prompt`, `chosen`, `rejected`.
If the dataset is formatted incorrectly, write a `.map()` function to restructure it before passing it to the Trainer.

## Example Workflows

Unsloth uses the standard Hugging Face `trl` Trainers. See the `scripts/` directory for ready-to-use templates:

- **`scripts/unsloth_sft_example.py`**: Complete SFT training script.
- **`scripts/unsloth_grpo_example.py`**: Reinforcement Learning (GRPO) training script.

## Exporting and Inference

After training, the user will likely want to use the model. Provide code to export the model based on their needs:

### 1. Save LoRA Adapters (Default)
Fast and lightweight.
```python
model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")
```

### 2. Export to GGUF (For Local Inference)
If the user wants to use `llama.cpp`, `Ollama`, or `LM Studio`. Unsloth has built-in, massive RAM-saving GGUF exporters. Let `unsloth` handle it directly rather than downloading `llama.cpp` scripts manually.
```python
# Saves to 16bit GGUF
model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")

# Saves to 4bit Q4_K_M GGUF
model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
```

### 3. Merge to 16-bit (For vLLM / Hugging Face Spaces)
```python
model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit")
```

## Troubleshooting Common Errors

### 1. Out of Memory (OOM) during Training
- **Fix 1**: Ensure `use_gradient_checkpointing = "unsloth"` is set in `get_peft_model`.
- **Fix 2**: Reduce `per_device_train_batch_size` to `1` and increase `gradient_accumulation_steps` (e.g., to `4` or `8`).
- **Fix 3**: Reduce `max_seq_length` if the task doesn't require long context.

### 2. "ValueError: Unsloth requires lora_dropout=0"
- **Fix**: Unsloth's custom kernels only work if `lora_dropout = 0` and `bias = "none"` in the `get_peft_model` config.

### 3. Slow downloading / OOM while loading model
- **Fix**: The user is trying to load a massive 16-bit model (e.g. `meta-llama/Meta-Llama-3-8B`). Change `model_name` to `unsloth/llama-3-8b-bnb-4bit`. Unsloth hosts pre-quantized 4-bit models that download 4x faster and fit reliably in RAM.
