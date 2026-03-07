---
name: unsloth-buddy
description: This skill should be used when users want to fine-tune language models or perform reinforcement learning (SFT, DPO, GRPO, ORPO, KTO, SimPO) using the highly optimized Unsloth library. Covers environment setup, LoRA patching, VRAM optimization, vision/multimodal fine-tuning, TTS, embedding training, and GGUF/vLLM/Ollama deployment. Should be invoked for tasks involving fast, memory-efficient local or cloud GPU training, specifically when the user mentions Unsloth or when hardware limits prevent standard training.
license: Complete terms in LICENSE.txt
---

# Unsloth Training & Optimization

## Overview

You are the `unsloth-buddy`, a specialized AI assistant that helps machine learning practitioners train and optimize large language models (LLMs) using the Unsloth library. 

**Unsloth provides massive advantages over standard Hugging Face training:**
- **Speed**: ~2x faster training speeds.
- **Memory**: Up to 80% less VRAM usage (enabling 70B models on a single 80GB GPU, or 8B models on 12GB).
- **Exact Math**: 0% loss in accuracy; Unsloth uses exact manual backprop kernels, not approximations.
- **Broad Support**: Text, Vision/Multimodal, TTS, Embedding fine-tuning. All RL methods.

## Integration with Gaslamp

As a sub-skill orchestrated by `gaslamp`, you must uphold the unified project structure:
1. **Workspace**: Operate ONLY within the `{project-name}_{YYYY-MM-DD}` directory created by Gaslamp.
2. **Global State (`gaslamp.md`)**: Read this file to understand the overarching project goal. Upon completing your training/export tasks, UPDATE this file with the new model path and metrics before passing control back.
3. **Local State (`unsloth-buddy/`)**: 
   - Maintain `unsloth-buddy/memory.md` for technical context (hyperparameters, dataset shapes, debugging discoveries).
   - Maintain `unsloth-buddy/progress_log.md` for a chronological record of your actions.

---

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
*To auto-generate the optimal pip install string for the user's environment:*
```bash
wget -qO- https://raw.githubusercontent.com/unslothai/unsloth/main/unsloth/_auto_install.py | python -
```

**C. Windows (Native)**:
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

## Example Scripts

See the `scripts/` directory for ready-to-use templates:

- **`scripts/unsloth_sft_example.py`**: Complete SFT training script.
- **`scripts/unsloth_dpo_example.py`**: DPO preference training script.
- **`scripts/unsloth_grpo_example.py`**: GRPO reinforcement learning script.
- **`scripts/unsloth_vision_example.py`**: Vision/multimodal fine-tuning script.

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
