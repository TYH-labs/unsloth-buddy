# Unsloth Buddy

![Unsloth Buddy Thumbnail](https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20logo%20white%20text.png)

**Unsloth Buddy** is an intelligent, automatic, end-to-end AI development agent skill designed to fine-tune Large Language Models (LLMs) and Vision Language Models (VLMs) using the highly optimized [Unsloth](https://github.com/unslothai/unsloth) library and [mlx-tune](https://github.com/ARahim3/mlx-tune) (for Apple Silicon).

This repository contains the `SKILL.md` instruction set that turns generic LLM coding agents into specialized fine-tuning experts.

## Key Features

The Unsloth Buddy provides massive advantages over standard Hugging Face pipelines by leveraging Unsloth's custom Triton kernels:

- 🏎️ **~2x Faster Training Speeds**
- 💾 **Up to 80% Less VRAM Usage** (Run 70B models on a single 80GB GPU)
- 🍎 **Apple Silicon Native Support** (Automatic fallback to `mlx-tune` on M1/M2/M3/M4)
- 🧠 **Math-Exact Exact Accuracy** (0% loss in performance)
- 🎨 **Broad Support**: Text (SFT), Vision/Multimodal, Reinforcement Learning (DPO, GRPO, ORPO, KTO, SimPO).

## The 4-Phase End-to-End Lifecycle

Unsloth Buddy isn't just a passive code generator; it operates as an active AI Developer across four distinct phases:

### Phase 1: Environment Analysis & Setup
The buddy proactively detects your OS hardware footprint (`uname -a`, `nvidia-smi`, `system_profiler`). 
- If standard Linux/Windows + CUDA is detected, it auto-installs the exact optimized `unsloth` whl version.
- If an Apple Silicon Mac is detected, it auto-installs `mlx-tune` to enable native training without CUDA.

### Phase 2: Code Generation & Execution
The buddy generates optimized training scripts (`train.py`) complete with PEFT LoRA patching, VRAM checks, and gradient accumulation fixes.
Instead of handing you the code, it uses its terminal tools to proactively ask: *"Should I execute the training script now?"* and runs the job locally.

### Phase 3: Evaluation & Metrics
Once training is finished, the buddy reads the log outputs to verify loss convergence and automatically writes an `eval.py` script to run inference on tests.

### Phase 4: Export & Conversion
The buddy handles finalizing the weights. It prompts the user for their desired deployment target and executes the necessary CLI commands to export the model to:
- **16-bit Fast Inference** (for vLLM / SGLang)
- **GGUF Format** (for Ollama / LM Studio / llama.cpp)
- **Hugging Face Hub**

## Example Scripts

This repository includes several template scripts that the buddy references to ensure best practices:

- `scripts/unsloth_sft_example.py`: Standard Supervised Fine-Tuning.
- `scripts/unsloth_mlx_sft_example.py`: SFT natively on Apple Silicon using `mlx-tune`.
- `scripts/unsloth_dpo_example.py`: Direct Preference Optimization.
- `scripts/unsloth_grpo_example.py`: DeepSeek-R1 style Group Relative Policy Optimization (GRPO) with Reward Functions.
- `scripts/unsloth_vision_example.py`: Multimodal/Vision fine-tuning for Qwen2.5-VL and Llama-3.2 Vision.

## Usage (Gaslamp / Agentic Systems)

To use this skill within an agentic loop (like [Gaslamp](https://github.com/TYH-labs/gaslamp) or Claude Code), copy the `SKILL.md` file into your agent's system prompt or workspace context. 

When invoking the agent, simply tell it your goal:
```text
"Hey buddy, I want to fine-tune Llama 3.1 8B on a custom CSV dataset of medical documents. Use Unsloth."
```

The agent will automatically read `SKILL.md` and initiate Phase 1 of the development lifecycle!

## License

This project incorporates instructions and patterns based on the open-source Unsloth library. Please refer to `LICENSE.txt` for details.
