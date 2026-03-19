# Unsloth Buddy

![Unsloth Buddy Thumbnail](https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20logo%20white%20text.png)

**Unsloth Buddy** is an intelligent, end-to-end AI development agent skill that fine-tunes Large Language Models (LLMs) and Vision Language Models (VLMs) using [Unsloth](https://github.com/unslothai/unsloth) and [mlx-tune](https://github.com/ARahim3/mlx-tune) (Apple Silicon). It features a real-time SSE-powered training dashboard and supports remote GPU training via [colab-mcp](https://github.com/googlecolab/colab-mcp).

## Key Features

- 🏎️ **~2x Faster Training** via Unsloth's custom Triton kernels
- 💾 **Up to 80% Less VRAM** (70B models on a single 80GB GPU, 8B on 12GB)
- 🍎 **Apple Silicon Native** — automatic fallback to `mlx-tune` on M1/M2/M3/M4
- ☁️ **Remote Colab Training** — offload to free T4/L4/A100 GPUs via `colab-mcp` (opt-in)
- 📊 **Live SSE Dashboard** — real-time training metrics with EMA smoothing, phase tracking, and ETA
- 🧠 **Math-Exact Accuracy** — 0% loss in performance
- 🎨 **Broad Support**: Text (SFT), Vision/Multimodal, RL (DPO, GRPO, ORPO, KTO, SimPO)

## The 7-Phase End-to-End Lifecycle

Unsloth Buddy operates as an active AI Developer across seven phases:

| Phase | What Happens |
|-------|-------------|
| **0. Project Init** | Creates a dated project directory with `data/`, `outputs/`, `logs/`, and state files |
| **1. Interview** | 5-Point Contract: training method, base model, hardware, data, deployment target |
| **2. Data Strategy** | Acquires/formats datasets to match TRL columns (`messages`, `chosen/rejected`, `prompt`) |
| **3. Environment Setup** | Auto-detects hardware, installs packages. Dual-path for Mac: local mlx-tune or Colab GPU |
| **4. Code Gen & Execution** | Generates `train.py`, attaches the Gaslamp dashboard callback, runs training |
| **5. Evaluation** | Runs inference tests, records qualitative results |
| **6. Export** | Exports to GGUF (Ollama), 16-bit (vLLM/SGLang), or Hugging Face Hub |

## Training Dashboard

The Gaslamp Dashboard provides real-time training observability:

- **SSE streaming** — instant updates via `EventSource`, no polling lag
- **EMA smoothed loss** — clear trend line over noisy raw loss, plus average line
- **Dynamic phase badge** — idle → training → completed/error
- **ETA & elapsed time** — estimated time remaining based on step progress
- **Evaluation metrics** — auto-reveals eval loss/accuracy charts with animated empty states
- **Gradient norm** — hidden until data arrives, then fades in with frosted glass transition

## Training Paths

| Path | Best For | Requirements |
|------|----------|-------------|
| **A. Standard Linux** | Linux/WSL + NVIDIA GPU | `pip install unsloth` |
| **B. Advanced Pip** | CUDA version mismatch, Ampere+ | Auto-generated install string |
| **C. Apple Silicon** | M1/M2/M3/M4 Macs, models ≤8B | `uv pip install mlx-tune` |
| **D. Docker** | Easiest GPU setup | `docker run unsloth/unsloth` |
| **E. Google Colab** | Mac users needing >8B models or CUDA features | `colab-mcp` MCP server |

### Path E: Colab via colab-mcp

For Apple Silicon users who need larger models, CUDA-only features (vLLM, FP8), or GRPO with vLLM:

1. Configure `colab-mcp` in your MCP settings
2. The agent runs `setup_colab.py` to auto-install Unsloth on the Colab VM
3. Dataset is uploaded (base64 for small files, HF Hub for large)
4. Training runs remotely; metrics stream back to the local dashboard
5. Adapters/GGUF downloaded back to local `outputs/` directory

**Local mlx-tune is still the default** — Colab is opt-in for when you need more power.

## Example Scripts

| Script | Description |
|--------|-------------|
| `scripts/unsloth_sft_example.py` | Standard Supervised Fine-Tuning |
| `scripts/unsloth_mlx_sft_example.py` | SFT on Apple Silicon via mlx-tune |
| `scripts/unsloth_dpo_example.py` | Direct Preference Optimization |
| `scripts/unsloth_grpo_example.py` | GRPO with reward functions |
| `scripts/unsloth_vision_example.py` | Vision/multimodal fine-tuning |
| `scripts/gaslamp_callback.py` | SSE dashboard callback for TRL trainers |
| `scripts/setup_colab.py` | Auto-setup Unsloth on Colab VM |
| `scripts/colab_training.py` | Remote Colab training helpers |

## Usage

Copy `SKILL.md` into your agent's workspace context (e.g., [Gaslamp](https://github.com/TYH-labs/gaslamp), Gemini CLI, Claude Code). Then tell it your goal:

```text
"Fine-tune Llama 3.1 8B on a custom medical dataset using Unsloth."
```

The agent reads `SKILL.md` and initiates the 7-phase lifecycle automatically.

## License

This project incorporates instructions and patterns based on the open-source Unsloth library. See `LICENSE.txt` for details.
