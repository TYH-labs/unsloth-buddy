# unsloth-buddy

<p align="center"><img src="images/unsloth_gaslamp.png" width="75%" alt="unsloth-buddy" /></p>

<p align="center">
  <a href="https://github.com/TYH-labs/unsloth-buddy"><img src="https://img.shields.io/github/stars/TYH-labs/unsloth-buddy?style=flat&logo=github&color=181717&logoColor=white" alt="GitHub" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="MIT License" /></a>
  <a href="#quick-start"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white" alt="Python 3.10+" /></a>
  <a href="https://gaslamp.dev/unsloth"><img src="https://img.shields.io/badge/%F0%9F%94%A5%20Gaslamp-Compatible-ff6b00?logoColor=white" alt="Gaslamp Compatible" /></a>
  <a href="#openclaw"><img src="https://img.shields.io/badge/%F0%9F%A6%9E%20OpenClaw-Compatible-ff4444" alt="OpenClaw Compatible" /></a>
  <a href="#quick-start"><img src="https://img.shields.io/badge/%F0%9F%A4%96%20Agent-Claude%20Code%20%2F%20Codex%20%2F%20Gemini-8b5cf6" alt="Agent Compatible" /></a>
  <a href="https://discord.gg/mZe4mbCQ6a"><img src="https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white" alt="Discord" /></a>
</p>

<p align="center"><code>/unsloth-buddy I have 500 customer support Q&As and want to fine-tune a summarization model. I only have a MacBook Air.</code></p>

<p align="center">
  <a href="#quick-start"><img src="https://img.shields.io/badge/Try%20It-1%20minute-black?style=for-the-badge" alt="Try It" /></a>
  <a href="demos/"><img src="https://img.shields.io/badge/Demos-Examples-6e40c9?style=for-the-badge" alt="Demos" /></a>
  <a href="SKILL.md"><img src="https://img.shields.io/badge/10%2B%20Features-Details-0969da?style=for-the-badge" alt="Features" /></a>
</p>

<p align="center">
  English | <a href="README_zh-Hans.md">简体中文</a> | <a href="README_zh-Hant.md">繁體中文</a>
</p>

---

## What is this?

A fine-tuning agent that talks like a colleague. Describe your goal, and it asks the right questions, finds or formats your data, picks the right technique and model, trains on your hardware, validates the result, and packages it for deployment.

Runs on NVIDIA GPUs via [Unsloth](https://github.com/unslothai/unsloth), natively on Apple Silicon via [mlx-tune](https://github.com/ml-explore/mlx-lm), and on free cloud GPUs via [colab-mcp](https://github.com/googlecolab/colab-mcp). Part of the [Gaslamp](https://gaslamp.dev/) AI development platform — [docs](https://gaslamp.dev/unsloth).

---

## One sentence, one fine-tuned model.

```
You: Fine-tune a small model on my customer support FAQ. I have a CSV file.

[Phase 0] Creating project: customer_faq_sft_2026_03_17/
[Phase 1] Requirements interview...
           Method: SFT   Model: Qwen2.5-0.5B   Deploy: Ollama
[Phase 2] Data strategy...
           Loading 1,200 rows from faq.csv → reformatting as chat messages
           Saved to data/train.jsonl (validated: messages column ✓)
[Phase 3] Environment: Apple M4 24GB, mlx-tune 0.4.3, Python 3.12
           Ready for training
[Phase 4] Training... 200 steps
           Final loss: 1.42 → saved to outputs/adapters/
[Phase 5] Evaluation (base vs fine-tuned, greedy decoding):
           Q: How do I reset my password?
           [Base]    I can help with that. Which password?
           [Fine-tuned]  Go to the login page → click "Forgot password" → check your email.
[Phase 6] Export → outputs/model-q4_k_m.gguf
           Run: ollama create my-faq-bot -f Modelfile && ollama run my-faq-bot
```

One conversation, seven phases, one deployable model.

---

## Quick Start

This skill includes sub-skills and utility scripts — install the full repository, not a single file.

**Claude Code** *(recommended)*
```
/plugin marketplace add TYH-labs/unsloth-buddy
/plugin install unsloth-buddy@TYH-labs/unsloth-buddy
```
Then describe what you want to fine-tune. The skill activates automatically.

**Gemini CLI**
```bash
gemini extensions install https://github.com/TYH-labs/unsloth-buddy --consent
```

**Any agent supporting the [Agent Skills](https://agentskills.io/) standard**
```bash
git clone https://github.com/TYH-labs/unsloth-buddy.git .agents/skills/unsloth-buddy
```

---

## How is it different?

Most tools assume you already know what to do. This one doesn't.

| Your concern | What actually happens |
|---|---|
| **"I don't know where to start"** | A 5-point interview locks in method, model, data, hardware, and deployment target before writing any code |
| **"I don't have data, or it's in the wrong format"** | A dedicated data phase acquires, generates, or reformats data to exactly match the trainer's required schema |
| **"SFT? DPO? GRPO? Which one?"** | Maps your goal to the right technique and explains why in plain language |
| **"Which model? Will it fit in my GPU?"** | Detects your hardware, maps to available model sizes, estimates cloud cost if needed |
| **"Unsloth won't install on my machine"** | Two-stage environment detection catches mismatches and prints the exact install command for your setup |
| **"I trained it, but does it work?"** | Runs the fine-tuned adapter alongside the base model so you can see the difference, not just a loss number |
| **"How do I deploy it?"** | You name the target (Ollama, vLLM, HF Hub) — it runs the conversion commands |

---

## How it works

Seven phases, each scoped to an isolated dated project directory that never touches your repo root.

| Phase | What happens | Output files |
|---|---|---|
| **0. Init** | Creates `{name}_{date}/` with standard directory structure | `progress_log.md` |
| **1. Interview** | 5-point Unsloth contract — method, model, data, hardware, deployment | `project_brief.md` |
| **2. Data** | Acquires, validates, and formats to trainer schema | `data_strategy.md` |
| **3. Environment** | Hardware scan → Python env check → blocks until ready | `detect_env_result.json` |
| **4. Training** | Generates and runs `train.py`, streams output to log | `outputs/adapters/` |
| **5. Evaluation** | Batch tests, interactive REPL, base vs fine-tuned comparison | `logs/eval.log` |
| **6. Export** | GGUF, merged 16-bit, or Hub push | `outputs/` |

```
customer_faq_sft_2026_03_17/
├── train.py              eval.py
├── data/                 outputs/adapters/
├── logs/
├── project_brief.md      data_strategy.md
├── memory.md             progress_log.md
```

---

## Hardware Support

| Hardware | Backend | What it can run |
|---|---|---|
| NVIDIA T4 (16 GB) | `unsloth` | 7B QLoRA, small-scale GRPO |
| NVIDIA A100 (80 GB) | `unsloth` | 70B QLoRA, 14B LoRA 16-bit |
| Apple M1 / M2 / M3 / M4 | `mlx-tune` | 7B on 10 GB unified memory, 13B on 24 GB |
| Google Colab (T4/L4/A100) | `unsloth` via `colab-mcp` | Free cloud GPU, opt-in |

Unsloth is ~2× faster than standard HuggingFace training, uses up to 80% less VRAM, and produces exact gradients.

**Supported training methods:** SFT, DPO, GRPO, ORPO, KTO, SimPO, Vision SFT (Qwen2.5-VL, Llama 3.2 Vision, Gemma 3)

---

## Training Dashboard

Every local training run automatically opens a real-time dashboard at **http://localhost:8080/**:

- **SSE streaming** — instant updates via `EventSource`, no polling lag
- **EMA smoothed loss** — clear trend line over noisy raw loss, plus running average
- **Dynamic phase badge** — idle → training → completed / error
- **ETA & elapsed time** — estimated time remaining based on step progress
- **Gradient norm** — fades in when data arrives
- **Evaluation metrics** — auto-reveals eval loss / accuracy with animated empty states
- **Peak VRAM** — tracks GPU (CUDA) and Apple MPS memory usage

Works on both NVIDIA (via `GaslampDashboardCallback`) and Apple Silicon (via `MlxGaslampDashboard` stdout interceptor).

---

## Google Colab Training

Apple Silicon users who need larger models or CUDA-only features can offload training to a free Colab GPU:

1. Install `colab-mcp` in Claude Code:
   ```bash
   uv python install 3.13
   claude mcp add colab-mcp -- uvx --from git+https://github.com/googlecolab/colab-mcp --python 3.13 colab-mcp
   ```
2. Open a Colab notebook, connect to a T4/L4 GPU runtime
3. The agent connects, installs Unsloth, starts training in a background thread, and polls metrics every 30s
4. Download adapters from the Colab file browser when done

Local mlx-tune remains the default — Colab is opt-in for when you need more power.

---

## Gaslamp

`unsloth-buddy` works standalone or as part of a larger [Gaslamp](https://gaslamp.dev/) project — an agentic platform that orchestrates the full ML lifecycle from research to training to deployment. When called via Gaslamp, the project directory and state are shared across skills, and results pass automatically to the next phase.

[gaslamp.dev/unsloth](https://gaslamp.dev/unsloth) — [gaslamp.dev](https://gaslamp.dev/)

---

## OpenClaw

**unsloth-buddy is an [OpenClaw](https://github.com/openclaw/openclaw)-compatible skill.** Share the repo URL with OpenClaw, describe what you want to fine-tune — it reads `AGENTS.md`, understands the workflow, and runs everything automatically.

```
1. Share https://github.com/TYH-labs/unsloth-buddy with OpenClaw
2. OpenClaw reads AGENTS.md → understands the 7-phase fine-tuning lifecycle
3. Say: "Fine-tune a model on my customer support data"
4. Done — OpenClaw runs the interview, formats data, trains, evaluates, and exports
```

For Claude Code, Gemini CLI, Codex, or any ACP-compatible agent: provide `AGENTS.md` as context and the agent will automatically follow the same workflow.

---

## Changelog

- **2026-03-18** — Added Google Colab training support via [colab-mcp](https://github.com/googlecolab/colab-mcp): free T4/L4/A100 GPU access from Claude Code, background-thread training with live polling, and adapter download workflow.

---

## License

See `LICENSE.txt`. Unsloth is MIT licensed, mlx-tune is MIT licensed.
