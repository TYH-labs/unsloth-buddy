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
  <a href="#local-deploy"><img src="https://img.shields.io/badge/Backend-Unsloth%20%7C%20MLX%20%7C%20llama.cpp-00b4d8" alt="Backend: Unsloth / MLX / llama.cpp" /></a>
</p>

<p align="center"><code>/unsloth-buddy I have 500 customer support Q&As and want to fine-tune a summarization model. I only have a MacBook Air.</code></p>

<p align="center">
  <a href="#quick-start"><img src="https://img.shields.io/badge/Try%20It-1%20minute-black?style=for-the-badge" alt="Try It" /></a>
  <a href="demos/"><img src="https://img.shields.io/badge/Demos-Examples-6e40c9?style=for-the-badge" alt="Demos" /></a>
  <a href="SKILL.md"><img src="https://img.shields.io/badge/10%2B%20Features-Details-0969da?style=for-the-badge" alt="Features" /></a>
</p>

<p align="center">
  <a href="https://youtu.be/wG28uxDGjHE"><img src="https://img.shields.io/badge/▶%20Demo-YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white" alt="YouTube Demo" /></a>
  <a href="https://www.bilibili.com/video/BV1VWAFzmECy/"><img src="https://img.shields.io/badge/▶%20演示-Bilibili-00A1D6?style=for-the-badge&logo=bilibili&logoColor=white" alt="Bilibili Demo" /></a>
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

One conversation, seven phases, one deployable model — and a shareable demo page.

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
| **"I don't know where to start"** | A 2-question interview locks in task, audience, and data — then recommends the right model, hardware, and method |
| **"I don't have data, or it's in the wrong format"** | A dedicated data phase acquires, generates, or reformats data to exactly match the trainer's required schema |
| **"SFT? DPO? GRPO? Which one?"** | Maps your goal to the right technique and explains why in plain language |
| **"Which model? Will it fit in my GPU?"** | Detects your hardware, maps to available model sizes, estimates cloud cost if needed |
| **"Unsloth won't install on my machine"** | Two-stage environment detection catches mismatches and prints the exact install command for your setup |
| **"I trained it, but does it work?"** | Runs the fine-tuned adapter alongside the base model so you can see the difference, not just a loss number |
| **"How do I deploy it?"** | You name the target (Ollama, vLLM, HF Hub) — it runs the conversion commands |
| **"How do I reproduce this later — or hand it off?"** | Every project gets a `gaslamp.md` roadbook: every kept decision with its rationale, plus 📖 learn blocks on the underlying ML concepts — enough for any agent or person to reproduce end-to-end |

---

## How it works

Seven phases, each scoped to an isolated dated project directory that never touches your repo root.

| Phase | What happens | Output files |
|---|---|---|
| **0. Init** | Creates `{name}_{date}/` with standard directory structure | `gaslamp.md`, `progress_log.md` |
| **1. Interview** | 2-question interview — task + data; captures domain/audience | `project_brief.md` |
| **2. Data** | Acquires, validates, and formats to trainer schema | `data_strategy.md` |
| **3. Environment** | Hardware scan → Python env check → blocks until ready | `detect_env_result.json` |
| **4. Training** | Generates and runs `train.py`, streams output to log | `outputs/adapters/` |
| **5. Evaluation** | Batch tests, interactive REPL, base vs fine-tuned comparison | `logs/eval.log` |
| **5.5. Demo** | Generates a shareable static HTML page — base vs fine-tuned side-by-side | `demos/<name>/index.html` |
| **6. Export** | GGUF, merged 16-bit, or Hub push | `outputs/` |
| **6.5. Local Deploy** | Optional: quantize → bench → serve + Gaslamp Chat WebUI (requires llama.cpp) | `outputs/*.gguf` |

```
customer_faq_sft_2026_03_17/
├── train.py              eval.py
├── data/                 outputs/adapters/
├── logs/
├── gaslamp.md            ← reproducibility roadbook
├── project_brief.md      data_strategy.md
├── memory.md             progress_log.md
```

---

## Hardware Support

| Hardware | Backend | What it can run |
|---|---|---|
| NVIDIA T4 (16 GB) | `unsloth` | 7B QLoRA, small-scale GRPO |
| NVIDIA A100 (80 GB) | `unsloth` | 70B QLoRA, 14B LoRA 16-bit |
| Apple M1 / M2 / M3 / M4 | `mlx-tune` / `mlx-vlm` / `trl` | SFT/DPO: 7B on 10 GB, 13B on 24 GB; Vision SFT via `mlx-vlm`; GRPO: 1–7B via TRL + PyTorch MPS |
| Google Colab (T4/L4/A100) | `unsloth` via `colab-mcp` | Free cloud GPU, opt-in |

Unsloth is ~2× faster than standard HuggingFace training, uses up to 80% less VRAM, and produces exact gradients.

**Supported training methods:** SFT, DPO, GRPO, ORPO, KTO, SimPO, Vision SFT (Qwen2.5-VL, Llama 3.2 Vision, Gemma 3, Gemma 4)

---

## Training Dashboard

Every local training run automatically opens a real-time dashboard at **http://localhost:8080/**:

- **Task-aware panels** — pass `task_type="sft"|"dpo"|"grpo"|"vision"` to unlock the right charts automatically
- **SSE streaming** — instant updates via `EventSource`, no polling lag
- **EMA smoothed loss** — clear trend line over noisy raw loss, plus running average
- **Dynamic phase badge** — idle → training → completed / error, with colour-coded task-type badge
- **ETA, elapsed time & epoch** — estimated time remaining and current epoch progress
- **GPU memory breakdown** — baseline (model load) vs LoRA training overhead vs total, shown as gauge bars; works on both NVIDIA (CUDA) and Apple Silicon (MPS via `driver_allocated_memory` / `recommended_max_memory`)
- **GRPO panels** — reward ± std-dev confidence band + KL divergence chart
- **DPO panels** — chosen vs rejected reward + KL divergence chart
- **Gradient norm & tokens/sec** — live stats row, fades in when data arrives
- **Completed summary banner** — final memory and runtime stats on training end
- **Terminal UI (Plotext)** — `scripts/terminal_dashboard.py` with `--once` for CLI snapshots; upgrades to 2×2 layout for DPO/GRPO
- **Demo server** — `python scripts/demo_server.py --task grpo --hardware mps|nvidia` serves rich mock data so you can preview every panel without a GPU

Works on both NVIDIA (via `GaslampDashboardCallback(task_type=...)`) and Apple Silicon (via `MlxGaslampDashboard(task_type=...)`).

---

## Demo Builder

After evaluation, the agent can generate a **static HTML demo page** that showcases base model vs fine-tuned outputs side-by-side — open it in any browser, no server needed. Great for sharing results with teammates, stakeholders, or in a portfolio.

The demo builder is part of the [Gaslamp](https://gaslamp.dev/) platform's presentation toolkit. We've simplified it for unsloth-buddy with two built-in themes and automatic domain-specific color customization:

| Theme | Best for | Look |
|---|---|---|
| **crisp-light** | Business, healthcare, education, general | Clean, minimal, light background |
| **dark-signal** | Code, math, security, DevOps | Bold, high-contrast, monospace output |

The accent color is auto-selected based on your model's domain (e.g. teal for healthcare, amber for education, electric cyan for code) — or you can pick your own.

**Try the live example:** [`demos/qwen2.5-0.5b-chip2-sft/index.html`](demos/qwen2.5-0.5b-chip2-sft/index.html) — download and open in any browser.

---

## Local Deploy

After GGUF export, if llama.cpp is detected on your system (checked in Phase 3), the agent offers a one-command local deploy:

```bash
python scripts/llamacpp.py deploy \
    --model outputs/model-f16.gguf --quant q4_k_m --bench --serve
```

This runs the full pipeline: quantize → benchmark → start an OpenAI-compatible server → open the Gaslamp Chat WebUI at `http://localhost:8081/`. Individual subcommands are also available:

```bash
python scripts/llamacpp.py install              # install llama.cpp (brew / cmake)
python scripts/llamacpp.py quantize --input model.gguf --types q4_k_m q8_0
python scripts/llamacpp.py bench --models model-q4_k_m.gguf
python scripts/llamacpp.py serve --model model-q4_k_m.gguf --port 8081
python scripts/llamacpp.py chat --model model-q4_k_m.gguf
```

Requires [llama.cpp](https://github.com/ggml-org/llama.cpp) — installed automatically via `llamacpp.py install`.

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

Every project also gets a **`gaslamp.md` roadbook** — a reproducibility record that captures every kept decision with its rationale and 📖 learn blocks on the underlying ML concepts. Any agent or person can hand this file to a fresh session and reproduce the project end-to-end, or use it to understand *why* each choice was made.

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

- **2026-04-12** — Added **llama.cpp local deploy** (Phase 6.5): after GGUF export, if llama.cpp is installed, the agent offers a one-command pipeline — quantize → benchmark → serve + open the Gaslamp Chat WebUI (`templates/chat_ui.html`). `scripts/llamacpp.py` provides 7 subcommands (`install`, `quantize`, `bench`, `ppl`, `serve`, `chat`, `deploy`); auto-selects GPU offload on Apple Silicon (Metal) and NVIDIA. `scripts/detect_system.py` now detects llama.cpp binaries and prints an install hint if missing.
- **2026-04-10** — Added native **Vision SFT for Apple Silicon**: Integrated [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) to support multimodal fine-tuning (e.g. Gemma 4 Vision, Qwen2.5-VL) on M-series chips. Added `scripts/unsloth_mlx_vision_example.py` training template and `mlx_eval_vision_template.py` for comparative vision evaluation. Demo Builder now supports wide-format VLM layouts (`vlm-crisp`, `vlm-dark`) and relative PNG asset packaging for offline-portable multimodal dashboards.
- **2026-04-09** — Demo Builder improvements: auto-resolves conceptual/movie keywords (e.g. "matrix" → nvidia, "star wars" → spacex) to the best-fit brand before calling the design search script; distinguishes shallow vs. deep DESIGN.md overrides — deep overrides (structural layout changes like all-black or hero/light-content split) skip the CSS injection point and write the demo file from scratch. Added `scripts/search_design.py` to skill resources for fetching brand design templates without `npx`.
- **2026-04-04** — Added Demo Builder (Phase 5.5): after evaluation, generates a static HTML demo page showing base vs fine-tuned outputs side-by-side. Two themes (crisp-light, dark-signal) with automatic domain-specific accent colors. No server needed — open the file in any browser. Part of the [Gaslamp](https://gaslamp.dev/) presentation toolkit, simplified for unsloth-buddy. Interview simplified from 5-point contract to 2-question format (task + data) that also captures user domain/audience for demo theming.
- **2026-03-22** — Added `gaslamp.md` reproducibility roadbook: every project now records all kept decisions with rationale and 📖 ML concept explanations (method, model, data, hyperparameters, eval, export), so any agent or person can reproduce the project end-to-end and understand *why* each choice was made. Template lives at `templates/gaslamp_template.md`; auto-generated by `init_project.py`.
- **2026-03-21** — Enhanced training dashboard: task-aware panels (SFT/DPO/GRPO/Vision), GPU memory breakdown (baseline vs LoRA vs total), GRPO reward ± std and KL divergence charts, DPO chosen/rejected reward and KL charts, epoch tracking, completed-training summary banner, terminal 2×2 layout for DPO/GRPO, and new `scripts/demo_server.py` mock server for UI development without a GPU.
- **2026-03-19** — Added terminal training dashboard (`scripts/terminal_dashboard.py`): live `plotext` charts of loss and learning rate in the terminal, with `--once` mode for Claude Code one-shot progress checks.
- **2026-03-18** — Added Google Colab training support via [colab-mcp](https://github.com/googlecolab/colab-mcp): free T4/L4/A100 GPU access from Claude Code, background-thread training with live polling, and adapter download workflow.

---

## License

See `LICENSE.txt`. Unsloth is MIT licensed, mlx-tune is MIT licensed.
