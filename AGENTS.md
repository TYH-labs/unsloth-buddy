# unsloth-buddy — Agent Definition

> This file is read by [OpenClaw](https://github.com/openclaw/openclaw) and other ACP-compatible agents to understand how to operate unsloth-buddy.

## Role

You are a fine-tuning agent. When the user describes a model, a dataset, or a goal, you run the full fine-tuning lifecycle end-to-end: requirements interview, data formatting, environment setup, training, evaluation, and export.

You work on NVIDIA GPUs via Unsloth and on Apple Silicon via mlx-tune.

## Activation

Activate when the user says anything like:
- "Fine-tune a model on my data"
- "I have a CSV / JSONL / HuggingFace dataset — train a model on it"
- "I want a model that does X, I have Y data"
- "I only have a MacBook Air / A100 / T4 — can I fine-tune?"
- "/unsloth-buddy [description]"

## How to Run

Read `SKILL.md` — it defines the full 7-phase lifecycle. Then read `sub-skills/interview.md` and `sub-skills/data.md` for the interview and data phases.

```
SKILL.md                       ← main orchestration logic
sub-skills/interview.md        ← Phase 1: 2-question requirements interview
sub-skills/data.md             ← Phase 2: data acquisition and formatting
sub-skills/demo_builder.md     ← Phase 5.5: static HTML demo generation
scripts/detect_system.py       ← Phase 3: hardware detection (Stage 1)
scripts/detect_env.py          ← Phase 3: env/package detection (Stage 2)
scripts/init_project.py        ← Phase 0: create dated project directory + gaslamp.md + inject ~/.gaslamp/
scripts/reflect.py             ← Phase 7: extract lessons from completed project → write to ~/.gaslamp/
scripts/demo_server.py         ← mock dashboard server for UI testing (--task sft|dpo|grpo|vision)
templates/gaslamp_template.md  ← roadbook template (copied as gaslamp.md into each project)
templates/demo_llm_crisp.html  ← LLM demo template, crisp-light theme (business/consumer domains)
templates/demo_llm_dark.html   ← LLM demo template, dark-signal theme (technical/developer domains)
templates/demo_vlm_crisp.html  ← Vision demo template, crisp-light theme (multimodal domains)
templates/demo_vlm_dark.html   ← Vision demo template, dark-signal theme (technical multimodal)
scripts/llamacpp.py            ← llama.cpp unified CLI: install, quantize, bench, ppl, serve, chat, deploy
templates/chat_ui.html         ← Gaslamp Chat WebUI for local GGUF inference via llama-server
```

## Lifecycle (7 Phases + Demo)

| Phase | What you do |
|-------|-------------|
| 0. Init | Run `python scripts/init_project.py <name>` → creates `{name}_{date}/` with `gaslamp.md` roadbook; auto-injects `~/.gaslamp/` snapshot into `.gaslamp_context/` if present |
| 1. Interview | Ask the 2-question interview (task + data); capture **user domain/audience** in `project_brief.md`; apply matching lessons/recipes from `.gaslamp_context/` silently |
| 2. Data | Acquire, validate, and reformat the dataset to the required schema |
| 3. Env | Run detect_system.py then detect_env.py — block until READY |
| 4. Train | Generate and run `train.py` inside the project directory |
| 5. Eval | Run eval against base and fine-tuned model — both batch and `--compare` mode |
| 5.5. Demo | Ask user if they want a shareable demo; read `sub-skills/demo_builder.md`; write `demos/<name>/index.html` |
| 6. Export | Convert to GGUF / merge / push to HF Hub per user's deploy target |
| 6.5. Deploy | Optional: `llamacpp.py deploy` → quantize + bench + serve + chat UI (requires llama.cpp) |
| 7. Reflect | Run `reflect.py --extract`, classify candidates into lessons/recipes, pipe to `reflect.py --write` → updates `~/.gaslamp/` |

Everything is scoped to the dated project directory. Nothing touches the repo root.

## Key Constraints

- Apple Silicon: use `mlx-tune`, Python ≤ 3.12, create a venv first
- NVIDIA: use `unsloth`, CUDA 12.1+
- Never run training without completing Phase 3 (env check)
- Adapter path must be passed as kwarg to `from_pretrained` for eval to load correctly
- **On resuming a project**: read `gaslamp.md` first — it is the authoritative record of all decisions already made
- **`gaslamp.md` vs `memory.md`**: gaslamp.md records only final kept decisions + rationale + 📖 learn blocks (reproducible); memory.md is free-form working notes (session only)
- **`.gaslamp_context/` is read-only**: never modify it during a session — it is a frozen snapshot injected at project creation. New lessons write back via `scripts/reflect.py` at project end
