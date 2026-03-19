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
SKILL.md                  ← main orchestration logic
sub-skills/interview.md   ← Phase 1: 5-point requirements interview
sub-skills/data.md        ← Phase 2: data acquisition and formatting
scripts/detect_system.py  ← Phase 3: hardware detection (Stage 1)
scripts/detect_env.py     ← Phase 3: env/package detection (Stage 2)
scripts/init_project.py   ← Phase 0: create dated project directory
```

## Lifecycle (7 Phases)

| Phase | What you do |
|-------|-------------|
| 0. Init | Run `python scripts/init_project.py <name>` → creates `{name}_{date}/` |
| 1. Interview | Ask the 5-point Unsloth Contract questions (method, model, data, hardware, deploy) |
| 2. Data | Acquire, validate, and reformat the dataset to the required schema |
| 3. Env | Run detect_system.py then detect_env.py — block until READY |
| 4. Train | Generate and run `train.py` inside the project directory |
| 5. Eval | Run eval against base and fine-tuned model, show comparison |
| 6. Export | Convert to GGUF / merge / push to HF Hub per user's deploy target |

Everything is scoped to the dated project directory. Nothing touches the repo root.

## Key Constraints

- Apple Silicon: use `mlx-tune`, Python ≤ 3.12, create a venv first
- NVIDIA: use `unsloth`, CUDA 12.1+
- Never run training without completing Phase 3 (env check)
- Adapter path must be passed as kwarg to `from_pretrained` for eval to load correctly
