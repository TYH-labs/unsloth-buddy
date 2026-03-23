# Gaslamp Gallery — Fine-Tuning Examples

A collection of real `gaslamp.md` roadbooks — each one a complete, reproducible fine-tuning run.

Every example is a single file. Hand it to any agent running [unsloth-buddy](https://gaslamp.dev/unsloth) and the entire project reproduces end-to-end: data download, environment setup, training, and evaluation.

```
/unsloth-buddy reproduce using demos/qwen2.5-0.5b-chip2-sft/gaslamp.md
```

---

## Gallery

| Example | Method | Model | Dataset | Hardware | What you get |
|---------|--------|-------|---------|----------|--------------|
| [qwen2.5-0.5b-chip2-sft](./qwen2.5-0.5b-chip2-sft/gaslamp.md) | SFT | Qwen2.5-0.5B-Instruct | OIG unified_chip2 (200k turns) | Apple Silicon | Compact instruction-following model, ~3 min on M-series Mac |

---

## What is a gaslamp.md?

A `gaslamp.md` is a **reproducibility roadbook** — not a log, not a README, not a notebook. It records every decision that was made and kept during a fine-tuning run, with:

- The reasoning behind each choice
- A 📖 learn block explaining the underlying ML concept and the tradeoffs
- The exact data schema, parse logic, and split parameters
- Full LoRA config and hyperparameters with one-line explanations
- A Source column in the file inventory showing what to copy vs regenerate vs write from scratch
- A load-and-generate snippet in the export section so you can verify the model works end-to-end

**The design test:** a fresh agent that has never seen the original project should be able to reproduce the full training run — from raw dataset download through evaluation — using only `gaslamp.md` and the installed skill, with no access to the original session or project files. This gallery was built by running that test.

---

## How to reproduce an example

1. Make sure unsloth-buddy is installed:
   ```
   /install-plugin https://github.com/TYH-labs/unsloth-buddy
   ```

2. Point it at a roadbook:
   ```
   /unsloth-buddy reproduce using demos/qwen2.5-0.5b-chip2-sft/gaslamp.md
   ```

The agent reads the roadbook, detects your hardware, and reconstructs the project in a fresh dated directory.

---

## How to contribute an example

A contribution is just a `gaslamp.md` file.

1. Run a fine-tuning project using unsloth-buddy (any method: SFT, DPO, GRPO, vision)
2. Verify it reproduces cleanly from the roadbook alone
3. Remove any personal or machine-specific paths (use `.venv/` not `/Users/yourname/...`)
4. Open a PR adding `demos/<descriptive-name>/gaslamp.md`

Good example names follow the pattern `{model}-{dataset}-{method}`, e.g.:
- `llama3-openhermes-dpo`
- `phi3-mini-gsm8k-grpo`
- `qwen2.5-vl-chartqa-vision-sft`

Each example should cover a distinct model family, dataset, or training method so the gallery remains varied and useful.
