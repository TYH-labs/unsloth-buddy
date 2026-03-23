# gaslamp.md — Roadbook for `qwen_chip2_sft`
# Powered by Gaslamp · https://gaslamp.dev
#
# PURPOSE: Every key decision made and kept, with the reasoning behind it and
# a brief conceptual warmup. This file alone should let another person (or agent)
# reproduce the project end-to-end — and understand *why* each choice was made.
#
# RULES:
#   ✅ Record only decisions that were kept (right choices)
#   ✅ Include WHY this choice was right for this project
#   ✅ Include a 📖 Learn block explaining the underlying concept + alternatives
#   ❌ Do NOT log failed attempts, timestamps, or session notes (that is memory.md)
#   ❌ Do NOT leave placeholder text — only fill in what is actually decided

---

## 1. Goal

- **Task:** SFT fine-tune Qwen2.5-0.5B-Instruct on the OIG unified_chip2 human/bot conversation dataset to produce a compact, local instruction-following model.
- **Success looks like:** Fine-tuned model gives more direct, helpful responses to common instruction prompts than the base model, with measurably lower eval loss and clearly better qualitative outputs on held-out prompts.

---

## 2. Method

**Chosen:** SFT (Supervised Fine-Tuning)
**Why for this project:** The unified_chip2 dataset contains 200k+ labeled `<human>` / `<bot>` conversation pairs — every example has a clear ground-truth response. SFT is the right fit when you have labeled input→output pairs and want the model to learn that specific response style.

> 📖 **Learn — Choosing a training method:**
> SFT (Supervised Fine-Tuning) trains the model on labeled input→output pairs using
> cross-entropy loss — the right default when your data has a ground-truth answer.
> DPO trains on preference pairs (chosen vs rejected response) with no reward model needed.
> GRPO uses a reward function (e.g. code correctness, format check) to optimize via RL —
> best when correctness is verifiable but hard to express as labeled pairs.
> Rule of thumb: if you have answers → SFT; if you have rankings → DPO; if you have a
> verifier → GRPO.

---

## 3. Model

**Chosen:** `Qwen/Qwen2.5-0.5B-Instruct`
**Why for this project:** Smallest Qwen2.5 Instruct variant. At 0.5B parameters it trains in ~2 minutes on Apple Silicon and fits comfortably in unified memory. The task (general instruction-following on chip2) is narrow and well-patterned — a 0.5B model has enough capacity. The Instruct variant already understands ChatML format, reducing the formatting mismatch with training data.

**Quantization:** 4-bit QLoRA (`load_in_4bit=True`)
**Why:** Halves memory footprint vs float16 on MPS. On a 0.5B model the quality loss from 4-bit is negligible for a narrow fine-tuning task.

> 📖 **Learn — Model size and quantization:**
> Smaller models (0.5B–3B) train faster, use less VRAM, and work well for narrow tasks
> with clear patterns. Larger models (7B–70B) have more world knowledge but need more
> hardware. QLoRA 4-bit lets you fine-tune a 7B model on ~8 GB VRAM by quantizing frozen
> weights while keeping the LoRA adapter in full precision. 8-bit and 16-bit trade VRAM
> for quality. Full fine-tuning updates all weights — rarely needed unless the task is
> very different from the pre-training distribution.

**LoRA config:**
- Rank (r): 8
- Alpha (α): 16 (= 2× rank — conventional scaling)
- Target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- Dropout: 0 (standard for fine-tuning; dropout hurts more than it helps at this scale)

> 📖 **Learn — LoRA rank and alpha:**
> LoRA adds two small trainable matrices (A and B, each of rank r) alongside frozen
> weights, so the weight update is ΔW = BA. Rank controls expressivity: r=8 is a good
> default for most tasks; go higher (32–64) if the task requires broad behavioural change.
> Alpha (α) is a scaling factor: effective update scale = α/r. Setting α = 2r is
> conventional. Target modules are the weight matrices that receive LoRA adapters;
> attention projections (q, k, v, o) plus MLP gates (gate_proj, up_proj, down_proj) is
> the recommended default for causal LLMs.

---

## 4. Data

**Source:** `laion/OIG` — `unified_chip2.jsonl` via HuggingFace
**URL:** `https://huggingface.co/datasets/laion/OIG/resolve/main/unified_chip2.jsonl`
**Format:** chat (`messages` column — OpenAI format)
**Size:** 199,774 train / 10,515 val (95/5 split, seed=42)

**Prompt template / schema:**
```
Input (raw):  {"text": "<human>: {question}\n<bot>: {answer}"}

Output (reformatted):
{
  "messages": [
    {"role": "user",      "content": "{question}"},
    {"role": "assistant", "content": "{answer}"}
  ]
}
```

**Key formatting decision:** Parsed `<human>:` / `<bot>:` delimiters from the raw `text` field into OpenAI chat format (`messages` column). 0 rows were malformed or skipped. No system prompt added — chip2 responses are direct answers that do not need a persona.

> 📖 **Learn — Why data format matters:**
> TRL's SFTTrainer expects a `messages` column in OpenAI chat format
> (list of {role, content} dicts). If your data is in a different shape, you must
> reformat before training — the trainer will silently drop mismatched rows otherwise.
> The chat template (applied by the tokenizer) adds special tokens around each turn.
> Using the model's native template (e.g. ChatML for Qwen, Llama-3 for Llama) is
> critical: mismatched templates cause the model to learn the wrong stop-token behaviour.

---

## 5. Environment

- **Hardware:** Apple Silicon (M-series Mac, unified memory)
- **Backend:** mlx-tune 0.4.2
- **Python:** 3.12.x
- **venv path:** `.venv/` (create with `python3.12 -m venv .venv`)
- **Key packages:** mlx-tune=0.4.2, transformers=5.3.0, datasets=4.7.0

---

## 6. Hyperparameters

| Parameter | Value | Why |
|-----------|-------|-----|
| Learning rate | 2e-4 | Standard for LoRA on a narrow task; higher than full FT because only adapter weights update |
| Batch size | 2 | MPS memory limit |
| Gradient accumulation | N/A (mlx-tune) | mlx-tune manages internally |
| Effective batch size | 2 | Small but sufficient for this dataset size |
| Steps / epochs | 200 iters | Short validation run; 2000+ iters recommended for production quality |
| LR scheduler | cosine (mlx-tune default) | Avoids sharp LR drop; stable on short runs |
| Warmup steps | mlx-tune default | |
| Max seq length | 512 | chip2 examples are short; 95th percentile well under 512 tokens |
| Optimizer | Adam (mlx-tune internal) | mlx-tune manages optimizer internally |

> 📖 **Learn — Key hyperparameter intuitions:**
> **Learning rate**: LoRA fine-tuning typically uses 1e-4 to 5e-4 (higher than full FT
> because fewer params update). Too high → loss spikes; too low → slow convergence.
> **Effective batch size** = batch_size × grad_accum × num_GPUs. Larger = more stable
> gradients but slower steps. Accumulate gradients when GPU VRAM limits physical batch
> size.
> **Cosine scheduler with warmup** is the standard: warmup stabilises early training
> when weights are far from optimum, cosine decay avoids overshooting at the end.
> **Max seq length** should match your data's 95th-percentile token length — longer
> wastes VRAM on padding; shorter truncates examples and loses information.

---

## 7. Training Outcome

**Canonical run — `train.py` (mlx-tune, use this to reproduce):**
- **Final loss:** train 2.704 → 1.249, val ~3.1 → 1.25 over 200 iters on full 199k dataset
- **Runtime:** ~3–4 min on Apple Silicon
- **Peak memory:** ~2.1 GB unified memory

---

## 8. Evaluation

**Method:** `--compare` mode (base vs fine-tuned, greedy decoding, side-by-side)

**Prompts tested:**
```
1. What are the best practices for writing clean code?
2. Explain the difference between RAM and ROM.
3. Give me three tips for staying productive while working from home.
4. What is the capital of Japan and what is it known for?
5. How do I make a good cup of coffee?
```

**Results:**
| Prompt | Base model | Fine-tuned |
|--------|-----------|------------|
| "Explain the difference between RAM and ROM." | Broad, sometimes off-topic | Concise, direct definition with clear contrast |
| "How do I make a good cup of coffee?" | Generic advice | Step-by-step with actionable detail |
| "Give me three tips for staying productive while working from home." | Verbose, hedged | Numbered list, direct and actionable |

**Verdict:** Fine-tuned model is consistently more direct and instruction-aligned than the base model on chip2-style prompts. Loss curve was stable (2.7 → 1.2, no spikes). A longer run (2000+ iters on the full 199k dataset) would improve quality further.

> 📖 **Learn — How to evaluate a fine-tuned model:**
> Loss tells you about training fit, not task quality. Always run qualitative tests on
> held-out prompts the model has never seen. Compare base vs fine-tuned on the same
> prompt — the delta is your signal. For instruction-following tasks, look for format
> compliance and factual accuracy. For GRPO/DPO, also check that the model is not
> reward-hacking (giving technically correct but useless outputs). A low eval loss with
> poor qualitative outputs usually means the prompt template is wrong.

---

## 9. File Inventory

| File | Source | Role |
|------|--------|------|
| `train.py` | copied from `scripts/unsloth_mlx_sft_example.py` | mlx-tune SFT script — 200 iters, full 199k dataset |
| `eval.py` | copied from `scripts/mlx_eval_template.py` | Evaluation: `--compare` base-vs-tuned, `--interactive` REPL, batch mode |
| `prepare_data.py` | write from scratch per § 4 | Downloads chip2.jsonl, reformats `<human>/<bot>` → `messages`, 95/5 split seed=42 |
| `mlx_gaslamp_dashboard.py` | copied from `scripts/mlx_gaslamp_dashboard.py` | Live dashboard context manager (mlx-tune path) |
| `templates/dashboard.html` | copied from `templates/dashboard.html` | Web dashboard UI |
| `data/train.jsonl` | generated by `prepare_data.py` | 199,774 formatted training rows |
| `data/val.jsonl` | generated by `prepare_data.py` | 10,515 formatted validation rows |
| `outputs/adapters/` | output of `train.py` | LoRA adapter weights |

**To reproduce from scratch:**
1. Copy scripts from skill root: `train.py` ← `scripts/unsloth_mlx_sft_example.py`, `eval.py` ← `scripts/mlx_eval_template.py`, `mlx_gaslamp_dashboard.py`, `templates/dashboard.html`
2. Write `prepare_data.py` from scratch using § 4 (URL, parse logic, 95/5 split seed=42)
3. `python prepare_data.py` → generates `data/`
4. Edit `train.py` config block: `MODEL_NAME`, `MAX_SEQ_LENGTH=512`, `LORA_RANK=8`, `LORA_ALPHA=16`, `ITERS=200`, `BATCH_SIZE=2`, `LEARNING_RATE=2e-4`
5. `python train.py` → generates `outputs/adapters/`
6. Edit `eval.py`: set `MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"`, `ADAPTER_PATH = "outputs/adapters"`, `TEMPERATURE = 0.0`
7. `python eval.py --compare` → verify base vs fine-tuned delta

---

## 10. Export

**Format:** Adapter-only (no merge or GGUF — local use only for this run)
**Why this format:** Adapters alone are sufficient for local inference with mlx-tune. No Hub push or GGUF needed for this validation run.
**Output path:** `outputs/adapters/`
**Run command (load + generate):**
```python
from mlx_tune import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name   = "Qwen/Qwen2.5-0.5B-Instruct",
    adapter_path = "outputs/adapters",   # full relative path — NOT the trainer's "adapters" shorthand
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

prompt = "<human>: Explain the difference between RAM and ROM.\n<bot>:"
print(model.generate(prompt=prompt, max_tokens=200))
```

> 📖 **Learn — Export formats:**
> **GGUF** (llama.cpp format) is for local CPU/GPU inference via Ollama, LM Studio, or
> llama.cpp. q4_K_M is the best quality/size tradeoff for most use cases.
> **Merged 16-bit safetensors** combines the LoRA adapter back into the base model
> weights — needed for vLLM serving or HuggingFace Hub deployment.
> **Adapter-only** (without merging) is the smallest option but requires the base model
> to be loaded separately at inference time — fine for local mlx-tune use.

---

## 11. Workarounds & Critical Notes

- **mlx-tune `adapter_path` double-nesting (train.py):** In the mlx-tune trainer config, set `adapter_path="adapters"` (not `"outputs/adapters"`). mlx-tune prepends `output_dir` automatically, so `"adapters"` → `outputs/adapters/`. Setting `"outputs/adapters"` silently produces `outputs/outputs/adapters/`, which breaks eval loading.

- **eval.py `ADAPTER_PATH` uses the full path, not the trainer shorthand:** In `eval.py`, set `ADAPTER_PATH = "outputs/adapters"` (the actual directory on disk). Do NOT use `"adapters"` — that shorthand only works inside the mlx-tune trainer config. `FastLanguageModel.from_pretrained(adapter_path=...)` expects the real relative path.

- **Qwen2.5 tokenizer has no `pad_token`:** If using TRL's SFTTrainer on MPS, set `tokenizer.pad_token = tokenizer.eos_token` before training. mlx-tune handles this internally; TRL does not.

- **`torch_dtype` deprecation in transformers ≥ 5.x:** Use `dtype=torch.float16` instead of `torch_dtype=torch.float16` in `AutoModelForCausalLM.from_pretrained`. The old kwarg still works but emits a deprecation warning.
