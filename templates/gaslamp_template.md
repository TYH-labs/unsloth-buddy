# gaslamp.md — Roadbook for `{project_name}`
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

- **Task:**
- **Success looks like:**

---

## 2. Method

**Chosen:** <!-- SFT | DPO | GRPO | Vision SFT | ORPO | KTO -->
**Why for this project:**

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

**Chosen:** <!-- e.g. Qwen/Qwen2.5-0.5B-Instruct -->
**Why for this project:**

**Quantization:** <!-- 4-bit QLoRA | 8-bit | 16-bit LoRA | full fine-tune -->
**Why:**

> 📖 **Learn — Model size and quantization:**
> Smaller models (0.5B–3B) train faster, use less VRAM, and work well for narrow tasks
> with clear patterns. Larger models (7B–70B) have more world knowledge but need more
> hardware. QLoRA 4-bit lets you fine-tune a 7B model on ~8 GB VRAM by quantizing frozen
> weights while keeping the LoRA adapter in full precision. 8-bit and 16-bit trade VRAM
> for quality. Full fine-tuning updates all weights — rarely needed unless the task is
> very different from the pre-training distribution.

**LoRA config:**
- Rank (r): <!-- typical: 8–64 -->
- Alpha (α): <!-- typical: 2× rank -->
- Target modules: <!-- e.g. q_proj, k_proj, v_proj, o_proj -->
- Dropout: <!-- 0 is standard for fine-tuning -->

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

**Source:** <!-- HF dataset ID, URL, or local path -->
**Format:** <!-- chat (messages column) | completion | DPO pairs | GRPO prompt -->
**Size:** <!-- N train rows / M val rows -->

**Prompt template / schema:**
```
<!-- paste the exact template used, e.g.:
<|im_start|>system
You are a helpful assistant.
<|im_end|>
<|im_start|>user
{input}
<|im_end|>
<|im_start|>assistant
{output}<|im_end|>
-->
```

**Key formatting decision:**
<!-- e.g. "Used system prompt to enforce JSON output format",
     "Filtered rows shorter than 10 tokens",
     "Mapped 'question' → 'input', 'answer' → 'output'" -->

> 📖 **Learn — Why data format matters:**
> TRL's SFTTrainer expects a `messages` column in OpenAI chat format
> (list of {role, content} dicts). If your data is in a different shape, you must
> reformat before training — the trainer will silently drop mismatched rows otherwise.
> The chat template (applied by the tokenizer) adds special tokens around each turn.
> Using the model's native template (e.g. ChatML for Qwen, Llama-3 for Llama) is
> critical: mismatched templates cause the model to learn the wrong stop-token behaviour.

---

## 5. Environment

- **Hardware:** <!-- e.g. Apple M4 Pro 24 GB unified memory | NVIDIA A100 40 GB -->
- **Backend:** <!-- mlx-tune | unsloth | colab (T4 / L4 / A100) -->
- **Python:** <!-- e.g. 3.12.x -->
- **venv path:** <!-- e.g. .venv/ -->
- **Key packages:** <!-- transformers=X, trl=X, peft=X, mlx-tune=X / unsloth=X -->

---

## 6. Hyperparameters

<!-- Fill in the final values that produced a good result — not the first attempt -->

| Parameter | Value | Why |
|-----------|-------|-----|
| Learning rate | | |
| Batch size | | |
| Gradient accumulation | | |
| Effective batch size | | |
| Steps / epochs | | |
| LR scheduler | | |
| Warmup steps | | |
| Max seq length | | |
| Optimizer | | |

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

<!-- If multiple scripts exist (e.g. a canonical training script + a dashboard test script),
     label each run separately. Only the canonical run is needed to reproduce. -->

**Canonical run — `train.py`:**
- **Final loss:** <!-- e.g. train 2.7 → 1.2, val 3.1 → 1.25 -->
- **Runtime:** <!-- e.g. 3–4 min -->
- **Peak VRAM / memory:** <!-- e.g. 2.1 GB -->

<!-- Add additional rows only if other scripts produce results worth recording. -->

---

## 8. Evaluation

**Method:** <!-- qualitative REPL | batch test | benchmark (e.g. MMLU, HumanEval) -->

**Prompts tested:**
```
<!-- paste 1–3 representative test prompts -->
```

**Results:**
| Prompt | Base model | Fine-tuned |
|--------|-----------|------------|
| | | |

**Verdict:** <!-- e.g. "Fine-tuned correctly follows JSON format; base model ignores it" -->

> 📖 **Learn — How to evaluate a fine-tuned model:**
> Loss tells you about training fit, not task quality. Always run qualitative tests on
> held-out prompts the model has never seen. Compare base vs fine-tuned on the same
> prompt — the delta is your signal. For instruction-following tasks, look for format
> compliance and factual accuracy. For GRPO/DPO, also check that the model is not
> reward-hacking (giving technically correct but useless outputs). A low eval loss with
> poor qualitative outputs usually means the prompt template is wrong.

---

## 9. File Inventory

<!-- Include a Source column so a reproducing agent knows where each file came from.
     "copied from X" = copy this file from the skill root.
     "custom" = written from scratch for this project.
     "generated by Y" = produced by running script Y — do not copy, re-run Y instead. -->

| File | Source | Role |
|------|--------|------|
| `train.py` | copied from `scripts/unsloth_mlx_sft_example.py` | Training script |
| `eval.py` | copied from `scripts/mlx_eval_template.py` | Evaluation script |
| `data/train.jsonl` | generated by `prepare_data.py` | Formatted training data |
| `outputs/adapters/` | output of `train.py` | LoRA adapter weights |

<!-- Add rows for extra files, e.g.:
| `gaslamp_callback.py`      | copied from `scripts/gaslamp_callback.py`      | Live dashboard callback (NVIDIA/TRL)          |
| `mlx_gaslamp_dashboard.py` | copied from `scripts/mlx_gaslamp_dashboard.py` | Live dashboard context manager (Apple Silicon) |
| `templates/dashboard.html` | copied from `templates/dashboard.html`         | Web dashboard UI                              |
| `demos/<name>/index.html`  | generated by demo_builder sub-skill            | Static HTML demo (base vs fine-tuned outputs) |
-->

**To reproduce from scratch:** copy scripts from the skill root (Source column). Re-run data scripts to regenerate `data/` and `outputs/` — do not copy them.

---

## 10. Export

**Format:** <!-- GGUF q4_k_m | merged 16-bit safetensors | adapter-only | HuggingFace Hub push -->
**Why this format:**
**Output path:** <!-- e.g. outputs/adapters/ | outputs/model-q4_k_m.gguf -->
**Run command (load + generate):** <!-- paste the full load-and-generate snippet, not just the load call -->

> 📖 **Learn — Export formats:**
> **GGUF** (llama.cpp format) is for local CPU/GPU inference via Ollama, LM Studio, or
> llama.cpp. q4_K_M is the best quality/size tradeoff for most use cases.
> **Merged 16-bit safetensors** combines the LoRA adapter back into the base model
> weights — needed for vLLM serving or HuggingFace Hub deployment.
> **Adapter-only push** (without merging) is the smallest upload but requires the base
> model to be loaded separately at inference time.

---

## 11. Workarounds & Critical Notes

<!-- Only non-obvious things a reproducing agent MUST know.
     Format: what the issue is + the exact fix that worked. -->
