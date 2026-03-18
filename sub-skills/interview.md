# Sub-skill: Interview & Requirements Gathering (Unsloth)

**Goal**: Transform vague user requests into a structured **Project Brief**. This brief acts as the "contract" for the remainder of the Unsloth 6-Phase Lifecycle, defining the exact model, task, and constraints before any code is written or data is prepared.

---

## 1. The 2-Question Interview

Instead of interrogating the user with a long list of questions, simply ask these two fundamental things:

### A. The Task Definition
- "What exactly are we trying to teach the model to do?"
- Based on their answer, map it to a TRL method internally:
    - Standard instruction following → **SFT**
    - Analyzing images/video → **Vision SFT**
    - Aligning with human preference (better/worse pairs) → **DPO / ORPO**
    - Reasoning, math, or complex problem solving → **GRPO**

### B. The Data Status
- "Do you already have a dataset prepared for this task, or do we need to hunt for one/generate one synthetically?"

---

## 2. Bucket Recommendations & Cost Estimation

Once you know their task, **do not ask them to pick a model or hardware**. Instead, proactively recommend one of these three implementation "Buckets" based on their assumed complexity, and state the estimated costs. Let them confirm or adjust.

### Tier 1: Testing, Edge Devices, or Simple Tasks
Use this if the task is simple (e.g. text classification, basic chat) or if the user specifically mentions running on a Mac, an old PC, or free Colab.
- **Recommended Models (0.5B - 3B)**: `Qwen2.5-0.5B`, `Qwen2.5-1.5B`, `Llama-3.2-1B`
- **Hardware Needed**: Local Mac/PC (8GB+ unified memory) or Free Google Colab (T4 16GB).
- **Estimated Cost**: **$0.00** (Free).

### Tier 2: Standard Production & Capable Chatbots
Use this as the **default recommendation** for most SFT and DPO tasks.
- **Recommended Models (7B - 9B)**: `Llama-3.1-8B`, `Qwen2.5-7B`, `Gemma-2-9B`
- **Hardware Needed**: Cloud GPU (L4 24GB or A10G 24GB) or high-end local (RTX 3090/4090, Mac M-series 32GB+).
- **Estimated Cost**: **~$0.30 - $1.00** per fine-tuning run (cheap cloud providers like RunPod/Lambda).

### Tier 3: Complex Reasoning (GRPO) & Heavy Lifting
Use this only if the user is doing complex math/coding tasks (GRPO) or specifically demands high capability.
- **Recommended Models (14B - 70B)**: `Qwen2.5-14B`, `Qwen2.5-32B`, `Llama-3.3-70B`
- **Hardware Needed**: A100 80GB (for 70B QLoRA) or multi-GPU instances.
- **Estimated Cost**: **$2.00 - $10.00+** per fine-tuning run.

---

## 3. Workflow

1. **Ask**: 
   - Ask the 2 core questions (Task & Data).

2. **Recommend**:
   - Internalize the answers and present the recommended Tier (Model + Cost + Hardware). Ask if they agree with this plan.

3. **Confirm & Log**:
   - Once they agree to a Tier, draft the `project_brief.md`.
   - "Here is my understanding of the fine-tuning project. Does this look correct?"
   - **Action**: Initialize `unsloth-buddy/progress_log.md`. Record the Problem Definition, chosen Base Model, and Target Hardware.

## 3. Output

**Mandatory Deliverable**: `project_brief.md` (saved in the project specific folder).

Only when the user confirms the Brief, transition to **Phase 2: Data Strategy**.
