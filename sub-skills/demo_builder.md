# Sub-skill: Demo Builder

Generate a static, zero-dependency HTML demo page that showcases **base model vs. fine-tuned model** outputs side-by-side. The page runs on any machine without a server.

---

## When to invoke

Invoke this sub-skill after **Phase 5 (Evaluation)** is complete — specifically once `eval.py --compare` has produced side-by-side output pairs. Skip if the user says they don't need a demo.

---

## Step 1: Gather Data from `gaslamp.md`

Read the project's `gaslamp.md` and extract:

| Field | Where in gaslamp.md |
|-------|-------------------|
| Project goal / domain | § 1 Goal |
| Base model name | § 3 Model |
| Fine-tuned model description | § 3 Model + § 7 Training Outcome |
| Training method | § 2 Method |
| Key metrics (eval loss, steps, dataset, etc.) | § 8 Evaluation + § 6 Hyperparameters |

If `gaslamp.md` is missing a field, ask the user directly.

---

## Step 2: Precompute 4–6 Example Pairs

Run the eval script in compare mode to generate output pairs. This is already covered by Phase 5 (saved to `logs/eval_compare.log`), so reuse those outputs. If compare mode wasn't run, execute:

```bash
python eval.py --compare 2>&1 | tee logs/eval_compare.log
```

Select **4–6 diverse prompts** that best showcase the model's improved capability. Include at least:
- 2–3 examples where the fine-tuned model is clearly better (`highlight_class: "positive"`)
- 1–2 neutral examples showing comparable quality (`highlight_class: "neutral"`)
- Optionally 1 honest regression if one exists (`highlight_class: "negative"`) — this builds trust

For **vision models**: collect the image URL/path alongside the prompt and set `"type": "image"`.

---

## Step 3: Choose Template and Theme

### 3a. Pick the template based on the user's domain

Ask yourself: **What domain or audience does this model serve?**
Do NOT base the decision on training method (SFT/DPO/GRPO).

| User domain / audience | Template | Default accent |
|---|---|---|
| Customer support / conversational chat | `demo_llm_crisp.html` | `#0052cc` blue |
| Healthcare / clinical / medical NLP | `demo_llm_crisp.html` | `#0891b2` cyan-teal |
| Finance / compliance / legal | `demo_llm_crisp.html` | `#047857` green |
| Education / tutoring / learning tools | `demo_llm_crisp.html` | `#d97706` amber |
| Code generation / DevOps / tooling | `demo_llm_dark.html` | `#00e5ff` electric cyan |
| Math / reasoning / science | `demo_llm_dark.html` | `#d4ff00` neon yellow |
| Security / threat detection / fraud | `demo_llm_dark.html` | `#ff4d6d` red |
| General / unspecified | `demo_llm_crisp.html` | `#0052cc` blue |

**No purple.** If the domain is ambiguous, default to `crisp-light` with blue.

### 3b. State your choice and invite override

Tell the user:
> "For a **[domain]** model, I'll use the **[crisp-light / dark-signal]** theme with **[color]** accent — feels [trustworthy / bold / technical]. Want a different color or theme?"

Wait for their confirmation before generating.

---

## Step 4: Build the Accent Override CSS

For the chosen accent color, construct a small CSS override to inject into the template:

**crisp-light variant:**
```css
/* domain accent override */
:root {
    --accent: #047857;
    --accent-light: rgba(4, 120, 87, 0.08);
    --accent-mid: rgba(4, 120, 87, 0.18);
}
```

**dark-signal variant:**
```css
/* domain accent override */
:root {
    --accent: #00e5ff;
    --accent-glow: rgba(0, 229, 255, 0.12);
    --accent-glow-strong: rgba(0, 229, 255, 0.22);
    --card-border-hover: rgba(0, 229, 255, 0.18);
}
```

Replace the `/* {{INJECT_ACCENT_OVERRIDE}} */` comment in the template with this block.

---

## Step 5: Fill in the Template Placeholders

Read the chosen template from `templates/demo_llm_crisp.html` (or `demo_llm_dark.html`) and perform these substitutions:

| Placeholder | Value |
|---|---|
| `{{MODEL_NAME}}` | Short display name, e.g. `"Qwen2.5-0.5B · Chip2 SFT"` |
| `{{MODEL_DESCRIPTION}}` | One-line description, e.g. `"Instruction-following model fine-tuned on OpenHermes chip2 dataset."` |
| `{{BASE_MODEL_NAME}}` | e.g. `"Qwen2.5-0.5B-Instruct (base)"` |
| `{{FINETUNED_MODEL_NAME}}` | e.g. `"Qwen2.5-0.5B + chip2-sft LoRA"` |
| `{{INJECT_EXAMPLES_JSON}}` | Replace with `const examples = [ ... ];` — the hardcoded JSON array |
| `{{INJECT_METRICS}}` | Replace with `<li>` chips (see format below) |
| `{{INJECT_ACCENT_OVERRIDE}}` | Replace with the CSS accent block from Step 4 |

### Examples JSON format

```js
const examples = [
    {
        "prompt": "Explain gradient descent in one sentence.",
        "base_output": "Gradient descent is an optimization algorithm.",
        "finetuned_output": "Gradient descent iteratively adjusts model parameters in the direction that minimizes the loss function, using the negative gradient as a guide.",
        "highlight_class": "positive"
    },
    {
        "prompt": "What is overfitting?",
        "base_output": "Overfitting is when a model learns the training data too well.",
        "finetuned_output": "Overfitting occurs when a model captures noise in training data rather than the underlying pattern, causing poor generalization to unseen examples.",
        "highlight_class": "positive"
    }
];
```

For **vision inputs**, use:
```js
{
    "type": "image",
    "image_url": "https://example.com/image.jpg",
    "image_caption": "A chest X-ray showing ...",
    "base_output": "I see a medical image.",
    "finetuned_output": "The chest X-ray shows a right lower lobe opacity consistent with pneumonia.",
    "highlight_class": "positive"
}
```

### Metrics `<li>` format

```html
<li><strong>87.3</strong>eval loss</li>
<li><strong>SFT</strong>method</li>
<li><strong>200 steps</strong>training</li>
<li><strong>Qwen2.5-0.5B</strong>base model</li>
<li><strong>chip2 (200k)</strong>dataset</li>
```

---

## Step 6: Write the Output File

Write the filled template to:

```
demos/<project-name>/index.html
```

Where `<project-name>` is the project directory name (e.g. `qwen2.5-0.5b-chip2-sft`).

Create the directory if it doesn't exist:
```bash
mkdir -p demos/<project-name>
```

---

## Step 7: Report to the User

After writing the file, tell the user:

> "Demo generated: `demos/<project-name>/index.html`
> Open it in any browser — no server needed.
> Theme: **[crisp-light / dark-signal]** · Accent: **[color]** · [N] example pairs"

Then update `gaslamp.md` **§ 9 File Inventory** with:
```
| demos/<project-name>/index.html | Demo | generated by demo_builder sub-skill |
```

---

## Checklist

- [ ] `gaslamp.md` read and fields extracted
- [ ] 4–6 example pairs precomputed from `eval.py --compare`
- [ ] Domain inferred → theme + accent selected → user confirmed
- [ ] Accent CSS override built
- [ ] All 7 placeholders substituted (no `{{...}}` strings remaining in final file)
- [ ] File written to `demos/<project-name>/index.html`
- [ ] `gaslamp.md` § 9 updated
