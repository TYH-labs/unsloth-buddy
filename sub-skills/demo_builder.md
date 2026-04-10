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

For **vision models**: extract the image, convert it to a Base64 data URI string (e.g. `data:image/png;base64,...`), and set that as `"image_url"`. This ensures the HTML file remains completely zero-dependency and locally portable without broken links.

---

## Step 3: Choose Template and Theme

### 3a. Pick the template based on the user's domain

Ask yourself: **What domain or audience does this model serve?**
Do NOT base the decision on training method (SFT/DPO/GRPO).

| User domain / audience | Text-only Model Template | Vision Model Template | Default accent |
|---|---|---|---|
| Customer support / conversational chat | `demo_llm_crisp.html` | `demo_vlm_crisp.html` | `#0052cc` blue |
| Healthcare / clinical / medical NLP | `demo_llm_crisp.html` | `demo_vlm_crisp.html` | `#0891b2` cyan-teal |
| Finance / compliance / legal | `demo_llm_crisp.html` | `demo_vlm_crisp.html` | `#047857` green |
| Education / tutoring / learning tools | `demo_llm_crisp.html` | `demo_vlm_crisp.html` | `#d97706` amber |
| Code generation / DevOps / tooling | `demo_llm_dark.html` | `demo_vlm_dark.html` | `#00e5ff` electric cyan |
| Math / reasoning / science | `demo_llm_dark.html` | `demo_vlm_dark.html` | `#d4ff00` neon yellow |
| Security / threat detection / fraud | `demo_llm_dark.html` | `demo_vlm_dark.html` | `#ff4d6d` red |
| General / unspecified | `demo_llm_crisp.html` | `demo_vlm_crisp.html` | `#0052cc` blue |

**No purple.** If the domain is ambiguous, default to `crisp-light` with blue.
*(Note: If the project involves vision data—i.e., you extracted `image_url` keys in Step 2—you MUST use the `demo_vlm_*` templates, which pre-pack layout grids optimized for wide image viewing rather than just text areas!)*

### 3b. Detect language and confirm theme

Detect the language of the current conversation. Ask the user in a single confirmation message about both theme and language. Importantly, offer an option to use a popular design system from `getdesign.md`.

> "For a **[domain]** model, I recommend the **[crisp-light / dark-signal]** base theme with a **[color]** accent. The demo UI will be in **[detected language]**. I can also customize this demo using a popular design system from getdesign.md. Want a different color, theme, language, or a specific design style?"

If the user has been conversing in a non-English language (e.g. Chinese, Japanese, Spanish), default to that language for all UI strings.

Wait for their confirmation before generating.

### 3c. Fetch DESIGN.md Context (If a specific design is chosen)

If the user requests a specific design — even with abstract, conceptual, or pop-culture keywords — do not wait or ask follow-up configuration questions. Follow "show don't tell":

**Step 0 — Resolve the keyword to a brand name (required before searching).**

The search script only accepts exact brand names from the catalog. Keywords like "matrix", "cyberpunk", "apple-like", "dark terminal", or movie/game references will return nothing. Before calling the script, reason from the keyword to the best-fit brand:

| User's keyword | Reasoning | Best brand match |
|---|---|---|
| "matrix", "terminal green", "hacker", "cyberpunk" | Matrix = black bg + neon green code rain + monospace. Best match: Warp (dark terminal IDE with green/neon palette) or NVIDIA (black + neon green, GPU/AI brand). | `warp` or `nvidia` |
| "star wars", "space", "sci-fi dark" | Dark with glowing accent on black. SpaceX fits: dark, technical, monospace. | `spacex` |
| "minimal white", "apple-like", "clean premium" | High whitespace, single accent, product-as-hero. | `apple` |
| "developer", "code editor", "VS Code vibe" | Dark editor aesthetic with file-tree chrome. | `cursor` |
| "startup", "SaaS", "clean dark" | Modern SaaS dark + subtle purple or blue. | `vercel` or `linear.app` |
| "fintech", "crypto", "exchange" | Dark with teal/blue, trust signals. | `kraken` or `coinbase` |
| "AI product", "model card" | Clean AI brand with gradient or dark theme. | `mistral.ai`, `cohere`, or `anthropic` → `claude` |

If the keyword doesn't map cleanly to one brand, pick the best single match and proceed — do not ask for confirmation.

1. Run the design search script with the resolved brand name:
   ```bash
   python scripts/search_design.py "<resolved_brand_name>"
   ```
2. Download the `DESIGN.md` using the official CLI command:
   ```bash
   npx getdesign@latest add <exact_brand_name>
   ```
   *(If `npx` is unavailable, use `python scripts/search_design.py "<exact_brand_name>" --fetch > DESIGN.md` as a fallback).*
3. Read the generated `DESIGN.md` to deeply understand that design's visual identity (fonts, colors, borders, shadows, spacing).

### 3d. Build UI strings for the chosen language

Use the table below for the target language. For languages not listed, translate from the English column — keep strings short and natural.

| Placeholder | English (default) | 简体中文 | 繁體中文 | 日本語 |
|---|---|---|---|---|
| `{{LANG}}` | `en` | `zh-Hans` | `zh-Hant` | `ja` |
| `{{UI_BADGE}}` | Fine-tuning Demo | 微调演示 | 微調展示 | ファインチューニング デモ |
| `{{UI_SELECT_PROMPT}}` | Select a Prompt | 选择提示词 | 選擇提示詞 | プロンプトを選択 |
| `{{UI_PROMPT}}` | Prompt | 提示词 | 提示詞 | プロンプト |
| `{{UI_MODEL_OUTPUTS}}` | Model Outputs | 模型输出 | 模型輸出 | モデル出力 |
| `{{UI_TRAINING_DETAILS}}` | Training Details | 训练详情 | 訓練詳情 | 学習の詳細 |
| `{{UI_QUALITY_POSITIVE}}` | ✦ Improved | ✦ 改善 | ✦ 改善 | ✦ 改善 |
| `{{UI_QUALITY_NEUTRAL}}` | ≈ Comparable | ≈ 相当 | ≈ 相當 | ≈ 同等 |
| `{{UI_QUALITY_NEGATIVE}}` | ↓ Regression | ↓ 回退 | ↓ 回退 | ↓ 低下 |

The `{{MODEL_NAME}}`, `{{MODEL_DESCRIPTION}}`, metric labels, and example outputs should also be written in the target language when it differs from English. The footer is always English — do not translate it.

---

## Step 4: Build the Accent Override CSS and Mockup

Once you have the `DESIGN.md` constraints (or default rules if no specific design was chosen), immediately build the `demos/<project-name>/index.html` mockup so the user can review a functioning UI. Do not just describe the CSS changes—generate the file!

**If using the default layout:**
Construct a small CSS override to inject into the template for the chosen accent color.

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

**If using a fetched DESIGN.md:**
Decide first whether the design is a **shallow override** (accent color + font swap) or a **deep override** (different page structure, background scheme, or border-radius system).

- **Shallow** — inject into `{{INJECT_ACCENT_OVERRIDE}}`: override `--font-display`, `--font-body`, `--bg-gradient`, `--card-bg`, `--text-main`, `--border-radius`, `--shadow-card`, and load any needed fonts via `@import` at the top of the block.
- **Deep** (e.g. NVIDIA-style all-black layout, Apple black-hero + light-content split) — write the output file from scratch. The template's single injection point cannot handle structural changes. Skip straight to Step 5 and build the full HTML directly.

Replace the `/* {{INJECT_ACCENT_OVERRIDE}} */` comment in the template with your CSS, or skip it entirely if doing a full rewrite.

---

## Step 5: Fill in the Template Placeholders

Read the chosen template from `templates/demo_llm_crisp.html` (or its `_dark` / `_vlm` variants) and perform these substitutions:

| Placeholder | Value |
|---|---|
| `{{LANG}}` | BCP 47 language tag, e.g. `en`, `zh-Hans`, `zh-Hant`, `ja` |
| `{{MODEL_NAME}}` | Short display name, e.g. `"Qwen2.5-0.5B · Chip2 SFT"` |
| `{{MODEL_DESCRIPTION}}` | One-line description (in target language) |
| `{{BASE_MODEL_NAME}}` | e.g. `"Qwen2.5-0.5B-Instruct (base)"` |
| `{{FINETUNED_MODEL_NAME}}` | e.g. `"Qwen2.5-0.5B + chip2-sft LoRA"` |
| `{{INJECT_EXAMPLES_JSON}}` | Replace with `const examples = [ ... ];` — the hardcoded JSON array |
| `{{INJECT_METRICS}}` | Replace with `<li>` chips (see format below) |
| `{{INJECT_ACCENT_OVERRIDE}}` | Replace with the CSS accent block from Step 4 |
| `{{UI_BADGE}}` | Header badge text — from Step 3c language table |
| `{{UI_SELECT_PROMPT}}` | Dropdown label — from Step 3c language table |
| `{{UI_PROMPT}}` | Prompt section heading — from Step 3c language table |
| `{{UI_MODEL_OUTPUTS}}` | Outputs section heading — from Step 3c language table |
| `{{UI_TRAINING_DETAILS}}` | Metrics section heading — from Step 3c language table |
| `{{UI_QUALITY_POSITIVE}}` | Quality badge: improved — from Step 3c language table |
| `{{UI_QUALITY_NEUTRAL}}` | Quality badge: comparable — from Step 3c language table |
| `{{UI_QUALITY_NEGATIVE}}` | Quality badge: regression — from Step 3c language table |

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

For **vision inputs**, explicitly convert local testing images into Base64 data URIs so the HTML remains completely zero-dependency and offline-ready:
```js
{
    "type": "image",
    "image_url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD...",
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
- [ ] Domain inferred → theme + accent selected → language confirmed → user approved
- [ ] Accent CSS override built
- [ ] All placeholders substituted (no `{{...}}` strings remaining in final file)
- [ ] File written to `demos/<project-name>/index.html`
- [ ] `gaslamp.md` § 9 updated
