"""
eval.py — General-purpose evaluation template for mlx-tune fine-tuned models.

Usage:
    python eval.py                        # batch mode with built-in test prompts
    python eval.py --interactive          # REPL chat mode
    python eval.py --compare              # side-by-side: base model vs fine-tuned
    python eval.py --style alpaca         # override prompt format

Supported prompt styles (auto-detected if not specified):
    chip2     <human>: {q}\n<bot>:          (OIG / unified_chip2)
    alpaca    ### Instruction:\n{q}\n\n### Response:
    chatml    applies model's built-in chat template via tokenizer
    raw       {q}  (direct completion, no wrapper)
"""

import argparse
import sys

from mlx_tune import FastLanguageModel
from mlx_lm.sample_utils import make_sampler

# ── Config — edit these to match your training run ──────────────────────────
MODEL_NAME   = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
ADAPTER_PATH = "outputs/adapters"   # set to None to run base model only
MAX_SEQ_LEN  = 512
MAX_TOKENS   = 200
TEMPERATURE  = 0.7                  # 0.0 = greedy / deterministic

# A few prompts representative of your training domain
TEST_PROMPTS = [
    "What is machine learning?",
    "Write a Python function that reverses a string.",
    "Give me three tips for better sleep.",
    "What is the difference between supervised and unsupervised learning?",
    "How do I center a div in CSS?",
]

# ── Prompt formatting ────────────────────────────────────────────────────────
STYLES = {
    "chip2":  {
        "wrap":    lambda q: f"<human>: {q}\n<bot>:",
        # truncate at next <human>: to prevent next-turn bleed
        "extract": lambda r: r.split("<bot>:")[-1].split("<human>:")[0].strip(),
    },
    "alpaca": {
        "wrap":    lambda q: f"### Instruction:\n{q}\n\n### Response:",
        "extract": lambda r: r.split("### Response:")[-1].split("### Instruction:")[0].strip(),
    },
    "raw": {
        "wrap":    lambda q: q,
        "extract": lambda r: r.strip(),
    },
    # chatml: uses the tokenizer's apply_chat_template — handled separately
}

def detect_style(adapter_path: str) -> str:
    """
    Guess the prompt style from the training data file saved alongside adapters.
    Falls back to 'chip2' if it can't be determined.
    """
    import os, json
    train_file = os.path.join(os.path.dirname(adapter_path), "train.jsonl") if adapter_path else None
    if train_file and os.path.exists(train_file):
        with open(train_file) as f:
            first = json.loads(f.readline())
        text = first.get("text", "")
        if "<human>:" in text:
            return "chip2"
        if "### Instruction:" in text:
            return "alpaca"
        if first.get("messages"):
            return "chatml"
    return "chip2"  # sensible default for OIG-style datasets

def build_prompt(question: str, style: str, tokenizer) -> str:
    if style == "chatml":
        messages = [{"role": "user", "content": question}]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return STYLES[style]["wrap"](question)

def extract_response(full_text: str, prompt: str, style: str) -> str:
    """Strip the echoed prompt from the generated text, then clean up."""
    # mlx_lm returns the full sequence including prompt — remove it
    if full_text.startswith(prompt):
        full_text = full_text[len(prompt):]
    # Also apply style-aware extraction as a fallback
    if style in STYLES:
        full_text = STYLES[style]["extract"](full_text)
    return full_text.strip()

def load_model(adapter_path, load_adapter=True, style="chip2"):
    kwargs = {}
    if load_adapter and adapter_path:
        kwargs["adapter_path"] = adapter_path

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = MODEL_NAME,
        max_seq_length = MAX_SEQ_LEN,
        load_in_4bit   = True,
        **kwargs,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def run_batch(model, tokenizer, style, prompts=TEST_PROMPTS, temperature=TEMPERATURE, max_tokens=MAX_TOKENS):
    sampler = make_sampler(temp=temperature) if temperature > 0 else None
    print(f"\n{'='*60}")
    print(f"BATCH EVAL  |  style={style}  |  adapter={ADAPTER_PATH}")
    print(f"{'='*60}")
    for q in prompts:
        prompt = build_prompt(q, style, tokenizer)
        raw = model.generate(prompt=prompt, max_tokens=max_tokens, sampler=sampler)
        answer = extract_response(raw, prompt, style)
        print(f"\nQ: {q}")
        print(f"A: {answer}")
        print("-" * 60)

def run_interactive(model, tokenizer, style):
    sampler = make_sampler(temp=TEMPERATURE) if TEMPERATURE > 0 else None
    print(f"\nInteractive mode  (style={style})  — type 'quit' to exit\n")
    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("quit", "exit", "q"):
            break
        if not q:
            continue
        prompt = build_prompt(q, style, tokenizer)
        raw = model.generate(prompt=prompt, max_tokens=MAX_TOKENS, sampler=sampler)
        print(f"Bot: {extract_response(raw, prompt, style)}\n")

def run_compare(tokenizer, style, prompts=TEST_PROMPTS):
    """Run same prompts through base model and fine-tuned model, print side-by-side."""
    sampler = make_sampler(temp=0.0)  # greedy for fair comparison
    print(f"\n{'='*60}")
    print(f"BASE vs FINE-TUNED comparison  |  style={style}")
    print(f"{'='*60}")

    print("\nLoading base model...")
    base_model, _ = load_model(ADAPTER_PATH, load_adapter=False, style=style)
    print("Loading fine-tuned model...")
    ft_model, _ = load_model(ADAPTER_PATH, load_adapter=True, style=style)

    for q in prompts:
        prompt = build_prompt(q, style, tokenizer)
        base_raw = base_model.generate(prompt=prompt, max_tokens=MAX_TOKENS, sampler=sampler)
        ft_raw   = ft_model.generate(prompt=prompt, max_tokens=MAX_TOKENS, sampler=sampler)

        print(f"\nQ: {q}")
        print(f"[BASE]  {extract_response(base_raw, prompt, style)}")
        print(f"[TUNED] {extract_response(ft_raw, prompt, style)}")
        print("-" * 60)

def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned mlx-tune model")
    parser.add_argument("--interactive", action="store_true", help="REPL chat mode")
    parser.add_argument("--compare",     action="store_true", help="Base vs fine-tuned comparison")
    parser.add_argument("--style",       default=None,        help="Prompt style: chip2 | alpaca | chatml | raw")
    parser.add_argument("--temperature", type=float,          default=TEMPERATURE)
    parser.add_argument("--max-tokens",  type=int,            default=MAX_TOKENS)
    args = parser.parse_args()

    temperature = args.temperature
    max_tokens  = args.max_tokens

    style = args.style or detect_style(ADAPTER_PATH)
    print(f"Detected prompt style: {style}")

    if args.compare:
        _, tokenizer = load_model(ADAPTER_PATH, load_adapter=False, style=style)
        run_compare(tokenizer, style)
        return

    model, tokenizer = load_model(ADAPTER_PATH, style=style)

    if args.interactive:
        run_interactive(model, tokenizer, style)
    else:
        run_batch(model, tokenizer, style, temperature=temperature, max_tokens=max_tokens)

if __name__ == "__main__":
    main()
