"""
mlx_eval_vision_template.py — Vision evaluation template for mlx-tune Vision Language Models.

Usage:
    python eval.py                        # batch mode with validation images
    python eval.py --compare              # side-by-side: base model vs fine-tuned
"""

import argparse
import sys
import os
import json

from mlx_tune import FastVisionModel
from mlx_vlm.generate import generate
from datasets import load_dataset
from PIL import Image

# ── Config — edit these to match your training run ──────────────────────────
MODEL_NAME   = "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
ADAPTER_PATH = "outputs/adapters"   # set to None to run base model only
MAX_SEQ_LEN  = 1024
MAX_TOKENS   = 200
DATASET_ID   = "your-dataset-here"

SYSTEM_PROMPT = "You are a helpful assistant."
USER_PROMPT = "Describe this image."

# ── 1. Fix Adapter JSON logic ───────────────────────────────────────────────
def patch_adapter_config(adapter_path):
    """
    mlx_tune writes 'rank' inside 'lora_parameters' config,
    but mlx_vlm.load expects 'rank' at the root of adapter_config.json.
    This patches the file smoothly if not patched already.
    """
    config_file = os.path.join(adapter_path, "adapter_config.json")
    if os.path.exists(config_file):
        with open(config_file, "r+") as f:
            cfg = json.load(f)
            if "lora_parameters" in cfg:
                for k, v in cfg["lora_parameters"].items():
                    cfg[k] = v
                cfg.pop("lora_parameters", None)
            
            cfg.pop("fine_tune_type", None)
            cfg.pop("num_layers", None)
            cfg.pop("keys", None)
            f.seek(0)
            json.dump(cfg, f, indent=2)
            f.truncate()

def load_model(adapter_path, load_adapter=True):
    kwargs = {}
    if load_adapter and adapter_path:
        patch_adapter_config(adapter_path)
        kwargs["adapter_path"] = adapter_path

    model, processor = FastVisionModel.from_pretrained(
        model_name     = MODEL_NAME,
        max_seq_length = MAX_SEQ_LEN,
        load_in_4bit   = True,
        **kwargs,
    )
    FastVisionModel.for_inference(model)
    return model, processor

# ── 2. Evaluation Scripts ───────────────────────────────────────────────────
def build_prompt(processor):
    # Adjust this layout if your model requires different prompt structuring 
    # instead of [SYSTEM, IMAGE, USER].
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": SYSTEM_PROMPT},
            {"type": "image"},
            {"type": "text", "text": USER_PROMPT},
        ]}
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def run_batch(model, processor, dataset, max_tokens=MAX_TOKENS):
    prompt = build_prompt(processor)
    
    print(f"\\n{'='*60}")
    print(f"BATCH EVAL  |  dataset={DATASET_ID}  |  adapter={ADAPTER_PATH}")
    print(f"{'='*60}")
    
    # Process the first 3 samples
    limit = min(3, len(dataset))
    for i in range(limit):
        sample = dataset[i]
        
        # Replace 'image' and 'text' with your dataset's actual column names
        image = sample.get("image", None)
        ground_truth = sample.get("text", sample.get("answer", "Unknown"))
        
        if image is None:
            continue
            
        os.makedirs("logs/eval_images", exist_ok=True)
        img_path = f"logs/eval_images/batch_sample_{i+1}.png"
        image.save(img_path)
        
        print(f"\nSample {i+1} (Image: {img_path}):")
        print(f"[GROUND TRUTH] {ground_truth}")
        
        try:
            res = generate(model, processor, prompt, [img_path], max_tokens=max_tokens, verbose=False)
            print(f"[TUNED MODEL]  {res.text.strip()}")
        except Exception as e:
            print(f"[ERROR] {e}")

def run_compare(dataset, max_tokens=MAX_TOKENS):
    print(f"\n{'='*60}")
    print(f"BASE vs FINE-TUNED comparison  |  dataset={DATASET_ID}")
    print(f"{'='*60}")

    print("\nLoading base model...")
    base_model, processor = load_model(ADAPTER_PATH, load_adapter=False)
    
    # Because MLX handles memory eagerly, we simply continue instead of resetting 
    print("\nLoading fine-tuned model...")
    ft_model, _ = load_model(ADAPTER_PATH, load_adapter=True)

    prompt = build_prompt(processor)
    
    limit = min(3, len(dataset))
    for i in range(limit):
        sample = dataset[i]
        image = sample.get("image", None)
        ground_truth = sample.get("text", sample.get("answer", "Unknown"))
        
        if image is None:
            continue
            
        os.makedirs("logs/eval_images", exist_ok=True)
        img_path = f"logs/eval_images/compare_sample_{i+1}.png"
        image.save(img_path)
        
        print(f"\nSample {i+1} (Image: {img_path}):")
        print(f"[GROUND TRUTH] {ground_truth}")
        
        try:
            base_res = generate(base_model, processor, prompt, [img_path], max_tokens=max_tokens, verbose=False)
            print(f"[BASE MODEL]   {base_res.text.strip()}")
            
            ft_res = generate(ft_model, processor, prompt, [img_path], max_tokens=max_tokens, verbose=False)
            print(f"[TUNED MODEL]  {ft_res.text.strip()}")
        except Exception as e:
            print(f"[ERROR] {e}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned mlx-tune Vision model")
    parser.add_argument("--compare",     action="store_true", help="Base vs fine-tuned comparison")
    parser.add_argument("--max-tokens",  type=int,            default=MAX_TOKENS)
    args = parser.parse_args()

    print(f"Loading validation dataset: {DATASET_ID}")
    try:
        val_dataset = load_dataset(DATASET_ID, split="validation")
    except Exception as e:
        print(f"Failed to load dataset: {e}. Create a mock or map your data source.")
        return

    if args.compare:
        run_compare(val_dataset, max_tokens=args.max_tokens)
        return

    model, processor = load_model(ADAPTER_PATH)
    run_batch(model, processor, val_dataset, max_tokens=args.max_tokens)

if __name__ == "__main__":
    main()
