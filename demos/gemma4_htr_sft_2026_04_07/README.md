# Gemma 4 HTR SFT — Handwriting Text Recognition

This demo showcases a fine-tuned **Gemma 4 2B Vision** model optimized for **Handwriting Text Recognition (HTR)** on the `Teklia/IAM-line` dataset.

The model was fine-tuned using `mlx-tune` on Apple Silicon.

## Files
- `gaslamp.md`: Reproducibility roadbook.
- `index.html`: Interactive side-by-side evaluation dashboard.
- `assets/`: Handwriting image samples used in the demo.

## How to Reproduce
Run the following command in your terminal with unsloth-buddy installed:
```bash
/unsloth-buddy reproduce using demos/gemma4_htr_sft_2026_04_07/gaslamp.md
```
