# Demos

Five end-to-end fine-tuning examples using unsloth-buddy.

| # | Use case | Dataset | Hardware | Method |
|---|----------|---------|----------|--------|
| 1 | Customer support FAQ bot | CSV (1,200 rows) | Apple M4 24 GB | SFT |
| 2 | Doctor-patient summarization | 500 consultation transcripts | NVIDIA A100 | SFT |
| 3 | Code assistant (Python) | HuggingFace `iamtarun/python_code_instructions_18k_alpaca` | NVIDIA T4 | SFT |
| 4 | Preference-tuned chat model | Custom DPO pairs | Apple M2 16 GB | DPO |
| 5 | Reasoning model (math) | HuggingFace `openai/gsm8k` | NVIDIA A100 | GRPO |

Each demo will include a `project_brief.md`, `data_strategy.md`, and the generated `train.py`.
