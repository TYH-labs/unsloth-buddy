# Sub-skill: Data Strategy (Unsloth)

After completing Phase 1 (Interview & Requirements Gathering), this skill guides the user through preparing the exact dataset required for their chosen `trl` Training Method.

Unsloth uses HuggingFace `trl` Trainers. Every training method mathematically demands a specific column format. Before writing any Python code to train the model, you must acquire the data and format it perfectly.

## Prerequisites

Before starting this phase, you MUST have a `project_brief.md` that confirms:
- **Task Type** (SFT, DPO, GRPO, etc.)
- **Base Model** (Needed if applying Chat Templates)

## Phase 2A: Data Discovery & Assessment

1. **Ask the User about Data Status**:
   - "Do you already have a prepared dataset (e.g., CSV, JSONL, Parquet), or do we need to build one?"
   - *If they have local data*: Ask them for the file path. Move to Phase 2B (Local Data Formatting).
   - *If they want to use a Hugging Face dataset*: Ask for the dataset identifier. Move to Phase 2C (Hugging Face SQL Integration).
   - *If they don't have data*: Offer to help them search Hugging Face Hub (using the HF MCP server) or generate synthetic data.

## Phase 2B: Local Data Formatting

1. **Load and Profile (If Local Data Exists)**:
   - Load their dataset (e.g., using `pandas` or `datasets`).
   - Output the column names and the first 2 rows so you can evaluate its current structure.
   - **Crucial Check**: Does this structure match the TRL method defined in Phase 1?
        - **SFT**: Needs `text` or `messages` (e.g., `[{"role": "user", "content": "..."}]`).
        - **DPO / ORPO**: Needs `prompt`, `chosen` (good response), and `rejected` (bad response).
        - **KTO**: Needs `prompt`, `completion`, and `label` (True/False).
        - **GRPO**: Needs `prompt` (plus you'll need to define a Python reward metric later).
   - If restructuring is needed, write a quick python `.map()` script to format it to `prepared_dataset.jsonl` or `.parquet`.

## Phase 2C: Hugging Face SQL Integration (The Power Feature)

If the user wants to train on an existing Hugging Face dataset, **DO NOT download the whole dataset to Python to format it.**

Instead, use the powerful `sql_manager.py` tool from the `hugging-face-datasets` skill to manipulate the remote dataset via DuckDB and instantly push the cleaned version to their account.

1. **Explore the Schema via SQL**:
   - First, run `uv run scripts/sql_manager.py describe --dataset "dataset_name"` to see its exact schema.
   - Sample it: `uv run scripts/sql_manager.py sample --dataset "dataset_name" --n 5`

2. **Format to TRL Columns via SQL**:
   - Write a DuckDB SQL query to perfectly map the upstream columns to TRL's required format.
   - For example, if the user chose SFT and the HF dataset has `question` and `answer` columns, you can map it instantly:
     ```bash
     uv run scripts/sql_manager.py export \
       --dataset "cais/mmlu" \
       --sql "SELECT question AS prompt, answer AS chosen FROM data WHERE LENGTH(question) > 50" \
       --output "prepared_dataset.parquet" \
       --format parquet
     ```
   - For complex JSON mapping (like chat templates), you can extract array items: `SELECT choices[answer] as correct`

3. **Save Locally or Push**:
   - Save the fully formatted dataset locally using `--output` (ideal for small runs), or if the dataset is massive, push it directly to their HF account using `--push-to "username/formatted-dataset-name"`.

## Phase 2D: Data Generation (If no data exists)

If the user has no dataset and no public repo fits:
- **Generate Synthetic Data**: Write a prompt template suggesting how the user can use LLMs (like Claude or GPT-4) to generate 50-100 high-quality training pairs formatted exactly for their chosen TRL method.

## Output Artifacts

By the end of this phase, you must deliver:
1. **Mandatory Final Deliverable**: `data_strategy.md`
   - Documenting the dataset source, total samples, and formatting logic.
   - Confirmation that the dataset currently exists on disk in a format compatible with `trl`.

Only when the dataset is fully prepped and the `data_strategy.md` is complete, transition to **Phase 3: Environment Setup** from the main `SKILL.md`.
