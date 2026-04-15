"""
init_project.py — Create a dated project working directory for a fine-tuning run.

Usage:
    python scripts/init_project.py <project_name>
    python scripts/init_project.py qwen_chip2_sft

Creates:
    ./{project_name}_{YYYY_MM_DD}/
    ├── data/               # dataset downloads / processed samples
    ├── outputs/
    │   └── adapters/       # LoRA adapter weights
    ├── logs/               # training stdout/stderr logs
    ├── gaslamp.md          # roadbook: key decisions + rationale + learning warmup (reproducible)
    ├── memory.md           # working notes: discoveries, debugging, in-progress findings
    ├── progress_log.md     # chronological session log of each phase
    ├── reflect.py          # pinned copy — long-term memory extraction + write
    └── add_reflect_hint.py # pinned copy — inline reflection hint appender

Reflection scripts are pinned at project-creation time so each project is
self-contained and independent of future skill updates.

Prints the project directory path to stdout so the caller can cd into it.
"""

import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

# ── Args ─────────────────────────────────────────────────────────────────────
if len(sys.argv) < 2:
    print("Usage: python scripts/init_project.py <project_name>", file=sys.stderr)
    sys.exit(1)

raw_name = sys.argv[1]

# Sanitise: lowercase, replace spaces/special chars with underscores
project_name = re.sub(r"[^\w]+", "_", raw_name.strip().lower()).strip("_")
date_str     = datetime.now().strftime("%Y_%m_%d")
project_dir  = Path(f"{project_name}_{date_str}")

# ── Create structure ──────────────────────────────────────────────────────────
if project_dir.exists():
    print(f"[init] Directory already exists: {project_dir}", file=sys.stderr)
else:
    (project_dir / "data").mkdir(parents=True)
    (project_dir / "outputs" / "adapters").mkdir(parents=True)
    (project_dir / "logs").mkdir(parents=True)
    print(f"[init] Created project directory: {project_dir}", file=sys.stderr)

# ── gaslamp.md (roadbook) — copied from templates/gaslamp_template.md ────────
gaslamp_file = project_dir / "gaslamp.md"
if not gaslamp_file.exists():
    # Locate the template relative to this script (scripts/ → templates/)
    template_path = Path(__file__).parent.parent / "templates" / "gaslamp_template.md"
    if template_path.exists():
        content = template_path.read_text()
        # Substitute {project_name} placeholder
        content = content.replace("{project_name}", project_name)
        gaslamp_file.write_text(content)
    else:
        print(f"[init] Warning: gaslamp_template.md not found at {template_path}", file=sys.stderr)

# ── memory.md ────────────────────────────────────────────────────────────────
memory_file = project_dir / "memory.md"
if not memory_file.exists():
    memory_file.write_text(f"""\
# memory.md — Technical context for `{project_name}`

## Model
- Base model:
- Quantization:
- LoRA rank / alpha:
- Max seq length:

## Dataset
- Source:
- Format:
- Size (train / val):
- Prompt style:

## Hyperparameters
- Learning rate:
- Batch size / grad accum:
- Steps / epochs:
- Scheduler:

## Discoveries & Notes
<!-- Record debugging findings, unexpected behaviour, tuning decisions -->
""")

# ── progress_log.md ──────────────────────────────────────────────────────────
log_file = project_dir / "progress_log.md"
if not log_file.exists():
    log_file.write_text(f"""\
# progress_log.md — Session log for `{project_name}`

## {datetime.now().strftime("%Y-%m-%d %H:%M")} — Project initialised

| Phase | Status |
|-------|--------|
| 0: Init | ✅ done |
| 1: Env setup | pending |
| 2: Training | pending |
| 3: Evaluation | pending |
| 4: Export | pending |
| 7: Reflection | pending |
""")

# ── Global Memory Injection (Frozen Snapshot) ────────────────────────────────
# If ~/.gaslamp/ exists, copy user.md / lessons.md / skills.md into
# .gaslamp_context/ inside the project as a read-only session snapshot.
# The agent reads these at session start and silently applies relevant
# lessons/recipes — it does NOT modify them during the session.
gaslamp_home = Path.home() / ".gaslamp"
if gaslamp_home.exists():
    context_dir = project_dir / ".gaslamp_context"
    context_dir.mkdir(exist_ok=True)
    injected = []
    for fname in ("user.md", "lessons.md", "skills.md"):
        src = gaslamp_home / fname
        if src.exists():
            shutil.copy2(src, context_dir / fname)
            injected.append(fname)
    if injected:
        print(
            f"[init] Injected global memory snapshot → {context_dir} "
            f"({', '.join(injected)})",
            file=sys.stderr,
        )

# ── Copy reflection scripts (pinned to this skill version) ───────────────────
# Projects are self-contained: they carry the exact script versions used at
# creation time, independent of future skill updates or repo location changes.
scripts_dir = Path(__file__).parent
for script_name in ("reflect.py", "add_reflect_hint.py"):
    src = scripts_dir / script_name
    dst = project_dir / script_name
    if src.exists() and not dst.exists():
        shutil.copy2(src, dst)
        print(f"[init] Copied {script_name} → {project_dir}/", file=sys.stderr)

# ── Print path for caller to use ──────────────────────────────────────────────
print(project_dir)
