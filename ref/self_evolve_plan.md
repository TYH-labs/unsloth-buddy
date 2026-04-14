# Self-Evolve: Long-Term Memory for unsloth-buddy

> Branch: `feat/self-evolve` | Status: v1 complete, v1.1 open items identified

Reference doc for the `~/.gaslamp/` long-term memory system. Read this before modifying
`scripts/reflect.py`, `scripts/init_project.py`, or the Phase 0 / Phase 7 sections of `SKILL.md`.

---

## Quick Progress Check

| Item | Status | Commit |
|---|---|---|
| `scripts/reflect.py` — `--extract` mode | ✅ Done | `a8e78a5` |
| `scripts/reflect.py` — `--write` mode (stdin) | ✅ Done | `a8e78a5` |
| `scripts/reflect.py` — `--write --input <file>` | ✅ Done | `aea617a` |
| `scripts/reflect.py` — `--all` flag | ✅ Done | `a8e78a5` |
| `scripts/reflect.py` — dedup hash fix | ✅ Fixed | `4b1cc72` |
| `scripts/init_project.py` — frozen snapshot injection | ✅ Done | `a8e78a5` |
| `SKILL.md` Phase 0 — Global Memory Injection block | ✅ Done | `a8e78a5` |
| `SKILL.md` Phase 7 — Reflection & Memory Synthesis | ✅ Done | `a8e78a5` |
| `AGENTS.md` — manifest + lifecycle + key constraints | ✅ Done | `a8e78a5` |
| `.gitignore` — `.gaslamp_context/` | ✅ Done | `a8e78a5` |
| End-to-end test (qwen_chip2_sft_2026_03_17) | ✅ Passed | — |
| **F-6: `user.md` category-replace (not append-only)** | 🔲 v1.1 | — |
| **F-7: Input validation in `--write`** | 🔲 v1.1 | — |
| **F-5: Phase 7 row in `progress_log.md` table** | 🔲 v1.1 | — |
| **F-2: SKILL.md §6/§9 classification hints** | 🔲 v1.1 | — |
| v2: `~/.gaslamp/benchmarks.json` | 🔲 v2 | — |
| v2: Autoresearch experiment loops | 🔲 v2 | — |

---

## Architecture

### The `~/.gaslamp/` Hierarchy

```
~/.gaslamp/
├── README.md                   ← auto-created on first write
├── user.md                     ← WHO: hardware, preferences, deploy targets (≤2000 chars)
├── lessons.md                  ← WHAT: isolated gotchas, model quirks, install traps (≤3000 chars)
├── skills.md                   ← HOW: scenario recipes with When: trigger conditions (≤3000 chars)
├── archive/
│   ├── lessons_2026_Q1.md      ← evicted oldest entries when lessons.md hits limit
│   └── skills_2026_Q1.md
└── index.json                  ← {schema_version, files: {char_count, updated, entry_count}}
```

### The Three Files — Critical Distinction

| File | Contains | Unit |
|---|---|---|
| `user.md` | Hardware, HF username, preferred models, language, verbosity | Category blocks (replace, not append) |
| `lessons.md` | Isolated facts — "X causes Y" | One dated `### [YYYY-MM-DD]` entry per fact |
| `skills.md` | Scenario recipes — "When task=X AND hw=Y, do Z in Phase N" | One dated recipe per scenario |

**Lessons vs Skills**: The critical distinction. A **lesson** is an isolated fact with no procedure. A **skill** connects multiple facts into a reusable `When: → steps` procedure. Skills are synthesized by the agent from cross-section patterns (§6 + §9 + §11 together), not extracted verbatim.

### Entry Format

**`lessons.md` / `user.md`** entries:
```markdown
### [YYYY-MM-DD] Title
Body text (≤120 chars for lessons, free-form for user).
Source: project_dir_name        ← lessons only, omit for user.md
```

**`skills.md`** recipe entries:
```markdown
### [YYYY-MM-DD] Scenario Name
When: task=<type> AND hardware=<hw> [AND model_size<=<N>B]
- Phase N: step description
- Phase N: step description
Source: project_dir_name
```

---

## reflect.py Design Rationale

> **Guardrail**: Do not add LLM calls to `reflect.py`. The agent IS the LLM. The script must remain stdlib-only.

### Why it exists as a subprocess tool

Three things that must be deterministic (not LLM-dependent):
1. **SHA-256 dedup** — comparing title+body hash against existing entries
2. **Char-limit eviction + quarterly archiving** — file-size accounting, date parsing, archive writes
3. **Front-matter updates + index.json bookkeeping** — mechanical counters

If the agent did these directly via file edits, they'd be inconsistently applied across LLM providers and invocations. The script makes them always-correct.

### Why two modes, not one

`--extract` and `--write` are deliberately separate because the **agent must classify in between**. The agent reads raw candidates, decides what is a lesson vs skill vs user preference, and writes a ≤120-char summary per lesson. This judgment is the only LLM-appropriate work in Phase 7.

### Interface: `--input <file>` preferred over stdin

> **Guardrail**: SKILL.md Phase 7 should instruct agents to write `.reflect_payload.json` first, then pass `--input`. Do not instruct agents to use heredoc or echo piping.

The agent writes the classified JSON to a file in the project dir using its native file tools. This:
- Eliminates shell quoting issues for lesson bodies containing `"`, backticks, or code
- Makes the payload a **durable artifact** — survives interruption between Steps 1 and 3
- Is auditable — the agent can review the payload before submitting

```bash
# Step 1
python3 scripts/reflect.py . --extract > .reflect_candidates.json

# Step 2 (agent classifies, writes file)
# Agent writes classified JSON to .reflect_payload.json using file tools

# Step 3
python3 scripts/reflect.py --write --input .reflect_payload.json
python3 scripts/reflect.py --write --input .reflect_payload.json --dry-run  # preview first
```

### Dedup mechanism

Hash is computed as `sha256(normalize(title + "\n" + final_body))` where `final_body` includes the `Source:` line (for lessons/skills) or `When:` + steps (for skills). The hash must match the stored body exactly — which is why it must be computed **after** assembling the full rendered body.

> **Guardrail**: If you change the body rendering in any formatter, update the hash computation in the same function. They must always hash the same string that gets written to disk.

---

## Frozen Snapshot Pattern

> **Guardrail**: `.gaslamp_context/` is **read-only during a session**. Never add logic that writes to it during training, eval, or any phase other than project init. New knowledge only flows back via Phase 7.

`init_project.py` copies `~/.gaslamp/{user,lessons,skills}.md` into `.gaslamp_context/` at project creation. The agent reads these once at session start and applies them silently. They are frozen for the rest of the session.

**Rationale (from Hermes agent design)**: If the agent could update its system-prompt-equivalent mid-session, every subsequent LLM call would be a cache miss. Freezing the snapshot at session start preserves prefix caching benefits for the entire session. New lessons write back only at project end.

**Agent behavior at session start** (from SKILL.md Phase 0):
1. Read `user.md` → pre-fill Phase 1 known answers (hardware, Python, deploy target)
2. Read `lessons.md` → silently apply matching gotchas for current task
3. Read `skills.md` → match `When:` triggers to current context; silently apply matching recipe steps
4. Log what was applied in `gaslamp.md` preamble: `> Applied from ~/.gaslamp/: ...`

---

## Archive Strategy

> **Guardrail**: Archive is **content-driven, quarterly-named**. Do not switch to monthly or time-driven triggers.

When a file exceeds its char limit during `--write`:
1. Sort entries by `date` (oldest first)
2. Evict oldest entries into `archive/{type}_YYYY_Q{N}.md` until file fits
3. Update front-matter `char_count` and `index.json`

Quarter calculation: `Q = (month - 1) // 3 + 1` (Q1=Jan-Mar, Q2=Apr-Jun, Q3=Jul-Sep, Q4=Oct-Dec).

**Why quarterly, not monthly**: A fine-tuning project cycle is weeks to months. Monthly archiving is too granular — you'd archive things that are still actively relevant after 31 days. Quarterly buckets match the actual cadence.

---

## Known Friction Points (v1.1 Backlog)

### F-6: `user.md` append problem (Medium — implement in v1.1)
`user.md` should replace entries by category, not append. Hardware upgrades create multiple conflicting entries. Solution: switch `user.md` from dated `### [YYYY-MM-DD]` entries to named `## Category` sections that get updated in-place.

### F-7: Input validation on `--write` (Medium — implement in v1.1)
Missing required fields (`title`, `body` for lessons; `when`, `steps` for skills) currently write malformed entries silently. Add validation in `write_entries()` that logs a warning and skips the entry.

### F-5: Phase 7 row in `progress_log.md` (Low — easy)
The progress table in `init_project.py` ends at Phase 4. Add Phase 7 row as "pending".

### F-2: SKILL.md §6/§9 classification hints (Low — easy)
Add a note in SKILL.md Phase 7 Step 2: "For §6, extract only non-default parameter values. For §9, extract the script source patterns (which templates were copied and as what). For §11, extract each bullet as a separate candidate."

---

## v2 Roadmap — ML-Specific Extensions

These extensions exploit the ML domain in ways Hermes (a general coding agent) cannot.

### `~/.gaslamp/benchmarks.json`
Structured numerical history indexed by `(model_family, hardware, task_type)`:
```json
{
  "runs": [
    {"project": "qwen_chip2_sft_2026_03_17", "model": "Qwen2.5-0.5B", "hw": "apple_m4",
     "task": "sft", "final_loss": 1.249, "train_time_min": 3, "peak_mem_gb": 2.1}
  ]
}
```
Enables comparative prediction, hyperparameter priors from similar past runs, and regression detection.

### Model Knowledge Graph
Tag `lessons.md` entries by model family to build a queryable MODEL → QUIRKS map. At Phase 1 interview, agent queries entries matching the chosen base model.

### Failure Mode Catalog
ML training fails in predictable patterns: loss spikes (lr too high), loss plateau (data mismatch or lr too low), OOM (batch/seq/rank), garbage generation (wrong chat template). A structured catalog accumulates diagnostic recipes.

### Autoresearch-Style Experiment Loops
With `benchmarks.json` + `skills.md` recipes + the existing 7-phase lifecycle:
- Agent proposes a hypothesis ("lr=1e-4 may converge better for 2B vision on M-series")
- Runs experiment using the existing training pipeline
- Records result in `benchmarks.json`
- Updates hyperparameter priors

Reference: [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — the same pattern applied to fine-tuning hyperparameter search, scoped to the user's own hardware and model preferences.

### Continuous / Inline Reflection
Evaluate the possibility of synthesizing what the agent learns *continuously during the project* (e.g., inline reflection capturing lessons at the exact moment a workaround is applied or a hyperparameter is chosen), rather than exclusively extracting everything as a batched post-mortem in Phase 7.

---

## Design Sources

| Source | What we borrowed | What we improved |
|---|---|---|
| [Hermes agent](../ref/memory_sys.md) | Frozen Snapshot, char-capped core memory, offline reflection concept | Agent-as-LLM (no second model), `--input <file>` interface, three-file split with Skills as trigger recipes |
| Hermes §5.1 (identified gap) | Offline reflection idea (Hermes never built it) | Built it as `reflect.py` |
| Hermes §5.3 (identified gap) | Skill/Fact blur problem | Explicit three-file split + recipe format with `When:` triggers |
| Hermes SKILL.md | "Self-improving through skills" framing | Applied to ML domain: recipes are phase-specific procedures, not generic scripts |
