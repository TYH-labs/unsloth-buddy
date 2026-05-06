# unsloth-buddy Skill Evals

This directory contains a stdlib-only V1 eval harness for the `unsloth-buddy`
agent skill.

For the short agent-facing procedure, read [EVAL.md](EVAL.md). This README is
the directory map and command reference.

V1 focuses on regression signals that do not require model downloads, package
installs, or real fine-tuning:

- static contract checks for `SKILL.md`, `AGENTS.md`, scripts, templates, and
  sub-skills
- prompt fixtures for activation and negative-control evals
- lifecycle smoke fixtures for data/env/demo/reflection gates
- JSONL trace, filesystem, and lifecycle graders
- unit tests for grader behavior

## Quick Commands

```bash
# Unit/regression tests for the harness
python3 -m unittest discover -s eval/tests

# Fast static contract suite
python3 eval/run_skill_evals.py --suite static

# Agent-run suites require the Codex CLI and run in disposable temp workspaces
python3 eval/run_skill_evals.py --suite activation --case explicit-sft-jsonl
python3 eval/run_skill_evals.py --suite lifecycle --case sft-local-jsonl-stop-before-env
```

Agent-run suites are intentionally separate from static checks. They invoke
`codex exec --json` and may take longer, but the prompts instruct the agent to
dry-run and stop before package installation, model downloads, or training.

## Artifacts

By default, suite output goes to:

```text
eval/artifacts/
  summary.md
  static/score.json
  <case-id>/
    trace.jsonl
    stderr.log
    final.txt
    score.json
```

Use `--artifacts-dir /path/to/dir` to redirect output. Use `--keep-workspaces`
to preserve the temp workspace path in each case directory.

## Layout

```text
eval/
  EVAL.md                         # agent-facing eval process
  README.md                       # this overview
  run_skill_evals.py              # suite runner
  graders/                        # static, trace, filesystem, lifecycle, report helpers
  prompts/                        # activation, negative-control, lifecycle CSVs
  fixtures/                       # tiny local data and resume/gaslamp fixtures
  schemas/                        # optional rubric output schema
  tests/                          # unittest regression tests for the harness
```

## Interpreting Results

Blocking failures are entries in `summary.md` under `## Failures`. Warnings are
non-blocking but should stay visible because they track known contract drift.
Each failed or warning check also carries a `fix` field in `score.json`, and the
suite summary prints these under `## Fix Suggestions`.
The current static suite is expected to warn about the obsolete
`scripts/eval_template.py` reference and older `progress_log.md` phase labels
until those source files are updated.
