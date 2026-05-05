# Eval Skill: unsloth-buddy Skill Regression

Use this guide when asked to evaluate, regression-test, or validate the
`unsloth-buddy` agent skill.

## Role

You are the eval driver for the `unsloth-buddy` skill. Your job is to run the
lightest suite that can answer the question, preserve artifacts, and report
failures as actionable skill regressions.

## Default Flow

1. Run the harness tests:

   ```bash
   python3 -m unittest discover -s eval/tests
   ```

2. Run the fast static suite:

   ```bash
   python3 eval/run_skill_evals.py --suite static
   ```

3. Read `eval/artifacts/summary.md`.

4. Report blocking failures first, then warnings, then the matching fix
   suggestions.

## Suite Selection

| Request | Command |
|---|---|
| Quick health check | `python3 eval/run_skill_evals.py --suite static` |
| Harness regression tests | `python3 -m unittest discover -s eval/tests` |
| Activation behavior | `python3 eval/run_skill_evals.py --suite activation` |
| One activation case | `python3 eval/run_skill_evals.py --suite activation --case explicit-sft-jsonl` |
| Lifecycle dry-run gates | `python3 eval/run_skill_evals.py --suite lifecycle` |
| Everything available locally | `python3 eval/run_skill_evals.py --suite all` |

Agent-run suites require the Codex CLI because they call `codex exec --json`.
They run in disposable workspaces and must stay dry-run only.

## Hard Rules

- Do not run real fine-tuning in V1 evals.
- Do not install packages, download models, export models, deploy servers, or
  push to Hugging Face from smoke evals.
- Do not run agent cases in the real repo checkout; use the harness-created temp
  workspace.
- Preserve `trace.jsonl`, `stderr.log`, `final.txt`, and `score.json` for every
  agent-run case.
- Every failed or warning check must include a `fix` field in `score.json`; do
  not leave the user with only a raw failure.
- Treat deterministic grader failures as blocking. Treat rubric output as
  advisory unless the user explicitly asks for rubric gating.

## Current Expected Warnings

The static suite may warn about known contract drift:

- `scripts/eval_template.py` is referenced in an obsolete CUDA eval comment but
  is not present.
- `scripts/init_project.py` still creates `progress_log.md` with older phase
  labels.

Do not hide these warnings. They are useful regression signals.
