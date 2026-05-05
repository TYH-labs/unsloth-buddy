#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from graders.common import CheckResult, fail, ok, suite_passed
from graders.filesystem import assert_no_root_training_artifacts, find_project_dirs
from graders.lifecycle import (
    PROHIBITED_SMOKE_COMMANDS,
    check_data_before_training,
    check_demo_after_eval,
    check_env_before_training,
    check_expected_activation,
    check_interview_contract,
    check_phase0_init,
    check_reflect_only_at_end,
)
from graders.report import case_payload, write_case_score, write_summary
from graders.static_contract import run_static_contract_checks
from graders.trace import extract_commands, load_events, check_no_prohibited_commands


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS = EVAL_ROOT / "artifacts"


def read_cases(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def bool_field(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y"}


def copy_minimal_workspace(source_root: Path, dest: Path) -> None:
    for file_name in ("AGENTS.md", "SKILL.md"):
        shutil.copy2(source_root / file_name, dest / file_name)
    for dir_name in ("scripts", "sub-skills", "templates"):
        shutil.copytree(
            source_root / dir_name,
            dest / dir_name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    shutil.copytree(EVAL_ROOT / "fixtures", dest / "eval_fixtures")


def prepare_home(work_root: Path) -> Path:
    home = work_root / "home"
    home.mkdir(parents=True, exist_ok=True)
    fixture_home = EVAL_ROOT / "fixtures" / "gaslamp_home"
    gaslamp = home / ".gaslamp"
    if fixture_home.exists():
        shutil.copytree(fixture_home, gaslamp)
    return home


def build_prompt(case: dict[str, str], workspace: Path) -> str:
    prompt = case["prompt"]
    replacements = {
        "{fixtures}": str(workspace / "eval_fixtures"),
        "{workspace}": str(workspace),
        "{resume_project}": str(workspace / "resume_fixture_2026_05_05"),
    }
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)
    if bool_field(case.get("dry_run", "true")):
        prompt += (
            "\n\nThis is an eval dry run. Do not install packages, download models, "
            "start training, export models, deploy servers, or write to the real user home. "
            "Stop at the first point where real environment setup or training would be required."
        )
    return prompt


def prepare_case_workspace(case: dict[str, str], work_root: Path) -> Path:
    workspace = work_root / "workspace"
    workspace.mkdir(parents=True)
    copy_minimal_workspace(REPO_ROOT, workspace)
    if case.get("fixture") == "resume_project":
        resume_dir = workspace / "resume_fixture_2026_05_05"
        shutil.copytree(EVAL_ROOT / "fixtures" / "resume_project", resume_dir)
    return workspace


def run_codex_case(
    case: dict[str, str],
    case_dir: Path,
    codex_bin: str,
    timeout: int,
    keep_workspace: bool,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix=f"unsloth-buddy-eval-{case['id']}-") as tmp:
        work_root = Path(tmp)
        workspace = prepare_case_workspace(case, work_root)
        home = prepare_home(work_root)
        prompt = build_prompt(case, workspace)

        cmd = [codex_bin, "exec", "--json"]
        if bool_field(case.get("full_auto", "true")):
            cmd.append("--full-auto")
        cmd.append(prompt)

        env = os.environ.copy()
        env["HOME"] = str(home)

        trace_path = case_dir / "trace.jsonl"
        stderr_path = case_dir / "stderr.log"
        final_path = case_dir / "final.txt"
        prompt_path = case_dir / "prompt.txt"
        case_dir.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(prompt, encoding="utf-8")

        checks: list[CheckResult] = []
        if shutil.which(codex_bin) is None:
            checks.append(fail("codex_cli_available", f"codex binary not found: {codex_bin}"))
            payload = case_payload(case["id"], case["suite"], checks, artifacts={"prompt": str(prompt_path)})
            write_case_score(case_dir / "score.json", payload)
            return payload

        try:
            res = subprocess.run(
                cmd,
                cwd=workspace,
                env=env,
                text=True,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            trace_path.write_text(exc.stdout or "", encoding="utf-8")
            stderr_path.write_text(exc.stderr or "", encoding="utf-8")
            checks.append(fail("codex_exec_completed", f"timed out after {timeout}s"))
            payload = case_payload(case["id"], case["suite"], checks, artifacts={"trace": str(trace_path), "stderr": str(stderr_path)})
            write_case_score(case_dir / "score.json", payload)
            return payload

        trace_path.write_text(res.stdout, encoding="utf-8")
        stderr_path.write_text(res.stderr, encoding="utf-8")
        final_path.write_text(extract_final_text(res.stdout), encoding="utf-8")

        events, malformed = load_events(trace_path)
        commands = extract_commands(events)
        final_text = final_path.read_text(encoding="utf-8")
        expected = bool_field(case.get("expected_activation", "false"))

        checks.append(ok("codex_exec_completed", f"exit_code={res.returncode}") if res.returncode == 0 else fail("codex_exec_completed", f"exit_code={res.returncode}; see stderr.log"))
        if malformed:
            checks.append(fail("trace_jsonl_valid", "; ".join(malformed)))
        else:
            checks.append(ok("trace_jsonl_valid", f"parsed {len(events)} JSONL events"))
        checks.append(check_no_prohibited_commands(commands, PROHIBITED_SMOKE_COMMANDS))
        checks.append(check_expected_activation(expected, workspace, final_text))
        checks.append(assert_no_root_training_artifacts(workspace, allowed_roots={"eval_fixtures"}))

        projects = find_project_dirs(workspace)
        project_dir = projects[0] if projects else None
        if expected and case["suite"] in {"activation", "lifecycle"}:
            checks.append(check_phase0_init(workspace))
        if expected and project_dir and case["suite"] == "lifecycle":
            checks.append(check_interview_contract(project_dir, final_text))
            checks.append(check_data_before_training(commands, project_dir))
            checks.append(check_env_before_training(commands))
            checks.append(check_demo_after_eval(project_dir, commands))
            checks.append(check_reflect_only_at_end(commands, project_dir))

        artifacts = {
            "trace": str(trace_path),
            "stderr": str(stderr_path),
            "final": str(final_path),
            "prompt": str(prompt_path),
        }
        if keep_workspace:
            saved = case_dir / "workspace"
            shutil.copytree(workspace, saved)
            artifacts["workspace"] = str(saved)

        payload = case_payload(case["id"], case["suite"], checks, artifacts=artifacts)
        write_case_score(case_dir / "score.json", payload)
        return payload


def extract_final_text(stdout_jsonl: str) -> str:
    texts: list[str] = []
    for line in stdout_jsonl.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        _collect_text(event, texts)
    return "\n".join(dict.fromkeys(texts))


def _collect_text(value: Any, out: list[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in {"text", "message", "content", "final_response"} and isinstance(child, str):
                if child.strip():
                    out.append(child.strip())
            elif isinstance(child, (dict, list)):
                _collect_text(child, out)
    elif isinstance(value, list):
        for child in value:
            _collect_text(child, out)


def static_suite(artifacts_dir: Path) -> dict[str, Any]:
    checks = run_static_contract_checks(REPO_ROOT)
    payload = case_payload("static-contract", "static", checks)
    write_case_score(artifacts_dir / "static" / "score.json", payload)
    return payload


def prompt_suite(
    suite: str,
    artifacts_dir: Path,
    case_filter: str | None,
    codex_bin: str,
    timeout: int,
    keep_workspace: bool,
) -> list[dict[str, Any]]:
    if suite == "activation":
        files = [EVAL_ROOT / "prompts" / "skill_activation.csv", EVAL_ROOT / "prompts" / "negative_controls.csv"]
    elif suite == "lifecycle":
        files = [EVAL_ROOT / "prompts" / "lifecycle_smoke.csv"]
    else:
        raise ValueError(f"unsupported prompt suite: {suite}")

    cases: list[dict[str, str]] = []
    for file in files:
        for case in read_cases(file):
            case["suite"] = suite
            cases.append(case)
    if case_filter:
        cases = [case for case in cases if case["id"] == case_filter]
    if not cases:
        raise SystemExit(f"no cases matched suite={suite!r} case={case_filter!r}")

    payloads = []
    for case in cases:
        payloads.append(run_codex_case(case, artifacts_dir / suite / case["id"], codex_bin, timeout, keep_workspace))
    return payloads


def main() -> int:
    parser = argparse.ArgumentParser(description="Run unsloth-buddy skill evals")
    parser.add_argument("--suite", choices=["static", "activation", "lifecycle", "smoke", "all"], default="static")
    parser.add_argument("--case", help="Run one prompt case by id")
    parser.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--keep-workspaces", action="store_true")
    args = parser.parse_args()

    artifacts_dir = args.artifacts_dir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    payloads: list[dict[str, Any]] = []
    if args.suite in {"static", "smoke", "all"}:
        payloads.append(static_suite(artifacts_dir))
    if args.suite in {"activation", "all"}:
        payloads.extend(prompt_suite("activation", artifacts_dir, args.case, args.codex_bin, args.timeout, args.keep_workspaces))
    if args.suite in {"lifecycle", "all"}:
        payloads.extend(prompt_suite("lifecycle", artifacts_dir, args.case, args.codex_bin, args.timeout, args.keep_workspaces))

    write_summary(artifacts_dir / "summary.md", payloads)
    print(f"Wrote summary: {artifacts_dir / 'summary.md'}")
    return 0 if all(payload["pass"] for payload in payloads) else 1


if __name__ == "__main__":
    raise SystemExit(main())

