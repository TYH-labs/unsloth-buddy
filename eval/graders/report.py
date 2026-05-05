from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import CheckResult, score_from_checks, suite_passed, write_text


def case_payload(case_id: str, suite: str, checks: list[CheckResult], artifacts: dict[str, str] | None = None) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "suite": suite,
        "pass": suite_passed(checks),
        "score": score_from_checks(checks),
        "checks": [check.to_json() for check in checks],
        "artifacts": artifacts or {},
    }


def write_case_score(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_summary(path: Path, payloads: list[dict[str, Any]]) -> None:
    by_suite: dict[str, list[dict[str, Any]]] = {}
    for payload in payloads:
        by_suite.setdefault(payload["suite"], []).append(payload)

    lines = ["# unsloth-buddy Skill Eval Summary", ""]
    lines.append("| Suite | Cases | Passed | Failed |")
    lines.append("|---|---:|---:|---:|")
    for suite, suite_payloads in sorted(by_suite.items()):
        passed = sum(1 for payload in suite_payloads if payload["pass"])
        total = len(suite_payloads)
        lines.append(f"| {suite} | {total} | {passed} | {total - passed} |")

    failures = []
    warnings = []
    fixes = []
    for payload in payloads:
        for check in payload["checks"]:
            if check["pass"]:
                continue
            key = f"{payload['suite']}/{payload['case_id']}/{check['id']}"
            entry = f"- `{key}`: {check['notes']}"
            if check["severity"] == "fail":
                failures.append(entry)
            else:
                warnings.append(entry)
            fix = check.get("fix")
            if fix:
                fixes.append(f"- `{key}`: {fix}")

    if failures:
        lines.extend(["", "## Failures", "", *failures])
    if warnings:
        lines.extend(["", "## Warnings", "", *warnings])
    if fixes:
        lines.extend(["", "## Fix Suggestions", "", *fixes])
    if not failures and not warnings:
        lines.extend(["", "No failures or warnings."])

    write_text(path, "\n".join(lines) + "\n")
