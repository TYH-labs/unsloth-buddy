from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class CheckResult:
    id: str
    passed: bool
    severity: str
    notes: str
    fix: str = ""

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["pass"] = data.pop("passed")
        return data


def ok(check_id: str, notes: str = "", severity: str = "fail") -> CheckResult:
    return CheckResult(check_id, True, severity, notes)


def fail(check_id: str, notes: str, severity: str = "fail", fix: str = "") -> CheckResult:
    if not fix:
        fix = "Inspect the case artifacts, update the violated skill contract or grader expectation, then rerun the suite."
    return CheckResult(check_id, False, severity, notes, fix)


def suite_passed(checks: list[CheckResult]) -> bool:
    return not any((not check.passed) and check.severity == "fail" for check in checks)


def score_from_checks(checks: list[CheckResult]) -> int:
    blocking = [check for check in checks if check.severity == "fail"]
    if not blocking:
        return 100
    passed = sum(1 for check in blocking if check.passed)
    return round(100 * passed / len(blocking))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
