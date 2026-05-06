from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from .common import CheckResult, fail, ok


PROJECT_DIR_RE = re.compile(r".+_\d{4}_\d{2}_\d{2}$")


def find_project_dirs(workspace: Path, date: str | None = None) -> list[Path]:
    suffix = date or datetime.now().strftime("%Y_%m_%d")
    dirs = []
    for child in workspace.iterdir() if workspace.exists() else []:
        if child.is_dir() and child.name.endswith(f"_{suffix}") and PROJECT_DIR_RE.match(child.name):
            dirs.append(child)
    return sorted(dirs)


def has_required_project_skeleton(project_dir: Path) -> CheckResult:
    required = [
        "data",
        "outputs/adapters",
        "logs",
        "gaslamp.md",
        "memory.md",
        "progress_log.md",
        "reflect.py",
        "add_reflect_hint.py",
    ]
    missing = [rel for rel in required if not (project_dir / rel).exists()]
    if missing:
        return fail("project_skeleton", f"missing: {', '.join(missing)}", fix="Update scripts/init_project.py so every required project skeleton path is created.")
    return ok("project_skeleton", f"required project skeleton exists in {project_dir.name}")


def assert_no_root_training_artifacts(workspace: Path, allowed_roots: set[str] | None = None) -> CheckResult:
    allowed = allowed_roots or set()
    prohibited = ["train.py", "eval.py", "detect_system_result.json", "detect_env_result.json", "outputs", "logs", "data", "demos"]
    hits = [name for name in prohibited if name not in allowed and (workspace / name).exists()]
    if hits:
        return fail("no_root_training_artifacts", f"root-level training artifacts found: {', '.join(hits)}", fix="Move training/eval/demo artifacts into the dated project directory and update the skill instructions to keep repo root clean.")
    return ok("no_root_training_artifacts", "no root-level training artifacts found")


def assert_no_unexpected_files(workspace: Path, allowlist: set[str]) -> CheckResult:
    seen = {child.name for child in workspace.iterdir()} if workspace.exists() else set()
    unexpected = sorted(seen - allowlist)
    if unexpected:
        return fail("no_unexpected_files", f"unexpected root entries: {', '.join(unexpected)}", fix="Either add expected fixture files to the allowlist or prevent the run from creating unexpected root entries.")
    return ok("no_unexpected_files", "workspace root matches allowlist")


def assert_resume_fixture(project_dir: Path) -> CheckResult:
    required = ["gaslamp.md", "progress_log.md", "project_brief.md"]
    missing = [rel for rel in required if not (project_dir / rel).exists()]
    if missing:
        return fail("resume_fixture", f"missing resume fixture files: {', '.join(missing)}", fix="Restore the resume fixture so it includes gaslamp.md, progress_log.md, and project_brief.md.")
    gaslamp = (project_dir / "gaslamp.md").read_text(encoding="utf-8")
    if "Authoritative Resume Fixture" not in gaslamp:
        return fail("resume_fixture", "gaslamp.md does not look like the resume fixture", fix="Restore eval/fixtures/resume_project/gaslamp.md from the expected fixture content.")
    return ok("resume_fixture", "resume fixture is present")
