from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .common import CheckResult, fail, ok
from .filesystem import has_required_project_skeleton


REQUIRED_SUBSKILLS = [
    "sub-skills/interview.md",
    "sub-skills/data.md",
    "sub-skills/demo_builder.md",
]

REQUIRED_SCRIPTS = [
    "scripts/init_project.py",
    "scripts/detect_system.py",
    "scripts/detect_env.py",
    "scripts/reflect.py",
    "scripts/add_reflect_hint.py",
    "scripts/unsloth_mlx_sft_example.py",
    "scripts/unsloth_mlx_vision_example.py",
    "scripts/unsloth_sft_example.py",
    "scripts/unsloth_dpo_example.py",
    "scripts/unsloth_grpo_example.py",
    "scripts/mps_grpo_example.py",
    "scripts/unsloth_vision_example.py",
    "scripts/mlx_eval_template.py",
    "scripts/mlx_eval_vision_template.py",
    "scripts/llamacpp.py",
]

REQUIRED_TEMPLATES = [
    "templates/gaslamp_template.md",
    "templates/dashboard.html",
    "templates/demo_llm_crisp.html",
    "templates/demo_llm_dark.html",
    "templates/demo_vlm_crisp.html",
    "templates/demo_vlm_dark.html",
    "templates/chat_ui.html",
]


def check_skill_frontmatter(repo_root: Path) -> CheckResult:
    skill = repo_root / "SKILL.md"
    if not skill.exists():
        return fail("skill_frontmatter", "SKILL.md is missing", fix="Restore SKILL.md or update the eval target to the correct skill file.")
    text = skill.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return fail("skill_frontmatter", "SKILL.md is missing YAML front matter", fix="Add YAML front matter with at least name and description fields.")
    frontmatter = text.split("---", 2)[1]
    missing = [field for field in ("name:", "description:") if field not in frontmatter]
    if missing:
        return fail("skill_frontmatter", f"front matter missing: {', '.join(missing)}", fix="Add the missing front matter fields so skill activation remains testable.")
    return ok("skill_frontmatter", "SKILL.md front matter contains name and description")


def check_required_paths(repo_root: Path) -> list[CheckResult]:
    checks = []
    for rel in REQUIRED_SUBSKILLS + REQUIRED_SCRIPTS + REQUIRED_TEMPLATES:
        path = repo_root / rel
        checks.append(
            ok(f"path_exists:{rel}", "exists")
            if path.exists()
            else fail(f"path_exists:{rel}", "missing", fix=f"Restore `{rel}` or remove/update the lifecycle contract that requires it.")
        )
    return checks


def _extract_referenced_paths(text: str) -> set[str]:
    paths = set()
    for match in re.finditer(r"\b(?:scripts|templates|sub-skills)/[A-Za-z0-9_./:-]+", text):
        path = match.group(0).split("::", 1)[0].rstrip(".,:)`'\"")
        if path.endswith("/"):
            path = path[:-1]
        if path.split("/")[-1] in {"X", "Y"}:
            continue
        paths.add(path)
    return paths


def check_referenced_paths(repo_root: Path) -> list[CheckResult]:
    paths: set[str] = set()
    for rel in ("SKILL.md", "AGENTS.md"):
        source = repo_root / rel
        if source.exists():
            paths |= _extract_referenced_paths(source.read_text(encoding="utf-8"))

    checks: list[CheckResult] = []
    for rel in sorted(paths):
        if (repo_root / rel).exists():
            checks.append(ok(f"referenced_path:{rel}", "exists"))
        elif rel == "scripts/eval_template.py":
            checks.append(fail(
                f"referenced_path:{rel}",
                "missing; appears to be an obsolete CUDA eval-template comment",
                severity="warn",
                fix="Update the CUDA eval comment in SKILL.md to point at an existing eval template, or add the missing template if it is still required.",
            ))
        else:
            checks.append(fail(f"referenced_path:{rel}", "missing", fix=f"Add `{rel}` or remove the stale reference from SKILL.md/AGENTS.md."))
    return checks


def check_interview_two_question_contract(repo_root: Path) -> CheckResult:
    text = (repo_root / "sub-skills" / "interview.md").read_text(encoding="utf-8").lower()
    required = ["the 2-question interview", "task definition", "data status", "domain/audience", "project_brief.md"]
    missing = [token for token in required if token not in text]
    if missing:
        return fail("interview_two_question_contract", f"missing: {', '.join(missing)}", fix="Update sub-skills/interview.md so the 2-question task/data interview and project_brief.md deliverable are explicit.")
    return ok("interview_two_question_contract", "interview sub-skill has the required contract")


def check_data_phase_prerequisites(repo_root: Path) -> CheckResult:
    text = (repo_root / "sub-skills" / "data.md").read_text(encoding="utf-8").lower()
    required = ["project_brief.md", "data_strategy.md", "sft", "dpo", "grpo", "messages", "chosen", "rejected", "prompt"]
    missing = [token for token in required if token not in text]
    if missing:
        return fail("data_phase_prerequisites", f"missing: {', '.join(missing)}", fix="Update sub-skills/data.md to require project_brief.md, data_strategy.md, and the TRL schemas for SFT/DPO/GRPO.")
    return ok("data_phase_prerequisites", "data sub-skill requires brief, strategy, and TRL schemas")


def check_detect_env_exit_code(repo_root: Path) -> CheckResult:
    text = (repo_root / "scripts" / "detect_env.py").read_text(encoding="utf-8")
    if "sys.exit(0 if ready else 1)" not in text:
        return fail("detect_env_exit_code", "detect_env.py must exit non-zero when not ready", fix="Restore readiness-based process exit behavior in scripts/detect_env.py.")
    return ok("detect_env_exit_code", "detect_env.py exits based on readiness")


def check_init_project_skeleton(repo_root: Path) -> list[CheckResult]:
    checks: list[CheckResult] = []
    with tempfile.TemporaryDirectory(prefix="unsloth-buddy-static-") as tmp:
        tmp_path = Path(tmp)
        home = tmp_path / "home"
        home.mkdir()
        import os
        # Preserve PATH so Python can resolve imports and subprocesses normally.
        env = os.environ.copy() | {"HOME": str(home)}
        res = subprocess.run(
            [sys.executable, str(repo_root / "scripts" / "init_project.py"), "eval_probe"],
            cwd=tmp_path,
            env=env,
            text=True,
            capture_output=True,
            timeout=20,
        )
        if res.returncode != 0:
            return [fail("init_project_runs", f"init_project.py failed: {res.stderr.strip()}", fix="Fix scripts/init_project.py so it can create a project from a clean temp directory.")]
        project_dir = tmp_path / res.stdout.strip().splitlines()[-1]
        checks.append(ok("init_project_runs", f"created {project_dir.name}"))
        checks.append(has_required_project_skeleton(project_dir))
        progress = project_dir / "progress_log.md"
        if progress.exists():
            text = progress.read_text(encoding="utf-8")
            expected = ["0: Init", "1: Interview", "2: Data", "3: Env", "4: Train", "5: Eval", "7: Reflection"]
            missing = [phase for phase in expected if phase not in text]
            if missing:
                checks.append(fail(
                    "progress_log_phase_labels",
                    f"progress_log template missing lifecycle labels: {', '.join(missing)}",
                    severity="warn",
                    fix="Update scripts/init_project.py progress_log.md template to use the 7-phase lifecycle labels from SKILL.md/AGENTS.md.",
                ))
            else:
                checks.append(ok("progress_log_phase_labels", "progress log labels match lifecycle"))
    return checks


def check_skill_phase_language_consistency(repo_root: Path) -> list[CheckResult]:
    skill = (repo_root / "SKILL.md").read_text(encoding="utf-8")
    agents = (repo_root / "AGENTS.md").read_text(encoding="utf-8")
    expected_phases = ["0", "1", "2", "3", "4", "5", "5.5", "6", "6.5", "7"]
    checks = []
    for phase in expected_phases:
        skill_has = f"Phase {phase}" in skill
        agents_has = f"| {phase}." in agents or f"| {phase} " in agents
        if skill_has and agents_has:
            checks.append(ok(f"phase_present:{phase}", "present in SKILL.md and AGENTS.md"))
        else:
            checks.append(fail(
                f"phase_present:{phase}",
                f"SKILL.md={skill_has}, AGENTS.md={agents_has}",
                severity="warn",
                fix=f"Align Phase {phase} language between SKILL.md and AGENTS.md, or update the evaluator if the lifecycle intentionally changed.",
            ))
    return checks


def run_static_contract_checks(repo_root: Path) -> list[CheckResult]:
    checks: list[CheckResult] = [check_skill_frontmatter(repo_root)]
    checks.extend(check_required_paths(repo_root))
    checks.extend(check_referenced_paths(repo_root))
    checks.append(check_interview_two_question_contract(repo_root))
    checks.append(check_data_phase_prerequisites(repo_root))
    checks.append(check_detect_env_exit_code(repo_root))
    checks.extend(check_init_project_skeleton(repo_root))
    checks.extend(check_skill_phase_language_consistency(repo_root))
    return checks
