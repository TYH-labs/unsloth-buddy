from __future__ import annotations

from pathlib import Path

from .common import CheckResult, fail, ok
from .filesystem import find_project_dirs, has_required_project_skeleton
from .trace import CommandEvent, command_contains, first_command_index


PROHIBITED_SMOKE_COMMANDS = [
    "pip install",
    "uv pip install",
    "conda install",
    "npm install",
    "trainer.train",
    "python train.py",
    "python3 train.py",
    "huggingface-cli upload",
    "git push",
    "llama-server",
    "llamacpp.py deploy",
]


def check_expected_activation(expected: bool, workspace: Path, final_text: str = "") -> CheckResult:
    projects = find_project_dirs(workspace)
    activated = bool(projects)
    if not activated:
        lower = final_text.lower()
        activated = "fine-tun" in lower and ("task" in lower and "data" in lower)
    if expected and not activated:
        return fail("expected_activation", "expected lifecycle activation, but no project or interview signal was found", fix="Tighten AGENTS.md/SKILL.md activation language so this prompt enters Phase 0/1, then rerun the case.")
    if not expected and activated:
        return fail("expected_activation", "negative-control prompt activated the fine-tuning lifecycle", fix="Narrow the skill description or add non-trigger guidance so adjacent non-training requests do not start Phase 0.")
    return ok("expected_activation", f"activation={activated}, expected={expected}")


def check_phase0_init(workspace: Path) -> CheckResult:
    projects = find_project_dirs(workspace)
    if len(projects) != 1:
        return fail("phase0_init", f"expected exactly one project dir, found {len(projects)}", fix="Fix Phase 0 instructions or scripts/init_project.py usage so exactly one dated project directory is created.")
    return has_required_project_skeleton(projects[0])


def check_interview_contract(project_dir: Path, final_text: str = "") -> CheckResult:
    brief = project_dir / "project_brief.md"
    if brief.exists():
        text = brief.read_text(encoding="utf-8").lower()
        required = ["problem", "domain", "audience", "model", "hardware", "training method"]
        missing = [token for token in required if token not in text]
        if missing:
            return fail("interview_contract", f"project_brief.md missing signals: {', '.join(missing)}", fix="Update Phase 1 behavior to write a complete project_brief.md with task, domain/audience, model, hardware, and method.")
        return ok("interview_contract", "project_brief.md contains required interview signals")
    lower = final_text.lower()
    if "what exactly" in lower and "dataset" in lower:
        return ok("interview_contract", "final response asks the required interview questions")
    return fail("interview_contract", "no project_brief.md or 2-question interview signal found", fix="Ensure the agent reads sub-skills/interview.md and either asks the 2-question interview or writes project_brief.md before Phase 2.")


def check_env_before_training(commands: list[CommandEvent]) -> CheckResult:
    training_patterns = ["python train.py", "python3 train.py", "trainer.train"]
    first_train = min(
        (index for pattern in training_patterns if (index := first_command_index(commands, pattern)) is not None),
        default=None,
    )
    if first_train is None:
        return ok("env_before_training", "no training command found")
    system_index = first_command_index(commands, "detect_system.py")
    env_index = first_command_index(commands, "detect_env.py")
    if system_index is None or env_index is None:
        return fail("env_before_training", "training command found before both env checks were present", fix="Block Phase 4 until detect_system.py and detect_env.py have both run and detect_env reports readiness.")
    if first_train < max(system_index, env_index):
        return fail("env_before_training", "training command appears before env checks", fix="Reorder lifecycle execution so Phase 3 completes before any train.py creation or execution.")
    return ok("env_before_training", "env checks appear before training")


def check_data_before_training(commands: list[CommandEvent], project_dir: Path | None) -> CheckResult:
    train_index = min(
        (index for pattern in ["python train.py", "python3 train.py", "trainer.train"] if (index := first_command_index(commands, pattern)) is not None),
        default=None,
    )
    if train_index is None:
        return ok("data_before_training", "no training command found")
    if project_dir and (project_dir / "data_strategy.md").exists():
        return ok("data_before_training", "data_strategy.md exists before scoring training")
    data_signal = first_command_index(commands, "data_strategy.md")
    if data_signal is not None and data_signal < train_index:
        return ok("data_before_training", "data strategy command appears before training")
    return fail("data_before_training", "training command appears without prior data_strategy.md signal", fix="Require Phase 2 to produce data_strategy.md and validate the TRL schema before Phase 4 training.")


def check_demo_after_eval(project_dir: Path, commands: list[CommandEvent]) -> CheckResult:
    demo_signal = command_contains(commands, "demo_builder.md") or (project_dir / "demos").exists()
    if not demo_signal:
        return ok("demo_after_eval", "no demo signal found")
    if (project_dir / "logs" / "eval_compare.log").exists():
        return ok("demo_after_eval", "demo has compare log input")
    if command_contains(commands, "eval.py --compare"):
        return ok("demo_after_eval", "compare mode command appears before demo grading")
    return fail("demo_after_eval", "demo signal found without eval compare output", fix="Gate demo generation on logs/eval_compare.log or run eval.py --compare before invoking sub-skills/demo_builder.md.")


def check_reflect_only_at_end(commands: list[CommandEvent], project_dir: Path | None = None) -> CheckResult:
    write_index = first_command_index(commands, "reflect.py --write")
    if write_index is None:
        return ok("reflect_only_at_end", "no reflection write found")
    completion_signals = ["llamacpp.py deploy", "outputs/adapters", "eval_compare.log", "Phase 7"]
    if any(command_contains(commands, signal) for signal in completion_signals):
        return ok("reflect_only_at_end", "reflection write has a completion signal")
    if project_dir and (project_dir / "logs" / "eval_compare.log").exists():
        return ok("reflect_only_at_end", "reflection write has fixture compare output")
    return fail("reflect_only_at_end", "reflect.py --write appears without project completion signal", fix="Move reflect.py --write to Phase 7 after eval/export completion evidence exists; use --dry-run earlier if needed.")
