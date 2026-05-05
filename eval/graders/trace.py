from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import CheckResult, fail, ok


@dataclass
class CommandEvent:
    index: int
    event_type: str
    command: str
    raw: dict[str, Any]


def load_events(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    events: list[dict[str, Any]] = []
    malformed: list[str] = []
    if not path.exists():
        return events, [f"missing trace file: {path}"]
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            malformed.append(f"line {line_no}: {exc}")
            continue
        if isinstance(value, dict):
            events.append(value)
        else:
            malformed.append(f"line {line_no}: expected object, got {type(value).__name__}")
    return events, malformed


def normalize_command(command: Any) -> str:
    if isinstance(command, list):
        text = " ".join(str(part) for part in command)
    elif isinstance(command, tuple):
        text = " ".join(str(part) for part in command)
    elif command is None:
        text = ""
    else:
        text = str(command)
    return re.sub(r"\s+", " ", text).strip()


def _candidate_command_values(event: dict[str, Any]) -> list[Any]:
    item = event.get("item") if isinstance(event.get("item"), dict) else {}
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    data = event.get("data") if isinstance(event.get("data"), dict) else {}
    return [
        event.get("command"),
        event.get("cmd"),
        item.get("command"),
        item.get("cmd"),
        payload.get("command"),
        data.get("command"),
    ]


def extract_commands(events: list[dict[str, Any]]) -> list[CommandEvent]:
    commands: list[CommandEvent] = []
    for index, event in enumerate(events):
        item = event.get("item") if isinstance(event.get("item"), dict) else {}
        event_type = str(event.get("type", ""))
        item_type = str(item.get("type", ""))
        looks_like_command = (
            item_type == "command_execution"
            or event_type in {"item.started", "item.completed", "command.started", "command.completed"}
            or "command" in event
        )
        if not looks_like_command:
            continue
        for candidate in _candidate_command_values(event):
            command = normalize_command(candidate)
            if command:
                commands.append(CommandEvent(index, event_type, command, event))
                break
    return commands


def command_contains(commands: list[CommandEvent], pattern: str) -> bool:
    return first_command_index(commands, pattern) is not None


def first_command_index(commands: list[CommandEvent], pattern: str) -> int | None:
    needle = pattern.lower()
    for command in commands:
        if needle in command.command.lower():
            return command.index
    return None


def check_no_prohibited_commands(commands: list[CommandEvent], prohibited: list[str]) -> CheckResult:
    hits = [
        f"{pattern!r} in {command.command!r}"
        for pattern in prohibited
        for command in commands
        if pattern.lower() in command.command.lower()
    ]
    if hits:
        return fail("no_prohibited_commands", "; ".join(hits), fix="Keep smoke prompts and skill behavior dry-run only; remove installs, training, deploy, upload, or network commands from this suite.")
    return ok("no_prohibited_commands", "no prohibited commands found")


def check_required_commands(commands: list[CommandEvent], patterns: list[str]) -> CheckResult:
    missing = [pattern for pattern in patterns if not command_contains(commands, pattern)]
    if missing:
        return fail("required_commands", f"missing command patterns: {', '.join(missing)}", fix="Update the skill flow or test fixture so the required command patterns appear in the trace.")
    return ok("required_commands", f"found required commands: {', '.join(patterns)}")


def check_no_command_before(
    commands: list[CommandEvent],
    prohibited_patterns: list[str],
    required_before_patterns: list[str],
    check_id: str,
) -> CheckResult:
    first_required = [
        first_command_index(commands, pattern)
        for pattern in required_before_patterns
    ]
    if any(index is None for index in first_required):
        return fail(check_id, f"missing prerequisite command(s): {', '.join(required_before_patterns)}", fix="Run the prerequisite lifecycle commands before the gated command, or adjust the grader if the prerequisite changed.")
    gate_index = max(index for index in first_required if index is not None)
    offenders = []
    for pattern in prohibited_patterns:
        index = first_command_index(commands, pattern)
        if index is not None and index < gate_index:
            offenders.append(pattern)
    if offenders:
        return fail(check_id, f"command(s) before prerequisites: {', '.join(offenders)}", fix="Reorder the lifecycle so gated commands run only after their required prerequisites.")
    return ok(check_id, "ordering satisfied")
