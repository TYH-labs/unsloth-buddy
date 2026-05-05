from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graders.trace import (  # noqa: E402
    check_no_command_before,
    check_no_prohibited_commands,
    command_contains,
    extract_commands,
    first_command_index,
    load_events,
)


class TraceTests(unittest.TestCase):
    def test_extracts_command_shapes(self) -> None:
        events = [
            {"type": "item.started", "item": {"type": "command_execution", "command": "python3 scripts/init_project.py demo"}},
            {"type": "item.completed", "item": {"type": "command_execution", "command": ["python3", "scripts/detect_env.py"]}},
            {"type": "command.completed", "command": "python train.py"},
            {"type": "message", "content": "done"},
        ]
        commands = extract_commands(events)
        self.assertEqual([cmd.command for cmd in commands], [
            "python3 scripts/init_project.py demo",
            "python3 scripts/detect_env.py",
            "python train.py",
        ])
        self.assertTrue(command_contains(commands, "detect_env.py"))
        self.assertEqual(first_command_index(commands, "train.py"), 2)

    def test_prohibited_command_check(self) -> None:
        commands = extract_commands([
            {"type": "item.completed", "item": {"type": "command_execution", "command": "uv pip install unsloth"}}
        ])
        result = check_no_prohibited_commands(commands, ["uv pip install"])
        self.assertFalse(result.passed)
        self.assertEqual(result.severity, "fail")

    def test_ordering_check_detects_training_before_env(self) -> None:
        commands = extract_commands([
            {"type": "item.completed", "item": {"type": "command_execution", "command": "python train.py"}},
            {"type": "item.completed", "item": {"type": "command_execution", "command": "python scripts/detect_system.py"}},
            {"type": "item.completed", "item": {"type": "command_execution", "command": "python scripts/detect_env.py"}},
        ])
        result = check_no_command_before(
            commands,
            prohibited_patterns=["python train.py"],
            required_before_patterns=["detect_system.py", "detect_env.py"],
            check_id="env_before_training",
        )
        self.assertFalse(result.passed)

    def test_load_events_preserves_malformed_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            trace.write_text(json.dumps({"type": "message"}) + "\nnot-json\n", encoding="utf-8")
            events, malformed = load_events(trace)
        self.assertEqual(len(events), 1)
        self.assertEqual(len(malformed), 1)


if __name__ == "__main__":
    unittest.main()

