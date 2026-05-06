from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graders.filesystem import assert_no_root_training_artifacts, find_project_dirs, has_required_project_skeleton  # noqa: E402
from graders.lifecycle import check_data_before_training, check_env_before_training, check_expected_activation  # noqa: E402
from graders.trace import extract_commands  # noqa: E402


class FilesystemLifecycleTests(unittest.TestCase):
    def _make_project(self, workspace: Path) -> Path:
        project = workspace / "demo_2026_05_05"
        for rel in ["data", "outputs/adapters", "logs"]:
            (project / rel).mkdir(parents=True, exist_ok=True)
        for rel in ["gaslamp.md", "memory.md", "progress_log.md", "reflect.py", "add_reflect_hint.py"]:
            (project / rel).write_text("fixture\n", encoding="utf-8")
        return project

    def test_project_skeleton_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            project = self._make_project(workspace)
            self.assertEqual(find_project_dirs(workspace, date="2026_05_05"), [project])
            self.assertTrue(has_required_project_skeleton(project).passed)
            self.assertTrue(check_expected_activation(True, workspace).passed)

    def test_root_training_artifact_detection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "train.py").write_text("bad root artifact\n", encoding="utf-8")
            result = assert_no_root_training_artifacts(workspace)
        self.assertFalse(result.passed)
        self.assertIn("train.py", result.notes)

    def test_env_before_training_passes_when_no_training(self) -> None:
        commands = extract_commands([
            {"type": "item.completed", "item": {"type": "command_execution", "command": "python3 scripts/detect_system.py"}}
        ])
        self.assertTrue(check_env_before_training(commands).passed)

    def test_data_before_training_requires_strategy(self) -> None:
        commands = extract_commands([
            {"type": "item.completed", "item": {"type": "command_execution", "command": "python train.py"}}
        ])
        with tempfile.TemporaryDirectory() as tmp:
            project = self._make_project(Path(tmp))
            result = check_data_before_training(commands, project)
        self.assertFalse(result.passed)


if __name__ == "__main__":
    unittest.main()

