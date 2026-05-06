from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graders.common import fail  # noqa: E402
from graders.report import case_payload, write_summary  # noqa: E402


class ReportTests(unittest.TestCase):
    def test_summary_includes_fix_suggestions(self) -> None:
        checks = [
            fail(
                "example_check",
                "something broke",
                fix="Change the fixture and rerun the eval.",
            )
        ]
        payload = case_payload("case-1", "static", checks)
        with tempfile.TemporaryDirectory() as tmp:
            summary = Path(tmp) / "summary.md"
            write_summary(summary, [payload])
            text = summary.read_text(encoding="utf-8")
        self.assertIn("## Fix Suggestions", text)
        self.assertIn("Change the fixture and rerun the eval.", text)


if __name__ == "__main__":
    unittest.main()

