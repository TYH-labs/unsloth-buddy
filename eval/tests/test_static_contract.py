from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graders.common import suite_passed  # noqa: E402
from graders.static_contract import (  # noqa: E402
    check_data_phase_prerequisites,
    check_detect_env_exit_code,
    check_interview_two_question_contract,
    check_referenced_paths,
    run_static_contract_checks,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


class StaticContractTests(unittest.TestCase):
    def test_core_static_checks_pass(self) -> None:
        self.assertTrue(check_interview_two_question_contract(REPO_ROOT).passed)
        self.assertTrue(check_data_phase_prerequisites(REPO_ROOT).passed)
        self.assertTrue(check_detect_env_exit_code(REPO_ROOT).passed)

    def test_missing_eval_template_reference_is_warning_not_blocker(self) -> None:
        checks = check_referenced_paths(REPO_ROOT)
        eval_template = [check for check in checks if check.id == "referenced_path:scripts/eval_template.py"]
        self.assertEqual(len(eval_template), 1)
        self.assertFalse(eval_template[0].passed)
        self.assertEqual(eval_template[0].severity, "warn")

    def test_full_static_suite_has_no_blocking_failures(self) -> None:
        checks = run_static_contract_checks(REPO_ROOT)
        failures = [check for check in checks if not check.passed and check.severity == "fail"]
        self.assertEqual(failures, [])
        self.assertTrue(suite_passed(checks))


if __name__ == "__main__":
    unittest.main()

