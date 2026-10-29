"""What the rollout's slim trace keeps of a parameter comparison (2026-10-29)."""
from __future__ import annotations

import unittest

from research.compound_errors.rollout import slim_trace


def _comparison_step() -> dict:
    candidates = [
        {"line_index": 3, "request_rank": 0, "execution_status": "success", "stage": "verification", "error_code": None,
         "verification_verdict": "ACCEPT_PARTIAL", "global_progress": 0.918, "max_normalized_residual": 3.96, "chi_square": 99.0},
        {"line_index": 6, "request_rank": 1, "execution_status": "failure", "stage": "correction", "error_code": "dispatch_error",
         "error_detail": "PermissionError", "verification_verdict": None},
    ]
    record = {"context_tool": "compare_parameter_candidates", "candidates": candidates, "ranked_line_indices": [3, 6],
              "best_line_index": 3, "internal_correction_calls": 2}
    return {"action": {"tool": "compare_parameter_candidates", "arguments": {"state_id": "episode:s1", "line_indices": [3, 6]}},
            "execution_status": "success", "error_code": None, "policy_tool_output": {"tool_metrics": record}}


class SlimTraceTests(unittest.TestCase):
    def test_a_comparison_keeps_its_ranking_and_every_candidates_fate(self) -> None:
        [step] = slim_trace([_comparison_step()])
        self.assertEqual(step["arguments"], {"line_indices": [3, 6]})
        self.assertEqual(step["comparison"]["ranked_lines"], [3, 6])
        self.assertEqual(step["comparison"]["best_line"], 3)
        # A failed fit keeps its error, so a refused case file is visible in the record.
        self.assertEqual(step["comparison"]["candidates"][1], {
            "line_index": 6, "execution_status": "failure", "stage": "correction", "error_code": "dispatch_error",
            "error_detail": "PermissionError", "verification_verdict": None})
        self.assertNotIn("chi_square", step["comparison"]["candidates"][0])


if __name__ == "__main__":
    unittest.main()
