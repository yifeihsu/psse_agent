from __future__ import annotations

import json
import unittest

from psse_env.sft.gates import GateError
from psse_env.sft.research_smoke import (
    PROBE_STAGES,
    _closed_loop_slice,
    _probe_policy,
    select_probe_rows,
    validate_selected_probe_rows,
)


def _row(
    example_id: str,
    tool: str,
    *,
    history: int = 0,
    candidate: bool = False,
    stratum: str = "",
) -> dict:
    state = {
        "history_window": [{"step": index} for index in range(history)],
        "has_open_candidate": candidate,
        "candidate_state_id": "candidate" if candidate else None,
    }
    return {
        "example_id": example_id,
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": json.dumps({"state": state}, sort_keys=True)},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": tool,
                            "arguments": {"state_id": "active"},
                        },
                    }
                ],
            },
        ],
        "metadata": {"recovery_stratum": stratum},
    }


def _probe_corpus() -> list[dict]:
    return [
        _row("initial", "wls_from_path"),
        _row("context", "get_measurement_context", history=1),
        _row("measurement", "correct_measurements_from_path", history=2),
        _row("parameter", "correct_parameters_from_path", history=2),
        _row("verify", "wls_from_path", history=3, candidate=True),
        _row("commit", "commit_state", history=3, candidate=True),
        _row("rollback", "rollback_state", history=3, candidate=True),
        _row(
            "unsupported",
            "get_topology_context",
            history=3,
            stratum="unsupported_correction_recovery",
        ),
        _row(
            "invalid",
            "wls_from_path",
            history=3,
            stratum="invalid_precondition_repair",
        ),
        _row("long", "finalize_diagnosis", history=4),
    ]


class ResearchSmokeSelectionTests(unittest.TestCase):
    def test_selects_exact_ten_probe_stages(self) -> None:
        selected = select_probe_rows(_probe_corpus())
        self.assertEqual(
            tuple(row["_research_probe_stage"] for row in selected),
            PROBE_STAGES,
        )
        self.assertEqual(len({row["example_id"] for row in selected}), 10)
        self.assertEqual(validate_selected_probe_rows(selected), selected)

    def test_missing_stage_fails_before_gpu_generation(self) -> None:
        with self.assertRaisesRegex(GateError, "candidate_rollback"):
            select_probe_rows(
                [row for row in _probe_corpus() if row["example_id"] != "rollback"]
            )

    def test_probe_report_distinguishes_schema_valid_and_exact(self) -> None:
        selected = select_probe_rows(_probe_corpus())

        class Policy:
            last_action_metrics = {
                "prompt_tokens": 10,
                "generated_tokens": 5,
                "hit_max_new_tokens": False,
                "truncated_input_tokens": 0,
                "repetition_loop_detected": False,
            }

            def act(self, observation):
                del observation
                return {
                    "tool": "wls_from_path",
                    "arguments": {"state_id": "active"},
                }

        report = _probe_policy(Policy(), selected)
        self.assertEqual(report["schema_valid_single_calls"], 10)
        self.assertEqual(report["exact_action_matches"], 3)
        self.assertEqual(report["maximum_token_hits"], 0)
        self.assertEqual(report["repetition_checks_available"], 10)
        self.assertEqual(report["repetition_loops"], 0)

    def test_closed_loop_slice_uses_unique_roots(self) -> None:
        suites = {
            "standard_success": [
                {"grouping": {"physical_root_fingerprint": "root-a"}}
            ],
            "invalid_action_recovery": [
                {"grouping": {"physical_root_fingerprint": "root-b"}}
            ],
            "forced_error_recovery": [
                {"grouping": {"physical_root_fingerprint": "root-c"}}
            ],
        }
        selected = _closed_loop_slice(suites, count=3)
        self.assertEqual(sum(len(rows) for rows in selected.values()), 3)


if __name__ == "__main__":
    unittest.main()
