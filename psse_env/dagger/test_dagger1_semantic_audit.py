from __future__ import annotations

import copy
import unittest

from psse_env.dagger.dagger1_semantic_audit import (
    audit_dagger1_union_realizability,
    stratified_approximate_realizability,
)


def _rows(count: int = 24) -> list[dict]:
    rows = []
    for index in range(count):
        rows.append(
            {
                "example_id": f"row-{index}",
                "physical_root_fingerprint": f"root-{index}",
                "scenario_family": "measurement+parameter",
                "state_class": "clean_successful",
                "recovery_stratum": "post_failure_no_candidate",
                "policy_observation": {
                    "active_state_id": "active",
                    "remaining_anomaly_score": 1.0 + index / 1000.0,
                    "anomaly_threshold": 2.0,
                    "history_window": [],
                },
                "preferred_action": {
                    "tool": "run_wls",
                    "arguments": {"state_id": "active"},
                },
            }
        )
    return rows


class Dagger1SemanticAuditTests(unittest.TestCase):
    def test_stratified_gate_requires_real_comparison_coverage(self) -> None:
        report = stratified_approximate_realizability(
            _rows(), "recovery_stratum"
        )["post_failure_no_candidate"]
        self.assertTrue(report["comparison_coverage_passed"], report)
        self.assertTrue(report["release_gate_passed"], report)

        singleton = stratified_approximate_realizability(
            _rows(1), "recovery_stratum"
        )["post_failure_no_candidate"]
        self.assertFalse(singleton["comparison_coverage_passed"])
        self.assertFalse(singleton["release_gate_passed"])

    def test_union_gate_detects_exact_conflict_in_natural_population(self) -> None:
        natural = _rows()
        conflicting = copy.deepcopy(natural[0])
        conflicting["example_id"] = "conflict"
        conflicting["physical_root_fingerprint"] = "conflict-root"
        conflicting["preferred_action"] = {
            "tool": "get_measurement_context",
            "arguments": {"state_id": "active"},
        }
        report = audit_dagger1_union_realizability(
            [*natural, conflicting], natural
        )
        self.assertFalse(report["passed"])
        self.assertFalse(
            report["natural_teacher_realizability"]["passed"]
        )

    def test_recovery_projection_excludes_unlabeled_d0_population(self) -> None:
        d1 = _rows()
        d0 = []
        for index, source in enumerate(_rows()):
            row = copy.deepcopy(source)
            row["example_id"] = f"d0-{index}"
            row["physical_root_fingerprint"] = f"d0-root-{index}"
            row.pop("recovery_stratum")
            d0.append(row)
        report = audit_dagger1_union_realizability([*d0, *d1], [*d0, *d1])
        self.assertNotIn(
            "unknown",
            report[
                "approximate_teacher_realizability_by_recovery_stratum"
            ],
        )
        self.assertEqual(
            report["recovery_stratum_comparison_population"]["rows"],
            len(d1),
        )


if __name__ == "__main__":
    unittest.main()
