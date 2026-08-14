from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

from psse_env.dagger.splits import (
    StratifiedSplitError,
    audit_physical_split_disjointness,
    audit_stratified_split_coverage,
    grouped_scenario_split,
    physical_root_fingerprint,
    scenario_split_stratum,
    stratified_grouped_scenario_split,
)


def _split_root(
    index: int,
    *,
    family: str = "measurement",
    case_id: str = "case14",
    cardinality: int = 1,
    source_tier: str = "measured",
) -> dict[str, object]:
    return {
        "scenario_id": f"root_{family}_{case_id}_{source_tier}_{index}",
        "root_scenario_id": f"root_{family}_{case_id}_{source_tier}_{index}",
        "physical_root_fingerprint": (
            f"physical_v3_{family}_{case_id}_{source_tier}_{index}"
        ),
        "case_id": case_id,
        "error_family_combination": family,
        "error_cardinality": cardinality,
        "source_tier": source_tier,
    }


def _ownership(
    splits: dict[str, list[dict[str, object]]],
) -> dict[str, str]:
    return {
        str(row["physical_root_fingerprint"]): split
        for split, rows in splits.items()
        for row in rows
    }


class PhysicalRootSplitTests(unittest.TestCase):
    def test_fingerprint_ignores_ids_but_changes_with_physics_and_telemetry(self) -> None:
        base = {
            "scenario_id": "root_a",
            "case": "case14",
            "measurements": [1.0, 2.0],
            "clean_measurements": [1.0, 1.0],
            "true_measurement_errors": [{"index": 1, "clean": 1.0}],
            "metadata": {"measurement_covariance": [0.1, 0.1]},
        }
        renamed = {**base, "scenario_id": "root_b", "root_scenario_id": "root_b"}
        self.assertEqual(
            physical_root_fingerprint(base), physical_root_fingerprint(renamed)
        )
        changed = {**renamed, "measurements": [1.0, 2.1]}
        self.assertNotEqual(
            physical_root_fingerprint(base), physical_root_fingerprint(changed)
        )

    def test_hif_fingerprint_ignores_derived_diagnostics_but_keeps_physics(self) -> None:
        truth = {
            "branch_row0": 3,
            "phase": "A",
            "split_ratio": 0.4,
            "r_hif_ohm": 120.0,
        }
        base = {
            "case": "case14",
            "measurements": [1.0, 2.0],
            "metadata": {
                "nlm_diagnostic": {
                    "detected_top1": True,
                    "top_hif_groups": [{"branch_row0": 3}],
                },
                "hif_scan_window": {
                    "scan_window_path": "/tmp/corpus-a/event-1",
                    "scans": [
                        {
                            "scan_index": 7,
                            "z_obs": [1.0, 2.0],
                            "z_clean": [0.9, 1.9],
                        }
                    ],
                    "window_metadata": {"source_kind": "benchmark-a"},
                },
            },
            "hidden_truth": {"true_hif_errors": [truth]},
        }
        changed_diagnostic = copy.deepcopy(base)
        changed_diagnostic["metadata"]["nlm_diagnostic"] = {
            "detected_top1": False,
            "top_hif_groups": [{"branch_row0": 18}],
        }
        changed_diagnostic["metadata"]["hif_scan_window"]["scan_window_path"] = (
            "/tmp/corpus-b/renamed"
        )
        changed_diagnostic["metadata"]["hif_scan_window"]["scans"][0][
            "z_clean"
        ] = [0.0, 0.0]
        self.assertEqual(
            physical_root_fingerprint(base),
            physical_root_fingerprint(changed_diagnostic),
        )

        changed_telemetry = copy.deepcopy(base)
        changed_telemetry["metadata"]["hif_scan_window"]["scans"][0][
            "z_obs"
        ][1] = 2.1
        self.assertNotEqual(
            physical_root_fingerprint(base),
            physical_root_fingerprint(changed_telemetry),
        )

        changed_fault = copy.deepcopy(base)
        changed_fault["hidden_truth"]["true_hif_errors"][0]["branch_row0"] = 4
        self.assertNotEqual(
            physical_root_fingerprint(base), physical_root_fingerprint(changed_fault)
        )

    def test_promoted_and_hidden_diagnostic_truth_have_one_root_identity(self) -> None:
        base = {"case": "case14", "measurements": [1.0, 2.0]}
        errors = [{"branch_row0": 2, "phase": "B"}]
        hidden = {**base, "hidden_truth": {"true_hif_errors": errors}}
        promoted = {**base, "true_hif_errors": errors}
        self.assertEqual(
            physical_root_fingerprint(hidden), physical_root_fingerprint(promoted)
        )

    def test_distinct_root_ids_with_same_physics_cannot_cross_splits(self) -> None:
        fingerprint = physical_root_fingerprint(
            {"case": "case14", "measurements": [1.0], "noise_seed": 7}
        )
        rows = [
            {
                "example_id": f"row_{index}",
                "root_scenario_id": root,
                "physical_root_fingerprint": fingerprint,
            }
            for index, root in enumerate(("root_a", "root_b"))
        ]
        splits = grouped_scenario_split(
            rows, train_fraction=0.34, validation_fraction=0.33, seed=11
        )
        owners = [name for name, split_rows in splits.items() if split_rows]
        self.assertEqual(len(owners), 1)
        self.assertTrue(audit_physical_split_disjointness(splits)["passed"])

    def test_copied_content_under_different_paths_has_same_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            left = root / "case_a.m"
            right = root / "renamed_case.m"
            left.write_bytes(b"same physical case\n")
            right.write_bytes(left.read_bytes())
            first = physical_root_fingerprint(
                {"case_path": str(left), "measurements": [1.0, 2.0]}
            )
            second = physical_root_fingerprint(
                {"case_path": str(right), "measurements": [1.0, 2.0]}
            )
        self.assertEqual(first, second)

    def test_derived_signature_text_does_not_change_fingerprint(self) -> None:
        base = {"case": "case14", "measurements": [1.0, 2.0]}
        first = {**base, "unresolved_signatures": ["measurement_bias"]}
        second = {**base, "unresolved_signatures": ["worded differently"]}
        self.assertEqual(
            physical_root_fingerprint(first), physical_root_fingerprint(second)
        )

    def test_nonexistent_runtime_binding_path_does_not_change_fingerprint(self) -> None:
        base = {"case": "case14", "measurements": [1.0, 2.0]}
        first = {
            **base,
            "metadata": {"scan_window_path": "/runtime/branch_a/scans.json"},
        }
        second = {
            **base,
            "metadata": {"scan_window_path": "/runtime/branch_b/scans.json"},
        }
        self.assertEqual(
            physical_root_fingerprint(first), physical_root_fingerprint(second)
        )

    def test_audit_detects_manual_fingerprint_leak(self) -> None:
        row = {"example_id": "row", "physical_root_fingerprint": "physical_v1_x"}
        report = audit_physical_split_disjointness(
            {"train": [row], "validation": [{**row, "example_id": "row2"}]}
        )
        self.assertFalse(report["passed"])
        self.assertIn("physical_v1_x", report["overlapping_fingerprints"])


class StratifiedPhysicalRootSplitTests(unittest.TestCase):
    def test_stratum_aliases_and_truth_lists_are_canonicalized(self) -> None:
        stratum = scenario_split_stratum(
            {
                "case": "case14",
                "scenario_family": "topology+measurement",
                "metadata": {"source_tier": "engineering"},
                "true_measurement_errors": [{"index": 1}],
                "hidden_truth": {
                    "true_measurement_errors": [{"index": 1}],
                    "true_topology_errors": [{"line_index1": 2}],
                },
            }
        )
        self.assertEqual(stratum.case_id, "case14")
        self.assertEqual(
            stratum.error_family_combination, "measurement+topology"
        )
        self.assertEqual(stratum.error_cardinality, 2)
        self.assertEqual(stratum.source_tier, "engineering")

    def test_deterministic_assignment_is_grouped_and_stratified(self) -> None:
        rows: list[dict[str, object]] = []
        for source_tier in ("measured", "synthetic"):
            for index in range(8):
                root = _split_root(index, source_tier=source_tier)
                rows.extend(
                    [
                        {**root, "example_id": f"{root['scenario_id']}_a"},
                        {**root, "example_id": f"{root['scenario_id']}_b"},
                    ]
                )

        first = stratified_grouped_scenario_split(
            rows, train_fraction=0.5, validation_fraction=0.25, seed=17
        )
        second = stratified_grouped_scenario_split(
            reversed(rows), train_fraction=0.5, validation_fraction=0.25, seed=17
        )
        self.assertEqual(_ownership(first), _ownership(second))
        self.assertTrue(audit_physical_split_disjointness(first)["passed"])

        for source_tier in ("measured", "synthetic"):
            actual = {
                split: len(
                    {
                        row["physical_root_fingerprint"]
                        for row in split_rows
                        if row["source_tier"] == source_tier
                    }
                )
                for split, split_rows in first.items()
            }
            self.assertEqual(actual, {"train": 4, "validation": 2, "test": 2})

    def test_critical_family_minima_rebalance_stratum_quotas(self) -> None:
        rows = [
            _split_root(index, source_tier="measured" if index < 6 else "synthetic")
            for index in range(12)
        ]
        splits = stratified_grouped_scenario_split(
            rows,
            train_fraction=0.8,
            validation_fraction=0.1,
            seed=23,
            critical_families=["measurement"],
            minimum_roots_per_critical_family={"validation": 3, "test": 3},
        )
        report = audit_stratified_split_coverage(
            splits,
            critical_families=["measurement"],
            minimum_roots_per_critical_family={"validation": 3, "test": 3},
        )
        self.assertTrue(report["passed"], report)
        self.assertEqual(
            report["root_counts_by_family_and_split"]["measurement"],
            {"train": 6, "validation": 3, "test": 3},
        )

    def test_infeasible_critical_family_coverage_fails_closed(self) -> None:
        rows = [_split_root(index) for index in range(9)]
        with self.assertRaises(StratifiedSplitError) as caught:
            stratified_grouped_scenario_split(
                rows,
                seed=2,
                critical_families=["measurement"],
                minimum_roots_per_critical_family={
                    "validation": 5,
                    "test": 5,
                },
            )
        diagnostic = caught.exception.diagnostics[
            "infeasible_critical_family_coverage"
        ][0]
        self.assertEqual(diagnostic["available_independent_roots"], 9)
        self.assertEqual(diagnostic["required_independent_roots"], 10)

    def test_missing_source_tier_fails_closed_with_row_diagnostic(self) -> None:
        row = _split_root(0)
        del row["source_tier"]
        with self.assertRaises(StratifiedSplitError) as caught:
            stratified_grouped_scenario_split([row])
        diagnostic = caught.exception.diagnostics
        self.assertEqual(diagnostic["input_error_count"], 1)
        self.assertEqual(
            diagnostic["input_errors"][0]["code"], "invalid_split_stratum"
        )
        self.assertIn("source_tier", diagnostic["input_errors"][0]["error"])

    def test_one_physical_root_cannot_carry_multiple_strata(self) -> None:
        first = _split_root(0)
        second = {**first, "source_tier": "synthetic", "example_id": "branch_b"}
        with self.assertRaises(StratifiedSplitError) as caught:
            stratified_grouped_scenario_split([first, second])
        self.assertEqual(
            caught.exception.diagnostics["input_errors"][0]["code"],
            "physical_root_has_multiple_strata",
        )

    def test_coverage_audit_reports_manual_family_deficit(self) -> None:
        validation = _split_root(0)
        report = audit_stratified_split_coverage(
            {"train": [], "validation": [validation], "test": []},
            critical_families=["measurement"],
            minimum_roots_per_critical_family={"validation": 1, "test": 1},
        )
        self.assertFalse(report["passed"])
        self.assertEqual(
            report["coverage_deficits"],
            [
                {
                    "family": "measurement",
                    "split": "test",
                    "required": 1,
                    "actual": 0,
                    "deficit": 1,
                }
            ],
        )

    def test_coverage_audit_rejects_one_root_with_multiple_strata(self) -> None:
        first = _split_root(0)
        report = audit_stratified_split_coverage(
            {
                "train": [first, {**first, "source_tier": "synthetic"}],
                "validation": [],
                "test": [],
            }
        )
        self.assertFalse(report["passed"])
        self.assertIn(
            first["physical_root_fingerprint"], report["inconsistent_root_strata"]
        )


if __name__ == "__main__":
    unittest.main()
