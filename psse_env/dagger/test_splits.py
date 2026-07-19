from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from psse_env.dagger.splits import (
    audit_physical_split_disjointness,
    grouped_scenario_split,
    physical_root_fingerprint,
)


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


if __name__ == "__main__":
    unittest.main()
