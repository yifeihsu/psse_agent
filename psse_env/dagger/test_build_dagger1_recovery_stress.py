from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Mapping

from psse_env.dagger.build_dagger1_recovery_stress import (
    RECOVERY_STRESS_DISTINCT_ROOT_COUNT,
    RECOVERY_STRESS_EPISODE_COUNT,
    RECOVERY_STRESS_ROOTS_PER_STRATUM,
    _source_bindings,
    build_recovery_stress_payload,
)
from psse_env.dagger.evaluator import (
    RECOVERY_STRESS_SUITES,
    RECOVERY_STRESS_SUITE_TO_STRATUM,
    _validate_recovery_stress_for_study,
)
from psse_env.dagger.root_sets import root_set_digest
from psse_env.dagger.test_evaluator import (
    _release_partitioned_resolved_scenario,
)
from psse_env.sft.provenance import file_sha256, stable_json_sha256


def _development_row(family: str, ordinal: int) -> dict[str, Any]:
    row = _release_partitioned_resolved_scenario()
    root = f"physical_v3_{ordinal:064x}"
    row["execution"]["scenario_id"] = f"stress-parent-{ordinal:03d}"
    row["execution"]["measurements"] = [0.0] * 8
    row["execution"]["measurements"][7] = 9.0
    row["audit"]["truth"]["true_measurement_errors"] = [
        {"index": 7, "clean": 1.0, "observed": 9.0}
    ]
    row["grouping"]["root_scenario_id"] = f"stress-parent-{ordinal:03d}"
    row["grouping"]["physical_root_fingerprint"] = root
    row["grouping"]["scenario_family"] = family
    row["grouping"]["error_cardinality"] = (
        2 if family == "measurement+parameter" else 3
    )
    return row


def _rows() -> list[dict[str, Any]]:
    return [
        *[_development_row("multi_measurement", index) for index in range(1, 13)],
        *[
            _development_row("measurement+parameter", index)
            for index in range(101, 113)
        ],
        *[_development_row("parameter", index) for index in range(201, 207)],
    ]


def _passing_validator(
    suite: str,
    scenario: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "passed": True,
        "suite": suite,
        "recovery_stratum": RECOVERY_STRESS_SUITE_TO_STRATUM[suite],
        "physical_root_fingerprint": scenario["grouping"][
            "physical_root_fingerprint"
        ],
        "pre_policy_step_count": 1,
        "expected_action_tool": "test-action",
        "expected_action_sha256": "a" * 64,
    }


class RecoveryStressBuilderTests(unittest.TestCase):
    def test_exact_seven_cell_payload_uses_twenty_parent_roots(self) -> None:
        payload, report = build_recovery_stress_payload(
            _rows(),
            validator=_passing_validator,
        )
        self.assertEqual(tuple(payload), RECOVERY_STRESS_SUITES)
        self.assertEqual(
            sum(len(rows) for rows in payload.values()),
            RECOVERY_STRESS_EPISODE_COUNT,
        )
        self.assertTrue(
            all(
                len(rows) == RECOVERY_STRESS_ROOTS_PER_STRATUM
                for rows in payload.values()
            )
        )
        roots = {
            row["grouping"]["physical_root_fingerprint"]
            for rows in payload.values()
            for row in rows
        }
        self.assertEqual(len(roots), RECOVERY_STRESS_DISTINCT_ROOT_COUNT)
        self.assertEqual(len(report["validation_records"]), 70)
        self.assertEqual(report["rejected_candidates"], [])
        self.assertTrue(
            all(
                row["grouping"]["split"] == "dagger_recovery_stress"
                for rows in payload.values()
                for row in rows
            )
        )
        self.assertTrue(
            all(
                "evaluation_intervention" in row["audit"]
                for rows in payload.values()
                for row in rows
            )
        )

    def test_candidate_must_pass_every_cell_for_its_family(self) -> None:
        rejected_root = _rows()[12]["grouping"][
            "physical_root_fingerprint"
        ]

        def validator(
            suite: str,
            scenario: Mapping[str, Any],
        ) -> dict[str, Any]:
            if (
                scenario["grouping"]["physical_root_fingerprint"]
                == rejected_root
                and suite == "recovery_rejected_candidate_rollback"
            ):
                raise RuntimeError("physical rejection setup did not validate")
            return _passing_validator(suite, scenario)

        payload, report = build_recovery_stress_payload(
            _rows(),
            validator=validator,
        )
        selected_mixed = {
            row["grouping"]["physical_root_fingerprint"]
            for row in payload[
                "recovery_measurement_parameter_sequential_handoff"
            ]
        }
        self.assertNotIn(rejected_root, selected_mixed)
        self.assertEqual(len(selected_mixed), 10)
        self.assertEqual(len(report["rejected_candidates"]), 1)

    def test_shortfall_fails_closed_without_a_partial_suite(self) -> None:
        rows = [
            row
            for row in _rows()
            if row["grouping"]["scenario_family"] != "measurement+parameter"
        ]
        rows.extend(
            copy.deepcopy(_development_row("measurement+parameter", index))
            for index in range(101, 110)
        )
        with self.assertRaisesRegex(
            RuntimeError,
            "insufficient validated measurement\\+parameter stress roots",
        ):
            build_recovery_stress_payload(
                rows,
                validator=_passing_validator,
            )

    def test_study_binding_recomputes_suite_manifest_and_physical_records(
        self,
    ) -> None:
        payload, selection = build_recovery_stress_payload(
            _rows(),
            validator=_passing_validator,
        )
        roots = {
            row["grouping"]["physical_root_fingerprint"]
            for rows in payload.values()
            for row in rows
        }
        repo_root = Path(__file__).resolve().parents[2]
        reviewed = "b" * 40
        frozen_hash = "c" * 64
        with tempfile.TemporaryDirectory() as directory:
            suite_path = Path(directory) / "stress.json"
            suite_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            input_bindings = {
                "development_holdout": "1" * 64,
                "development_holdout_manifest": "2" * 64,
                "development_holdout_generator_report": "3" * 64,
                "d0_raw": "4" * 64,
                "d0_generation_provenance": "5" * 64,
                "d0_manifest": "6" * 64,
                "d1_training_scenarios": "7" * 64,
                "d1_training_manifest": "8" * 64,
                "recovery_probes": "9" * 64,
                "recovery_probe_manifest": "a" * 64,
                "frozen_suite": frozen_hash,
            }
            protected = {
                "d0": set(),
                "d1_training": set(),
                "recovery_probes": set(),
                "frozen_evaluation": set(),
            }
            manifest = {
                "schema_version": 1,
                "scenario_schema_version": 1,
                "artifact_type": "dagger1_recovery_stress_suite",
                "contract": "dagger1_root_disjoint_recovery_stress_v1",
                "source_state": {
                    "release_eligible_source": True,
                    "source_commit": reviewed,
                },
                "source_bindings": _source_bindings(repo_root),
                "suite_format": "evaluation_suite_mapping_v1",
                "suite_names": list(RECOVERY_STRESS_SUITES),
                "split": "dagger_recovery_stress",
                "evaluator_seed": 20260723,
                "rows": 70,
                "rows_by_suite": {
                    name: 10 for name in sorted(RECOVERY_STRESS_SUITES)
                },
                "distinct_physical_roots": 20,
                "distinct_roots_by_suite": {
                    name: 10 for name in sorted(RECOVERY_STRESS_SUITES)
                },
                "minimum_distinct_roots_per_stratum": 10,
                "development_parent_root_count": 30,
                "development_parent_subset": True,
                "training_eligible": False,
                "model_selection_eligible": False,
                "recovery_test_evidence_eligible": True,
                "natural_coverage_eligible": False,
                "probe_training_root_overlap": [],
                "protected_root_overlap": {
                    name: [] for name in sorted(protected)
                },
                "root_set_sha256": {
                    "stress": root_set_digest(roots),
                    "development_parent": "b" * 64,
                    **{
                        name: root_set_digest(value)
                        for name, value in sorted(protected.items())
                    },
                },
                "input_bindings": input_bindings,
                "input_bindings_sha256": stable_json_sha256(input_bindings),
                "selection": selection,
                "output_sha256": file_sha256(suite_path),
            }
            manifest_path = Path(directory) / "stress.manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            binding = _validate_recovery_stress_for_study(
                suite_path=suite_path,
                recovery_stress_manifest_path=manifest_path,
                study_manifest={
                    "bindings": {
                        "evaluation": {"suite_sha256": frozen_hash}
                    }
                },
                reviewed_source_commit=reviewed,
                repo_root=repo_root,
            )
        self.assertEqual(binding["recovery_stress_physical_roots"], 20)
        self.assertEqual(binding["recovery_stress_episode_count"], 70)
        self.assertEqual(
            binding["recovery_stress_root_set_sha256"],
            root_set_digest(roots),
        )
        self.assertEqual(
            binding["recovery_stress_development_parent_sha256"],
            input_bindings["development_holdout"],
        )


if __name__ == "__main__":
    unittest.main()
