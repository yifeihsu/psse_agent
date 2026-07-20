from __future__ import annotations

import copy
import unittest

from psse_env.dagger.release_audit import (
    ACCEPTED_TARGET_NONREGRESSION_CHECK,
    ACCEPTED_TARGETS_CHECK,
    DIAGNOSTIC_LOCALIZATION_CHECK,
    EXPLANATION_ONLY_DIAGNOSTIC_CONTRACT,
    FINAL_CASE_CHECK,
    FINAL_MEASUREMENTS_CHECK,
    HEALTHY_CASE_CHECK,
    HEALTHY_MEASUREMENTS_CHECK,
    REMAINING_FAULTS_CHECK,
    audit_episode_against_truth,
)


def _case() -> dict[str, object]:
    return {
        "baseMVA": 100.0,
        "bus": [[1.0, 3.0], [2.0, 1.0]],
        "branch": [
            [1.0, 2.0, 0.01, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [2.0, 3.0, 0.03, 0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
    }


def _base_scenario() -> dict[str, object]:
    case = _case()
    return {
        "scenario_id": "strict-audit",
        "scenario_family": "measurement",
        "case": copy.deepcopy(case),
        "clean_case": copy.deepcopy(case),
        "measurements": [1.0, 99.0, 3.0],
        "clean_measurements": [1.0, 2.0, 3.0],
        "true_measurement_errors": [{"index": 1, "clean": 2.0}],
        "true_parameter_errors": [],
        "true_topology_errors": [],
    }


def _active(
    *, measurements: list[float] | None = None, case: object | None = None
) -> dict[str, object]:
    return {
        "state_id": "episode:s2",
        "case": copy.deepcopy(_case() if case is None else case),
        "measurements": list(measurements or [1.0, 2.0, 3.0]),
    }


def _final_state(
    *actions: dict[str, object],
    explanations: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "accepted_corrections": [
            {"source_action": copy.deepcopy(action)} for action in actions
        ],
        "explained_anomalies": copy.deepcopy(explanations or []),
    }


def _audit_resolved(
    scenario: dict[str, object],
    final_state: dict[str, object],
    *,
    active: dict[str, object] | None = None,
    remaining: dict[str, object] | None = None,
    not_applicable: dict[str, str] | None = None,
    tolerances: dict[str, float] | None = None,
) -> dict[str, object]:
    supplied_ledger = copy.deepcopy(remaining)
    if supplied_ledger is not None and "truth_complete" not in supplied_ledger:
        hidden = supplied_ledger.get("hidden_truth")
        if not isinstance(hidden, dict) or "truth_complete" not in hidden:
            supplied_ledger["truth_complete"] = True
    return audit_episode_against_truth(
        scenario,
        final_state,
        terminal=True,
        terminal_outcome="resolved",
        active_physical_state=active,
        remaining_truth=supplied_ledger,
        not_applicable=not_applicable,
        tolerances=tolerances,
    )


class AcceptedCorrectionTargetTests(unittest.TestCase):
    def test_true_measurement_subset_passes(self) -> None:
        scenario = _base_scenario()
        result = _audit_resolved(
            scenario,
            _final_state(
                {
                    "tool": "correct_measurements",
                    "arguments": {
                        "suspect_group": [1],
                        "measurement_updates": {"1": 2.0},
                    },
                }
            ),
            active=_active(),
            remaining={"remaining_true_fault_count": 0, "remaining_true_faults": []},
        )
        self.assertFalse(result["quarantined"], result["problems"])

    def test_broad_group_touching_healthy_meter_fails(self) -> None:
        scenario = _base_scenario()
        result = _audit_resolved(
            scenario,
            _final_state(
                {
                    "tool": "correct_measurements",
                    "arguments": {"suspect_group": [1, 2]},
                }
            ),
            active=_active(),
            remaining={"remaining_true_fault_count": 0},
        )
        self.assertTrue(result["quarantined"])
        self.assertIn(
            "accepted_measurement_targets_outside_truth", result["problems"]
        )

    def test_measurement_updates_are_unioned_with_suspect_group(self) -> None:
        scenario = _base_scenario()
        result = _audit_resolved(
            scenario,
            _final_state(
                {
                    "tool": "correct_measurements",
                    "arguments": {
                        "suspect_group": [1],
                        "measurement_updates": {2: 3.0},
                    },
                }
            ),
            active=_active(),
            remaining={"remaining_true_fault_count": 0},
        )
        self.assertIn(
            "accepted_measurement_targets_outside_truth", result["problems"]
        )

    def test_fractional_measurement_index_fails_closed(self) -> None:
        scenario = _base_scenario()
        result = audit_episode_against_truth(
            scenario,
            _final_state(
                {
                    "tool": "correct_measurements",
                    "arguments": {"suspect_group": [1.5]},
                }
            ),
            terminal=True,
            terminal_outcome="operator_escalation",
        )
        self.assertIn("accepted_measurement_targets_malformed", result["problems"])

    def test_parameter_correction_on_topology_truth_line_fails(self) -> None:
        scenario = _base_scenario()
        scenario["scenario_family"] = "topology"
        scenario["true_measurement_errors"] = []
        scenario["measurements"] = [1.0, 2.0, 3.0]
        scenario["true_topology_errors"] = [
            {"branch_row0": 1, "line_index1": 2, "expected_status": 0}
        ]
        result = audit_episode_against_truth(
            scenario,
            _final_state(
                {
                    "tool": "correct_parameters",
                    "arguments": {"line_index1": 2},
                }
            ),
            terminal=True,
            terminal_outcome="operator_escalation",
        )
        self.assertIn(
            "accepted_parameter_target_outside_same_family_truth",
            result["problems"],
        )

    def test_topology_correction_on_parameter_truth_line_fails(self) -> None:
        scenario = _base_scenario()
        scenario["scenario_family"] = "parameter"
        scenario["true_measurement_errors"] = []
        scenario["measurements"] = [1.0, 2.0, 3.0]
        scenario["true_parameter_errors"] = [
            {"branch_row0": 1, "line_index1": 2, "parameter": "rx"}
        ]
        result = audit_episode_against_truth(
            scenario,
            _final_state(
                {
                    "tool": "correct_topology",
                    "arguments": {"branch_row0": 1, "status": 0},
                }
            ),
            terminal=True,
            terminal_outcome="operator_escalation",
        )
        self.assertIn(
            "accepted_topology_target_outside_same_family_truth",
            result["problems"],
        )


class AcceptedTargetNonregressionTests(unittest.TestCase):
    def _measurement_action(self) -> dict[str, object]:
        return {
            "tool": "correct_measurements",
            "arguments": {"suspect_group": [1]},
        }

    def _parameter_scenario(self) -> dict[str, object]:
        scenario = _base_scenario()
        scenario["scenario_family"] = "parameter"
        scenario["measurements"] = [1.0, 2.0, 3.0]
        scenario["clean_measurements"] = [1.0, 2.0, 3.0]
        scenario["true_measurement_errors"] = []
        clean = _case()
        clean["branch"][1][2] = 0.01  # type: ignore[index]
        clean["branch"][1][3] = 0.02  # type: ignore[index]
        scenario["clean_case"] = clean
        scenario["true_parameter_errors"] = [
            {
                "branch_row0": 1,
                "line_index1": 2,
                "parameter": "rx",
                "clean_r": 0.01,
                "clean_x": 0.02,
            }
        ]
        return scenario

    def _topology_scenario(self) -> dict[str, object]:
        scenario = _base_scenario()
        scenario["scenario_family"] = "topology"
        scenario["measurements"] = [1.0, 2.0, 3.0]
        scenario["clean_measurements"] = [1.0, 2.0, 3.0]
        scenario["true_measurement_errors"] = []
        clean = _case()
        clean["branch"][1][10] = 0.0  # type: ignore[index]
        scenario["clean_case"] = clean
        scenario["true_topology_errors"] = [
            {"branch_row0": 1, "line_index1": 2, "expected_status": 0}
        ]
        return scenario

    def test_operator_escalation_rejects_worsened_measurement_target(self) -> None:
        result = audit_episode_against_truth(
            _base_scenario(),
            _final_state(self._measurement_action()),
            terminal=True,
            terminal_outcome="operator_escalation",
            active_physical_state=_active(measurements=[1.0, 150.0, 3.0]),
        )

        self.assertTrue(result["quarantined"])
        self.assertIn("accepted_measurement_target_regressed", result["problems"])
        self.assertEqual(
            result["checks"][ACCEPTED_TARGET_NONREGRESSION_CHECK]["status"],
            "failed",
        )

    def test_operator_escalation_allows_unresolved_but_improved_target(self) -> None:
        result = audit_episode_against_truth(
            _base_scenario(),
            _final_state(self._measurement_action()),
            terminal=True,
            terminal_outcome="operator_escalation",
            active_physical_state=_active(measurements=[1.0, 50.0, 3.0]),
        )

        self.assertFalse(result["quarantined"], result["problems"])
        check = result["checks"][ACCEPTED_TARGET_NONREGRESSION_CHECK]
        self.assertEqual(check["status"], "passed")
        self.assertEqual(check["target_evidence"][0]["status"], "passed")

    def test_missing_measurement_clean_evidence_fails_closed(self) -> None:
        scenario = _base_scenario()
        scenario.pop("clean_measurements")
        scenario["true_measurement_errors"] = [{"index": 1}]
        result = audit_episode_against_truth(
            scenario,
            _final_state(self._measurement_action()),
            terminal=True,
            terminal_outcome="operator_escalation",
            active_physical_state=_active(),
        )

        self.assertIn(
            "accepted_measurement_nonregression_evidence_missing_or_malformed",
            result["problems"],
        )

    def test_operator_escalation_rejects_worsened_parameter_target(self) -> None:
        scenario = self._parameter_scenario()
        final_case = _case()
        final_case["branch"][1][2] = 0.06  # type: ignore[index]
        final_case["branch"][1][3] = 0.08  # type: ignore[index]
        result = audit_episode_against_truth(
            scenario,
            _final_state(
                {
                    "tool": "correct_parameters",
                    "arguments": {"line_index1": 2},
                }
            ),
            terminal=True,
            terminal_outcome="operator_escalation",
            active_physical_state=_active(case=final_case),
        )

        self.assertIn("accepted_parameter_target_regressed", result["problems"])
        evidence = result["checks"][ACCEPTED_TARGET_NONREGRESSION_CHECK][
            "target_evidence"
        ]
        self.assertEqual(evidence[0]["family"], "parameter")
        self.assertGreater(evidence[0]["final_distance"], evidence[0]["initial_distance"])

    def test_missing_parameter_initial_case_fails_closed(self) -> None:
        scenario = self._parameter_scenario()
        scenario.pop("case")
        result = audit_episode_against_truth(
            scenario,
            _final_state(
                {
                    "tool": "correct_parameters",
                    "arguments": {"branch_row0": 1},
                }
            ),
            terminal=True,
            terminal_outcome="operator_escalation",
            active_physical_state=_active(case=scenario["clean_case"]),
        )

        self.assertIn(
            "accepted_parameter_nonregression_evidence_missing_or_malformed",
            result["problems"],
        )

    def test_unresolved_topology_target_is_nonregressing(self) -> None:
        scenario = self._topology_scenario()
        result = audit_episode_against_truth(
            scenario,
            _final_state(
                {
                    "tool": "correct_topology",
                    "arguments": {"branch_row0": 1, "status": 0},
                }
            ),
            terminal=True,
            terminal_outcome="operator_escalation",
            active_physical_state=_active(case=_case()),
        )

        self.assertFalse(result["quarantined"], result["problems"])
        evidence = result["checks"][ACCEPTED_TARGET_NONREGRESSION_CHECK][
            "target_evidence"
        ]
        self.assertEqual(evidence[0]["family"], "topology")
        self.assertEqual(evidence[0]["initial_distance"], evidence[0]["final_distance"])

    def test_malformed_topology_status_fails_closed(self) -> None:
        scenario = self._topology_scenario()
        final_case = _case()
        final_case["branch"][1][10] = 2.0  # type: ignore[index]
        result = audit_episode_against_truth(
            scenario,
            _final_state(
                {
                    "tool": "correct_topology",
                    "arguments": {"branch_row0": 1, "status": 0},
                }
            ),
            terminal=True,
            terminal_outcome="operator_escalation",
            active_physical_state=_active(case=final_case),
        )

        self.assertIn(
            "accepted_topology_nonregression_evidence_missing_or_malformed",
            result["problems"],
        )


class ResolvedPhysicalStateTests(unittest.TestCase):
    def test_derived_ledger_is_sufficient_when_no_external_ledger_is_supplied(self) -> None:
        result = _audit_resolved(
            _base_scenario(),
            _final_state(
                {
                    "tool": "correct_measurements",
                    "arguments": {"suspect_group": [1]},
                }
            ),
            active=_active(),
        )
        self.assertFalse(result["quarantined"], result["problems"])
        self.assertEqual(
            result["checks"][REMAINING_FAULTS_CHECK][
                "derived_remaining_fault_count"
            ],
            0,
        )
        self.assertEqual(
            result["checks"][REMAINING_FAULTS_CHECK]["evidence_source"],
            "offline_scenario_truth_derivation",
        )

    def test_false_empty_incomplete_ledger_is_never_trusted(self) -> None:
        result = _audit_resolved(
            _base_scenario(),
            _final_state(
                {
                    "tool": "correct_measurements",
                    "arguments": {"suspect_group": [1]},
                }
            ),
            active=_active(),
            remaining={
                "remaining_true_faults": [],
                "hidden_truth": {
                    "truth_complete": False,
                    "remaining_true_faults": [],
                    "remaining_true_fault_count": 0,
                },
            },
        )
        self.assertIn(
            "supplied_remaining_truth_ledger_incomplete", result["problems"]
        )
        self.assertEqual(
            result["checks"][REMAINING_FAULTS_CHECK][
                "derived_remaining_fault_count"
            ],
            0,
        )

    def test_derived_ledger_catches_incomplete_accepted_faults(self) -> None:
        scenario = _base_scenario()
        scenario["measurements"] = [9.0, 99.0, 3.0]
        scenario["true_measurement_errors"] = [
            {"index": 0, "clean": 1.0},
            {"index": 1, "clean": 2.0},
        ]
        result = _audit_resolved(
            scenario,
            _final_state(
                {
                    "tool": "correct_measurements",
                    "arguments": {"suspect_group": [1]},
                }
            ),
            active=_active(),
            remaining={
                "truth_complete": True,
                "remaining_true_faults": [],
                "remaining_true_fault_count": 0,
            },
        )
        self.assertIn("resolved_episode_has_remaining_true_faults", result["problems"])
        self.assertIn(
            "supplied_remaining_truth_ledger_disagrees_with_derived",
            result["problems"],
        )
        self.assertEqual(
            result["checks"][REMAINING_FAULTS_CHECK][
                "derived_remaining_fault_count"
            ],
            1,
        )

    def test_missing_remaining_fault_and_active_state_evidence_fails_closed(self) -> None:
        result = _audit_resolved(_base_scenario(), _final_state())
        self.assertTrue(result["quarantined"])
        self.assertIn(
            "resolved_episode_has_remaining_true_faults",
            result["problems"],
        )
        self.assertIn(
            "healthy_measurement_preservation_evidence_missing_or_malformed",
            result["problems"],
        )

    def test_nonzero_remaining_fault_count_fails(self) -> None:
        result = _audit_resolved(
            _base_scenario(),
            _final_state(),
            active=_active(),
            remaining={
                "remaining_true_fault_count": 1,
                "remaining_true_faults": [{"index": 1}],
            },
        )
        self.assertIn("resolved_episode_has_remaining_true_faults", result["problems"])

    def test_inconsistent_remaining_count_and_list_fails_closed(self) -> None:
        result = _audit_resolved(
            _base_scenario(),
            _final_state(),
            active=_active(),
            remaining={
                "remaining_true_fault_count": 0,
                "remaining_true_faults": [{"index": 1}],
            },
        )
        self.assertIn(
            "supplied_remaining_truth_ledger_missing_or_malformed",
            result["problems"],
        )

    def test_healthy_measurement_mutation_fails(self) -> None:
        result = _audit_resolved(
            _base_scenario(),
            _final_state(),
            active=_active(measurements=[9.0, 2.0, 3.0]),
            remaining={"remaining_true_fault_count": 0},
        )
        self.assertIn("healthy_measurement_modified", result["problems"])

    def test_true_target_can_be_preserved_but_final_state_must_match_clean(self) -> None:
        result = _audit_resolved(
            _base_scenario(),
            _final_state(),
            active=_active(measurements=[1.0, 99.0, 3.0]),
            remaining={"remaining_true_fault_count": 0},
        )
        self.assertNotIn("healthy_measurement_modified", result["problems"])
        self.assertIn("final_measurements_outside_clean_tolerance", result["problems"])

    def test_healthy_branch_mutation_fails(self) -> None:
        scenario = _base_scenario()
        scenario["scenario_family"] = "topology"
        scenario["true_measurement_errors"] = []
        scenario["measurements"] = [1.0, 2.0, 3.0]
        scenario["true_topology_errors"] = [
            {"branch_row0": 1, "line_index1": 2, "expected_status": 0}
        ]
        clean = _case()
        clean["branch"][1][-1] = 0.0  # type: ignore[index]
        scenario["clean_case"] = copy.deepcopy(clean)
        final_case = copy.deepcopy(clean)
        # Mutate a non-target field on the target branch itself.  Only the
        # topology status column is allowed to change.
        final_case["branch"][1][5] = 9.0  # type: ignore[index]
        result = _audit_resolved(
            scenario,
            _final_state(
                {
                    "tool": "correct_topology",
                    "arguments": {"branch_row0": 1, "status": 0},
                }
            ),
            active=_active(case=final_case),
            remaining={"remaining_true_fault_count": 0},
            tolerances={
                "case_abs": 1e-9,
                "case_rel": 1e-9,
                "final_case_abs": 10.0,
                "final_case_rel": 10.0,
            },
        )
        self.assertIn("healthy_case_component_modified", result["problems"])
        self.assertNotIn("final_case_outside_clean_tolerance", result["problems"])

    def test_broad_parameter_tolerance_cannot_hide_wrong_topology_status(self) -> None:
        scenario = _base_scenario()
        scenario["scenario_family"] = "topology"
        scenario["true_measurement_errors"] = []
        scenario["measurements"] = [1.0, 2.0, 3.0]
        scenario["true_topology_errors"] = [
            {"branch_row0": 1, "line_index1": 2, "expected_status": 0}
        ]
        clean = _case()
        clean["branch"][1][-1] = 0.0  # type: ignore[index]
        scenario["clean_case"] = clean
        result = _audit_resolved(
            scenario,
            _final_state(
                {
                    "tool": "correct_topology",
                    "arguments": {"branch_row0": 1, "status": 0},
                }
            ),
            active=_active(),
            remaining={"remaining_true_fault_count": 0},
            tolerances={"final_case_abs": 10.0, "final_case_rel": 10.0},
        )
        self.assertIn("final_case_outside_clean_tolerance", result["problems"])

    def test_distinct_case_references_require_loader(self) -> None:
        scenario = _base_scenario()
        scenario["case"] = "initial"
        scenario["clean_case"] = "clean"
        cases = {"initial": _case(), "clean": _case(), "final": _case()}
        correction = {
            "tool": "correct_measurements",
            "arguments": {"suspect_group": [1]},
        }
        without_loader = _audit_resolved(
            scenario,
            _final_state(correction),
            active=_active(case="final"),
            remaining={"remaining_true_fault_count": 0},
        )
        self.assertIn(
            "final_clean_case_evidence_missing_or_unloadable",
            without_loader["problems"],
        )
        with_loader = audit_episode_against_truth(
            scenario,
            _final_state(correction),
            terminal=True,
            terminal_outcome="resolved",
            active_physical_state=_active(case="final"),
            remaining_truth={
                "truth_complete": True,
                "remaining_true_fault_count": 0,
            },
            case_loader=lambda reference: copy.deepcopy(cases[str(reference)]),
        )
        self.assertFalse(with_loader["quarantined"], with_loader["problems"])


class DiagnosticLocalizationTests(unittest.TestCase):
    def _diagnostic_scenario(
        self, family: str, truth_key: str, truth: dict[str, object]
    ) -> dict[str, object]:
        scenario = _base_scenario()
        scenario["scenario_family"] = family
        scenario["measurements"] = [10.0, 20.0, 30.0]
        scenario["clean_measurements"] = [1.0, 2.0, 3.0]
        scenario["true_measurement_errors"] = []
        scenario["hidden_truth"] = {truth_key: [truth]}
        scenario["release_audit"] = {
            "explanation_only_contract": EXPLANATION_ONLY_DIAGNOSTIC_CONTRACT,
            "not_applicable": {
                FINAL_MEASUREMENTS_CHECK: "diagnostic-only anomaly is explained, not repaired"
            }
        }
        return scenario

    def test_measurement_recovery_cannot_spoof_explanation_only_waiver(self) -> None:
        scenario = _base_scenario()
        scenario["release_audit"] = {
            "explanation_only_contract": EXPLANATION_ONLY_DIAGNOSTIC_CONTRACT,
            "not_applicable": {
                FINAL_MEASUREMENTS_CHECK: "spoofed explanation-only waiver"
            },
        }
        result = _audit_resolved(
            scenario,
            _final_state(
                {
                    "tool": "correct_measurements",
                    "arguments": {"suspect_group": [1]},
                }
            ),
            active=_active(measurements=[1.0, 50.0, 3.0]),
            remaining={"remaining_true_fault_count": 0},
        )

        self.assertTrue(result["quarantined"])
        self.assertIn(
            "final_measurements_not_applicable_requires_pure_diagnostic_family",
            result["problems"],
        )
        self.assertIn(
            "final_measurements_not_applicable_prohibits_correction_truth",
            result["problems"],
        )
        self.assertIn(
            "final_measurements_not_applicable_prohibits_accepted_corrections",
            result["problems"],
        )
        self.assertIn("final_measurements_outside_clean_tolerance", result["problems"])
        self.assertEqual(
            result["checks"][FINAL_MEASUREMENTS_CHECK]["status"], "failed"
        )

    def test_explanation_only_waiver_requires_generator_contract_marker(self) -> None:
        scenario = self._diagnostic_scenario(
            "harmonic", "true_harmonic_errors", {"bus_1based": 5}
        )
        scenario["release_audit"].pop("explanation_only_contract")  # type: ignore[union-attr]
        result = _audit_resolved(
            scenario,
            _final_state(
                explanations=[
                    {"family": "harmonic", "detail": {"bus_1based": 5}}
                ]
            ),
            active=_active(measurements=[10.0, 20.0, 30.0]),
            remaining={"remaining_true_fault_count": 0},
        )

        self.assertIn(
            "final_measurements_not_applicable_contract_marker_missing_or_invalid",
            result["problems"],
        )
        self.assertEqual(
            result["checks"][FINAL_MEASUREMENTS_CHECK]["status"], "failed"
        )

    def test_operator_escalation_does_not_claim_or_require_localization(self) -> None:
        scenario = self._diagnostic_scenario(
            "hif", "true_hif_errors", {"branch_row0": 1}
        )
        result = audit_episode_against_truth(
            scenario,
            _final_state(),
            terminal=True,
            terminal_outcome="operator_escalation",
            active_physical_state=_active(measurements=[10.0, 20.0, 30.0]),
        )
        self.assertFalse(result["quarantined"], result["problems"])
        self.assertEqual(
            result["checks"][DIAGNOSTIC_LOCALIZATION_CHECK]["status"],
            "not_required",
        )

    def test_operator_escalation_rejects_a_false_localization_claim(self) -> None:
        scenario = self._diagnostic_scenario(
            "hif", "true_hif_errors", {"branch_row0": 1}
        )
        result = audit_episode_against_truth(
            scenario,
            _final_state(
                explanations=[
                    {
                        "family": "hif",
                        "detail": {"candidate_branch_row0": 0},
                    }
                ]
            ),
            terminal=True,
            terminal_outcome="operator_escalation",
            active_physical_state=_active(measurements=[10.0, 20.0, 30.0]),
        )

        self.assertTrue(result["quarantined"])
        self.assertIn(
            "diagnostic_explanation_does_not_localize_any_truth",
            result["problems"],
        )

    def test_operator_escalation_rejects_healthy_component_corruption(self) -> None:
        scenario = _base_scenario()
        result = audit_episode_against_truth(
            scenario,
            _final_state(),
            terminal=True,
            terminal_outcome="operator_escalation",
            active_physical_state=_active(measurements=[7.0, 99.0, 3.0]),
        )

        self.assertTrue(result["quarantined"])
        self.assertIn("healthy_measurement_modified", result["problems"])
        self.assertEqual(
            result["checks"][HEALTHY_MEASUREMENTS_CHECK]["status"], "failed"
        )
        self.assertEqual(
            result["checks"][HEALTHY_CASE_CHECK]["status"], "passed"
        )

    def test_harmonic_bus_within_declared_tolerance_passes(self) -> None:
        scenario = self._diagnostic_scenario(
            "harmonic",
            "true_harmonic_errors",
            {"bus_1based": 5, "bus_index_tolerance": 1},
        )
        result = _audit_resolved(
            scenario,
            _final_state(
                explanations=[
                    {
                        "family": "harmonic",
                        "kind": "harmonic_source_localized",
                        "detail": {"bus_1based": 6},
                    }
                ]
            ),
            active=_active(measurements=[10.0, 20.0, 30.0]),
            remaining={"remaining_true_fault_count": 0},
        )
        self.assertFalse(result["quarantined"], result["problems"])
        self.assertEqual(
            result["checks"][FINAL_MEASUREMENTS_CHECK]["status"],
            "not_applicable",
        )

    def test_harmonic_bus_outside_tolerance_fails(self) -> None:
        scenario = self._diagnostic_scenario(
            "harmonic", "true_harmonic_errors", {"source_bus": 5}
        )
        result = _audit_resolved(
            scenario,
            _final_state(
                explanations=[
                    {"family": "harmonic", "detail": {"bus_1based": 6}}
                ]
            ),
            active=_active(measurements=[10.0, 20.0, 30.0]),
            remaining={"remaining_true_fault_count": 0},
        )
        self.assertIn("diagnostic_localization_outside_tolerance", result["problems"])

    def test_hif_branch_phase_and_alpha_are_checked(self) -> None:
        scenario = self._diagnostic_scenario(
            "hif",
            "true_hif_errors",
            {
                "branch_row0": 1,
                "phase": "A",
                "split_ratio": 0.4,
                "alpha_tolerance": 0.1,
            },
        )
        result = _audit_resolved(
            scenario,
            _final_state(
                explanations=[
                    {
                        "family": "hif",
                        "kind": "hif_model_accepted_over_null",
                        "detail": {
                            "candidate_branch_row0": 1,
                            "estimated": {
                                "phase": "A",
                                "alpha_from_from_bus": 0.45,
                            },
                        },
                    }
                ]
            ),
            active=_active(measurements=[10.0, 20.0, 30.0]),
            remaining={"remaining_true_fault_count": 0},
        )
        self.assertFalse(result["quarantined"], result["problems"])

        failed = _audit_resolved(
            scenario,
            _final_state(
                explanations=[
                    {
                        "family": "hif",
                        "detail": {
                            "candidate_branch_row0": 1,
                            "estimated": {
                                "phase": "B",
                                "alpha_from_from_bus": 0.45,
                            },
                        },
                    }
                ]
            ),
            active=_active(measurements=[10.0, 20.0, 30.0]),
            remaining={"remaining_true_fault_count": 0},
        )
        self.assertIn("diagnostic_localization_outside_tolerance", failed["problems"])

    def test_unbalance_localization_uses_top_vuf_bus(self) -> None:
        scenario = self._diagnostic_scenario(
            "three_phase_unbalance",
            "true_three_phase_unbalance_errors",
            {"unbalance_bus_name": "b7"},
        )
        result = _audit_resolved(
            scenario,
            _final_state(
                explanations=[
                    {
                        "family": "three_phase_unbalance",
                        "detail": {
                            "top_vuf_buses": [{"bus": "b7", "vuf": 0.1}]
                        },
                    }
                ]
            ),
            active=_active(measurements=[10.0, 20.0, 30.0]),
            remaining={"remaining_true_fault_count": 0},
        )
        self.assertFalse(result["quarantined"], result["problems"])

    def test_unbalance_can_declare_ranked_localization_tolerance(self) -> None:
        scenario = self._diagnostic_scenario(
            "three_phase_unbalance",
            "true_unbalance_errors",
            {"unbalance_bus": 7, "localization_top_k": 2},
        )
        result = _audit_resolved(
            scenario,
            _final_state(
                explanations=[
                    {
                        "family": "three_phase_unbalance",
                        "detail": {
                            "top_vuf_buses": [
                                {"bus": 3, "vuf": 0.2},
                                {"bus": 7, "vuf": 0.1},
                            ]
                        },
                    }
                ]
            ),
            active=_active(measurements=[10.0, 20.0, 30.0]),
            remaining={"remaining_true_fault_count": 0},
        )
        self.assertFalse(result["quarantined"], result["problems"])

    def test_wrong_family_explanation_fails_even_on_same_numeric_location(self) -> None:
        scenario = self._diagnostic_scenario(
            "hif", "true_hif_errors", {"branch_row0": 1}
        )
        result = _audit_resolved(
            scenario,
            _final_state(
                explanations=[
                    {"family": "harmonic", "detail": {"bus_1based": 1}}
                ]
            ),
            active=_active(measurements=[10.0, 20.0, 30.0]),
            remaining={"remaining_true_fault_count": 0},
        )
        self.assertIn(
            "diagnostic_explanation_without_same_family_truth", result["problems"]
        )
        self.assertIn(
            "diagnostic_truth_has_no_same_family_explanation", result["problems"]
        )

    def test_missing_localization_evidence_fails_closed(self) -> None:
        scenario = self._diagnostic_scenario(
            "harmonic", "true_harmonic_errors", {"bus_1based": 5}
        )
        result = _audit_resolved(
            scenario,
            _final_state(explanations=[{"family": "harmonic", "detail": {}}]),
            active=_active(measurements=[10.0, 20.0, 30.0]),
            remaining={"remaining_true_fault_count": 0},
        )
        self.assertIn("diagnostic_localization_evidence_missing", result["problems"])

    def test_localization_not_applicable_is_prohibited(self) -> None:
        scenario = self._diagnostic_scenario(
            "three_phase_unbalance",
            "true_three_phase_unbalance_errors",
            {"classification": "three_phase_unbalance"},
        )
        result = _audit_resolved(
            scenario,
            _final_state(
                explanations=[
                    {
                        "family": "three_phase_unbalance",
                        "kind": "nlm_non_hif_unbalance_classified",
                        "detail": {
                            "diagnostic_classification": "three_phase_unbalance"
                        },
                    }
                ]
            ),
            active=_active(measurements=[10.0, 20.0, 30.0]),
            remaining={"remaining_true_fault_count": 0},
            not_applicable={
                DIAGNOSTIC_LOCALIZATION_CHECK: "source has classification truth only"
            },
        )
        self.assertTrue(result["quarantined"])
        self.assertIn(
            "not_applicable_check_unknown_or_prohibited", result["problems"]
        )
        self.assertIn("resolved_episode_has_remaining_true_faults", result["problems"])
        self.assertEqual(
            result["checks"][DIAGNOSTIC_LOCALIZATION_CHECK]["status"],
            "failed",
        )

    def test_final_case_not_applicable_field_form_is_prohibited(self) -> None:
        scenario = _base_scenario()
        scenario["release_audit"] = {
            FINAL_CASE_CHECK: {
                "status": "not_applicable",
                "reason": "fixture validates measurement recovery only",
            }
        }
        scenario.pop("clean_case")
        result = _audit_resolved(
            scenario,
            _final_state(
                {
                    "tool": "correct_measurements",
                    "arguments": {"suspect_group": [1]},
                }
            ),
            active=_active(),
            remaining={"remaining_true_fault_count": 0},
        )
        self.assertTrue(result["quarantined"])
        self.assertIn(
            "not_applicable_check_unknown_or_prohibited", result["problems"]
        )
        self.assertEqual(
            result["checks"][FINAL_CASE_CHECK]["status"], "failed"
        )

    def test_not_applicable_requires_known_check_and_reason(self) -> None:
        result = _audit_resolved(
            _base_scenario(),
            _final_state(),
            active=_active(),
            remaining={"remaining_true_fault_count": 0},
            not_applicable={
                "unknown_check": "because",
                FINAL_MEASUREMENTS_CHECK: "",
            },
        )
        self.assertIn("not_applicable_check_unknown_or_prohibited", result["problems"])
        self.assertIn("not_applicable_reason_missing", result["problems"])

    def test_core_invariant_not_applicable_attempts_fail_on_escalation(self) -> None:
        prohibited = (
            ACCEPTED_TARGETS_CHECK,
            REMAINING_FAULTS_CHECK,
            HEALTHY_MEASUREMENTS_CHECK,
            HEALTHY_CASE_CHECK,
        )
        for name in prohibited:
            with self.subTest(check=name):
                scenario = _base_scenario()
                supplied = None
                if name == ACCEPTED_TARGETS_CHECK:
                    # Cover the embedded per-check status form as well as the
                    # generic supplied map exercised by the other invariants.
                    scenario["release_audit"] = {
                        name: {
                            "status": "not_applicable",
                            "reason": "attempted core-invariant waiver",
                        }
                    }
                else:
                    supplied = {name: "attempted core-invariant waiver"}
                result = audit_episode_against_truth(
                    scenario,
                    _final_state(),
                    terminal=True,
                    terminal_outcome="operator_escalation",
                    active_physical_state=_active(),
                    not_applicable=supplied,
                )

                self.assertTrue(result["quarantined"])
                self.assertIn(
                    "not_applicable_check_unknown_or_prohibited",
                    result["problems"],
                )
                self.assertEqual(
                    result["checks"]["audit_evidence_contract"]["status"],
                    "failed",
                )


class AuditBoundaryTests(unittest.TestCase):
    def test_audit_does_not_mutate_truth_or_policy_state(self) -> None:
        scenario = _base_scenario()
        final_state = _final_state()
        before_scenario = copy.deepcopy(scenario)
        before_final = copy.deepcopy(final_state)
        _audit_resolved(
            scenario,
            final_state,
            active=_active(),
            remaining={"remaining_true_fault_count": 0},
        )
        self.assertEqual(scenario, before_scenario)
        self.assertEqual(final_state, before_final)


if __name__ == "__main__":
    unittest.main()
