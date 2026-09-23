"""Preserve acquired sensor noise while auditing healthy and mixed-HIF recovery."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from psse_env.dagger.release_audit import (
    EXPLANATION_ONLY_DIAGNOSTIC_CONTRACT,
    FINAL_MEASUREMENTS_CHECK,
    HEALTHY_ACQUISITION_REFERENCE,
    HIF_METER_ACQUISITION_REFERENCE,
    audit_episode_against_truth,
    audit_truth_audited_task_success,
)


def _case():
    return {"baseMVA": 100.0, "bus": [[1.0, 3.0], [2.0, 1.0]],
            "branch": [[1.0, 2.0, .01, .02, 0., 0., 0., 0., 0., 0., 1.]]}


def _healthy():
    return {
        "scenario_id": "healthy-noisy", "scenario_family": "telemetry_no_disturbance",
        "truth_complete": True, "true_measurement_errors": [],
        "true_parameter_errors": [], "true_topology_errors": [], "true_unbalance_errors": [],
        "case": _case(), "clean_case": _case(),
        "measurements": [1.003, .224, -.031], "clean_measurements": [1., .2, -.01],
    }


def _hif_explanation(truth=None):
    truth = truth or {"branch_row0": 0, "phase": "A", "split_ratio": .4}
    return {"family": "hif", "kind": "hif_model_accepted_over_null", "detail": {
        "candidate_branch_row0": truth["branch_row0"],
        "estimated": {"phase": truth.get("phase", "A"),
                      "alpha_from_from_bus": truth.get("split_ratio", .4)},
    }}


def _mixed():
    scenario = _healthy()
    scenario.update(
        scenario_id="mixed-hif-meter", scenario_family="measurement+hif",
        measurements=[1.003, 20.32, 30.02], clean_measurements=[1., 2., 3.],
        true_measurement_errors=[{"index": 1, "observed": 20.32, "clean": 20.02}],
        true_hif_errors=[{"branch_row0": 0, "phase": "A", "split_ratio": .4}],
        release_audit={
            "explanation_only_contract": EXPLANATION_ONLY_DIAGNOSTIC_CONTRACT,
            "not_applicable": {FINAL_MEASUREMENTS_CHECK: "Inherited pure-HIF explanation-only declaration"},
            "tolerances": {"measurement_abs": .03},
        },
    )
    return scenario


def _audit(scenario, measurements=None, *, accepted=(), explanations=(), case=None):
    return audit_episode_against_truth(
        scenario,
        {"accepted_corrections": [{"source_action": {
            "tool": "correct_measurements", "arguments": {"suspect_group": [index]},
        }} for index in accepted], "explained_anomalies": list(explanations)},
        terminal=True, terminal_outcome="resolved",
        active_physical_state={"case": deepcopy(scenario["case"] if case is None else case),
                               "measurements": list(scenario["measurements"] if measurements is None else measurements)},
    )


class AcquisitionReferenceAuditTests(unittest.TestCase):
    def test_healthy_control_preserves_noise_and_records_its_reference(self):
        for family in ("telemetry_no_disturbance", "no_error"):
            with self.subTest(family=family):
                scenario = _healthy()
                scenario["scenario_family"] = family
                original = deepcopy(scenario)
                result = _audit(scenario)
                self.assertFalse(result["quarantined"], result["problems"])
                self.assertEqual(result["checks"][FINAL_MEASUREMENTS_CHECK]["reference_kind"], HEALTHY_ACQUISITION_REFERENCE)
                self.assertEqual(result["checks"][FINAL_MEASUREMENTS_CHECK]["status"], "passed")
                self.assertEqual(scenario, original)

    def test_healthy_control_denoising_or_even_small_mutation_fails(self):
        scenario = _healthy()
        modified = list(scenario["measurements"])
        modified[0] += 1e-8
        for measurements in (scenario["clean_measurements"], modified):
            with self.subTest(measurements=measurements):
                result = _audit(scenario, measurements)
                self.assertTrue(result["quarantined"])
                self.assertIn("healthy_measurement_modified", result["problems"])

    def test_healthy_target_correction_and_case_mutation_still_fail(self):
        scenario = _healthy()
        correction = _audit(scenario, accepted=[1])
        self.assertIn("accepted_measurement_targets_outside_truth", correction["problems"])
        changed = deepcopy(scenario["case"])
        changed["branch"][0][2] += .01
        result = _audit(scenario, case=changed)
        self.assertIn("healthy_case_component_modified", result["problems"])

    def test_unknown_truth_or_fault_mislabeled_healthy_cannot_select_healthy_reference(self):
        for alteration in ({"truth_complete": False}, {"true_hif_errors": [{"branch_row0": 0}]},
                           {"true_measurement_errors": [{"index": 1, "clean": .2}]},
                           {"true_unrecognized_errors": [{"target": 1}]},
                           {"true_measurement_errors": "malformed"}):
            with self.subTest(alteration=alteration):
                scenario = _healthy()
                scenario.update(alteration)
                result = _audit(scenario)
                self.assertTrue(result["quarantined"])
                self.assertEqual(result["checks"][FINAL_MEASUREMENTS_CHECK]["reference_kind"], "scenario_clean_measurements")

    def test_missing_healthy_initial_acquisition_fails_closed(self):
        scenario = _healthy()
        scenario.pop("measurements")
        result = _audit(scenario, measurements=[1., .2, -.01])
        self.assertTrue(result["quarantined"])
        self.assertIn("healthy_measurement_preservation_evidence_missing_or_malformed", result["problems"])

    def test_mixed_hif_diagnosis_plus_repaired_meter_runs_full_reference_check(self):
        scenario = _mixed()
        original = deepcopy(scenario)
        result = _audit(scenario, [1.003, 20.025, 30.02], accepted=[1], explanations=[_hif_explanation()])
        self.assertFalse(result["quarantined"], result["problems"])
        check = result["checks"][FINAL_MEASUREMENTS_CHECK]
        self.assertEqual(check["status"], "passed")
        self.assertEqual(check["reference_kind"], HIF_METER_ACQUISITION_REFERENCE)
        self.assertTrue(check["superseded_explanation_only_waiver"])
        self.assertEqual(result["checks"]["accepted_target_nonregression"]["status"], "passed")
        self.assertEqual(scenario, original)

    def test_mixed_unrepaired_meter_cannot_use_the_inherited_hif_waiver(self):
        result = _audit(_mixed(), explanations=[_hif_explanation()])
        self.assertTrue(result["quarantined"])
        self.assertIn("resolved_episode_has_remaining_true_faults", result["problems"])
        self.assertIn("final_measurements_outside_clean_tolerance", result["problems"])
        self.assertEqual(result["checks"][FINAL_MEASUREMENTS_CHECK]["status"], "failed")

    def test_mixed_repair_requires_correct_hif_diagnosis_and_exact_true_target(self):
        for accepted, explanations in (([1], []), ([0], [_hif_explanation()]),
                                      ([1], [_hif_explanation({"branch_row0": 1})])):
            with self.subTest(accepted=accepted, explanations=explanations):
                result = _audit(_mixed(), [1.003, 20.02, 30.02], accepted=accepted, explanations=explanations)
                self.assertTrue(result["quarantined"])

    def test_mixed_healthy_channel_mutation_is_not_hidden_by_meter_noise_tolerance(self):
        result = _audit(_mixed(), [1.00300001, 20.02, 30.02], accepted=[1], explanations=[_hif_explanation()])
        self.assertIn("healthy_measurement_modified", result["problems"])

    def test_mixed_reference_rejects_missing_clean_duplicate_or_unbound_observed_values(self):
        for rows in ([{"index": 1}], [{"index": 1, "clean": 20.02}] * 2,
                     [{"index": 1, "clean": 20.02, "observed": 999.}],
                     [{"index": 999, "clean": 20.02}], [{"index": 1, "clean": float("nan")} ]):
            with self.subTest(rows=rows):
                scenario = _mixed()
                scenario["true_measurement_errors"] = rows
                result = _audit(scenario, [1.003, 20.02, 30.02], accepted=[1], explanations=[_hif_explanation()])
                self.assertTrue(result["quarantined"])
                self.assertIn("mixed_hif_measurement_reference_truth_invalid", result["problems"])

    def test_explicit_new_mixed_contract_needs_no_waiver(self):
        scenario = _mixed()
        scenario["release_audit"] = {"measurement_reference_kind": HIF_METER_ACQUISITION_REFERENCE,
                                      "tolerances": {"measurement_abs": .03}}
        result = _audit(scenario, [1.003, 20.02, 30.02], accepted=[1], explanations=[_hif_explanation()])
        self.assertFalse(result["quarantined"], result["problems"])
        self.assertFalse(result["checks"][FINAL_MEASUREMENTS_CHECK]["superseded_explanation_only_waiver"])

    def test_recorded_eight_r1_healthy_acquisitions_pass_without_denoising(self):
        artifact = Path(__file__).resolve().parents[2]/"output/r1_failure_analysis_20260922/candidate_artifacts/telemetry_control_reference_check.json"
        if not artifact.exists():
            self.skipTest("Frozen R1 acquisition audit is a local artifact")
        rows = json.loads(artifact.read_text())
        self.assertEqual(len(rows), 8)
        for row in rows:
            with self.subTest(scenario=row["scenario_id"]):
                self.assertFalse(row["faulted"])
                self.assertEqual(row["initial_state_id"], row["final_state_id"])
                self.assertTrue(all(step["state_mutated"] is False for step in row["trace"]))
                scenario = _healthy()
                scenario.update(scenario_id=row["scenario_id"], measurements=row["initial_measurements"],
                                clean_measurements=row["clean_reference_measurements"])
                result = _audit(scenario)
                self.assertFalse(result["quarantined"], result["problems"])
                assessment = audit_truth_audited_task_success(
                    scenario, {"accepted_corrections": [], "explained_anomalies": []},
                    actual_terminal=True, actual_terminal_outcome="resolved",
                    active_physical_state={"case": scenario["case"], "measurements": row["initial_measurements"]},
                )
                self.assertTrue(assessment["eligible"], assessment["reasons"])

    def test_frozen_eight_mixed_roots_require_meter_repair_even_with_correct_hif_diagnosis(self):
        artifact = Path(__file__).resolve().parents[2]/"output/hif_continuation_fix_20260922/frozen_mixed_scenarios.json"
        if not artifact.exists():
            self.skipTest("Frozen mixed-HIF scenarios are local audit artifacts")
        from psse_env.dagger.evaluator import _strict_audit_scenario
        from psse_env.dagger.release_factories import deterministic_case_loader

        rows = json.loads(artifact.read_text())
        self.assertEqual(len(rows), 8)
        for row in rows:
            scenario = _strict_audit_scenario(row)
            with self.subTest(scenario=scenario["scenario_id"]):
                # This is an audit-contract counterfactual using exact hidden
                # truth, not a claim that a policy performed these repairs.
                explanations = [_hif_explanation(truth) for truth in scenario["true_hif_errors"]]
                initial = list(scenario["measurements"])
                unrepaired = audit_episode_against_truth(
                    scenario, {"accepted_corrections": [], "explained_anomalies": explanations},
                    terminal=True, terminal_outcome="resolved",
                    active_physical_state={"case": scenario["case"], "measurements": initial},
                    case_loader=deterministic_case_loader,
                )
                self.assertTrue(unrepaired["quarantined"])
                self.assertIn("resolved_episode_has_remaining_true_faults", unrepaired["problems"])
                self.assertIn("final_measurements_outside_clean_tolerance", unrepaired["problems"])
                repaired = list(initial)
                corrections = []
                for fault in scenario["true_measurement_errors"]:
                    repaired[fault["index"]] = fault["clean"]
                    corrections.append({"source_action": {"tool": "correct_measurements",
                        "arguments": {"suspect_group": [fault["index"]]}}})
                recovered = audit_episode_against_truth(
                    scenario, {"accepted_corrections": corrections, "explained_anomalies": explanations},
                    terminal=True, terminal_outcome="resolved",
                    active_physical_state={"case": scenario["case"], "measurements": repaired},
                    case_loader=deterministic_case_loader,
                )
                self.assertFalse(recovered["quarantined"], recovered["problems"])
                self.assertEqual(recovered["checks"][FINAL_MEASUREMENTS_CHECK]["status"], "passed")


if __name__ == "__main__":
    unittest.main()
