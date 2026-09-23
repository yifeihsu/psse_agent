"""A meter-only repair may retain, never erase, pre-existing HIF voltage violations."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from psse_env.oracle.candidate_quality import CandidateDisposition, CandidateQualityOracle


def _fixture(target=2):
    bus = [[1., 3., 0., 0., 0., 0., 1., 1., 0., 69., 1., 1.05, .95],
           [2., 1., 0., 0., 0., 0., 1., 1., 0., 69., 1., 1.05, .95]]
    case = {"baseMVA": 100., "bus": bus,
            "branch": [[1., 2., .01, .02, 0., 100., 0., 0., 0., 0., 1.]]}
    parent = {"state_id": "s0", "state_hash": "parent-hash", "case": case,
              "measurements": [1.12, 1., .5, .2, .1, .1, .5, .1, .4, .1]}
    candidate = deepcopy(parent)
    candidate.update(state_id="s1", state_hash="candidate-hash", parent_state_id="s0")
    candidate["measurements"][target] = .1
    action = {"tool": "correct_measurements", "arguments": {"state_id": "s0", "suspect_group": [target]}}
    scope = "observed_snapshot_topology_vm_rate_a"
    verification = {
        "target_fixed": True, "target_progress": .99, "global_progress": .7,
        "globally_resolved": True, "no_material_anomaly_remaining": True,
        "chi_square_alarm": False, "normalized_residual_alarm": False,
        "chi_square_statistic": 50., "chi_square_threshold": 100.,
        "max_normalized_residual": 2., "normalized_residual_threshold": 4.,
        "physical_constraints_ok": False, "physical_evidence_complete": True,
        "physical_evidence_scope": scope,
        "conditional_meter_scores": [0.] * 10,
        "hif_conditioning": {"status": "ready", "state_id": "s1", "state_hash": "candidate-hash",
            "method": "paired_opendss_effect_compensation", "physical_fault_still_present": True,
            "remaining_meter_candidate_indices": [], "failure_reasons": []},
        "physical_bound_violations": [{"type": "bus_voltage_out_of_bounds", "bus": 1,
            "measurement_index0": 0, "observed_vm_pu": 1.12, "vmin_pu": .95, "vmax_pu": 1.05}],
        "steady_state_physical_evidence": {"scope": scope, "complete": True, "input_errors": [],
            "violation_count": 1,
            "topology_connectivity": {"checked": True, "connected": True, "component_count": 1},
            "bus_voltage_bounds": {"checked": True, "within_bounds": False, "violation_count": 1},
            "active_branch_rate_a_bounds": {"checked": True, "within_defined_rate_a_bounds": True, "violation_count": 0}},
        "unresolved_signatures": ["hif_suspected_zero_sequence"],
    }
    return parent, candidate, action, verification


def _assess(parent, candidate, action, verification, **oracle_options):
    return CandidateQualityOracle(mode="deployment", **oracle_options).label_candidate(
        parent_state=parent, candidate_state=candidate, source_action=action,
        verification_output=verification,
    )


class HIFMeterNonregressionTests(unittest.TestCase):
    def test_meter_injection_or_flow_repair_is_partial_never_physical_resolution(self):
        for target in (2, 6):
            with self.subTest(target=target):
                parent, candidate, action, verification = _fixture(target)
                original = deepcopy(verification)
                result = _assess(parent, candidate, action, verification)
                self.assertEqual(result.disposition, CandidateDisposition.ACCEPT_PARTIAL)
                self.assertEqual(result.progress_class, "meter_repaired_preexisting_hif_voltage_violation")
                self.assertIn("operator_handoff_required", result.rationale_codes)
                self.assertIn("physical_fault_still_present", result.rationale_codes)
                self.assertEqual(result.unresolved_signatures, ["hif_suspected_zero_sequence"])
                self.assertEqual(verification, original)
                self.assertFalse(verification["physical_constraints_ok"])

    def test_path_case_requires_an_independent_case_loader(self):
        parent, candidate, action, verification = _fixture()
        case = parent["case"]
        parent["case"] = candidate["case"] = "case-path"
        self.assertEqual(_assess(parent, candidate, action, verification).disposition, CandidateDisposition.REJECT)
        result = _assess(parent, candidate, action, verification, case_loader=lambda _: case)
        self.assertEqual(result.disposition, CandidateDisposition.ACCEPT_PARTIAL)

    def test_case_or_off_target_edit_remains_collateral_damage(self):
        for change in ("case", "measurement"):
            with self.subTest(change=change):
                parent, candidate, action, verification = _fixture()
                if change == "case":
                    candidate["case"]["branch"][0][2] += .01
                else:
                    candidate["measurements"][3] += .01
                result = _assess(parent, candidate, action, verification)
                self.assertEqual(result.disposition, CandidateDisposition.REJECT)
                self.assertTrue(result.collateral_damage)

    def test_changed_voltage_cannot_use_exception_even_if_declared_target(self):
        parent, candidate, action, verification = _fixture()
        action["arguments"]["suspect_group"] = [0, 2]
        candidate["measurements"][0] = 1.13
        verification["physical_bound_violations"][0]["observed_vm_pu"] = 1.13
        self.assertEqual(_assess(parent, candidate, action, verification).disposition, CandidateDisposition.REJECT)

    def test_new_voltage_violation_cannot_be_hidden_by_retaining_old_violation_record(self):
        parent, candidate, action, verification = _fixture()
        action["arguments"]["suspect_group"] = [1, 2]
        candidate["measurements"][1] = 1.2
        self.assertEqual(_assess(parent, candidate, action, verification).disposition, CandidateDisposition.REJECT)

    def test_wrong_bounds_or_violation_channel_fail_closed(self):
        for patch in ({"measurement_index0": 6}, {"observed_vm_pu": 1.13},
                      {"vmax_pu": 1.06}, {"bus": 2}, {"type": "topology_disconnected"}):
            with self.subTest(patch=patch):
                parent, candidate, action, verification = _fixture()
                verification["physical_bound_violations"][0].update(patch)
                self.assertEqual(_assess(parent, candidate, action, verification).disposition, CandidateDisposition.REJECT)

    def test_thermal_topology_or_incomplete_physics_remain_rejected(self):
        for field, patch in (
            ("active_branch_rate_a_bounds", {"within_defined_rate_a_bounds": False, "violation_count": 1}),
            ("active_branch_rate_a_bounds", {"checked": False}),
            ("topology_connectivity", {"connected": False}),
            ("bus_voltage_bounds", {"checked": False}),
            (None, {"complete": False}), (None, {"input_errors": ["bad_input"]}),
        ):
            with self.subTest(field=field, patch=patch):
                parent, candidate, action, verification = _fixture()
                physical = verification["steady_state_physical_evidence"]
                (physical[field] if field else physical).update(patch)
                self.assertEqual(_assess(parent, candidate, action, verification).disposition, CandidateDisposition.REJECT)

    def test_stale_or_unavailable_hif_conditioning_cannot_relax_physics(self):
        for patch in ({"state_id": "s0"}, {"state_hash": "old"}, {"status": "unavailable"},
                      {"remaining_meter_candidate_indices": [2]}, {"failure_reasons": ["wide_prediction"]},
                      {"physical_fault_still_present": False}):
            with self.subTest(patch=patch):
                parent, candidate, action, verification = _fixture()
                verification["hif_conditioning"].update(patch)
                self.assertEqual(_assess(parent, candidate, action, verification).disposition, CandidateDisposition.REJECT)

    def test_unresolved_or_unknown_meter_evidence_cannot_use_exception(self):
        for patch in ({"target_fixed": False}, {"chi_square_alarm": True},
                      {"normalized_residual_alarm": True}, {"global_progress": -.001},
                      {"global_progress": None}, {"conditional_meter_scores": [1.] * 10},
                      {"conditional_meter_scores": [None] * 10},
                      {"power_flow_converged": False}, {"topology_feasible": False},
                      {"max_normalized_residual": 4.}, {"chi_square_statistic": 100.},
                      {"chi_square_threshold": 0.}, {"normalized_residual_threshold": float("nan")}):
            with self.subTest(patch=patch):
                parent, candidate, action, verification = _fixture()
                verification.update(patch)
                self.assertEqual(_assess(parent, candidate, action, verification).disposition, CandidateDisposition.REJECT)

    def test_ordinary_meter_without_hif_keeps_original_physical_rejection(self):
        parent, candidate, action, verification = _fixture()
        verification.pop("hif_conditioning")
        result = _assess(parent, candidate, action, verification)
        self.assertEqual(result.disposition, CandidateDisposition.REJECT)
        self.assertIn("physical_constraints_failed", result.rationale_codes)

    def test_explicit_values_or_unmodified_group_members_cannot_use_exception(self):
        for patch in ({"measurement_updates": {2: .1}}, {"suspect_group": [2, 3]},
                      {"suspect_group": [2, 2]}):
            with self.subTest(patch=patch):
                parent, candidate, action, verification = _fixture()
                action["arguments"].update(patch)
                self.assertEqual(_assess(parent, candidate, action, verification).disposition, CandidateDisposition.REJECT)

    def test_candidate_parent_mismatch_still_overrides_partial_route(self):
        parent, candidate, action, verification = _fixture()
        candidate["parent_state_id"] = "other-parent"
        result = _assess(parent, candidate, action, verification)
        self.assertEqual(result.disposition, CandidateDisposition.REJECT)
        self.assertIn("candidate_parent_mismatch", result.rationale_codes)

    def test_captured_real_candidates_pass_quality_without_relaxing_voltage_constraints(self):
        from psse_env.providers.matpower import MatpowerDeploymentProviders

        root = Path(__file__).resolve().parents[2]/"output/hif_continuation_fix_20260922"
        frozen_path = root/"frozen_mixed_scenarios.json"
        root_ids = ("r0_de51c28ced3e", "r0_b32f59dc6fa0", "r0_e50050413ab6")
        if not frozen_path.exists() or not all((root/"final_regression"/(name + ".json")).exists() for name in root_ids):
            self.skipTest("Captured provider regressions are local audit artifacts")
        frozen = {row["execution"]["scenario_id"]: row["execution"]
                  for row in json.loads(frozen_path.read_text())}
        for root_id in root_ids:
            with self.subTest(scenario=root_id):
                captured = json.loads((root/"final_regression"/(root_id + ".json")).read_text())
                events = captured["events"]
                correction_index = next(i for i, event in enumerate(events)
                                        if event["action"]["tool"] == "correct_measurements")
                correction = events[correction_index]
                creation = correction["tool_output"]["tool_metrics"]
                verification_event = events[correction_index + 1]
                self.assertEqual(verification_event["tool_output"]["error_detail"], "physical_constraint_evidence_missing")
                parent = {**deepcopy(frozen[root_id]), "state_id": creation["parent_state_id"],
                          "state_hash": creation["state_hash_before"]}
                candidate = {**deepcopy(parent), "state_id": creation["candidate_state_id"],
                    "state_hash": creation["state_hash_after"], "parent_state_id": parent["state_id"],
                    "status": "candidate", "source_action": correction["action"],
                    "policy_observation": verification_event["policy_observation"]}
                for index in correction["action"]["arguments"]["suspect_group"]:
                    candidate["measurements"][index] = captured["prediction"]["predicted_hif_measurements"][index]
                provider = MatpowerDeploymentProviders(normalized_residual_threshold=4.)
                # Reuse the recorded physical prediction, not hidden clean
                # truth. Run real balanced WLS and observable physical checks;
                # this regression requires no new HIF fitting/OpenDSS solve.
                with patch("psse_env.providers.matpower.conditioned_prediction", return_value=captured["prediction"]):
                    verification = provider.run_wls(candidate)
                parent_wls = next(event["tool_output"]["tool_metrics"] for event in reversed(events[:correction_index])
                                  if event["action"]["tool"] == "run_wls")
                parent_j = parent_wls["chi_square_statistic"]
                verification["global_progress"] = (parent_j - verification["chi_square_statistic"]) / parent_j
                oracle = provider._deployment_candidate_quality_oracle()
                result = oracle.label_candidate(parent_state=parent, candidate_state=candidate,
                    source_action=correction["action"], verification_output=verification)
                self.assertEqual(result.disposition, CandidateDisposition.ACCEPT_PARTIAL)
                self.assertEqual(result.progress_class, "meter_repaired_preexisting_hif_voltage_violation")
                self.assertFalse(verification["physical_constraints_ok"])
                self.assertEqual(parent["measurements"][:14], candidate["measurements"][:14])


if __name__ == "__main__":
    unittest.main()
