"""The learned-policy boundary retains the controller's bounded HIF proof."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import unittest

from psse_env.dagger.dataset_builder import prepare_model_policy_observation, validate_policy_payload
from psse_env.dagger.policy_adapter import LocalAliasPolicyAdapter


def _observation():
    parent, candidate = "hif-boundary-episode:s0", "hif-boundary-episode:s1"
    parent_hash, candidate_hash = "a" * 64, "b" * 64
    correction = {"tool": "correct_measurements", "arguments": {"state_id": parent, "suspect_group": [76]}}
    condition = {"status": "ready", "state_id": candidate, "state_hash": candidate_hash,
        "evidence_source": "deployment_wls:lagrangian_port", "method": "paired_opendss_effect_compensation",
        "remaining_meter_candidate_indices": [], "failure_reasons": [], "physical_fault_still_present": True}
    proof = {"contract": "hif_conditioned_meter_nonregression_v1",
        "evidence_source": "controller_observed:stored_state_hif_meter_nonregression",
        "parent_state_id": parent, "parent_state_hash": parent_hash,
        "candidate_state_id": candidate, "candidate_state_hash": candidate_hash,
        "case_unchanged": True, "non_target_measurements_unchanged": True,
        "preexisting_voltage_violations_unchanged": True,
        "physical_fault_still_present": True, "operator_review_required": True,
        "source_action": correction}
    verification = {"state_id": candidate, "state_hash": candidate_hash,
        "evidence_source": "deployment_wls:lagrangian_port", "globally_resolved": True,
        "physical_constraints_ok": False, "physical_evidence_complete": True,
        "hif_conditioning": condition, "hif_meter_nonregression": proof}
    output = {"execution_status": "success", "state_mutated": True,
        "active_state_id": parent, "candidate_state_id": candidate, "tool_metrics": verification}
    return {"active_state_id": parent, "candidate_state_id": candidate,
        "candidate_status": "verified", "candidate_lifecycle": "VERIFIED_CANDIDATE",
        "has_verified_candidate": True, "remaining_budget": 30,
        "last_tool": "run_wls", "last_verification": verification, "last_tool_output": output,
        "fresh_context_evidence": {"hif_conditioning": {
            **condition, "state_id": parent, "state_hash": parent_hash, "remaining_meter_candidate_indices": [76]}},
        "history_window": [{"action": {"tool": "run_wls", "arguments": {"state_id": candidate}},
                             "tool_output": output}]}


class HIFPolicySerializationTests(unittest.TestCase):
    def _assert_receipt(self, model):
        verification = model["last_verification"]
        proof = verification["hif_meter_nonregression"]
        condition = verification["hif_conditioning"]
        self.assertEqual(proof["contract"], "hif_conditioned_meter_nonregression_v1")
        self.assertEqual(proof["parent_state_id"], model["active_state_id"])
        self.assertEqual(proof["candidate_state_id"], model["candidate_state_id"])
        self.assertEqual(proof["parent_state_hash"], model["fresh_context_evidence"]["hif_conditioning"]["state_hash"])
        self.assertEqual(proof["candidate_state_hash"], verification["state_hash"])
        self.assertEqual(proof["candidate_state_hash"], condition["state_hash"])
        self.assertRegex(proof["parent_state_hash"], r"^h\d+$")
        self.assertRegex(proof["candidate_state_hash"], r"^h\d+$")
        self.assertEqual(proof["source_action"]["arguments"]["state_id"], model["active_state_id"])
        self.assertEqual(proof["source_action"]["arguments"]["suspect_group"], [76])
        self.assertTrue(proof["physical_fault_still_present"])
        self.assertTrue(proof["operator_review_required"])
        self.assertTrue(proof["preexisting_voltage_violations_unchanged"])
        self.assertFalse(verification["physical_constraints_ok"])
        self.assertEqual(condition["remaining_meter_candidate_indices"], [])
        self.assertEqual(condition["failure_reasons"], [])
        self.assertNotIn("_omitted_fields", proof)

    def test_whole_small_receipt_survives_both_alias_orders(self):
        raw = _observation()
        original = deepcopy(raw)
        for early_alias in (False, True):
            with self.subTest(alias_before_compaction=early_alias):
                model, _ = prepare_model_policy_observation(raw, alias_before_compaction=early_alias)
                self._assert_receipt(model)
                last_output = model["last_tool_output"]["observable_metrics"]
                self.assertEqual(last_output["hif_meter_nonregression"], model["last_verification"]["hif_meter_nonregression"])
                self.assertEqual(model["history_window"][-1]["observable_metrics"]["hif_conditioning"], model["last_verification"]["hif_conditioning"])
                validate_policy_payload({"state": model})
        self.assertEqual(raw, original)

    def test_durable_proof_survives_history_character_budget(self):
        model, _ = prepare_model_policy_observation(_observation(), max_history_chars=64)
        self._assert_receipt(model)
        self.assertEqual(model["history_window"][0]["summary"], "history omitted for size")

    def test_receipt_whitelist_drops_vectors_truth_and_unknown_action_arguments(self):
        raw = _observation()
        verification = raw["last_verification"]
        proof = verification["hif_meter_nonregression"]
        proof["clean_measurements"] = [999.] * 122
        proof["true_measurement_errors"] = [{"index": 76}]
        proof["prediction_vector"] = [111.] * 122
        proof["source_action"]["arguments"]["measurement_updates"] = {76: 999.}
        verification["hif_conditioning"]["predicted_hif_measurements"] = [222.] * 122
        model, _ = prepare_model_policy_observation(raw)
        self._assert_receipt(model)
        rendered = json.dumps(model["last_verification"], sort_keys=True)
        for forbidden in ("clean_measurements", "true_measurement_errors", "prediction_vector", "measurement_updates", "predicted_hif_measurements"):
            self.assertNotIn(forbidden, rendered)

    def test_serializer_never_mints_missing_or_successful_evidence(self):
        raw = _observation()
        raw["last_verification"] = {"physical_constraints_ok": False}
        model, _ = prepare_model_policy_observation(raw)
        self.assertEqual(model["last_verification"], {"physical_constraints_ok": False})
        raw = _observation()
        raw["last_verification"]["hif_conditioning"].update(status="unavailable", failure_reasons=["wide_prediction"])
        model, _ = prepare_model_policy_observation(raw)
        self.assertEqual(model["last_verification"]["hif_conditioning"]["status"], "unavailable")
        self.assertEqual(model["last_verification"]["hif_conditioning"]["failure_reasons"], ["wide_prediction"])

    def test_real_controller_candidate_proof_reaches_canonical_model_adapter(self):
        artifact = Path(__file__).resolve().parents[2]/"output/hif_continuation_fix_20260922/final_verified/r0_de51c28ced3e.json"
        if not artifact.exists():
            self.skipTest("Real HIF continuation trace is a local audit artifact")
        captured = json.loads(artifact.read_text())
        raw = next(event["policy_observation"] for event in captured["events"] if event["action"]["tool"] == "commit_state")
        class CapturePolicy:
            observation = None
            def act(self, observation):
                self.observation = deepcopy(observation)
                return {"tool": "commit_state", "arguments": {"case_path": "candidate"}}
        actor = CapturePolicy()
        action = LocalAliasPolicyAdapter(actor, protocol="canonical").act(raw)
        self._assert_receipt(actor.observation)
        self.assertEqual(action, {"tool": "commit_state", "arguments": {"candidate_state_id": raw["candidate_state_id"]}})
        validate_policy_payload({"state": actor.observation})


if __name__ == "__main__":
    unittest.main()
