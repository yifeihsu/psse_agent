"""Controller integrity fingerprints never reach the model-visible observation.

Regression for the 2026-09-24 stage-0 failure: an accepted HIF fit carries a
``conditioning_fit`` receipt (case, acquisition and model digests) in
``explained_anomalies[].detail``; under wls_gated_diagnostics explanations are
recorded, so the SFT export found an unaliased 64-hex identifier.
"""
from copy import deepcopy

from psse_env.dagger.dataset_builder import (
    find_model_identifier_leaks,
    prepare_model_policy_observation,
    without_controller_fingerprints,
)

DIGEST = "ab" * 32


def _receipt():
    return {
        "success": True,
        "candidate_branch_row0": 3,
        "estimated": {"alpha": 0.41, "r_hif_ohm": 152.9},
        "case_sha256": DIGEST,
        "independent_of_current_scada": True,
        "acquisition_sha256": "cd" * 32,
        "model_sha256": "ef" * 32,
    }


def _observation():
    explanation = {"family": "hif", "signature": "hif_suspected_line_differential",
                   "detail": {"branch_row0": 3, "phase": "A", "conditioning_fit": _receipt()}}
    return {
        "active_state_id": "r0_abc_episode0:s0",
        "evidence_profile": "wls_gated_diagnostics",
        "unresolved_signatures": [],
        "explained_anomalies": [explanation],
        "last_tool": "estimate_hif_location_magnitude_multiscan_from_path",
        "last_tool_output": {"execution_status": "success",
                             "tool_metrics": {"anomaly_explanation": deepcopy(explanation)}},
        "history_window": [{"step": 3, "tool": "estimate_hif_location_magnitude_multiscan_from_path",
                            "tool_metrics": {"anomaly_explanation": deepcopy(explanation)}}],
    }


def test_fingerprint_keys_are_dropped_recursively_without_mutating_the_input():
    value = {"a_sha256": DIGEST, "keep": [{"model_fingerprint": "x", "b": 1}], "nested": (1, {"c_sha256": DIGEST})}
    original = deepcopy(value)
    assert without_controller_fingerprints(value) == {"keep": [{"b": 1}], "nested": [1, {}]}
    assert value == original


def test_accepted_hif_explanation_exports_without_identifier_leaks():
    observation = _observation()
    original = deepcopy(observation)
    prepared, _ = prepare_model_policy_observation(observation)
    assert find_model_identifier_leaks({"state": prepared}) == []
    fit = prepared["explained_anomalies"][0]["detail"]["conditioning_fit"]
    assert not any(key.endswith("_sha256") for key in fit)
    assert fit["candidate_branch_row0"] == 3 and fit["independent_of_current_scada"] is True
    # The environment still reads the receipt from its own observation.
    assert observation == original
    assert observation["explained_anomalies"][0]["detail"]["conditioning_fit"]["case_sha256"] == DIGEST


def test_replay_stable_view_also_drops_fingerprints():
    prepared, _ = prepare_model_policy_observation(_observation(), alias_before_compaction=True)
    assert find_model_identifier_leaks({"state": prepared}) == []


def test_durable_nlm_localization_stays_controller_only():
    observation = _observation()
    observation["fresh_context_evidence"] = {"three_phase": {
        "state_id": "r0_abc_episode0:s0", "status": "available",
        "nlm_localization": {"top_hif_branch_rows": [3, 7], "suspected_phase": "B"}}}
    prepared, _ = prepare_model_policy_observation(observation)
    assert "nlm_localization" not in prepared["fresh_context_evidence"]["three_phase"]
    assert prepared["fresh_context_evidence"]["three_phase"]["status"] == "available"
    assert observation["fresh_context_evidence"]["three_phase"]["nlm_localization"]["suspected_phase"] == "B"
