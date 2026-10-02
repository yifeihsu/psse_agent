"""The unexplained-discrepancy handoff needs phasors that named no event (2026-10-02).

Round-2 research collection of the 2026-10-01 ranked cell stopped on a pure
HIF root: the phasors named the HIF, its estimate was accepted, the
HIF-conditioned WLS still flagged one active-power residual, and the
measurement context offered no correction.  The expert then asked for an
unexplained-discrepancy handoff because the phasors had been examined, and
the environment refused it: the phasors had named a diagnosable event and no
spectra had been taken on the state.  The handoff is unexplained only when
both acquisition tiers answered on the active state and named nothing;
otherwise the balanced recovery options are exhausted, which the environment
accepts on the same shared conditions.
"""
from __future__ import annotations

from copy import deepcopy

import pytest

from psse_env.actions import (
    ASK_FOR_MORE_EVIDENCE, GET_HARMONIC_CONTEXT, RECOVERY_OPTIONS_EXHAUSTED_REQUEST, RUN_WLS,
    UNEXPLAINED_DISCREPANCY_REQUEST, action_signature, unexplained_handoff_supported,
)
from psse_env.evidence_profile import SUSPICION_GATED_PROFILE
from psse_env.oracle import ExpertPolicyOracle, ProcessValidityOracle

ACTIVE = "gated:s1"
ANCESTOR = "gated:s0"
HASH = "content-current"
RESIDUAL = "wls_residual_outlier index=93 channel=Pt"
HIF_EXPLANATION = {"tool": "estimate_hif_location_magnitude_from_path", "state_id": ACTIVE, "family": "hif",
                   "kind": "hif_pmu_snapshot_accepted", "detail": {"candidate_branch_row0": 5}}


def _state(classification=None, *, nlm_attempted=True, spectra_state=None, explained=()):
    """A suspicion-gated observation whose balanced routes are exhausted while one residual stands."""
    contexts = {
        "wls": {"state_id": ACTIVE, "state_hash": HASH, "evidence_source": "deployment_wls:test", "successful": True,
                "anomalous": True, "chi_square_alarm": False, "normalized_residual_alarm": True,
                "normalized_residual_threshold": 4.0, "max_normalized_residual": 4.6},
        "three_phase": {"state_id": ACTIVE, "state_hash": HASH, "request_attempted": True,
                        "three_phase_context_status": "available", "nlm_attempted": nlm_attempted},
    }
    if classification is not None:
        contexts["three_phase"]["nlm_classification"] = classification
    tried = [action_signature({"tool": RUN_WLS, "arguments": {"state_id": ACTIVE}})]
    if spectra_state is not None:
        tried.append(action_signature({"tool": GET_HARMONIC_CONTEXT, "arguments": {"state_id": spectra_state}}))
        if spectra_state == ACTIVE:
            contexts["harmonic"] = {"state_id": ACTIVE, "state_hash": HASH, "request_attempted": True,
                                    "harmonic_context_status": "available", "harmonic_distortion_detected": False}
    return {
        "evidence_profile": SUSPICION_GATED_PROFILE, "active_state_id": ACTIVE, "candidate_state_id": None,
        "has_open_candidate": False, "remaining_budget": 30, "remaining_anomaly_score": 1.2,
        "no_material_anomaly_remaining": False, "unresolved_signatures": [RESIDUAL],
        "explained_anomalies": list(explained), "accepted_corrections": [], "rejected_hypotheses": [],
        "available_evidence": [], "tried_action_signatures": tried, "fresh_context_evidence": contexts,
        "has_fresh_measurement_context": True, "measurement_context_state_id": ACTIVE,
    }


def _handoff(state):
    expert = ExpertPolicyOracle(process_oracle=ProcessValidityOracle(executor_hydrated_corrections=True))
    proposals = expert._recovery_exhaustion_proposals(deepcopy(state), [])
    assert len(proposals) == 1, proposals
    action = proposals[0].action
    assert action["tool"] == ASK_FOR_MORE_EVIDENCE and action["arguments"]["state_id"] == ACTIVE
    return action["arguments"]["request"], list(proposals[0].evidence_codes)


def test_only_balanced_phasors_and_same_state_spectra_support_the_unexplained_handoff():
    assert unexplained_handoff_supported(_state("balanced_three_phase", spectra_state=ACTIVE))
    assert unexplained_handoff_supported(_state("unresolved", spectra_state=ACTIVE))
    for classification in ("hif_suspected", "three_phase_unbalance"):
        assert not unexplained_handoff_supported(_state(classification, spectra_state=ACTIVE)), classification
    # No classification on record: the NLM never answered (or failed) on this state.
    assert not unexplained_handoff_supported(_state(None, spectra_state=ACTIVE))
    assert not unexplained_handoff_supported(_state("balanced_three_phase", nlm_attempted=False, spectra_state=ACTIVE))
    # Spectra on an ancestor state, or none at all, do not count for this state.
    assert not unexplained_handoff_supported(_state("balanced_three_phase", spectra_state=ANCESTOR))
    assert not unexplained_handoff_supported(_state("balanced_three_phase"))
    # Phasors carried over from another state are not this state's ledger.
    moved = _state("balanced_three_phase", spectra_state=ACTIVE)
    moved["fresh_context_evidence"]["three_phase"]["state_id"] = ANCESTOR
    assert not unexplained_handoff_supported(moved)
    # A spectra request that failed or is still pending, or an answer carried
    # over a commit, is no successful spectra evidence on this state.
    for status in ("failed", "pending"):
        broken = _state("balanced_three_phase", spectra_state=ACTIVE)
        broken["fresh_context_evidence"]["harmonic"]["harmonic_context_status"] = status
        assert not unexplained_handoff_supported(broken), status
    carried = _state("balanced_three_phase", spectra_state=ACTIVE)
    carried["fresh_context_evidence"]["harmonic"]["carried_from_state_id"] = ANCESTOR
    assert not unexplained_handoff_supported(carried)


def test_an_alarm_that_outlived_an_accepted_hif_is_an_exhausted_recovery():
    """The round-2 failure: phasors named the HIF, its estimate was accepted, a residual stood."""
    request, evidence = _handoff(_state("hif_suspected", explained=[HIF_EXPLANATION]))
    assert request == RECOVERY_OPTIONS_EXHAUSTED_REQUEST
    assert "phasors_examined_on_state" in evidence
    assert "unexplained_balanced_discrepancy_requires_operator_handoff" not in evidence
    request, _ = _handoff(_state("three_phase_unbalance"))
    assert request == RECOVERY_OPTIONS_EXHAUSTED_REQUEST


def test_balanced_phasors_and_spectra_on_the_state_end_the_ladder_unexplained():
    request, evidence = _handoff(_state("balanced_three_phase", spectra_state=ACTIVE))
    assert request == UNEXPLAINED_DISCREPANCY_REQUEST
    assert "unexplained_balanced_discrepancy_requires_operator_handoff" in evidence


@pytest.mark.parametrize("spectra_state", [ANCESTOR])
def test_spectra_taken_on_an_ancestor_state_leave_an_exhausted_recovery(spectra_state):
    # The spectra tier is closed for the episode, but the environment audits
    # an unexplained handoff against spectra bound to the active state.
    request, _ = _handoff(_state("balanced_three_phase", spectra_state=spectra_state))
    assert request == RECOVERY_OPTIONS_EXHAUSTED_REQUEST


def test_without_examined_phasors_the_handoff_is_unchanged():
    state = _state("balanced_three_phase", spectra_state=ACTIVE)
    del state["fresh_context_evidence"]["three_phase"]
    del state["fresh_context_evidence"]["harmonic"]
    request, evidence = _handoff(state)
    assert request == RECOVERY_OPTIONS_EXHAUSTED_REQUEST
    assert "phasors_examined_on_state" not in evidence
