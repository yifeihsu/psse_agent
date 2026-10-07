"""Teacher fix of 2026-10-06: after balanced phasors, spectra come before a voltage-meter edit.

The 2026-10-02 round-2 losses were all at one decision under
suspicion_gated_diagnostics: what follows balanced phasors on a voltage-meter
suspicion.  The teacher edited the meter when the screen had explained the
alarm and asked for spectra when it had not, and the balanced evidence
cannot tell the two cases apart.  Now the spectra are asked for first in
both cases; a later state of the episode that already looked at them edits
the meter directly.
"""
from __future__ import annotations

from psse_env.actions import (
    CORRECT_MEASUREMENTS, GET_HARMONIC_CONTEXT, GET_MEASUREMENT_CONTEXT, GET_THREE_PHASE_CONTEXT, RUN_THREE_PHASE_NLM_FROM_PATH,
    RUN_WLS, action_signature,
)
from psse_env.oracle import ExpertPolicyOracle, ProcessValidityOracle

ACTIVE = "s0"
HASH = "h"
VM_CHANNEL = 6  # bus 7's voltage magnitude on IEEE 14 (channel index 6)


def _screen(*, explained: bool):
    """A valid screen with a voltage-meter suspicion; ``explained`` says whether its hypotheses explain the alarm."""
    accepted = [{"class": "meter", "channel_index0": VM_CHANNEL, "J": 95.0}] if explained else []
    return {
        "status": "valid", "suspected": False, "outcome": "meter" if explained else None, "explained": explained,
        "unexplained": not explained, "accepted_hypotheses": accepted, "voltage_meter_channels": [VM_CHANNEL],
        "phasor_suspicion": {"hif": False, "voltage_meter": True, "unexplained": not explained},
        "rounds": [{"winner": "meter", "clean_after_removal": explained, "set_aside_channels": [VM_CHANNEL],
                    "scores": {"meter": 40.0, "parameter": 2.0, "topology": -1.0, "hif": 1.0},
                    "best": {"meter": {"channel_index0": VM_CHANNEL, "J": 95.0}}, "ranked": {"meter": [{"channel_index0": VM_CHANNEL, "J": 95.0}]}}],
    }


def _state(*, explained: bool, phase=None, tried=()):
    contexts = {"wls": {"state_id": ACTIVE, "state_hash": HASH, "evidence_source": "deployment_wls:test", "successful": True,
                        "anomalous": True, "chi_square_alarm": True, "normalized_residual_alarm": True,
                        "max_normalized_residual": 9.5, "anomaly_breadth": 0.05, "bus_count": 14, "hif_screen": _screen(explained=explained)}}
    tried_signatures = [action_signature({"tool": RUN_WLS, "arguments": {"state_id": ACTIVE}})]
    available = []
    if phase is not None:
        contexts["three_phase"] = {"state_id": ACTIVE, "state_hash": HASH, "evidence_source": "deployment_context:three_phase_measurements",
                                   "request_attempted": True, "three_phase_context_status": "available",
                                   "available_evidence_channels": ["three_phase_voltages", "three_phase_branch_currents"], **phase}
        tried_signatures.append(action_signature({"tool": GET_THREE_PHASE_CONTEXT, "arguments": {"state_id": ACTIVE}}))
        available = ["three_phase_voltages", "three_phase_branch_currents"]
    tried_signatures += [action_signature(action) for action in tried]
    return {
        "evidence_profile": "suspicion_gated_diagnostics", "active_state_id": ACTIVE, "candidate_state_id": None,
        "has_open_candidate": False, "remaining_budget": 30, "remaining_anomaly_score": 4.2, "no_material_anomaly_remaining": False,
        "unresolved_signatures": [f"wls_residual_outlier_dominant index={VM_CHANNEL} channel=Vm", f"wls_voltage_meter_suspected index={VM_CHANNEL}"],
        "explained_anomalies": [], "accepted_corrections": [], "rejected_hypotheses": [], "available_evidence": available,
        "last_tool": RUN_WLS, "last_tool_status": "success", "tried_action_signatures": tried_signatures,
        "fresh_context_evidence": contexts,
    }


def _first(state):
    oracle = ExpertPolicyOracle(process_oracle=ProcessValidityOracle(executor_hydrated_corrections=True))
    actions = oracle.next_actions(state, [])
    assert actions, state
    return actions[0]["tool"]


def test_the_voltage_meter_suspicion_acquires_phasors_then_tests_them():
    for explained in (True, False):
        assert _first(_state(explained=explained)) == GET_THREE_PHASE_CONTEXT
        assert _first(_state(explained=explained, phase={"nlm_attempted": False})) == RUN_THREE_PHASE_NLM_FROM_PATH


def test_balanced_phasors_ask_for_spectra_before_the_meter_edit_whether_or_not_the_screen_explained_the_alarm():
    balanced = {"nlm_attempted": True, "nlm_classification": "balanced_three_phase"}
    for explained in (True, False):
        assert _first(_state(explained=explained, phase=balanced)) == GET_HARMONIC_CONTEXT, explained


def test_spectra_already_examined_in_the_episode_release_the_meter_edit():
    balanced = {"nlm_attempted": True, "nlm_classification": "balanced_three_phase"}
    state = _state(explained=True, phase=balanced)
    state["fresh_context_evidence"]["harmonic"] = {
        "state_id": ACTIVE, "state_hash": HASH, "evidence_source": "deployment_context:harmonic_measurements",
        "request_attempted": True, "harmonic_context_status": "available", "harmonic_distortion_detected": False,
        "available_evidence_channels": [],
    }
    state["tried_action_signatures"].append(action_signature({"tool": GET_HARMONIC_CONTEXT, "arguments": {"state_id": ACTIVE}}))
    assert _first(state) in {GET_MEASUREMENT_CONTEXT, CORRECT_MEASUREMENTS}
