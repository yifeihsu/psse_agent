"""Expert and operator-escalation audit agree on what was tried (DAgger round 1).

Round-1 research collection stopped on a measurement+hif root with
``same_state_supported_corrections_unexhausted``: the learner had issued a
correction the process gate refused, the expert then treated that refused
attempt as tried and handed off, while the environment audit (correctly) saw
an untried, provider-supported correction.  Three mechanisms are covered:

* M1: a process-gate refusal (``missing_precondition``, lifecycle, ...) tests
  nothing, so it neither retires a correction nor a context request.
* M2: a balanced context inventory the controller retired without a state
  change (telemetry answer, accepted HIF fit) must be requested again.
* M3: the HIF ladder keeps the NLM localization after the NLM output has left
  the bounded policy history window.

The unit fixtures pin the expert decision; the real-environment tests drive the
research environment and require the collector's training-decision check to
pass at every visited state.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from psse_env.actions import (
    ASK_FOR_MORE_EVIDENCE, CORRECT_MEASUREMENTS, ESTIMATE_HIF_FROM_PATH,
    ESTIMATE_HIF_MULTISCAN_FROM_PATH, GET_HARMONIC_CONTEXT, GET_MEASUREMENT_CONTEXT,
    GET_PARAMETER_CONTEXT, GET_THREE_PHASE_CONTEXT, GET_TOPOLOGY_CONTEXT,
    PROCESS_REJECTION_ERROR_CODES, RECOVERY_OPTIONS_EXHAUSTED_REQUEST,
    RUN_THREE_PHASE_NLM_FROM_PATH, RUN_WLS, action_signature,
)
from psse_env.evidence_profile import WLS_GATED_PROFILE
from psse_env.oracle import ExpertPolicyOracle, ProcessValidityOracle

ACTIVE = "gated:s0"
HASH = "content-current"
WLS_SIGNATURES = [
    "wls_residual_outlier_dominant index=26 channel=Pinj",
    "wls_branch_multiplier line_status_or_parameter line=4",
]
MINTED_HIF = "hif_suspected_line_differential"
PHASE_CHANNELS = ["three_phase_voltages", "three_phase_branch_currents"]
METER_26 = {"tool": CORRECT_MEASUREMENTS, "arguments": {"state_id": ACTIVE, "suspect_group": [26]}}


# ------------------------------------------------------------------ fixtures


def _state(*, signatures=None):
    """Policy observation after an alarming opening WLS on the active state."""
    return {
        "evidence_profile": WLS_GATED_PROFILE, "active_state_id": ACTIVE, "has_open_candidate": False,
        "remaining_budget": 30, "remaining_anomaly_score": 2.0, "no_material_anomaly_remaining": False,
        "unresolved_signatures": list(WLS_SIGNATURES if signatures is None else signatures),
        "explained_anomalies": [], "accepted_corrections": [], "rejected_hypotheses": [],
        "available_evidence": [], "last_tool": RUN_WLS, "last_tool_status": "success",
        "tried_action_signatures": [action_signature({"tool": RUN_WLS, "arguments": {"state_id": ACTIVE}})],
        "fresh_context_evidence": {"wls": {
            "state_id": ACTIVE, "state_hash": HASH, "evidence_source": "deployment_wls:test",
            "successful": True, "anomalous": True, "anomaly_breadth": 0.05,
            "chi_square_alarm": True, "normalized_residual_alarm": True,
            "normalized_residual_threshold": 4.0, "max_normalized_residual": 8.0,
        }},
    }


def _telemetry(state, family, *, available, channels=(), nlm_attempted=None):
    record = {
        "state_id": ACTIVE, "state_hash": HASH, "evidence_source": f"deployment_context:{family}_measurements",
        "request_attempted": True, f"{family}_context_status": "available" if available else "unavailable",
        "available_evidence_channels": list(channels),
    }
    if family == "harmonic":
        record["harmonic_distortion_detected"] = False
    if nlm_attempted is not None:
        record["nlm_attempted"] = nlm_attempted
    state["fresh_context_evidence"][family] = record
    if available:
        state["available_evidence"] = sorted(set(state["available_evidence"]) | set(channels))
    return record


def _balanced_state():
    """Both telemetry requests answered unavailable: the balanced ladder is open."""
    state = _state()
    _telemetry(state, "three_phase", available=False)
    _telemetry(state, "harmonic", available=False)
    return state


def _context(state, family, targets=()):
    tool = {"measurement": CORRECT_MEASUREMENTS, "parameter": "correct_parameters",
            "topology": "correct_topology"}[family]
    state[f"has_fresh_{family}_context"] = True
    state[f"{family}_context_state_id"] = ACTIVE
    state["fresh_context_evidence"][family] = {
        "state_id": ACTIVE, "state_hash": HASH, "evidence_source": f"deployment_context:{family}",
        "route_status": "actionable" if targets else "complete_negative",
        "supported_corrections": [{"tool": tool, "arguments": {"state_id": ACTIVE, **target}} for target in targets],
    }


def _retire(state, family):
    """What the controller does after a telemetry answer or an accepted HIF fit."""
    state[f"has_fresh_{family}_context"] = False
    state[f"{family}_context_state_id"] = None
    state["fresh_context_evidence"].pop(family, None)


def _tried(state, action):
    state["tried_action_signatures"].append(action_signature(action))


def _event(action, *, status="success", error_code=None, metrics=None):
    output = {"execution_status": status, "tool_metrics": dict(metrics or {})}
    if error_code is not None:
        output["error_code"] = error_code
    return {"state_id": ACTIVE, "action": deepcopy(action), "tool_output": output}


def _conditioning(candidates):
    return {"status": "ready", "state_id": ACTIVE, "state_hash": HASH,
            "method": "paired_opendss_effect_compensation", "physical_fault_still_present": True,
            "remaining_meter_candidate_indices": list(candidates), "failure_reasons": []}


def _accepted_hif_state(candidates=(26,)):
    state = _state(signatures=[*WLS_SIGNATURES, MINTED_HIF])
    _telemetry(state, "three_phase", available=True, channels=PHASE_CHANNELS, nlm_attempted=True)
    state["available_evidence"] = sorted(set(state["available_evidence"]) | {"hif_scan_window"})
    state["explained_anomalies"] = [{
        "family": "hif", "tool": ESTIMATE_HIF_MULTISCAN_FROM_PATH, "state_id": ACTIVE,
        "explained_signatures": [MINTED_HIF],
        "detail": {"conditioning_fit": {"success": True}, "candidate_branch_row0": 3}}]
    state["fresh_context_evidence"]["hif_conditioning"] = _conditioning(candidates)
    return state


def _oracle():
    return ExpertPolicyOracle(process_oracle=ProcessValidityOracle(executor_hydrated_corrections=True))


def _first(state, history=()):
    actions = _oracle().next_actions(deepcopy(state), [deepcopy(item) for item in history])
    assert actions, state
    return actions[0]


# ------------------------------------------------ M1: refused corrections


@pytest.mark.parametrize("window", ["visible", "outside_window"])
def test_a2_refused_correction_does_not_retire_the_conditioned_meter(window):
    """measurement+hif: the learner copied the conditioned meter before the context."""
    state = _accepted_hif_state()
    _context(state, "measurement", [{"suspect_group": [26]}])
    _tried(state, METER_26)
    history = [_event(METER_26, status="failure", error_code="missing_precondition")] if window == "visible" else []
    assert _first(state, history) == METER_26


def test_c_refused_balanced_correction_does_not_retire_the_meter():
    """measurement: a correction refused for a missing context is still untried."""
    state = _balanced_state()
    _context(state, "measurement", [{"suspect_group": [26]}])
    _tried(state, METER_26)
    history = [_event(METER_26, status="failure", error_code="missing_precondition")]
    assert _first(state, history) == METER_26


@pytest.mark.parametrize("code", sorted(PROCESS_REJECTION_ERROR_CODES))
def test_every_process_rejection_code_leaves_the_correction_untried(code):
    state = _balanced_state()
    _context(state, "measurement", [{"suspect_group": [26]}])
    _tried(state, METER_26)
    assert _first(state, [_event(METER_26, status="failure", error_code=code)]) == METER_26


def test_a_real_test_still_retires_the_correction():
    """A verification rollback or executor failure is durable exhaustion."""
    for kind in ("rollback", "executor_failure"):
        state = _balanced_state()
        _context(state, "measurement", [{"suspect_group": [26]}])
        _tried(state, METER_26)
        record = {"candidate_parent_id": ACTIVE, "source_action": deepcopy(METER_26),
                  "action_signature": action_signature(METER_26)}
        if kind == "rollback":
            record["candidate_state_id"] = "gated:s1"
        else:
            record.update(rejection_kind="executor_failure", error_code="measurement_correction_failure")
        state["rejected_hypotheses"] = [record]
        first = _first(state)
        assert first != METER_26, kind


def test_a_visible_non_process_failure_still_retires_the_correction():
    """A route refusal is exhaustion for the audit, so it stays tried here."""
    state = _balanced_state()
    _context(state, "measurement", [{"suspect_group": [26]}])
    _tried(state, METER_26)
    history = [_event(METER_26, status="failure", error_code="correction_route_not_actionable")]
    assert _first(state, history) != METER_26


# ------------------------------------------ M1/M2: context requests


def test_b_context_refused_before_wls_is_requested_again():
    state = _balanced_state()
    request = {"tool": GET_MEASUREMENT_CONTEXT, "arguments": {"state_id": ACTIVE}}
    _tried(state, request)
    history = [_event(request, status="failure", error_code="missing_precondition")]
    assert _first(state, history) == request
    # Out of the window the refusal is not a test either.
    assert _first(state, []) == request


def test_provider_no_evidence_answer_is_not_requested_again_while_visible():
    state = _balanced_state()
    request = {"tool": GET_MEASUREMENT_CONTEXT, "arguments": {"state_id": ACTIVE}}
    _tried(state, request)
    history = [_event(request, status="failure", error_code="insufficient_observable_evidence")]
    assert GET_MEASUREMENT_CONTEXT not in {action["tool"] for action in _oracle().next_actions(state, history)}


def test_m2_retired_context_is_requested_again_instead_of_a_handoff():
    """A telemetry answer retired the served measurement inventory (no state change)."""
    state = _balanced_state()
    request = {"tool": GET_MEASUREMENT_CONTEXT, "arguments": {"state_id": ACTIVE}}
    _context(state, "measurement", [{"suspect_group": [26]}])
    _tried(state, request)
    history = [_event(request, metrics={"supported_corrections": [deepcopy(METER_26)]})]
    # While fresh the context is not repeated: the supported correction follows.
    assert _first(state, history) == METER_26
    _retire(state, "measurement")
    _tried(state, {"tool": GET_THREE_PHASE_CONTEXT, "arguments": {"state_id": ACTIVE}})
    history.append(_event({"tool": GET_THREE_PHASE_CONTEXT, "arguments": {"state_id": ACTIVE}}))
    assert _first(state, history) == request
    assert _first(state, history)["arguments"].get("request") != RECOVERY_OPTIONS_EXHAUSTED_REQUEST


def test_m2_parameter_and_topology_contexts_follow_the_same_freshness_rule():
    state = _state(signatures=["wls_branch_multiplier_dominant line_status_or_parameter line=3"])
    _telemetry(state, "three_phase", available=False)
    _telemetry(state, "harmonic", available=False)
    history = []
    for tool in (GET_PARAMETER_CONTEXT, GET_TOPOLOGY_CONTEXT):
        request = {"tool": tool, "arguments": {"state_id": ACTIVE}}
        _tried(state, request)
        history.append(_event(request))
    assert _first(state, history)["tool"] in {GET_PARAMETER_CONTEXT, GET_TOPOLOGY_CONTEXT}


# ------------------------------------------------ M3: durable NLM localization


def _hif_suspected_state(*, localization=True, bound=True):
    state = _state(signatures=[*WLS_SIGNATURES, MINTED_HIF])
    record = _telemetry(state, "three_phase", available=True, channels=PHASE_CHANNELS, nlm_attempted=True)
    state["available_evidence"] = sorted(set(state["available_evidence"]) | {"hif_scan_window"})
    if localization:
        record["nlm_localization"] = {"top_hif_branch_rows": [3, 7], "suspected_phase": "B"}
    if not bound:
        record["state_id"] = "gated:previous"
    for tool in (RUN_THREE_PHASE_NLM_FROM_PATH, GET_THREE_PHASE_CONTEXT):
        _tried(state, {"tool": tool, "arguments": {"state_id": ACTIVE}})
    return state


def test_d_hif_ladder_uses_the_durable_nlm_localization_outside_the_window():
    state = _hif_suspected_state()
    # Four unrelated learner actions pushed the NLM output out of the window.
    history = [_event({"tool": GET_PARAMETER_CONTEXT, "arguments": {"state_id": ACTIVE}}) for _ in range(4)]
    assert _first(state, history) == {"tool": ESTIMATE_HIF_MULTISCAN_FROM_PATH, "arguments": {
        "state_id": ACTIVE, "candidate_branch_row0": 3, "candidate_phase": "B"}}


def test_d_ladder_continues_to_the_single_scan_estimator_from_the_ledger():
    state = _hif_suspected_state()
    multiscan = {"tool": ESTIMATE_HIF_MULTISCAN_FROM_PATH,
                 "arguments": {"state_id": ACTIVE, "candidate_branch_row0": 3, "candidate_phase": "B"}}
    _tried(state, multiscan)
    history = [_event(multiscan, metrics={"state_id": ACTIVE, "state_hash": HASH,
                                          "diagnostic_acceptance": {"accepted": False}})]
    assert _first(state, history) == {"tool": ESTIMATE_HIF_FROM_PATH, "arguments": {
        "state_id": ACTIVE, "candidate_branch_row0": 3, "candidate_phase": "B"}}


def test_d_localization_bound_to_another_acquisition_is_ignored():
    state = _hif_suspected_state(bound=False)
    actions = _oracle().next_actions(state, [])
    assert not any(action["tool"] in {ESTIMATE_HIF_MULTISCAN_FROM_PATH, ESTIMATE_HIF_FROM_PATH} for action in actions)


def test_window_nlm_output_still_drives_the_ladder_without_a_ledger_record():
    """Legacy observations without the ledger record keep the history route."""
    state = _hif_suspected_state(localization=False)
    nlm = _event({"tool": RUN_THREE_PHASE_NLM_FROM_PATH, "arguments": {"state_id": ACTIVE}}, metrics={
        "state_id": ACTIVE, "state_hash": HASH, "evidence_source": "deployment_diagnostic:three_phase_nlm",
        "nlm_summary": {"screening_mode": True, "diagnostic_classification": "hif_suspected",
                        "top_hif_groups": [{"branch_row0": 5}], "suspected_phase": "C"},
        "diagnostic_acceptance": {"accepted": False}})
    assert _first(state, [nlm])["arguments"]["candidate_branch_row0"] == 5


def _ledger_env(state_id=ACTIVE, state_hash=HASH, **three_phase):
    record = {"state_id": state_id, "state_hash": state_hash, "request_attempted": True,
              "three_phase_context_status": "available", "available_evidence_channels": list(PHASE_CHANNELS),
              **three_phase}
    return SimpleNamespace(context_flags={"fresh_context_evidence": {"three_phase": record}})


def test_controller_records_the_localization_only_on_the_screened_acquisition():
    from psse_env.transactional_env import TransactionalPSSEEnv

    metrics = {"nlm_summary": {"top_hif_groups": [{"branch_row0": 3}, {"branch_row0": "7"}, {"branch_row0": 3}],
                               "suspected_phase": "b", "fit_sha256": "not-kept"}}
    fake = _ledger_env(nlm_attempted=True)
    TransactionalPSSEEnv._record_nlm_localization(fake, ACTIVE, HASH, metrics)
    record = fake.context_flags["fresh_context_evidence"]["three_phase"]["nlm_localization"]
    assert record == {"top_hif_branch_rows": [3, 7], "suspected_phase": "B"}
    for fake in (_ledger_env(nlm_attempted=False), _ledger_env(state_hash="other", nlm_attempted=True),
                 _ledger_env(state_id="gated:other", nlm_attempted=True)):
        TransactionalPSSEEnv._record_nlm_localization(fake, ACTIVE, HASH, metrics)
        assert "nlm_localization" not in fake.context_flags["fresh_context_evidence"]["three_phase"]
    # A balanced screen clears an earlier localization on the same acquisition.
    fake = _ledger_env(nlm_attempted=True, nlm_localization={"top_hif_branch_rows": [3], "suspected_phase": None})
    TransactionalPSSEEnv._record_nlm_localization(fake, ACTIVE, HASH, {"nlm_summary": {"top_hif_groups": []}})
    assert "nlm_localization" not in fake.context_flags["fresh_context_evidence"]["three_phase"]


def test_localization_is_dropped_with_the_acquisition_on_commit_and_state_change():
    from psse_env.transactional_env import TransactionalPSSEEnv

    fake = _ledger_env(nlm_attempted=True, nlm_localization={"top_hif_branch_rows": [3], "suspected_phase": "A"})
    TransactionalPSSEEnv._invalidate_context_flags(fake)
    assert "three_phase" not in fake.context_flags["fresh_context_evidence"]
    fake = _ledger_env(nlm_attempted=True, nlm_localization={"top_hif_branch_rows": [3], "suspected_phase": "A"})
    TransactionalPSSEEnv._rebind_telemetry_requests(fake, "gated:s1", "content-next")
    assert "three_phase" not in fake.context_flags["fresh_context_evidence"]


# ----------------------------------------------------------- real environment


def _hif_corpora():
    from psse_env.providers.scenario_generator import PHYSICAL_HIF_SAMPLE_PATHS

    paths = [PHYSICAL_HIF_SAMPLE_PATHS[1], PHYSICAL_HIF_SAMPLE_PATHS[0]]
    return paths if all(path.is_file() for path in paths) else None


@pytest.fixture(scope="module")
def real_roots():
    pytest.importorskip("opendssdirect")
    paths = _hif_corpora()
    if paths is None:
        pytest.skip("20260923opf physical HIF corpora are not checked out")
    from psse_env.providers.scenario_generator import Round0ScenarioGenerator

    generator = Round0ScenarioGenerator(
        seed=20260925, source_partition="train", hif_sample_paths=paths,
        normalized_residual_threshold=4.0, parameter_ranking_dominance_threshold=1.2, hif_max_scans=10,
    )
    roots = {}
    for scenario in generator.build({"measurement+hif": 1, "measurement": 1}):
        roots.setdefault(scenario["scenario_family"], scenario)
    return roots


@pytest.fixture(scope="module")
def real_env():
    """One research environment reused across tests so the HIF fit is memoised."""
    from psse_env.providers import MatpowerDeploymentProviders
    from psse_env.transactional_env import TransactionalPSSEEnv

    providers = MatpowerDeploymentProviders(
        evidence_profile=WLS_GATED_PROFILE, chi2_alpha=0.01, normalized_residual_threshold=4.0,
        parameter_ranking_dominance_threshold=1.2, hif_alpha_grid_size=7, hif_r_grid_size=9, hif_max_scans=10,
    )
    env = TransactionalPSSEEnv(**providers.env_kwargs(), production_dataset_mode=True, max_steps=40, history_window=4)
    return env, ExpertPolicyOracle(process_oracle=env.process_oracle, candidate_oracle=env.candidate_quality_oracle)


def _drive(env, expert, scenario, forced, *, max_steps=40):
    """Label every visited state exactly as the DAgger-1 collector does.

    ``forced`` maps a step to a function of the observation returning the
    learner action executed instead of the expert's.  Every expert label must
    pass ``assert_training_decision_evidence``.
    """
    from psse_env.dagger.release_factories import select_observable_expert_actions
    from psse_env.state_store import policy_safe_copy

    env.reset(deepcopy(scenario))
    history, trace = [], []
    for step in range(max_steps):
        if env.is_terminal():
            break
        observation = replace(env.get_policy_observation(history), remaining_budget=max_steps - step).as_dict()
        selection = select_observable_expert_actions(policy_observation=observation, expert_oracle=expert)
        preferred = selection.preferred_action
        assert preferred is not None, (step, trace)
        env.assert_training_decision_evidence(preferred)
        if step in forced:
            action, by = forced[step](observation), "learner"
        else:
            action, by = preferred, "expert"
        _, output = env.step(deepcopy(action))
        history.append({"state_id": observation["active_state_id"], "action": policy_safe_copy(action),
                        "tool_output": policy_safe_copy(output)})
        trace.append((by, action, output.get("execution_status"), output.get("error_code")))
    return trace


def _conditioned_meter(observation):
    for signature in observation["unresolved_signatures"]:
        if "conditioned_meter index=" in signature:
            index = int(signature.split("index=")[1].split()[0])
            return {"tool": CORRECT_MEASUREMENTS, "arguments": {"state_id": observation["active_state_id"],
                                                                "suspect_group": [index]}}
    raise AssertionError(observation["unresolved_signatures"])


def _top_residual_meter(observation):
    for signature in observation["unresolved_signatures"]:
        if "residual" in signature and "index=" in signature:
            index = int(signature.split("index=")[1].split()[0])
            return {"tool": CORRECT_MEASUREMENTS, "arguments": {"state_id": observation["active_state_id"],
                                                                "suspect_group": [index]}}
    raise AssertionError(observation["unresolved_signatures"])


def _request(tool):
    return lambda observation: {"tool": tool, "arguments": {"state_id": observation["active_state_id"]}}


def _true_meters(scenario):
    return {int(row["index"]) for row in scenario.get("true_measurement_errors") or []}


def test_real_a2_conditioned_meter_is_repaired_after_a_refused_copy(real_roots, real_env):
    env, expert = real_env
    scenario = real_roots["measurement+hif"]
    trace = _drive(env, expert, scenario, {5: _conditioned_meter})
    tools = [(by, action["tool"]) for by, action, _, _ in trace]
    assert tools[:5] == [("expert", RUN_WLS), ("expert", GET_THREE_PHASE_CONTEXT),
                         ("expert", RUN_THREE_PHASE_NLM_FROM_PATH),
                         ("expert", ESTIMATE_HIF_MULTISCAN_FROM_PATH), ("expert", RUN_WLS)]
    assert trace[5][0] == "learner" and trace[5][3] == "missing_precondition"
    assert tools[6] == ("expert", GET_MEASUREMENT_CONTEXT)
    repaired = trace[7][1]
    assert repaired["tool"] == CORRECT_MEASUREMENTS and trace[7][2] == "success"
    assert set(repaired["arguments"]["suspect_group"]) <= _true_meters(scenario)
    assert env.terminal_outcome == "operator_escalation"


def test_real_b_and_c_measurement_root_recovers_after_refused_learner_actions(real_roots, real_env):
    env, expert = real_env
    scenario = real_roots["measurement"]
    for forced in ({0: _request(GET_MEASUREMENT_CONTEXT)}, {3: _top_residual_meter}):
        trace = _drive(env, expert, scenario, forced)
        corrections = [action for by, action, status, _ in trace
                       if by == "expert" and action["tool"] == CORRECT_MEASUREMENTS and status == "success"]
        assert corrections, trace
        assert env.terminal_outcome == "operator_escalation", trace


def test_real_m2_measurement_context_retired_by_telemetry_is_requested_again(real_roots, real_env):
    env, expert = real_env
    scenario = real_roots["measurement"]
    on_policy = _drive(env, expert, scenario, {})
    step = next(index for index, (_, action, _, _) in enumerate(on_policy) if action["tool"] == GET_MEASUREMENT_CONTEXT)
    trace = _drive(env, expert, scenario, {step + 1: _request(GET_THREE_PHASE_CONTEXT)})
    assert trace[step + 1][0] == "learner" and trace[step + 1][2] == "success"
    assert trace[step + 2][1]["tool"] == GET_MEASUREMENT_CONTEXT
    assert env.terminal_outcome == "operator_escalation"


def test_real_d_hif_ladder_survives_the_nlm_leaving_the_window(real_roots, real_env):
    env, expert = real_env
    scenario = real_roots["measurement+hif"]
    forced = {3: _request(GET_PARAMETER_CONTEXT), 4: _request(GET_TOPOLOGY_CONTEXT),
              5: _request(GET_PARAMETER_CONTEXT), 6: _request(GET_TOPOLOGY_CONTEXT)}
    trace = _drive(env, expert, scenario, forced)
    assert trace[2][1]["tool"] == RUN_THREE_PHASE_NLM_FROM_PATH
    assert trace[7][0] == "expert" and trace[7][1]["tool"] == ESTIMATE_HIF_MULTISCAN_FROM_PATH
    assert env.terminal_outcome is not None
