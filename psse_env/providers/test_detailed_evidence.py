"""The detailed evidence of a correction and its verification (``psse_env.providers.detailed_evidence``).

A three-bus triangle: lines 1 (bus 1 to 2), 2 (bus 2 to 3) and 3 (bus 1 to 3).
Measurement order is Vm, Pinj, Qinj per bus, then Pf, Qf, Pt, Qt per line.
"""
from __future__ import annotations

import numpy as np

from psse_env.dagger.dataset_builder import (
    CANONICAL_DAGGER_SYSTEM_PROMPT,
    DETAILED_EVIDENCE_PROMPT_PARAGRAPH,
    HISTORY_METRIC_KEYS,
    prepare_model_policy_observation,
    system_prompt_for_observation,
)
from psse_env.evidence_profile import SCADA_ONLY_PROFILE
from psse_env.providers.detailed_evidence import (
    DETAILED_EVIDENCE_CONTRACT,
    locate,
    measurement_edit,
    parameter_edit,
    solve_view,
    target_neighbourhood,
    topology_edit,
)
from psse_env.state_store import PolicyObservation

NB, NL = 3, 3


def _case(status=(1, 1, 1)):
    bus = np.array([[1, 3, 0, 0, 0, 0], [2, 1, 0, 0, 0, 0], [3, 1, 0, 0, 0, 0]], dtype=float)
    branch = np.zeros((NL, 13))
    for row, (f, t) in enumerate(((1, 2), (2, 3), (1, 3))):
        branch[row, 0], branch[row, 1], branch[row, 2], branch[row, 3], branch[row, 10] = f, t, 0.01, 0.1, status[row]
    return {"bus": bus, "branch": branch}


def _view(residual=None, lam=None, status=(1, 1, 1)):
    r = np.zeros(3 * NB + 4 * NL) if residual is None else np.asarray(residual, dtype=float)
    multipliers = np.zeros(2 * NL) if lam is None else np.asarray(lam, dtype=float)
    return solve_view({"r": r.tolist(), "lambdaN": multipliers.tolist()}, _case(status))


def _pf(row):
    return 3 * NB + row


def test_locate_reads_the_channel_and_its_offset():
    assert locate(0, NB, NL) == ("Vm", 0)
    assert locate(NB + 2, NB, NL) == ("Pinj", 2)
    assert locate(_pf(1), NB, NL) == ("Pf", 1)
    assert locate(3 * NB + 3 * NL + 2, NB, NL) == ("Qt", 2)


def test_solve_view_refuses_a_vector_of_the_wrong_length():
    assert solve_view({"r": [0.0] * 5, "lambdaN": [0.0] * 6}, _case()) is None


def test_a_parameter_target_reads_its_own_multiplier_and_what_stands_next_to_it():
    before_r = np.zeros(3 * NB + 4 * NL)
    before_r[_pf(0)] = -9.0     # a flow on line 1, which meets line 2 at bus 2
    before_r[NB + 2] = 4.0      # P injection at bus 3, an end of line 2
    before = _view(before_r, [0, 1.0, 12.0, -30.0, 0, 5.0])   # line 2 carries the largest multiplier
    after_r = np.zeros(3 * NB + 4 * NL)
    after_r[_pf(0)] = 7.5
    after = _view(after_r, [0, 6.0, 0.5, -1.0, 0, 2.0])
    block = target_neighbourhood({"tool": "correct_parameters", "arguments": {"line_index": 2}}, before, after)
    assert block["target"] == {"kind": "parameter", "lines": [2], "buses": [2, 3]}
    assert block["adjacent_lines"] == [1, 3]
    assert block["own_metric"] == {"after": 1.0, "before": 30.0}
    assert block["adjacent_residual_max"] == {"after": 7.5, "before": 9.0}
    assert block["adjacent_multiplier_max"] == {"after": 6.0, "before": 5.0}


def test_a_meter_target_leaves_its_own_channel_out_of_the_neighbourhood():
    residual = np.zeros(3 * NB + 4 * NL)
    residual[_pf(0)] = 20.0                 # the corrected meter itself
    residual[3 * NB + 2 * NL + 0] = 6.0     # Pt of the same line
    before = _view(residual, [0, 8.0, 0, 3.0, 0, 0])
    after_r = residual.copy()
    after_r[_pf(0)] = 0.0
    after = _view(after_r, [0, 8.0, 0, 3.0, 0, 0])
    block = target_neighbourhood({"tool": "correct_measurements", "arguments": {"suspect_group": [_pf(0)]}}, before, after)
    assert block["target"]["kind"] == "meter" and block["target"]["buses"] == [1, 2]
    assert block["own_metric"] == {"after": 0.0, "before": 20.0}
    # The line the meter sits on stays in the neighbourhood: its multiplier is what a wrong meter edit leaves behind.
    assert block["adjacent_multiplier_max"] == {"after": 8.0, "before": 8.0}
    assert block["adjacent_residual_max"] == {"after": 6.0, "before": 6.0}


def test_an_opened_line_reads_its_own_flows_and_drops_out_of_the_adjacent_lines():
    before_r = np.zeros(3 * NB + 4 * NL)
    before_r[_pf(2)] = 15.0
    before = _view(before_r, [0, 0, 0, 0, 0, 40.0])
    after_r = np.zeros(3 * NB + 4 * NL)
    after_r[_pf(2)] = 1.2
    after = _view(after_r, None, status=(1, 1, 0))
    block = target_neighbourhood({"tool": "correct_topology", "arguments": {"cb_name": "CB_X", "line_index": 3, "status": 0}}, before, after)
    assert block["target"]["kind"] == "open_line"
    assert block["own_metric"] == {"after": 1.2, "before": 40.0}
    assert block["adjacent_lines"] == [1, 2]


def test_without_the_parent_solve_only_the_after_numbers_are_reported():
    block = target_neighbourhood({"tool": "correct_parameters", "arguments": {"line_index": 1}}, None, _view(lam=[0, 2.0, 0, 0, 0, 0]))
    assert block["own_metric"] == {"after": 2.0}
    assert "before" not in block["adjacent_residual_max"]
    assert target_neighbourhood({"tool": "run_wls", "arguments": {}}, None, _view()) is None


def test_edits_report_old_and_new_values():
    edit = parameter_edit(7, [0.02, 0.2], [0.005, 0.05])
    assert edit == {"line": 7, "r": {"before": 0.02, "after": 0.005, "ratio": 0.25}, "x": {"before": 0.2, "after": 0.05, "ratio": 0.25}}
    previous = [0.0] * (3 * NB + 4 * NL)
    previous[_pf(1)] = 0.50
    meters = measurement_edit({_pf(1): 0.35}, previous, [0.01] * len(previous), NB, NL)
    assert meters == [{"index0": _pf(1), "channel": "Pf", "channel_offset": 1, "before": 0.5, "after": 0.35, "change_sigma": -15.0}]
    assert topology_edit({"cb_name": "CB_1", "cb_previous_status": "closed", "cb_new_status": "open",
                          "breaker_effect": "dangling_line_terminal", "line_index": 4, "previous_status": 1, "new_status": 0}) == {
        "breaker": "CB_1", "breaker_status": {"before": "closed", "after": "open"}, "effect": "dangling_line_terminal",
        "line": 4, "line_status": {"before": 1, "after": 0}}
    assert topology_edit({}) is None


def test_the_observation_carries_the_records_only_when_set():
    plain = PolicyObservation(active_state_id="s0", evidence_profile=SCADA_ONLY_PROFILE).as_dict()
    assert "detailed_evidence" not in plain and "lifecycle_choice" not in plain
    rich = PolicyObservation(active_state_id="s0", evidence_profile=SCADA_ONLY_PROFILE,
                             detailed_evidence={"contract": DETAILED_EVIDENCE_CONTRACT}).as_dict()
    assert rich["detailed_evidence"] == {"contract": DETAILED_EVIDENCE_CONTRACT}
    assert system_prompt_for_observation(CANONICAL_DAGGER_SYSTEM_PROMPT, rich) == \
        system_prompt_for_observation(CANONICAL_DAGGER_SYSTEM_PROMPT, plain) + DETAILED_EVIDENCE_PROMPT_PARAGRAPH


def test_the_prompt_admits_the_evidence_under_one_key():
    assert "detailed_evidence" in HISTORY_METRIC_KEYS
    block = {"contract": DETAILED_EVIDENCE_CONTRACT,
             "target_neighbourhood": {"target": {"kind": "parameter", "lines": [2], "buses": [2, 3]}, "adjacent_lines": [1, 3],
                                      "own_metric": {"after": 1.0, "before": 30.0},
                                      "adjacent_residual_max": {"after": 7.5, "before": 9.0},
                                      "adjacent_multiplier_max": {"after": 6.0, "before": 5.0}}}
    observation = {
        "active_state_id": "e:s0", "candidate_state_id": "e:s1", "evidence_profile": SCADA_ONLY_PROFILE, "last_tool": "run_wls",
        "last_tool_output": {"execution_status": "success", "tool_metrics": {"detailed_evidence": block, "corrected_r": 0.1}},
        "history_window": [],
    }
    prepared, _meta = prepare_model_policy_observation(observation)
    metrics = prepared["last_tool_output"]["observable_metrics"]
    assert metrics["detailed_evidence"] == block       # whole, not cut by the depth bound
    assert "corrected_r" not in metrics                 # keys outside the allow-list stay out, as before
