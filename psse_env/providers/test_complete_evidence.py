"""The estimator's output, whole (``psse_env.providers.complete_evidence``).

A three-bus triangle: lines 1 (bus 1 to 2), 2 (bus 2 to 3) and 3 (bus 1 to 3).
Measurement order is Vm, Pinj, Qinj per bus, then Pf, Qf, Pt, Qt per line.
Bus 3 has no load, shunt or generator: a zero-injection bus.
"""
from __future__ import annotations

import copy

import numpy as np

from psse_env.dagger.dataset_builder import (
    CANONICAL_DAGGER_SYSTEM_PROMPT,
    COMPLETE_EVIDENCE_PROMPT_PARAGRAPH,
    COMPLETE_EVIDENCE_TABLE_KEYS,
    DETAILED_EVIDENCE_CORRECTION_SENTENCES,
    DETAILED_EVIDENCE_PROMPT_PARAGRAPH,
    RULE_SCREEN_KEYS,
    prepare_model_policy_observation,
    system_prompt_for_observation,
    table_to_records,
    without_rule_screens,
)
from psse_env.evidence_profile import SCADA_ONLY_PROFILE
from psse_env.providers.complete_evidence import (
    COMPLETE_EVIDENCE_CONTRACT,
    TABLE_KEYS,
    breaker_table,
    line_table,
    meter_table,
    signed_residuals,
    solve_tables,
)
from psse_env.providers.detailed_evidence import DETAILED_EVIDENCE_CONTRACT
from psse_env.state_store import PolicyObservation

NB, NL = 3, 3
NZ = 3 * NB + 4 * NL


def _case(status=(1, 1, 1)):
    bus = np.array([[1, 3, 0, 0, 0, 0], [2, 1, 20, 5, 0, 0], [3, 1, 0, 0, 0, 0]], dtype=float)
    branch = np.zeros((NL, 13))
    for row, (f, t) in enumerate(((1, 2), (2, 3), (1, 3))):
        branch[row, 0], branch[row, 1], branch[row, 2], branch[row, 3], branch[row, 10] = f, t, 0.01 * (row + 1), 0.1 * (row + 1), status[row]
    gen = np.zeros((1, 21))
    gen[0, 0], gen[0, 7] = 1, 1
    return {"bus": bus, "branch": branch, "gen": gen}


def _pf(row):
    return 3 * NB + row


def test_the_residual_keeps_the_sign_of_measured_minus_estimated():
    payload = {"r": [2.0, 0.5, 3.0], "raw_residual": [-0.01, 0.002, 0.0]}
    assert signed_residuals(payload) == [-2.0, 0.5, 3.0]
    # Without the raw residual, or with one of another length, the magnitude stands.
    assert signed_residuals({"r": [2.0, 0.5]}) == [2.0, 0.5]
    assert signed_residuals({"r": [2.0, 0.5], "raw_residual": [1.0]}) == [2.0, 0.5]


def test_the_solve_tables_hold_every_bus_and_line_in_the_case_order():
    residual = np.zeros(NZ)
    residual[1] = -0.2                    # Vm at bus 2: far below any threshold, still reported
    residual[_pf(2)] = 14.0               # Pf of line 3
    residual[3 * NB + 3 * NL + 0] = -6.5  # Qt of line 1
    lam = [0.1, -0.2, 0.0, 0.0, -31.0, 4.0]
    tables = solve_tables(residual, lam, _case(status=(1, 1, 0)))
    assert set(tables) == {"bus_table", "branch_table"}
    assert [row["bus"] for row in tables["bus_table"]] == [1, 2, 3]
    assert [row["line"] for row in tables["branch_table"]] == [1, 2, 3]
    assert tables["bus_table"][1] == {"bus": 2, "type": "PQ", "vm": -0.2, "p": 0.0, "q": 0.0}
    # A zero-injection bus has no injection meter to read.
    assert tables["bus_table"][2]["p"] is None and tables["bus_table"][2]["q"] is None
    assert tables["branch_table"][0] == {"line": 1, "from": 1, "to": 2, "pf": 0.0, "qf": 0.0, "pt": 0.0, "qt": -6.5,
                                         "lr": 0.1, "lx": -0.2}
    assert tables["branch_table"][2]["pf"] == 14.0 and tables["branch_table"][2]["lr"] == -31.0
    assert tables["branch_table"][2]["out"] is True and "out" not in tables["branch_table"][0]
    # With the solve's state estimate each bus carries it; one of the wrong length is not attached.
    stated = solve_tables(residual, lam, _case(), vm_estimate=[1.06, 1.0123456, 0.98], angle_estimate_rad=[0.0, -0.1, 0.05])
    assert stated["bus_table"][1] == {"bus": 2, "type": "PQ", "v": 1.0123, "angle": -5.73, "vm": -0.2, "p": 0.0, "q": 0.0}
    assert "v" not in solve_tables(residual, lam, _case(), vm_estimate=[1.0, 1.0], angle_estimate_rad=[0.0, 0.0])["bus_table"][0]


def test_the_meter_table_names_every_measurement_by_the_index_a_correction_takes():
    values = np.linspace(-1.0, 1.0, NZ)
    residual = np.arange(NZ, dtype=float) - 10.0
    rows = meter_table(values, residual, _case())
    assert [row["index"] for row in rows] == list(range(NZ))
    assert rows[0] == {"index": 0, "channel": "Vm", "at": 1, "value": -1.0, "residual": -10.0}
    assert rows[NB + 1]["channel"] == "Pinj" and rows[NB + 1]["at"] == 2
    # Injections at the zero-injection bus 3: the value is shown, the residual is not a meter's.
    assert rows[NB + 2]["residual"] is None and rows[2 * NB + 2]["residual"] is None
    assert rows[2]["residual"] == -8.0   # its voltage meter is a meter like any other
    assert rows[_pf(1)] == {"index": _pf(1), "channel": "Pf", "at": 2, "value": round(float(values[_pf(1)]), 4),
                            "residual": float(_pf(1)) - 10.0}
    assert rows[-1]["channel"] == "Qt" and rows[-1]["at"] == 3


def test_the_line_table_gives_every_line_its_parameters_and_multipliers():
    rows = line_table([0.1, -0.2, 0.0, 0.0, -31.0, 4.0], _case(status=(1, 0, 1)))
    assert rows[0] == {"line": 1, "from": 1, "to": 2, "r": 0.01, "x": 0.1, "lr": 0.1, "lx": -0.2}
    assert rows[1]["out"] is True
    assert rows[2]["lr"] == -31.0 and rows[2]["lx"] == 4.0 and rows[2]["r"] == 0.03


def test_the_breaker_table_lists_every_breaker_and_what_the_estimator_tried():
    ranking = [
        {"cb_name": "CB_2_B", "score": 40.0, "multipliers": {"theta": -40.0, "vm": 3.0}, "reported_status": "closed"},
        {"cb_name": "CB_1_A", "score": 1.5, "multipliers": {"theta": 1.5, "vm": -0.25}, "reported_status": "closed"},
        {"cb_name": "CB_3_C", "score": 9.0, "multipliers": {}, "reported_status": "open"},
    ]
    effects = {"CB_2_B": {"effect": "dangling_line_terminal", "branch_row0": 1, "branch_status": 0},
               "CB_1_A": {"effect": "bus_split", "affected_planning_buses": [1]},
               "CB_3_C": {"effect": "equivalent"}}
    flips = {"CB_2_B": {"success": True, "chi_square": 120.0}, "CB_3_C": {"success": False, "chi_square": float("nan")}}
    rows = breaker_table(ranking, effects, flips, 480.0)
    assert [row["cb"] for row in rows] == ["CB_1_A", "CB_2_B", "CB_3_C"]   # by name, not by score
    assert rows[0] == {"cb": "CB_1_A", "status": "closed", "score": 1.5, "theta": 1.5, "vm": -0.25,
                       "flip_effect": "bus_split", "bus": "1"}
    assert rows[1]["line"] == 2 and rows[1]["flip_converged"] is True
    assert rows[1]["flip_chi_square"] == 120.0 and rows[1]["flip_progress"] == 0.75
    # A breaker whose re-estimation did not converge says so and carries no chi-square.
    assert rows[2]["flip_converged"] is False and "flip_chi_square" not in rows[2] and rows[2]["theta"] is None


def test_the_observation_carries_the_contract_only_when_set():
    plain = PolicyObservation(active_state_id="s0", evidence_profile=SCADA_ONLY_PROFILE).as_dict()
    assert "complete_evidence" not in plain
    complete = PolicyObservation(active_state_id="s0", evidence_profile=SCADA_ONLY_PROFILE,
                                 complete_evidence={"contract": COMPLETE_EVIDENCE_CONTRACT}).as_dict()
    assert complete["complete_evidence"] == {"contract": COMPLETE_EVIDENCE_CONTRACT}
    base = system_prompt_for_observation(CANONICAL_DAGGER_SYSTEM_PROMPT, plain)
    assert system_prompt_for_observation(CANONICAL_DAGGER_SYSTEM_PROMPT, complete) == base + COMPLETE_EVIDENCE_PROMPT_PARAGRAPH
    # With the detailed evidence too, its sentences about the neighbourhood tables give way to the complete ones.
    both = PolicyObservation(active_state_id="s0", evidence_profile=SCADA_ONLY_PROFILE,
                             detailed_evidence={"contract": DETAILED_EVIDENCE_CONTRACT},
                             complete_evidence={"contract": COMPLETE_EVIDENCE_CONTRACT}).as_dict()
    text = system_prompt_for_observation(CANONICAL_DAGGER_SYSTEM_PROMPT, both)
    assert text == base + COMPLETE_EVIDENCE_PROMPT_PARAGRAPH + DETAILED_EVIDENCE_CORRECTION_SENTENCES
    assert "near the alarm" not in text
    assert DETAILED_EVIDENCE_PROMPT_PARAGRAPH.endswith(DETAILED_EVIDENCE_CORRECTION_SENTENCES)


def _observation(complete: bool):
    residual = np.zeros(NZ)
    residual[_pf(2)] = 14.0
    lam = [0.1, -0.2, 0.0, 0.0, -31.0, 4.0]
    tables = solve_tables(residual, lam, _case())
    summary = {"success": True, "top_residuals": [{"index0": _pf(2), "channel": "Pf", "channel_offset": 2, "value": 14.0}],
               "top_lagrange": [{"lambda_index0": 4, "line_row0": 2, "parameter": "R", "value": -31.0}], **tables}
    supported = [{"tool": "correct_parameters", "arguments": {"state_id": "ep:s0", "line_index": 3}}]
    observation = {
        "active_state_id": "ep:s0", "candidate_state_id": None, "episode_id": "ep", "evidence_profile": SCADA_ONLY_PROFILE,
        "unresolved_signatures": ["wls_branch_multiplier_dominant line_status_or_parameter line=3"],
        "remaining_budget": 30, "last_tool": "run_wls", "last_tool_status": "success",
        "last_tool_output": {"execution_status": "success", "state_mutated": False, "active_state_id": "ep:s0",
                             "tool_metrics": {"chi_square_statistic": 300.0, "chi_square_threshold": 20.0,
                                              "unresolved_signatures": ["wls_residual_outlier index=11 channel=Pf"],
                                              "wls_summary": copy.deepcopy(summary)}},
        "fresh_context_evidence": {
            "wls": {"state_id": "ep:s0", "successful": True, "anomalous": True, "anomaly_breadth": 0.1,
                    "flow_at_noise_lines": [], "route_confidence": {"parameter": 0.7, "topology": 0.3},
                    "hypothesis_confidence": {"ranking": ["parameter", "topology", "measurement"]},
                    **(copy.deepcopy(tables) if complete else {})},
            "parameter": {"context_tool": "get_parameter_context", "state_id": "ep:s0", "route_status": "actionable",
                          "route_status_reason": "supported_parameter_candidates", "supported_corrections": supported,
                          "parameter_findings": [{"line_row0": 2, "parameter": "R", "value": -31.0}],
                          "parameter_ranking_distinct_lines": [{"line_index1": 3, "abs_lambda_score": 31.0}],
                          "parameter_ranking_dominant": True,
                          **({"line_table": line_table(lam, _case())} if complete else {})},
        },
        "history_window": [],
    }
    if complete:
        observation["complete_evidence"] = {"contract": COMPLETE_EVIDENCE_CONTRACT}
    return observation


def _keys(value, found=None):
    found = set() if found is None else found
    if isinstance(value, dict):
        for key, item in value.items():
            found.add(key)
            _keys(item, found)
    elif isinstance(value, list):
        for item in value:
            _keys(item, found)
    return found


def test_the_model_view_holds_the_numbers_and_none_of_the_screens_conclusions():
    view, _ = prepare_model_policy_observation(_observation(complete=True))
    keys = _keys(view)
    assert not keys & RULE_SCREEN_KEYS
    wls = view["fresh_context_evidence"]["wls"]
    # Each table once, as its columns and its rows, every component in it.
    assert wls["bus_table"]["columns"] == ["bus", "type", "vm", "p", "q"] and len(wls["bus_table"]["rows"]) == NB
    assert len(wls["branch_table"]["rows"]) == NL
    assert table_to_records(wls["branch_table"])[2]["pf"] == 14.0
    lines = view["fresh_context_evidence"]["parameter"]["line_table"]
    assert table_to_records(lines)[2] == {"line": 3, "from": 1, "to": 3, "r": 0.03, "x": 0.3, "lr": -31.0, "lx": 4.0}
    # The last tool output is the same solve of the active case: it points at the ledger's tables.
    summary = view["last_tool_output"]["observable_metrics"]["wls_summary"]
    assert summary == {"success": True, "tables": {"same_as": "fresh_context_evidence.wls"}}
    # What says an anomaly stands is kept; so is the contract, by which the system prompt describes the tables.
    assert wls["anomalous"] is True and view["complete_evidence"] == {"contract": COMPLETE_EVIDENCE_CONTRACT}


def test_a_candidates_solve_keeps_its_own_tables_in_the_last_tool_output():
    observation = _observation(complete=True)
    other = np.zeros(NZ)
    other[_pf(2)] = 1.5
    observation["last_tool_output"]["tool_metrics"]["wls_summary"].update(solve_tables(other, [0.0] * 6, _case()))
    view, _ = prepare_model_policy_observation(observation)
    summary = view["last_tool_output"]["observable_metrics"]["wls_summary"]
    assert table_to_records(summary["branch_table"])[2]["pf"] == 1.5
    assert table_to_records(view["fresh_context_evidence"]["wls"]["branch_table"])[2]["pf"] == 14.0


def test_without_the_contract_the_model_view_is_as_it_was():
    plain = _observation(complete=False)
    view, _ = prepare_model_policy_observation(plain)
    assert "supported_corrections" in view["fresh_context_evidence"]["parameter"]
    assert view["unresolved_signatures"] == plain["unresolved_signatures"]
    summary = view["last_tool_output"]["observable_metrics"]["wls_summary"]
    assert isinstance(summary["bus_table"], list) and "top_residuals" in summary


def test_the_screens_are_removed_at_any_depth_and_nothing_else_is():
    value = {"keep": 1, "supported_corrections": [1], "inner": [{"route_status": "x", "score": 2.0}], "tables": {"line_table": []}}
    assert without_rule_screens(value) == {"keep": 1, "inner": [{"score": 2.0}], "tables": {"line_table": []}}
    assert COMPLETE_EVIDENCE_TABLE_KEYS == set(TABLE_KEYS) and not COMPLETE_EVIDENCE_TABLE_KEYS & RULE_SCREEN_KEYS


# ---------------------------------------------------------------------------------------------------
# On the IEEE 14 node/breaker model: a real root, the provider with the option and without it.


def _state(scenario: dict) -> dict:
    return {"state_id": "episode:s0", "state_hash": "hash0", "status": "active", "case": scenario["case"],
            "measurements": list(scenario["measurements"]), "metadata": copy.deepcopy(scenario["metadata"])}


def _real_root():
    from psse_env.providers.scenario_generator import Round0ScenarioGenerator

    return Round0ScenarioGenerator(seed=31, topology_effects=("dangling_line_terminal",)).build({"topology": 1})[0]


def test_on_a_real_root_every_component_is_reported_and_nothing_else_changes():
    from psse_env.providers.matpower import MatpowerDeploymentProviders

    scenario = _real_root()
    truth = scenario["true_topology_errors"][0]
    plain = MatpowerDeploymentProviders(chi2_alpha=0.01)
    whole = MatpowerDeploymentProviders(chi2_alpha=0.01, complete_evidence=True)

    solve, plain_solve = whole.run_wls(_state(scenario)), plain.run_wls(_state(scenario))
    summary = solve["wls_summary"]
    assert len(summary["bus_table"]) == 14 and len(summary["branch_table"]) == 20
    # The estimated state of every bus: the reference bus holds the angle's zero, voltages sit near 1 per unit.
    assert summary["bus_table"][0]["angle"] == 0.0
    assert all(0.9 < row["v"] < 1.15 for row in summary["bus_table"])
    assert all(abs(row["v"] - meter) < 0.02 for row, meter in zip(summary["bus_table"], scenario["measurements"][:14]))
    assert [row["line"] for row in summary["branch_table"]] == list(range(1, 21))
    assert "omitted" not in summary and "bus_table" not in plain_solve["wls_summary"]
    # A solve reports residuals of both signs, and the largest magnitude is the solve's own.
    flows = [row[key] for row in summary["branch_table"] for key in ("pf", "qf", "pt", "qt")]
    cells = flows + [row[key] for row in summary["bus_table"] for key in ("vm", "p", "q") if row[key] is not None]
    assert min(cells) < 0.0 < max(cells)
    assert abs(max(abs(cell) for cell in cells) - solve["max_normalized_residual"]) < 0.01
    assert {key: value for key, value in solve.items() if key != "wls_summary"} == \
        {key: value for key, value in plain_solve.items() if key != "wls_summary"}

    meters = whole.get_measurement_context(_state(scenario))
    assert [row["index"] for row in meters["meter_table"]] == list(range(122))
    assert [row["value"] for row in meters["meter_table"][:14]] == [round(v, 4) for v in scenario["measurements"][:14]]
    lines = whole.get_parameter_context(_state(scenario))
    assert len(lines["line_table"]) == 20
    # The parameter context lists at most five entries; the table holds all forty multipliers.
    assert len(lines["parameter_findings"]) <= 5

    topology = whole.get_topology_context(_state(scenario))
    plain_topology = plain.get_topology_context(_state(scenario))
    table = {row["cb"]: row for row in topology["breaker_table"]}
    assert len(table) == 73 and list(table) == sorted(table)
    tested = [row for row in table.values() if "flip_converged" in row]
    assert len(tested) == len(topology["breaker_findings"]) == whole.breaker_candidate_count
    faulted = table[truth["cb_name"]]
    assert faulted["flip_effect"] == "dangling_line_terminal" and faulted["line"] == int(truth["branch_row0"]) + 1
    assert faulted["flip_progress"] > 0.5
    for name, context, other in (("measurement", meters, plain.get_measurement_context(_state(scenario))),
                                 ("parameter", lines, plain.get_parameter_context(_state(scenario))),
                                 ("topology", topology, plain_topology)):
        added = set(context) - set(other)
        assert added == {{"measurement": "meter_table", "parameter": "line_table", "topology": "breaker_table"}[name]}
        assert {key: context[key] for key in other} == other


def test_a_candidates_solve_that_changes_one_table_keeps_both():
    observation = _observation(complete=True)
    other = np.zeros(NZ)
    other[_pf(2)] = 1.5   # the flows change, the bus table does not
    observation["last_tool_output"]["tool_metrics"]["wls_summary"].update(solve_tables(other, [0.0] * 6, _case()))
    view, _ = prepare_model_policy_observation(observation)
    summary = view["last_tool_output"]["observable_metrics"]["wls_summary"]
    assert "tables" not in summary and "bus_table" in summary and "branch_table" in summary


def test_the_topology_schema_points_at_the_breaker_table_only_under_the_contract():
    from psse_env.dagger.dataset_builder import COMPLETE_EVIDENCE_TOPOLOGY_LINE_DESCRIPTION, tool_schemas_for_observation
    from psse_env.dagger.protocol_bridge import unified_tool_schemas

    registry = unified_tool_schemas()

    def line_description(observation):
        schemas = tool_schemas_for_observation(registry, observation)
        topology = next(tool["function"] for tool in schemas if tool["function"]["name"] == "correct_topology_from_path")
        return topology["parameters"]["properties"]["line_index1"]["description"]

    plain = PolicyObservation(active_state_id="s0", evidence_profile=SCADA_ONLY_PROFILE).as_dict()
    complete = PolicyObservation(active_state_id="s0", evidence_profile=SCADA_ONLY_PROFILE,
                                 complete_evidence={"contract": COMPLETE_EVIDENCE_CONTRACT}).as_dict()
    assert "supported correction" in line_description(plain)
    assert line_description(complete) == COMPLETE_EVIDENCE_TOPOLOGY_LINE_DESCRIPTION
    # The shared registry is not edited in place.
    assert "supported correction" in line_description(plain)


def test_the_open_rules_paragraph_does_not_speak_of_a_list_the_model_no_longer_reads():
    from psse_env.actions import OPEN_RULES, open_rules_record
    from psse_env.dagger.dataset_builder import open_rules_prompt_paragraph

    plain = open_rules_prompt_paragraph(OPEN_RULES)
    complete = open_rules_prompt_paragraph(OPEN_RULES, complete=True)
    assert "does not list" in plain and "not actionable" in plain and "every listed correction" in plain
    assert "does not list" not in complete and "not actionable" not in complete and "listed" not in complete
    assert "A verified candidate may be committed whatever its verification shows" in complete
    assert "may be sent at any step." in complete
    # The system prompt of an observation picks the form by its contract.
    record = open_rules_record(OPEN_RULES)
    opened = PolicyObservation(active_state_id="s0", evidence_profile=SCADA_ONLY_PROFILE, open_rules=record).as_dict()
    both = PolicyObservation(active_state_id="s0", evidence_profile=SCADA_ONLY_PROFILE, open_rules=record,
                             complete_evidence={"contract": COMPLETE_EVIDENCE_CONTRACT}).as_dict()
    assert system_prompt_for_observation(CANONICAL_DAGGER_SYSTEM_PROMPT, opened).endswith(plain)
    assert system_prompt_for_observation(CANONICAL_DAGGER_SYSTEM_PROMPT, both).endswith(complete)
