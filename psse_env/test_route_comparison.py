"""Item 7 step 7 (the user's request of 2026-10-28): compare the impedance edit and the breaker flip before committing.

A toy network of six lines whose line 6 has an open terminal: opening its
breaker clears the alarm, an impedance edit on it only lowers the objective.
``compare_routes`` fits both on isolated clones and ranks the verdicts, and
changes nothing on the live state.
"""
from __future__ import annotations

import unittest

from psse_env.actions import (
    COMPARE_ROUTES,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
    GET_MEASUREMENT_CONTEXT,
    GET_PARAMETER_CONTEXT,
    GET_TOPOLOGY_CONTEXT,
    ROUTE_COMPARISON_FAMILY,
    RUN_WLS,
    route_comparison_best_family,
)
from psse_env.evidence_profile import SCADA_ONLY_PROFILE
from psse_env.oracle.candidate_quality import CandidateAssessment, CandidateDisposition
from psse_env.oracle.process_validity import ProcessValidityOracle
from psse_env.state_store import PolicyObservation
from psse_env.transactional_env import TransactionalPSSEEnv

ALARM = 10.0


def _wls(state):
    rows = (state.get("case") or {}).get("branch") or []
    opened = int(rows[5].get("br_status", 1)) == 0
    refit = float(rows[5].get("x", 1.0)) != 1.0
    objective = 0.0 if opened else 30.0 if refit else 40.0
    alarm = objective >= ALARM
    return {"wls_objective": objective, "remaining_anomaly_score": objective / ALARM, "chi_square_statistic": objective,
            "chi_square_threshold": ALARM, "chi_square_alarm": alarm, "converged": True, "globally_resolved": not alarm,
            "no_material_anomaly_remaining": not alarm,
            "unresolved_signatures": ["wls_branch_multiplier_dominant line_status_or_parameter line=6"] if alarm else []}


def _actionable(state, tool, corrections):
    return {"state_id": state["state_id"], "state_hash": state["state_hash"], "route_status": "actionable",
            "supported_corrections": [{"tool": tool, "arguments": {"state_id": state["state_id"], **c}} for c in corrections]}


def _parameter_context(state):
    return _actionable(state, CORRECT_PARAMETERS, [{"line_index": 6}, {"line_index": 3}])


def _topology_context(state):
    return _actionable(state, CORRECT_TOPOLOGY, [{"cb_name": "CB_6", "status": 0, "line_index": 6}])


def _empty(state):
    return {"state_id": state["state_id"], "state_hash": state["state_hash"], "supported_corrections": [], "route_status": "complete_negative"}


def _correct_parameters(state, action):
    del state
    return {"modification": {"line_index1": int(action["arguments"]["line_index"]), "parameter": "x", "value": 0.5}}


def _correct_topology(state, action):
    del state
    return {"modification": {"line_index1": int(action["arguments"]["line_index"]), "status": int(action["arguments"]["status"]),
                             "status_field": "br_status"}}


class _Quality:
    def label_candidate(self, *, parent_state, source_action, candidate_state, verification_output, hidden_truth):
        del parent_state, source_action, candidate_state, hidden_truth
        if not verification_output.get("chi_square_alarm"):
            disposition = CandidateDisposition.ACCEPT_FINAL
        elif float(verification_output.get("global_progress") or 0.0) > 0.0:
            disposition = CandidateDisposition.ACCEPT_PARTIAL
        else:
            disposition = CandidateDisposition.REJECT
        return CandidateAssessment(disposition=disposition, progress_class="toy")


def _env(enabled: bool = True) -> TransactionalPSSEEnv:
    return TransactionalPSSEEnv(
        evidence_profile=SCADA_ONLY_PROFILE,
        process_oracle=ProcessValidityOracle(executor_hydrated_corrections=True),
        wls_runner=_wls,
        candidate_quality_oracle=_Quality(),
        context_providers={
            GET_MEASUREMENT_CONTEXT: _empty,
            GET_PARAMETER_CONTEXT: _parameter_context,
            GET_TOPOLOGY_CONTEXT: _topology_context,
        },
        correction_executors={CORRECT_PARAMETERS: _correct_parameters, CORRECT_TOPOLOGY: _correct_topology},
        max_steps=30,
        route_comparison=enabled,
    )


SCENARIO = {"scenario_id": "routes-toy", "case": {"branch": [{"x": 1.0, "br_status": 1} for _ in range(6)]}, "measurements": [1.0]}


def _step(env, tool, **arguments):
    _, output = env.step({"tool": tool, "arguments": arguments})
    return output


def _prepared(env):
    root = env.reset(SCENARIO)["active_state_id"]
    for tool in (RUN_WLS, GET_TOPOLOGY_CONTEXT):
        assert _step(env, tool, state_id=root)["execution_status"] == "success"
    return root


class RouteComparisonContractTests(unittest.TestCase):
    def test_without_the_option_the_tool_is_unknown_and_the_observation_lists_no_research_tools(self) -> None:
        env = _env(enabled=False)
        root = _prepared(env)
        _step(env, GET_PARAMETER_CONTEXT, state_id=root)
        self.assertNotIn("research_tools", env.get_policy_observation().as_dict())
        self.assertNotIn("research_tools", PolicyObservation(active_state_id="s0").as_dict())
        output = _step(env, COMPARE_ROUTES, state_id=root, parameter_line=6, topology_line=6)
        self.assertEqual(output["error_code"], "unknown_tool")

    def test_both_routes_are_fitted_on_clones_and_ranked_without_touching_the_state(self) -> None:
        env = _env()
        root = _prepared(env)
        self.assertEqual(env.get_policy_observation().as_dict()["research_tools"], [COMPARE_ROUTES])
        # Both branch contexts are required.
        self.assertEqual(_step(env, COMPARE_ROUTES, state_id=root, parameter_line=6, topology_line=6)["error_detail"],
                         "route_comparison_needs_both_branch_contexts")
        _step(env, GET_PARAMETER_CONTEXT, state_id=root)
        self.assertEqual(_step(env, COMPARE_ROUTES, state_id=root, parameter_line=6, topology_line=4)["error_detail"],
                         "route_comparison_target_not_supported")
        hash_before = env.store.state_hash(root)
        output = _step(env, COMPARE_ROUTES, state_id=root, parameter_line=6, topology_line=6)
        self.assertEqual(output["execution_status"], "success")
        self.assertFalse(output["state_mutated"])
        self.assertEqual((env.store.active_state_id, env.current_candidate_id, env.store.state_hash(root)), (root, None, hash_before))
        metrics = output["tool_metrics"]
        self.assertEqual([c["route"] for c in metrics["candidates"]], ["topology", "parameter"])
        self.assertEqual([c["verification_verdict"] for c in metrics["candidates"]], ["ACCEPT_FINAL", "ACCEPT_PARTIAL"])
        self.assertEqual((metrics["best_route"], metrics["internal_correction_calls"], metrics["internal_wls_calls"]), ("topology", 2, 2))
        observation = env.get_policy_observation().as_dict()
        self.assertEqual(observation["fresh_context_evidence"][ROUTE_COMPARISON_FAMILY]["best_route"], "topology")
        self.assertEqual(route_comparison_best_family(observation), "topology")
        # Once per pair of lines on a state.
        self.assertEqual(_step(env, COMPARE_ROUTES, state_id=root, parameter_line=6, topology_line=6)["error_detail"],
                         "route_comparison_already_recorded")

    def test_an_open_candidate_blocks_the_comparison(self) -> None:
        env = _env()
        root = _prepared(env)
        _step(env, GET_PARAMETER_CONTEXT, state_id=root)
        _step(env, CORRECT_PARAMETERS, state_id=root, line_index=3)
        self.assertEqual(_step(env, COMPARE_ROUTES, state_id=root, parameter_line=6, topology_line=6)["error_detail"],
                         "route_comparison_with_open_candidate")


if __name__ == "__main__":
    unittest.main()
