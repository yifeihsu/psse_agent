"""The bounded candidate comparison for ambiguous parameter rankings (2026-10-19).

``compare_parameter_candidates`` fits two or three supported parameter lines on
isolated branches of the active state, runs each candidate's verification solve
there, and reports the observable verdicts; the expert variant
``ledger_ranked_compare`` proposes it when the ranking does not observably
dominate and then orders the corrections by the verdicts.
"""

from __future__ import annotations

import copy
import unittest

from psse_env.actions import (
    COMMIT_STATE,
    COMPARE_PARAMETER_CANDIDATES,
    DEFAULT_COMPARISON_DOMINANCE_BAND,
    CORRECT_MEASUREMENTS,
    CORRECT_PARAMETERS,
    CORRECT_TOPOLOGY,
    GET_MEASUREMENT_CONTEXT,
    GET_PARAMETER_CONTEXT,
    GET_TOPOLOGY_CONTEXT,
    PARAMETER_COMPARISON_FAMILY,
    RUN_WLS,
    comparison_line_indices,
    parameter_comparison_ranked_lines,
    rank_parameter_comparison,
)
from psse_env.dagger.protocol_bridge import (
    canonical_to_internal_action,
    internal_to_canonical_action,
    unified_tool_schemas,
)
from psse_env.dagger.release_factories import ObservableExpertPolicy
from psse_env.evidence_profile import AUXILIARY_EVIDENCE_PROFILE, SCADA_ONLY_PROFILE
from psse_env.oracle import ExpertPolicyOracle, ProcessValidityOracle
from psse_env.oracle.expert_variants import expert_variant_options
from psse_env.state_store import find_forbidden_policy_paths
from psse_env.transactional_env import TransactionalPSSEEnv

ANOMALY_THRESHOLD = 10.0
TRUE_LINE = 6
WRONG_LINE = 3


def _deterministic(function):
    function.provider_kind = "deterministic"
    return function


def _reactances(state) -> dict[str, float]:
    rows = (state.get("case") or {}).get("branch") or []
    return {str(index + 1): float(row.get("x", 1.0)) for index, row in enumerate(rows)}


def _objective(x: dict[str, float]) -> float:
    """A two-line toy fit: the true line clears it, the wrong line only lowers it."""
    x3, x6 = x.get("3", 1.0), x.get("6", 1.0)
    return 100.0 * (x3 * x6 - 0.5) ** 2 + 60.0 * (x6 - 0.5) ** 2


def _corrected_line(state) -> int | None:
    """The refitted row, read from the case itself (strict profiles strip provider metadata)."""
    for line, value in _reactances(state).items():
        if value != 1.0:
            return int(line)
    return None


@_deterministic
def _wls(state):
    x = _reactances(state)
    objective = _objective(x)
    resolved = objective < ANOMALY_THRESHOLD
    line = _corrected_line(state) if state.get("parent_state_id") else None
    metrics = {
        "wls_objective": objective,
        "remaining_anomaly_score": objective,
        "anomaly_threshold": ANOMALY_THRESHOLD,
        "chi_square_statistic": objective,
        "chi_square_threshold": ANOMALY_THRESHOLD,
        "chi_square_alarm": not resolved,
        "remaining_suspect_count": 0 if resolved else 1,
        "globally_resolved": resolved,
        "physical_constraints_ok": True,
        "new_constraint_violations": 0,
        "unresolved_signatures": [] if resolved else ["wls_branch_multiplier_dominant line=3"],
        "converged": True,
    }
    if line is not None:
        multiplier = 0.5 if line == TRUE_LINE else 4.0
        metrics.update({
            "target_metric_kind": "branch_multiplier",
            "target_metric_value": multiplier,
            "target_metric_threshold": 3.0,
            "target_fixed": multiplier < 3.0,
            "target_progress": 0.9 if line == TRUE_LINE else 0.1,
        })
    return metrics


def _parameter_context_factory(*, dominant: bool, ratio: float):
    @_deterministic
    def context(state):
        state_id = state["state_id"]
        return {
            "state_id": state_id,
            "state_hash": state["state_hash"],
            "unresolved_signatures": ["wls_branch_multiplier_dominant line=3"],
            "parameter_findings": [{"line_index1": WRONG_LINE}, {"line_index1": TRUE_LINE}],
            "parameter_ranking_distinct_lines": [
                {"line_index1": WRONG_LINE, "abs_lambda_score": 13.9},
                {"line_index1": TRUE_LINE, "abs_lambda_score": 13.9 / ratio},
            ],
            "parameter_ranking_top_abs_lambda": 13.9,
            "parameter_ranking_runner_up_abs_lambda": 13.9 / ratio,
            "parameter_ranking_dominance_ratio": ratio,
            "parameter_ranking_dominance_threshold": 1.2,
            "parameter_ranking_dominant": dominant,
            "route_status": "actionable",
            "route_status_reason": "supported_parameter_candidates" if dominant else "parameter_ranking_ambiguous_top_candidates",
            "supported_corrections": [
                {"tool": CORRECT_PARAMETERS, "arguments": {"state_id": state_id, "line_index": line}}
                for line in (WRONG_LINE, TRUE_LINE)
            ],
        }

    return context


@_deterministic
def _empty_context(state):
    return {
        "state_id": state["state_id"], "state_hash": state["state_hash"],
        "supported_corrections": [], "route_status": "complete_negative",
    }


@_deterministic
def _correct_parameters(state, action):
    del state
    line = int(action["arguments"]["line_index"])
    return {"modification": {"line_index": line, "parameter": "x", "value": 0.5}, "executor_receipt": "toy_refit"}


@_deterministic
def _correct_other(state, action):
    del state
    return {"modification": {key: value for key, value in action["arguments"].items() if key != "state_id"}}


def _env(*, profile: str = AUXILIARY_EVIDENCE_PROFILE, dominant: bool = False, ratio: float = 1.18) -> TransactionalPSSEEnv:
    adapters = {RUN_WLS, GET_MEASUREMENT_CONTEXT, GET_PARAMETER_CONTEXT, GET_TOPOLOGY_CONTEXT,
                CORRECT_MEASUREMENTS, CORRECT_PARAMETERS, CORRECT_TOPOLOGY}
    return TransactionalPSSEEnv(
        evidence_profile=profile,
        production_dataset_mode=True,
        # The deployment providers configure this gate: corrections name a
        # target and the executor computes the values.
        process_oracle=ProcessValidityOracle(executor_hydrated_corrections=True),
        approved_deterministic_providers=adapters,
        wls_runner=_wls,
        context_providers={
            GET_MEASUREMENT_CONTEXT: _empty_context,
            GET_PARAMETER_CONTEXT: _parameter_context_factory(dominant=dominant, ratio=ratio),
            GET_TOPOLOGY_CONTEXT: _empty_context,
        },
        correction_executors={
            CORRECT_MEASUREMENTS: _correct_other,
            CORRECT_PARAMETERS: _correct_parameters,
            CORRECT_TOPOLOGY: _correct_other,
        },
        max_steps=12,
    )


def _scenario() -> dict:
    return {
        "scenario_id": "two-line-ambiguity",
        "case": {"branch": [{"x": 1.0} for _ in range(6)]},
        "measurements": [1.0, 1.0],
    }


def _prepared(env: TransactionalPSSEEnv) -> str:
    state = env.reset(_scenario())
    active = state["active_state_id"]
    _, wls = env.step({"tool": RUN_WLS, "arguments": {"state_id": active}})
    assert wls["execution_status"] == "success", wls
    _, context = env.step({"tool": GET_PARAMETER_CONTEXT, "arguments": {"state_id": active}})
    assert context["execution_status"] == "success", context
    return active


def _compare(active: str, lines) -> dict:
    return {"tool": COMPARE_PARAMETER_CANDIDATES, "arguments": {"state_id": active, "line_indices": list(lines)}}


class PureHelperTests(unittest.TestCase):
    def test_line_indices_are_bounded_distinct_positive_integers(self) -> None:
        self.assertEqual(comparison_line_indices([3, 6]), [3, 6])
        self.assertEqual(comparison_line_indices((6, 3, 7)), [6, 3, 7])
        for malformed in ([3], [3, 6, 7, 8], [3, 3], [0, 6], [True, 6], ["3", 6], None, "3,6"):
            self.assertIsNone(comparison_line_indices(malformed), malformed)

    def test_verdicts_rank_accepted_first_then_progress_then_statistic_then_request_order(self) -> None:
        ranked = rank_parameter_comparison([
            {"line_index": 3, "request_rank": 0, "execution_status": "success", "verification_verdict": "ACCEPT_PARTIAL", "global_progress": 0.6, "chi_square_statistic": 15.0},
            {"line_index": 6, "request_rank": 1, "execution_status": "success", "verification_verdict": "ACCEPT_FINAL", "global_progress": 1.0, "chi_square_statistic": 0.0},
            {"line_index": 7, "request_rank": 2, "execution_status": "success", "verification_verdict": "REJECT", "global_progress": -0.2, "chi_square_statistic": 50.0},
            {"line_index": 9, "request_rank": 3, "execution_status": "failure", "error_code": "parameter_correction_failure", "verification_verdict": None},
        ])
        self.assertEqual([item["line_index"] for item in ranked], [6, 3, 7, 9])
        self.assertEqual([item["comparison_rank"] for item in ranked], [1, 2, 3, 4])
        # Within one class the larger progress wins, then the smaller statistic, then the request order.
        partials = rank_parameter_comparison([
            {"line_index": 1, "request_rank": 0, "execution_status": "success", "verification_verdict": "ACCEPT_PARTIAL", "global_progress": 0.4, "chi_square_statistic": 20.0},
            {"line_index": 2, "request_rank": 1, "execution_status": "success", "verification_verdict": "ACCEPT_PARTIAL", "global_progress": 0.7, "chi_square_statistic": 12.0},
            {"line_index": 4, "request_rank": 2, "execution_status": "success", "verification_verdict": "ACCEPT_PARTIAL", "global_progress": 0.7, "chi_square_statistic": 11.0},
            {"line_index": 5, "request_rank": 3, "execution_status": "success", "verification_verdict": "ACCEPT_PARTIAL", "global_progress": 0.7, "chi_square_statistic": 11.0},
        ])
        self.assertEqual([item["line_index"] for item in partials], [4, 5, 2, 1])

    def test_ranked_lines_need_a_comparison_bound_to_the_active_state(self) -> None:
        record = {"state_id": "s0", "candidates": [
            {"line_index": 3, "request_rank": 0, "execution_status": "success", "verification_verdict": "REJECT"},
            {"line_index": 6, "request_rank": 1, "execution_status": "success", "verification_verdict": "ACCEPT_FINAL"},
        ]}
        observation = {"active_state_id": "s0", "fresh_context_evidence": {PARAMETER_COMPARISON_FAMILY: record}}
        self.assertEqual(parameter_comparison_ranked_lines(observation), [6, 3])
        self.assertEqual(parameter_comparison_ranked_lines({**observation, "active_state_id": "s1"}), [])
        self.assertEqual(parameter_comparison_ranked_lines({"active_state_id": "s0", "fresh_context_evidence": {}}), [])


class ProtocolSurfaceTests(unittest.TestCase):
    def test_the_tool_is_on_the_canonical_surface_and_bridges_both_ways(self) -> None:
        schemas = {row["function"]["name"]: row["function"]["parameters"] for row in unified_tool_schemas()}
        self.assertIn(COMPARE_PARAMETER_CANDIDATES, schemas)
        self.assertEqual(sorted(schemas[COMPARE_PARAMETER_CANDIDATES]["properties"]), ["case_path", "line_indices"])
        self.assertEqual(schemas[COMPARE_PARAMETER_CANDIDATES]["properties"]["line_indices"]["type"], "array")
        internal = {"tool": COMPARE_PARAMETER_CANDIDATES, "arguments": {"state_id": "s0", "line_indices": [3, 6]}}
        canonical = internal_to_canonical_action(internal)
        self.assertEqual(canonical, {"tool": COMPARE_PARAMETER_CANDIDATES, "arguments": {"case_path": "s0", "line_indices": [3, 6]}})
        self.assertEqual(canonical_to_internal_action(canonical), internal)

    def test_the_tool_stays_visible_under_the_scada_only_profile(self) -> None:
        from psse_env.dagger.dataset_builder import tool_schemas_for_observation

        visible = {row["function"]["name"] for row in
                   tool_schemas_for_observation(unified_tool_schemas(), {"evidence_profile": SCADA_ONLY_PROFILE})}
        self.assertIn(COMPARE_PARAMETER_CANDIDATES, visible)

    def test_the_rank_one_proof_accepts_the_comparison_best_line_as_the_target(self) -> None:
        from psse_env.dagger.rollout_collector import observable_rank_one_target_proof

        best = {"tool": CORRECT_PARAMETERS, "arguments": {"state_id": "s0", "line_index": 6}}
        other = {"tool": CORRECT_PARAMETERS, "arguments": {"state_id": "s0", "line_index": 3}}
        observation = {
            "active_state_id": "s0", "has_fresh_parameter_context": True, "parameter_context_state_id": "s0",
            "fresh_context_evidence": {PARAMETER_COMPARISON_FAMILY: {"state_id": "s0", "candidates": [
                {"line_index": 3, "request_rank": 0, "execution_status": "success", "verification_verdict": "ACCEPT_PARTIAL", "global_progress": 0.3},
                {"line_index": 6, "request_rank": 1, "execution_status": "success", "verification_verdict": "ACCEPT_FINAL", "global_progress": 0.9},
            ]}},
        }
        proof = observable_rank_one_target_proof(observation, preferred_action=best, expert_actions=[best, other])
        self.assertTrue(proof["passed"], proof)
        self.assertEqual(proof["basis"], "branch_verified_comparison_best")
        # The multiplier-ranked line is no longer the deterministic target once the comparison is on the state.
        stale = observable_rank_one_target_proof(observation, preferred_action=other, expert_actions=[other, best])
        self.assertFalse(stale["passed"])

    def test_the_comparison_variant_turns_the_option_on_only_for_itself(self) -> None:
        self.assertNotIn("candidate_comparison", expert_variant_options("ledger_ranked"))
        self.assertIs(expert_variant_options("ledger_ranked_compare")["candidate_comparison"], True)


class ProcessGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.oracle = ProcessValidityOracle(executor_hydrated_corrections=True)
        self.state = {
            "active_state_id": "s0", "candidate_state_id": None, "evidence_profile": AUXILIARY_EVIDENCE_PROFILE,
            "has_open_candidate": False, "has_unverified_candidate": False, "has_verified_candidate": False,
            "has_fresh_parameter_context": True, "parameter_context_state_id": "s0",
            "fresh_context_evidence": {
                "wls": {"successful": True, "state_id": "s0"},
                "parameter": {
                    "state_id": "s0", "route_status": "actionable",
                    "supported_corrections": [
                        {"tool": CORRECT_PARAMETERS, "arguments": {"state_id": "s0", "line_index": line}} for line in (3, 6, 7)
                    ],
                },
            },
        }

    def _check(self, lines, **overrides):
        state = copy.deepcopy(self.state)
        state.update(overrides)
        return self.oracle.check(state, _compare("s0", lines))

    def test_a_bounded_request_of_supported_lines_on_a_fresh_context_is_valid(self) -> None:
        self.assertTrue(self._check([3, 6])["process_valid"])
        self.assertTrue(self._check([7, 6, 3])["process_valid"])

    def test_the_gate_refuses_malformed_unsupported_stale_and_repeated_requests(self) -> None:
        self.assertEqual(self._check([3])["error_code"], "schema_error")
        self.assertEqual(self._check([3, 6, 7, 9])["error_code"], "schema_error")
        self.assertEqual(self._check([3, 9])["error_detail"], "comparison_candidate_not_supported")
        self.assertEqual(self._check([3, 6], has_fresh_parameter_context=False)["error_detail"], "parameter_context_missing")
        self.assertEqual(self._check([3, 6], has_open_candidate=True, candidate_state_id="c1")["error_code"], "candidate_lifecycle_violation")
        recorded = copy.deepcopy(self.state["fresh_context_evidence"])
        recorded[PARAMETER_COMPARISON_FAMILY] = {"state_id": "s0", "candidates": [
            {"line_index": 3, "request_rank": 0, "execution_status": "success", "verification_verdict": "ACCEPT_PARTIAL", "global_progress": 0.3},
            {"line_index": 6, "request_rank": 1, "execution_status": "success", "verification_verdict": "ACCEPT_FINAL", "global_progress": 1.0},
        ]}
        repeat = self._check([3, 6], fresh_context_evidence=recorded)
        self.assertEqual(repeat["error_detail"], "parameter_comparison_already_recorded")
        self.assertEqual(repeat["valid_next_actions"],
                         [{"tool": CORRECT_PARAMETERS, "arguments": {"state_id": "s0", "line_index": 6}}])
        # A different supported line may still be compared on the same state.
        self.assertTrue(self._check([6, 7], fresh_context_evidence=recorded)["process_valid"])

    def test_strict_profiles_need_the_current_balanced_solve_first(self) -> None:
        state = copy.deepcopy(self.state)
        state["evidence_profile"] = SCADA_ONLY_PROFILE
        state["fresh_context_evidence"]["wls"] = {"successful": False}
        self.assertEqual(self.oracle.check(state, _compare("s0", [3, 6]))["error_detail"], "current_balanced_wls_required")


class EnvironmentComparisonTests(unittest.TestCase):
    def _assert_comparison_contract(self, env: TransactionalPSSEEnv, active: str) -> None:
        before_hash = env.store.episode_hash()
        before_history = len(env.history)
        before_budget = env.current_state()["remaining_budget"]
        _, output = env.step(_compare(active, [WRONG_LINE, TRUE_LINE]))
        self.assertEqual(output["execution_status"], "success", output)
        self.assertFalse(output["state_mutated"])
        record = output["tool_metrics"]
        self.assertEqual(record["ranked_line_indices"], [TRUE_LINE, WRONG_LINE])
        self.assertEqual(record["best_line_index"], TRUE_LINE)
        self.assertEqual(record["candidates"][0]["verification_verdict"], "ACCEPT_FINAL")
        self.assertNotEqual(record["candidates"][1]["verification_verdict"], "ACCEPT_FINAL")
        self.assertEqual((record["internal_correction_calls"], record["internal_wls_calls"]), (2, 2))
        self.assertEqual(record["state_id"], active)
        # The live episode did not move: no candidate, same store, one policy action charged.
        self.assertIsNone(env.current_candidate_id)
        self.assertEqual(env.store.episode_hash(), before_hash)
        self.assertEqual(len(env.history), before_history + 1)
        self.assertEqual(env.current_state()["remaining_budget"], before_budget - 1)
        observation = env.get_policy_observation().as_dict()
        self.assertEqual(find_forbidden_policy_paths(observation), [])
        self.assertEqual(parameter_comparison_ranked_lines(observation), [TRUE_LINE, WRONG_LINE])
        self.assertEqual(observation["fresh_context_evidence"]["parameter"]["state_id"], active)

    def test_the_comparison_reports_branch_verdicts_without_touching_the_live_state(self) -> None:
        env = _env()
        active = _prepared(env)
        self._assert_comparison_contract(env, active)
        # A repeat of the same lines is refused with the best line as the repair.
        _, repeat = env.step(_compare(active, [WRONG_LINE, TRUE_LINE]))
        self.assertEqual(repeat["error_detail"], "parameter_comparison_already_recorded")
        self.assertEqual(repeat["valid_next_actions"][0]["arguments"]["line_index"], TRUE_LINE)
        # Unsupported and malformed requests are refused before any fit runs.
        self.assertEqual(env.step(_compare(active, [WRONG_LINE, 9]))[1]["error_detail"], "comparison_candidate_not_supported")
        self.assertEqual(env.step(_compare(active, [WRONG_LINE]))[1]["error_code"], "schema_error")

    def test_the_record_survives_the_scada_only_sanitizer(self) -> None:
        env = _env(profile=SCADA_ONLY_PROFILE)
        active = _prepared(env)
        self._assert_comparison_contract(env, active)

    def test_the_record_is_dropped_with_the_state_it_was_taken_on(self) -> None:
        env = _env()
        active = _prepared(env)
        env.step(_compare(active, [WRONG_LINE, TRUE_LINE]))
        _, corrected = env.step({"tool": CORRECT_PARAMETERS, "arguments": {"state_id": active, "line_index": TRUE_LINE}})
        self.assertEqual(corrected["execution_status"], "success", corrected)
        candidate = corrected["candidate_state_id"]
        self.assertEqual(env.step(_compare(active, [WRONG_LINE, TRUE_LINE]))[1]["error_code"], "candidate_lifecycle_violation")
        _, verified = env.step({"tool": RUN_WLS, "arguments": {"state_id": candidate}})
        self.assertEqual(verified["execution_status"], "success", verified)
        self.assertEqual(env.current_state()["candidate_disposition"], "ACCEPT_FINAL")
        _, committed = env.step({"tool": COMMIT_STATE, "arguments": {"candidate_state_id": candidate}})
        self.assertEqual(committed["execution_status"], "success", committed)
        self.assertNotIn(PARAMETER_COMPARISON_FAMILY, env.get_policy_observation().fresh_context_evidence)


class ExpertComparisonTests(unittest.TestCase):
    """Under scada_only no waveform screening precedes the balanced corrections."""

    def _expert(self, env: TransactionalPSSEEnv, **options) -> ExpertPolicyOracle:
        return ExpertPolicyOracle(process_oracle=env.process_oracle, **options)

    def test_the_comparison_variant_compares_an_ambiguous_ranking_then_follows_the_verdicts(self) -> None:
        env = _env(profile=SCADA_ONLY_PROFILE)
        active = _prepared(env)
        observation = env.get_policy_observation().as_dict()
        baseline = self._expert(env).next_actions(observation)
        self.assertEqual(baseline[0], {"tool": CORRECT_PARAMETERS, "arguments": {"state_id": active, "line_index": WRONG_LINE}})
        compare = self._expert(env, candidate_comparison=True)
        proposals = compare.next_action_proposals(observation)
        first = proposals[0]
        self.assertEqual(first.action, _compare(active, [WRONG_LINE, TRUE_LINE]))
        self.assertIn("parameter_ranking_not_observably_dominant", first.evidence_codes)
        self.assertEqual(proposals[1].action["tool"], CORRECT_PARAMETERS)
        # The observable wrapper hands the same action to the environment.
        self.assertEqual(ObservableExpertPolicy(compare).act(observation), first.action)
        _, output = env.step(first.action)
        self.assertEqual(output["execution_status"], "success", output)
        after = env.get_policy_observation().as_dict()
        ordered = compare.next_action_proposals(after)
        self.assertEqual(ordered[0].action, {"tool": CORRECT_PARAMETERS, "arguments": {"state_id": active, "line_index": TRUE_LINE}})
        self.assertIn("parameter_comparison_best", ordered[0].evidence_codes)
        self.assertNotIn(COMPARE_PARAMETER_CANDIDATES, [proposal.action["tool"] for proposal in ordered])
        # The baseline expert still corrects the top-ranked line on the same observation.
        self.assertEqual(self._expert(env).next_actions(after)[0]["arguments"]["line_index"], WRONG_LINE)

    def test_a_clearly_dominant_ranking_is_corrected_without_a_comparison(self) -> None:
        env = _env(profile=SCADA_ONLY_PROFILE, dominant=True, ratio=2.0)
        _prepared(env)
        observation = env.get_policy_observation().as_dict()
        actions = self._expert(env, candidate_comparison=True).next_actions(observation)
        self.assertEqual(actions[0]["tool"], CORRECT_PARAMETERS)
        self.assertNotIn(COMPARE_PARAMETER_CANDIDATES, [action["tool"] for action in actions])

    def test_a_ranking_dominant_inside_the_band_is_still_compared(self) -> None:
        env = _env(profile=SCADA_ONLY_PROFILE, dominant=True, ratio=1.32)
        _prepared(env)
        observation = env.get_policy_observation().as_dict()
        actions = self._expert(env, candidate_comparison=True).next_actions(observation)
        self.assertEqual(actions[0]["tool"], COMPARE_PARAMETER_CANDIDATES)
        narrow = self._expert(env, candidate_comparison=True, comparison_dominance_band=1.3).next_actions(observation)
        self.assertEqual(narrow[0]["tool"], CORRECT_PARAMETERS)

    def test_comparison_boundary_is_strict_and_independent_of_dominant_flag(self) -> None:
        for ratio, dominant, expected in (
            (1.19, False, COMPARE_PARAMETER_CANDIDATES),
            (1.2, True, COMPARE_PARAMETER_CANDIDATES),
            (1.30, True, COMPARE_PARAMETER_CANDIDATES),
            (1.499999, True, COMPARE_PARAMETER_CANDIDATES),
            (1.5, True, CORRECT_PARAMETERS),
            (1.500001, True, CORRECT_PARAMETERS),
        ):
            with self.subTest(ratio=ratio, dominant=dominant):
                env = _env(profile=SCADA_ONLY_PROFILE, dominant=dominant, ratio=ratio)
                _prepared(env)
                observation = env.get_policy_observation().as_dict()
                expert = self._expert(env, candidate_comparison=True)
                self.assertEqual(expert.next_actions(observation)[0]["tool"], expected)

    def test_student_sees_the_band_and_durable_dominant_ratio(self) -> None:
        from psse_env.dagger.dataset_builder import prepare_model_policy_observation

        env = _env(profile=SCADA_ONLY_PROFILE, dominant=True, ratio=1.30)
        _prepared(env)
        view, _ = prepare_model_policy_observation(
            env.get_policy_observation().as_dict(), history=[],
        )
        parameter = view["fresh_context_evidence"]["parameter"]
        self.assertIs(parameter["parameter_ranking_dominant"], True)
        self.assertEqual(parameter["parameter_ranking_dominance_ratio"], 1.30)
        self.assertEqual(find_forbidden_policy_paths(view), [])
        schema = next(row["function"] for row in unified_tool_schemas()
                      if row["function"]["name"] == COMPARE_PARAMETER_CANDIDATES)
        description = schema["description"]
        self.assertIn(f"below {DEFAULT_COMPARISON_DOMINANCE_BAND:g}", description)
        self.assertIn("even if the dominant flag is true", description)
        self.assertIn("reuse its verdict order", description)
        self.assertEqual(schema["parameters"]["required"], ["case_path", "line_indices"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
