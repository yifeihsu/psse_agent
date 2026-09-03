from __future__ import annotations

import unittest

from hif_search_limits import HIF_ALPHA_GRID_SIZE_MAX
from three_phase_nlm.ieee14_adapter import ELIGIBLE_HIF_BRANCHES

from psse_env.actions import (
    ASK_FOR_MORE_EVIDENCE,
    CORRECT_PARAMETERS,
    ESTIMATE_HIF_FROM_PATH,
    ESTIMATE_HIF_MULTISCAN_FROM_PATH,
    HIF_DIAGNOSTICS_EXHAUSTED_REQUEST,
    RUN_HSE_FROM_PATH,
    RUN_THREE_PHASE_NLM_FROM_PATH,
    safe_normalize_action,
)
from psse_env.dagger.counterfactual_generator import (
    CounterfactualGenerator,
    latest_nlm_top_branch,
)
from psse_env.dagger.error_injectors import (
    NLM_TOP_BRANCH_PLACEHOLDER,
    NLM_WRONG_BRANCH_PLACEHOLDER,
    diagnostic_wrong_actions,
    plausible_wrong_actions,
    resolve_nlm_branch_placeholder,
)


def _state(**overrides) -> dict:
    state = {
        "active_state_id": "episode:s0",
        "candidate_state_id": None,
        "available_evidence": [],
        "unresolved_signatures": [],
    }
    state.update(overrides)
    return state


def _nlm_history(top_branch_row0: int, *, status: str = "success") -> list[dict]:
    return [
        {
            "state_id": "episode:s0",
            "candidate_state_id": None,
            "action": {
                "tool": RUN_THREE_PHASE_NLM_FROM_PATH,
                "arguments": {"state_id": "episode:s0"},
            },
            "tool_output": {
                "execution_status": status,
                "tool_metrics": {
                    "nlm_summary": {
                        "top_hif_groups": [{"branch_row0": top_branch_row0}]
                    }
                },
            },
        }
    ]


class _Branch:
    def __init__(self, history: list[dict]) -> None:
        self.history = history

    @staticmethod
    def current_state() -> dict:
        return {"active_state_id": "episode:s0", "candidate_state_id": None}


class DiagnosticInjectionTests(unittest.TestCase):
    def test_correction_roots_get_no_diagnostic_mistakes(self) -> None:
        expert = [
            {"tool": "get_parameter_context", "arguments": {"state_id": "episode:s0"}}
        ]
        self.assertEqual(diagnostic_wrong_actions(_state(), expert), [])
        families = {item.family for item in plausible_wrong_actions(_state(), expert)}
        self.assertFalse(
            families
            & {
                "premature_hif_estimation",
                "wrong_hif_candidate_branch",
                "hif_search_budget_overrun",
                "premature_diagnostic_escalation",
                "wrong_diagnostic_family",
                "masking_correction_on_diagnostic_anomaly",
            }
        )

    def test_nlm_root_emits_the_full_ladder_of_mistakes(self) -> None:
        state = _state(
            available_evidence=["hif_scan_window", "three_phase_branch_currents"],
            unresolved_signatures=["hif_suspected_zero_sequence"],
        )
        expert = [
            {"tool": RUN_THREE_PHASE_NLM_FROM_PATH, "arguments": {"state_id": "episode:s0"}}
        ]
        injected = {item.family: item for item in diagnostic_wrong_actions(state, expert)}
        self.assertEqual(
            sorted(injected),
            [
                "hif_search_budget_overrun",
                "masking_correction_on_diagnostic_anomaly",
                "premature_diagnostic_escalation",
                "premature_hif_estimation",
                "wrong_diagnostic_family",
                "wrong_hif_candidate_branch",
            ],
        )
        for item in injected.values():
            normalized = safe_normalize_action(item.action)
            self.assertNotEqual(normalized["tool"], "__invalid_action__", item.family)

        premature = injected["premature_hif_estimation"]
        self.assertEqual(premature.action["tool"], ESTIMATE_HIF_MULTISCAN_FROM_PATH)
        self.assertEqual(
            premature.action["arguments"]["candidate_branch_row0"],
            int(ELIGIBLE_HIF_BRANCHES[0]),
        )
        self.assertEqual(premature.setup_actions, ())

        wrong_branch = injected["wrong_hif_candidate_branch"]
        self.assertEqual(len(wrong_branch.setup_actions), 1)
        self.assertEqual(
            wrong_branch.setup_actions[0]["tool"], RUN_THREE_PHASE_NLM_FROM_PATH
        )
        self.assertEqual(
            wrong_branch.action["arguments"]["candidate_branch_row0"],
            NLM_WRONG_BRANCH_PLACEHOLDER,
        )

        overrun = injected["hif_search_budget_overrun"]
        self.assertEqual(
            overrun.action["arguments"]["candidate_branch_row0"],
            NLM_TOP_BRANCH_PLACEHOLDER,
        )
        self.assertEqual(
            overrun.action["arguments"]["alpha_grid_size"], HIF_ALPHA_GRID_SIZE_MAX + 1
        )

        escalation = injected["premature_diagnostic_escalation"]
        self.assertEqual(escalation.action["tool"], ASK_FOR_MORE_EVIDENCE)
        self.assertEqual(
            escalation.action["arguments"]["request"], HIF_DIAGNOSTICS_EXHAUSTED_REQUEST
        )

        self.assertEqual(injected["wrong_diagnostic_family"].action["tool"], RUN_HSE_FROM_PATH)
        masking = injected["masking_correction_on_diagnostic_anomaly"]
        self.assertEqual(masking.action["tool"], CORRECT_PARAMETERS)
        self.assertEqual(masking.action["arguments"]["branch_row0"], int(ELIGIBLE_HIF_BRANCHES[0]))

    def test_single_scan_rows_use_the_single_scan_estimator(self) -> None:
        state = _state(available_evidence=["nlm_diagnostic", "three_phase_voltages"])
        expert = [
            {"tool": RUN_THREE_PHASE_NLM_FROM_PATH, "arguments": {"state_id": "episode:s0"}}
        ]
        tools = {
            item.family: item.action["tool"]
            for item in diagnostic_wrong_actions(state, expert)
        }
        self.assertEqual(tools["premature_hif_estimation"], ESTIMATE_HIF_FROM_PATH)
        self.assertEqual(tools["wrong_hif_candidate_branch"], ESTIMATE_HIF_FROM_PATH)

    def test_harmonic_root_swaps_family_and_escalates_early(self) -> None:
        expert = [
            {"tool": "get_harmonic_context", "arguments": {"state_id": "episode:s0"}}
        ]
        injected = {item.family: item for item in diagnostic_wrong_actions(_state(), expert)}
        self.assertEqual(
            sorted(injected), ["premature_diagnostic_escalation", "wrong_diagnostic_family"]
        )
        self.assertEqual(
            injected["wrong_diagnostic_family"].action["tool"],
            RUN_THREE_PHASE_NLM_FROM_PATH,
        )

    def test_injected_actions_never_reference_hidden_truth(self) -> None:
        state = _state(
            available_evidence=["hif_scan_window"],
            true_hif_errors=[{"branch_row0": 13, "phase": "B"}],
        )
        expert = [
            {"tool": RUN_THREE_PHASE_NLM_FROM_PATH, "arguments": {"state_id": "episode:s0"}}
        ]
        for item in diagnostic_wrong_actions(state, expert):
            arguments = item.action["arguments"]
            self.assertNotEqual(arguments.get("candidate_branch_row0"), 13, item.family)
            self.assertNotIn("phase", arguments)
            self.assertNotIn("candidate_phase", arguments)


class BranchPlaceholderBindingTests(unittest.TestCase):
    def test_top_placeholder_binds_to_localized_line(self) -> None:
        self.assertEqual(resolve_nlm_branch_placeholder(NLM_TOP_BRANCH_PLACEHOLDER, 13), 13)

    def test_wrong_placeholder_moves_to_the_next_eligible_line(self) -> None:
        eligible = [int(row) for row in ELIGIBLE_HIF_BRANCHES]
        for top in eligible:
            wrong = resolve_nlm_branch_placeholder(NLM_WRONG_BRANCH_PLACEHOLDER, top)
            self.assertNotEqual(wrong, top)
            self.assertIn(wrong, eligible)
        # Transformer rows are never HIF-eligible targets.
        self.assertEqual(
            resolve_nlm_branch_placeholder(NLM_WRONG_BRANCH_PLACEHOLDER, eligible[-1]),
            eligible[0],
        )
        wrong_from_transformer = resolve_nlm_branch_placeholder(NLM_WRONG_BRANCH_PLACEHOLDER, 8)
        self.assertIn(wrong_from_transformer, eligible)

    def test_missing_localization_falls_back_to_first_eligible_line(self) -> None:
        for placeholder in (NLM_TOP_BRANCH_PLACEHOLDER, NLM_WRONG_BRANCH_PLACEHOLDER):
            self.assertEqual(
                resolve_nlm_branch_placeholder(placeholder, None), int(ELIGIBLE_HIF_BRANCHES[0])
            )
        with self.assertRaises(ValueError):
            resolve_nlm_branch_placeholder("__unknown__", 3)

    def test_latest_nlm_top_branch_reads_observable_history_only(self) -> None:
        self.assertEqual(latest_nlm_top_branch(_nlm_history(13)), 13)
        self.assertIsNone(latest_nlm_top_branch(_nlm_history(13, status="failure")))
        self.assertIsNone(latest_nlm_top_branch([]))

    def test_generator_binds_placeholders_from_branch_history(self) -> None:
        branch = _Branch(_nlm_history(13))
        bound = CounterfactualGenerator._bind_dynamic_state_ids(
            branch,
            {
                "tool": ESTIMATE_HIF_MULTISCAN_FROM_PATH,
                "arguments": {
                    "state_id": "__active__",
                    "candidate_branch_row0": NLM_WRONG_BRANCH_PLACEHOLDER,
                },
            },
        )
        self.assertEqual(bound["arguments"]["state_id"], "episode:s0")
        self.assertNotEqual(bound["arguments"]["candidate_branch_row0"], 13)
        self.assertIn(bound["arguments"]["candidate_branch_row0"], [int(r) for r in ELIGIBLE_HIF_BRANCHES])
        top = CounterfactualGenerator._bind_dynamic_state_ids(
            branch,
            {
                "tool": ESTIMATE_HIF_MULTISCAN_FROM_PATH,
                "arguments": {
                    "state_id": "__active__",
                    "candidate_branch_row0": NLM_TOP_BRANCH_PLACEHOLDER,
                },
            },
        )
        self.assertEqual(top["arguments"]["candidate_branch_row0"], 13)
        literal = CounterfactualGenerator._bind_dynamic_state_ids(
            branch,
            {"tool": ESTIMATE_HIF_FROM_PATH, "arguments": {"state_id": "__active__", "candidate_branch_row0": 4}},
        )
        self.assertEqual(literal["arguments"]["candidate_branch_row0"], 4)


if __name__ == "__main__":
    unittest.main()
