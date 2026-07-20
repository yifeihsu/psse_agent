from __future__ import annotations

import copy
import unittest

from psse_env.dagger.replay_buffer import build_balanced_training_view


def _row(
    example_id: str,
    *,
    tool: str = "run_wls",
    state_class: str = "clean_successful",
    physical_root: str | None = None,
    scenario_family: str = "measurement",
    error_cardinality: int = 1,
    terminal_outcome: str = "resolved",
    cost_margin: float = 0.2,
) -> dict[str, object]:
    return {
        "example_id": example_id,
        "physical_root_fingerprint": physical_root or f"physical_{example_id}",
        "state_class": state_class,
        "preferred_action": {"tool": tool, "arguments": {}},
        "scenario_family": scenario_family,
        "error_cardinality": error_cardinality,
        "episode_terminal_outcome": terminal_outcome,
        "cost_margin": cost_margin,
    }


class BalancedTrainingViewTests(unittest.TestCase):
    def test_feasible_targets_pass_with_explicit_achieved_deviation_report(self) -> None:
        rows = [
            _row(f"baseline-{index}", tool="run_wls")
            for index in range(5)
        ] + [
            _row(f"context-{index}", tool="get_measurement_context")
            for index in range(5)
        ]

        _, report = build_balanced_training_view(
            rows,
            size=10,
            seed=7,
            tool_category_weights={
                "baseline_diagnostics": 0.5,
                "context_acquisition": 0.5,
            },
            max_duplicate_count=1,
            max_rows_per_root=1,
        )

        expected_counts = {
            "baseline_diagnostics": 5,
            "context_acquisition": 5,
        }
        self.assertTrue(report["passed"])
        self.assertTrue(report["release_ready"])
        self.assertEqual(report["target_counts"]["tool_category"], expected_counts)
        self.assertEqual(report["achieved_counts"]["tool_category"], expected_counts)
        self.assertEqual(report["necessary_feasibility_shortfalls"], {})
        self.assertEqual(report["necessary_feasibility_shortfall_total"], 0)
        category_deviation = report["target_deviation"]["tool_category"]
        self.assertEqual(category_deviation["total_shortfall"], 0)
        self.assertEqual(category_deviation["total_excess"], 0)
        self.assertEqual(category_deviation["total_absolute_deviation"], 0)
        self.assertEqual(category_deviation["relative_deviation"], 0.0)
        for details in category_deviation["values"].values():
            self.assertEqual(details["shortfall"], 0)
            self.assertEqual(details["excess"], 0)
            self.assertEqual(details["absolute_deviation"], 0)
            self.assertEqual(details["relative_deviation"], 0.0)

    def test_infeasible_targets_fail_even_when_deviation_tolerance_is_loose(self) -> None:
        rows = [
            _row(f"baseline-{index}", tool="run_wls")
            for index in range(9)
        ] + [_row("context-only", tool="get_measurement_context")]

        _, report = build_balanced_training_view(
            rows,
            size=10,
            seed=2,
            tool_category_weights={
                "baseline_diagnostics": 0.5,
                "context_acquisition": 0.5,
            },
            max_duplicate_count=1,
            max_rows_per_root=1,
            maximum_tool_category_target_deviation=1.0,
        )

        self.assertFalse(report["passed"])
        self.assertFalse(report["release_ready"])
        self.assertEqual(
            report["necessary_feasibility_shortfalls"]["tool_category"][
                "context_acquisition"
            ],
            {"target": 5, "maximum_achievable": 1, "shortfall": 4},
        )
        self.assertGreater(report["necessary_feasibility_shortfall_total"], 0)
        context_deviation = report["target_deviation"]["tool_category"]["values"][
            "context_acquisition"
        ]
        self.assertEqual(context_deviation["shortfall"], 4)
        self.assertEqual(context_deviation["absolute_deviation"], 4)
        self.assertEqual(context_deviation["relative_deviation"], 0.8)

    def test_secondary_axes_are_capacity_aware_without_relaxing_category_gate(self) -> None:
        rows = [
            _row(f"common-{index}", state_class="clean_successful")
            for index in range(8)
        ] + [
            _row("rare", state_class="rejected_candidate_recovery")
        ]

        _, report = build_balanced_training_view(
            rows,
            size=9,
            seed=17,
            tool_category_weights={"baseline_diagnostics": 1.0},
            max_duplicate_count=1,
            max_rows_per_root=1,
        )

        self.assertTrue(report["release_ready"])
        self.assertEqual(
            report["target_contract"]["strict_target_axes"], ["tool_category"]
        )
        self.assertEqual(
            report["unconstrained_target_counts"]["state_class"],
            {"clean_successful": 4, "rejected_candidate_recovery": 5},
        )
        self.assertEqual(
            report["target_counts"]["state_class"],
            {"clean_successful": 8, "rejected_candidate_recovery": 1},
        )
        self.assertEqual(
            report["capacity_adjustments"]["state_class"],
            {
                "clean_successful": {
                    "unconstrained_target": 4,
                    "capacity_adjusted_target": 8,
                    "maximum_achievable": 8,
                    "necessary_reduction": 0,
                    "redistributed_increase": 4,
                },
                "rejected_candidate_recovery": {
                    "unconstrained_target": 5,
                    "capacity_adjusted_target": 1,
                    "maximum_achievable": 1,
                    "necessary_reduction": 4,
                    "redistributed_increase": 0,
                },
            },
        )
        self.assertNotIn("tool_category", report["capacity_adjustments"])
        self.assertEqual(report["necessary_feasibility_shortfalls"], {})
        self.assertEqual(report["capacity_adjustment_total"], 4)

    def test_achieved_category_deviation_is_an_independent_release_gate(self) -> None:
        rows = [
            _row(
                "e0",
                tool="run_wls",
                physical_root="r1",
                scenario_family="c",
                error_cardinality=1,
                terminal_outcome="resolved",
            ),
            _row(
                "e1",
                tool="get_measurement_context",
                physical_root="r1",
                scenario_family="a",
                error_cardinality=2,
                terminal_outcome="operator_escalation",
            ),
            _row(
                "e2",
                tool="run_wls",
                physical_root="r0",
                scenario_family="b",
                error_cardinality=2,
                terminal_outcome="operator_escalation",
            ),
            _row(
                "e3",
                tool="get_measurement_context",
                physical_root="r1",
                scenario_family="a",
                error_cardinality=2,
                terminal_outcome="resolved",
            ),
        ]
        options = {
            "size": 3,
            "seed": 0,
            "tool_category_weights": {
                "baseline_diagnostics": 0.5,
                "context_acquisition": 0.5,
            },
            "max_duplicate_count": 1,
            "max_rows_per_root": 2,
        }

        _, strict = build_balanced_training_view(
            rows,
            **options,
            maximum_tool_category_target_deviation=0.10,
        )
        self.assertEqual(strict["necessary_feasibility_shortfall_total"], 0)
        self.assertEqual(strict["achieved_tool_category_target_deviation"], 1.0)
        self.assertFalse(strict["passed"])

        _, permissive = build_balanced_training_view(
            rows,
            **options,
            maximum_tool_category_target_deviation=1.0,
        )
        self.assertTrue(permissive["passed"])

    def test_available_tool_categories_cannot_have_zero_weight_mass(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "Available tool categories must receive positive training-weight mass: "
            "baseline_diagnostics",
        ):
            build_balanced_training_view(
                [_row("baseline", tool="run_wls")],
                size=1,
                tool_category_weights={
                    "baseline_diagnostics": 0.0,
                    "context_acquisition": 1.0,
                },
            )

    def test_is_deterministic_reports_all_axes_and_reduces_tool_skew(self) -> None:
        tool_populations = (
            ("run_wls", 40),
            ("get_measurement_context", 4),
            ("correct_measurements", 4),
            ("verify_candidate", 4),
            ("finalize_diagnosis", 4),
            ("diagnose_harmonic", 4),
        )
        rows = [
            _row(
                f"{tool}-{index}",
                tool=tool,
                scenario_family=(
                    "measurement" if index % 2 == 0 else "measurement+topology"
                ),
                error_cardinality=1 if index % 2 == 0 else 2,
                terminal_outcome="resolved" if index % 2 == 0 else "operator_escalation",
            )
            for tool, count in tool_populations
            for index in range(count)
        ]

        first_view, first_report = build_balanced_training_view(
            rows,
            size=30,
            seed=11,
            max_duplicate_count=1,
            max_rows_per_root=1,
        )
        second_view, second_report = build_balanced_training_view(
            reversed(rows),
            size=30,
            seed=11,
            max_duplicate_count=1,
            max_rows_per_root=1,
        )
        self.assertEqual(first_view, second_view)
        self.assertEqual(first_report, second_report)

        review_axes = {
            "state_class",
            "target_tool",
            "scenario_family",
            "error_cardinality",
            "physical_root",
            "terminal_outcome",
        }
        self.assertLessEqual(review_axes, set(first_report["before"]))
        self.assertLessEqual(review_axes, set(first_report["after"]))

        before = first_report["before"]
        after = first_report["after"]
        before_share = before["tool_category"]["baseline_diagnostics"] / before["rows"]
        after_share = after["tool_category"]["baseline_diagnostics"] / after["rows"]
        self.assertLess(after_share, before_share)
        self.assertGreater(
            after["tool_category"]["context_acquisition"] / after["rows"],
            before["tool_category"]["context_acquisition"] / before["rows"],
        )

    def test_physical_root_cap_is_enforced_even_when_duplication_is_needed(self) -> None:
        rows = [
            _row(f"dominant-{index}", physical_root="dominant")
            for index in range(10)
        ]
        rows.extend(
            _row(f"rare-{index}", physical_root=f"rare-{index}")
            for index in range(3)
        )
        view, report = build_balanced_training_view(
            rows,
            size=8,
            seed=4,
            max_duplicate_count=2,
            max_rows_per_root=2,
        )
        self.assertEqual(len(view), 8)
        self.assertEqual(report["max_rows_per_root"], 2)
        self.assertEqual(
            report["after"]["physical_root"],
            {"dominant": 2, "rare-0": 2, "rare-1": 2, "rare-2": 2},
        )
        self.assertLessEqual(max(report["after"]["physical_root"].values()), 2)

    def test_known_low_margin_rows_are_excluded_without_mutating_source(self) -> None:
        rows = [
            _row("top-level-low", cost_margin=0.01),
            {
                **_row("label-threshold", cost_margin=0.2),
                "cost_margin": None,
                "labels": {"cost_margin": 0.05},
            },
            {
                **_row("metadata-high", cost_margin=0.2),
                "cost_margin": None,
                "metadata": {"cost_margin": 0.051},
            },
        ]
        original = copy.deepcopy(rows)
        view, report = build_balanced_training_view(
            rows,
            size=1,
            seed=9,
            max_duplicate_count=1,
            low_cost_margin_threshold=0.05,
        )
        self.assertEqual(rows, original)
        self.assertEqual([row["example_id"] for row in view], ["metadata-high"])
        self.assertEqual(report["excluded_low_margin_rows"], 2)
        self.assertEqual(report["before"]["rows"], 3)
        self.assertEqual(report["after"]["rows"], 1)

    def test_new_terminal_state_classes_are_eligible_and_reported(self) -> None:
        rows = [
            _row(
                "terminal-resolved",
                tool="finalize_diagnosis",
                state_class="terminal_resolved",
                terminal_outcome="resolved",
            ),
            _row(
                "terminal-escalation",
                tool="ask_for_more_evidence",
                state_class="terminal_operator_escalation",
                terminal_outcome="operator_escalation",
            ),
        ]
        view, report = build_balanced_training_view(
            rows, size=2, seed=3, max_duplicate_count=1
        )
        self.assertEqual(len(view), 2)
        self.assertEqual(
            report["after"]["state_class"],
            {"terminal_operator_escalation": 1, "terminal_resolved": 1},
        )
        self.assertEqual(
            report["after"]["terminal_outcome"],
            {"operator_escalation": 1, "resolved": 1},
        )

    def test_unknown_state_class_fails_closed(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "Unknown training-view state class: unreviewed_terminal"
        ):
            build_balanced_training_view(
                [_row("unknown", state_class="unreviewed_terminal")],
                size=1,
                seed=0,
            )


if __name__ == "__main__":
    unittest.main()
