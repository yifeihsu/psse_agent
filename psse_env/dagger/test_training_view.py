from __future__ import annotations

import copy
import unittest

from psse_env.dagger.replay_buffer import (
    build_balanced_training_view,
    build_dagger1_training_view,
)


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
    def test_d0_d1_builder_enforces_deterministic_recovery_share_and_caps(
        self,
    ) -> None:
        d0 = [
            {
                **_row(f"d0-{index}", physical_root=f"d0-root-{index}"),
                "production_label_eligible": True,
                "iteration": 0,
            }
            for index in range(6)
        ]
        d1 = [
            {
                **_row(
                    f"d1-{index}",
                    physical_root=f"d1-root-{index}",
                    state_class="invalid_precondition_recovery",
                ),
                "production_label_eligible": True,
                "iteration": 1,
                "collection_role": "training",
                "collection_beta": 0.25,
                "state_origin": "learner_policy",
                "recovery_stratum": "post_failure_no_candidate",
                "recovery_label_contract": (
                    "observable_rank_one_learner_state_v1"
                ),
                "collection_training_eligible": True,
            }
            for index in range(2)
        ]
        first, first_report = build_dagger1_training_view(
            d0,
            d1,
            size=8,
            seed=73,
            max_duplicate_count=1,
            max_rows_per_root=1,
        )
        second, second_report = build_dagger1_training_view(
            d0,
            d1,
            size=8,
            seed=73,
            max_duplicate_count=1,
            max_rows_per_root=1,
        )
        self.assertEqual(
            [row["example_id"] for row in first],
            [row["example_id"] for row in second],
        )
        self.assertEqual(first_report, second_report)
        self.assertTrue(first_report["release_ready"])
        self.assertEqual(
            first_report["source_allocation"]["observed_d1_share"], 0.25
        )
        self.assertEqual(
            sum(row["replay_source"] == "d1_recovery" for row in first), 2
        )
        self.assertTrue(first_report["physical_root_contract"]["passed"])
        self.assertTrue(first_report["duplicate_contract"]["passed"])

        diagnostic = copy.deepcopy(d1)
        diagnostic[0]["collection_role"] = "diagnostic"
        diagnostic[0]["collection_beta"] = 0.0
        with self.assertRaisesRegex(ValueError, "mixed-policy training"):
            build_dagger1_training_view(d0, diagnostic, size=8)
        overlap = copy.deepcopy(d1)
        overlap[0]["physical_root_fingerprint"] = "d0-root-0"
        with self.assertRaisesRegex(ValueError, "physical roots must be disjoint"):
            build_dagger1_training_view(d0, overlap, size=8)

    def test_targetless_rows_are_excluded_before_balancing(self) -> None:
        targetless = {
            **_row("targetless", tool="run_wls"),
            "preferred_action": None,
            "valid_next_actions": [],
        }
        rows = [
            targetless,
            _row("baseline", tool="run_wls"),
            _row("context", tool="get_measurement_context"),
        ]

        view, report = build_balanced_training_view(
            rows,
            seed=7,
            tool_category_weights={
                "baseline_diagnostics": 0.5,
                "context_acquisition": 0.5,
            },
            max_duplicate_count=1,
            max_rows_per_root=1,
            minimum_tool_category_natural_rows=0,
            minimum_tool_category_distinct_roots=0,
        )

        self.assertEqual(
            {row["example_id"] for row in view}, {"baseline", "context"}
        )
        self.assertEqual(report["input_rows"], 3)
        self.assertEqual(report["target_bearing_input_rows"], 2)
        self.assertEqual(report["excluded_targetless_rows"], 1)
        self.assertEqual(report["requested_size"], 2)
        self.assertEqual(report["returned_size"], 2)
        self.assertEqual(report["before"]["rows"], 2)
        self.assertNotIn("unknown", report["before"]["target_tool"])
        self.assertNotIn(
            "specialized_diagnostics", report["before"]["tool_category"]
        )

    def test_all_targetless_rows_fail_closed(self) -> None:
        row = {
            **_row("targetless", tool="run_wls"),
            "preferred_action": None,
            "valid_next_actions": [],
        }

        with self.assertRaisesRegex(ValueError, "all training rows are targetless"):
            build_balanced_training_view([row])

    def test_already_exported_messages_do_not_become_balancer_targets(self) -> None:
        row = {
            **_row("chat-only", tool="run_wls"),
            "preferred_action": None,
            "valid_next_actions": [],
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"function": {"name": "wls_from_path", "arguments": {}}}
                    ],
                }
            ],
        }

        with self.assertRaisesRegex(ValueError, "all training rows are targetless"):
            build_balanced_training_view([row])

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
            minimum_tool_category_natural_rows=0,
            minimum_tool_category_distinct_roots=0,
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

    def test_natural_support_floor_passes_at_rows_and_roots_boundary(self) -> None:
        rows = [
            _row(f"baseline-{index}", tool="run_wls")
            for index in range(16)
        ] + [
            _row(f"context-{index}", tool="get_measurement_context")
            for index in range(16)
        ]

        _, report = build_balanced_training_view(
            rows,
            size=32,
            seed=4,
            tool_category_weights={
                "baseline_diagnostics": 0.5,
                "context_acquisition": 0.5,
            },
            max_duplicate_count=1,
            max_rows_per_root=1,
        )

        self.assertTrue(report["tool_category_natural_support_passed"])
        self.assertEqual(report["tool_category_natural_support_shortfalls"], {})
        self.assertTrue(report["release_ready"])
        for support in report["tool_category_natural_support"].values():
            self.assertEqual(support["natural_target_bearing_rows"], 16)
            self.assertEqual(support["distinct_roots"], 16)
            self.assertTrue(support["passed"])

    def test_zero_capacity_configured_category_fails_natural_support_floor(self) -> None:
        rows = [
            _row(f"baseline-{index}", tool="run_wls")
            for index in range(16)
        ] + [
            _row(f"context-{index}", tool="get_measurement_context")
            for index in range(16)
        ]

        _, report = build_balanced_training_view(
            rows,
            size=32,
            seed=5,
            tool_category_weights={
                "baseline_diagnostics": 0.45,
                "context_acquisition": 0.45,
                "specialized_diagnostics": 0.10,
            },
            max_duplicate_count=1,
            max_rows_per_root=1,
        )

        self.assertEqual(
            report["unconstrained_target_counts"]["tool_category"]
            ["specialized_diagnostics"],
            3,
        )
        self.assertEqual(
            report["target_counts"]["tool_category"]["specialized_diagnostics"],
            0,
        )
        self.assertEqual(
            report["tool_category_natural_support_shortfalls"]
            ["specialized_diagnostics"],
            {
                "natural_target_bearing_rows": 0,
                "distinct_roots": 0,
                "row_shortfall": 16,
                "root_shortfall": 10,
            },
        )
        self.assertFalse(report["tool_category_natural_support_passed"])
        self.assertFalse(report["release_ready"])

    def test_category_targets_clip_and_redistribute_before_deviation_gate(self) -> None:
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
            minimum_tool_category_natural_rows=0,
            minimum_tool_category_distinct_roots=0,
        )

        self.assertTrue(report["passed"])
        self.assertTrue(report["release_ready"])
        self.assertEqual(
            report["unconstrained_target_counts"]["tool_category"],
            {"baseline_diagnostics": 5, "context_acquisition": 5},
        )
        self.assertEqual(
            report["target_counts"]["tool_category"],
            {"baseline_diagnostics": 9, "context_acquisition": 1},
        )
        self.assertEqual(
            report["capacity_adjustments"]["tool_category"],
            {
                "baseline_diagnostics": {
                    "unconstrained_target": 5,
                    "capacity_adjusted_target": 9,
                    "maximum_achievable": 9,
                    "necessary_reduction": 0,
                    "redistributed_increase": 4,
                },
                "context_acquisition": {
                    "unconstrained_target": 5,
                    "capacity_adjusted_target": 1,
                    "maximum_achievable": 1,
                    "necessary_reduction": 4,
                    "redistributed_increase": 0,
                },
            },
        )
        self.assertEqual(report["necessary_feasibility_shortfalls"], {})
        self.assertEqual(report["achieved_tool_category_target_deviation"], 0.0)

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
            minimum_tool_category_natural_rows=0,
            minimum_tool_category_distinct_roots=0,
        )

        self.assertTrue(report["release_ready"])
        self.assertEqual(
            report["target_contract"]["strict_target_axes"], []
        )
        self.assertEqual(
            report["target_contract"]["deviation_gated_target_axes"],
            ["tool_category"],
        )
        self.assertIn(
            "tool_category", report["target_contract"]["capacity_aware_target_axes"]
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
                "baseline-shared-root",
                tool="run_wls",
                physical_root="shared",
            ),
            _row(
                "context-shared-root",
                tool="get_measurement_context",
                physical_root="shared",
            ),
            _row(
                "terminal-r1",
                tool="finalize_diagnosis",
                physical_root="r1",
            ),
            _row(
                "terminal-r2",
                tool="finalize_diagnosis",
                physical_root="r2",
            ),
        ]
        options = {
            "size": 3,
            "seed": 0,
            "tool_category_weights": {
                "baseline_diagnostics": 1.0,
                "context_acquisition": 1.0,
                "terminal_or_handoff": 1.0,
            },
            "max_duplicate_count": 1,
            "max_rows_per_root": 1,
            "minimum_tool_category_natural_rows": 0,
            "minimum_tool_category_distinct_roots": 0,
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

    def test_exact_action_root_gate_uses_unique_explicit_physical_roots(self) -> None:
        missing_root = _row(
            "parameter-missing-root",
            tool="correct_parameters",
        )
        missing_root.pop("physical_root_fingerprint")
        rows = [
            _row(
                "parameter-r1-a",
                tool="correct_parameters",
                physical_root="r1",
            ),
            _row(
                "parameter-r1-b",
                tool="correct_parameters",
                physical_root="r1",
            ),
            _row(
                "parameter-r2",
                tool="correct_parameters",
                physical_root="r2",
            ),
            missing_root,
        ]

        options = {
            "seed": 3,
            "tool_category_weights": {"corrections": 1.0},
            "max_duplicate_count": 1,
            "minimum_tool_category_natural_rows": 0,
            "minimum_tool_category_distinct_roots": 0,
        }
        _, passing_report = build_balanced_training_view(
            rows,
            **options,
            target_tool_minimum_distinct_roots={"correct_parameters": 2},
        )
        self.assertTrue(passing_report["target_tool_unique_root_support_passed"])
        self.assertTrue(passing_report["release_ready"])

        _, report = build_balanced_training_view(
            rows,
            **options,
            target_tool_minimum_distinct_roots={"correct_parameters": 3},
        )

        natural = report["target_tool_unique_root_support"][
            "eligible_natural_source"
        ]["correct_parameters"]
        self.assertEqual(natural["target_bearing_rows"], 4)
        self.assertEqual(natural["distinct_physical_roots"], 2)
        self.assertEqual(natural["rows_missing_physical_root"], 1)
        self.assertEqual(natural["root_shortfall"], 1)
        reservation = report["requirement_aware_reservation"]
        self.assertTrue(
            reservation["feasible_requirements_satisfied_by_reservation"]
        )
        self.assertEqual(
            reservation["requirements"][0][
                "natural_distinct_physical_roots"
            ],
            2,
        )
        self.assertFalse(
            reservation["requirements"][0]["natural_support_feasible"]
        )
        self.assertEqual(
            reservation["requirements"][0][
                "selected_distinct_physical_roots"
            ],
            2,
        )
        self.assertEqual(
            reservation["requirements"][0]["selected_root_shortfall"],
            1,
        )
        self.assertFalse(report["target_tool_unique_root_support_passed"])
        self.assertFalse(report["release_ready"])

    def test_parameter_corrections_retain_same_root_context_before_duplication(
        self,
    ) -> None:
        rows = []
        for root, family in (
            ("parameter-root", "parameter"),
            ("mixed-root", "measurement+parameter"),
        ):
            rows.extend(
                [
                    _row(
                        f"{root}-context",
                        tool="get_parameter_context",
                        physical_root=root,
                        scenario_family=family,
                    ),
                    _row(
                        f"{root}-correction",
                        tool="correct_parameters",
                        physical_root=root,
                        scenario_family=family,
                    ),
                ]
            )

        view, report = build_balanced_training_view(
            rows,
            size=6,
            seed=19,
            tool_category_weights={
                "context_acquisition": 0.25,
                "corrections": 0.75,
            },
            max_duplicate_count=2,
            max_rows_per_root=3,
            minimum_tool_category_natural_rows=0,
            minimum_tool_category_distinct_roots=0,
            target_tool_minimum_distinct_roots={"correct_parameters": 2},
            same_root_prerequisite_rules={
                "correct_parameters": {
                    "prerequisite_tool": "get_parameter_context",
                    "scenario_families": [
                        "parameter",
                        "measurement+parameter",
                    ],
                }
            },
        )

        selected_tools_by_root: dict[str, list[str]] = {}
        for row in view:
            selected_tools_by_root.setdefault(
                str(row["physical_root_fingerprint"]), []
            ).append(str(row["preferred_action"]["tool"]))
        for tools in selected_tools_by_root.values():
            if "correct_parameters" in tools:
                self.assertIn("get_parameter_context", tools)
        self.assertEqual(
            sum(
                tool == "correct_parameters"
                for tools in selected_tools_by_root.values()
                for tool in tools
            ),
            4,
        )
        self.assertTrue(report["same_root_prerequisite_support_passed"])
        self.assertTrue(report["release_ready"])
        self.assertIn(
            "same_root_target_prerequisites",
            report["target_contract"]["strict_target_axes"],
        )
        self.assertEqual(
            report["requirement_aware_reservation"]["policy"],
            "constrained_first_with_same_root_prerequisites_v2",
        )

    def test_unpaired_parameter_correction_fails_closed_and_is_not_selected(
        self,
    ) -> None:
        rows = [
            _row(
                "unpaired-correction",
                tool="correct_parameters",
                physical_root="parameter-root",
                scenario_family="parameter",
            ),
            _row(
                "safe-baseline",
                tool="run_wls",
                physical_root="baseline-root",
                scenario_family="measurement",
            ),
        ]

        view, report = build_balanced_training_view(
            rows,
            size=2,
            seed=7,
            tool_category_weights={
                "baseline_diagnostics": 0.5,
                "corrections": 0.5,
            },
            max_duplicate_count=2,
            max_rows_per_root=2,
            minimum_tool_category_natural_rows=0,
            minimum_tool_category_distinct_roots=0,
            same_root_prerequisite_rules={
                "correct_parameters": {
                    "prerequisite_tool": "get_parameter_context",
                    "scenario_families": ["parameter"],
                }
            },
        )

        self.assertEqual(
            {row["example_id"] for row in view}, {"safe-baseline"}
        )
        shortfall = report["same_root_prerequisite_shortfalls"][
            "eligible_natural_source"
        ]["correct_parameters"]["parameter"]
        self.assertEqual(shortfall["unpaired_target_distinct_physical_roots"], 1)
        self.assertEqual(
            shortfall["unpaired_target_physical_roots"], ["parameter-root"]
        )
        self.assertFalse(report["same_root_prerequisite_support_passed"])
        self.assertFalse(report["release_ready"])

    def test_mixed_parameter_correction_accepts_only_bound_bundled_context(
        self,
    ) -> None:
        bundle = {
            **_row(
                "mixed-bundled-context",
                tool="get_measurement_context",
                physical_root="mixed-root",
                scenario_family="measurement+parameter",
            ),
            "tool_output": {
                "tool_metrics": {
                    "branch_route_screening": {
                        "parameter": {
                            "context_tool": "get_parameter_context",
                            "route_status": "actionable",
                            "state_id": "active",
                            "state_hash": "abc123",
                            "supported_corrections": [
                                {
                                    "tool": "correct_parameters",
                                    "arguments": {
                                        "state_id": "active",
                                        "line_index": 3,
                                    },
                                }
                            ],
                        }
                    }
                }
            },
        }
        rules = {
            "correct_parameters": {
                "prerequisite_options_by_family": {
                    "measurement+parameter": [
                        {"tool": "get_parameter_context"},
                        {
                            "tool": "get_measurement_context",
                            "evidence_path": (
                                "tool_output.tool_metrics."
                                "branch_route_screening.parameter"
                            ),
                            "evidence_contract": (
                                "bound_supported_parameter_inventory_v1"
                            ),
                        },
                    ],
                    "parameter": [{"tool": "get_parameter_context"}],
                }
            }
        }
        rows = [
            bundle,
            _row(
                "mixed-correction",
                tool="correct_parameters",
                physical_root="mixed-root",
                scenario_family="measurement+parameter",
            ),
            _row(
                "pure-context",
                tool="get_parameter_context",
                physical_root="pure-root",
                scenario_family="parameter",
            ),
            _row(
                "pure-correction",
                tool="correct_parameters",
                physical_root="pure-root",
                scenario_family="parameter",
            ),
        ]
        view, report = build_balanced_training_view(
            rows,
            size=4,
            seed=11,
            tool_category_weights={
                "context_acquisition": 0.5,
                "corrections": 0.5,
            },
            max_duplicate_count=1,
            max_rows_per_root=2,
            minimum_tool_category_natural_rows=0,
            minimum_tool_category_distinct_roots=0,
            same_root_prerequisite_rules=rules,
        )
        self.assertEqual(len(view), 4)
        self.assertTrue(report["same_root_prerequisite_support_passed"])
        mixed = report["same_root_prerequisite_support"]["training_view"][
            "correct_parameters"
        ]["measurement+parameter"]
        self.assertEqual(mixed["paired_distinct_physical_roots"], 1)
        self.assertEqual(
            mixed["prerequisite_option_distinct_physical_roots"][
                "get_measurement_context:bound_supported_parameter_inventory_v1"
            ],
            1,
        )

        invalid_bundle = copy.deepcopy(bundle)
        invalid_bundle["example_id"] = "empty-bundle"
        invalid_bundle["physical_root_fingerprint"] = "unbound-root"
        invalid_bundle["tool_output"]["tool_metrics"]["branch_route_screening"] = {}
        invalid_correction = _row(
            "unbound-correction",
            tool="correct_parameters",
            physical_root="unbound-root",
            scenario_family="measurement+parameter",
        )
        _, invalid_report = build_balanced_training_view(
            [invalid_bundle, invalid_correction],
            size=1,
            seed=11,
            tool_category_weights={
                "context_acquisition": 0.5,
                "corrections": 0.5,
            },
            max_duplicate_count=1,
            max_rows_per_root=2,
            minimum_tool_category_natural_rows=0,
            minimum_tool_category_distinct_roots=0,
            same_root_prerequisite_rules=rules,
        )
        self.assertFalse(
            invalid_report["same_root_prerequisite_support_passed"]
        )

    def test_production_ineligible_rows_cannot_satisfy_action_root_gate(self) -> None:
        auxiliary = {
            **_row(
                "parameter-auxiliary",
                tool="correct_parameters",
                physical_root="r3",
            ),
            "labels": {"production_label_eligible": False},
        }
        rows = [
            {
                **_row(
                    "parameter-r1",
                    tool="correct_parameters",
                    physical_root="r1",
                ),
                "production_label_eligible": True,
            },
            {
                **_row(
                    "parameter-r2",
                    tool="correct_parameters",
                    physical_root="r2",
                ),
                "production_label_eligible": True,
            },
            auxiliary,
        ]

        view, report = build_balanced_training_view(
            rows,
            seed=4,
            tool_category_weights={"corrections": 1.0},
            max_duplicate_count=1,
            minimum_tool_category_natural_rows=0,
            minimum_tool_category_distinct_roots=0,
            target_tool_minimum_distinct_roots={"correct_parameters": 3},
            require_production_label_eligible=True,
        )

        self.assertEqual({row["example_id"] for row in view}, {
            "parameter-r1",
            "parameter-r2",
        })
        self.assertEqual(
            report["explicitly_production_ineligible_input_rows"], 1
        )
        self.assertEqual(report["training_view_candidate_input_rows"], 2)
        natural = report["target_tool_unique_root_support"][
            "eligible_natural_source"
        ]["correct_parameters"]
        self.assertEqual(natural["distinct_physical_roots"], 2)
        self.assertEqual(natural["root_shortfall"], 1)
        self.assertFalse(report["release_ready"])

    def test_reservation_does_not_fabricate_absent_required_action(self) -> None:
        _, report = build_balanced_training_view(
            [_row("baseline", tool="run_wls")],
            seed=4,
            tool_category_weights={"baseline_diagnostics": 1.0},
            max_duplicate_count=1,
            minimum_tool_category_natural_rows=0,
            minimum_tool_category_distinct_roots=0,
            target_tool_minimum_distinct_roots={"rollback_state": 1},
        )

        shortfall = report["target_tool_unique_root_shortfalls"][
            "eligible_natural_source"
        ]["rollback_state"]
        self.assertEqual(shortfall["distinct_physical_roots"], 0)
        self.assertEqual(shortfall["root_shortfall"], 1)
        requirement = report["requirement_aware_reservation"][
            "requirements"
        ][0]
        self.assertEqual(
            requirement["reservation_target_distinct_physical_roots"],
            0,
        )
        self.assertEqual(requirement["selected_distinct_physical_roots"], 0)
        self.assertEqual(requirement["selected_root_shortfall"], 1)
        self.assertFalse(report["release_ready"])

    def test_critical_joint_root_gates_keep_state_and_family_cells_separate(
        self,
    ) -> None:
        rows = [
            _row(
                "rollback",
                tool="rollback_state",
                state_class="rejected_candidate_recovery",
                physical_root="rollback-root",
            ),
            _row(
                "wrong-state-rollback",
                tool="rollback_state",
                state_class="clean_successful",
                physical_root="wrong-state-root",
            ),
            _row(
                "hif-handoff",
                tool="ask_for_more_evidence",
                state_class="terminal_operator_escalation",
                scenario_family="hif",
                physical_root="hif-root",
            ),
            _row(
                "multi-handoff",
                tool="ask_for_more_evidence",
                state_class="terminal_operator_escalation",
                scenario_family="multi_measurement",
                physical_root="multi-root",
            ),
        ]

        options = {
            "seed": 8,
            "tool_category_weights": {
                "verification_lifecycle": 0.5,
                "terminal_or_handoff": 0.5,
            },
            "max_duplicate_count": 1,
            "minimum_tool_category_natural_rows": 0,
            "minimum_tool_category_distinct_roots": 0,
            "target_tool_scenario_family_minimum_distinct_roots": {
                "ask_for_more_evidence": {
                    "hif": 1,
                    "multi_measurement": 1,
                }
            },
        }
        _, passing_report = build_balanced_training_view(
            rows,
            **options,
            target_tool_state_class_minimum_distinct_roots={
                "rollback_state": {"rejected_candidate_recovery": 1}
            },
        )
        self.assertTrue(
            passing_report["critical_joint_unique_root_support_passed"]
        )
        self.assertTrue(passing_report["release_ready"])

        _, report = build_balanced_training_view(
            rows,
            **options,
            target_tool_state_class_minimum_distinct_roots={
                "rollback_state": {"rejected_candidate_recovery": 2}
            },
        )

        state_support = report["critical_joint_unique_root_support"][
            "target_tool_x_state_class"
        ]["eligible_natural_source"]["rollback_state"]
        self.assertEqual(
            state_support["rejected_candidate_recovery"][
                "distinct_physical_roots"
            ],
            1,
        )
        self.assertEqual(
            state_support["clean_successful"]["distinct_physical_roots"],
            1,
        )
        self.assertEqual(
            report["critical_joint_unique_root_shortfalls"][
                "target_tool_x_state_class"
            ]["eligible_natural_source"]["rollback_state"][
                "rejected_candidate_recovery"
            ]["root_shortfall"],
            1,
        )
        family_support = report["critical_joint_unique_root_support"][
            "target_tool_x_scenario_family"
        ]["eligible_natural_source"]["ask_for_more_evidence"]
        self.assertEqual(family_support["hif"]["distinct_physical_roots"], 1)
        self.assertEqual(
            family_support["multi_measurement"]["distinct_physical_roots"],
            1,
        )
        self.assertFalse(report["critical_joint_unique_root_support_passed"])
        self.assertFalse(report["release_ready"])

    def test_feasible_critical_cell_is_reserved_before_greedy_balancing(
        self,
    ) -> None:
        rows = [
            _row(
                f"baseline-{index}",
                tool="run_wls",
                physical_root=f"baseline-root-{index}",
                scenario_family="measurement",
            )
            for index in range(9)
        ] + [
            _row(
                "hif-handoff",
                tool="ask_for_more_evidence",
                state_class="terminal_operator_escalation",
                physical_root="hif-root",
                scenario_family="hif",
                terminal_outcome="operator_escalation",
            ),
            _row(
                "multi-handoff",
                tool="ask_for_more_evidence",
                state_class="terminal_operator_escalation",
                physical_root="multi-root",
                scenario_family="multi_measurement",
                terminal_outcome="operator_escalation",
            ),
        ]
        options = {
            "size": 10,
            "seed": 0,
            "tool_category_weights": {
                "baseline_diagnostics": 0.9,
                "terminal_or_handoff": 0.1,
            },
            "max_duplicate_count": 1,
            "max_rows_per_root": 1,
            "minimum_tool_category_natural_rows": 0,
            "minimum_tool_category_distinct_roots": 0,
            "target_tool_scenario_family_minimum_distinct_roots": {
                "ask_for_more_evidence": {"hif": 1}
            },
        }

        view, report = build_balanced_training_view(rows, **options)
        reversed_view, reversed_report = build_balanced_training_view(
            reversed(rows),
            **options,
        )

        self.assertEqual(view, reversed_view)
        self.assertEqual(report, reversed_report)
        self.assertIn(
            "hif-handoff",
            {row["example_id"] for row in view},
        )
        self.assertEqual(
            len(
                {
                    row["physical_root_fingerprint"]
                    for row in view
                }
            ),
            len(view),
        )
        reservation = report["requirement_aware_reservation"]
        self.assertEqual(reservation["reserved_rows"], 1)
        self.assertTrue(
            reservation["feasible_requirements_satisfied_by_reservation"]
        )
        self.assertEqual(
            reservation["requirements"],
            [
                {
                    "axis": "scenario_family",
                    "target_tool": "ask_for_more_evidence",
                    "value": "hif",
                    "minimum_distinct_physical_roots": 1,
                    "natural_distinct_physical_roots": 1,
                    "natural_support_feasible": True,
                    "reservation_target_distinct_physical_roots": 1,
                    "reserved_distinct_physical_roots": 1,
                    "reservation_shortfall": 0,
                    "selected_distinct_physical_roots": 1,
                    "selected_root_shortfall": 0,
                }
            ],
        )
        self.assertTrue(report["release_ready"])

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
