"""Breaker-level topology investigation: substation measurements, the node/breaker
normalized-multiplier estimator, and the breaker-named correction."""
from __future__ import annotations

import copy
import unittest

from psse_env.actions import normalize_action
from psse_env.providers.matpower import MatpowerDeploymentProviders
from psse_env.providers.scenario_generator import Round0ScenarioGenerator


def _state(scenario: dict, state_id: str = "episode:s0") -> dict:
    return {
        "state_id": state_id,
        "state_hash": "hash0",
        "status": "active",
        "case": scenario["case"],
        "measurements": list(scenario["measurements"]),
        "metadata": copy.deepcopy(scenario["metadata"]),
    }


class NodeBreakerTopologyScenarioTests(unittest.TestCase):
    """A breaker whose true state isolates one line terminal: the operator model
    keeps its 14 buses and the fix takes that line out of service."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.generator = Round0ScenarioGenerator(
            seed=31, topology_effects=("dangling_line_terminal",)
        )
        cls.scenario = cls.generator.build({"topology": 1})[0]
        cls.truth = cls.scenario["true_topology_errors"][0]
        cls.providers = MatpowerDeploymentProviders(chi2_alpha=0.01)

    def test_scenario_carries_substation_channel_and_reported_statuses(self) -> None:
        metadata = self.scenario["metadata"]
        telemetry = metadata["substation_telemetry"]
        self.assertEqual(len(telemetry["node_vm"]), 65)
        self.assertEqual(len(telemetry["cb_p"]), 73)
        self.assertEqual(len(telemetry["branch_pf"]), 20)
        reported = metadata["reported_breaker_status"]
        self.assertEqual(len(reported), 73)
        # The reported statuses are the schematic-normal ones: the truth differs
        # on exactly the sampled breaker, and only the hidden truth says which.
        expected = "closed" if self.truth["reported_cb_closed"] else "open"
        self.assertEqual(reported[self.truth["cb_name"]], expected)
        self.assertNotEqual(reported[self.truth["cb_name"]], "closed" if self.truth["true_cb_closed"] else "open")
        self.assertEqual(len(metadata["operator_voltage_meter_nodes"]), 14)
        for key in metadata:
            self.assertFalse(str(key).startswith(("true_", "clean_", "hidden_")), key)
        self.assertNotIn(self.truth["cb_name"], str(metadata["operator_voltage_meter_nodes"]))
        ranking = self.scenario["topology_ranking"]
        self.assertEqual(ranking["true_breaker_rank"], 1)
        self.assertNotIn("top_breaker", ranking)
        self.assertGreater(ranking["gse_chi_square"], ranking["gse_threshold"])

    def test_operator_vector_reads_the_telemetry_meters(self) -> None:
        metadata = self.scenario["metadata"]
        telemetry = metadata["substation_telemetry"]
        meters = metadata["operator_voltage_meter_nodes"]
        z = self.scenario["measurements"]
        for bus in range(1, 15):
            self.assertAlmostEqual(z[bus - 1], telemetry["node_vm"][meters[str(bus)]], places=9)
        self.assertEqual(z[42:62], list(telemetry["branch_pf"]))

    def test_context_requests_substation_measurements_and_names_the_breaker(self) -> None:
        context = self.providers.get_topology_context(_state(self.scenario))
        self.assertNotIn("execution_status", context)
        self.assertTrue(context["substation_measurements_requested"])
        self.assertEqual(
            context["evidence_source"], "deployment_context:node_breaker_nlm_candidate_screened"
        )
        estimate = context["node_breaker_estimate"]
        self.assertTrue(estimate["anomalous"])
        self.assertEqual(
            estimate["method"], "generalized_state_estimation_normalized_lagrange_multipliers"
        )
        findings = context["breaker_findings"]
        self.assertEqual(findings[0]["cb_name"], self.truth["cb_name"])
        self.assertTrue(findings[0]["flip_explains_substation_measurements"])
        self.assertEqual(findings[0]["line_index"], self.truth["line_index1"])
        self.assertEqual(context["route_status"], "actionable")
        supported = context["supported_corrections"]
        self.assertEqual(len(supported), 1)
        arguments = supported[0]["arguments"]
        self.assertEqual(arguments["cb_name"], self.truth["cb_name"])
        self.assertEqual(arguments["line_index"], self.truth["line_index1"])
        self.assertEqual(arguments["status"], 0)
        normalized = normalize_action(supported[0])
        self.assertEqual(normalized["arguments"]["cb_name"], self.truth["cb_name"])
        screened = context["topology_candidate_screening"][0]
        self.assertEqual(screened["cb_name"], self.truth["cb_name"])
        self.assertTrue(screened["eligible"])
        self.assertEqual(screened["hypothesis_source"], "node_breaker_nlm_ranking")

    def test_non_converged_candidate_flip_is_reported_finite_and_not_offered(self) -> None:
        """A candidate whose flipped re-estimation diverges (NaN chi-square)
        must reach the model-visible context as a finite, non-explaining finding
        and never as a supported correction; the true breaker stays supported."""
        import json
        import math
        from unittest import mock

        import Transmission.ieee14_full_gse as gse

        real = gse.screen_breaker_flips
        poisoned: dict[str, str] = {}

        def diverging(model, reference, reported_status, telemetry, candidates, **kwargs):
            results = real(model, reference, reported_status, telemetry, candidates, **kwargs)
            for item in results:
                if item["cb_name"] != self.truth["cb_name"]:
                    item["success"] = False
                    item["chi_square"] = float("nan")
                    poisoned["cb_name"] = item["cb_name"]
                    break
            return results

        with mock.patch.object(gse, "screen_breaker_flips", diverging):
            context = self.providers.get_topology_context(_state(self.scenario))
        self.assertIn("cb_name", poisoned)
        json.dumps(context, allow_nan=False)  # no NaN reaches the export
        by_name = {item["cb_name"]: item for item in context["breaker_findings"]}
        bad = by_name[poisoned["cb_name"]]
        self.assertFalse(bad["flip_estimate_converged"])
        self.assertIsNone(bad["gse_chi_square_after_flip"])
        self.assertIsNone(bad["gse_progress_after_flip"])
        self.assertFalse(bad["flip_explains_substation_measurements"])
        supported = {action["arguments"]["cb_name"] for action in context["supported_corrections"]}
        self.assertNotIn(poisoned["cb_name"], supported)
        self.assertIn(self.truth["cb_name"], supported)
        good = by_name[self.truth["cb_name"]]
        self.assertTrue(good["flip_estimate_converged"])
        self.assertTrue(math.isfinite(good["gse_chi_square_after_flip"]))

    def test_breaker_correction_derives_case_and_records_the_switch(self) -> None:
        state = _state(self.scenario)
        result = self.providers.correct_topology(
            state,
            {
                "tool": "correct_topology",
                "arguments": {
                    "state_id": "episode:s0",
                    "cb_name": self.truth["cb_name"],
                    "status": 0,
                    "line_index": self.truth["line_index1"],
                },
            },
        )
        self.assertNotIn("execution_status", result)
        self.assertEqual(result["evidence_source"], "deployment_correction:breaker_status")
        self.assertEqual(result["breaker_effect"], "dangling_line_terminal")
        self.assertEqual(result["line_index"], self.truth["line_index1"])
        updates = result["modification"]["metadata_updates"]
        self.assertEqual(updates["reported_breaker_status"][self.truth["cb_name"]], "open")
        self.assertEqual(updates["last_topology_correction"]["cb_name"], self.truth["cb_name"])
        # The derived operator case verifies clean and the target evidence
        # records the breaker identity.
        candidate = {
            **state,
            "state_id": "episode:s1",
            "status": "candidate",
            "case": result["modification"]["case"],
            "metadata": {**state["metadata"], **updates},
            "source_action": {
                "tool": "correct_topology",
                "arguments": {
                    "state_id": "episode:s0",
                    "cb_name": self.truth["cb_name"],
                    "status": 0,
                    "line_index": self.truth["line_index1"],
                },
            },
        }
        verification = self.providers.run_wls(candidate)
        self.assertNotIn("execution_status", verification)
        self.assertTrue(verification["target_fixed"])
        self.assertTrue(verification["topology_target_breaker_matches_requested"])
        self.assertEqual(verification["topology_target_breaker"], self.truth["cb_name"])
        self.assertTrue(verification["no_material_anomaly_remaining"])

    def test_breaker_correction_refuses_unrepresentable_or_inconsistent_requests(self) -> None:
        state = _state(self.scenario)

        def attempt(**arguments):
            return self.providers.correct_topology(
                state,
                {"tool": "correct_topology", "arguments": {"state_id": "episode:s0", **arguments}},
            )

        # A bus split is rendered, not refused: one more bus, re-projected vector.
        split = attempt(cb_name="CB_6_B1_B2", status=0)
        self.assertNotIn("execution_status", split)
        self.assertEqual(split["breaker_effect"], "bus_split")
        self.assertEqual(split["operator_bus_count"], 15)
        self.assertEqual(len(split["modification"]["measurements"]), 125)
        island = attempt(cb_name="CB_5_I_B1", status=0)
        self.assertEqual(island["error_code"], "topology_correction_unsupported_effect")
        self.assertEqual(island["breaker_effect"], "unsupplied_island")
        equivalent = attempt(cb_name="CB_5_L51_B1", status=0)
        self.assertEqual(equivalent["error_code"], "topology_correction_unsupported_effect")
        self.assertEqual(equivalent["breaker_effect"], "equivalent")
        inconsistent = attempt(cb_name=self.truth["cb_name"], status=0, line_index=99)
        self.assertEqual(inconsistent["error_code"], "topology_correction_inconsistent_target")
        unknown = attempt(cb_name="CB_NOPE", status=0)
        self.assertEqual(unknown["error_code"], "topology_correction_unknown_breaker")
        unchanged = attempt(cb_name=self.truth["cb_name"], status=1)
        self.assertEqual(unchanged["error_code"], "topology_correction_no_change")
        without_binding = self.providers.correct_topology(
            {**state, "metadata": {}},
            {
                "tool": "correct_topology",
                "arguments": {"state_id": "episode:s0", "cb_name": self.truth["cb_name"], "status": 0},
            },
        )
        self.assertEqual(without_binding["error_code"], "topology_correction_breaker_unsupported")

    def test_wrong_breaker_on_the_same_line_is_not_a_fix(self) -> None:
        # Another breaker whose flip would isolate the same line terminal
        # produces the same operator case, but the private truth is retired
        # only by the breaker that actually changed state.
        from psse_env.private_target_matching import action_targets_private_fault

        right = normalize_action(
            {
                "tool": "correct_topology",
                "arguments": {
                    "state_id": "episode:s0",
                    "cb_name": self.truth["cb_name"],
                    "status": 0,
                    "line_index": self.truth["line_index1"],
                },
            }
        )
        wrong = normalize_action(
            {
                "tool": "correct_topology",
                "arguments": {
                    "state_id": "episode:s0",
                    "cb_name": "CB_OTHER",
                    "status": 0,
                    "line_index": self.truth["line_index1"],
                },
            }
        )
        self.assertTrue(action_targets_private_fault(right, self.truth))
        self.assertFalse(action_targets_private_fault(wrong, self.truth))

    def test_expert_requests_context_then_fixes_the_named_breaker(self) -> None:
        from psse_env.oracle import ExpertPolicyOracle
        from psse_env.transactional_env import TransactionalPSSEEnv

        env = TransactionalPSSEEnv(
            **self.providers.env_kwargs(), production_dataset_mode=True, max_steps=18
        )
        oracle = ExpertPolicyOracle(process_oracle=env.process_oracle)
        env.reset(self.scenario)
        executed = []
        for _ in range(18):
            if env.is_terminal():
                break
            actions = oracle.next_actions(env.get_oracle_state(env.history), env.history)
            self.assertTrue(actions, f"expert stalled after {executed}")
            _, output = env.step(actions[0])
            executed.append((actions[0], output))
        self.assertTrue(env.is_terminal())
        tools = [action["tool"] for action, _ in executed]
        self.assertIn("get_topology_context", tools)
        context_output = next(
            output for action, output in executed if action["tool"] == "get_topology_context"
        )
        self.assertTrue(context_output["tool_metrics"]["substation_measurements_requested"])
        corrections = [
            action["arguments"]
            for action, output in executed
            if action["tool"] == "correct_topology" and output["execution_status"] == "success"
        ]
        self.assertTrue(corrections)
        self.assertEqual(corrections[-1]["cb_name"], self.truth["cb_name"])
        self.assertEqual(int(corrections[-1]["line_index"]), int(self.truth["line_index1"]))
        self.assertFalse(env.get_oracle_state().true_topology_errors)


if __name__ == "__main__":
    unittest.main()


class BusSplitTopologyScenarioTests(unittest.TestCase):
    """A breaker whose true state splits a bus: the operator model gains a bus."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.generator = Round0ScenarioGenerator(seed=31, topology_effects=("bus_split",))
        cls.scenario = cls.generator.build({"topology": 1})[0]
        cls.truth = cls.scenario["true_topology_errors"][0]
        cls.providers = MatpowerDeploymentProviders(chi2_alpha=0.01)

    def test_split_root_declares_a_structural_truth_and_a_rendered_clean_case(self) -> None:
        self.assertEqual(self.truth["physical_effect"], "bus_split")
        self.assertNotIn("branch_row0", self.truth)
        self.assertEqual(self.truth["operator_bus_count_after_fix"], 15)
        self.assertEqual(len(self.scenario["measurements"]), 122)
        self.assertEqual(len(self.scenario["clean_measurements"]), 125)
        self.assertEqual(self.scenario["metadata"]["operator_layout"]["bus_count"], 14)
        self.assertEqual(self.scenario["topology_ranking"]["true_breaker_rank"], 1)

    def test_context_offers_the_split_breaker_without_a_line(self) -> None:
        context = self.providers.get_topology_context(_state(self.scenario))
        self.assertEqual(context["route_status"], "actionable")
        finding = context["breaker_findings"][0]
        self.assertEqual(finding["cb_name"], self.truth["cb_name"])
        self.assertEqual(finding["bus_branch_effect"], "bus_split")
        self.assertEqual(finding["affected_planning_buses"], self.truth["affected_planning_buses"])
        self.assertNotIn("line_index", finding)
        arguments = context["supported_corrections"][0]["arguments"]
        self.assertEqual(set(arguments), {"state_id", "cb_name", "status"})
        self.assertEqual(arguments["cb_name"], self.truth["cb_name"])

    def test_split_correction_re_renders_the_operator_model(self) -> None:
        state = _state(self.scenario)
        action = {
            "tool": "correct_topology",
            "arguments": {"state_id": "episode:s0", "cb_name": self.truth["cb_name"], "status": 0},
        }
        result = self.providers.correct_topology(state, action)
        self.assertNotIn("execution_status", result)
        self.assertEqual(result["breaker_effect"], "bus_split")
        self.assertEqual(result["operator_bus_count"], 15)
        modification = result["modification"]
        self.assertEqual(len(modification["measurements"]), 125)
        updates = modification["metadata_updates"]
        self.assertTrue(updates["last_topology_correction"]["operator_layout_changed"])
        self.assertEqual(updates["last_topology_correction"]["derived_case"], modification["case"])
        candidate = {
            **state,
            "state_id": "episode:s1",
            "status": "candidate",
            "case": modification["case"],
            "measurements": modification["measurements"],
            "metadata": {**state["metadata"], **updates},
            "source_action": action,
        }
        verification = self.providers.run_wls(candidate)
        self.assertNotIn("execution_status", verification)
        self.assertTrue(verification["target_fixed"])
        self.assertEqual(verification["target_metric_kind"], "breaker_status_mismatch")
        self.assertTrue(verification["topology_target_breaker_matches_requested"])
        self.assertEqual(verification["topology_target_effect"], "bus_split")
        self.assertTrue(verification["no_material_anomaly_remaining"])
        # A line index is meaningless for a split and is refused.
        wrong = self.providers.correct_topology(
            state,
            {
                "tool": "correct_topology",
                "arguments": {"state_id": "episode:s0", "cb_name": self.truth["cb_name"], "status": 0, "line_index": 1},
            },
        )
        self.assertEqual(wrong["error_code"], "topology_correction_inconsistent_target")

    def test_expert_fixes_the_split_and_the_release_audit_passes(self) -> None:
        from psse_env.examples.generate_round0_aggregate import audit_episode_against_truth
        from psse_env.oracle import ExpertPolicyOracle
        from psse_env.transactional_env import TransactionalPSSEEnv

        env = TransactionalPSSEEnv(
            **self.providers.env_kwargs(), production_dataset_mode=True, max_steps=18
        )
        oracle = ExpertPolicyOracle(process_oracle=env.process_oracle)
        env.reset(self.scenario)
        executed = []
        for _ in range(18):
            if env.is_terminal():
                break
            actions = oracle.next_actions(env.get_oracle_state(env.history), env.history)
            self.assertTrue(actions, f"expert stalled after {executed}")
            _, output = env.step(actions[0])
            executed.append((actions[0], output))
        self.assertTrue(env.is_terminal())
        corrections = [
            action["arguments"]
            for action, output in executed
            if action["tool"] == "correct_topology" and output["execution_status"] == "success"
        ]
        self.assertEqual(corrections[-1]["cb_name"], self.truth["cb_name"])
        self.assertFalse(env.get_oracle_state().true_topology_errors)
        final = env.current_state()
        active = env.store.get_state(str(final["active_state_id"]))
        self.assertEqual(len(active["measurements"]), 125)
        audit = audit_episode_against_truth(
            self.scenario,
            final,
            terminal=True,
            terminal_outcome=env.terminal_outcome,
            active_physical_state=active,
            remaining_truth=None,
        )
        self.assertEqual(audit["problems"], [])


class MergeTopologyScenarioTests(unittest.TestCase):
    """A breaker in the shared 10/14 yard that is truly closed while reported open:
    buses 10 and 14 are one electrical bus and the operator model loses a bus.

    On the operator's 14-bus model a merge is residual-dominant, so the default
    admission (branch-dominant WLS evidence) rejects every merge draw; these
    tests switch that gate off to exercise the breaker-level machinery.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.generator = Round0ScenarioGenerator(
            seed=31, topology_effects=("merge",), require_branch_dominant_topology=False
        )
        cls.scenario = cls.generator.build({"topology": 1})[0]
        cls.truth = cls.scenario["true_topology_errors"][0]
        cls.providers = MatpowerDeploymentProviders(chi2_alpha=0.01)

    def test_merge_root_declares_a_structural_truth(self) -> None:
        self.assertEqual(self.truth["physical_effect"], "merge")
        self.assertEqual(self.truth["expected_status"], 1)
        self.assertEqual(self.truth["affected_planning_buses"], [10, 14])
        self.assertEqual(self.truth["operator_bus_count_after_fix"], 13)
        self.assertNotIn("branch_row0", self.truth)
        self.assertEqual(len(self.scenario["clean_measurements"]), 3 * 13 + 80)
        self.assertEqual(self.scenario["metadata"]["reported_breaker_status"][self.truth["cb_name"]], "open")
        self.assertEqual(self.scenario["topology_ranking"]["true_breaker_rank"], 1)

    def test_context_offers_closing_the_merge_breaker(self) -> None:
        context = self.providers.get_topology_context(_state(self.scenario))
        self.assertEqual(context["route_status"], "actionable")
        finding = context["breaker_findings"][0]
        self.assertEqual(finding["cb_name"], self.truth["cb_name"])
        self.assertEqual(finding["bus_branch_effect"], "merge")
        self.assertEqual(finding["proposed_status"], "closed")
        arguments = context["supported_corrections"][0]["arguments"]
        self.assertEqual(set(arguments), {"state_id", "cb_name", "status"})
        self.assertEqual(arguments["status"], 1)

    def test_merge_correction_renders_thirteen_buses_and_verifies(self) -> None:
        state = _state(self.scenario)
        action = {
            "tool": "correct_topology",
            "arguments": {"state_id": "episode:s0", "cb_name": self.truth["cb_name"], "status": 1},
        }
        result = self.providers.correct_topology(state, action)
        self.assertNotIn("execution_status", result)
        self.assertEqual(result["breaker_effect"], "merge")
        self.assertEqual(result["operator_bus_count"], 13)
        modification = result["modification"]
        self.assertEqual(len(modification["measurements"]), 3 * 13 + 80)
        layout = modification["metadata_updates"]["operator_layout"]
        self.assertEqual(layout["main_section_by_bus"]["14"], layout["main_section_by_bus"]["10"])
        candidate = {
            **state,
            "state_id": "episode:s1",
            "status": "candidate",
            "case": modification["case"],
            "measurements": modification["measurements"],
            "metadata": {**state["metadata"], **modification["metadata_updates"]},
            "source_action": action,
        }
        verification = self.providers.run_wls(candidate)
        self.assertNotIn("execution_status", verification)
        self.assertTrue(verification["target_fixed"])
        self.assertEqual(verification["topology_target_effect"], "merge")
        self.assertTrue(verification["no_material_anomaly_remaining"])

    def test_default_admission_rejects_merges_as_residual_dominant(self) -> None:
        generator = Round0ScenarioGenerator(seed=31, topology_effects=("merge",))
        self.assertEqual(generator.build({"topology": 1}), [])
        reasons = {record["reason"] for record in generator.skipped}
        self.assertIn("topology_root_not_branch_dominant", reasons)

    def test_expert_fixes_the_merge_through_the_breaker_route(self) -> None:
        # The operator-level route reaches the breaker fix; because the merge is
        # residual-dominant on the 14-bus model the expert may try a meter
        # correction first, which is why merges are not in the default mix.
        from psse_env.oracle import ExpertPolicyOracle
        from psse_env.transactional_env import TransactionalPSSEEnv

        env = TransactionalPSSEEnv(
            **self.providers.env_kwargs(), production_dataset_mode=True, max_steps=18
        )
        oracle = ExpertPolicyOracle(process_oracle=env.process_oracle)
        env.reset(self.scenario)
        executed = []
        for _ in range(18):
            if env.is_terminal():
                break
            actions = oracle.next_actions(env.get_oracle_state(env.history), env.history)
            self.assertTrue(actions, f"expert stalled after {executed}")
            _, output = env.step(actions[0])
            executed.append((actions[0], output))
        self.assertTrue(env.is_terminal())
        corrections = [
            action["arguments"]
            for action, output in executed
            if action["tool"] == "correct_topology" and output["execution_status"] == "success"
        ]
        self.assertEqual(corrections[-1]["cb_name"], self.truth["cb_name"])
        self.assertEqual(corrections[-1]["status"], 1)
        self.assertFalse(env.get_oracle_state().true_topology_errors)
        final = env.current_state()
        active = env.store.get_state(str(final["active_state_id"]))
        self.assertEqual(len(active["measurements"]), 3 * 13 + 80)
        self.assertEqual(
            active["metadata"]["reported_breaker_status"][self.truth["cb_name"]], "closed"
        )
