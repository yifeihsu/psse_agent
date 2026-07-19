from __future__ import annotations

import json
import unittest

from psse_env.actions import MACRO_ACTIONS
from psse_env.dagger.dataset_builder import (
    TOOL_JSON_SCHEMAS,
    bind_controller_action,
    examples_to_chat_sft as _examples_to_chat_sft,
    validate_tool_schemas,
)
from psse_env.dagger.sft_audit import (
    audit_approximate_teacher_realizability,
    audit_chat_sft_rows,
    audit_teacher_realizability,
    canonical_semantic_action,
    policy_observation_hash,
)
from psse_env.dagger.policy_adapter import LocalAliasPolicyAdapter


def examples_to_chat_sft(*args, **kwargs):
    """Controller-surface helper for this legacy-format unit-test module."""
    kwargs.setdefault("protocol", "controller")
    return _examples_to_chat_sft(*args, **kwargs)


def _observation(episode: str = "case14_measurement_seed42_episode0") -> dict:
    return {
        "active_state_id": f"{episode}:s0",
        "candidate_state_id": None,
        "candidate_parent_id": None,
        "episode_id": episode,
        "remaining_budget": 6,
        "history_window": [],
        "unresolved_signatures": [],
        "remaining_anomaly_score": None,
        "no_material_anomaly_remaining": False,
    }


def _example(
    *,
    episode: str = "case14_measurement_seed42_episode0",
    tool: str = "run_wls",
    provenance: dict | None = None,
) -> dict:
    observation = _observation(episode)
    example = {
        "example_id": f"dagger_iter0_{episode}_step0",
        "scenario_id": episode.removesuffix("_episode0"),
        "root_scenario_id": episode.removesuffix("_episode0"),
        "policy_observation": observation,
        "history_window": [],
        "preferred_action": {
            "tool": tool,
            "arguments": {"state_id": observation["active_state_id"]},
        },
        "labels": {},
    }
    if provenance is not None:
        example["semantic_field_provenance"] = provenance
    return example


class NativeToolExportTests(unittest.TestCase):
    def test_full_macro_surface_has_portable_json_schemas(self) -> None:
        validate_tool_schemas(TOOL_JSON_SCHEMAS)
        self.assertEqual(
            {schema["function"]["name"] for schema in TOOL_JSON_SCHEMAS},
            set(MACRO_ACTIONS),
        )
        encoded = json.dumps(TOOL_JSON_SCHEMAS, allow_nan=False)
        self.assertEqual(json.loads(encoded), TOOL_JSON_SCHEMAS)
        for schema in TOOL_JSON_SCHEMAS:
            parameters = schema["function"]["parameters"]
            self.assertEqual(parameters["type"], "object")
            self.assertIsInstance(parameters["properties"], dict)

    def test_every_row_has_schemas_and_dictionary_arguments(self) -> None:
        row = examples_to_chat_sft([_example()])[0]
        self.assertEqual(row["tools"], TOOL_JSON_SCHEMAS)
        arguments = row["messages"][2]["tool_calls"][0]["function"]["arguments"]
        self.assertIsInstance(arguments, dict)
        self.assertEqual(arguments, {"state_id": "active"})
        self.assertTrue(audit_chat_sft_rows([row])["passed"])

    def test_production_dataset_mode_tag_is_preserved_outside_messages(self) -> None:
        example = _example()
        example["dataset_mode"] = "production"
        example["labels"]["dataset_mode"] = "production"
        row = examples_to_chat_sft([example])[0]
        self.assertEqual(row["dataset_mode"], "production")
        self.assertEqual(row["metadata"]["dataset_mode"], "production")
        self.assertNotIn("production", row["messages"][1]["content"])

    def test_explicitly_ineligible_auxiliary_row_fails_closed(self) -> None:
        example = _example()
        example["dataset_source"] = "synthetic_counterfactual"
        example["production_label_eligible"] = False
        with self.assertRaisesRegex(ValueError, "ineligible for production SFT"):
            examples_to_chat_sft([example])
        rows = examples_to_chat_sft(
            [example], allow_ineligible_auxiliary=True
        )
        self.assertEqual(len(rows), 1)

    def test_measurement_update_keys_round_trip_as_strict_json_strings(self) -> None:
        example = _example()
        example["preferred_action"] = {
            "tool": "correct_measurements",
            "arguments": {
                "state_id": example["policy_observation"]["active_state_id"],
                "measurement_updates": {0: 1.0},
            },
        }
        row = examples_to_chat_sft([example])[0]
        arguments = row["messages"][2]["tool_calls"][0]["function"]["arguments"]
        self.assertEqual(arguments["measurement_updates"], {"0": 1.0})
        rebound = bind_controller_action(
            {"tool": "correct_measurements", "arguments": arguments},
            row["metadata"]["controller"]["state_aliases"],
        )
        self.assertEqual(rebound["arguments"]["measurement_updates"], {0: 1.0})

    def test_name_filter_emits_schemas_and_requires_target_schema(self) -> None:
        row = examples_to_chat_sft([_example()], available_tools=["run_wls"])[0]
        self.assertEqual([tool["function"]["name"] for tool in row["tools"]], ["run_wls"])
        with self.assertRaisesRegex(ValueError, "has no schema"):
            examples_to_chat_sft([_example()], available_tools=["get_measurement_context"])

    def test_malformed_custom_schema_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "parameters"):
            examples_to_chat_sft(
                [_example()],
                available_tools=[{"type": "function", "function": {"name": "run_wls"}}],
            )


class ControllerAliasTests(unittest.TestCase):
    def test_full_state_and_scenario_ids_are_not_model_visible(self) -> None:
        example = _example()
        row = examples_to_chat_sft([example])[0]
        visible = json.dumps(row["messages"], sort_keys=True)
        full_state_id = example["policy_observation"]["active_state_id"]
        self.assertNotIn(full_state_id, visible)
        self.assertNotIn(example["scenario_id"], visible)
        user = json.loads(row["messages"][1]["content"])
        self.assertEqual(user["state"]["active_state_id"], "active")
        self.assertEqual(user["state"]["episode_id"], "episode")
        self.assertEqual(row["messages"][2]["tool_calls"][0]["id"], "call_0")
        controller = row["metadata"]["controller"]
        self.assertEqual(controller["state_aliases"]["active"], full_state_id)
        self.assertEqual(controller["scenario_id"], example["scenario_id"])
        self.assertEqual(controller["root_scenario_id"], example["root_scenario_id"])
        self.assertEqual(row["root_scenario_id"], example["root_scenario_id"])

    def test_controller_binds_generated_alias_after_model_output(self) -> None:
        row = examples_to_chat_sft([_example()])[0]
        function = row["messages"][2]["tool_calls"][0]["function"]
        rebound = bind_controller_action(
            {"function": function},
            row["metadata"]["controller"]["state_aliases"],
        )
        self.assertEqual(
            rebound["arguments"]["state_id"],
            "case14_measurement_seed42_episode0:s0",
        )
        with self.assertRaisesRegex(ValueError, "Unknown controller state alias"):
            bind_controller_action(
                {"tool": "run_wls", "arguments": {"state_id": "foreign"}},
                row["metadata"]["controller"]["state_aliases"],
            )

    def test_ids_embedded_in_action_signatures_are_redacted(self) -> None:
        example = _example()
        state_id = example["policy_observation"]["active_state_id"]
        example["policy_observation"]["tried_action_signatures"] = [
            f'run_wls:{{"state_id":"{state_id}"}}'
        ]
        row = examples_to_chat_sft([example])[0]
        visible = row["messages"][1]["content"]
        self.assertNotIn(state_id, visible)
        self.assertIn("active", visible)

    def test_short_episode_id_does_not_corrupt_semantic_strings(self) -> None:
        example = _example(episode="e", tool="verify_candidate")
        row = examples_to_chat_sft([example])
        function = row[0]["messages"][-1]["tool_calls"][0]["function"]
        self.assertEqual(function["name"], "verify_candidate")
        self.assertEqual(function["arguments"], {"state_id": "active"})
        self.assertNotIn("vepisoderify", json.dumps(row[0]["messages"]))

    def test_model_policy_adapter_matches_export_view_and_rebinds_action(self) -> None:
        seen: list[dict] = []

        class ModelPolicy:
            def act(self, observation):
                seen.append(observation)
                return {"tool": "run_wls", "arguments": {"state_id": "active"}}

        raw = _observation()
        raw["semantic_field_provenance"] = {
            "remaining_anomaly_score": "controller_default"
        }
        rebound = LocalAliasPolicyAdapter(ModelPolicy(), protocol="controller").act(raw)
        self.assertEqual(seen[0]["active_state_id"], "active")
        self.assertNotIn("semantic_field_provenance", seen[0])
        self.assertEqual(rebound["arguments"]["state_id"], raw["active_state_id"])

    def test_state_hashes_use_local_equality_preserving_aliases(self) -> None:
        example = _example()
        digest = "a" * 64
        example["policy_observation"]["last_tool_output"] = {
            "execution_status": "success",
            "tool_metrics": {"state_hash": digest, "state_hash_before": digest},
        }
        row = examples_to_chat_sft([example])[0]
        visible = row["messages"][1]["content"]
        self.assertNotIn(digest, visible)
        user = json.loads(visible)
        metrics = user["state"]["last_tool_output"]["observable_metrics"]
        self.assertEqual(metrics["state_hash"], metrics["state_hash_before"])
        self.assertEqual(metrics["state_hash"], "h0")
        self.assertEqual(row["metadata"]["controller"]["hash_aliases"]["h0"], digest)
        self.assertTrue(audit_chat_sft_rows([row])["passed"])

    def test_distinct_state_hash_aliases_have_stable_semantic_order(self) -> None:
        example = _example()
        before = "a" * 64
        after = "b" * 64
        example["policy_observation"]["last_tool_output"] = {
            "execution_status": "success",
            "tool_metrics": {
                # Deliberately reverse source insertion order.  Export order
                # must still be state_hash_before followed by state_hash_after.
                "state_hash_after": after,
                "state_hash_before": before,
            },
        }
        row = examples_to_chat_sft([example])[0]
        aliases = row["metadata"]["controller"]["hash_aliases"]
        self.assertEqual(aliases, {"h0": before, "h1": after})
        metrics = json.loads(row["messages"][1]["content"])["state"][
            "last_tool_output"
        ]["observable_metrics"]
        self.assertEqual(metrics["state_hash_before"], "h0")
        self.assertEqual(metrics["state_hash_after"], "h1")


class HistoryAndProvenanceTests(unittest.TestCase):
    def test_history_has_one_bounded_structured_location(self) -> None:
        example = _example()
        state_id = example["policy_observation"]["active_state_id"]
        example["history_window"] = [
            {
                "state_id": state_id,
                "action": {"tool": "run_wls", "arguments": {"state_id": state_id}},
                "tool_output": {
                    "execution_status": "success",
                    "tool_metrics": {
                        "wls_objective": float(index),
                        "raw_solver_matrix": "x" * 10000,
                    },
                },
            }
            for index in range(20)
        ]
        row = examples_to_chat_sft(
            [example],
            max_history_events=3,
            max_history_chars=1200,
        )[0]
        user = json.loads(row["messages"][1]["content"])
        self.assertNotIn("recent_action_observation_history", user)
        history = user["state"]["history_window"]
        self.assertLessEqual(len(history), 3)
        self.assertLessEqual(len(json.dumps(history, sort_keys=True)), 1200)
        self.assertNotIn("raw_solver_matrix", json.dumps(history))
        self.assertEqual(row["metadata"]["history"]["source_events"], 20)
        self.assertTrue(audit_chat_sft_rows([row])["passed"])

    def test_bounded_context_findings_and_supported_correction_remain_visible(self) -> None:
        example = _example(tool="correct_measurements")
        state_id = example["policy_observation"]["active_state_id"]
        supported = {
            "tool": "correct_measurements",
            "arguments": {
                "state_id": state_id,
                "measurement_updates": {0: 1.0},
            },
        }
        example["preferred_action"] = supported
        example["policy_observation"]["last_tool"] = "get_measurement_context"
        example["policy_observation"]["last_tool_output"] = {
            "execution_status": "success",
            "tool_metrics": {
                "measurement_findings": [
                    {
                        "measurement_index": 0,
                        "observed": 1.2,
                        "reference": 1.0,
                    }
                ],
                "supported_corrections": [supported],
            },
        }
        row = examples_to_chat_sft([example])[0]
        state = json.loads(row["messages"][1]["content"])["state"]
        metrics = state["last_tool_output"]["observable_metrics"]
        self.assertEqual(metrics["measurement_findings"][0]["reference"], 1.0)
        visible_action = metrics["supported_corrections"][0]
        self.assertEqual(visible_action["arguments"]["state_id"], "active")
        self.assertEqual(
            visible_action["arguments"]["measurement_updates"],
            {"0": 1.0},
        )

    def test_hidden_truth_and_nested_oracle_hint_provenance_are_rejected(self) -> None:
        example = _example(provenance={"remaining_anomaly_score": {"source": "hidden_truth"}})
        example["policy_observation"]["remaining_anomaly_score"] = 0.8
        with self.assertRaisesRegex(ValueError, "Privileged provenance"):
            examples_to_chat_sft([example])

        nested = _example(
            provenance={
                "remaining_anomaly_score": {
                    "source": {"source_kind": "oracle_hint"},
                }
            }
        )
        nested["policy_observation"]["remaining_anomaly_score"] = 0.8
        with self.assertRaisesRegex(ValueError, "Privileged provenance"):
            examples_to_chat_sft([nested])

        synthetic = _example(
            provenance={"remaining_anomaly_score": "synthetic_placeholder"}
        )
        synthetic["policy_observation"]["remaining_anomaly_score"] = 0.8
        with self.assertRaisesRegex(ValueError, "Privileged provenance"):
            examples_to_chat_sft([synthetic])

    def test_informative_derived_field_requires_observable_provenance(self) -> None:
        missing = _example()
        missing["policy_observation"]["remaining_anomaly_score"] = 0.8
        with self.assertRaisesRegex(ValueError, "Missing observable provenance"):
            examples_to_chat_sft([missing])

        safe = _example(
            provenance={
                "remaining_anomaly_score": {
                    "source": "observable_provider",
                    "provider": "run_wls",
                }
            }
        )
        safe["policy_observation"]["remaining_anomaly_score"] = 0.8
        row = examples_to_chat_sft([safe])[0]
        user = json.loads(row["messages"][1]["content"])
        self.assertNotIn("semantic_field_provenance", user["state"])
        self.assertEqual(
            row["metadata"]["semantic_field_provenance"]["remaining_anomaly_score"]["source"],
            "observable_provider",
        )


class TeacherRealizabilityAuditTests(unittest.TestCase):
    def test_identifier_only_differences_hash_to_same_observation(self) -> None:
        left = _observation("measurement_template_episode0")
        right = _observation("parameter_template_episode9")
        self.assertEqual(policy_observation_hash(left), policy_observation_hash(right))
        left_action = {"tool": "run_wls", "arguments": {"state_id": left["active_state_id"]}}
        right_action = {"tool": "run_wls", "arguments": {"state_id": right["active_state_id"]}}
        self.assertEqual(
            canonical_semantic_action(left_action, left),
            canonical_semantic_action(right_action, right),
        )

    def test_hidden_truth_teacher_family_conflict_is_reported(self) -> None:
        measurement = _example(
            episode="measurement_template_episode0",
            tool="get_measurement_context",
        )
        parameter = _example(
            episode="parameter_template_episode9",
            tool="get_parameter_context",
        )
        report = audit_teacher_realizability([measurement, parameter])
        self.assertFalse(report["passed"])
        self.assertEqual(report["unique_observations"], 1)
        self.assertEqual(report["conflict_observations"], 1)
        self.assertEqual(report["conflicting_examples"], 2)
        self.assertEqual(report["conflict_rate"], 1.0)
        self.assertEqual(
            {action["tool"] for action in report["conflicts"][0]["semantic_actions"]},
            {"get_measurement_context", "get_parameter_context"},
        )

    def test_nearby_continuous_teacher_conflict_is_reported_approximately(self) -> None:
        measurement = _example(
            episode="measurement_template_episode0",
            tool="get_measurement_context",
        )
        parameter = _example(
            episode="parameter_template_episode9",
            tool="get_parameter_context",
        )
        for example, score in ((measurement, 1.01), (parameter, 1.02)):
            example["policy_observation"]["last_tool"] = "run_wls"
            example["policy_observation"]["last_tool_output"] = {
                "execution_status": "success",
                "tool_metrics": {
                    "max_normalized_residual": score,
                    "chi_square_statistic": 101.0 + score,
                    "chi_square_threshold": 100.0,
                    "wls_summary": {
                        "top_residuals": [
                            {"channel": "Pinj", "index0": 7, "value": score}
                        ]
                    },
                },
            }
        self.assertTrue(audit_teacher_realizability([measurement, parameter])["passed"])
        report = audit_approximate_teacher_realizability(
            [measurement, parameter],
            quantization_bin=0.25,
            conflict_tolerance=0.0,
            nearest_neighbor_tolerance=0.0,
        )
        self.assertFalse(report["passed"])
        self.assertEqual(report["conflict_buckets"], 1)
        self.assertEqual(report["approximate_conflict_rate"], 1.0)
        self.assertEqual(report["nearest_neighbor_action_disagreement_rate"], 1.0)

    def test_nlm_localized_branch_separates_hif_targets_in_approximate_audit(self) -> None:
        examples = []
        for branch in (2, 12):
            example = _example(
                episode=f"hif_branch_{branch}_episode0",
                tool="estimate_hif_location_magnitude_from_path",
            )
            state_id = example["policy_observation"]["active_state_id"]
            example["preferred_action"]["arguments"] = {
                "state_id": state_id,
                "candidate_branch_row0": branch,
            }
            example["history_window"] = [
                {
                    "state_id": state_id,
                    "action": {
                        "tool": "run_three_phase_nlm_from_path",
                        "arguments": {"state_id": state_id},
                    },
                    "tool_output": {
                        "execution_status": "success",
                        "tool_metrics": {
                            "nlm_summary": {
                                "top_hif_groups": [
                                    {"rank": 1, "branch_row0": branch, "score": 0.9}
                                ]
                            }
                        },
                    },
                }
            ]
            examples.append(example)

        report = audit_approximate_teacher_realizability(
            examples,
            conflict_tolerance=0.0,
            nearest_neighbor_tolerance=0.0,
        )

        self.assertTrue(report["passed"])
        self.assertEqual(report["conflict_buckets"], 0)
        self.assertEqual(report["nearest_neighbor_compared_examples"], 0)

    def test_cost_margin_coverage_and_low_margin_rate_are_explicit(self) -> None:
        first = _example(episode="margin_a_episode0")
        second = _example(episode="margin_b_episode0")
        first["cost_margin"] = 0.01
        second["action_costs"] = [{"q_cost": 1.0}, {"q_cost": 1.2}]
        report = audit_approximate_teacher_realizability([first, second])
        self.assertEqual(report["cost_margin_coverage"], 1.0)
        self.assertEqual(report["low_cost_margin_rate"], 0.5)

    def test_verbose_history_and_provenance_do_not_hide_same_visible_conflict(self) -> None:
        measurement = _example(
            episode="measurement_template_episode0",
            tool="get_measurement_context",
        )
        parameter = _example(
            episode="parameter_template_episode9",
            tool="get_parameter_context",
        )
        measurement["policy_observation"]["semantic_field_provenance"] = {
            "remaining_anomaly_score": "controller_default"
        }
        parameter["policy_observation"]["semantic_field_provenance"] = {
            "remaining_anomaly_score": "observable_wls"
        }
        for example, digest in ((measurement, "a" * 64), (parameter, "b" * 64)):
            state_id = example["policy_observation"]["active_state_id"]
            example["history_window"] = [
                {
                    "state_id": state_id,
                    "action": {"tool": "run_wls", "arguments": {"state_id": state_id}},
                    "tool_output": {
                        "execution_status": "success",
                        "tool_metrics": {
                            "wls_objective": 1.0,
                            "state_hash": digest,
                            "raw_solver_blob": digest * 100,
                        },
                    },
                }
            ]
        report = audit_teacher_realizability([measurement, parameter])
        self.assertFalse(report["passed"])
        self.assertEqual(report["unique_observations"], 1)
        self.assertEqual(report["conflict_observations"], 1)

    def test_same_semantic_label_with_different_ids_passes(self) -> None:
        first = _example(episode="template_a_episode0", tool="run_wls")
        second = _example(episode="template_b_episode8", tool="run_wls")
        report = audit_teacher_realizability([first, second])
        self.assertTrue(report["passed"])
        self.assertEqual(report["unique_observations"], 1)
        self.assertEqual(report["conflict_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
