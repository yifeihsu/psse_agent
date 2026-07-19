from __future__ import annotations

import json
import unittest

from psse_env.actions import MACRO_ACTIONS
from psse_env.dagger.dataset_builder import (
    examples_to_chat_sft,
    validate_tool_schemas,
)
from psse_env.dagger.policy_adapter import LocalAliasPolicyAdapter
from psse_env.dagger.protocol_bridge import (
    CANONICAL_TO_INTERNAL_TOOL,
    INTERNAL_TO_CANONICAL_TOOL,
    canonical_to_internal_action,
    canonical_tool_names,
    internal_to_canonical_action,
    unified_tool_schemas,
)


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
    preferred_action: dict | None = None,
) -> dict:
    observation = _observation(episode)
    return {
        "example_id": f"dagger_iter0_{episode}_step0",
        "scenario_id": episode.removesuffix("_episode0"),
        "root_scenario_id": episode.removesuffix("_episode0"),
        "policy_observation": observation,
        "history_window": [],
        "preferred_action": preferred_action
        or {
            "tool": "run_wls",
            "arguments": {"state_id": observation["active_state_id"]},
        },
        "labels": {},
    }


class UnifiedSchemaTests(unittest.TestCase):
    def test_unified_registry_is_portable_and_covers_both_surfaces(self) -> None:
        schemas = unified_tool_schemas()
        validate_tool_schemas(schemas)
        names = {schema["function"]["name"] for schema in schemas}
        # Canonical production tools, including the specialized diagnostics.
        for name in (
            "wls_from_path",
            "get_harmonic_context",
            "run_hse_from_path",
            "run_three_phase_nlm_from_path",
            "estimate_hif_location_magnitude_from_path",
            "estimate_hif_location_magnitude_multiscan_from_path",
        ):
            self.assertIn(name, names)
        # Transactional recovery tools preserved from the controller surface.
        for name in ("commit_state", "rollback_state", "finalize_diagnosis"):
            self.assertIn(name, names)
        # Every internal macro action must be expressible on the unified surface.
        for internal, canonical in INTERNAL_TO_CANONICAL_TOOL.items():
            self.assertIn(canonical, names, f"{internal} has no canonical schema")

    def test_canonical_wls_schema_is_unmodified(self) -> None:
        schemas = {s["function"]["name"]: s for s in unified_tool_schemas()}
        wls = schemas["wls_from_path"]["function"]["parameters"]
        self.assertEqual(sorted(wls["properties"]), ["case_path"])
        self.assertEqual(wls["required"], ["case_path"])


class ActionMappingTests(unittest.TestCase):
    def test_every_macro_action_maps_and_round_trips_by_name(self) -> None:
        self.assertEqual(set(INTERNAL_TO_CANONICAL_TOOL), set(MACRO_ACTIONS))
        for internal, canonical in INTERNAL_TO_CANONICAL_TOOL.items():
            self.assertEqual(CANONICAL_TO_INTERNAL_TOOL[canonical], internal)

    def test_run_wls_maps_state_reference(self) -> None:
        action = internal_to_canonical_action(
            {"tool": "run_wls", "arguments": {"state_id": "active"}}
        )
        self.assertEqual(action, {"tool": "wls_from_path", "arguments": {"case_path": "active"}})
        back = canonical_to_internal_action(action)
        self.assertEqual(back, {"tool": "run_wls", "arguments": {"state_id": "active"}})

    def test_measurement_correction_drops_values_and_keeps_targets(self) -> None:
        action = internal_to_canonical_action(
            {
                "tool": "correct_measurements",
                "arguments": {
                    "state_id": "active",
                    "measurement_updates": {"7": 1.02, "3": 0.98},
                },
            }
        )
        self.assertEqual(action["tool"], "correct_measurements_from_path")
        self.assertEqual(action["arguments"], {"case_path": "active", "suspect_group": [3, 7]})
        back = canonical_to_internal_action(action)
        self.assertEqual(back["tool"], "correct_measurements")
        self.assertEqual(back["arguments"], {"state_id": "active", "suspect_group": [3, 7]})

    def test_measurement_correction_without_targets_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "measurement_updates or suspect_group"):
            internal_to_canonical_action(
                {"tool": "correct_measurements", "arguments": {"state_id": "active"}}
            )

    def test_parameter_correction_normalizes_branch_targets(self) -> None:
        by_row0 = internal_to_canonical_action(
            {
                "tool": "correct_parameters",
                "arguments": {"state_id": "active", "branch_row0": 10, "field": "x", "value": 0.2},
            }
        )
        self.assertEqual(
            by_row0["arguments"], {"case_path": "active", "line_index": 11}
        )
        by_index1 = internal_to_canonical_action(
            {
                "tool": "correct_parameters",
                "arguments": {"state_id": "active", "line_index1": 11},
            }
        )
        self.assertEqual(
            by_index1["arguments"], {"case_path": "active", "line_index": 11}
        )
        by_branch_id = internal_to_canonical_action(
            {
                "tool": "correct_parameters",
                "arguments": {"state_id": "active", "branch_id": "L2", "value": 0.2},
            }
        )
        self.assertEqual(
            by_branch_id["arguments"], {"case_path": "active", "branch_id": "L2"}
        )

    def test_topology_correction_maps_status_to_boolean_and_back(self) -> None:
        action = internal_to_canonical_action(
            {
                "tool": "correct_topology",
                "arguments": {"state_id": "active", "cb_name": "CB_4_5", "status": 0},
            }
        )
        self.assertEqual(
            action["arguments"],
            {"case_path": "active", "cb_name": "CB_4_5", "desired_status": False},
        )
        back = canonical_to_internal_action(action)
        self.assertEqual(
            back["arguments"], {"state_id": "active", "cb_name": "CB_4_5", "status": 0}
        )

    def test_commit_and_rollback_reference_the_candidate_case(self) -> None:
        for tool in ("commit_state", "rollback_state"):
            action = internal_to_canonical_action(
                {"tool": tool, "arguments": {"candidate_state_id": "candidate"}}
            )
            self.assertEqual(action["arguments"], {"case_path": "candidate"})
            back = canonical_to_internal_action(action)
            self.assertEqual(back["arguments"], {"candidate_state_id": "candidate"})

    def test_verification_snapshot_defaults_to_open_candidate(self) -> None:
        action = internal_to_canonical_action(
            {"tool": "verify_candidate", "arguments": {"state_id": "candidate"}}
        )
        self.assertEqual(action["tool"], "get_verification_snapshot")
        self.assertEqual(action["arguments"], {"case_path": "candidate"})
        stage_only = canonical_to_internal_action(
            {
                "tool": "get_verification_snapshot",
                "arguments": {"stage": "post_measurement_correction"},
            }
        )
        self.assertEqual(stage_only["tool"], "verify_candidate")
        self.assertEqual(stage_only["arguments"]["state_id"], "candidate")

    def test_specialized_diagnostics_map_state_references(self) -> None:
        internal = {
            "tool": "estimate_hif_location_magnitude_from_path",
            "arguments": {"state_id": "active", "candidate_branch_row0": 12},
        }
        canonical = internal_to_canonical_action(internal)
        self.assertEqual(
            canonical["arguments"], {"case_path": "active", "candidate_branch_row0": 12}
        )
        self.assertEqual(canonical_to_internal_action(canonical), internal)
        for tool in ("get_harmonic_context", "run_hse_from_path", "run_three_phase_nlm_from_path"):
            round_trip = canonical_to_internal_action(
                internal_to_canonical_action({"tool": tool, "arguments": {"state_id": "active"}})
            )
            self.assertEqual(round_trip["arguments"], {"state_id": "active"})

    def test_multiscan_hif_binds_scan_window_to_state(self) -> None:
        internal = {
            "tool": "estimate_hif_location_magnitude_multiscan_from_path",
            "arguments": {"state_id": "active", "candidate_branch_row0": 7},
        }
        canonical = internal_to_canonical_action(internal)
        self.assertEqual(
            canonical["arguments"],
            {"scan_window_path": "active", "candidate_branch_row0": 7},
        )
        # A stray canonical case_path is dropped in favor of the window binding.
        canonical["arguments"]["case_path"] = "case14"
        self.assertEqual(canonical_to_internal_action(canonical), internal)

    def test_unknown_tool_fails_closed_on_export(self) -> None:
        with self.assertRaisesRegex(ValueError, "No canonical mapping"):
            internal_to_canonical_action({"tool": "made_up_tool", "arguments": {}})


class CanonicalExportTests(unittest.TestCase):
    def test_export_emits_canonical_target_and_unified_tools(self) -> None:
        row = examples_to_chat_sft([_example()], protocol="canonical")[0]
        self.assertEqual(row["metadata"]["protocol"], "canonical")
        tool_names = {tool["function"]["name"] for tool in row["tools"]}
        self.assertEqual(tool_names, canonical_tool_names())
        call = row["messages"][2]["tool_calls"][0]["function"]
        self.assertEqual(call["name"], "wls_from_path")
        self.assertEqual(call["arguments"], {"case_path": "active"})
        # Controller identifiers must never be model-visible.
        visible = json.dumps(row["messages"], sort_keys=True)
        self.assertNotIn("case14_measurement_seed42_episode0", visible)
        self.assertNotIn("state_id", json.dumps(call))

    def test_export_correction_target_uses_canonical_arguments(self) -> None:
        example = _example()
        example["preferred_action"] = {
            "tool": "correct_measurements",
            "arguments": {
                "state_id": example["policy_observation"]["active_state_id"],
                "measurement_updates": {5: 1.01},
            },
        }
        row = examples_to_chat_sft([example], protocol="canonical")[0]
        call = row["messages"][2]["tool_calls"][0]["function"]
        self.assertEqual(call["name"], "correct_measurements_from_path")
        self.assertEqual(call["arguments"], {"case_path": "active", "suspect_group": [5]})

    def test_export_supports_specialized_diagnostic_targets(self) -> None:
        example = _example()
        example["preferred_action"] = {
            "tool": "estimate_hif_location_magnitude_from_path",
            "arguments": {
                "state_id": example["policy_observation"]["active_state_id"],
                "candidate_branch_row0": 12,
                "candidate_phase": "B",
            },
        }
        row = examples_to_chat_sft([example], protocol="canonical")[0]
        call = row["messages"][2]["tool_calls"][0]["function"]
        self.assertEqual(call["name"], "estimate_hif_location_magnitude_from_path")
        self.assertEqual(
            call["arguments"],
            {"case_path": "active", "candidate_branch_row0": 12, "candidate_phase": "B"},
        )

    def test_controller_protocol_remains_the_default(self) -> None:
        row = examples_to_chat_sft([_example()])[0]
        self.assertEqual(row["metadata"]["protocol"], "controller")
        call = row["messages"][2]["tool_calls"][0]["function"]
        self.assertEqual(call["name"], "run_wls")
        self.assertEqual(call["arguments"], {"state_id": "active"})

    def test_invalid_protocol_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "protocol must be one of"):
            examples_to_chat_sft([_example()], protocol="deployment")


class CanonicalAdapterTests(unittest.TestCase):
    def test_canonical_model_call_binds_to_controller_state(self) -> None:
        observation = _observation()

        def model_policy(model_observation: dict) -> dict:
            self.assertEqual(model_observation["active_state_id"], "active")
            return {
                "tool": "wls_from_path",
                "arguments": {"case_path": model_observation["active_state_id"]},
            }

        adapter = LocalAliasPolicyAdapter(model_policy, protocol="canonical")
        bound = adapter.act(observation)
        self.assertEqual(bound["tool"], "run_wls")
        self.assertEqual(bound["arguments"], {"state_id": observation["active_state_id"]})

    def test_adapter_rejects_unknown_protocol(self) -> None:
        with self.assertRaisesRegex(ValueError, "protocol must be one of"):
            LocalAliasPolicyAdapter(lambda observation: observation, protocol="prod")


if __name__ == "__main__":
    unittest.main()
