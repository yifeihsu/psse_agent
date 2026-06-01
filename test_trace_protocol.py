import json
import unittest

from Transmission.build_sft_traces import (
    apply_measurement_corrections_to_snapshot,
    build_verification_summary,
    choose_measurement_suspect_group,
    correction_family_order,
    explicit_snapshot_compatible_with_current_z,
    get_explicit_verification_snapshot,
    rejection_reason,
    verification_z_obs_from_snapshot,
)
from Transmission.generate_multi_error_measurements import (
    _project_measurement_label_to_snapshot,
    combo_requires_structural_coupling,
    coupling_metadata,
)
from trace_protocol import (
    DECISION_SCHEMA_TEXT,
    chi2_threshold,
    extract_conversation_context,
    hydrate_tool_arguments,
    resolve_case_path_alias,
    summarize_tool_result_for_conversation,
)


def make_initial_user_message() -> dict:
    payload = {
        "case_path": "case14",
        "z_obs": [1.1234567, 2.2345678, 3.3456789],
        "index_map": {"Vm": [0, 1], "Pinj": [1, 2], "Qinj": [2, 3]},
        "branch_info": [{"from_bus": 1, "to_bus": 2}, {"from_bus": 2, "to_bus": 3}],
        "meta_hint": {"nb": 5, "nl": 2, "case": "case14"},
    }
    return {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}


class TraceProtocolTests(unittest.TestCase):
    def test_chi2_threshold_matches_ieee14_reference(self) -> None:
        self.assertAlmostEqual(chi2_threshold(95), 118.7516, places=3)

    def test_schema_describes_scalar_error_family_and_index_bases(self) -> None:
        self.assertIsInstance(DECISION_SCHEMA_TEXT["verdict"]["error_family"], str)
        self.assertIn("scalar enum", DECISION_SCHEMA_TEXT["verdict"]["error_family"])
        self.assertIn("error_families", DECISION_SCHEMA_TEXT["verdict"])
        self.assertIn("suspect_locations", DECISION_SCHEMA_TEXT)
        self.assertIn("applied_tools", DECISION_SCHEMA_TEXT["action"])
        self.assertIn("first_applied_tool", DECISION_SCHEMA_TEXT["action"])
        self.assertIn("last_applied_tool", DECISION_SCHEMA_TEXT["action"])
        self.assertIn("correction_steps", DECISION_SCHEMA_TEXT["action"])
        self.assertIn("0-based", DECISION_SCHEMA_TEXT["evidence"]["top_residuals"][0]["index0"])
        self.assertIn("applied_tool", DECISION_SCHEMA_TEXT["action"])
        self.assertIn("1-based", DECISION_SCHEMA_TEXT["action"]["arguments_hint"])
        self.assertIn("object or null", DECISION_SCHEMA_TEXT["action"]["verification_summary"])

    def test_measurement_suspect_group_prefers_gold_label_indices(self) -> None:
        rec = {"label": {"indices": [2, 4], "channel": "Vm"}}
        idx_map = {"Vm": slice(0, 5)}
        tool_payload = {"r": [0.1, 0.2, 0.3, 30.0, 0.4]}

        self.assertEqual(
            choose_measurement_suspect_group(rec, idx_map, tool_payload),
            [2, 4],
        )

    def test_apply_measurement_corrections_updates_all_suspect_indices(self) -> None:
        z_obs = [1.0, 2.0, 3.0, 4.0]
        corr_payload = {
            "corrected_measurements": [
                {"index0": 0, "corrected": 10.0},
                {"index0": 2, "corrected": 30.0},
                {"index0": 3, "corrected": 40.0},
            ]
        }

        self.assertEqual(
            apply_measurement_corrections_to_snapshot(z_obs, corr_payload, [0, 2]),
            [10.0, 2.0, 30.0, 4.0],
        )

    def test_verification_summary_compares_against_previous_step_payload(self) -> None:
        meta = {"branch_info": []}
        idx_map = {"Vm": slice(0, 1)}
        post_payload = {"global_residual_sum": 200.0, "global_residual_threshold": 100.0}

        improved = build_verification_summary(
            post_payload,
            meta,
            idx_map,
            pre_action_payload={"global_residual_sum": 300.0, "global_residual_threshold": 100.0},
        )
        not_improved = build_verification_summary(
            post_payload,
            meta,
            idx_map,
            pre_action_payload={"global_residual_sum": 150.0, "global_residual_threshold": 100.0},
        )

        self.assertTrue(improved["post_action_improved"])
        self.assertFalse(not_improved["post_action_improved"])

    def test_parameter_topology_combo_is_marked_curriculum_only(self) -> None:
        self.assertTrue(combo_requires_structural_coupling(["parameter_error", "topology_error"]))
        metadata = coupling_metadata(["measurement_error", "parameter_error", "topology_error"])

        self.assertEqual(metadata["coupling_mode"], "curriculum_independent_components")
        self.assertFalse(metadata["physically_coupled"])

    def test_correction_family_order_is_structural_first_by_default(self) -> None:
        rec = {
            "scenario": "multi_error",
            "label": {
                "error_families": ["measurement_error", "parameter_error", "topology_error", "harmonic_anomaly"]
            },
        }

        self.assertEqual(
            correction_family_order(rec, {}, {}, {}),
            ["topology_error", "parameter_error", "measurement_error", "harmonic_anomaly"],
        )
        self.assertEqual(
            correction_family_order(rec, {}, {}, {}, policy="measurement_first"),
            ["measurement_error", "parameter_error", "topology_error", "harmonic_anomaly"],
        )

    def test_explicit_structural_snapshot_can_preserve_current_z_obs(self) -> None:
        rec = {
            "verification_snapshots": {
                "post_topology_correction": {
                    "case_path": "corrected_topology.m",
                    "z_obs_policy": "preserve_current_z_obs",
                    "remaining_families": ["measurement_error"],
                }
            }
        }
        snapshot = get_explicit_verification_snapshot(rec, "post_topology_correction")

        self.assertIsNotNone(snapshot)
        self.assertEqual(verification_z_obs_from_snapshot(snapshot, [1.0, 2.0]), [1.0, 2.0])

    def test_explicit_snapshot_rejects_stale_materialized_vector_length(self) -> None:
        snapshot = {"case_path": "corrected_topology.m", "z_obs": [1.0, 2.0, 3.0]}

        self.assertFalse(explicit_snapshot_compatible_with_current_z(snapshot, [1.0, 2.0]))
        self.assertTrue(
            explicit_snapshot_compatible_with_current_z(
                {"case_path": "corrected_topology.m", "z_obs_policy": "preserve_current_z_obs"},
                [1.0, 2.0],
            )
        )

    def test_measurement_label_projection_preserves_channel_offset_on_compacted_topology_vector(self) -> None:
        source_idx_map = {"Pf": slice(42, 62)}
        target_idx_map = {"Pf": slice(42, 61)}
        label = {
            "error_type": "measurement_error",
            "channel": "Pf",
            "index": 47,
            "amplitude": 0.5,
        }
        z_obs = [0.0] * 118

        projected = _project_measurement_label_to_snapshot(z_obs, label, source_idx_map, target_idx_map)

        self.assertEqual(projected[47], 0.5)
        self.assertEqual(sum(projected), 0.5)

    def test_hydrate_tool_arguments_uses_visible_user_payloads(self) -> None:
        messages = [
            {"role": "system", "content": "system"},
            make_initial_user_message(),
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "case_path": "case14",
                        "z_scans": [[1.0, 2.0], [1.1, 2.1]],
                        "initial_states": [[0.9, 1.9], [1.0, 2.0]],
                    },
                    ensure_ascii=False,
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "case_path": "case14",
                        "harmonic_measurements": [{"bus": 3, "order": 5, "vmag": 0.12}],
                        "harmonic_orders": [5],
                    },
                    ensure_ascii=False,
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "case_path": "case14",
                        "breaker_context": {"cb_name": "CB-12", "desired_status": False},
                    },
                    ensure_ascii=False,
                ),
            },
        ]

        wls_args, wls_notes = hydrate_tool_arguments("wls_from_path", {"case_path": "case14"}, messages)
        self.assertEqual(wls_args["z"], [1.123457, 2.234568, 3.345679])
        self.assertIn("hydrated_wls_from_path_z_from_user", wls_notes)

        param_args, param_notes = hydrate_tool_arguments(
            "correct_parameters_from_path",
            {"case_path": "case14", "line_index": 2},
            messages,
        )
        self.assertEqual(param_args["z_scans"], [[1.0, 2.0], [1.1, 2.1]])
        self.assertEqual(param_args["initial_states"], [[0.9, 1.9], [1.0, 2.0]])
        self.assertIn("hydrated_correct_parameters_z_scans", param_notes)
        self.assertIn("hydrated_correct_parameters_initial_states", param_notes)

        hse_args, hse_notes = hydrate_tool_arguments("run_hse_from_path", {"case_path": "case14"}, messages)
        self.assertEqual(hse_args["harmonic_orders"], [5])
        self.assertEqual(hse_args["harmonic_measurements"][0]["bus"], 3)
        self.assertIn("hydrated_hse_measurements", hse_notes)

        topo_args, topo_notes = hydrate_tool_arguments(
            "correct_topology_from_path",
            {"case_path": "case14"},
            messages,
        )
        self.assertEqual(topo_args["cb_name"], "CB-12")
        self.assertFalse(topo_args["desired_status"])
        self.assertIn("hydrated_topology_cb_name", topo_notes)

    def test_compact_wls_summary_is_significance_gated_and_smaller(self) -> None:
        messages = [make_initial_user_message()]
        meta, index_map = extract_conversation_context(messages)
        raw_payload = {
            "success": True,
            "r": [0.2] * 120 + [3.8, -4.2],
            "lambdaN": [0.1, 0.2, 4.6, -4.9] + [0.1] * 36,
            "z_est": [1.0] * 122,
            "Omega_diag": [0.5] * 122,
            "Omega": [[0.0] * 40 for _ in range(40)],
            "global_residual_sum": 25.43219,
        }

        compact = summarize_tool_result_for_conversation("wls_from_path", raw_payload, meta, index_map)

        self.assertEqual(compact["global_metrics"]["global_residual_sum"], 25.4322)
        self.assertIsNotNone(compact["global_metrics"]["global_residual_threshold"])
        self.assertIsNotNone(compact["global_metrics"]["global_residual_ratio"])
        self.assertEqual(len(compact["top_residuals"]), 2)
        self.assertEqual(len(compact["top_lagrange"]), 2)
        self.assertEqual(compact["top_residuals"][0]["channel"], "unknown")
        self.assertEqual(compact["top_lagrange"][0]["from_bus"], 2)
        self.assertEqual(compact["top_lagrange"][0]["to_bus"], 3)

        raw_len = len(json.dumps(raw_payload, ensure_ascii=False))
        compact_len = len(json.dumps(compact, ensure_ascii=False))
        self.assertLess(compact_len, raw_len * 0.2)

    def test_helper_context_hydration_and_alias_resolution(self) -> None:
        messages = [make_initial_user_message()]
        hidden_context = {
            "case_aliases": {
                "case14::parameter_case::abc12345": "D:/hidden/case_param_err_1.m",
                "case14::measurement_verify::deadbeef": "case14",
                "case14::parameter_verify::beaded00": "case14",
                "case14::topology_verify::feedface": "D:/hidden/corrected_topology_case.m",
            },
            "parameter_context": {
                "case_path": "case14::parameter_case::abc12345",
                "z_scans": [[1.0, 2.0], [1.1, 2.1]],
                "initial_states": [[0.9, 1.9], [1.0, 2.0]],
                "suspect_line": {"line_row0": 1, "from_bus": 2, "to_bus": 3},
            },
            "snapshot_context": {
                "case_path": "case14::measurement_verify::deadbeef",
                "z_obs": [9.8765432, 8.7654321, 7.654321],
                "stage": "post_parameter_correction",
            },
        }

        param_args, param_notes = hydrate_tool_arguments(
            "correct_parameters_from_path",
            {"line_index": 2},
            messages,
            hidden_context=hidden_context,
        )
        self.assertEqual(param_args["case_path"], "case14::parameter_case::abc12345")
        self.assertEqual(param_args["z_scans"], [[1.0, 2.0], [1.1, 2.1]])
        self.assertIn("hydrated_correct_parameters_case_path", param_notes)

        wls_args, wls_notes = hydrate_tool_arguments(
            "wls_from_path",
            {"case_path": "case14"},
            messages,
            hidden_context=hidden_context,
        )
        self.assertEqual(wls_args["case_path"], "case14::measurement_verify::deadbeef")
        self.assertEqual(wls_args["z"], [9.876543, 8.765432, 7.654321])
        self.assertIn("hydrated_wls_from_path_z_from_user", wls_notes)

        resolved = resolve_case_path_alias("case14::parameter_case::abc12345", hidden_context)
        self.assertEqual(resolved, "D:/hidden/case_param_err_1.m")
        self.assertEqual(resolve_case_path_alias(wls_args["case_path"], hidden_context), "case14")
        self.assertEqual(
            resolve_case_path_alias("case14::topology_verify::feedface", hidden_context),
            "D:/hidden/corrected_topology_case.m",
        )

        compact = summarize_tool_result_for_conversation(
            "get_parameter_context",
            hidden_context["parameter_context"],
            {},
            {},
        )
        self.assertEqual(compact["case_path"], "case14::parameter_case::abc12345")
        self.assertEqual(compact["scans"], 2)
        self.assertEqual(compact["measurement_vector_length"], 2)

    def test_verification_alias_contract_covers_snapshot_and_backend_patterns(self) -> None:
        messages = [make_initial_user_message()]
        hidden_context = {
            "case_aliases": {
                "case14::measurement_verify::abc11111": "case14",
                "case14::parameter_verify::abc22222": "case14",
                "case14::topology_verify::abc33333": "out_measurements_balanced/models_topology/case_topology_corrected.m",
            },
            "snapshot_context": {
                "case_path": "case14::parameter_verify::abc22222",
                "z_obs": [4.4444444, 5.5555555, 6.6666666],
                "stage": "post_parameter_correction",
            },
        }

        hydrated_args, _ = hydrate_tool_arguments(
            "wls_from_path",
            {"case_path": "case14::parameter_verify::abc22222"},
            messages,
            hidden_context=hidden_context,
        )
        self.assertEqual(hydrated_args["case_path"], "case14::parameter_verify::abc22222")
        self.assertEqual(hydrated_args["z"], [4.444444, 5.555555, 6.666667])
        self.assertEqual(resolve_case_path_alias(hydrated_args["case_path"], hidden_context), "case14")
        self.assertEqual(
            resolve_case_path_alias("case14::topology_verify::abc33333", hidden_context),
            "out_measurements_balanced/models_topology/case_topology_corrected.m",
        )

    def test_build_verification_summary_reports_execution_improvement_and_resolution(self) -> None:
        messages = [make_initial_user_message()]
        meta, index_map = extract_conversation_context(messages)
        pre_action = {
            "success": True,
            "global_residual_sum": 150.0,
            "global_residual_threshold": 100.0,
        }
        post_action = {
            "success": True,
            "global_residual_sum": 80.0,
            "global_residual_threshold": 100.0,
        }

        summary = build_verification_summary(
            post_action,
            meta,
            index_map,
            pre_action_payload=pre_action,
        )

        self.assertTrue(summary["post_action_executed"])
        self.assertTrue(summary["post_action_improved"])
        self.assertTrue(summary["post_action_resolved"])
        self.assertEqual(summary["post_action_global_residual_ratio"], 0.8)

    def test_rejection_reason_enforces_strict_clean_boundary(self) -> None:
        borderline_no_error = {
            "verdict": {"has_error": False, "error_family": "no_error", "confidence": 0.99},
            "evidence": {
                "global_metrics": {
                    "global_residual_sum": 100.0,
                    "global_residual_threshold": 100.0,
                    "global_residual_ratio": 1.0,
                },
                "top_residuals": [],
                "top_lagrange": [],
            },
            "suspect_location": {"domain": "none", "details": {}},
            "action": {"verification_summary": None, "request_more_data": False},
            "summary": "clean",
        }
        self.assertEqual(rejection_reason(borderline_no_error), "borderline_no_error")

        missing_verification = {
            "verdict": {"has_error": True, "error_family": "measurement_error", "confidence": 0.95},
            "evidence": {
                "global_metrics": {
                    "global_residual_sum": 120.0,
                    "global_residual_threshold": 100.0,
                    "global_residual_ratio": 1.2,
                },
                "top_residuals": [{"index0": 7, "channel": "Pf", "channel_offset": 0, "value": 4.1}],
                "top_lagrange": [],
            },
            "suspect_location": {"domain": "measurement", "details": {}},
            "action": {"verification_summary": None, "request_more_data": False},
            "summary": "bad measurement",
        }
        self.assertEqual(rejection_reason(missing_verification), "measurement_error_missing_verification")

    def test_rejection_reason_accepts_backward_compatible_multi_error_target(self) -> None:
        multi_target = {
            "verdict": {
                "has_error": True,
                "error_family": "measurement_error",
                "error_families": ["measurement_error", "parameter_error"],
                "confidence": 0.96,
            },
            "evidence": {
                "global_metrics": {
                    "global_residual_sum": 180.0,
                    "global_residual_threshold": 100.0,
                    "global_residual_ratio": 1.8,
                },
                "top_residuals": [{"index0": 7, "channel": "Pf", "channel_offset": 0, "value": 4.1}],
                "top_lagrange": [{"lambda_index0": 2, "line_row0": 1, "value": 5.2}],
            },
            "suspect_location": {"domain": "measurement", "details": {"index0": 7}},
            "suspect_locations": [
                {"domain": "measurement", "details": {"index0": 7}},
                {"domain": "parameter", "details": {"line_row0": 1}},
            ],
            "action": {
                "applied_tool": "correct_parameters_from_path",
                "applied_tools": ["correct_measurements_from_path", "correct_parameters_from_path"],
                "request_more_data": False,
                "verification_summary": None,
            },
            "summary": "multi",
        }
        self.assertIsNone(rejection_reason(multi_target))

        multi_target["action"]["applied_tools"] = ["correct_measurements_from_path"]
        self.assertEqual(rejection_reason(multi_target), "multi_error_missing_applied_tools")


if __name__ == "__main__":
    unittest.main()
