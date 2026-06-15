import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from Transmission.build_sft_traces import (
    BuilderConfig,
    HARDENING_TEMPLATES,
    apply_measurement_corrections_to_snapshot,
    build_final_target,
    build_tool_precondition_hardening_trace,
    build_verification_summary,
    choose_measurement_suspect_group,
    correction_family_order,
    explicit_snapshot_compatible_with_current_z,
    get_explicit_verification_snapshot,
    measurement_indices_for_index_space,
    measurement_correction_policy_payload,
    multi_error_semantic_rejection_reason,
    rejection_reason,
    trace_metadata_for_record,
    verification_z_obs_from_snapshot,
)
from Transmission.generate_multi_error_measurements import (
    _project_measurement_label_to_snapshot,
    choose_base_family,
    combo_requires_structural_coupling,
    coupling_metadata,
    make_multi_error_record,
)
from Transmission.generate_measurements import (
    load_case,
    make_index_map,
)
from trace_protocol import (
    DECISION_SCHEMA_TEXT,
    SCADA_HARMONIC_SYSTEM_PROMPT,
    canonical_tool_schemas,
    chi2_threshold,
    extract_conversation_context,
    hydrate_tool_arguments,
    resolve_case_path_alias,
    scada_harmonic_tool_schemas,
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
    def _mock_builder_config(self) -> BuilderConfig:
        return BuilderConfig(
            samples_path=Path("samples.jsonl"),
            meta_path=Path("meta.json"),
            imbalance_samples_path=None,
            imbalance_meta_path=None,
            hif_samples_path=None,
            hif_meta_path=None,
            hardening_source_samples_path=None,
            case_name=None,
            endpoint="http://localhost:3929/tools",
            out_path=Path("out.jsonl"),
            analysis_out_path=None,
            mock=True,
            seed=7,
            add_no_error=0,
            with_correction=True,
            corr_max_iter=2,
            corr_tol=1e-3,
            allow_hif_metadata_fallback=False,
            hardening_examples=0,
        )

    def test_chi2_threshold_matches_ieee14_reference(self) -> None:
        self.assertAlmostEqual(chi2_threshold(95), 118.7516, places=3)

    def test_schema_describes_scalar_error_family_and_index_bases(self) -> None:
        self.assertIsInstance(DECISION_SCHEMA_TEXT["verdict"]["error_family"], str)
        self.assertIn("scalar enum", DECISION_SCHEMA_TEXT["verdict"]["error_family"])
        self.assertIn("error_families", DECISION_SCHEMA_TEXT["verdict"])
        self.assertIn("suspect_locations", DECISION_SCHEMA_TEXT)
        self.assertIn("evidence_by_stage", DECISION_SCHEMA_TEXT)
        self.assertIn("applied_tools", DECISION_SCHEMA_TEXT["action"])
        self.assertIn("high_impedance_fault", DECISION_SCHEMA_TEXT["verdict"]["error_family"])
        self.assertIn("top_hif_groups", DECISION_SCHEMA_TEXT["evidence"])
        self.assertIn("first_applied_tool", DECISION_SCHEMA_TEXT["action"])
        self.assertIn("last_applied_tool", DECISION_SCHEMA_TEXT["action"])
        self.assertIn("verification_summaries", DECISION_SCHEMA_TEXT["action"])
        self.assertIn("last_verified_summary", DECISION_SCHEMA_TEXT["action"])
        self.assertIn("tool_steps", DECISION_SCHEMA_TEXT["action"])
        self.assertIn("correction_steps", DECISION_SCHEMA_TEXT["action"])
        self.assertIn("0-based", DECISION_SCHEMA_TEXT["evidence"]["top_residuals"][0]["index0"])
        self.assertIn("applied_tool", DECISION_SCHEMA_TEXT["action"])
        self.assertIn("1-based", DECISION_SCHEMA_TEXT["action"]["arguments_hint"])
        self.assertIn("object or null", DECISION_SCHEMA_TEXT["action"]["verification_summary"])
        self.assertIn("diagnosis_status", DECISION_SCHEMA_TEXT["action"])
        self.assertIn("curriculum_only", DECISION_SCHEMA_TEXT["action"]["diagnosis_status"])
        self.assertIn("sequence_only", DECISION_SCHEMA_TEXT["action"]["diagnosis_status"])
        self.assertIn("remaining_candidate_families", DECISION_SCHEMA_TEXT["action"])

    def test_scada_harmonic_prompt_scope_excludes_hif_and_imbalance_tools(self) -> None:
        tool_names = {tool["function"]["name"] for tool in scada_harmonic_tool_schemas()}

        self.assertIn("run_hse_from_path", tool_names)
        self.assertNotIn("run_three_phase_nlm_from_path", tool_names)
        self.assertNotIn("high_impedance_fault", SCADA_HARMONIC_SYSTEM_PROMPT)
        self.assertNotIn("three_phase_imbalance", SCADA_HARMONIC_SYSTEM_PROMPT)
        self.assertNotIn("top_hif_groups", SCADA_HARMONIC_SYSTEM_PROMPT)

    def test_verification_snapshot_schema_is_stage_only(self) -> None:
        schemas = canonical_tool_schemas()
        verify_schema = next(
            tool for tool in schemas if tool.get("function", {}).get("name") == "get_verification_snapshot"
        )
        params = verify_schema["function"]["parameters"]

        self.assertEqual(params.get("required"), [])
        self.assertIn("stage", params.get("properties", {}))
        self.assertNotIn("case_path", params.get("properties", {}))

    def test_measurement_suspect_group_prefers_gold_label_indices(self) -> None:
        rec = {"label": {"indices": [2, 4], "channel": "Vm"}}
        idx_map = {"Vm": slice(0, 5)}
        tool_payload = {"r": [0.1, 0.2, 0.3, 30.0, 0.4]}

        self.assertEqual(
            choose_measurement_suspect_group(rec, idx_map, tool_payload),
            [2, 4],
        )

    def test_measurement_suspect_group_uses_projected_index_space_after_topology(self) -> None:
        rec = {
            "label": {
                "index": 77,
                "channel": "Qf",
                "index_spaces": {
                    "post_topology_correction_indices0": [76],
                    "post_topology_correction": {
                        "index_space": "post_topology_correction",
                        "original_indices0": [77],
                        "current_indices0": [76],
                    },
                },
            }
        }
        idx_map = {"Qf": slice(62, 82)}
        tool_payload = {"r": [0.0] * 122}

        self.assertEqual(measurement_indices_for_index_space(rec, "post_topology_correction"), [76])
        self.assertEqual(
            choose_measurement_suspect_group(
                rec,
                idx_map,
                tool_payload,
                prefer_label=False,
                index_space="post_topology_correction",
            ),
            [76],
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
        weakly_improved = build_verification_summary(
            {"global_residual_sum": 292.0, "global_residual_threshold": 100.0},
            meta,
            idx_map,
            pre_action_payload={"global_residual_sum": 300.0, "global_residual_threshold": 100.0},
        )

        self.assertTrue(improved["post_action_improved"])
        self.assertFalse(not_improved["post_action_improved"])
        self.assertFalse(weakly_improved["post_action_improved"])

    def test_parameter_topology_combo_is_marked_curriculum_only(self) -> None:
        self.assertTrue(combo_requires_structural_coupling(["parameter_error", "topology_error"]))
        metadata = coupling_metadata(["measurement_error", "parameter_error", "topology_error"])

        self.assertEqual(metadata["coupling_mode"], "curriculum_independent_components")
        self.assertFalse(metadata["physically_coupled"])
        self.assertEqual(choose_base_family(["parameter_error", "topology_error"]), "topology_error")
        self.assertEqual(choose_base_family(["measurement_error", "parameter_error"]), "parameter_error")
        self.assertEqual(choose_base_family(["measurement_error", "harmonic_anomaly"]), "clean_scada")

    def test_parameter_topology_trace_metadata_marks_sequence_only_curriculum(self) -> None:
        rec = {
            "scenario": "multi_error",
            "label": {
                "combo": "parameter+topology",
                "error_families": ["parameter_error", "topology_error"],
                "coupling_mode": "curriculum_independent_components",
                "physically_coupled": False,
                "errors": [
                    {"error_type": "parameter_error", "line_row": 5},
                    {"error_type": "topology_error", "cb_name": "CB_2"},
                ],
            },
        }

        metadata = trace_metadata_for_record(rec)

        self.assertEqual(metadata["trace_kind"], "multi_error")
        self.assertEqual(metadata["trace_type"], "representative_multi_error")
        self.assertFalse(metadata["physically_coupled"])
        self.assertTrue(metadata["curriculum_only"])
        self.assertEqual(metadata["sequence_only_families"], ["parameter_error"])

    def test_physical_parameter_topology_record_has_verified_snapshots(self) -> None:
        ppc_base = load_case("14")
        idx_map = make_index_map(ppc_base["bus"].shape[0], ppc_base["branch"].shape[0])
        with tempfile.TemporaryDirectory() as tmp:
            with contextlib.redirect_stdout(io.StringIO()):
                rec = make_multi_error_record(
                    np.random.default_rng(20260615),
                    ppc_base,
                    idx_map,
                    Path(tmp),
                    ["parameter_error", "topology_error"],
                    scans=2,
                    load_scale_min=0.8,
                    load_scale_max=1.25,
                    mode="physical",
                )

        self.assertIsNotNone(rec)
        assert rec is not None
        self.assertTrue(rec["label"]["physically_coupled"])
        self.assertEqual(
            rec["label"]["coupling_mode"],
            "physical_parameter_on_topology_corrected_model",
        )
        self.assertEqual(rec["op_point"]["base_family"], "coupled_parameter_topology")
        snapshots = rec["verification_snapshots"]
        self.assertIn("post_topology_correction", snapshots)
        self.assertIn("post_parameter_correction", snapshots)
        self.assertEqual(
            snapshots["post_topology_correction"]["remaining_families"],
            ["parameter_error"],
        )
        self.assertEqual(snapshots["post_parameter_correction"]["remaining_families"], [])
        self.assertEqual(
            len(snapshots["post_topology_correction"]["z_obs"]),
            len(rec["z_scans"][0]),
        )
        parameter_error = next(
            item for item in rec["label"]["errors"] if item["error_type"] == "parameter_error"
        )
        self.assertEqual(parameter_error["index_space"], "post_topology_correction")

    def test_physical_parameter_topology_final_target_has_staged_evidence(self) -> None:
        messages = [make_initial_user_message()]
        meta, idx_map = extract_conversation_context(messages)
        initial_branch_info = list(meta["branch_info"])
        active_branch_info = list(initial_branch_info)
        active_branch_info[1] = {"i": 1, "from_bus": 2, "to_bus": 4, "is_line": True}
        rec = {
            "scenario": "multi_error",
            "label": {
                "error_families": ["parameter_error", "topology_error"],
                "physically_coupled": True,
                "errors": [
                    {
                        "error_type": "parameter_error",
                        "line_row": 1,
                        "from_bus": 2,
                        "to_bus": 4,
                        "subtype": "RX",
                    },
                    {
                        "error_type": "topology_error",
                        "substation": 3,
                        "cb_name": "CB_3_L34_B2",
                        "old_status": "closed",
                        "new_status": "open",
                    },
                ],
            },
        }
        primary_wls = {
            "success": True,
            "r": [0.0] * 122,
            "lambdaN": [0.0, 0.0, 4.5, 6.0],
            "global_residual_sum": 200.0,
            "branch_info": initial_branch_info,
        }
        post_topology_wls = {
            "success": True,
            "r": [0.0] * 122,
            "lambdaN": [0.0, 0.0, 5.0, 8.0],
            "global_residual_sum": 150.0,
            "branch_info": active_branch_info,
        }
        post_parameter_wls = {
            "success": True,
            "r": [0.0] * 122,
            "lambdaN": [0.0] * 4,
            "global_residual_sum": 20.0,
            "branch_info": active_branch_info,
        }

        final = build_final_target(
            rec,
            meta,
            idx_map,
            primary_wls,
            verification_payloads={
                "topology_error": post_topology_wls,
                "parameter_error": post_parameter_wls,
            },
            verification_pre_payloads={
                "topology_error": primary_wls,
                "parameter_error": post_topology_wls,
            },
            applied_tools=["correct_topology_from_path", "correct_parameters_from_path"],
            tool_steps=[
                {"family": "topology_error", "tool": "correct_topology_from_path", "verification_policy": "verified_wls"},
                {"family": "parameter_error", "tool": "correct_parameters_from_path", "verification_policy": "verified_wls"},
            ],
            correction_steps=[],
        )

        self.assertEqual(final["evidence"]["top_lagrange"][0]["from_bus"], 2)
        self.assertEqual(final["evidence"]["top_lagrange"][0]["to_bus"], 3)
        staged = final["evidence_by_stage"]
        self.assertEqual(staged["initial"]["top_lagrange"][0]["to_bus"], 3)
        self.assertEqual(staged["post_topology_correction"]["top_lagrange"][0]["line_row0"], 1)
        self.assertEqual(staged["post_topology_correction"]["top_lagrange"][0]["from_bus"], 2)
        self.assertEqual(staged["post_topology_correction"]["top_lagrange"][0]["to_bus"], 4)
        self.assertEqual(staged["post_parameter_correction"]["top_lagrange"], [])
        parameter_location = next(
            loc for loc in final["suspect_locations"] if loc.get("domain") == "parameter"
        )
        self.assertEqual(parameter_location["details"]["to_bus"], 4)

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

    def test_tool_precondition_hardening_templates_recover_via_measurement_correction(self) -> None:
        meta = {
            "case": "case14",
            "nb": 2,
            "nl": 1,
            "baseMVA": 100.0,
            "branch_info": [{"from_bus": 1, "to_bus": 2}],
            "index_map": {
                "Vm": [0, 2],
                "Va": [2, 4],
                "Pinj": [4, 6],
                "Qinj": [6, 8],
                "Pf": [8, 9],
                "Qf": [9, 10],
                "Pt": [10, 10],
                "Qt": [10, 10],
            },
        }
        idx_map = {k: slice(v[0], v[1]) for k, v in meta["index_map"].items()}
        rec = {
            "id": "meas_hardening_source",
            "scenario": "measurement_error",
            "z_obs": [1.0, 9.0, 1.0, 1.0, 0.1, 0.2, 0.1, 0.2, 0.3, 0.4],
            "z_true": [1.0, 1.1, 1.0, 1.0, 0.1, 0.2, 0.1, 0.2, 0.3, 0.4],
            "label": {
                "error_type": "measurement_error",
                "channel": "Vm",
                "index": 1,
                "subtype": "single_gross_outlier",
            },
        }

        for template in HARDENING_TEMPLATES:
            with self.subTest(template=template):
                row, reason = build_tool_precondition_hardening_trace(
                    config=self._mock_builder_config(),
                    rec=rec,
                    template=template,
                    sid=f"{rec['id']}::{template}",
                    meta=meta,
                    idx_map=idx_map,
                    base_case_backend="case14",
                    base_case_visible="case14",
                    rng_np=np.random.default_rng(123),
                )

                self.assertIsNone(reason)
                self.assertIsNotNone(row)
                messages = row["messages"]
                tool_names = [
                    call["function"]["name"]
                    for message in messages
                    for call in message.get("tool_calls", [])
                ]
                self.assertEqual(tool_names[0], "wls_from_path")
                self.assertIn("correct_measurements_from_path", tool_names)
                self.assertIn("get_verification_snapshot", tool_names)
                self.assertNotIn("correct_parameters_from_path", tool_names)
                self.assertNotIn("run_hse_from_path", tool_names)

                if template == "parameter_helper_unavailable":
                    self.assertIn("get_parameter_context", tool_names)
                elif template == "harmonic_helper_unavailable":
                    self.assertIn("get_harmonic_context", tool_names)
                else:
                    self.assertEqual(tool_names.count("get_verification_snapshot"), 2)

                missing_context_tools = [
                    message
                    for message in messages
                    if message.get("role") == "tool"
                    and "Missing runtime context" in str(message.get("content", ""))
                ]
                self.assertEqual(len(missing_context_tools), 1)
                missing_summary = json.loads(missing_context_tools[0]["content"])
                self.assertFalse(missing_summary["success"])
                self.assertEqual(missing_summary["allowed_next_tools"], ["correct_measurements_from_path"])
                self.assertEqual(missing_summary["available_context_tools"], ["correct_measurements_from_path"])

                masked_bad_calls = [
                    message
                    for message in messages
                    if message.get("role") == "assistant"
                    and message.get("trainable") is False
                    and message.get("train_on_assistant") is False
                    and message.get("loss_mask") is False
                    and message.get("tool_calls")
                ]
                self.assertEqual(len(masked_bad_calls), 1)
                self.assertIn(
                    masked_bad_calls[0]["tool_calls"][0]["function"]["name"],
                    {"get_parameter_context", "get_harmonic_context", "get_verification_snapshot"},
                )

                final = json.loads(messages[-1]["content"])
                self.assertEqual(final["verdict"]["error_family"], "measurement_error")
                self.assertEqual(final["verdict"]["error_families"], ["measurement_error"])
                self.assertEqual(len(final["suspect_locations"]), 1)
                self.assertEqual(final["action"]["applied_tool"], "correct_measurements_from_path")
                self.assertEqual(final["action"]["applied_tools"], ["correct_measurements_from_path"])
                self.assertIn("measurement_error", final["action"]["arguments_hint"])
                self.assertIn("diagnosis_status", final["action"])
                self.assertIn("remaining_candidate_families", final["action"])
                self.assertTrue(final["action"]["measurement_correction_policy"]["allowed"])
                self.assertEqual(
                    final["action"]["measurement_correction_policy"]["residual_pattern"],
                    "localized",
                )
                self.assertEqual(final["action"]["tool_steps"][0]["family"], "measurement_error")
                self.assertIn("pre_global_residual_ratio", final["action"]["tool_steps"][0])
                self.assertIn("post_global_residual_ratio", final["action"]["tool_steps"][0])
                self.assertIn("post_action_improved", final["action"]["tool_steps"][0])
                self.assertIn("post_action_resolved", final["action"]["tool_steps"][0])
                self.assertEqual(final["action"]["correction_steps"][0]["family"], "measurement_error")
                self.assertIn("pre_global_residual_ratio", final["action"]["correction_steps"][0])
                self.assertEqual(row["trace_metadata"]["trace_kind"], "tool_precondition_hardening")
                self.assertEqual(row["trace_type"], "hardening_recovery")
                self.assertEqual(row["trace_metadata"]["trace_type"], "hardening_recovery")
                self.assertIn("tools", row)
                self.assertNotIn(
                    "run_three_phase_nlm_from_path",
                    {tool["function"]["name"] for tool in row["tools"]},
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

    def test_wls_summary_prefers_active_case_branch_info(self) -> None:
        messages = [make_initial_user_message()]
        meta, index_map = extract_conversation_context(messages)
        raw_payload = {
            "success": True,
            "r": [0.0] * 122,
            "lambdaN": [0.1, 0.2, 4.6, -4.9] + [0.1] * 36,
            "global_residual_sum": 25.0,
            "branch_info": [
                {"i": 0, "from_bus": 10, "to_bus": 11, "is_line": True},
                {"i": 1, "from_bus": 20, "to_bus": 21, "is_line": True},
            ],
        }

        compact = summarize_tool_result_for_conversation("wls_from_path", raw_payload, meta, index_map)

        self.assertEqual(compact["top_lagrange"][0]["line_row0"], 1)
        self.assertEqual(compact["top_lagrange"][0]["from_bus"], 20)
        self.assertEqual(compact["top_lagrange"][0]["to_bus"], 21)

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

    def test_hif_tool_schema_hydration_and_summary(self) -> None:
        schemas = canonical_tool_schemas()
        names = [tool["function"]["name"] for tool in schemas]
        self.assertIn("run_three_phase_nlm_from_path", names)

        messages = [make_initial_user_message()]
        hidden_context = {
            "hif_context": {
                "case_path": "case14",
                "label": {"branch_row0": 2, "dss_element": "Line.2-3"},
                "nlm_diagnostic": {
                    "success": True,
                    "converged": True,
                    "top_hif_groups": [
                        {
                            "rank": 1,
                            "branch_row0": 2,
                            "line_index1": 3,
                            "dss_element": "Line.2-3",
                            "from_bus": 2,
                            "to_bus": 3,
                            "score": 31.25,
                        }
                    ],
                },
            }
        }

        args, notes = hydrate_tool_arguments(
            "run_three_phase_nlm_from_path",
            {"case_path": "case14"},
            messages,
            hidden_context=hidden_context,
        )
        self.assertEqual(args["target_branch_row0"], 2)
        self.assertEqual(args["target_dss_element"], "Line.2-3")
        self.assertIn("hydrated_hif_nlm_diagnostic", notes)

        compact = summarize_tool_result_for_conversation(
            "run_three_phase_nlm_from_path",
            args["nlm_diagnostic"],
            {},
            {},
        )
        self.assertTrue(compact["success"])
        self.assertEqual(compact["top_hif_groups"][0]["branch_row0"], 2)

    def test_hif_final_target_uses_visible_nlm_not_hidden_label(self) -> None:
        meta = {
            "nb": 14,
            "nl": 20,
            "branch_info": [{"from_bus": i + 1, "to_bus": i + 2} for i in range(20)],
        }
        idx_map = {
            "Vm": slice(0, 14),
            "Pinj": slice(14, 28),
            "Qinj": slice(28, 42),
            "Pf": slice(42, 62),
            "Qf": slice(62, 82),
            "Pt": slice(82, 102),
            "Qt": slice(102, 122),
        }
        rec = {
            "scenario": "high_impedance_fault",
            "label": {
                "branch_row0": 2,
                "line_index1": 3,
                "dss_element": "Line.2-3",
                "from_bus": 2,
                "to_bus": 3,
                "phase": "A",
            },
        }
        nlm_payload = {
            "success": True,
            "converged": True,
            "top_hif_groups": [
                {
                    "rank": 1,
                    "branch_row0": 4,
                    "line_index1": 5,
                    "dss_element": "Line.2-5",
                    "from_bus": 2,
                    "to_bus": 5,
                    "score": 19.0,
                }
            ],
        }
        primary_wls = {
            "success": True,
            "r": [0.0] * 122,
            "lambdaN": [],
            "global_residual_sum": 10.0,
            "global_residual_threshold": 118.0,
        }

        final = build_final_target(rec, meta, idx_map, primary_wls, nlm_payload=nlm_payload)
        details = final["suspect_location"]["details"]

        self.assertEqual(details["branch_row0"], 4)
        self.assertEqual(details["dss_element"], "Line.2-5")
        self.assertNotIn("phase", details)

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

    def test_single_error_final_target_includes_family_keyed_verification_summaries(self) -> None:
        meta = {"branch_info": []}
        idx_map = {"Vm": slice(0, 1)}
        rec = {
            "scenario": "measurement_error",
            "label": {"channel": "Vm", "index": 0, "subtype": "single_gross_outlier"},
        }
        primary_wls = {"success": True, "global_residual_sum": 240.0, "global_residual_threshold": 100.0}
        verification_payload = {"success": True, "global_residual_sum": 80.0, "global_residual_threshold": 100.0}

        final = build_final_target(
            rec,
            meta,
            idx_map,
            primary_wls,
            measurement_suspect_group=[0],
            verification_payload=verification_payload,
            correction_tool_name="correct_measurements_from_path",
            applied_tools=["correct_measurements_from_path"],
        )

        action = final["action"]
        self.assertEqual(action["verification_summaries"]["measurement_error"], action["verification_summary"])

    def test_parameter_topology_final_target_marks_curriculum_and_last_verified_summary(self) -> None:
        meta = {"branch_info": []}
        idx_map = {"Vm": slice(0, 1)}
        rec = {
            "scenario": "multi_error",
            "label": {
                "error_families": ["parameter_error", "topology_error"],
                "primary_error_family": "parameter_error",
                "physically_coupled": False,
                "errors": [
                    {"error_type": "parameter_error", "line_row": 5},
                    {"error_type": "topology_error", "cb_name": "CB_2", "old_status": "closed"},
                ],
            },
        }
        primary_wls = {"success": True, "global_residual_sum": 260.0, "global_residual_threshold": 100.0}
        topology_wls = {"success": True, "global_residual_sum": 88.5, "global_residual_threshold": 100.0}

        final = build_final_target(
            rec,
            meta,
            idx_map,
            primary_wls,
            verification_payloads={"topology_error": topology_wls},
            verification_pre_payloads={"topology_error": primary_wls},
            applied_tools=["correct_topology_from_path", "correct_parameters_from_path"],
            correction_steps=[
                {
                    "step": 1,
                    "family": "topology_error",
                    "tool": "correct_topology_from_path",
                    "verification_policy": "verified_wls",
                },
                {
                    "step": 2,
                    "family": "parameter_error",
                    "tool": "correct_parameters_from_path",
                    "verification_policy": "sequence_only",
                },
            ],
            tool_steps=[
                {
                    "step": 1,
                    "family": "topology_error",
                    "tool": "correct_topology_from_path",
                    "verification_policy": "verified_wls",
                },
                {
                    "step": 2,
                    "family": "parameter_error",
                    "tool": "correct_parameters_from_path",
                    "verification_policy": "sequence_only",
                },
            ],
        )

        action = final["action"]
        self.assertEqual(action["diagnosis_status"], "curriculum_only")
        self.assertTrue(action["curriculum_only"])
        self.assertFalse(action["physically_coupled"])
        self.assertEqual(action["sequence_only_families"], ["parameter_error"])
        self.assertIsNone(action["verification_summary"])
        self.assertEqual(action["last_verified_summary"]["family"], "topology_error")
        self.assertEqual(action["last_verified_summary"]["post_action_global_residual_ratio"], 0.885)

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

    def test_multi_error_semantic_rejects_unresolved_measurement_harmonic(self) -> None:
        rec = {
            "scenario": "multi_error",
            "label": {
                "error_families": ["measurement_error", "harmonic_anomaly"],
                "errors": [
                    {"error_type": "measurement_error", "channel": "Pf", "index": 49},
                    {"error_type": "harmonic_anomaly", "source_bus": 14},
                ],
            },
        }
        final = {
            "action": {
                "correction_steps": [
                    {
                        "family": "measurement_error",
                        "pre_global_residual_ratio": 514.0,
                        "post_global_residual_ratio": 512.0,
                    }
                ]
            }
        }

        self.assertEqual(
            multi_error_semantic_rejection_reason(rec, final),
            "measurement_harmonic_unresolved_after_measurement_correction",
        )

    def test_measurement_policy_allows_structural_first_localized_cleanup(self) -> None:
        rec = {
            "scenario": "multi_error",
            "label": {
                "error_families": ["measurement_error", "parameter_error"],
                "errors": [
                    {
                        "error_type": "measurement_error",
                        "subtype": "single_gross_outlier",
                        "channel": "Pf",
                        "index": 49,
                    },
                    {"error_type": "parameter_error", "line_row": 2},
                ],
            },
        }
        action = {
            "applied_tools": ["get_parameter_context", "correct_parameters_from_path", "correct_measurements_from_path"],
            "arguments_hint": {"measurement_error": {"suspect_group": [49]}},
        }

        policy = measurement_correction_policy_payload(
            rec,
            ["measurement_error", "parameter_error"],
            action,
            [49],
        )

        self.assertTrue(policy["allowed"])
        self.assertEqual(policy["residual_pattern"], "localized")
        self.assertTrue(policy["structural_tools_before_measurement"])
        self.assertEqual(policy["structural_checks_completed"], ["parameter_error"])
        self.assertIsNone(multi_error_semantic_rejection_reason(rec, {"action": action}))

    def test_measurement_policy_inherits_action_request_more_data_for_partial_residual(self) -> None:
        rec = {
            "scenario": "multi_error",
            "label": {
                "error_families": ["measurement_error", "parameter_error"],
                "errors": [
                    {
                        "error_type": "measurement_error",
                        "subtype": "single_gross_outlier",
                        "channel": "Pf",
                        "index": 49,
                    },
                    {"error_type": "parameter_error", "line_row": 2},
                ],
            },
        }
        action = {
            "applied_tools": ["correct_parameters_from_path", "correct_measurements_from_path"],
            "request_more_data": True,
            "diagnosis_status": "partial",
        }

        policy = measurement_correction_policy_payload(
            rec,
            ["measurement_error", "parameter_error"],
            action,
            [49],
        )

        self.assertTrue(policy["allowed"])
        self.assertTrue(policy["request_more_data"])
        self.assertIn("final residual remains above threshold", policy["reason"])

    def test_multi_error_semantic_rejects_measurement_before_structural_cleanup(self) -> None:
        rec = {
            "scenario": "multi_error",
            "label": {
                "error_families": ["measurement_error", "parameter_error"],
                "errors": [
                    {"error_type": "measurement_error", "subtype": "single_gross_outlier", "index": 10},
                    {"error_type": "parameter_error", "line_row": 1},
                ],
            },
        }
        final = {
            "action": {
                "applied_tools": ["correct_measurements_from_path", "correct_parameters_from_path"],
                "arguments_hint": {"measurement_error": {"suspect_group": [10]}},
            }
        }

        self.assertEqual(
            multi_error_semantic_rejection_reason(rec, final),
            "measurement_before_structural_correction",
        )

    def test_multi_error_semantic_rejects_distributed_measurement_cleanup(self) -> None:
        rec = {
            "scenario": "multi_error",
            "label": {
                "error_families": ["measurement_error", "harmonic_anomaly"],
                "errors": [
                    {
                        "error_type": "measurement_error",
                        "subtype": "distributed_meter_bias",
                        "indices": [1, 2, 3, 4, 5, 6],
                    },
                    {"error_type": "harmonic_anomaly", "source_bus": 14},
                ],
            },
        }
        final = {
            "action": {
                "applied_tools": ["correct_measurements_from_path", "run_hse_from_path"],
                "arguments_hint": {"measurement_error": {"suspect_group": [1, 2, 3, 4, 5, 6]}},
                "measurement_correction_policy": {
                    "allowed": False,
                    "residual_pattern": "distributed",
                },
            }
        }

        self.assertEqual(
            multi_error_semantic_rejection_reason(rec, final),
            "measurement_distributed_subtype_requires_dedicated_policy",
        )

    def test_multi_error_semantic_requires_topology_projected_measurement_indices(self) -> None:
        rec = {
            "scenario": "multi_error",
            "label": {
                "error_families": ["measurement_error", "topology_error"],
                "errors": [
                    {
                        "error_type": "measurement_error",
                        "channel": "Qf",
                        "index": 77,
                        "index_spaces": {
                            "post_topology_correction_indices0": [76],
                            "post_topology_correction": {
                                "index_space": "post_topology_correction",
                                "original_indices0": [77],
                                "current_indices0": [76],
                            },
                        },
                    },
                    {"error_type": "topology_error", "cb_name": "CB_1_N2_B2"},
                ],
            },
        }
        final = {
            "action": {
                "arguments_hint": {"measurement_error": {"suspect_group": [76]}},
                "correction_steps": [],
            }
        }

        self.assertIsNone(multi_error_semantic_rejection_reason(rec, final))
        final["action"]["arguments_hint"]["measurement_error"]["suspect_group"] = [77]
        self.assertEqual(
            multi_error_semantic_rejection_reason(rec, final),
            "measurement_topology_suspect_group_mismatch",
        )

    def test_multi_error_semantic_marks_curriculum_parameter_topology_sequence_only(self) -> None:
        rec = {
            "scenario": "multi_error",
            "label": {
                "error_families": ["parameter_error", "topology_error"],
                "physically_coupled": False,
                "errors": [
                    {"error_type": "parameter_error", "line_row": 5},
                    {"error_type": "topology_error", "cb_name": "CB_2"},
                ],
            },
        }
        final = {
            "action": {
                "tool_steps": [
                    {
                        "family": "topology_error",
                        "tool": "correct_topology_from_path",
                        "verification_policy": "verified_wls",
                    },
                    {
                        "family": "parameter_error",
                        "tool": "correct_parameters_from_path",
                        "verification_policy": "sequence_only",
                    },
                ],
                "correction_steps": [],
            }
        }

        self.assertIsNone(multi_error_semantic_rejection_reason(rec, final))
        final["action"]["tool_steps"][-1]["verification_policy"] = "verified_wls"
        self.assertEqual(
            multi_error_semantic_rejection_reason(rec, final),
            "parameter_topology_requires_sequence_only_verification_policy",
        )

    def test_multi_error_semantic_requires_verified_physical_parameter_topology(self) -> None:
        rec = {
            "scenario": "multi_error",
            "label": {
                "error_families": ["parameter_error", "topology_error"],
                "physically_coupled": True,
                "errors": [
                    {"error_type": "parameter_error", "line_row": 5},
                    {"error_type": "topology_error", "cb_name": "CB_2"},
                ],
            },
        }
        final = {
            "action": {
                "tool_steps": [
                    {
                        "family": "topology_error",
                        "tool": "correct_topology_from_path",
                        "verification_policy": "verified_wls",
                    },
                    {
                        "family": "parameter_error",
                        "tool": "correct_parameters_from_path",
                        "verification_policy": "verified_wls",
                    },
                ],
                "correction_steps": [],
            }
        }

        self.assertIsNone(multi_error_semantic_rejection_reason(rec, final))
        final["action"]["tool_steps"][-1]["verification_policy"] = "sequence_only"
        self.assertEqual(
            multi_error_semantic_rejection_reason(rec, final),
            "parameter_topology_physical_coupling_rejects_sequence_only",
        )


if __name__ == "__main__":
    unittest.main()
