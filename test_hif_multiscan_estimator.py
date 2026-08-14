from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from IEEE_14_OpenDSS.constants import (
    IEEE14_GENERATOR_DISPATCH_KW,
    IEEE14_GENERATOR_VOLTAGE_PU,
    IEEE14_LOAD_BASE_KW,
    IEEE14_OPERATING_POINT_KEYS,
)
from three_phase_nlm.hif_multiscan_estimator import (
    _distinct_parameter_candidates,
    _normalized_joint_residual,
    _observability_payload,
    _parse_scans,
    _with_diagnostic_coverage,
    estimate_hif_location_magnitude_multiscan,
)
from three_phase_nlm.hif_operating_point import canonicalize_ieee14_operating_point
from trace_protocol import hydrate_tool_arguments, summarize_tool_result_for_conversation


class HIFMultiscanEstimatorTests(unittest.TestCase):
    def test_joint_residual_normalization_preserves_duplicate_linear_information(self) -> None:
        block = np.asarray([2.0, -1.0, 0.5])
        single = _normalized_joint_residual([block])
        duplicate = _normalized_joint_residual([block, block])

        self.assertAlmostEqual(float(single @ single), float(duplicate @ duplicate), places=12)

    def test_partial_observability_diagnostic_disables_point_claim(self) -> None:
        payload = _with_diagnostic_coverage(
            {"status": "full_rank_well_conditioned", "parameter_identifiable": True},
            requested_scan_count=3,
            successful_scan_indices=[0, 2],
            failures=[{"scan_index": 1, "error": "did not converge"}],
        )

        self.assertEqual(payload["status"], "diagnostic_partial")
        self.assertFalse(payload["parameter_identifiable"])
        self.assertFalse(payload["diagnostic_complete"])
        self.assertEqual(payload["diagnostic_scan_count"], 2)
        self.assertEqual(payload["diagnostic_failed_scan_count"], 1)

    def test_duplicate_optimizer_solutions_do_not_create_false_top_two_tie(self) -> None:
        candidates = [
            {"alpha": 0.47, "r_hif_pu": 80.0, "phase": "A", "score": 1.0},
            {"alpha": 0.47000001, "r_hif_pu": 80.00001, "phase": "A", "score": 1.0},
            {"alpha": 0.62, "r_hif_pu": 110.0, "phase": "A", "score": 1.4},
        ]

        distinct = _distinct_parameter_candidates(candidates, limit=3)

        self.assertEqual(len(distinct), 2)
        self.assertEqual(distinct[0]["alpha"], 0.47)
        self.assertEqual(distinct[1]["alpha"], 0.62)

    def test_identical_rank_one_scans_do_not_recover_rank(self) -> None:
        single = np.asarray([[4.0, 0.0], [0.0, 0.0]])
        payload = _observability_payload(
            effective_information=2.0 * single,
            per_scan_information=[single, single],
            scan_count=2,
            weighted_residual_norm=1.0,
            residual_reduction=0.8,
            condition_limit=1e6,
            correlation_limit=0.98,
            diagnostic_method="test",
        )

        self.assertEqual(payload["effective_rank"], 1)
        self.assertEqual(payload["best_single_scan_rank"], 1)
        self.assertEqual(payload["rank_gain_vs_best_single_scan"], 0)
        self.assertEqual(payload["status"], "rank_deficient")
        self.assertFalse(payload["parameter_identifiable"])
        self.assertEqual(payload["scan_diversity_score"], 0.0)

    def test_complementary_rank_one_scans_recover_rank(self) -> None:
        alpha_only = np.asarray([[4.0, 0.0], [0.0, 0.0]])
        resistance_only = np.asarray([[0.0, 0.0], [0.0, 3.0]])
        payload = _observability_payload(
            effective_information=alpha_only + resistance_only,
            per_scan_information=[alpha_only, resistance_only],
            scan_count=2,
            weighted_residual_norm=1.0,
            residual_reduction=0.8,
            condition_limit=1e6,
            correlation_limit=0.98,
            diagnostic_method="test",
        )

        self.assertEqual(payload["effective_rank"], 2)
        self.assertEqual(payload["best_single_scan_rank"], 1)
        self.assertEqual(payload["rank_gain_vs_best_single_scan"], 1)
        self.assertEqual(payload["status"], "full_rank_well_conditioned")
        self.assertTrue(payload["parameter_identifiable"])
        self.assertGreater(payload["scan_diversity_score"], 0.5)

    def test_scan_window_rejects_topology_changes(self) -> None:
        scans = [
            {"scan_index": 0, "z_obs": [1.0, 2.0], "op_point": {}, "topology_id": "base"},
            {"scan_index": 1, "z_obs": [1.0, 2.0], "op_point": {}, "topology_id": "changed"},
        ]
        with self.assertRaisesRegex(ValueError, "same topology_id"):
            _parse_scans(scans=scans, scan_window_path=None)

    def test_hydrated_window_sigma_applies_to_every_scan(self) -> None:
        scans = [
            {"scan_index": 0, "z_obs": [1.0, 2.0], "op_point": {}},
            {"scan_index": 1, "z_obs": [1.1, 2.1], "op_point": {}},
        ]

        parsed, _ = _parse_scans(
            scans=scans,
            scan_window_path="bound://hif_window/example",
            default_sigma_z=[0.1, 0.2],
        )

        self.assertTrue(all(np.array_equal(scan.sigma_z, np.asarray([0.1, 0.2])) for scan in parsed))

    @unittest.skipUnless(importlib.util.find_spec("opendssdirect"), "opendssdirect is not installed")
    def test_diverse_generator_keeps_reference_scan_and_varies_later_scans(self) -> None:
        from Transmission.generate_measurements_hif_ieee14 import _scan_operating_points

        points = _scan_operating_points(
            np.random.default_rng(7),
            scan_count=4,
            mode="diverse",
            event_load_scale=1.0,
            load_log_std=0.08,
            dispatch_fraction=0.20,
            voltage_std=0.008,
        )

        self.assertEqual(len(points), 4)
        self.assertTrue(all(tuple(point) == IEEE14_OPERATING_POINT_KEYS for point in points))
        self.assertTrue(all(set(point["bus_load_scales"]) == set(IEEE14_LOAD_BASE_KW) for point in points))
        self.assertTrue(
            all(set(point["generator_dispatch_kw"]) == set(IEEE14_GENERATOR_DISPATCH_KW) for point in points)
        )
        self.assertTrue(
            all(set(point["voltage_setpoints_pu"]) == set(IEEE14_GENERATOR_VOLTAGE_PU) for point in points)
        )
        self.assertTrue(all(value == 1.0 for value in points[0]["bus_load_scales"].values()))
        self.assertNotEqual(points[1]["bus_load_scales"], points[2]["bus_load_scales"])

    def test_sparse_operating_point_expands_to_canonical_schema(self) -> None:
        point = canonicalize_ieee14_operating_point(
            {
                "load_scale": 1.05,
                "bus_load_scales": {"b2": 0.95},
                "generator_dispatch_kw": {"b2": 42000.0},
            }
        )

        self.assertEqual(tuple(point), IEEE14_OPERATING_POINT_KEYS)
        self.assertEqual(point["bus_load_scales"]["b2"], 0.95)
        self.assertEqual(point["bus_load_scales"]["b3"], 1.0)
        self.assertEqual(point["generator_dispatch_kw"]["b2"], 42000.0)
        self.assertEqual(point["voltage_setpoints_pu"], IEEE14_GENERATOR_VOLTAGE_PU)

    def test_multiscan_protocol_hydrates_bound_scans_and_compacts_observability(self) -> None:
        scans = [
            {"scan_index": 0, "z_obs": [1.0, 2.0], "op_point": {"load_scale": 1.0}},
            {"scan_index": 1, "z_obs": [1.1, 2.1], "op_point": {"load_scale": 1.1}},
        ]
        messages = [
            {"role": "user", "content": "{}"},
            {
                "role": "tool",
                "name": "run_three_phase_nlm_from_path",
                "content": '{"top_hif_groups":[{"branch_row0":3}]}',
            },
        ]
        hidden = {
            "tool_context": {
                "hif_context": {
                    "case_path": "case14",
                    "scan_window_path": "bound://hif_window/example",
                    "scans": scans,
                    "sigma_z": [0.1, 0.2],
                    "pristine_model_dir": "/tmp/model",
                }
            }
        }

        arguments, notes = hydrate_tool_arguments(
            "estimate_hif_location_magnitude_multiscan_from_path",
            {"scan_window_path": "bound://hif_window/example"},
            messages,
            hidden_context=hidden,
        )
        self.assertEqual(arguments["candidate_branch_row0"], 3)
        self.assertEqual(arguments["scans"], scans)
        self.assertEqual(arguments["sigma_z"], [0.1, 0.2])
        self.assertEqual(arguments["case_path"], "case14")
        self.assertIn("hydrated_hif_multiscan_scans", notes)

        compact = summarize_tool_result_for_conversation(
            "estimate_hif_location_magnitude_multiscan_from_path",
            {
                "success": True,
                "method": "multiscan_augmented_hif_parameter_estimation",
                "parameter_identifiable": True,
                "scan_count": 2,
                "selected_scan_count": 2,
                "selected_scan_indices": [0, 1],
                "estimated": {"alpha_from_from_bus": 0.47, "r_hif_pu": 83.0},
                "fit": {"weighted_residual_norm": 1.2, "ambiguity": False},
                "observability": {
                    "parameter_dimension": 2,
                    "effective_rank": 2,
                    "condition_number": 120.0,
                    "alpha_log_r_correlation": -0.7,
                    "status": "full_rank_well_conditioned",
                    "parameter_identifiable": True,
                },
            },
            {},
            {},
        )
        self.assertEqual(compact["observability"]["effective_rank"], 2)
        self.assertEqual(compact["selected_scan_indices"], [0, 1])
        self.assertTrue(compact["parameter_identifiable"])

    @unittest.skipUnless(importlib.util.find_spec("opendssdirect"), "opendssdirect is not installed")
    def test_real_opendss_two_scan_grid_recovers_exact_candidate(self) -> None:
        from three_phase_nlm.hif_parameter_estimator import (
            _line_tokens,
            _resolve_model_dir,
            _simulate_candidate,
        )
        from three_phase_nlm.ieee14_adapter import branch_info_for_row0

        branch = 0
        branch_info = branch_info_for_row0(branch)
        model_dir = _resolve_model_dir(None, "case14")
        tokens, _ = _line_tokens(model_dir, branch_info["dss_element"])
        operating_points = [
            {
                "load_scale": 1.0,
                "bus_load_scales": {"b2": 0.93, "b4": 1.08, "b14": 1.05},
                "generator_dispatch_kw": {"b2": 43000.0},
                "voltage_setpoints_pu": {"b2": 1.04},
            },
            {"load_scale": 1.0},
        ]
        scans = []
        for scan_index, op_point in enumerate(operating_points):
            simulated = _simulate_candidate(
                model_dir=model_dir,
                original_tokens=tokens,
                dss_element=branch_info["dss_element"],
                alpha=0.5,
                phase="A",
                r_hif_pu=100.0,
                op_point=op_point,
            )
            scans.append(
                {
                    "scan_index": scan_index,
                    "z_obs": simulated["z"],
                    "three_phase_voltages": simulated["three_phase_voltages"],
                    "op_point": op_point,
                }
            )

        result = estimate_hif_location_magnitude_multiscan(
            candidate_branch_row0=branch,
            scans=scans,
            candidate_phase="A",
            scan_selection="all",
            max_scans=2,
            alpha_grid_size=3,
            r_grid_size=3,
            r_hif_pu_min=50.0,
            r_hif_pu_max=200.0,
            robust_loss="linear",
            refine_top_n=0,
            local_max_nfev=0,
        )

        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["estimated"]["alpha_from_from_bus"], 0.5, places=9)
        self.assertAlmostEqual(result["estimated"]["r_hif_pu"], 100.0, places=9)
        self.assertAlmostEqual(result["fit"]["weighted_residual_norm"], 0.0, places=9)
        self.assertEqual(result["observability"]["effective_rank"], 2)
        self.assertEqual(result["selected_scan_indices"], [0, 1])

        reordered = estimate_hif_location_magnitude_multiscan(
            candidate_branch_row0=branch,
            scans=list(reversed(scans)),
            candidate_phase="A",
            scan_selection="all",
            max_scans=2,
            alpha_grid_size=3,
            r_grid_size=3,
            r_hif_pu_min=50.0,
            r_hif_pu_max=200.0,
            robust_loss="linear",
            refine_top_n=0,
            local_max_nfev=0,
        )
        self.assertAlmostEqual(
            reordered["estimated"]["alpha_from_from_bus"],
            result["estimated"]["alpha_from_from_bus"],
            places=12,
        )
        self.assertAlmostEqual(
            reordered["estimated"]["r_hif_pu"], result["estimated"]["r_hif_pu"], places=12
        )

        with tempfile.TemporaryDirectory(prefix="hif_hidden_label_") as temporary:
            path = Path(temporary) / "window.json"
            path.write_text(
                json.dumps(
                    {
                        "id": "hidden-label-check",
                        "shared_label": {
                            "split_ratio": 0.9,
                            "phase": "C",
                            "r_hif_pu": 500.0,
                        },
                        "scans": scans,
                    }
                ),
                encoding="utf-8",
            )
            searched = estimate_hif_location_magnitude_multiscan(
                candidate_branch_row0=branch,
                scan_window_path=path,
                candidate_phase=None,
                scan_selection="all",
                max_scans=2,
                alpha_grid_size=3,
                r_grid_size=3,
                r_hif_pu_min=50.0,
                r_hif_pu_max=200.0,
                robust_loss="linear",
                refine_top_n=0,
                local_max_nfev=0,
            )
        self.assertEqual(searched["estimated"]["phase"], "A")
        self.assertAlmostEqual(searched["estimated"]["alpha_from_from_bus"], 0.5, places=9)
        self.assertAlmostEqual(searched["estimated"]["r_hif_pu"], 100.0, places=9)
        self.assertEqual(set(searched["phase_scores"]), {"A", "B", "C"})

    @unittest.skipUnless(importlib.util.find_spec("opendssdirect"), "opendssdirect is not installed")
    def test_canonical_operating_point_replays_clean_measurements(self) -> None:
        from three_phase_nlm.hif_parameter_estimator import (
            _line_tokens,
            _resolve_model_dir,
            _simulate_candidate,
        )
        from three_phase_nlm.ieee14_adapter import branch_info_for_row0

        branch_info = branch_info_for_row0(0)
        model_dir = _resolve_model_dir(None, "case14")
        tokens, _ = _line_tokens(model_dir, branch_info["dss_element"])
        op_point = canonicalize_ieee14_operating_point(
            {
                "load_scale": 1.03,
                "bus_load_scales": {"b2": 0.94, "b4": 1.07},
                "generator_dispatch_kw": {"b2": 43000.0},
                "voltage_setpoints_pu": {"b2": 1.04},
                "source_voltage_pu": 1.055,
            }
        )

        first = _simulate_candidate(
            model_dir=model_dir,
            original_tokens=tokens,
            dss_element=branch_info["dss_element"],
            alpha=0.5,
            phase="A",
            r_hif_pu=100.0,
            op_point=op_point,
        )
        second = _simulate_candidate(
            model_dir=model_dir,
            original_tokens=tokens,
            dss_element=branch_info["dss_element"],
            alpha=0.5,
            phase="A",
            r_hif_pu=100.0,
            op_point=op_point,
        )

        np.testing.assert_allclose(first["z"], second["z"], rtol=0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
