from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mcp_server.matpower_server import _load_python_case
from psse_env.providers import MatpowerDeploymentProviders
from psse_env.providers.scenario_generator import build_measurement_vector
from psse_env.transactional_env import TransactionalPSSEEnv

FIXTURE = Path(__file__).parent / "fixtures" / "case14_z.json"


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text())


class WlsRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.providers = MatpowerDeploymentProviders()
        data = _fixture()
        self.state = {
            "state_id": "episode:s0",
            "state_hash": "hash0",
            "case": data["case_path"],
            "measurements": list(data["z_obs"]),
            "metadata": {},
        }

    @staticmethod
    def _clean_candidate_state() -> dict:
        ppc = _load_python_case("case14")
        # Build a self-consistent observable snapshot whose Vm telemetry is
        # inside the case's declared operating bounds.
        ppc["bus"][:, 7] = np.minimum(
            np.maximum(ppc["bus"][:, 7], ppc["bus"][:, 12]),
            ppc["bus"][:, 11],
        )
        return {
            "state_id": "episode:s1",
            "state_hash": "hash1",
            "status": "candidate",
            "source_action": {
                "tool": "correct_measurements",
                "arguments": {"suspect_group": [0]},
            },
            "case": "case14",
            "measurements": build_measurement_vector(ppc).tolist(),
            "metadata": {},
        }

    def test_run_wls_returns_observable_decision_evidence(self) -> None:
        metrics = self.providers.run_wls(self.state)
        self.assertNotIn("execution_status", metrics)
        self.assertEqual(metrics["state_id"], "episode:s0")
        self.assertEqual(metrics["state_hash"], "hash0")
        self.assertTrue(metrics["evidence_source"].startswith("deployment_wls"))
        for key in (
            "wls_objective",
            "chi_square_statistic",
            "chi_square_threshold",
            "max_normalized_residual",
            "remaining_anomaly_score",
            "converged",
        ):
            self.assertIsNotNone(metrics.get(key), key)
        summary = metrics["wls_summary"]
        self.assertTrue(summary["top_residuals"])
        self.assertIn("global_metrics", summary)
        # The fixture snapshot carries injected errors; the chi-square test
        # must flag it as anomalous rather than clean.
        self.assertGreater(metrics["chi_square_statistic"], metrics["chi_square_threshold"])
        self.assertFalse(metrics["no_material_anomaly_remaining"])

    def test_run_wls_rejects_wrong_measurement_length(self) -> None:
        self.state["measurements"] = [1.0, 2.0, 3.0]
        metrics = self.providers.run_wls(self.state)
        self.assertEqual(metrics["execution_status"], "failure")

    def test_topology_target_uses_structural_status_not_branch_multiplier(self) -> None:
        ppc = _load_python_case("case14")
        ppc["branch"][0, 10] = 0.0

        evidence = self.providers._target_evidence(
            {
                "tool": "correct_topology",
                "arguments": {"branch_row0": 0, "status": 0},
            },
            residuals=[12.0],
            lambda_values=[250.0, -175.0],
            nl=len(ppc["branch"]),
            candidate_case=ppc,
        )

        self.assertIsNotNone(evidence)
        self.assertIs(evidence["target_fixed"], True)
        self.assertEqual(evidence["target_metric_kind"], "branch_status_mismatch")
        self.assertEqual(evidence["target_metric_value"], 0.0)
        self.assertIs(evidence["topology_target_status_matches_requested"], True)
        self.assertEqual(evidence["topology_target_branch_multiplier"], 250.0)
        self.assertIs(evidence["topology_target_branch_multiplier_cleared"], False)

    def test_topology_target_rejects_candidate_status_mismatch(self) -> None:
        ppc = _load_python_case("case14")
        ppc["branch"][0, 10] = 0.0

        evidence = self.providers._target_evidence(
            {
                "tool": "correct_topology",
                "arguments": {"branch_row0": 0, "status": 1},
            },
            residuals=[0.0],
            lambda_values=[0.0, 0.0],
            nl=len(ppc["branch"]),
            candidate_case=ppc,
        )

        self.assertIsNotNone(evidence)
        self.assertIs(evidence["target_fixed"], False)
        self.assertEqual(evidence["target_metric_value"], 1.0)
        self.assertIs(evidence["topology_target_status_matches_requested"], False)

    def test_parameter_target_retains_branch_multiplier_evidence(self) -> None:
        ppc = _load_python_case("case14")

        evidence = self.providers._target_evidence(
            {
                "tool": "correct_parameters",
                "arguments": {"branch_row0": 0},
            },
            residuals=[0.0],
            lambda_values=[250.0, -175.0],
            nl=len(ppc["branch"]),
            candidate_case=ppc,
        )

        self.assertIsNotNone(evidence)
        self.assertIs(evidence["target_fixed"], False)
        self.assertEqual(evidence["target_metric_kind"], "max_abs_branch_multiplier")
        self.assertEqual(evidence["target_metric_value"], 250.0)

    def test_hif_evidence_inventory_reports_only_completed_ladder(self) -> None:
        state = {
            **self.state,
            "evidence_request": "operator_escalation:hif_diagnostics_exhausted",
            "policy_observation": {
                "unresolved_signatures": ["hif_suspected_zero_sequence"],
                "explained_anomalies": [],
                "available_evidence": ["nlm_diagnostic", "hif_scan_window"],
                "tried_action_signatures": [
                    "run_three_phase_nlm_from_path:{}",
                    "estimate_hif_location_magnitude_multiscan_from_path:{}",
                    "estimate_hif_location_magnitude_from_path:{}",
                ],
            },
        }
        metrics = self.providers.request_additional_evidence(state)

        self.assertNotIn("execution_status", metrics)
        self.assertIs(metrics["additional_evidence_available"], False)
        self.assertIs(metrics["operator_review_required"], True)
        self.assertEqual(metrics["state_id"], self.state["state_id"])
        self.assertEqual(metrics["state_hash"], self.state["state_hash"])

    def test_hif_evidence_inventory_fails_before_all_estimators_are_attempted(self) -> None:
        state = {
            **self.state,
            "evidence_request": "operator_escalation:hif_diagnostics_exhausted",
            "policy_observation": {
                "unresolved_signatures": ["hif_suspected_zero_sequence"],
                "explained_anomalies": [],
                "available_evidence": ["nlm_diagnostic", "hif_scan_window"],
                "tried_action_signatures": [
                    "run_three_phase_nlm_from_path:{}",
                    "estimate_hif_location_magnitude_multiscan_from_path:{}",
                ],
            },
        }
        metrics = self.providers.request_additional_evidence(state)
        self.assertEqual(metrics["execution_status"], "failure")
        self.assertEqual(metrics["error_code"], "hif_diagnostic_ladder_incomplete")

    def test_generic_recovery_inventory_reports_operator_handoff_not_resolution(self) -> None:
        state = {
            **self.state,
            "evidence_request": "operator_escalation:recovery_options_exhausted",
            "policy_observation": {
                "unresolved_signatures": ["wls_residual_outlier index=3"],
                "remaining_anomaly_score": 2.0,
                "available_evidence": [],
                "tried_action_signatures": [
                    "run_wls:{}",
                    "get_measurement_context:{}",
                ],
            },
        }
        metrics = self.providers.request_additional_evidence(state)

        self.assertIs(metrics["additional_evidence_available"], False)
        self.assertIs(metrics["operator_review_required"], True)
        self.assertNotIn("no_material_anomaly_remaining", metrics)
        self.assertNotIn("anomaly_explanation", metrics)

    def test_clean_candidate_emits_scoped_snapshot_physical_evidence(self) -> None:
        metrics = self.providers.run_wls(self._clean_candidate_state())

        self.assertLess(metrics["chi_square_statistic"], metrics["chi_square_threshold"])
        self.assertIs(metrics["physical_constraints_ok"], True)
        self.assertNotIn("power_flow_converged", metrics)
        evidence = metrics["steady_state_physical_evidence"]
        self.assertEqual(evidence["scope"], "observed_snapshot_topology_vm_rate_a")
        self.assertTrue(evidence["complete"])
        self.assertTrue(evidence["topology_connectivity"]["connected"])
        self.assertTrue(evidence["bus_voltage_bounds"]["within_bounds"])
        self.assertTrue(
            evidence["active_branch_rate_a_bounds"]["within_defined_rate_a_bounds"]
        )

    def test_anomalous_candidate_gets_independent_scoped_physical_evidence(self) -> None:
        state = {
            **self.state,
            "status": "candidate",
            "source_action": {
                "tool": "correct_measurements",
                "arguments": {"suspect_group": [0]},
            },
        }
        metrics = self.providers.run_wls(state)
        self.assertGreaterEqual(
            metrics["chi_square_statistic"], metrics["chi_square_threshold"]
        )
        self.assertIs(metrics["physical_constraints_ok"], True)
        self.assertFalse(metrics["globally_resolved"])
        self.assertNotIn("power_flow_converged", metrics)
        evidence = metrics["steady_state_physical_evidence"]
        self.assertEqual(
            evidence["method"],
            "matpower_case_limits_with_observed_wls_telemetry",
        )
        self.assertTrue(evidence["complete"])

    def test_measured_vm_violation_fails_complete_physical_check(self) -> None:
        state = self._clean_candidate_state()
        ppc = _load_python_case("case14")
        ppc["bus"][0, 11] = 0.95
        state["case"] = self.providers._derived_case(ppc, "test_low_vmax")

        metrics = self.providers.run_wls(state)
        self.assertLess(metrics["chi_square_statistic"], metrics["chi_square_threshold"])
        self.assertIs(metrics["physical_constraints_ok"], False)
        self.assertTrue(metrics["physical_evidence_complete"])
        self.assertIn(
            "bus_voltage_out_of_bounds",
            {item["type"] for item in metrics["physical_bound_violations"]},
        )

    def test_active_branch_rate_a_violation_uses_measured_terminal_mva(self) -> None:
        state = self._clean_candidate_state()
        ppc = _load_python_case("case14")
        ppc["branch"][0, 5] = 1.0
        state["case"] = self.providers._derived_case(ppc, "test_low_rate_a")

        metrics = self.providers.run_wls(state)
        self.assertLess(metrics["chi_square_statistic"], metrics["chi_square_threshold"])
        self.assertIs(metrics["physical_constraints_ok"], False)
        violation = next(
            item
            for item in metrics["physical_bound_violations"]
            if item["type"] == "active_branch_rate_a_exceeded"
        )
        self.assertEqual(violation["branch_row0"], 0)
        self.assertGreater(max(violation["from_mva"], violation["to_mva"]), 1.0)

    def test_disconnected_snapshot_is_a_complete_physical_violation(self) -> None:
        solved = self.providers._solve(self._clean_candidate_state())
        solved = copy.deepcopy(solved)
        branch = solved["ppc"]["branch"]
        for row in range(branch.shape[0]):
            if 14 in {int(branch[row, 0]), int(branch[row, 1])}:
                branch[row, 10] = 0.0

        metrics = self.providers._steady_state_physical_evidence(solved)
        self.assertIs(metrics["physical_constraints_ok"], False)
        self.assertTrue(metrics["physical_evidence_complete"])
        self.assertFalse(
            metrics["steady_state_physical_evidence"]["topology_connectivity"][
                "connected"
            ]
        )

    def test_incomplete_physical_inputs_are_inconclusive_and_fail_closed(self) -> None:
        solved = self.providers._solve(self._clean_candidate_state())
        solved = copy.deepcopy(solved)
        solved["index_map"] = {}

        metrics = self.providers._steady_state_physical_evidence(solved)
        self.assertIsNone(metrics["physical_constraints_ok"])
        self.assertFalse(metrics["physical_evidence_complete"])
        errors = metrics["steady_state_physical_evidence"]["input_errors"]
        self.assertEqual(errors.count("measurement_index_map_invalid"), 1)
        self.assertFalse(metrics["physical_bound_violations"])


class MeasurementContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.providers = MatpowerDeploymentProviders()
        data = _fixture()
        z = list(data["z_obs"])
        self.error_index = 5
        z[self.error_index] += 5.0
        self.state = {
            "state_id": "episode:s0",
            "state_hash": "hash0",
            "case": data["case_path"],
            "measurements": z,
            "metadata": {},
        }

    def test_context_preserves_singletons_before_bounded_fallbacks(self) -> None:
        metrics = self.providers.get_measurement_context(self.state)
        self.assertNotIn("execution_status", metrics)
        findings = metrics["measurement_findings"]
        self.assertTrue(findings)
        flagged = {item["index0"] for item in findings}
        self.assertIn(self.error_index, flagged)
        supported = metrics["supported_corrections"]
        self.assertTrue(supported)
        self.assertEqual(len(supported[0]["arguments"]["suspect_group"]), 1)
        self.assertIn(supported[0]["arguments"]["suspect_group"][0], flagged)
        singleton_targets = {
            proposal["arguments"]["suspect_group"][0]
            for proposal in supported
            if len(proposal["arguments"]["suspect_group"]) == 1
        }
        self.assertEqual(singleton_targets, flagged)
        self.assertTrue(
            all(
                set(proposal["arguments"]["suspect_group"]) <= flagged
                for proposal in supported
            )
        )
        self.assertEqual(supported[0]["tool"], "correct_measurements")

    def test_context_appends_bounded_vm_group_after_singletons(self) -> None:
        baseline = self.providers.get_measurement_context(self.state)
        bounded_targets = sorted(
            item["index0"] for item in baseline["measurement_findings"][:2]
        )
        with patch.object(
            self.providers,
            "_physical_vm_joint_targets",
            return_value=bounded_targets,
        ):
            metrics = self.providers.get_measurement_context(self.state)

        self.assertEqual(metrics["physical_vm_joint_targets"], bounded_targets)
        groups = [
            action["arguments"]["suspect_group"]
            for action in metrics["supported_corrections"]
        ]
        self.assertIn(sorted(bounded_targets), groups)
        self.assertEqual(len(groups[0]), 1)
        finding_indices = {
            item["index0"] for item in metrics["measurement_findings"]
        }
        self.assertLessEqual(
            set(metrics["physical_vm_closure_targets"]), finding_indices
        )
        self.assertTrue(
            any(
                len(action["arguments"]["suspect_group"]) == 1
                for action in metrics["supported_corrections"]
            )
        )

    def test_coupled_fallback_contains_only_ranked_residual_targets(self) -> None:
        with patch.object(
            self.providers,
            "_physical_vm_joint_targets",
            return_value=[],
        ):
            metrics = self.providers.get_measurement_context(self.state)
        findings = [item["index0"] for item in metrics["measurement_findings"]]

        self.assertEqual(
            metrics["coupled_measurement_fallback_targets"],
            sorted(findings[:2]),
        )
        self.assertLessEqual(
            set(metrics["coupled_measurement_fallback_targets"]), set(findings)
        )
        executable_groups = [
            action["arguments"]["suspect_group"]
            for action in metrics["supported_corrections"]
        ]
        self.assertNotIn(
            metrics["coupled_measurement_fallback_targets"], executable_groups
        )

    def test_in_bound_vm_residuals_never_enter_the_physical_joint_group(self) -> None:
        solved = self.providers._solve(self.state)
        bus = np.asarray(solved["ppc"]["bus"], dtype=float)
        z = list(solved["z"])
        for row in (0, 1):
            z[row] = 0.5 * (float(bus[row, 11]) + float(bus[row, 12]))
        solved = {**solved, "z": z}
        evidence = [
            {
                "index0": row,
                "channel": "Vm",
                "channel_offset": row,
                "value": self.providers.residual_threshold + 1.0,
            }
            for row in (0, 1)
        ]

        self.assertEqual(
            self.providers._physical_vm_joint_targets(solved, evidence), []
        )

    def test_non_vm_outliers_never_enter_the_physical_joint_group(self) -> None:
        solved = self.providers._solve(self.state)
        pf_slice = solved["index_map"]["Pf"]
        evidence = [
            {
                "index0": int(pf_slice.start) + row,
                "channel": "Pf",
                "channel_offset": row,
                "value": self.providers.residual_threshold + 10.0,
            }
            for row in (0, 1)
        ]

        self.assertEqual(
            self.providers._physical_vm_joint_targets(solved, evidence), []
        )

    def test_only_residual_ranked_out_of_bound_vm_members_enter_joint_group(
        self,
    ) -> None:
        solved = self.providers._solve(self.state)
        bus = np.asarray(solved["ppc"]["bus"], dtype=float)
        z = list(solved["z"])
        for row in (0, 1):
            z[row] = (
                float(bus[row, 11])
                + self.providers.vm_bound_tolerance_pu
                + 0.01
            )
        # A third out-of-bound Vm is intentionally absent from evidence and
        # therefore cannot be added to the executable group.
        z[2] = (
            float(bus[2, 11])
            + self.providers.vm_bound_tolerance_pu
            + 0.01
        )
        solved = {**solved, "z": z}
        evidence = [
            {
                "index0": row,
                "channel": "Vm",
                "channel_offset": row,
                "value": self.providers.residual_threshold + 1.0,
            }
            for row in (0, 1)
        ]

        self.assertEqual(
            self.providers._physical_vm_joint_targets(solved, evidence), [0, 1]
        )

    def test_context_can_jointly_refine_previously_accepted_targets(self) -> None:
        state = copy.deepcopy(self.state)
        state["policy_observation"] = {
            "accepted_corrections": [
                {
                    "source_action": {
                        "tool": "correct_measurements",
                        "arguments": {"suspect_group": [self.error_index]},
                    }
                },
                {
                    "source_action": {
                        "tool": "correct_measurements",
                        "arguments": {"suspect_group": [self.error_index + 1]},
                    }
                },
            ]
        }

        metrics = self.providers.get_measurement_context(state)

        groups = [
            proposal["arguments"]["suspect_group"]
            for proposal in metrics["supported_corrections"]
        ]
        self.assertIn([self.error_index, self.error_index + 1], groups)
        self.assertTrue(metrics["accepted_target_refinement"])

    def test_context_defers_refinement_while_new_singleton_dominates(self) -> None:
        state = copy.deepcopy(self.state)
        accepted = [self.error_index + 1, self.error_index + 2]
        state["policy_observation"] = {
            "accepted_corrections": [
                {
                    "source_action": {
                        "tool": "correct_measurements",
                        "arguments": {"suspect_group": [index]},
                    }
                }
                for index in accepted
            ]
        }

        metrics = self.providers.get_measurement_context(state)

        groups = [
            proposal["arguments"]["suspect_group"]
            for proposal in metrics["supported_corrections"]
        ]
        self.assertFalse(metrics["accepted_target_refinement"])
        self.assertNotIn(accepted, groups)
        self.assertIn(self.error_index, metrics["accepted_target_refinement_blocked_by"])
        self.assertTrue(
            metrics["accepted_target_refinement_dominant_target_unaccepted"]
        )

    def test_context_refines_accepted_targets_in_budgeted_ambiguity_band(self) -> None:
        state = copy.deepcopy(self.state)
        accepted = [self.error_index + 1, self.error_index + 2]
        state["policy_observation"] = {
            "remaining_budget": 8,
            "accepted_corrections": [
                {
                    "source_action": {
                        "tool": "correct_measurements",
                        "arguments": {"suspect_group": [index]},
                    }
                }
                for index in accepted
            ],
        }
        statistic = self.providers.get_measurement_context(state)[
            "chi_square_statistic"
        ]

        with patch(
            "psse_env.providers.matpower.chi2_threshold",
            return_value=statistic / 1.05,
        ):
            metrics = self.providers.get_measurement_context(state)

        groups = [
            proposal["arguments"]["suspect_group"]
            for proposal in metrics["supported_corrections"]
        ]
        self.assertTrue(metrics["accepted_target_refinement"])
        self.assertTrue(
            metrics["accepted_target_refinement_near_threshold_override"]
        )
        self.assertAlmostEqual(
            metrics["accepted_target_refinement_anomaly_ratio"], 1.05
        )
        self.assertEqual(
            metrics["accepted_target_refinement_remaining_budget"], 8
        )
        self.assertIn(accepted, groups)

    def test_context_does_not_repeat_an_accepted_joint_refinement(self) -> None:
        state = copy.deepcopy(self.state)
        accepted = [self.error_index + 1, self.error_index + 2]
        state["policy_observation"] = {
            "accepted_corrections": [
                {
                    "source_action": {
                        "tool": "correct_measurements",
                        "arguments": {"suspect_group": accepted},
                    }
                }
            ]
        }

        metrics = self.providers.get_measurement_context(state)

        groups = [
            proposal["arguments"]["suspect_group"]
            for proposal in metrics["supported_corrections"]
        ]
        self.assertFalse(metrics["accepted_target_refinement"])
        self.assertTrue(metrics["accepted_target_refinement_already_accepted"])
        self.assertNotIn(accepted, groups)

    def test_context_reestimates_an_accepted_meter_after_branch_repair(self) -> None:
        state = copy.deepcopy(self.state)
        state["policy_observation"] = {
            "accepted_corrections": [
                {
                    "source_action": {
                        "tool": "correct_measurements",
                        "arguments": {"suspect_group": [self.error_index]},
                    }
                },
                {
                    "source_action": {
                        "tool": "correct_parameters",
                        "arguments": {"line_index": 1},
                    }
                },
            ]
        }

        metrics = self.providers.get_measurement_context(state)

        groups = [
            proposal["arguments"]["suspect_group"]
            for proposal in metrics["supported_corrections"]
        ]
        self.assertTrue(metrics["accepted_target_refinement"])
        self.assertEqual(
            metrics["accepted_target_refinement_kind"],
            "post_branch_model_reestimate",
        )
        self.assertIn([self.error_index], groups)

    def test_post_branch_reestimate_excludes_colocated_flow_target(self) -> None:
        state = copy.deepcopy(self.state)
        direct_flow_index = 3 * 14
        state["measurements"] = list(_fixture()["z_obs"])
        state["measurements"][direct_flow_index] += 5.0
        state["policy_observation"] = {
            "accepted_corrections": [
                {
                    "source_action": {
                        "tool": "correct_measurements",
                        "arguments": {"suspect_group": [direct_flow_index]},
                    }
                },
                {
                    "source_action": {
                        "tool": "correct_parameters",
                        "arguments": {"line_index": 1},
                    }
                },
            ]
        }

        metrics = self.providers.get_measurement_context(state)

        groups = [
            proposal["arguments"]["suspect_group"]
            for proposal in metrics["supported_corrections"]
        ]
        self.assertFalse(metrics["accepted_target_refinement"])
        self.assertIn(
            direct_flow_index,
            metrics["accepted_target_refinement_suppressed_colocated_indices"],
        )
        self.assertNotIn([direct_flow_index], groups)

    def test_context_suppresses_direct_flow_residual_on_repaired_branch(self) -> None:
        state = copy.deepcopy(self.state)
        direct_flow_index = 3 * 14
        state["measurements"] = list(_fixture()["z_obs"])
        state["measurements"][direct_flow_index] += 5.0
        state["policy_observation"] = {
            "accepted_corrections": [
                {
                    "source_action": {
                        "tool": "correct_parameters",
                        "arguments": {"line_index": 1},
                    }
                }
            ]
        }

        metrics = self.providers.get_measurement_context(state)

        groups = [
            proposal["arguments"]["suspect_group"]
            for proposal in metrics["supported_corrections"]
        ]
        self.assertIn(
            direct_flow_index,
            metrics["suppressed_colocated_post_branch_indices"],
        )
        self.assertNotIn([direct_flow_index], groups)

    def test_lambda_contexts_expose_branch_targets(self) -> None:
        parameter_without_scans = self.providers.get_parameter_context(self.state)
        self.assertNotIn("execution_status", parameter_without_scans)
        self.assertTrue(parameter_without_scans["parameter_findings"])
        self.assertIs(parameter_without_scans["parameter_scans_available"], False)
        self.assertEqual(parameter_without_scans["parameter_scan_count"], 0)
        self.assertEqual(parameter_without_scans["supported_corrections"], [])
        self.assertEqual(
            parameter_without_scans["route_status"],
            "unavailable_or_inconclusive",
        )

        state_with_scans = copy.deepcopy(self.state)
        state_with_scans["metadata"] = {
            "parameter_scans": {"z_scans": [list(self.state["measurements"])]}
        }
        parameter = self.providers.get_parameter_context(state_with_scans)
        self.assertNotIn("execution_status", parameter)
        self.assertIn("parameter_findings", parameter)
        self.assertIs(parameter["parameter_scans_available"], True)
        self.assertEqual(parameter["parameter_scan_count"], 1)
        self.assertTrue(parameter["supported_corrections"])
        self.assertEqual(parameter["route_status"], "actionable")
        for proposal in parameter["supported_corrections"]:
            self.assertEqual(proposal["tool"], "correct_parameters")
            self.assertGreaterEqual(proposal["arguments"]["line_index"], 1)

        for malformed_scans in (
            [[1.0]],
            [[float("nan")] * len(self.state["measurements"])],
        ):
            with self.subTest(malformed_scans=len(malformed_scans[0])):
                malformed = copy.deepcopy(self.state)
                malformed["metadata"] = {
                    "parameter_scans": {"z_scans": malformed_scans}
                }
                context = self.providers.get_parameter_context(malformed)
                self.assertIs(context["parameter_scans_available"], False)
                self.assertEqual(context["parameter_scan_count"], 1)
                self.assertEqual(context["supported_corrections"], [])
        topology = self.providers.get_topology_context(self.state)
        self.assertNotIn("execution_status", topology)
        for proposal in topology["supported_corrections"]:
            self.assertEqual(proposal["tool"], "correct_topology")
            self.assertIn(proposal["arguments"]["status"], (0, 1))

    def test_topology_screen_preserves_mapping_case_transaction_parity(self) -> None:
        wrapped = copy.deepcopy(self.state)
        wrapped["case"] = {"case_path": self.state["case"]}

        topology = self.providers.get_topology_context(wrapped)

        self.assertTrue(topology["topology_candidate_screening"])
        self.assertEqual(topology["supported_corrections"], [])
        self.assertTrue(
            all(
                item["disposition"] == "REJECT"
                for item in topology["topology_candidate_screening"]
            )
        )
        self.assertTrue(
            any(
                item["progress_class"] == "healthy_component_corruption"
                for item in topology["topology_candidate_screening"]
            )
        )


class CorrectionExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.providers = MatpowerDeploymentProviders()
        data = _fixture()
        z = list(data["z_obs"])
        self.error_index = 5
        self.original_value = z[self.error_index]
        z[self.error_index] += 5.0
        self.state = {
            "state_id": "episode:s0",
            "state_hash": "hash0",
            "case": data["case_path"],
            "measurements": z,
            "metadata": {},
        }

    def test_suspect_group_is_hydrated_into_indexed_value_updates(self) -> None:
        result = self.providers.correct_measurements(
            self.state,
            {
                "tool": "correct_measurements",
                "arguments": {"state_id": "episode:s0", "suspect_group": [self.error_index]},
            },
        )
        self.assertNotIn("execution_status", result)
        updates = result["modification"]["measurement_updates"]
        self.assertIn(self.error_index, updates)
        corrected = updates[self.error_index]
        corrupted = self.state["measurements"][self.error_index]
        self.assertLess(
            abs(corrected - self.original_value), abs(corrupted - self.original_value)
        )
        self.assertTrue(result["applied_any_correction"])
        self.assertIn("correction_summary", result)

    def test_missing_targets_fail_closed(self) -> None:
        result = self.providers.correct_measurements(
            self.state,
            {"tool": "correct_measurements", "arguments": {"state_id": "episode:s0"}},
        )
        self.assertEqual(result["execution_status"], "failure")
        self.assertEqual(result["error_code"], "measurement_correction_target_missing")

    def test_parameter_correction_without_scans_fails_closed(self) -> None:
        result = self.providers.correct_parameters(
            self.state,
            {
                "tool": "correct_parameters",
                "arguments": {"state_id": "episode:s0", "line_index": 3},
            },
        )
        self.assertEqual(result["execution_status"], "failure")
        self.assertEqual(result["error_code"], "parameter_scans_missing")

    @patch(
        "psse_env.providers.matpower._param_correction_json",
        return_value={"success": True, "corrected_params": [0.02, 0.08]},
    )
    def test_parameter_initial_states_are_derived_from_observations(self, mocked) -> None:
        state = copy.deepcopy(self.state)
        scan = list(_fixture()["z_obs"])
        state["metadata"] = {
            "parameter_scans": {
                "z_scans": [scan],
                "initial_states": [[999.0] * 28],
            }
        }

        result = self.providers.correct_parameters(
            state,
            {
                "tool": "correct_parameters",
                "arguments": {"state_id": "episode:s0", "line_index": 3},
            },
        )

        self.assertNotIn("execution_status", result)
        supplied_starts = mocked.call_args.args[3]
        self.assertEqual(supplied_starts[0][:14], scan[:14])
        self.assertNotEqual(supplied_starts, state["metadata"]["parameter_scans"]["initial_states"])
        configured = _load_python_case(state["case"])["bus"][:, 8]
        configured = configured - configured[0]
        np.testing.assert_allclose(supplied_starts[0][14:], configured)

    def test_topology_correction_writes_derived_case(self) -> None:
        result = self.providers.correct_topology(
            self.state,
            {
                "tool": "correct_topology",
                "arguments": {"state_id": "episode:s0", "line_index": 3, "status": 0},
            },
        )
        self.assertNotIn("execution_status", result)
        derived = result["modification"]["case"]
        self.assertTrue(Path(derived).is_file())
        self.assertNotEqual(derived, self.state["case"])
        # The derived case must be solvable and reflect the status change.
        rerun = self.providers.run_wls({**self.state, "case": derived})
        self.assertNotIn("execution_status", rerun)
        repeat = self.providers.correct_topology(
            self.state,
            {
                "tool": "correct_topology",
                "arguments": {"state_id": "episode:s0", "line_index": 3, "status": 1},
            },
        )
        self.assertEqual(repeat["execution_status"], "failure")
        self.assertEqual(repeat["error_code"], "topology_correction_no_change")

    def test_content_addressed_case_replaces_stale_existing_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            providers = MatpowerDeploymentProviders(derived_case_dir=directory)
            case = _load_python_case("case14")
            derived = Path(providers._derived_case(case, "determinism"))
            expected = derived.read_bytes()
            derived.write_bytes(b"stale-or-corrupt-case")

            rebuilt = Path(providers._derived_case(case, "determinism"))

            self.assertEqual(rebuilt, derived)
            self.assertEqual(rebuilt.read_bytes(), expected)
            self.assertEqual(_load_python_case(str(rebuilt))["branch"].shape, (20, 13))


class DiagnosticProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.providers = MatpowerDeploymentProviders()
        self.data = _fixture()

    def _env(
        self, metadata: dict | None = None, **scenario_overrides
    ) -> TransactionalPSSEEnv:
        env = TransactionalPSSEEnv(**self.providers.env_kwargs(), production_dataset_mode=True)
        scenario = {
            "scenario_id": "diagnostics",
            "case": self.data["case_path"],
            "measurements": list(self.data["z_obs"]),
            "metadata": metadata or {},
        }
        scenario.update(scenario_overrides)
        env.reset(scenario)
        return env

    @staticmethod
    def _harmonic_metadata() -> dict:
        return {
            "harmonic_measurements": [
                {"h": 5, "bus": bus, "Vm": 0.02 + 0.001 * bus, "Va_deg": 10.0 * bus, "sigma": 1e-4}
                for bus in range(1, 15)
            ]
        }

    def test_harmonic_context_and_hse_run_through_the_environment(self) -> None:
        env = self._env(self._harmonic_metadata())
        active = env.current_state()["active_state_id"]
        _, context_output = env.step(
            {"tool": "get_harmonic_context", "arguments": {"state_id": active}}
        )
        self.assertEqual(context_output["execution_status"], "success")
        metrics = context_output["tool_metrics"]
        self.assertEqual(metrics["harmonic_orders"], [5])
        self.assertEqual(metrics["finding_count"], 14)

        _, hse_output = env.step({"tool": "run_hse_from_path", "arguments": {"state_id": active}})
        self.assertEqual(hse_output["execution_status"], "success")
        hse_metrics = hse_output["tool_metrics"]
        self.assertIsNotNone(hse_metrics["best_candidate_bus_1based"])
        self.assertTrue(hse_metrics["hse_summary"]["ranking_top5"])
        self.assertTrue(hse_metrics["diagnostic_acceptance"]["accepted"])

    def test_harmonic_best_bus_without_threshold_crossing_is_not_explained(self) -> None:
        state = {
            "state_id": "episode:s0",
            "state_hash": "hash0",
            "case": self.data["case_path"],
            "measurements": list(self.data["z_obs"]),
            "metadata": self._harmonic_metadata(),
        }
        payload = {
            "success": True,
            "best_candidate_bus_1based": 2,
            "estimated_thd_percent": {"2": 0.25},
            "ranking_top10": [{"bus_1based": 2, "score": 1.0}],
        }
        with patch("psse_env.providers.matpower._run_hse_logic", return_value=payload):
            metrics = self.providers.run_hse(
                state, {"tool": "run_hse_from_path", "arguments": {}}
            )
        self.assertFalse(metrics["diagnostic_acceptance"]["accepted"])
        self.assertNotIn("anomaly_explanation", metrics)

    @staticmethod
    def _three_phase_voltages(*, unbalanced: bool) -> list[dict]:
        return [
            {
                "bus": "b1",
                "vln_pu": [1.0, 0.78 if unbalanced else 1.0, 1.0],
                "ang_deg": [0.0, -120.0, 120.0],
            }
        ]

    def test_three_phase_voltage_channel_is_model_visible(self) -> None:
        env = self._env(
            {"three_phase_voltages": self._three_phase_voltages(unbalanced=True)}
        )
        self.assertIn(
            "three_phase_voltages", env.get_policy_observation().available_evidence
        )

    def test_pure_unbalance_is_explained_without_hif_escalation(self) -> None:
        from psse_env.oracle import ExpertPolicyOracle

        env = self._env(
            {"three_phase_voltages": self._three_phase_voltages(unbalanced=True)},
            unresolved_signatures=["three_phase_unbalance vuf_threshold_exceeded"],
            semantic_field_provenance={
                "unresolved_signatures": "deployment_sensor:sequence_voltage"
            },
        )
        oracle = ExpertPolicyOracle(process_oracle=env.process_oracle)
        executed: list[str] = []
        for _ in range(3):
            if env.is_terminal():
                break
            actions = oracle.next_actions(env.get_oracle_state(env.history), env.history)
            self.assertTrue(actions)
            self.assertNotIn("estimate_hif", actions[0]["tool"])
            env.assert_training_decision_evidence(actions[0])
            _, output = env.step(actions[0])
            self.assertEqual(output["execution_status"], "success")
            executed.append(actions[0]["tool"])
        self.assertEqual(
            executed, ["run_three_phase_nlm_from_path", "finalize_diagnosis"]
        )
        self.assertTrue(env.is_terminal())
        explanation = env.get_policy_observation().explained_anomalies[0]
        self.assertEqual(explanation["family"], "three_phase_unbalance")

    def test_balanced_voltage_null_does_not_create_terminal_explanation(self) -> None:
        env = self._env(
            {"three_phase_voltages": self._three_phase_voltages(unbalanced=False)},
            unresolved_signatures=["three_phase_unbalance suspected"],
            semantic_field_provenance={
                "unresolved_signatures": "deployment_sensor:sequence_voltage"
            },
        )
        active = env.current_state()["active_state_id"]
        _, output = env.step(
            {"tool": "run_three_phase_nlm_from_path", "arguments": {"state_id": active}}
        )
        self.assertEqual(output["execution_status"], "success")
        self.assertFalse(output["tool_metrics"]["diagnostic_acceptance"]["accepted"])
        self.assertFalse(env.get_policy_observation().explained_anomalies)
        _, final = env.step({"tool": "finalize_diagnosis", "arguments": {}})
        self.assertEqual(final["execution_status"], "failure")
        self.assertEqual(final["error_code"], "terminal_condition_not_met")

    def test_sanitized_cached_nlm_output_does_not_invent_detection_label(self) -> None:
        diagnostic = {
            "success": True,
            "converged": True,
            "method": "legacy_three_phase_nlm",
            "top_hif_groups": [
                {"rank": 1, "branch_row0": 3, "score": 0.9}
            ],
        }
        env = self._env(
            {"nlm_diagnostic": diagnostic},
            unresolved_signatures=["hif_suspected_zero_sequence"],
            semantic_field_provenance={
                "unresolved_signatures": "deployment_sensor:waveform_capture"
            },
        )
        active = env.current_state()["active_state_id"]
        _, output = env.step(
            {
                "tool": "run_three_phase_nlm_from_path",
                "arguments": {"state_id": active},
            }
        )
        self.assertEqual(output["execution_status"], "success")
        summary = output["tool_metrics"]["nlm_summary"]
        self.assertNotIn("detected", summary)
        self.assertEqual(summary["top_hif_groups"][0]["branch_row0"], 3)

    def test_hif_optimizer_requires_null_model_improvement_to_explain(self) -> None:
        state = {
            "state_id": "episode:s0",
            "state_hash": "hash0",
            "case": self.data["case_path"],
            "measurements": list(self.data["z_obs"]),
            "metadata": {},
        }
        action = {
            "tool": "estimate_hif_location_magnitude_from_path",
            "arguments": {"candidate_branch_row0": 2},
        }
        base_payload = {
            "success": True,
            "candidate_branch_row0": 2,
            "estimated": {"alpha_from_from_bus": 0.5, "r_hif_pu": 100.0},
            "fit": {
                "weighted_residual_norm": 1.0,
                "residual_reduction_vs_no_refinement": 0.05,
            },
        }
        with patch(
            "psse_env.providers.matpower._estimate_hif_location_magnitude_logic",
            return_value=base_payload,
        ):
            rejected = self.providers.estimate_hif(state, action)
        self.assertFalse(rejected["diagnostic_acceptance"]["accepted"])
        self.assertNotIn("anomaly_explanation", rejected)

        accepted_payload = {
            **base_payload,
            "fit": {
                "weighted_residual_norm": 1.0,
                "residual_reduction_vs_no_refinement": 0.50,
            },
        }
        with patch(
            "psse_env.providers.matpower._estimate_hif_location_magnitude_logic",
            return_value=accepted_payload,
        ):
            accepted = self.providers.estimate_hif(state, action)
        self.assertTrue(accepted["diagnostic_acceptance"]["accepted"])
        self.assertEqual(
            accepted["anomaly_explanation"]["kind"], "hif_model_accepted_over_null"
        )

    def test_diagnostics_without_runtime_data_fail_closed_as_noop(self) -> None:
        env = self._env()
        active = env.current_state()["active_state_id"]
        before_hash = env.store.episode_hash()
        for tool, expected_code in (
            ("get_harmonic_context", "harmonic_context_missing"),
            ("run_hse_from_path", "hse_runtime_missing"),
            ("run_three_phase_nlm_from_path", "nlm_runtime_missing"),
        ):
            _, output = env.step({"tool": tool, "arguments": {"state_id": active}})
            self.assertEqual(output["execution_status"], "failure", tool)
            self.assertEqual(output["error_code"], expected_code, tool)
        _, output = env.step(
            {
                "tool": "estimate_hif_location_magnitude_multiscan_from_path",
                "arguments": {"state_id": active, "candidate_branch_row0": 7},
            }
        )
        self.assertEqual(output["execution_status"], "failure")
        self.assertEqual(output["error_code"], "hif_scan_window_missing")
        self.assertEqual(env.store.episode_hash(), before_hash)

    def test_hif_estimator_requires_branch_target(self) -> None:
        env = self._env()
        active = env.current_state()["active_state_id"]
        _, output = env.step(
            {
                "tool": "estimate_hif_location_magnitude_from_path",
                "arguments": {"state_id": active},
            }
        )
        self.assertEqual(output["execution_status"], "failure")
        self.assertEqual(output["error_code"], "hif_target_missing")

    def test_diagnostics_are_blocked_while_candidate_is_unverified(self) -> None:
        env = self._env(self._harmonic_metadata())
        active = env.current_state()["active_state_id"]
        env.step({"tool": "run_wls", "arguments": {"state_id": active}})
        _, ctx = env.step({"tool": "get_measurement_context", "arguments": {"state_id": active}})
        _, correction = env.step(ctx["tool_metrics"]["supported_corrections"][0])
        self.assertEqual(correction["execution_status"], "success")
        _, output = env.step(
            {"tool": "get_harmonic_context", "arguments": {"state_id": active}}
        )
        self.assertEqual(output["execution_status"], "failure")
        self.assertEqual(output["error_code"], "candidate_lifecycle_violation")


class EndToEndEnvironmentTests(unittest.TestCase):
    def test_production_env_runs_wls_context_correct_verify_cycle(self) -> None:
        providers = MatpowerDeploymentProviders()
        data = _fixture()
        z = list(data["z_obs"])
        error_index = 5
        z[error_index] += 5.0
        env = TransactionalPSSEEnv(
            **providers.env_kwargs(),
            production_dataset_mode=True,
        )
        env.reset(
            {
                "scenario_id": "deployment_smoke",
                "case": data["case_path"],
                "measurements": z,
            }
        )
        active = env.current_state()["active_state_id"]

        _, wls_output = env.step({"tool": "run_wls", "arguments": {"state_id": active}})
        self.assertEqual(wls_output["execution_status"], "success")
        self.assertIn("chi_square_statistic", wls_output["tool_metrics"])

        _, context_output = env.step(
            {"tool": "get_measurement_context", "arguments": {"state_id": active}}
        )
        self.assertEqual(context_output["execution_status"], "success")
        supported = context_output["tool_metrics"]["supported_corrections"]
        self.assertTrue(supported)

        _, correction_output = env.step(supported[0])
        self.assertEqual(correction_output["execution_status"], "success")
        candidate_id = correction_output["candidate_state_id"]
        self.assertIsNotNone(candidate_id)

        _, verify_output = env.step({"tool": "run_wls", "arguments": {"state_id": candidate_id}})
        self.assertEqual(verify_output["execution_status"], "success")
        state = env.current_state()
        self.assertTrue(state.get("has_verified_candidate"))
        self.assertIn(
            state.get("candidate_disposition"),
            {"ACCEPT_FINAL", "ACCEPT_PARTIAL", "REJECT", "INCONCLUSIVE"},
        )
        # The strongest-residual singleton must reduce the observable WLS objective.
        metrics = verify_output["tool_metrics"]
        self.assertLess(
            metrics["wls_objective"], wls_output["tool_metrics"]["wls_objective"]
        )
        # An accepted candidate must be committable under production evidence.
        if state.get("candidate_disposition") in {"ACCEPT_FINAL", "ACCEPT_PARTIAL"}:
            _, commit_output = env.step(
                {"tool": "commit_state", "arguments": {"candidate_state_id": candidate_id}}
            )
            self.assertEqual(commit_output["execution_status"], "success")


if __name__ == "__main__":
    unittest.main()
