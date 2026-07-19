from __future__ import annotations

import json
import unittest
from pathlib import Path

from psse_env.providers import MatpowerDeploymentProviders
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

    def test_context_flags_injected_error_and_supports_group_correction(self) -> None:
        metrics = self.providers.get_measurement_context(self.state)
        self.assertNotIn("execution_status", metrics)
        findings = metrics["measurement_findings"]
        self.assertTrue(findings)
        flagged = {item["index0"] for item in findings}
        self.assertIn(self.error_index, flagged)
        supported = metrics["supported_corrections"]
        self.assertTrue(supported)
        group = supported[0]["arguments"]["suspect_group"]
        self.assertEqual(sorted(group), sorted(flagged))
        self.assertEqual(supported[0]["tool"], "correct_measurements")

    def test_lambda_contexts_expose_branch_targets(self) -> None:
        parameter = self.providers.get_parameter_context(self.state)
        self.assertNotIn("execution_status", parameter)
        self.assertIn("parameter_findings", parameter)
        for proposal in parameter["supported_corrections"]:
            self.assertEqual(proposal["tool"], "correct_parameters")
            self.assertGreaterEqual(proposal["arguments"]["line_index"], 1)
        topology = self.providers.get_topology_context(self.state)
        self.assertNotIn("execution_status", topology)
        for proposal in topology["supported_corrections"]:
            self.assertEqual(proposal["tool"], "correct_topology")
            self.assertIn(proposal["arguments"]["status"], (0, 1))


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
        # The grouped correction must reduce the observable WLS objective.
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
