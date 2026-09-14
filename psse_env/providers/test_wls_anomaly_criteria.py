"""Global and local WLS alarms must agree across routing and verification."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from psse_env.providers.matpower import MatpowerDeploymentProviders


class WlsAnomalyCriteriaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fixture = json.loads((Path(__file__).parent / "fixtures/case14_z.json").read_text())
        cls.state = {
            "state_id": "alarm:s0", "state_hash": "initial",
            "case": fixture["case_path"], "measurements": fixture["z_obs"],
            "metadata": {},
        }
        cls.solved = MatpowerDeploymentProviders()._solve(cls.state)

    def provider_and_solve(self, *, chi_ratio, maximum):
        provider = MatpowerDeploymentProviders(chi2_alpha=0.05, normalized_residual_threshold=4.0)
        solved = copy.deepcopy(self.solved)
        limit = provider._wls_detection_metrics(solved)["chi_square_threshold"]
        solved["payload"]["global_residual_sum"] = limit * chi_ratio
        solved["payload"]["r"] = [0.0] * len(solved["payload"]["r"])
        solved["payload"]["r"][1] = maximum
        solved["payload"]["lambdaN"] = [0.0] * len(solved["payload"]["lambdaN"])
        return provider, solved

    def test_either_alarm_prevents_clean_and_emits_local_evidence(self):
        for chi_ratio, maximum, chi_alarm, local_alarm in (
            (0.9, 3.99, False, False),
            (1.0, 0.5, True, False),
            (0.9, 4.0, False, True),
            (1.1, 8.0, True, True),
        ):
            with self.subTest(chi_ratio=chi_ratio, maximum=maximum):
                provider, solved = self.provider_and_solve(chi_ratio=chi_ratio, maximum=maximum)
                with patch.object(provider, "_solve", return_value=solved):
                    metrics = provider.run_wls(self.state)
                self.assertEqual(metrics["chi_square_alarm"], chi_alarm)
                self.assertEqual(metrics["normalized_residual_alarm"], local_alarm)
                self.assertEqual(metrics["no_material_anomaly_remaining"], not (chi_alarm or local_alarm))
                self.assertEqual(metrics["globally_resolved"], not (chi_alarm or local_alarm))
                self.assertAlmostEqual(metrics["remaining_anomaly_score"], max(chi_ratio, maximum / 4.0))
                self.assertAlmostEqual(provider._remaining_anomaly_score(solved), metrics["remaining_anomaly_score"])
                if local_alarm:
                    self.assertTrue(any("index=1" in signature for signature in metrics["unresolved_signatures"]))

    def test_candidate_cannot_resolve_while_another_meter_exceeds_local_limit(self):
        provider, solved = self.provider_and_solve(chi_ratio=0.8, maximum=9.0)
        candidate = copy.deepcopy(self.state)
        candidate.update(status="candidate", source_action={
            "tool": "correct_measurements", "arguments": {"suspect_group": [0]},
        })
        with patch.object(provider, "_solve", return_value=solved):
            metrics = provider.run_wls(candidate)
        self.assertTrue(metrics["target_fixed"])
        self.assertFalse(metrics["post_action_resolved"])
        self.assertFalse(metrics["globally_resolved"])

    def test_post_measurement_parameter_screening_runs_on_residual_only_alarm(self):
        provider, solved = self.provider_and_solve(chi_ratio=0.8, maximum=9.0)
        state = copy.deepcopy(self.state)
        state["policy_observation"] = {"accepted_corrections": [{
            "source_action": {"tool": "correct_measurements", "arguments": {"suspect_group": [0]}}
        }]}
        with patch.object(provider, "_solve", return_value=solved), patch.object(
            provider, "_post_measurement_branch_route_screening", return_value={}
        ) as screening:
            context = provider.get_measurement_context(state)
        self.assertTrue(screening.call_args.kwargs["anomaly_unresolved"])
        self.assertTrue(context["normalized_residual_alarm"])

    def test_legacy_mode_and_summary_use_configured_chi_square_level(self):
        _, solved = self.provider_and_solve(chi_ratio=0.8, maximum=9.0)
        provider = MatpowerDeploymentProviders(chi2_alpha=0.01)
        with patch.object(provider, "_solve", return_value=solved):
            metrics = provider.run_wls(self.state)
        self.assertTrue(metrics["no_material_anomaly_remaining"])
        self.assertFalse(metrics["normalized_residual_alarm"])
        self.assertIsNone(metrics["normalized_residual_threshold"])
        self.assertEqual(metrics["anomaly_detection_rule"], "chi_square_only")
        self.assertAlmostEqual(
            metrics["wls_summary"]["global_metrics"]["global_residual_threshold"],
            metrics["chi_square_threshold"], places=3,
        )

    def test_nonfinite_evidence_fails_closed(self):
        for value in (float("nan"), float("inf")):
            provider, solved = self.provider_and_solve(chi_ratio=0.8, maximum=value)
            with patch.object(provider, "_solve", return_value=solved):
                metrics = provider.run_wls(self.state)
            self.assertEqual(metrics["execution_status"], "failure")
            self.assertEqual(metrics["error_code"], "wls_evidence_error")

    def test_invalid_alarm_configuration_is_rejected(self):
        for alpha in (0, 1, -0.1, float("nan"), float("inf")):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                MatpowerDeploymentProviders(chi2_alpha=alpha)
        for threshold in (0, -1, float("nan"), float("inf")):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                MatpowerDeploymentProviders(normalized_residual_threshold=threshold)


if __name__ == "__main__":
    unittest.main()
