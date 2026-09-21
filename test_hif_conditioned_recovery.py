from __future__ import annotations

from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from three_phase_nlm.hif_conditioned_recovery import replay_hif_measurement_effect


def _fit() -> dict:
    return {
        "success": True,
        "candidate_branch_row0": 2,
        "estimated": {
            "alpha_from_from_bus": 0.45,
            "phase": "A",
            "r_hif_pu": 100.0,
            "resistance_model": "shared",
        },
        "fit": {"weighted_residual_norm": 0.8, "ambiguity": True},
        "parameter_identifiable": False,
    }


class HIFConditionedRecoveryTests(unittest.TestCase):
    def _replay(self, fit: dict | None = None, **kwargs):
        return replay_hif_measurement_effect(
            _fit() if fit is None else fit,
            op_point=kwargs.pop("op_point", {"load_scale": 0.91}),
            scan_index=kwargs.pop("scan_index", 3),
            snapshot_id=kwargs.pop("snapshot_id", "snapshot-3"),
            **kwargs,
        )

    @patch("three_phase_nlm.hif_conditioned_recovery._simulate_base")
    @patch("three_phase_nlm.hif_conditioned_recovery.simulate_hif_candidate")
    def test_paired_replay_preserves_context_identity_and_uncertainty(self, present, absent):
        present.return_value = {"z": np.linspace(1.0, 2.0, 122)}
        absent.return_value = {"z": np.ones(122)}
        fit = _fit()
        before = deepcopy(fit)
        op = {"load_scale": 0.91, "bus_load_scales": {"b2": 0.93}}
        result = self._replay(fit, op_point=op, time_tag="2026-09-16T10:00:00Z")
        self.assertEqual(fit, before)
        self.assertEqual(op, {"load_scale": 0.91, "bus_load_scales": {"b2": 0.93}})
        self.assertEqual(present.call_args.kwargs["op_point"], absent.call_args.kwargs["op_point"])
        self.assertEqual(Path(present.call_args.kwargs["pristine_model_dir"]), absent.call_args.args[0])
        self.assertEqual(result["binding"], {
            "snapshot_id": "snapshot-3", "scan_index": 3, "time_tag": "2026-09-16T10:00:00Z"
        })
        self.assertEqual(len(set(result["channel_ids"])), 122)
        self.assertEqual(result["channel_ids"][2], "Vm:b3:phase_A")
        self.assertEqual(result["channel_ids"][44], "Pf:Line.2-3:external_terminal")
        self.assertEqual(result["channel_ids"][84], "Pt:Line.2-3:external_terminal")
        self.assertFalse(result["diagnostic_fit_evidence"]["parameter_identifiable"])
        self.assertTrue(result["physical_fault_still_present"])
        self.assertFalse(result["uncertainty_statement"]["effect_covariance_available"])
        np.testing.assert_allclose(result["measurement_effect"], np.linspace(0.0, 1.0, 122))

    @patch("three_phase_nlm.hif_conditioned_recovery._simulate_base", return_value={"z": [1.0] * 122})
    @patch("three_phase_nlm.hif_conditioned_recovery.simulate_hif_candidate", return_value={"z": [1.0] * 122})
    def test_current_scan_uses_its_resistance_not_the_window_median(self, present, absent):
        fit = _fit()
        fit["estimated"].update({
            "resistance_model": "scan_specific_smooth",
            "per_scan_r_hif_pu": [
                {"scan_index": 3, "r_hif_pu": 25.0},
                {"scan_index": 8, "r_hif_pu": 180.0},
            ],
        })
        result = self._replay(fit)
        self.assertEqual(present.call_args.kwargs["r_hif_pu"], 25.0)
        self.assertEqual(result["parameters_used"]["r_hif_pu"], 25.0)
        with self.assertRaisesRegex(ValueError, "no fitted resistance for current scan 4"):
            self._replay(fit, scan_index=4)
        self.assertEqual(present.call_count, 1)
        self.assertEqual(absent.call_count, 1)

    @patch("three_phase_nlm.hif_conditioned_recovery.simulate_hif_candidate")
    def test_scan_specific_requires_unique_explicit_current_scan_fit(self, present):
        fit = _fit()
        fit["estimated"]["resistance_model"] = "scan_specific_smooth"
        with self.assertRaisesRegex(ValueError, "requires per_scan_r_hif_pu"):
            self._replay(fit)
        fit["estimated"]["per_scan_r_hif_pu"] = [
            {"scan_index": 3, "r_hif_pu": 20.0},
            {"scan_index": 3, "r_hif_pu": 40.0},
        ]
        with self.assertRaisesRegex(ValueError, "duplicate resistance"):
            self._replay(fit)
        present.assert_not_called()

    @patch("three_phase_nlm.hif_conditioned_recovery.simulate_hif_candidate")
    def test_invalid_parameters_fail_before_any_physical_replay(self, present):
        for field, value in (("alpha_from_from_bus", 0.0), ("alpha_from_from_bus", 1.0),
                             ("r_hif_pu", float("nan")), ("r_hif_pu", -1), ("phase", "AB")):
            with self.subTest(field=field, value=value):
                fit = _fit()
                fit["estimated"][field] = value
                with self.assertRaises(ValueError):
                    self._replay(fit)
        with self.assertRaisesRegex(ValueError, "snapshot_id"):
            self._replay(snapshot_id="")
        with self.assertRaisesRegex(ValueError, "scan_index"):
            self._replay(scan_index=3.2)
        with self.assertRaisesRegex(ValueError, "operating point"):
            self._replay(op_point=None)
        fit = _fit()
        fit["candidate_branch_row0"] = 7
        with self.assertRaisesRegex(ValueError, "Line"):
            self._replay(fit)
        present.assert_not_called()

    @patch("three_phase_nlm.hif_conditioned_recovery.simulate_hif_candidate")
    def test_explicit_missing_model_does_not_fall_back_to_default(self, present):
        with self.assertRaisesRegex(ValueError, "pristine_model_dir"):
            self._replay(pristine_model_dir="__nonexistent_hif_replay_model__")
        present.assert_not_called()

    @patch("three_phase_nlm.hif_conditioned_recovery._simulate_base", return_value={"z": [1.0] * 122})
    @patch("three_phase_nlm.hif_conditioned_recovery.simulate_hif_candidate")
    def test_invalid_simulation_output_cannot_become_a_compensation(self, present, absent):
        for values in ([1.0] * 123, [float("nan")] * 122):
            with self.subTest(length=len(values)):
                present.return_value = {"z": values}
                with self.assertRaisesRegex(ValueError, "122 finite external measurements"):
                    self._replay()

    @unittest.skipUnless(importlib.util.find_spec("opendssdirect"), "opendssdirect is not installed")
    def test_real_replay_preserves_split_line_and_same_channel_meter_error(self):
        result = self._replay()
        present = np.asarray(result["predicted_hif_measurements"])
        absent = np.asarray(result["predicted_base_measurements"])
        effect = np.asarray(result["measurement_effect"])
        self.assertEqual(present.shape, (122,))
        # A hidden midspan shunt changes the external terminal power balance;
        # the two terminals still occupy the original 20-branch channel map.
        self.assertGreater(float(np.linalg.norm(effect)), 1e-4)
        branch = result["parameters_used"]["candidate_branch_row0"]
        from_index, to_index = 42 + branch, 82 + branch
        self.assertGreater(
            abs((present[from_index] + present[to_index]) - (absent[from_index] + absent[to_index])),
            1e-5,
        )
        corrupted = present.copy()
        corrupted[from_index] += 0.08
        remaining = corrupted - effect - absent
        expected = np.zeros(122)
        expected[from_index] = 0.08
        np.testing.assert_allclose(remaining, expected, atol=1e-12)
        self.assertEqual(result["measurement_semantics"]["voltage"], "phase_A_line_to_neutral_magnitude_pu")


if __name__ == "__main__":
    unittest.main()
