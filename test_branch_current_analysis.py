"""Pure-numpy tests for the per-phase branch-current analysis module.

These tests use KCL-consistent synthetic telemetry built from the IEEE-14
line parameters, so they run without OpenDSS.  The OpenDSS-backed checks live
in ``test_export_measurement_series.py`` and ``test_hif_multiscan_estimator.py``.
"""

from __future__ import annotations

import unittest

import numpy as np

from three_phase_nlm.branch_current_analysis import (
    DEFAULT_BRANCH_CURRENT_SIGMA_PU,
    add_branch_current_noise,
    balanced_branch_current_control,
    branch_current_rows_to_phasors,
    branch_current_rows_valid,
    bus_shunt_power_unbalance,
    line_differential_currents,
    line_differential_null_test,
    sequence_components,
    terminal_current_hif_localization,
    two_terminal_hif_estimate,
    unbalance_source_localization,
)
from three_phase_nlm.synthetic_branch_telemetry import (
    balanced_phasors,
    disjoint_healthy_line,
    synthetic_line_fault_rows,
    synthetic_unbalance_rows,
)


class SequenceComponentTests(unittest.TestCase):
    def test_balanced_set_is_pure_positive_sequence(self) -> None:
        zero, positive, negative = sequence_components(*balanced_phasors(1.0, 10.0))
        self.assertAlmostEqual(abs(zero), 0.0, places=12)
        self.assertAlmostEqual(abs(negative), 0.0, places=12)
        self.assertAlmostEqual(abs(positive), 1.0, places=12)

    def test_row_parsing_rejects_malformed_entries(self) -> None:
        rows = [
            {
                "branch_row0": 0,
                "from_bus": "b1",
                "to_bus": "b2",
                "i_from_pu": [1, 1],
                "ang_from_deg": [0, 0],
                "i_to_pu": [1, 1, 1],
                "ang_to_deg": [0, 0, 0],
            },
            {
                "branch_row0": 1,
                "from_bus": "b1",
                "to_bus": "b5",
                "i_from_pu": [1, 1, 1],
                "ang_from_deg": [0, 0, 0],
                "i_to_pu": [1, 1, 1],
                "ang_to_deg": [0, 0, 0],
            },
        ]
        parsed = branch_current_rows_to_phasors(rows)
        self.assertEqual(sorted(parsed), [1])
        self.assertFalse(branch_current_rows_valid(rows, expected_branches=2))
        self.assertTrue(branch_current_rows_valid(rows[1:], expected_branches=1))


class UnbalanceLocalizationTests(unittest.TestCase):
    def test_source_bus_has_dominant_phase_power_spread(self) -> None:
        for source in (2, 3):
            with self.subTest(source=source):
                voltage_rows, current_rows = synthetic_unbalance_rows(
                    source_bus=source, split=(0.5, 0.3, 0.2)
                )
                ranking = bus_shunt_power_unbalance(voltage_rows, current_rows)
                self.assertEqual(ranking[0]["bus"], source)
                self.assertAlmostEqual(ranking[0]["phase_power_spread_rel"], 0.5, places=9)
                # The balanced load draws exactly equal per-phase power; the
                # source (bus 1) absorbs the unbalance but at a lower relative spread.
                by_bus = {item["bus"]: item for item in ranking}
                other_load = 3 if source == 2 else 2
                self.assertLess(by_bus[other_load]["phase_power_spread_rel"], 1e-9)
                self.assertLess(by_bus[1]["phase_power_spread_rel"], 0.5)
                localization = unbalance_source_localization(voltage_rows, current_rows, top_k=3)
                self.assertEqual(localization["bus_1based"], source)
                self.assertEqual(localization["top_unbalance_source_buses"][0]["bus"], source)
                self.assertGreater(localization["separation_ratio"], 1.0)

    def test_balanced_control_has_no_source_and_passes_null(self) -> None:
        voltage_rows, current_rows = synthetic_unbalance_rows(
            source_bus=2, split=(1 / 3, 1 / 3, 1 / 3)
        )
        ranking = bus_shunt_power_unbalance(voltage_rows, current_rows)
        self.assertTrue(all(item["phase_power_spread_rel"] < 1e-9 for item in ranking))
        null = line_differential_null_test(voltage_rows, current_rows)
        self.assertFalse(null["hif_like_differential_present"])
        balanced = balanced_branch_current_control(current_rows)
        self.assertEqual(len(balanced), len(current_rows))
        for row in balanced:
            self.assertEqual(len(set(row["i_from_pu"])), 1)
            a, b, c = row["ang_from_deg"]
            self.assertAlmostEqual(a - b, 120.0)
            self.assertAlmostEqual(c - a, 120.0)


class TwoTerminalHIFTests(unittest.TestCase):
    CASES = ((2, 0.37, "B", 100.0), (0, 0.8, "A", 20.0), (16, 0.25, "C", 500.0))

    def test_disjoint_healthy_line_shares_no_bus(self) -> None:
        from three_phase_nlm.branch_current_analysis import case14_line_parameters

        params = case14_line_parameters()
        for row0, *_ in self.CASES:
            healthy = params[disjoint_healthy_line(row0)]
            faulted = params[row0]
            self.assertFalse(
                {healthy["from_bus"], healthy["to_bus"]}
                & {faulted["from_bus"], faulted["to_bus"]}
            )

    def test_closed_form_recovers_position_phase_and_resistance(self) -> None:
        for row0, alpha, phase, r_hif_pu in self.CASES:
            with self.subTest(row0=row0, alpha=alpha, phase=phase, r=r_hif_pu):
                voltage_rows, current_rows, i_fault = synthetic_line_fault_rows(
                    row0=row0, alpha=alpha, phase=phase, r_hif_pu=r_hif_pu
                )
                ranking = line_differential_currents(voltage_rows, current_rows)
                self.assertEqual(ranking[0]["branch_row0"], row0)
                self.assertEqual(ranking[0]["phase"], phase)
                # The differential removes healthy-line charging, so it matches
                # the fault current up to the small charging redistribution.
                self.assertAlmostEqual(
                    ranking[0]["score"] / abs(i_fault), 1.0, delta=0.01
                )
                self.assertLess(ranking[1]["score"], 1e-9)
                estimate = two_terminal_hif_estimate(
                    voltage_rows, current_rows, branch_row0=row0, phase=phase
                )
                self.assertAlmostEqual(estimate["alpha_from_from_bus"], alpha, places=6)
                self.assertAlmostEqual(estimate["r_hif_pu"] / r_hif_pu, 1.0, places=5)
                self.assertAlmostEqual(estimate["x_hif_pu"], 0.0, places=4)
                self.assertAlmostEqual(estimate["i_hif_pu"] / abs(i_fault), 1.0, places=6)
                self.assertLess(estimate["fit_mismatch_pu"], 1e-9)
                self.assertLess(estimate["consistency_ratio"], 1e-6)
                self.assertGreater(estimate["line_impedance_pu"], 0.0)

    def test_bad_terminal_sensor_looks_like_a_fault_at_that_terminal(self) -> None:
        voltage_rows, current_rows, _ = synthetic_line_fault_rows(
            row0=2, alpha=0.37, phase="B", r_hif_pu=100.0
        )
        # Replace the genuine fault with a spurious 1% error on the to-terminal
        # current of a healthy line.  The differential is real, and the only
        # (alpha, R) that reconciles both ends puts the "fault" at the corrupted
        # terminal: the estimate must flag that endpoint ambiguity.
        healthy_rows = [row for row in current_rows if row["branch_row0"] != 2]
        healthy_row0 = healthy_rows[0]["branch_row0"]
        corrupted = [dict(row) for row in healthy_rows]
        corrupted[0]["i_to_pu"] = [corrupted[0]["i_to_pu"][0] + 0.01, *corrupted[0]["i_to_pu"][1:]]
        payload = terminal_current_hif_localization(voltage_rows, corrupted, sigma_pu=1e-3)
        self.assertEqual(payload["top_hif_groups"][0]["branch_row0"], healthy_row0)
        self.assertTrue(payload["differential_detected"])
        estimate = payload["terminal_current_estimate"]
        self.assertGreater(estimate["alpha_from_from_bus"], 0.98)
        self.assertTrue(estimate["endpoint_ambiguous"])
        # A genuine interior fault is not ambiguous.
        genuine = terminal_current_hif_localization(voltage_rows, current_rows, sigma_pu=1e-3)
        self.assertFalse(genuine["terminal_current_estimate"]["endpoint_ambiguous"])

    def test_localization_payload_has_nlm_shape_and_phase_evidence(self) -> None:
        voltage_rows, current_rows, _ = synthetic_line_fault_rows(
            row0=2, alpha=0.37, phase="B", r_hif_pu=100.0
        )
        payload = terminal_current_hif_localization(voltage_rows, current_rows, top_k=3)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["method"], "terminal_current_differential")
        self.assertEqual(payload["top_hif_groups"][0]["branch_row0"], 2)
        self.assertEqual(payload["top_hif_groups"][0]["line_index1"], 3)
        self.assertEqual(payload["top_hif_groups"][0]["dss_element"], "Line.2-3")
        self.assertEqual(payload["suspected_phase"], "B")
        self.assertTrue(payload["differential_detected"])
        self.assertGreater(payload["separation_ratio"], 5.0)
        self.assertAlmostEqual(
            payload["terminal_current_estimate"]["alpha_from_from_bus"], 0.37, places=6
        )
        self.assertNotIn("detected", payload)

    def test_null_test_flags_hif_like_differential_only_under_fault(self) -> None:
        voltage_rows, current_rows, _ = synthetic_line_fault_rows(
            row0=2, alpha=0.37, phase="B", r_hif_pu=100.0
        )
        faulted = line_differential_null_test(
            voltage_rows, current_rows, sigma_pu=DEFAULT_BRANCH_CURRENT_SIGMA_PU
        )
        self.assertTrue(faulted["hif_like_differential_present"])
        self.assertEqual(faulted["max_differential_branch_row0"], 2)
        healthy_rows = [row for row in current_rows if row["branch_row0"] != 2]
        healthy = line_differential_null_test(
            voltage_rows, healthy_rows, sigma_pu=DEFAULT_BRANCH_CURRENT_SIGMA_PU
        )
        self.assertFalse(healthy["hif_like_differential_present"])

    def test_noise_helper_perturbs_components_deterministically(self) -> None:
        _, current_rows, _ = synthetic_line_fault_rows(
            row0=2, alpha=0.37, phase="B", r_hif_pu=100.0
        )
        noisy_a = add_branch_current_noise(current_rows, np.random.default_rng(3), 1e-3)
        noisy_b = add_branch_current_noise(current_rows, np.random.default_rng(3), 1e-3)
        self.assertEqual(noisy_a, noisy_b)
        clean = branch_current_rows_to_phasors(current_rows)
        noisy = branch_current_rows_to_phasors(noisy_a)
        deltas = [
            abs(noisy[row0]["i_from"][index] - clean[row0]["i_from"][index])
            for row0 in clean
            for index in range(3)
        ]
        self.assertGreater(max(deltas), 0.0)
        self.assertLess(max(deltas), 1e-2)
        self.assertEqual(
            add_branch_current_noise(current_rows, np.random.default_rng(3), 0.0), current_rows
        )

    def test_fault_survives_sensor_noise_when_above_detection_floor(self) -> None:
        # I_fault ~ 1/R = 0.01 pu; the 6-sigma floor at sigma=1e-3 is ~8.5e-3 pu.
        voltage_rows, current_rows, _ = synthetic_line_fault_rows(
            row0=10, alpha=0.6, phase="C", r_hif_pu=100.0
        )
        noisy = add_branch_current_noise(current_rows, np.random.default_rng(11), 1e-3)
        payload = terminal_current_hif_localization(voltage_rows, noisy, sigma_pu=1e-3)
        self.assertEqual(payload["top_hif_groups"][0]["branch_row0"], 10)
        self.assertEqual(payload["suspected_phase"], "C")
        self.assertTrue(payload["differential_detected"])

    def test_multiscan_aggregation_lowers_the_noise_floor(self) -> None:
        from three_phase_nlm.branch_current_analysis import (
            terminal_current_hif_localization_multiscan,
        )

        voltage_rows, current_rows, _ = synthetic_line_fault_rows(
            row0=10, alpha=0.6, phase="C", r_hif_pu=200.0
        )
        scans = [
            {
                "scan_index": index,
                "three_phase_voltages": voltage_rows,
                "three_phase_branch_currents": add_branch_current_noise(
                    current_rows, np.random.default_rng(100 + index), 1e-3
                ),
            }
            for index in range(9)
        ]
        payload = terminal_current_hif_localization_multiscan(scans, sigma_pu=1e-3)
        self.assertEqual(payload["scan_count"], 9)
        self.assertEqual(payload["aggregation"], "coherent_mean_differential_across_scans")
        self.assertEqual(payload["top_hif_groups"][0]["branch_row0"], 10)
        self.assertEqual(payload["suspected_phase"], "C")
        # Floor scales as 6*sqrt(2)*sigma/sqrt(N): 8.5e-3 -> 2.8e-3 for nine scans.
        self.assertAlmostEqual(
            payload["differential_detection_floor_pu"], 6.0 * (2.0**0.5) * 1e-3 / 3.0, places=12
        )
        self.assertTrue(payload["differential_detected"])
        estimate = payload["terminal_current_estimate"]
        self.assertEqual(estimate["scan_count"], 9)
        self.assertLess(abs(estimate["alpha_from_from_bus"] - 0.6), 0.15)
        self.assertEqual(len(estimate["per_scan"]), 9)
        single = terminal_current_hif_localization_multiscan(scans[:1], sigma_pu=1e-3)
        self.assertEqual(single["scan_count"], 1)
        self.assertNotIn("aggregation", single)

    def test_detection_floor_tracks_declared_sensor_sigma(self) -> None:
        # A 200 pu fault draws ~5e-3 pu: below the 6-sigma floor at sigma=1e-3
        # (still ranked first, but not claimed as detected), above it at 2e-4.
        voltage_rows, current_rows, _ = synthetic_line_fault_rows(
            row0=10, alpha=0.6, phase="C", r_hif_pu=200.0
        )
        coarse = terminal_current_hif_localization(voltage_rows, current_rows, sigma_pu=1e-3)
        self.assertEqual(coarse["top_hif_groups"][0]["branch_row0"], 10)
        self.assertFalse(coarse["differential_detected"])
        fine = terminal_current_hif_localization(voltage_rows, current_rows, sigma_pu=2e-4)
        self.assertTrue(fine["differential_detected"])
        self.assertAlmostEqual(
            fine["differential_detection_floor_pu"], 6.0 * (2.0**0.5) * 2e-4, places=12
        )


if __name__ == "__main__":
    unittest.main()
