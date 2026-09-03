from __future__ import annotations

import math
import statistics
import tempfile
import unittest
from pathlib import Path

import numpy as np
import opendssdirect as dss

from IEEE_14_OpenDSS.constants import BRANCH_ORDER, BUS_ORDER
from IEEE_14_OpenDSS.export_measurement_series import (
    _compile_and_solve,
    _phase_terminal_complex_currents,
    _phase_terminal_complex_powers,
    element_pq_3ph_per_terminal,
    extract_measurement_series,
    extract_three_phase_branch_current_measurements,
    extract_three_phase_voltage_measurements,
)
from three_phase_nlm.branch_current_analysis import (
    branch_current_rows_to_phasors,
    bus_shunt_injections,
    line_differential_currents,
    terminal_current_hif_localization,
    voltage_rows_to_phasors,
)
from three_phase_nlm.dss_hif_injector import (
    copy_ieee14_model,
    inject_midspan_hif_ieee14,
    write_balanced_ieee14_load_override,
)


REPO_ROOT = Path(__file__).resolve().parent
MODEL_DIR = REPO_ROOT / "IEEE_14_OpenDSS"
TRANSFORMERS = [name for name in BRANCH_ORDER if name.lower().startswith("transformer.")]


class TerminalConductorIndexingTests(unittest.TestCase):
    def test_four_conductor_terminals_exclude_neutral(self) -> None:
        values = [
            10 + 1j,
            20 + 2j,
            30 + 3j,
            999 + 999j,
            -11 - 1j,
            -21 - 2j,
            -31 - 3j,
            888 + 888j,
        ]
        powers = [component for value in values for component in (value.real, value.imag)]

        terminals = _phase_terminal_complex_powers(
            powers,
            [1, 2, 3, 0, 1, 2, 3, 0],
            n_conductors=4,
            n_terminals=2,
        )

        self.assertEqual(terminals, [60 + 6j, -63 - 6j])


class OpenDSSExportPhysicsTests(unittest.TestCase):
    def setUp(self) -> None:
        caller_cwd = Path.cwd()
        _compile_and_solve(str(MODEL_DIR))
        self.assertEqual(Path.cwd(), caller_cwd)
        self.assertTrue(dss.Solution.Converged())

    def test_transformer_terminal_sums_match_opendss_losses(self) -> None:
        for element in TRANSFORMERS:
            with self.subTest(element=element):
                self.assertTrue(dss.Circuit.SetActiveElement(element))
                terminals = element_pq_3ph_per_terminal()
                self.assertEqual(len(terminals), 2)
                losses = [float(value) for value in dss.CktElement.Losses()]

                self.assertAlmostEqual(
                    sum(value[0] for value in terminals), losses[0] / 1e6, places=9
                )
                self.assertAlmostEqual(
                    sum(value[1] for value in terminals), losses[1] / 1e6, places=9
                )
                ratio = abs(terminals[1][0]) / max(abs(terminals[0][0]), 1e-12)
                self.assertGreater(ratio, 0.95)
                self.assertLess(ratio, 1.05)

    def test_measurement_vector_order_and_bus_balance(self) -> None:
        series, buses, branches = extract_measurement_series()
        self.assertEqual(buses, BUS_ORDER)
        self.assertEqual(branches, BRANCH_ORDER)
        self.assertEqual(len(series), 122)
        self.assertTrue(np.isfinite(np.asarray(series, dtype=float)).all())

        nb = len(BUS_ORDER)
        nl = len(BRANCH_ORDER)
        values = np.asarray(series, dtype=float)
        pinj = values[nb : 2 * nb]
        qinj = values[2 * nb : 3 * nb]
        pf = values[3 * nb : 3 * nb + nl]
        qf = values[3 * nb + nl : 3 * nb + 2 * nl]
        pt = values[3 * nb + 2 * nl : 3 * nb + 3 * nl]
        qt = values[3 * nb + 3 * nl : 3 * nb + 4 * nl]

        terminal_p = {bus: 0.0 for bus in BUS_ORDER}
        terminal_q = {bus: 0.0 for bus in BUS_ORDER}
        for index, element in enumerate(BRANCH_ORDER):
            self.assertTrue(dss.Circuit.SetActiveElement(element))
            element_buses = [
                str(value).split(".")[0].lower() for value in dss.CktElement.BusNames()
            ]
            terminal_p[element_buses[0]] += float(pf[index])
            terminal_q[element_buses[0]] += float(qf[index])
            terminal_p[element_buses[1]] += float(pt[index])
            terminal_q[element_buses[1]] += float(qt[index])

        residuals = {
            bus: math.hypot(
                float(pinj[index]) - terminal_p[bus],
                float(qinj[index]) - terminal_q[bus],
            )
            for index, bus in enumerate(BUS_ORDER)
        }
        secondary = [residuals[bus] for bus in ("b6", "b7", "b9")]
        others = [
            residual for bus, residual in residuals.items() if bus not in {"b6", "b7", "b9"}
        ]
        self.assertLessEqual(
            statistics.median(secondary),
            max(0.03, 2.0 * statistics.median(others)),
        )

    def test_hidden_split_line_preserves_external_vector_mapping(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ieee14_export_hif_") as temporary:
            scenario_dir = Path(temporary)
            copy_ieee14_model(MODEL_DIR, scenario_dir, overwrite=True)
            write_balanced_ieee14_load_override(scenario_dir)
            injection = inject_midspan_hif_ieee14(
                scenario_dir,
                "Line.7-8",
                split_ratio=0.47,
                phase="A",
                r_hif_ohm=1.0,
                fault_bus="Fault_7_8_EXPORT_TEST",
            )
            _compile_and_solve(str(scenario_dir))
            self.assertTrue(dss.Solution.Converged())

            series, buses, branches = extract_measurement_series(
                branch_element_overrides=injection.branch_element_overrides
            )

            self.assertEqual(len(series), 122)
            self.assertEqual(buses, BUS_ORDER)
            self.assertEqual(branches, BRANCH_ORDER)
            self.assertNotIn(injection.fault_bus.lower(), {bus.lower() for bus in buses})
            self.assertEqual(branches[13], "Line.7-8")


class TerminalCurrentIndexingTests(unittest.TestCase):
    def test_phase_conductors_follow_node_numbers_not_positions(self) -> None:
        values = [1 + 0j, 2 + 0j, 3 + 0j, 99 + 99j, -1 + 0j, -2 + 0j, -3 + 0j, 88 + 88j]
        currents = [component for value in values for component in (value.real, value.imag)]
        # Terminal 1 lists nodes out of order (2, 1, 3, neutral).
        terminals = _phase_terminal_complex_currents(
            currents, [2, 1, 3, 0, 1, 2, 3, 0], n_conductors=4, n_terminals=2
        )
        self.assertEqual(terminals[0], [2 + 0j, 1 + 0j, 3 + 0j])
        self.assertEqual(terminals[1], [-1 + 0j, -2 + 0j, -3 + 0j])


class BranchCurrentExportPhysicsTests(unittest.TestCase):
    def setUp(self) -> None:
        _compile_and_solve(str(MODEL_DIR))
        self.assertTrue(dss.Solution.Converged())

    def test_rows_follow_branch_order_with_finite_per_unit_phasors(self) -> None:
        rows = extract_three_phase_branch_current_measurements()
        self.assertEqual([row["branch"] for row in rows], BRANCH_ORDER)
        self.assertEqual([row["branch_row0"] for row in rows], list(range(len(BRANCH_ORDER))))
        parsed = branch_current_rows_to_phasors(rows)
        self.assertEqual(len(parsed), len(BRANCH_ORDER))
        for row in rows:
            self.assertGreater(row["ibase_from_a"], 0.0)
            self.assertTrue(all(math.isfinite(value) for value in row["i_from_pu"]))
            self.assertTrue(all(0.0 < value < 5.0 for value in row["i_from_pu"]))

    def test_terminal_currents_satisfy_kcl_against_bus_injections(self) -> None:
        series, _, _ = extract_measurement_series()
        voltages = voltage_rows_to_phasors(extract_three_phase_voltage_measurements())
        currents = branch_current_rows_to_phasors(
            extract_three_phase_branch_current_measurements()
        )
        injections = bus_shunt_injections(voltages, currents)
        nb = len(BUS_ORDER)
        worst = 0.0
        for index, _bus in enumerate(BUS_ORDER):
            bus = index + 1
            complex_power = sum(
                voltages[bus][phase] * injections[bus][phase].conjugate() for phase in range(3)
            )
            # Bus injections in z are three-phase totals on the 100 MVA base;
            # the per-phase per-unit powers sum to three times that.
            worst = max(
                worst,
                abs(complex_power.real / 3.0 - float(series[nb + index])),
                abs(complex_power.imag / 3.0 - float(series[2 * nb + index])),
            )
        self.assertLess(worst, 1e-3)

    def test_healthy_lines_carry_only_modeled_charging_differential(self) -> None:
        ranking = line_differential_currents(
            extract_three_phase_voltage_measurements(),
            extract_three_phase_branch_current_measurements(),
        )
        self.assertEqual(len(ranking), 17)
        self.assertLess(ranking[0]["score"], 1e-8)

    def test_hidden_split_line_currents_localize_the_fault(self) -> None:
        with tempfile.TemporaryDirectory(prefix="ieee14_export_hif_currents_") as temporary:
            scenario_dir = Path(temporary)
            copy_ieee14_model(MODEL_DIR, scenario_dir, overwrite=True)
            write_balanced_ieee14_load_override(scenario_dir)
            injection = inject_midspan_hif_ieee14(
                scenario_dir,
                "Line.7-8",
                split_ratio=0.47,
                phase="A",
                r_hif_ohm=1.0,  # 100 pu on the 0.01 ohm base
                fault_bus="Fault_7_8_CURRENT_TEST",
            )
            _compile_and_solve(str(scenario_dir))
            self.assertTrue(dss.Solution.Converged())
            voltages = extract_three_phase_voltage_measurements()
            currents = extract_three_phase_branch_current_measurements(
                branch_element_overrides=injection.branch_element_overrides
            )

        self.assertEqual([row["branch"] for row in currents], BRANCH_ORDER)
        self.assertEqual(currents[13]["from_bus"], "b7")
        self.assertEqual(currents[13]["to_bus"], "b8")
        payload = terminal_current_hif_localization(voltages, currents, top_k=3)
        self.assertEqual(payload["top_hif_groups"][0]["branch_row0"], 13)
        self.assertEqual(payload["suspected_phase"], "A")
        # Noise-free telemetry: every healthy line has zero differential, and
        # the separation ratio is bounded only by the declared sensor floor.
        self.assertLess(payload["second_line_differential_pu"], 1e-8)
        self.assertTrue(payload["differential_detected"])
        self.assertGreater(payload["separation_ratio"], 5.0)
        estimate = payload["terminal_current_estimate"]
        self.assertAlmostEqual(estimate["alpha_from_from_bus"], 0.47, places=6)
        self.assertAlmostEqual(estimate["r_hif_pu"] / 100.0, 1.0, places=6)
        self.assertAlmostEqual(estimate["r_hif_ohm"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
