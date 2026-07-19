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
    _phase_terminal_complex_powers,
    element_pq_3ph_per_terminal,
    extract_measurement_series,
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


if __name__ == "__main__":
    unittest.main()
