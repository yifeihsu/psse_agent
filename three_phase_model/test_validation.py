"""Compiled-engine equivalence and failure-detection checks for the exporter."""

from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np
from pypower.api import case9, case57

from three_phase_model.exporter import export_model, load_assumptions
from three_phase_model.runtime import compile_model
from three_phase_model.validation import validate_model


class CompiledModelValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="compiled_three_phase_")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.diagonal = export_model(case57(), Path(cls.temporary.name) / "diagonal", case_id="case57")
        cls.coupled = export_model(
            case57(), Path(cls.temporary.name) / "coupled", case_id="case57",
            assumptions=load_assumptions("coupled_sensitivity"),
        )

    @staticmethod
    def engine(model):
        return compile_model(Path(model["output_dir"]) / "Master.dss")

    @staticmethod
    def audit(engine, model, *, balanced=True, registry=None):
        return validate_model(
            engine, model["reference"], registry or model["registry"],
            model["assumptions"], balanced=balanced,
        )

    def test_ieee57_compiled_model_matches_reference_and_physical_equations(self) -> None:
        report = self.audit(self.engine(self.diagonal), self.diagonal)
        self.assertTrue(report["passed"], report["failed_checks"])
        self.assertEqual(len(report["buses"]), 57)
        self.assertEqual(len(report["branches"]), 80)
        self.assertEqual(report["checks"]["phase_node_kcl"]["node_count"], 171)
        self.assertLess(report["checks"]["branch_positive_sequence_terminal_admittance"]["max_error"], 1e-10)
        self.assertLess(report["checks"]["balanced_bus_voltage_magnitude"]["max_error"], 1e-6)
        self.assertLess(report["checks"]["balanced_branch_both_end_power"]["max_error"], 1e-5)

    def test_coupled_completion_preserves_positive_sequence_and_balanced_solution(self) -> None:
        engine = self.engine(self.coupled)
        report = self.audit(engine, self.coupled)
        self.assertTrue(report["passed"], report["failed_checks"])
        engine.Circuit.SetActiveElement("Line.br_0001")
        raw = np.asarray(engine.CktElement.YPrim())
        actual_y = (raw[::2] + 1j * raw[1::2]).reshape(6, 6, order="F")
        self.assertGreater(abs(actual_y[0, 1]), 1.0)

    def test_removed_actual_line_charging_fails_independent_primitive_check(self) -> None:
        engine = self.engine(self.diagonal)
        engine.Text.Command("Edit Line.br_0001 Cmatrix=[0 | 0 0 | 0 0 0]")
        engine.Solution.Solve()
        report = self.audit(engine, self.diagonal)
        self.assertFalse(report["passed"])
        self.assertIn("branch_positive_sequence_terminal_admittance", report["failed_checks"])
        self.assertGreater(report["branches"][0]["terminal_y_max_error_pu"], 0.01)

    def test_wrong_actual_transformer_tap_fails_independent_primitive_check(self) -> None:
        engine = self.engine(self.diagonal)
        transformer = next(row for row in self.diagonal["registry"]["branches"]
                           if row["dss_element"].lower().startswith("transformer."))
        engine.Text.Command(f"Edit {transformer['dss_element']} Wdg=1 Tap=1.1")
        engine.Solution.Solve()
        report = self.audit(engine, self.diagonal)
        self.assertIn("branch_positive_sequence_terminal_admittance", report["failed_checks"])

    def test_registry_phase_and_terminal_labels_are_verified_against_engine(self) -> None:
        registry = copy.deepcopy(self.diagonal["registry"])
        registry["loads"][0]["phase"] = 2
        registry["branches"][0]["from_terminal"] = 2
        report = self.audit(self.engine(self.diagonal), self.diagonal, registry=registry)
        self.assertFalse(report["checks"]["asset_terminal_identity"]["passed"])
        self.assertEqual(len(report["checks"]["asset_terminal_identity"]["problems"]), 2)

    def test_unbalanced_resolve_preserves_kcl_and_component_equations(self) -> None:
        engine = self.engine(self.coupled)
        loads = self.coupled["registry"]["loads"]
        target_bus = max(loads, key=lambda item: item["kw"])["bus"]
        for load in loads:
            if load["bus"] != target_bus:
                continue
            factor = {1: 1.2, 2: 0.8, 3: 1.0}[load["phase"]]
            engine.Text.Command(f"Edit {load['element']} kW={load['kw'] * factor:.16g} kvar={load['kvar'] * factor:.16g}")
        engine.Solution.Solve()
        report = self.audit(engine, self.coupled, balanced=False)
        self.assertTrue(report["passed"], report["failed_checks"])
        self.assertNotIn("balanced_voltage_sequence_null", report["checks"])
        balanced_claim = self.audit(engine, self.coupled)
        self.assertIn("balanced_voltage_sequence_null", balanced_claim["failed_checks"])
        self.assertTrue(balanced_claim["checks"]["phase_node_kcl"]["passed"])

    def test_changed_zero_sequence_source_impedance_is_detected_under_balance(self) -> None:
        engine = self.engine(self.diagonal)
        engine.Text.Command("Edit Vsource.source Z0=[2e-8 2e-8]")
        engine.Solution.Solve()
        report = self.audit(engine, self.diagonal)
        self.assertTrue(report["checks"]["balanced_bus_voltage_magnitude"]["passed"])
        self.assertIn("source_sequence_impedance", report["failed_checks"])

    def test_unregistered_enabled_device_fails_coverage(self) -> None:
        engine = self.engine(self.diagonal)
        engine.Text.Command("New Load.unregistered Phases=1 Bus1=b2.1.0 kV=.5773502691896258 kW=100 kvar=0 Model=1")
        engine.Solution.Solve()
        report = self.audit(engine, self.diagonal)
        self.assertIn("asset_coverage", report["failed_checks"])
        self.assertIn("load.unregistered", report["checks"]["asset_coverage"]["missing_or_unregistered_enabled_elements"])

    def test_tapped_charging_transformer_nominal_transformer_and_gs_bs_shunts(self) -> None:
        source = case9()
        source["branch"][0, [2, 4, 8]] = [0.004, 0.02, 1.025]
        source["branch"][3, 8] = 1.0
        source["bus"][4, [4, 5]] = [1.2, 2.3]
        model = export_model(source, Path(self.temporary.name) / "tap_charging_gs", case_id="tap_charging_gs")
        self.assertTrue(model["registry"]["branches"][0]["charging_elements"]["from"])
        self.assertTrue(model["registry"]["branches"][3]["dss_element"].startswith("Transformer."))
        report = self.audit(self.engine(model), model)
        self.assertTrue(report["passed"], report["failed_checks"])

    def test_inactive_transformer_and_its_charging_contribute_no_admittance(self) -> None:
        source = case57()
        # One of the parallel 4->18 transformers is removed; its stored
        # charging must disappear with it while the second circuit remains.
        source["branch"][18, [4, 10]] = [0.02, 0]
        model = export_model(source, Path(self.temporary.name) / "inactive_tap", case_id="inactive_tap")
        report = self.audit(self.engine(model), model)
        self.assertTrue(report["passed"], report["failed_checks"])
        self.assertEqual(report["branches"][18]["terminal_y_max_error_pu"], 0.0)


if __name__ == "__main__":
    unittest.main()
