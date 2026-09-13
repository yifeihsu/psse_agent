"""Physical logical-topology measurements and formulation-aware WLS checks."""
from __future__ import annotations

import copy
import unittest

import numpy as np
from pypower.api import ppoption, runpf
from threadpoolctl import threadpool_limits

from psse_env.providers.matpower import MatpowerDeploymentProviders
from psse_env.providers.scenario_generator import build_measurement_vector
from psse_env.systems import resolve_system
from .inventory import build_inventory, process_topology
from .measurements import (aggregate_measurements, build_measurement_inventory,
                           expected_measurements, sample_measurements)
from .estimation import _MeasurementModel, estimate


class LogicalMeasurementEstimationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.thread_limit = threadpool_limits(limits=1)
        cls.addClassCleanup(cls.thread_limit.restore_original_limits)
        cls.case = resolve_system("case57").load_case()
        cls.core = build_inventory("case57", split_buses=[])
        cls.section = build_inventory("case57", split_buses=[4])
        cls.full = build_inventory("case57")
        cls.options = ppoption(VERBOSE=0, OUT_ALL=0, PF_TOL=1e-10)

    def physical(self, inventory, statuses=None, case=None):
        case = self.case if case is None else case
        statuses = dict(inventory["normal_statuses"]) if statuses is None else statuses
        processed = process_topology(case, inventory, statuses)
        result, success = runpf(processed["case"], self.options)
        self.assertTrue(success)
        return result

    def data(self, inventory, statuses=None, *, profile="direct", noise=False, case=None):
        case = self.case if case is None else case
        statuses = dict(inventory["normal_statuses"]) if statuses is None else statuses
        sensors = build_measurement_inventory(inventory, profile)
        physical = self.physical(inventory, statuses, case)
        truth = expected_measurements(case, inventory, statuses, physical, sensors)
        return sample_measurements(truth, sensors, seed=427, noise=noise), sensors, physical

    def test_branch_only_ieee57_all491_and_existing_wls_agree(self):
        observations, sensors, physical = self.data(self.core, noise=True)
        truth = expected_measurements(self.case, self.core, self.core["normal_statuses"], physical, sensors)
        np.testing.assert_allclose(truth, build_measurement_vector(physical), rtol=0, atol=1e-8)
        result = estimate(self.case, self.core, self.core["normal_statuses"], observations, sensors)
        baseline = MatpowerDeploymentProviders(chi2_alpha=.05, normalized_residual_threshold=4).run_wls(
            {"state_id": "logical57_comparison:s0", "state_hash": "logical57_hash", "case": "case57",
             "measurements": observations["values"], "metadata": {}})
        self.assertTrue(result["converged"])
        self.assertTrue(result["observable"])
        self.assertEqual((result["raw_measurement_count"], result["rank"], result["chi_square_dof"]), (491, 113, 378))
        self.assertAlmostEqual(result["wls_objective"], baseline["wls_objective"], delta=1e-5)
        self.assertAlmostEqual(result["max_normalized_residual"], baseline["max_normalized_residual"], delta=1e-4)

    def test_ieee14_bridge_has_original_dimensions(self):
        inventory = build_inventory("case14", split_buses=[])
        case = resolve_system("case14").load_case()
        observations, sensors, _ = self.data(inventory, case=case)
        result = estimate(case, inventory, inventory["normal_statuses"], observations, sensors)
        self.assertTrue(result["plausible"])
        self.assertEqual((result["raw_measurement_count"], result["state_dimension"]), (122, 27))

    def test_closed_and_open_sections_keep_effective_state_dimension(self):
        closed = dict(self.section["normal_statuses"])
        opened = dict(closed)
        opened[self.section["couplers"][0]["device_id"]] = 0
        outcomes = []
        for statuses in (closed, opened):
            observations, sensors, _ = self.data(self.section, statuses)
            result = estimate(self.case, self.section, statuses, observations, sensors)
            self.assertTrue(result["plausible"], result)
            outcomes.append(result)
        self.assertEqual([r["state_dimension"] for r in outcomes], [115, 115])
        self.assertEqual([r["rank"] for r in outcomes], [115, 115])
        self.assertEqual([r["chi_square_dof"] for r in outcomes], [379, 379])
        self.assertEqual([r["closed_coupler_nuisance_count"] for r in outcomes], [2, 0])
        self.assertGreater(abs(next(iter(outcomes[0]["state"]["closed_coupler_flows_pu"].values()))["p"]), .01)

    def test_all_declared_sections_are_observable_with_nuisance_flows(self):
        observations, sensors, _ = self.data(self.full)
        result = estimate(self.case, self.full, self.full["normal_statuses"], observations, sensors)
        self.assertTrue(result["plausible"])
        self.assertEqual((result["raw_measurement_count"], result["state_dimension"], result["rank"]), (530, 139, 139))

    def test_section_injections_come_from_equipment_allocation(self):
        inventory = build_inventory("case57", split_buses=[12])
        observations, sensors, physical = self.data(inventory)
        readings = {record["node_id"]: observations["values"][index]
                    for index, record in enumerate(sensors["records"])
                    if record["kind"] == "Pinj" and ":bus:12:" in record["node_id"]}
        self.assertEqual(len(readings), 2)
        self.assertNotAlmostEqual(*readings.values())
        expected = sum(row[1] for row in physical["gen"] if int(row[0]) == 12)/100 - self.case["bus"][11, 2]/100
        self.assertAlmostEqual(sum(readings.values()), expected)
        wrong = copy.deepcopy(physical)
        wrong["bus"][0, 2] += 1
        with self.assertRaisesRegex(ValueError, "allocation"):
            expected_measurements(self.case, inventory, inventory["normal_statuses"], wrong, sensors)

    def test_offline_branch_keeps_available_constant_zero_model_rows(self):
        observations, sensors, _ = self.data(self.core)
        wrong = dict(self.core["normal_statuses"])
        wrong[self.core["branches"][0]["device_id"]] = 0
        result = estimate(self.case, self.core, wrong, observations, sensors)
        self.assertTrue(result["candidate_connected"])
        self.assertEqual(result["available_measurement_count"], 491)
        self.assertFalse(result["plausible"])
        for index, record in enumerate(sensors["records"]):
            if record.get("branch_row0") == 0:
                self.assertEqual(result["predicted_values"][index], 0.0)
                self.assertIsNotNone(result["normalized_residuals"][index])

    def test_predetermined_indirect_masks_and_sparse_unobservability(self):
        for profile, parity in (("indirect_even", 0), ("indirect_odd", 1)):
            sensors = build_measurement_inventory(self.core, profile)
            self.assertEqual(sensors["masked_branch_rows0"], list(range(parity, 80, 2)))
            self.assertEqual(sum(sensors["available_mask"]), 331)
            self.assertEqual(len(sensors["records"]), 491)
        observations, sensors, _ = self.data(self.full, profile="voltage_only")
        result = estimate(self.case, self.full, self.full["normal_statuses"], observations, sensors)
        self.assertTrue(result["converged"])
        self.assertFalse(result["observable"])
        self.assertFalse(result["plausible"])
        self.assertEqual(result["failure_reason"], "state_unobservable")

    def test_unavailable_values_are_absent_not_zero_and_cannot_leak_truth(self):
        sensors = build_measurement_inventory(self.core, "indirect_even")
        physical = self.physical(self.core)
        truth = expected_measurements(self.case, self.core, self.core["normal_statuses"], physical, sensors)
        changed_truth = list(truth)
        for index, available in enumerate(sensors["available_mask"]):
            if not available:
                changed_truth[index] += 1e6
        first = sample_measurements(truth, sensors, seed=521, noise=True)
        second = sample_measurements(changed_truth, sensors, seed=521, noise=True)
        self.assertEqual(first, second)
        masked_index = sensors["available_mask"].index(False)
        self.assertIsNone(first["values"][masked_index])
        result = estimate(self.case, self.core, self.core["normal_statuses"], first, sensors)
        self.assertTrue(result["converged"])
        self.assertIsNone(result["raw_residuals"][masked_index])
        self.assertIsNone(result["normalized_residuals"][masked_index])
        unredacted = copy.deepcopy(first); unredacted["values"][masked_index] = 0.0
        with self.assertRaisesRegex(ValueError, "redacted"):
            estimate(self.case, self.core, self.core["normal_statuses"], unredacted, sensors)
        matrix = np.zeros((1, len(truth))); matrix[0, masked_index] = 1
        with self.assertRaisesRegex(ValueError, "unavailable"):
            aggregate_measurements(first, sensors, matrix)
        matrix[0, masked_index] = 0; matrix[0, 0] = 1
        aggregate = aggregate_measurements(first, sensors, matrix)
        self.assertEqual(aggregate["values"], [first["values"][0]])

    def test_covariance_aggregation_preserves_correlations_and_raw_records(self):
        observations, sensors, _ = self.data(self.section)
        n = len(sensors["records"])
        indices = [index for index, row in enumerate(sensors["records"])
                   if row["kind"] == "Pinj" and ":bus:4:" in row["node_id"]]
        a = np.zeros((2, n))
        a[0, indices] = 1
        a[1, indices[0]] = 1
        result = aggregate_measurements(observations, sensors, a)
        expected = a@np.array(sensors["covariance"])@a.T
        np.testing.assert_allclose(result["covariance"], expected)
        self.assertEqual(result["covariance"][0][1], 1e-4)
        self.assertEqual(result["raw_observations"], observations)
        self.assertEqual(result["raw_available_mask"], sensors["available_mask"])
        a = np.zeros((1, n)); a[0, :2] = .5
        with self.assertRaisesRegex(ValueError, "voltage"):
            aggregate_measurements(observations, sensors, a)

    def test_full_correlated_covariance_and_analytic_jacobian(self):
        statuses = self.section["normal_statuses"]
        observations, sensors, _ = self.data(self.section)
        covariance = np.array(sensors["covariance"])
        covariance[60, 61] = covariance[61, 60] = .5e-4
        correlated = build_measurement_inventory(self.section, covariance=covariance)
        observations = sample_measurements(observations["values"], correlated, seed=29, noise=True)
        result = estimate(self.case, self.section, statuses, observations, correlated)
        self.assertTrue(result["converged"])
        self.assertTrue(result["observable"])
        residual = np.asarray(result["raw_residuals"])
        self.assertAlmostEqual(result["wls_objective"], float(residual@np.linalg.solve(covariance, residual)), places=7)
        model = _MeasurementModel(self.case, self.section, statuses, process_topology(self.case, self.section, statuses), sensors)
        x = np.zeros(model.nstate); x[model.vm_start:model.voltage_states] = 1
        x[0] = .02; x[-2:] = [.1, -.05]
        _, jac = model.evaluate(x)
        for column in (0, model.vm_start, model.vm_start+3, model.nstate-2, model.nstate-1):
            plus, minus = x.copy(), x.copy(); plus[column] += 1e-6; minus[column] -= 1e-6
            finite_difference = (model.evaluate(plus)[0]-model.evaluate(minus)[0])/(2e-6)
            np.testing.assert_allclose(jac[:, column], finite_difference, rtol=1e-6, atol=1e-7)

    def test_nonconvergence_unknown_and_disconnected_candidates_remain_distinct(self):
        observations, sensors, _ = self.data(self.core)
        result = estimate(self.case, self.core, self.core["normal_statuses"], observations, sensors, max_nfev=1)
        self.assertFalse(result["converged"])
        self.assertEqual(result["failure_reason"], "wls_nonconvergence")
        unknown = dict(self.core["normal_statuses"]); unknown[self.core["branches"][0]["device_id"]] = None
        result = estimate(self.case, self.core, unknown, observations, sensors)
        self.assertEqual(result["failure_reason"], "invalid_or_unknown_candidate_status")
        for branch in self.core["branches"]:
            candidate = dict(self.core["normal_statuses"]); candidate[branch["device_id"]] = 0
            if not process_topology(self.case, self.core, candidate)["connectivity"]["connected"]:
                break
        else:
            self.fail("case57 must exercise an islanding branch candidate")
        result = estimate(self.case, self.core, candidate, observations, sensors)
        self.assertFalse(result["candidate_connected"])
        self.assertTrue(result["excluded_by_declared_scope"])
        self.assertFalse(result["plausible"])


if __name__ == "__main__":
    unittest.main()
