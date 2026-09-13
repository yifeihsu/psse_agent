"""Statistical-envelope mathematics and guarded scan-decision tests.

The Monte Carlo checks below are Gaussian linear-model checks. They do not
claim calibrated finite-sample error rates for nonlinear power-system WLS.
"""
from __future__ import annotations

import copy
import unittest

import numpy as np
from scipy.stats import chi2, norm

from .calibration import calibrate_scan, common_envelope_df, gaussian_zero_flow_rejection
from .runtime import evidence_hash


def fixture():
    inventory = {"layout_hash": "linear_fixture", "nodes": [{"node_id": f"n{i}"} for i in range(1, 5)],
                 "branches": [{"device_id": f"b{i}", "device_kind": "branch_status", "row0": i-1,
                               "from_node": f"n{a}", "to_node": f"n{b}"}
                              for i, (a, b) in enumerate(((1, 2), (2, 3), (1, 3), (3, 4)), 1)],
                 "couplers": [{"device_id": "c1", "device_kind": "bus_coupler", "node_a": "n3", "node_b": "n4"}]}
    records = [{"sensor_id": f"z{index}", "kind": ("Pf", "Qf", "Pt", "Qt")[index] if index < 4 else "Pinj",
                "branch_row0": 0, "available": True} for index in range(331)]
    sensors = {"layout_hash": "linear_fixture", "sensor_inventory_hash": "fixed_linear_sensors",
               "records": records, "available_mask": [True]*331, "covariance": np.eye(331).tolist()}
    observations = {"sensor_inventory_hash": sensors["sensor_inventory_hash"], "sensor_ids": [row["sensor_id"] for row in records],
                    "values": [0.0]*331}
    statuses = {row["device_id"]: 1 for rows in (inventory["branches"], inventory["couplers"]) for row in rows}
    return inventory, sensors, observations, statuses


def fit(objective, maximum=3.0):
    return {"converged": True, "observable": True, "rank": 113, "state_dimension": 113,
            "available_measurement_count": 331, "wls_objective": objective, "max_normalized_residual": maximum,
            "chi_square_alpha": .05, "normalized_residual_threshold": 4.0}


def scan_for(inventory, sensors, observations, chosen_statuses, rival_statuses, *, chosen_j=200, rival_j=350,
             rival_resolution="rejected", rival_fit=True):
    binding = evidence_hash({"measurement_inventory": sensors, "observations": observations})
    def candidate(name, statuses, selected):
        return {"candidate_id": name, "parent_model_hash": "fixed_model", "fixed_evidence_hash": binding,
                "statuses": copy.deepcopy(statuses), "current_model": not selected,
                "resolution": "plausible" if selected else rival_resolution, "plausible": selected,
                "estimation": fit(chosen_j if selected else rival_j) if selected or rival_fit else None}
    selected = candidate("selected", chosen_statuses, True)
    rival = candidate("current", rival_statuses, False)
    return {"decision": "unique_within_declared_scope", "scope_complete": True,
            "unique_candidate_id": "selected", "parent_model_hash": "fixed_model", "fixed_evidence_hash": binding,
            "current": rival, "candidates": [selected, rival],
            "hypothesis_scope": {"connected_energized_models_only": True, "global_status_uniqueness_claimed": False}}


class SeparationCalibrationTests(unittest.TestCase):
    def test_near_threshold_healthy_control_is_not_a_status_identification(self):
        inventory, sensors, observations, current = fixture()
        selected = dict(current); selected["b1"] = 0
        scan = scan_for(inventory, sensors, observations, selected, current,
                        chosen_j=250.857, rival_j=254.738)
        self.assertLess(scan["candidates"][0]["estimation"]["wls_objective"], chi2.ppf(.95, 218))
        self.assertGreater(scan["current"]["estimation"]["wls_objective"], chi2.ppf(.95, 218))
        result = calibrate_scan(scan, inventory, sensors, observations)
        self.assertFalse(result["allowed"])
        comparison = result["comparisons"][0]
        self.assertAlmostEqual(comparison["objective_gain"], 3.881)
        self.assertEqual(comparison["df_upper_bound"], 4)
        self.assertEqual(comparison["reason"], "absolute_threshold_crossing_without_material_separation")

    def test_strong_fixed_family_separation_passes_conditionally(self):
        inventory, sensors, observations, current = fixture()
        selected = dict(current); selected["b1"] = 0
        result = calibrate_scan(scan_for(inventory, sensors, observations, selected, current), inventory, sensors, observations)
        self.assertTrue(result["allowed"])
        self.assertFalse(result["exact_nonlinear_false_positive_guarantee"])
        self.assertFalse(result["global_minima_verified"])

    def test_envelope_cannot_spend_the_gaussian_method_share(self):
        inventory, sensors, observations, current = fixture()
        selected = dict(current); selected["b1"] = 0
        old_critical, shared_critical = chi2.isf(.05, 4), chi2.isf(.05/2, 4)
        gain = (old_critical+shared_critical)/2
        scan = scan_for(inventory, sensors, observations, selected, current,
                        chosen_j=244, rival_j=244+gain)
        self.assertGreater(scan["current"]["estimation"]["wls_objective"], chi2.isf(.05, 218))
        result = calibrate_scan(scan, inventory, sensors, observations)
        self.assertFalse(result["allowed"])
        self.assertEqual(result["contract"], "logical_topology_pairwise_separation_guard_v2")
        self.assertEqual(result["method_count"], 2)
        self.assertEqual(result["method_budget_allocation"], "equal_bonferroni_split")
        self.assertEqual(result["method_family_alpha"], .025)
        self.assertEqual(result["pairwise_alpha"], .05)
        self.assertEqual(result["pairwise_method_alpha"], .025)
        comparison = result["comparisons"][0]
        self.assertAlmostEqual(comparison["critical_gain"], shared_critical)
        self.assertAlmostEqual(comparison["multiplicity_adjusted_asymptotic_p_upper_bound"], 2*chi2.sf(gain, 4))
        self.assertGreater(comparison["multiplicity_adjusted_asymptotic_p_upper_bound"], .05)

    def test_gaussian_witness_and_fallback_share_one_family_budget(self):
        inventory, sensors, observations, selected = fixture()
        rival = dict(selected); rival["b1"] = 0
        alpha = 1e-4
        # A legitimate >4-sigma analytical witness whose old one-method
        # adjusted p is .75*alpha cannot use the other method's allocation.
        z = float(norm.isf(.75*alpha/(2*4)))
        self.assertGreater(z, 4)
        observations["values"][0] = z
        scan = scan_for(inventory, sensors, observations, selected, rival,
                        rival_resolution="analytical_rejection", rival_fit=False)
        standalone = gaussian_zero_flow_rejection(inventory, rival, sensors, observations, rival_count=1, alpha=alpha)
        self.assertTrue(standalone["passed"])
        result = calibrate_scan(scan, inventory, sensors, observations, alpha=alpha)
        self.assertEqual(result["requires_full_fit"], ["current"])
        flow = result["comparisons"][0]["zero_flow_evidence"]
        self.assertFalse(flow["passed"])
        self.assertEqual(flow["method_count"], 2)
        self.assertEqual(flow["method_family_alpha"], alpha/2)
        self.assertEqual(flow["pairwise_method_alpha"], alpha/2)
        self.assertAlmostEqual(flow["multiplicity_adjusted_p_upper_bound"], 1.5*alpha)
        self.assertAlmostEqual(flow["log_adjusted_p_upper_bound"], np.log(1.5*alpha))
        # The fitted fallback also lies between the old full-alpha cutoff and
        # the shared cutoff. Failing the witness cannot release its budget.
        gain = float((chi2.isf(alpha, 4)+chi2.isf(alpha/2, 4))/2)
        scan["current"]["estimation"] = fit(200+gain, maximum=z)
        result = calibrate_scan(scan, inventory, sensors, observations, alpha=alpha)
        self.assertFalse(result["allowed"])
        self.assertEqual(result["requires_full_fit"], [])
        self.assertEqual(result["comparisons"][0]["method"], "asymptotic_common_relaxation_envelope")
        self.assertGreater(result["comparisons"][0]["multiplicity_adjusted_asymptotic_p_upper_bound"], alpha)

    def test_combined_method_union_bound_in_linear_gaussian_nulls(self):
        # Independent valid null statistics make the two routes nearly
        # disjoint. This checks the combined alpha allocation, not grid-model
        # regularity or the empirical error rate of the IEEE57 experiment.
        rng = np.random.default_rng(20260912)
        samples, rivals, rows, alpha = 50000, 8, 4, .05
        witnesses = rng.normal(size=(samples, rivals, rows))
        gains = np.sum(rng.normal(size=(samples, rivals, 4))**2, axis=2)
        gaussian = np.max(np.abs(witnesses), axis=(1, 2)) >= norm.isf(alpha/(2*2*rivals*rows))
        envelope = np.max(gains, axis=1) >= chi2.isf(alpha/(2*rivals), 4)
        empirical_false_rejection = np.mean(gaussian | envelope)
        self.assertLess(empirical_false_rejection, .055)
        self.assertGreater(empirical_false_rejection, .04)

    def test_branch_and_coupler_common_relaxations_have_different_df_bounds(self):
        inventory, _, _, current = fixture()
        branch = dict(current); branch["b1"] = 0
        coupler = dict(current); coupler["c1"] = 0
        both = dict(branch); both["c1"] = 0
        self.assertEqual(common_envelope_df(inventory, current, branch)["df_upper_bound"], 4)
        self.assertEqual(common_envelope_df(inventory, current, coupler)["df_upper_bound"], 2)
        self.assertEqual(common_envelope_df(inventory, current, both)["df_upper_bound"], 6)

    def test_zero_flow_rejection_adjusts_for_rows_and_all_rivals(self):
        inventory, sensors, observations, statuses = fixture()
        statuses["b1"] = 0
        observations["values"][0] = 4.0
        weak = gaussian_zero_flow_rejection(inventory, statuses, sensors, observations, rival_count=1000)
        self.assertFalse(weak["passed"])
        self.assertAlmostEqual(weak["multiplicity_adjusted_p_upper_bound"], 2*norm.sf(4)*4*1000)
        observations["values"][0] = 8.0
        strong = gaussian_zero_flow_rejection(inventory, statuses, sensors, observations, rival_count=1000)
        self.assertTrue(strong["passed"])
        covariance = np.asarray(sensors["covariance"]); covariance[0, 1] = covariance[1, 0] = .7
        sensors["covariance"] = covariance.tolist()
        correlated = gaussian_zero_flow_rejection(inventory, statuses, sensors, observations, rival_count=1000)
        self.assertEqual(correlated["multiplicity_adjusted_p_upper_bound"], strong["multiplicity_adjusted_p_upper_bound"])

    def test_weak_analytic_receipt_requests_a_fit_then_can_use_fitted_separation(self):
        inventory, sensors, observations, selected = fixture()
        rival = dict(selected); rival["b1"] = 0
        observations["values"][0] = 4.0
        scan = scan_for(inventory, sensors, observations, selected, rival,
                        rival_resolution="analytical_rejection", rival_fit=False)
        result = calibrate_scan(scan, inventory, sensors, observations, alpha=1e-6)
        self.assertFalse(result["allowed"])
        self.assertEqual(result["requires_full_fit"], ["current"])
        scan["current"]["estimation"] = fit(350)
        result = calibrate_scan(scan, inventory, sensors, observations, alpha=1e-6)
        self.assertTrue(result["allowed"])
        self.assertEqual(result["comparisons"][0]["method"], "asymptotic_common_relaxation_envelope")

    def test_unresolved_rivals_and_unproven_scope_exclusions_block(self):
        inventory, sensors, observations, selected = fixture()
        rival = dict(selected); rival["b1"] = 0
        scan = scan_for(inventory, sensors, observations, selected, rival, rival_resolution="unresolved", rival_fit=False)
        self.assertFalse(calibrate_scan(scan, inventory, sensors, observations)["allowed"])
        isolated = dict(selected); isolated["b4"] = isolated["c1"] = 0
        scan = scan_for(inventory, sensors, observations, selected, isolated, rival_resolution="excluded", rival_fit=False)
        scan["current"]["connectivity"] = {"connected": False}
        self.assertTrue(calibrate_scan(scan, inventory, sensors, observations)["allowed"])
        scan["hypothesis_scope"]["connected_energized_models_only"] = False
        self.assertFalse(calibrate_scan(scan, inventory, sensors, observations)["allowed"])

    def test_changed_evidence_and_bound_active_fits_fail_closed(self):
        inventory, sensors, observations, selected = fixture()
        rival = dict(selected); rival["b1"] = 0
        scan = scan_for(inventory, sensors, observations, selected, rival)
        changed = copy.deepcopy(observations); changed["values"][10] = 1.0
        self.assertEqual(calibrate_scan(scan, inventory, sensors, changed)["decision"], "invalid_fixed_evidence_or_scan")
        scan["current"]["estimation"]["state"] = {"node_voltage_magnitude_pu": {"n1": .2}}
        result = calibrate_scan(scan, inventory, sensors, observations)
        self.assertFalse(result["allowed"])
        self.assertIn("voltage_bound", result["comparisons"][0]["reason"])

    def test_linear_gaussian_envelope_bonferroni_monte_carlo(self):
        # Eight independent 4-dimensional linear relaxations of a correct
        # Gaussian null. Their exact LRT gains are chi-square(4); selecting
        # the largest is permitted only after the predeclared family correction.
        rng = np.random.default_rng(20260911)
        gains = np.sum(rng.normal(size=(30000, 8, 4))**2, axis=2)
        critical = chi2.isf(.05/8, 4)
        empirical_false_rejection = np.mean(np.max(gains, axis=1) >= critical)
        self.assertLess(empirical_false_rejection, .055)
        self.assertGreater(empirical_false_rejection, .04)

    def test_equal_dimension_nonnested_linear_coupler_gain_is_bounded_by_common_fit(self):
        # Closed/open models each have five free coefficients. They share
        # three and constrain two different coordinates in a seven-dimensional
        # common model, representing flow-zero versus voltage-equality choices.
        samples = np.random.default_rng(96).normal(size=(20000, 12))
        common_j = np.sum(samples[:, 7:]**2, axis=1)
        rival_j = common_j+np.sum(samples[:, 5:7]**2, axis=1)
        selected_j = common_j+np.sum(samples[:, 3:5]**2, axis=1)
        gain, envelope_gain = rival_j-selected_j, rival_j-common_j
        self.assertTrue(np.all(gain <= envelope_gain+1e-12))
        self.assertLess(np.mean(gain >= chi2.isf(.05, 2)), .055)


if __name__ == "__main__":
    unittest.main()
