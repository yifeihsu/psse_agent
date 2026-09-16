"""Physical and feature-contract checks with actual IEEE14/57 WLS solves."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest
import numpy as np

from mcp_server.matpower_server import _load_python_case
from tools import lagrangian_port as lp
from research.gnn_screen.feature_schema import ScreenInputError
from research.gnn_screen.graph_builder import build_graph, FeatureScaler
from research.gnn_screen.wls_features import (build_wls_features, configured_case,
    default_measurement_sigma, residual_covariance, state_measurements_and_jacobian)

REPO = Path(__file__).resolve().parents[3]


def snapshot(network="case14", seed=12):
    case = _load_python_case(str(REPO / "mcp_server" / f"{network}.m"))
    clean = configured_case(case)
    theta = np.deg2rad(case["bus"][:, 8])
    vm = case["bus"][:, 7].copy()
    z, _ = state_measurements_and_jacobian(clean, theta, vm)
    sigma = default_measurement_sigma(len(case["bus"]), len(case["branch"]))
    return case, z + np.random.default_rng(seed).normal(0, sigma)


class PhysicalWLSFeaturesTests(unittest.TestCase):
    def test_actual_ieee14_and_ieee57_shapes_and_current_covariance(self):
        for name, nb, nl in (("case14", 14, 20), ("case57", 57, 80)):
            with self.subTest(network=name):
                case, z = snapshot(name)
                evidence = build_wls_features(case, z)
                graph = build_graph(case, z)
                self.assertEqual(graph["x"].shape, (nb, 27))
                self.assertEqual(graph["edge_attr"].shape, (2 * nl, 40))
                self.assertEqual(graph["edge_index"].shape, (2, 2 * nl))
                self.assertEqual(graph["u"].shape, (4,))
                self.assertEqual(graph["x"].dtype, np.float64)
                h, variance = evidence["measurement_jacobian"], evidence["variance"]
                # Independent QR residual projector avoids the implementation's
                # normal-equation/Cholesky route and verifies the true covariance.
                q, _ = np.linalg.qr(h / np.sqrt(variance[:, None]), mode="reduced")
                reference_omega = variance * (1 - np.sum(q * q, axis=1))
                np.testing.assert_allclose(evidence["residual_covariance_diag"], reference_omega,
                                           rtol=1e-8, atol=1e-14)
                expected_j = np.dot(evidence["raw_residual"], evidence["raw_residual"] / variance)
                self.assertAlmostEqual(graph["u"][0], expected_j / evidence["dof"])
                signed = evidence["signed_normalized_residual"]
                self.assertTrue(np.any(signed > 0) and np.any(signed < 0))

    def test_custom_covariance_is_actual_solver_covariance(self):
        case, z = snapshot()
        nb, nl = len(case["bus"]), len(case["branch"])
        sigma = default_measurement_sigma(nb, nl) * np.linspace(0.5, 2.0, len(z))
        evidence = build_wls_features(case, z, measurement_sigma=sigma)
        np.testing.assert_array_equal(evidence["variance"], sigma ** 2)
        h, r = evidence["measurement_jacobian"], evidence["raw_residual"]
        correction = np.linalg.lstsq(h / sigma[:, None], r / sigma, rcond=None)[0]
        self.assertLess(np.max(np.abs(correction)), 1e-8)
        graph = build_graph(case, z, measurement_sigma=sigma)
        np.testing.assert_allclose(graph["x"][:, 3], np.log(sigma[:nb] / 0.001))

    def test_jacobian_is_at_estimated_state_by_finite_difference(self):
        case, z = snapshot()
        evidence = build_wls_features(case, z)
        clean = configured_case(case)
        theta, vm = evidence["theta_est_rad"], evidence["vm_est_pu"]
        ref = int(np.flatnonzero(clean["bus"][:, 1] == 3)[0])
        full = np.r_[theta, vm]
        numeric = []
        for column in range(len(full)):
            if column == ref:
                continue
            plus, minus = full.copy(), full.copy()
            plus[column] += 1e-6
            minus[column] -= 1e-6
            hp, _ = state_measurements_and_jacobian(clean, plus[:len(vm)], plus[len(vm):])
            hm, _ = state_measurements_and_jacobian(clean, minus[:len(vm)], minus[len(vm):])
            numeric.append((hp - hm) / 2e-6)
        np.testing.assert_allclose(evidence["measurement_jacobian"], np.array(numeric).T,
                                   rtol=3e-5, atol=2e-7)

    def test_global_angle_reference_invariance_and_stale_details_rejected(self):
        case, z = snapshot()
        clean = configured_case(case)
        details = lp.lagrangian_m_singlephase_details(z, clean, 0, clean["bus"], tol=1e-8)
        original = build_graph(case, z, wls_details=details)
        shifted = deepcopy(details)
        shifted["theta_est_rad"] += 1.234
        changed = build_graph(case, z, wls_details=shifted)
        for name in ("x", "edge_attr", "u"):
            np.testing.assert_allclose(original[name], changed[name], rtol=1e-7, atol=1e-9)
        with self.assertRaises(ScreenInputError):
            build_graph(case, z + 0.1, wls_details=details)

    def test_invalid_inputs_never_emit_negative_screen(self):
        case, z = snapshot()
        disconnected = deepcopy(case)
        disconnected["branch"][:, 10] = 0
        with self.assertRaises(ScreenInputError) as context:
            build_graph(disconnected, z)
        self.assertEqual(context.exception.status, "inadequate_observability")
        with self.assertRaises(ScreenInputError) as context:
            build_graph(case, z, wls_details={"success": False})
        self.assertEqual(context.exception.status, "wls_failure")
        with self.assertRaises(ScreenInputError) as context:
            build_graph(case, z, measurement_mask=np.zeros_like(z))
        self.assertEqual(context.exception.status, "unsupported_input")
        with self.assertRaises(ScreenInputError):
            build_graph(case, np.r_[z[:-1], np.nan])

    def test_critical_measurement_variance_is_not_clipped_to_huge_score(self):
        # A critical first measurement is exactly absorbed; its variance is 0.
        h = np.array([[1., 0.], [0., 1.], [0., 1.]])
        omega, leverage = residual_covariance(h, np.ones(3))
        self.assertAlmostEqual(omega[0], 0, places=12)
        self.assertEqual(leverage[0], 1.0)


class GraphMappingTests(unittest.TestCase):
    def test_native_terminal_reversal_parallel_open_branches_and_tap_side(self):
        case, _ = snapshot()
        case["branch"] = np.vstack((case["branch"], case["branch"][0]))
        case["branch"][-1, 10] = 0
        case["branch"][7, 9] = 9.5
        z, _ = state_measurements_and_jacobian(configured_case(case),
                    np.deg2rad(case["bus"][:, 8]), case["bus"][:, 7])
        # Report nonzero power on a configured-open record: preserve observation.
        z[3 * len(case["bus"]) + len(case["branch"]) - 1] = 0.1
        graph = build_graph(case, z)
        self.assertEqual(len(graph["edge_pair"]), 42)
        np.testing.assert_array_equal(graph["edge_index"][:, 0], graph["edge_index"][:, -2])
        forward, reverse = graph["edge_attr"][0::2], graph["edge_attr"][1::2]
        np.testing.assert_array_equal(forward[:, :14], reverse[:, 14:28])
        np.testing.assert_array_equal(forward[:, 14:28], reverse[:, :14])
        np.testing.assert_array_equal(forward[:, 28:37], reverse[:, 28:37])
        np.testing.assert_array_equal(forward[:, 37], np.ones(21))
        np.testing.assert_array_equal(reverse[:, 37], -np.ones(21))
        self.assertEqual(forward[-1, 34], 0)
        self.assertEqual(forward[-1, 0], 0.1)
        self.assertEqual(forward[7, 31], case["branch"][7, 8])
        self.assertAlmostEqual(forward[7, 33], np.sin(np.deg2rad(9.5)))

    def test_bus_branch_permutation_and_external_ids(self):
        case, z = snapshot()
        graph = build_graph(case, z)
        rng = np.random.default_rng(2)
        bp, ep = rng.permutation(14), rng.permutation(20)
        permuted = deepcopy(case)
        # Change arbitrary external labels before permuting rows.
        idmap = {int(old): int(old * 13 + 101) for old in case["bus"][:, 0]}
        permuted["bus"][:, 0] = [idmap[int(i)] for i in case["bus"][:, 0]]
        for column in (0, 1):
            permuted["branch"][:, column] = [idmap[int(i)] for i in case["branch"][:, column]]
        permuted["bus"] = permuted["bus"][bp]
        permuted["branch"] = permuted["branch"][ep]
        zp = np.r_[*[z[k * 14:(k + 1) * 14][bp] for k in range(3)],
                   *[z[42 + k * 20:42 + (k + 1) * 20][ep] for k in range(4)]]
        changed = build_graph(permuted, zp)
        np.testing.assert_allclose(changed["x"], graph["x"][bp], atol=1e-7)
        directions = np.array([[2 * p, 2 * p + 1] for p in ep]).ravel()
        np.testing.assert_allclose(changed["edge_attr"], graph["edge_attr"][directions], atol=1e-7)
        np.testing.assert_allclose(changed["u"], graph["u"], atol=1e-7)
        inverse = np.argsort(bp)
        np.testing.assert_array_equal(changed["edge_index"], inverse[graph["edge_index"][:, directions]])

    def test_simulator_state_load_dispatch_truth_and_phase_data_are_not_features(self):
        case, z = snapshot()
        original = build_graph(case, z)
        contaminated = deepcopy(case)
        contaminated["bus"][:, [2, 3, 7, 8]] = np.nan
        contaminated["gen"] = {"hidden_dispatch": "forbidden"}
        contaminated.update({"scenario_name": "hif", "true_branch": 7,
            "three_phase_voltages": np.ones((14, 3)), "harmonic_spectra": [100]})
        changed = build_graph(contaminated, z)
        for name in ("x", "edge_attr", "u"):
            np.testing.assert_array_equal(original[name], changed[name])
        with self.assertRaises(TypeError):
            build_graph(case, z, three_phase_voltages=np.ones((14, 3)))

    def test_scaler_training_only_shared_types_and_serialization(self):
        case, z = snapshot()
        graph = build_graph(case, z)
        scaler = FeatureScaler().fit([graph])
        restored = FeatureScaler.from_dict(scaler.to_dict())
        scaled = restored.transform(graph)
        np.testing.assert_allclose(scaled["x"][:, 0].mean(), 0, atol=1e-12)
        np.testing.assert_allclose(scaled["x"][:, 0].std(), 1)
        np.testing.assert_array_equal(scaled["x"][:, 2::7], graph["x"][:, 2::7])
        np.testing.assert_array_equal(scaled["edge_attr"][0::2, :14], scaled["edge_attr"][1::2, 14:28])
        self.assertFalse(graph["metadata"]["scaler_applied"])
        with self.assertRaises(ValueError):
            scaler.fit([graph], split="test")
        with self.assertRaises(ValueError):
            scaler.transform(scaled)


if __name__ == "__main__":
    unittest.main()
