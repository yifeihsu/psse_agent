"""Ground-truth tests for the numerical foundations behind the agentic tools.

These tests check *physics*, not plumbing: derivative correctness by finite
differences, chi-square calibration under clean noise, and exact recovery of
injected measurement, parameter, and topology errors on IEEE case14 through
the same code paths the deployment providers use.
"""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scipy.stats import chi2  # noqa: E402

from mcp_server.matpower_server import (  # noqa: E402
    _load_python_case,
    _meas_correction_json,
    _param_correction_json,
    _run_hse_logic,
    _wls_json,
)
from psse_env.providers.matpower import (  # noqa: E402
    MatpowerDeploymentProviders,
    _render_matpower_case,
    matpower_case_differ,
    observable_parameter_initial_states,
)
from tools import branch_param_jacobian as bj  # noqa: E402
from tools import correct_parameter_group_multi_scan_port as pp  # noqa: E402
from tools import lagrangian_correct_port as lc  # noqa: E402
from tools import lagrangian_port as lp  # noqa: E402

CASE14 = os.path.join(REPO_ROOT, "mcp_server", "case14.m")
CASE9 = os.path.join(REPO_ROOT, "mcp_server", "case9.m")


def _copy(ppc: dict) -> dict:
    return {key: (value.copy() if hasattr(value, "copy") else value) for key, value in ppc.items()}


def _internal(ppc: dict) -> dict:
    return lp._copy_result_to_internal(ppc)


def _case_state(case_int: dict) -> tuple[np.ndarray, np.ndarray]:
    bus = case_int["bus"]
    ref = int(np.flatnonzero(bus[:, lp.BUS_TYPE].astype(int) == lp.REF)[0])
    theta = np.deg2rad(bus[:, lp.VA])
    theta = theta - theta[ref]
    return theta, bus[:, lp.VM].copy()


def _sigma(nb: int, nl: int) -> np.ndarray:
    """Measurement noise model hard-coded in the WLS port and the tracked corpora."""
    return np.r_[0.001 * np.ones(nb), 0.01 * np.ones(2 * nb), 0.01 * np.ones(4 * nl)]


def _h(ppc: dict, theta: np.ndarray | None = None, V: np.ndarray | None = None) -> np.ndarray:
    case_int = _internal(ppc)
    if theta is None or V is None:
        theta, V = _case_state(case_int)
    return pp.calculate_hx(case_int, theta, V)


def _write_case(directory: str, ppc: dict, tag: str) -> str:
    path = os.path.join(directory, f"{tag}.m")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(_render_matpower_case(ppc, tag))
    return path


class BranchJacobianTests(unittest.TestCase):
    """The shared d h / d [R, X] must agree with finite differences everywhere."""

    def _check_case(self, path: str, tol: float) -> None:
        ppc = _load_python_case(path)
        case_int = _internal(ppc)
        nb = case_int["bus"].shape[0]
        nl = case_int["branch"].shape[0]
        theta, V = _case_state(case_int)
        rng = np.random.default_rng(1)
        ref = int(np.flatnonzero(case_int["bus"][:, lp.BUS_TYPE].astype(int) == lp.REF)[0])
        theta = theta + rng.normal(0.0, 0.05, nb)
        theta[ref] = 0.0
        V = V + rng.normal(0.0, 0.02, nb)
        for k in range(nl):
            analytic = bj.branch_rx_jacobian(case_int, k, theta, V)
            numeric = bj.branch_rx_jacobian_fd(case_int, k, theta, V)
            scale = max(float(np.abs(numeric).max()), 1e-12)
            rel = float(np.abs(analytic - numeric).max()) / scale
            with self.subTest(case=os.path.basename(path), branch=k):
                self.assertLess(rel, tol)

    def test_analytic_matches_finite_difference_case14(self) -> None:
        # case14 rows 7-9 are off-nominal-tap, zero-resistance transformers.
        self._check_case(CASE14, 1e-5)

    def test_analytic_matches_finite_difference_case9(self) -> None:
        self._check_case(CASE9, 1e-5)

    def test_phase_shifter_with_line_charging(self) -> None:
        ppc = _load_python_case(CASE14)
        case_int = _internal(ppc)
        branch = case_int["branch"]
        branch[7, lp.SHIFT] = 7.5
        branch[7, lp.BR_R] = 0.01
        branch[7, lp.BR_B] = 0.05
        theta, V = _case_state(case_int)
        analytic = bj.branch_rx_jacobian(case_int, 7, theta, V)
        numeric = bj.branch_rx_jacobian_fd(case_int, 7, theta, V)
        self.assertLess(np.abs(analytic - numeric).max() / np.abs(numeric).max(), 1e-6)

    def test_out_of_service_branch_has_zero_sensitivity(self) -> None:
        ppc = _load_python_case(CASE14)
        case_int = _internal(ppc)
        case_int["branch"][0, lp.BR_STATUS] = 0.0
        theta, V = _case_state(case_int)
        self.assertEqual(np.abs(bj.branch_rx_jacobian(case_int, 0, theta, V)).max(), 0.0)
        self.assertEqual(np.abs(bj.branch_rx_jacobian_fd(case_int, 0, theta, V)).max(), 0.0)

    def test_textbook_signs_at_review_operating_point(self) -> None:
        """dP_ij/db = -Vi Vj sin(d) and dQ_ij/dg = -Vi Vj sin(d) for a plain line."""
        R, X, Vi, Vj, delta = 0.02, 0.06, 1.02, 0.98, 0.1
        bus = np.zeros((2, 13))
        bus[:, 0] = [0, 1]
        bus[:, 1] = [3, 1]
        bus[:, 7] = [Vi, Vj]
        branch = np.zeros((1, 13))
        branch[0, :4] = [0, 1, R, X]
        branch[0, lp.BR_STATUS] = 1.0
        case_int = {"baseMVA": 100.0, "bus": bus, "branch": branch}
        H = bj.branch_rx_jacobian(case_int, 0, np.array([delta, 0.0]), np.array([Vi, Vj]))
        # Rows: Pf = 3*nb + 0 = 6, Qf = 3*nb + nl + 0 = 7 for nb=2, nl=1.
        dP_dR, dP_dX = H[6]
        dQ_dR, dQ_dX = H[7]
        np.testing.assert_allclose([dP_dR, dQ_dR], [-5.8103, -26.8278], atol=2e-3)
        np.testing.assert_allclose([dP_dX, dQ_dX], [-26.8278, 5.8103], atol=2e-3)

    def test_fd_helper_survives_zero_resistance_transformer(self) -> None:
        ppc = _load_python_case(CASE14)
        case_int = _internal(ppc)
        theta, V = _case_state(case_int)
        self.assertEqual(float(case_int["branch"][7, lp.BR_R]), 0.0)
        numeric = pp.calculate_param_jacobian_for_line_fd(case_int, 7, theta, V)
        analytic = pp.calculate_param_jacobian_for_line(case_int, 7, np.r_[theta, V])
        self.assertLess(np.abs(analytic - numeric).max() / np.abs(numeric).max(), 1e-5)


class NormalizedMultiplierTests(unittest.TestCase):
    """With correct derivatives the NLM ranks the perturbed line first."""

    def test_reactance_error_is_localized_top1(self) -> None:
        ppc = _load_python_case(CASE14)
        case_int = _internal(ppc)
        theta, V = _case_state(case_int)
        nb, nl = case_int["bus"].shape[0], case_int["branch"].shape[0]
        rng = np.random.default_rng(3)
        for row, scale in ((0, 1.4), (2, 0.7), (4, 1.5), (3, 1.4)):
            true_case = _copy(ppc)
            true_case["branch"][row, 3] *= scale
            z = _h(true_case, theta, V) + rng.normal(0.0, _sigma(nb, nl))
            out = lp.lagrangian_m_singlephase_details(z, ppc, 0, ppc["bus"])
            self.assertEqual(out["success"], 1)
            per_line = np.abs(out["lambdaN"]).reshape(nl, 2).max(axis=1)
            with self.subTest(row=row, scale=scale):
                self.assertEqual(int(np.argmax(per_line)), row)
                self.assertGreater(per_line[row], 3.0)

    def test_open_branch_multiplier_is_zero_not_nan(self) -> None:
        ppc = _load_python_case(CASE14)
        ppc["branch"][1, 10] = 0.0
        z = _h(ppc)
        out = lp.lagrangian_m_singlephase_details(z, ppc, 0, ppc["bus"])
        self.assertTrue(np.all(np.isfinite(out["lambdaN"])))
        self.assertEqual(out["lambdaN"][2], 0.0)
        self.assertEqual(out["lambdaN"][3], 0.0)


class GlobalChiSquareTests(unittest.TestCase):
    """The global bad-data test must use J = e' R^-1 e, not sum(r_norm^2)."""

    def test_clean_noise_false_alarm_rate_matches_alpha(self) -> None:
        ppc = _load_python_case(CASE14)
        case_int = _internal(ppc)
        nb, nl = case_int["bus"].shape[0], case_int["branch"].shape[0]
        z_true = _h(ppc)
        sigma = _sigma(nb, nl)
        dof = z_true.size - (2 * nb - 1)
        rng = np.random.default_rng(0)
        draws = 100
        raw, normalized = [], []
        for _ in range(draws):
            out = lp.lagrangian_m_singlephase_details(z_true + rng.normal(0.0, sigma), ppc, 0, ppc["bus"])
            self.assertEqual(out["success"], 1)
            raw.append(out["wls_objective"])
            normalized.append(out["sum_normalized_residual_sq"])
        raw = np.asarray(raw)
        normalized = np.asarray(normalized)
        threshold = chi2.ppf(0.95, dof)
        false_alarm = float(np.mean(raw > threshold))
        # Nominal 5%; with 100 draws allow generous binomial slack.
        self.assertLessEqual(false_alarm, 0.12)
        self.assertAlmostEqual(float(raw.mean()) / dof, 1.0, delta=0.15)
        # The previously used statistic has expectation ~m, not m - n, and
        # therefore rejects clean data far above alpha.
        self.assertGreater(float(normalized.mean()), 1.15 * dof)
        self.assertGreater(float(np.mean(normalized > threshold)), 0.3)

    def test_wls_payload_carries_raw_objective(self) -> None:
        ppc = _load_python_case(CASE14)
        payload = _wls_json(CASE14, _h(ppc).tolist())
        self.assertTrue(payload["success"])
        for key in (
            "wls_objective",
            "global_residual_sum",
            "sum_normalized_residual_sq",
            "raw_residual",
            "theta_est_rad",
            "vm_est_pu",
            "dof",
        ):
            self.assertIn(key, payload)
        self.assertEqual(payload["global_residual_sum"], payload["wls_objective"])
        self.assertLess(payload["wls_objective"], 1e-6)
        self.assertEqual(payload["dof"], 122 - 27)
        self.assertEqual(payload["lambda_layout"], "per_branch_R_X_interleaved")

    def test_gross_error_is_detected(self) -> None:
        ppc = _load_python_case(CASE14)
        case_int = _internal(ppc)
        nb, nl = case_int["bus"].shape[0], case_int["branch"].shape[0]
        z = _h(ppc) + np.random.default_rng(5).normal(0.0, _sigma(nb, nl))
        z[3 * nb + 4] += 0.2  # 20 sigma on a Pf channel
        payload = _wls_json(CASE14, z.tolist())
        self.assertGreater(payload["wls_objective"], chi2.ppf(0.99, payload["dof"]))
        self.assertEqual(int(np.argmax(np.abs(payload["r"]))), 3 * nb + 4)


class MeasurementCorrectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ppc = _load_python_case(CASE14)
        case_int = _internal(self.ppc)
        self.nb, self.nl = case_int["bus"].shape[0], case_int["branch"].shape[0]
        self.sigma = _sigma(self.nb, self.nl)
        self.z_true = _h(self.ppc)
        self.rng = np.random.default_rng(7)

    def _check_recovery(self, errors: dict[int, float]) -> None:
        z = self.z_true + self.rng.normal(0.0, self.sigma)
        z_obs = z.copy()
        for index, error in errors.items():
            z_obs[index] += error
        payload = _meas_correction_json(
            CASE14,
            z_obs.tolist(),
            suspect_group=sorted(errors),
            enable_correction=True,
            max_correction_iterations=2,
            error_tolerance=1e-3,
        )
        self.assertTrue(payload["success"])
        corrected = {item["index0"]: item["corrected"] for item in payload["corrected_measurements"]}
        self.assertEqual(set(corrected), set(errors))
        for index in errors:
            with self.subTest(index=index):
                self.assertLess(abs(corrected[index] - self.z_true[index]), 4.0 * self.sigma[index])
        repaired = z_obs.copy()
        for index, value in corrected.items():
            repaired[index] = value
        verify = _wls_json(CASE14, repaired.tolist())
        self.assertLess(verify["wls_objective"], chi2.ppf(0.95, verify["dof"]))
        # Healthy meters are untouched by construction of the grouped update.
        untouched = [i for i in range(repaired.size) if i not in errors]
        np.testing.assert_array_equal(repaired[untouched], z_obs[untouched])

    def test_singleton_gross_error_recovery(self) -> None:
        self._check_recovery({3 * self.nb + 3: 0.30})

    def test_two_simultaneous_cross_channel_errors(self) -> None:
        self._check_recovery({self.nb + 3: -0.25, 3 * self.nb + 3 * self.nl + 6: 0.20})

    def test_three_simultaneous_errors_incl_voltage(self) -> None:
        self._check_recovery({4: 0.03, self.nb + 8: 0.2, 3 * self.nb + self.nl + 12: -0.15})


class ParameterCorrectionTests(unittest.TestCase):
    """Multi-scan R/X recovery from observable-only starts through the tool path."""

    def _scans(self, true_case: dict, count: int, seed: int) -> list[list[float]]:
        case_int = _internal(true_case)
        nb, nl = case_int["bus"].shape[0], case_int["branch"].shape[0]
        theta, V = _case_state(case_int)
        ref = int(np.flatnonzero(case_int["bus"][:, lp.BUS_TYPE].astype(int) == lp.REF)[0])
        rng = np.random.default_rng(seed)
        scans = []
        for k in range(count):
            # Distinct operating points that are exactly consistent with the
            # true network model (WLS only needs z = h(x) for some x).
            theta_k = theta * (0.7 + 0.2 * k) + rng.normal(0.0, 0.01, nb)
            theta_k[ref] = 0.0
            V_k = V + rng.normal(0.0, 0.01, nb)
            z = pp.calculate_hx(case_int, theta_k, V_k) + rng.normal(0.0, _sigma(nb, nl))
            scans.append(z.tolist())
        return scans

    def _recover(self, row0: int, r_scale: float, x_scale: float, seed: int = 11) -> tuple[float, float, dict]:
        truth = _load_python_case(CASE14)
        model = _copy(truth)
        model["branch"][row0, 2] *= r_scale
        model["branch"][row0, 3] *= x_scale
        scans = self._scans(truth, 4, seed)
        with tempfile.TemporaryDirectory() as tmp:
            model_path = _write_case(tmp, model, f"model_l{row0}")
            initial = observable_parameter_initial_states(model, scans)
            payload = _param_correction_json(model_path, row0 + 1, scans, initial)
        self.assertTrue(payload["success"], payload.get("error"))
        r_est, x_est = payload["corrected_params"]
        return r_est, x_est, payload

    def test_recovers_coupled_rx_error_on_plain_line(self) -> None:
        row0 = 2  # bus 2-3
        truth = _load_python_case(CASE14)
        r_est, x_est, _ = self._recover(row0, 1.3, 1.4)
        self.assertAlmostEqual(r_est / truth["branch"][row0, 2], 1.0, delta=0.05)
        self.assertAlmostEqual(x_est / truth["branch"][row0, 3], 1.0, delta=0.05)

    def test_recovers_reactance_only_error(self) -> None:
        row0 = 6  # bus 4-5
        truth = _load_python_case(CASE14)
        r_est, x_est, _ = self._recover(row0, 1.0, 0.75, seed=12)
        self.assertAlmostEqual(x_est / truth["branch"][row0, 3], 1.0, delta=0.05)
        self.assertAlmostEqual(r_est / truth["branch"][row0, 2], 1.0, delta=0.10)

    def test_recovers_transformer_reactance_with_zero_resistance(self) -> None:
        row0 = 7  # bus 4-7, tap 0.978, R = 0
        truth = _load_python_case(CASE14)
        model = _copy(truth)
        model["branch"][row0, 3] *= 1.25
        scans = self._scans(truth, 4, 13)
        diagnostics: dict = {}
        params, success = pp.correct_parameter_group_multi_scan(
            model,
            row0 + 1,
            np.asarray(scans, dtype=float).T,
            np.asarray(observable_parameter_initial_states(model, scans), dtype=float).T,
            _sigma(14, 20) ** 2,
            None,
            diagnostics=diagnostics,
        )
        self.assertEqual(success, 1, diagnostics)
        self.assertAlmostEqual(params[1] / truth["branch"][row0, 3], 1.0, delta=0.05)
        self.assertGreater(diagnostics["objective_reduction"], 0.9)
        self.assertEqual(diagnostics["termination_reason"], "converged")

    def test_convergence_diagnostics_report_objective_reduction(self) -> None:
        row0 = 2
        truth = _load_python_case(CASE14)
        model = _copy(truth)
        model["branch"][row0, 3] *= 1.4
        scans = self._scans(truth, 3, 14)
        diagnostics: dict = {}
        pp.correct_parameter_group_multi_scan(
            model,
            row0 + 1,
            np.asarray(scans, dtype=float).T,
            np.asarray(observable_parameter_initial_states(model, scans), dtype=float).T,
            _sigma(14, 20) ** 2,
            None,
            diagnostics=diagnostics,
        )
        self.assertTrue(diagnostics["converged"])
        self.assertFalse(diagnostics["parameter_at_floor"])
        self.assertLess(diagnostics["objective_final"], diagnostics["objective_initial"])


class TopologyIdentificationTests(unittest.TestCase):
    """Both error directions through the deployment context and executor."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.providers = MatpowerDeploymentProviders(derived_case_dir=self.tmp.name)
        self.truth = _load_python_case(CASE14)
        # IEEE case14 stores bus 8 at 1.09 p.u. against its own 1.06 limit.  The
        # deployment candidate screen fails closed on measured Vm outside the
        # declared bounds, so the synthetic truth must respect them.
        bus = self.truth["bus"]
        bus[:, 7] = np.minimum(np.maximum(bus[:, 7], bus[:, 12]), bus[:, 11])
        case_int = _internal(self.truth)
        self.nb, self.nl = case_int["bus"].shape[0], case_int["branch"].shape[0]
        self.rng = np.random.default_rng(21)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _state(self, model_path: str, z: np.ndarray) -> dict:
        return {
            "state_id": "episode:s0",
            "state_hash": "hash0",
            "case": model_path,
            "measurements": z.tolist(),
            "metadata": {},
        }

    def test_modeled_closed_true_open_line_is_identified_and_repaired(self) -> None:
        row0 = 1  # bus 1-5; opening it keeps the network connected
        true_case = _copy(self.truth)
        true_case["branch"][row0, 10] = 0.0
        z = _h(true_case) + self.rng.normal(0.0, _sigma(self.nb, self.nl))
        state = self._state(CASE14, z)
        wls = self.providers.run_wls(state)
        self.assertFalse(wls["no_material_anomaly_remaining"])
        context = self.providers.get_topology_context(state)
        self.assertNotIn("execution_status", context)
        self.assertEqual(context["topology_findings"][0]["line_row0"], row0)
        proposals = [
            (a["arguments"]["line_index"], a["arguments"]["status"])
            for a in context["supported_corrections"]
        ]
        self.assertIn((row0 + 1, 0), proposals)
        correction = self.providers.correct_topology(
            state, {"tool": "correct_topology", "arguments": {"line_index": row0 + 1, "status": 0}}
        )
        derived = correction["modification"]["case"]
        diff = matpower_case_differ(CASE14, derived)
        self.assertEqual(diff["changed_branch_rows"], {row0: [10]})
        verify = _wls_json(derived, z.tolist())
        self.assertLess(verify["wls_objective"], chi2.ppf(0.95, verify["dof"]))

    def test_modeled_open_true_closed_line_is_enumerated_and_repaired(self) -> None:
        row0 = 1
        model = _copy(self.truth)
        model["branch"][row0, 10] = 0.0
        model_path = _write_case(self.tmp.name, model, "model_open_l2")
        z = _h(self.truth) + self.rng.normal(0.0, _sigma(self.nb, self.nl))
        state = self._state(model_path, z)
        context = self.providers.get_topology_context(state)
        self.assertNotIn("execution_status", context)
        self.assertIn(row0 + 1, context["enumerated_close_hypotheses"])
        proposals = [
            (a["arguments"]["line_index"], a["arguments"]["status"])
            for a in context["supported_corrections"]
        ]
        self.assertIn((row0 + 1, 1), proposals)
        screening = {
            int(item["line_index"]): item for item in context["topology_candidate_screening"]
        }
        self.assertEqual(
            screening[row0 + 1]["hypothesis_source"], "out_of_service_status_enumeration"
        )
        correction = self.providers.correct_topology(
            state, {"tool": "correct_topology", "arguments": {"line_index": row0 + 1, "status": 1}}
        )
        verify = _wls_json(correction["modification"]["case"], z.tolist())
        self.assertLess(verify["wls_objective"], chi2.ppf(0.95, verify["dof"]))

    def test_executor_rejects_non_binary_status(self) -> None:
        z = _h(self.truth)
        state = self._state(CASE14, z)
        for bad in (2, -1, 0.5, "open"):
            with self.subTest(status=bad):
                result = self.providers.correct_topology(
                    state, {"tool": "correct_topology", "arguments": {"line_index": 2, "status": bad}}
                )
                self.assertEqual(result.get("execution_status"), "failure")
                self.assertEqual(result["error_code"], "topology_correction_invalid_status")


class HarmonicStateEstimationTests(unittest.TestCase):
    def setUp(self) -> None:
        from Harmonics import hse_utils

        self.hse = hse_utils
        self.ppc = _load_python_case(CASE14)
        self.bus = np.asarray(self.ppc["bus"], dtype=float)
        self.branch = np.asarray(self.ppc["branch"], dtype=float)
        self.base = float(self.ppc["baseMVA"])
        self.nb = self.bus.shape[0]

    def _voltages(self, source_bus0: int | None, branch: np.ndarray, h: int = 5) -> np.ndarray:
        unknown = [i for i in range(self.nb) if i != 0]
        yh = self.hse.build_ybus_harmonic(self.bus, branch, self.base, h)
        yuu = yh[np.ix_(unknown, unknown)]
        injection = np.zeros(len(unknown), dtype=complex)
        if source_bus0 is not None:
            injection[unknown.index(source_bus0)] = 0.08 * np.exp(1j * 0.4)
        v = np.zeros(self.nb, dtype=complex)
        v[unknown] = np.linalg.solve(yuu, injection)
        return v

    def _measurements(self, v: np.ndarray, noise: float, seed: int) -> list[dict]:
        rng = np.random.default_rng(seed)
        noisy = v + rng.normal(0.0, noise, self.nb) + 1j * rng.normal(0.0, noise, self.nb)
        return [
            {"h": 5, "bus": i + 1, "V_real": float(noisy[i].real), "V_imag": float(noisy[i].imag), "sigma": 1e-4}
            for i in range(self.nb)
        ]

    def test_open_branch_is_removed_from_harmonic_ybus(self) -> None:
        closed = self.hse.build_ybus_harmonic(self.bus, self.branch, self.base, 5)
        opened_branch = self.branch.copy()
        opened_branch[0, 10] = 0.0  # bus 1-2
        opened = self.hse.build_ybus_harmonic(self.bus, opened_branch, self.base, 5)
        self.assertNotEqual(closed[0, 1], 0.0)
        self.assertEqual(opened[0, 1], 0.0)

    def test_single_source_is_localized_and_beats_null_model(self) -> None:
        for source in (8, 11, 2):  # buses 9, 12, 3
            measurements = self._measurements(self._voltages(source, self.branch), 1e-4, source)
            payload = _run_hse_logic(CASE14, measurements, [5], 0)
            with self.subTest(source_bus_1based=source + 1):
                self.assertTrue(payload["success"], payload.get("error"))
                self.assertEqual(payload["best_candidate_bus_1based"], source + 1)
                self.assertGreater(payload["sse_reduction_vs_null"], 0.9)

    def test_no_source_does_not_beat_null_model(self) -> None:
        measurements = self._measurements(self._voltages(None, self.branch), 1e-4, 99)
        payload = _run_hse_logic(CASE14, measurements, [5], 0)
        self.assertTrue(payload["success"])
        self.assertLess(payload["sse_reduction_vs_null"], 0.5)

    def test_localization_under_open_line_topology(self) -> None:
        opened_branch = self.branch.copy()
        opened_branch[1, 10] = 0.0  # bus 1-5
        measurements = self._measurements(self._voltages(8, opened_branch), 1e-4, 3)
        with tempfile.TemporaryDirectory() as tmp:
            model = _copy(self.ppc)
            model["branch"][1, 10] = 0.0
            path = _write_case(tmp, model, "hse_open")
            payload = _run_hse_logic(path, measurements, [5], 0)
        self.assertEqual(payload["best_candidate_bus_1based"], 9)
        self.assertGreater(payload["sse_reduction_vs_null"], 0.9)

    def test_observed_fundamental_voltage_is_used_for_thd(self) -> None:
        measurements = self._measurements(self._voltages(8, self.branch), 1e-4, 4)
        observed = _h(self.ppc)[: self.nb] * 0.5  # halve |V1| -> THD doubles
        base_payload = _run_hse_logic(CASE14, measurements, [5], 0)
        scaled_payload = _run_hse_logic(CASE14, measurements, [5], 0, fundamental_vm=observed.tolist())
        self.assertEqual(scaled_payload["fundamental_voltage_source"], "observed_vm")
        self.assertAlmostEqual(
            scaled_payload["estimated_thd_percent"]["9"] / base_payload["estimated_thd_percent"]["9"],
            2.0,
            delta=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
