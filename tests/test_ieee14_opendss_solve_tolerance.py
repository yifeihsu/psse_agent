"""The IEEE-14 OpenDSS model must converge tightly on every compile path.

At OpenDSS's default tolerance (1e-4) a solve with the four regulating PV
generators stopped up to 1.7 measurement sigmas short of the converged power
flow, and the HIF operating-point path and the unbalance generator's
load-scaling path stopped on different sides of it (1.5 sigma apart on the
bus-2/3 injections). Run_IEEE14Bus.dss now sets tolerance 1e-8.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("opendssdirect")

from IEEE_14_OpenDSS.export_measurement_series import extract_measurement_series  # noqa: E402
from three_phase_nlm.hif_operating_point import apply_hif_operating_point, capture_operating_point_baseline  # noqa: E402
from three_phase_nlm.hif_parameter_estimator import (  # noqa: E402
    _compile_base_model, _resolve_model_dir, _simulate_base, _solve_or_raise,
)
from Transmission import generate_measurements_imbalance as gi  # noqa: E402

SIGMA = np.r_[np.full(14, 1e-3), np.full(108, 1e-2)]


def _z():
    return np.asarray(extract_measurement_series(shunt_convention="ybus")[0], dtype=float)


def _solution_settings():
    import opendssdirect as dss

    return float(dss.Solution.Convergence()), int(dss.Solution.MaxIterations())


def test_a_diverged_solve_is_redone_from_a_fresh_compile(monkeypatch):
    """One sporadic divergence is retried from scratch; a persistent one raises after the attempts."""
    import three_phase_nlm.hif_parameter_estimator as hpe

    model = _resolve_model_dir(None, "case14")
    reference = _simulate_base(model, op_point={"load_scale": 1.1}, shunt_convention="ybus")["z"]
    real = hpe._solve_or_raise
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("OpenDSS solve did not converge (simulated divergence)")
        real()

    monkeypatch.setattr(hpe, "_solve_or_raise", flaky)
    with pytest.warns(UserWarning, match="retrying from a fresh compile"):
        z = _simulate_base(model, op_point={"load_scale": 1.1}, shunt_convention="ybus")["z"]
    assert calls["n"] == 2 and z == reference

    def stuck():
        raise RuntimeError("OpenDSS solve did not converge (simulated divergence)")

    monkeypatch.setattr(hpe, "_solve_or_raise", stuck)
    with pytest.warns(UserWarning), pytest.raises(RuntimeError, match="3 fresh attempts"):
        _simulate_base(model, op_point={"load_scale": 1.1}, shunt_convention="ybus")


@pytest.mark.parametrize("load_scale", [0.8, 1.0, 1.25])
def test_both_compile_paths_converge_to_the_same_balanced_solution(load_scale):
    import opendssdirect as dss

    model = _resolve_model_dir(None, "case14")
    _compile_base_model(model)
    assert _solution_settings() == (1e-8, 200)
    apply_hif_operating_point(capture_operating_point_baseline(), {"load_scale": load_scale})
    _solve_or_raise()
    z_hif, iterations = _z(), int(dss.Solution.Iterations())
    _solve_or_raise()
    assert np.max(np.abs(_z() - z_hif) / SIGMA) < 1e-3  # a re-solve does not move a converged solution

    gi._compile_ieee14_opendss(str(model))
    assert _solution_settings() == (1e-8, 200)
    gi._scale_all_loads(gi._read_base_loads(), load_scale)
    dss.Text.Command("Solve")
    assert dss.Solution.Converged()
    z_unbalance_path = _z()
    assert np.max(np.abs(z_hif - z_unbalance_path) / SIGMA) < 1e-3
    assert iterations < 200
