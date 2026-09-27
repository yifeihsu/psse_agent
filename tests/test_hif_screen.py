"""The balanced HIF screen (psse_env/providers/hif_screen.py).

The screen refits the operator's balanced model under a meter, a branch
parameter, a line outage and a split-line shunt, and flags an HIF when the
shunt wins the penalized chi-square comparison.  These tests pin its measurement
model to the repository's WLS, its decisions on real roots, and the two-round
rule that finds an HIF behind a bad meter.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from mcp_server.matpower_server import _load_python_case
from psse_env.providers.hif_screen import (
    DEFAULT_HIF_SCREEN_CONFIG, HifScreenCache, _h_and_jac, _Operator, _ybus_dense, default_hif_lines, screen_hif,
)
from psse_env.providers.scenario_generator import PHYSICAL_HIF_SAMPLE_PATHS, Round0ScenarioGenerator
from tools.lagrangian_port import _copy_result_to_internal, make_jaco, make_ybus

CASE = _load_python_case("case14")
PPC = _copy_result_to_internal(CASE)
LINES = default_hif_lines(CASE)


def _hif_rows(count: int) -> list[dict]:
    path = PHYSICAL_HIF_SAMPLE_PATHS[0]
    if not path.is_file():
        pytest.skip("tracked 20260923opf HIF corpus is not checked out")
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if (row.get("label") or {}).get("error_type") != "no_error":
            rows.append(row)
        if len(rows) == count:
            break
    return rows


def _screen(z, sigma, exact=()):
    return screen_hif(PPC["baseMVA"], PPC["bus"], PPC["branch"], z, sigma, exact, lines=LINES)


def test_candidates_are_the_sixteen_same_voltage_lines():
    # Every IEEE 14 line except the 13.8/18 kV branch 7-8 (row 13); rows 7-9 are transformers.
    assert LINES == [0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 14, 15, 16, 17, 18, 19]


def test_dense_model_matches_the_repository_admittance_and_jacobian():
    operator = _Operator(PPC["baseMVA"], PPC["bus"], PPC["branch"])
    rng = np.random.default_rng(0)
    for model in (operator.base_model(), operator.split_model(3, 0.3), operator.outage_model(5)):
        ybus, yf, yt = make_ybus(model.base_mva, model.bus, model.branch)
        dense = _ybus_dense(model.base_mva, model.bus, model.branch)
        assert np.max(np.abs(ybus.toarray() - dense[0])) < 1e-12
        assert np.max(np.abs(yf.toarray() - dense[1])) < 1e-12
        va = rng.normal(0.0, 0.2, model.nb)
        vm = 1.0 + rng.normal(0.0, 0.03, model.nb)
        voltage = vm * np.exp(1j * va)
        reference, _sf, _st = make_jaco(np.r_[va, vm], ybus, yf, yt, model.nb, model.nl,
                                        model.branch[:, 0].astype(int), model.branch[:, 1].astype(int), voltage)
        _h, jac = _h_and_jac(*dense, va, vm)
        assert np.max(np.abs(reference.toarray() - jac)) < 1e-10


def test_an_hif_window_is_flagged_on_its_line():
    row = _hif_rows(1)[0]
    report = _screen(row["z_obs"], row["sigma_z"])
    assert report["status"] == "valid"
    assert report["suspected"] is True
    assert report["outcome"] == "hif"
    assert report["branch_row0"] == int(row["label"]["branch_row0"])
    assert 0.0 < report["alpha"] < 1.0 and report["shunt_conductance_pu"] > 0.0


def test_an_hif_behind_a_bad_meter_needs_the_second_round():
    row = _hif_rows(1)[0]
    z = list(row["z_obs"])
    bad = 31  # Qinj at bus 4, a gross error of 25 sigma
    z[bad] += 25.0 * float(row["sigma_z"][bad])
    report = _screen(z, row["sigma_z"])
    assert report["outcome"] == "meter>hif"
    assert report["suspected"] is True
    assert report["meter_set_aside_index0"] == bad
    assert report["branch_row0"] == int(row["label"]["branch_row0"])


def test_balanced_errors_are_not_flagged():
    generator = Round0ScenarioGenerator(seed=20260927, normalized_residual_threshold=4.0)
    built = generator.build({"measurement": 2, "multi_measurement": 1, "parameter": 2, "topology": 1})
    for scenario in built:
        metadata = scenario["metadata"]
        case = _copy_result_to_internal(_load_python_case(scenario["case"]))
        report = screen_hif(
            case["baseMVA"], case["bus"], case["branch"], scenario["measurements"], metadata["sigma_z"],
            metadata.get("structural_zero_indices") or (), lines=default_hif_lines(_load_python_case(scenario["case"])),
        )
        assert report["status"] == "valid", scenario["scenario_family"]
        assert report["suspected"] is False, (scenario["scenario_family"], report["outcome"])


def test_cache_keys_bind_every_input():
    base = HifScreenCache.key([1.0, 2.0], [0.1, 0.1], config=DEFAULT_HIF_SCREEN_CONFIG)
    assert base == HifScreenCache.key([1.0, 2.0], [0.1, 0.1], config=DEFAULT_HIF_SCREEN_CONFIG)
    assert base != HifScreenCache.key([1.0, 2.0 + 1e-12], [0.1, 0.1], config=DEFAULT_HIF_SCREEN_CONFIG)
    assert base != HifScreenCache.key([1.0, 2.0], [0.1, 0.2], config=DEFAULT_HIF_SCREEN_CONFIG)
    cache = HifScreenCache(limit=1)
    cache.put("a", {"x": 1})
    cache.put("b", {"x": 2})
    assert cache.get("a") is None and cache.get("b") == {"x": 2}
