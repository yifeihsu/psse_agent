"""IEEE118 three-phase realizations: source voltage bases, the Newton solve
fallback, and voltage-regulated generators with reactive limits."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import opendssdirect
import pytest
from pypower.idx_bus import BASE_KV, PD, QD, VM
from pypower.idx_gen import GEN_BUS, QG, QMAX, QMIN, VG

from psse_env.systems import resolve_system
from three_phase_model import runtime
from three_phase_model.diagnostics import capture_nominal_model, screen_measurements
from three_phase_model.disturbances import inject_midspan_hif
from three_phase_model.exporter import PV_GENERATORS, export_model, solve_reference
from three_phase_model.measurements import extract_measurements
from three_phase_model.validation import validate_model
from three_phase_model.voltage_bases import (
    IEEE118_SOURCE_KV, IEEE118_VOLTAGE_BASE_PROFILE_ID, apply_ieee118_voltage_bases,
    ieee118_hif_branch_eligibility,
)


@pytest.fixture(scope="module")
def source():
    return resolve_system("case118").load_case()


@pytest.fixture(scope="module")
def models(tmp_path_factory, source):
    root = tmp_path_factory.mktemp("ieee118")
    return {
        "normalized": export_model(source, root / "normalized", case_id="case118"),
        "physical_pv": export_model(source, root / "physical_pv", case_id="case118",
                                    voltage_profile=IEEE118_VOLTAGE_BASE_PROFILE_ID, generator_control="pv_q_limits"),
    }


def _master(model):
    return Path(model["output_dir"]) / "Master.dss"


def test_profile_is_the_source_basekv_and_excludes_the_two_zero_tap_cross_voltage_branches(source):
    assert {int(row[0]): float(row[BASE_KV]) for row in source["bus"]} == IEEE118_SOURCE_KV
    eligibility = ieee118_hif_branch_eligibility(apply_ieee118_voltage_bases(source))
    eligible = [row for row in eligibility["branch_rows"] if row["eligible"]]
    assert len(eligible) == 175
    assert sum(row["kv_ll"] == 345.0 for row in eligible) == 10
    zero_tap_cross = [(row["from_bus"], row["to_bus"]) for row in eligibility["excluded_branches"]
                      if row["exclusion_reasons"] == ["cross_voltage_branch"]]
    assert zero_tap_cross == [(86, 87), (68, 116)]
    conflicting = deepcopy(source)
    conflicting["bus"][115, BASE_KV] = 345.0
    with pytest.raises(ValueError, match="differs from the source profile"):
        apply_ieee118_voltage_bases(conflicting)


def test_opendss_alone_lands_off_the_operating_point_and_the_fallback_recovers_it(models):
    model = models["normalized"]
    raw = opendssdirect.NewContext()
    raw.Basic.AllowChangeDir(False)
    raw.Text.Command(f'Compile "{_master(model).resolve()}"')
    # The current-injection iteration "converges" with loads on constant-Z fallback.
    assert raw.Solution.Converged()
    assert runtime.constant_pq_setpoint_deviation(raw)["relative_deviation"] > 1e-2
    dss = runtime.compile_model(_master(model))
    assert runtime.device_control_deviation(dss)["passed"]
    report = validate_model(dss, model["reference"], model["registry"], model["assumptions"])
    assert report["passed"], report["failed_checks"]
    assert report["checks"]["balanced_bus_voltage_magnitude"]["max_error"] < 1e-9


def test_reactive_limited_reference_satisfies_complementarity(source):
    reference = solve_reference(source, PV_GENERATORS)
    limited = reference["reactive_limited_gen_rows0"]
    assert sorted(int(reference["gen"][i, GEN_BUS]) for i in limited) == [19, 32, 34, 92, 103, 105]
    rows = {int(number): i for i, number in enumerate(reference["bus"][:, 0])}
    for i, gen in enumerate(reference["gen"]):
        if int(gen[GEN_BUS]) == 69:
            continue
        voltage = reference["bus"][rows[int(gen[GEN_BUS])], VM]
        assert gen[QMIN] - 1e-6 <= gen[QG] <= gen[QMAX] + 1e-6
        if limited.get(i) == "max":
            assert gen[QG] == pytest.approx(gen[QMAX]) and voltage <= gen[VG] + 1e-9
        elif limited.get(i) == "min":
            assert gen[QG] == pytest.approx(gen[QMIN]) and voltage >= gen[VG] - 1e-9
        else:
            assert voltage == pytest.approx(gen[VG], abs=1e-9)


def test_regulated_physical_model_validates_and_opendss_holds_the_control_law(models):
    model = models["physical_pv"]
    assert sum(row["dss_element"].startswith("Transformer.") for row in model["registry"]["branches"]) == 11
    generators = model["registry"]["generators"]
    assert len(generators) == 53 and all(row["control"] == "pv" and row["phases"] == [1, 2, 3] for row in generators)
    assert sorted(row["bus"] for row in generators if row["reference_reactive_limit"]) == [19, 32, 34, 92, 103, 105]
    dss = runtime.compile_model(_master(model))
    report = validate_model(dss, model["reference"], model["registry"], model["assumptions"])
    assert report["passed"], report["failed_checks"]
    runtime.redistribute_load(dss, model["registry"], bus=12, delta=0.2)
    control = runtime.regulated_generator_deviation(dss)
    assert control["passed"] and control["generator_count"] == 53
    before = runtime._regulated_reactive(dss)
    dss.Text.Command("Set MaxIterations=3")
    dss.Solution.Solve()
    assert max(abs(runtime._regulated_reactive(dss)[name] - q) for name, q in before.items()) < 1.0  # var


def test_strong_hif_solves_with_regulation_and_healthy_screen_stays_quiet(models):
    model = models["physical_pv"]
    dss = runtime.compile_model(_master(model))
    nominal = capture_nominal_model(dss, model["registry"], model["assumptions"])
    assert sum(bool(row.get("regulated_generator")) for row in nominal["buses"]) == 53
    healthy = extract_measurements(dss, model["registry"], model["assumptions"])
    assert screen_measurements(healthy, nominal)["classification"] == "no_detectable_anomaly"
    # 100 ohm on line 1-2 at 138 kV has no fixed-PQ operating point (bus 10 rises
    # without bound on the charged 345 kV corridor); voltage regulation holds it.
    receipt = inject_midspan_hif(dss, model["registry"], model["assumptions"], branch_row0=0, phase=1,
                                 resistance_ohm=100.0)
    assert runtime.device_control_deviation(dss)["passed"]
    faulted = extract_measurements(dss, model["registry"], model["assumptions"],
                                   branch_overrides=receipt["branch_overrides"])
    screen = screen_measurements(faulted, nominal)
    assert screen["classification"] == "hif_like_branch_mismatch"
    assert (screen["hif_candidate"]["branch_row0"], screen["hif_candidate"]["phase"]) == (0, 1)


def test_fixed_pq_generators_have_no_operating_point_for_the_same_fault(tmp_path, source):
    model = export_model(source, tmp_path / "physical_pq", case_id="case118",
                         voltage_profile=IEEE118_VOLTAGE_BASE_PROFILE_ID)
    dss = runtime.compile_model(_master(model))
    with pytest.raises(RuntimeError, match="No operating point|did not converge|control law"):
        inject_midspan_hif(dss, model["registry"], model["assumptions"], branch_row0=0, phase=1,
                           resistance_ohm=100.0)


def test_load_scaled_regulated_reference_limits_more_units(source):
    scaled = deepcopy(source)
    scaled["bus"][:, [PD, QD]] *= 0.8
    assert len(solve_reference(scaled, PV_GENERATORS)["reactive_limited_gen_rows0"]) == 16
