"""OPF-driven operating points for the OpenDSS-generated corpora (2026-09-23).

The pypower scenario families sit on the case14 AC-OPF dispatch at their load
scale, while the OpenDSS HIF and unbalance corpora ran the checked-in model
dispatch (bus 2 at 40 MW, 1 kW condensers, bus 8 near 1.09 pu). Balanced SCADA
alone separated the two groups. Both generators now take every window's and
scan's generator dispatch, PV setpoints and source voltage from the same
AC-OPF at the same per-bus loads, store them in the canonical op_point, and
every replay path reproduces them. The fault labels and load profiles of a
seed are unchanged, and an OPF failure skips the window instead of falling
back to the model dispatch.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("opendssdirect")

from three_phase_nlm.hif_operating_point import (  # noqa: E402
    DISPATCH_MODE_CASE14,
    DISPATCH_MODE_OPF,
    OPF_DISPATCHED_GENERATORS,
    OPFDispatchError,
    apply_hif_operating_point,
    canonicalize_ieee14_operating_point,
    capture_operating_point_baseline,
    ieee14_opf_operating_point,
    operating_point_from_opf_solution,
    solve_ieee14_opf,
)
from three_phase_nlm.hif_parameter_estimator import (  # noqa: E402
    _compile_base_model, _resolve_model_dir, _simulate_base, _solve_or_raise, simulate_hif_candidate,
)
from Transmission import generate_measurements_hif_ieee14 as ghi  # noqa: E402
from Transmission import generate_measurements_imbalance as gi  # noqa: E402
from Transmission.generate_measurements import compute_measurements_pu  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SIGMA = np.r_[np.full(14, 1e-3), np.full(108, 1e-2)]
PINJ_BUS1, PINJ_BUS3, VM_BUS8, QINJ_BUS8 = 14, 16, 7, 28 + 7
PROFILE = {"b3": 0.9, "b4": 1.1, "b9": 1.05}


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _noise(scan: dict) -> np.ndarray:
    return np.asarray(scan["z_obs"]) - np.asarray(scan["z_clean"])


# ------------------------------------------------------------------ replay path
def test_opf_operating_point_is_reproduced_by_the_opendss_replay_path():
    """The canonical op_point carries the OPF dispatch; _simulate_base lands on the pypower OPF solution."""
    opf = ieee14_opf_operating_point(1.1, PROFILE)
    from pypower.idx_gen import GEN_BUS, PG
    pg = {f"b{int(row[GEN_BUS])}": float(row[PG]) * 1000.0 for row in opf.solution["gen"]}
    assert set(opf.op_point) == set(canonicalize_ieee14_operating_point({}))
    assert opf.op_point["generator_dispatch_kw"] == {bus: pg[bus] for bus in OPF_DISPATCHED_GENERATORS}
    assert opf.op_point["bus_load_scales"]["b3"] == 0.9 and opf.op_point["load_scale"] == 1.1
    assert opf.receipt["mode"] == DISPATCH_MODE_OPF and opf.receipt["slack_pg_kw"] == pg["b1"]

    z_opendss = np.asarray(_simulate_base(_resolve_model_dir(None, "case14"), op_point=opf.op_point,
                                          shunt_convention="ybus")["z"])
    z_pypower = compute_measurements_pu(opf.solution)
    assert np.max(np.abs(z_opendss - z_pypower) / SIGMA) < 0.1  # 0.02 sigma measured; the models agree
    assert abs(z_opendss[VM_BUS8] - opf.op_point["voltage_setpoints_pu"]["b8"]) < 1e-4


def test_slack_is_folded_into_the_source_voltage():
    """Vsource pu = OPF bus-1 voltage; the source then supplies the OPF slack active power."""
    import opendssdirect as dss
    from pypower.idx_bus import VM

    opf = ieee14_opf_operating_point(1.2)
    assert opf.op_point["source_voltage_pu"] == float(opf.solution["bus"][0, VM])
    model = _resolve_model_dir(None, "case14")
    _compile_base_model(model)
    applied = apply_hif_operating_point(capture_operating_point_baseline(), opf.op_point)
    assert applied["source_voltage_pu"] == {"source": opf.op_point["source_voltage_pu"]}
    assert float(dss.Vsources.PU()) == pytest.approx(opf.op_point["source_voltage_pu"])
    _solve_or_raise()
    z = _simulate_base(model, op_point=opf.op_point, shunt_convention="ybus")["z"]
    assert abs(z[PINJ_BUS1] - opf.receipt["slack_pg_kw"] / 1e5) < 1e-4  # pu on 100 MVA


def test_zero_kw_units_from_the_opf_still_regulate():
    """At light load the OPF idles the bus-3/6/8 units; a 0 kW PV unit keeps regulating in OpenDSS."""
    import opendssdirect as dss

    point = ieee14_opf_operating_point(0.8).op_point
    zero = canonicalize_ieee14_operating_point({**point, "generator_dispatch_kw": {**point["generator_dispatch_kw"],
                                                                                    "b3": 0.0, "b6": 0.0, "b8": 0.0}})
    model = _resolve_model_dir(None, "case14")
    _compile_base_model(model)
    apply_hif_operating_point(capture_operating_point_baseline(), zero)
    _solve_or_raise()
    for bus in ("b3", "b6", "b8"):
        dss.Circuit.SetActiveBus(bus)
        assert abs(float(np.mean(dss.Bus.puVmagAngle()[0::2][:3])) - zero["voltage_setpoints_pu"][bus]) < 1e-3
    assert _simulate_base(model, op_point=zero, shunt_convention="ybus")["z"][QINJ_BUS8] > 0.01


# ------------------------------------------------------------------ scan sampler
def test_scan_sampler_keeps_the_load_profiles_and_takes_the_dispatch_from_the_opf():
    kwargs = dict(scan_count=3, mode="diverse", event_load_scale=1.05, load_log_std=0.08,
                  dispatch_fraction=0.20, voltage_std=0.008)
    case14 = ghi._resolve_scan_operating_points(np.random.default_rng(11), dispatch_mode=DISPATCH_MODE_CASE14, **kwargs)
    opf = ghi._resolve_scan_operating_points(np.random.default_rng(11), dispatch_mode=DISPATCH_MODE_OPF, **kwargs)
    assert [p.op_point["bus_load_scales"] for p in case14] == [p.op_point["bus_load_scales"] for p in opf]
    assert all(p.op_point["load_scale"] == 1.05 for p in opf)
    assert all(v == 1.0 for v in opf[0].op_point["bus_load_scales"].values())
    for point in opf:
        expected = ieee14_opf_operating_point(point.op_point["load_scale"], point.op_point["bus_load_scales"]).op_point
        assert point.op_point == expected
        assert point.dispatch["mode"] == DISPATCH_MODE_OPF and point.opf is not None
        assert set(point.op_point) == set(canonicalize_ieee14_operating_point({}))  # canonical schema, nothing added
    assert case14[0].op_point["generator_dispatch_kw"]["b2"] == 40000.0
    assert opf[0].op_point["generator_dispatch_kw"]["b2"] != 40000.0
    assert case14[0].dispatch["mode"] == DISPATCH_MODE_CASE14 and case14[0].opf is None
    # the public helper keeps its list-of-dicts contract
    assert ghi._scan_operating_points(np.random.default_rng(11), dispatch_mode=DISPATCH_MODE_OPF, **kwargs) == [p.op_point for p in opf]


def test_case14_mode_reproduces_the_tracked_20260923b_operating_points():
    """Same seed: case14 mode gives the stored op_points; opf mode keeps their load profiles."""
    path = ROOT / "artifacts/measurements/hif_physical69_main_valid_detectable_8x10_20260923b/samples.jsonl"
    if not path.is_file():
        pytest.skip("tracked 20260923b subset not present")
    row = next(r for r in _rows(path) if r["scenario"] == "high_impedance_fault")
    seed, index = row["window_metadata"]["seed"], row["window_metadata"]["sample_index"]
    stored = [scan["op_point"] for scan in row["scans"]]
    kwargs = dict(scan_count=len(stored), mode=row["window_metadata"]["operating_point_mode"],
                  event_load_scale=row["op_point"]["load_scale"], load_log_std=0.08, dispatch_fraction=0.20, voltage_std=0.008)
    rng = lambda: np.random.default_rng(np.random.SeedSequence([int(seed), int(index), 2]))  # noqa: E731
    assert ghi._scan_operating_points(rng(), dispatch_mode=DISPATCH_MODE_CASE14, **kwargs) == stored
    opf = ghi._scan_operating_points(rng(), dispatch_mode=DISPATCH_MODE_OPF, **kwargs)
    assert [p["bus_load_scales"] for p in opf] == [p["bus_load_scales"] for p in stored]
    assert [p["load_scale"] for p in opf] == [p["load_scale"] for p in stored]
    assert all(p["voltage_setpoints_pu"]["b8"] < 1.061 for p in opf)  # OPF keeps bus 8 at its 1.06 bound


# ------------------------------------------------------------------ generators, tiny seed
@pytest.fixture(scope="module")
def hif_corpora(tmp_path_factory):
    out = tmp_path_factory.mktemp("hif")
    common = ["--n-hif", "2", "--n-no-error", "1", "--seed", "7", "--scans-per-window", "2", "--resistance-units", "ohm",
              "--r-hif-ohm-sweep", "200", "--voltage-stratum", "69kv"]
    flags = {DISPATCH_MODE_CASE14: ["--dispatch-mode", "case14"],
             DISPATCH_MODE_OPF: ["--dispatch-mode", "opf", "--three-phase-noise-pu", "1e-4", "--branch-current-noise-pu", "1e-4"]}
    for mode, extra in flags.items():
        subprocess.run([sys.executable, str(ROOT / "Transmission/generate_measurements_hif_ieee14.py"), "--out", str(out / mode),
                        *common, *extra], check=True, cwd=ROOT, capture_output=True, text=True)
    return out


def test_hif_generator_keeps_labels_load_profiles_and_scada_noise_across_dispatch_modes(hif_corpora):
    case14 = _rows(hif_corpora / DISPATCH_MODE_CASE14 / "samples.jsonl")
    opf = _rows(hif_corpora / DISPATCH_MODE_OPF / "samples.jsonl")
    assert [r["id"] for r in case14] == [r["id"] for r in opf] and len(opf) == 3
    for a, b in zip(case14, opf):
        assert b["dispatch"]["mode"] == DISPATCH_MODE_OPF
        if a["scenario"] == "no_error":
            assert a["z_obs"] == b["z_obs"]  # controls are pypower OPF rows in both modes
            continue
        assert a["label"] == b["label"]  # line, phase, resistance, split ratio
        assert a["op_point"]["load_scale"] == b["op_point"]["load_scale"]
        assert b["window_metadata"]["dispatch_mode"] == DISPATCH_MODE_OPF
        assert b["op_point"] == b["scans"][0]["op_point"] and b["dispatch"] == b["scans"][0]["dispatch"]
        for sa, sb in zip(a["scans"], b["scans"]):
            assert sa["op_point"]["bus_load_scales"] == sb["op_point"]["bus_load_scales"]
            assert set(sb["op_point"]) == set(sa["op_point"])  # canonical schema kept; mode sits beside it
            np.testing.assert_allclose(_noise(sa), _noise(sb), rtol=0, atol=1e-12)  # phasor sigma leaves SCADA draws alone
            assert sb["three_phase_sigma"] == 1e-4 and sb["branch_current_sigma_pu"] == 1e-4
            assert sb["noise_contract"]["channels"]["three_phase_voltages"]["applied_sigma_per_component"] == 1e-4
            assert sb["noise_contract"]["channels"]["three_phase_branch_currents"]["applied_sigma_per_component"] == 1e-4
            assert sb["dispatch"]["mode"] == DISPATCH_MODE_OPF and "slack_pg_kw" in sb["dispatch"]
            assert sb["op_point"]["generator_dispatch_kw"]["b2"] != sa["op_point"]["generator_dispatch_kw"]["b2"]
        # z_true (OpenDSS, fault removed) and z_reference_opf (pypower) are now the same operating point
        assert max(abs(x - y) for x, y in zip(b["z_true"], b["z_reference_opf"])) < 5e-4
        assert b["z_true"][VM_BUS8] < 1.061 and a["z_true"][VM_BUS8] > 1.07
    meta = json.loads((hif_corpora / DISPATCH_MODE_OPF / "meta.json").read_text(encoding="utf-8"))
    assert meta["three_phase_sigma"] == 1e-4 and meta["branch_current_sigma_pu"] == 1e-4
    assert meta["hif"]["generation"]["dispatch_mode"] == DISPATCH_MODE_OPF == meta["hif"]["dispatch"]["mode"]
    assert meta["hif"]["scan_window"]["dispatch_mode"] == DISPATCH_MODE_OPF
    assert meta["hif"]["generation"]["skipped_window_count"] == 0 and meta["hif"]["generation"]["skipped_windows"] == []
    assert meta["hif"]["generation"]["three_phase_noise_pu"] == 1e-4


def test_opf_hif_corpus_replays_exactly_through_the_estimator_simulator(hif_corpora):
    row = next(r for r in _rows(hif_corpora / DISPATCH_MODE_OPF / "samples.jsonl") if r["scenario"] == "high_impedance_fault")
    label = row["label"]
    for scan in row["scans"]:
        simulated = simulate_hif_candidate(candidate_branch_row0=label["branch_row0"], alpha=label["split_ratio"],
                                           phase=label["phase"], r_hif_pu=label["r_hif_pu"], op_point=scan["op_point"],
                                           shunt_convention="ybus")
        assert max(abs(x - y) for x, y in zip(simulated["z"], scan["z_clean"])) <= 1e-9
    subprocess.run([sys.executable, str(ROOT / "scripts/validate_hif_multiscan_dataset.py"),
                    str(hif_corpora / DISPATCH_MODE_OPF / "samples.jsonl"), "--meta",
                    str(hif_corpora / DISPATCH_MODE_OPF / "meta.json"), "--strict-physics"],
                   check=True, cwd=ROOT, capture_output=True, text=True)


def test_unbalance_generator_keeps_labels_and_replays_the_opf_dispatch(tmp_path):
    from IEEE_14_OpenDSS.export_measurement_series import extract_measurement_series

    common = ["--n-imbalance", "2", "--n-no-error", "1", "--seed", "5"]
    for mode in (DISPATCH_MODE_CASE14, DISPATCH_MODE_OPF):
        subprocess.run([sys.executable, str(ROOT / "Transmission/generate_measurements_imbalance.py"), "--out", str(tmp_path / mode),
                        *common, "--dispatch-mode", mode], check=True, cwd=ROOT, capture_output=True, text=True)
    case14 = _rows(tmp_path / DISPATCH_MODE_CASE14 / "samples.jsonl")
    opf = _rows(tmp_path / DISPATCH_MODE_OPF / "samples.jsonl")
    assert [r["id"] for r in case14] == [r["id"] for r in opf] and len(opf) == 3
    repo = str(ROOT / "IEEE_14_OpenDSS")
    gi._compile_ieee14_opendss(repo)
    base = gi._read_base_loads()
    for a, b in zip(case14, opf):
        if a["scenario"] == "no_error":
            assert a["z_obs"] == b["z_obs"] and b["dispatch"]["mode"] == DISPATCH_MODE_OPF
            continue
        assert a["label"]["unbalance_bus"] == b["label"]["unbalance_bus"]
        assert a["label"]["load_split"]["fractions"] == b["label"]["load_split"]["fractions"]
        assert a["op_point"]["load_scale"] == b["op_point"]["load_scale"]
        np.testing.assert_allclose(_noise(a), _noise(b), rtol=0, atol=1e-12)
        assert set(a["op_point"]) == {"load_scale", "target_bus"}
        assert set(b["op_point"]) == {"load_scale", "target_bus", *canonicalize_ieee14_operating_point({})}
        assert b["dispatch"]["mode"] == DISPATCH_MODE_OPF and a["dispatch"]["mode"] == DISPATCH_MODE_CASE14
        assert b["z_true"][VM_BUS8] < 1.061 and a["z_true"][VM_BUS8] > 1.07
        assert max(abs(x - y) for x, y in zip(b["z_true"], b["z_reference_opf"])) < 5e-4
        # exact replay with the stored dispatch, balanced and unbalanced
        split, scale = b["label"]["load_split"], b["op_point"]["load_scale"]
        gi._compile_ieee14_opendss(repo)
        gi._scale_all_loads(base, scale)
        assert gi._apply_operating_point_dispatch(b["op_point"]) is not None
        gi._solve_or_raise()
        assert np.max(np.abs(np.asarray(extract_measurement_series(shunt_convention="ybus")[0]) - np.asarray(b["z_true"]))) <= 1e-9
        gi._compile_ieee14_opendss(repo)
        gi._set_loads_scaled_with_bus_unbalance(base, target_bus=split["bus"], load_scale=scale,
                                                bus_fracs=tuple(split["fractions"][p] for p in "abc"))
        gi._apply_operating_point_dispatch(b["op_point"])
        gi._solve_or_raise()
        assert np.max(np.abs(np.asarray(extract_measurement_series(shunt_convention="ybus")[0]) - np.asarray(b["z_clean"]))) <= 1e-9
        # the scenario generator's balanced replay canonicalizes the row op_point (target_bus ignored) and lands on z_true
        replay = _simulate_base(_resolve_model_dir(None, "case14"), op_point=canonicalize_ieee14_operating_point(b["op_point"]),
                                shunt_convention="ybus")["z"]
        assert np.max(np.abs(np.asarray(replay) - np.asarray(b["z_true"])) / SIGMA) < 1e-2
    gi._compile_ieee14_opendss(repo)
    assert gi._apply_operating_point_dispatch({"load_scale": 1.0, "target_bus": "b9"}) is None  # load-only rows: no-op
    meta = json.loads((tmp_path / DISPATCH_MODE_OPF / "meta.json").read_text(encoding="utf-8"))
    assert meta["imbalance"]["dispatch_mode"] == DISPATCH_MODE_OPF == meta["imbalance"]["dispatch"]["mode"]
    assert meta["imbalance"]["generation"]["skipped_window_count"] == 0
    assert meta["three_phase_sigma"] == 5e-3 and meta["branch_current_sigma_pu"] == 1e-3


# ------------------------------------------------------------------ failure policy
def test_opf_failure_raises_and_never_falls_back(monkeypatch):
    import three_phase_nlm.hif_operating_point as hop

    monkeypatch.setattr(hop, "solve_ieee14_opf", lambda *args, **kwargs: None)
    with pytest.raises(OPFDispatchError, match="did not converge"):
        hop.ieee14_opf_operating_point(1.0)
    with pytest.raises(OPFDispatchError):
        ghi._resolve_scan_operating_points(np.random.default_rng(1), scan_count=2, mode="diverse", event_load_scale=1.0,
                                           load_log_std=0.08, dispatch_fraction=0.2, voltage_std=0.008,
                                           dispatch_mode=DISPATCH_MODE_OPF)
    bad = solve_ieee14_opf(1.0)
    bad["gen"][2, 1] = -0.5  # an OPF output below PMIN is not a dispatch either
    with pytest.raises(OPFDispatchError, match="invalid"):
        operating_point_from_opf_solution(bad, load_scale=1.0)


def test_hif_generator_skips_and_records_windows_whose_opf_fails(tmp_path, monkeypatch):
    def failing(load_scale, bus_load_scales=None):
        raise OPFDispatchError(f"AC-OPF did not converge at load_scale={load_scale:.6f} (simulated)")

    monkeypatch.setattr(ghi, "ieee14_opf_operating_point", failing)
    ghi.generate_dataset(out_dir=str(tmp_path / "hif"), n_hif=2, n_no_error=0, seed=7, load_scale_min=0.8, load_scale_max=1.25,
                         split_min=0.25, split_max=0.75, noise_scale=1.0, keep_scenarios=False, branch_sampling="balanced",
                         scans_per_window=1, resistance_units="ohm", r_hif_ohm_sweep="200", voltage_stratum="69kv",
                         dispatch_mode=DISPATCH_MODE_OPF)
    assert _rows(tmp_path / "hif" / "samples.jsonl") == []  # no row with a substituted case14 dispatch
    meta = json.loads((tmp_path / "hif" / "meta.json").read_text(encoding="utf-8"))
    skipped = meta["hif"]["generation"]["skipped_windows"]
    assert meta["hif"]["generation"]["skipped_window_count"] == 2 == len(skipped)
    assert [s["id"] for s in skipped] == ["ieee14_hif_000000", "ieee14_hif_000001"]
    assert all("did not converge" in s["reason"] for s in skipped)
    # case14 mode does not consult the OPF for the dispatch and still generates
    ghi.generate_dataset(out_dir=str(tmp_path / "case14"), n_hif=1, n_no_error=0, seed=7, load_scale_min=0.8, load_scale_max=1.25,
                         split_min=0.25, split_max=0.75, noise_scale=1.0, keep_scenarios=False, branch_sampling="balanced",
                         scans_per_window=1, resistance_units="ohm", r_hif_ohm_sweep="200", voltage_stratum="69kv",
                         dispatch_mode=DISPATCH_MODE_CASE14)
    rows = _rows(tmp_path / "case14" / "samples.jsonl")
    assert len(rows) == 1 and rows[0]["dispatch"]["mode"] == DISPATCH_MODE_CASE14


def test_unbalance_generator_skips_and_records_windows_whose_opf_fails(tmp_path, monkeypatch):
    def failing(load_scale, bus_load_scales=None):
        raise OPFDispatchError(f"AC-OPF did not converge at load_scale={load_scale:.6f} (simulated)")

    monkeypatch.setattr(gi, "ieee14_opf_operating_point", failing)
    gi.generate_dataset(out_dir=str(tmp_path / "imb"), n_imbalance=2, n_no_error=0, seed=5, load_scale_min=0.8,
                        load_scale_max=1.25, dirichlet_alpha=3.0, dispatch_mode=DISPATCH_MODE_OPF)
    assert _rows(tmp_path / "imb" / "samples.jsonl") == []
    meta = json.loads((tmp_path / "imb" / "meta.json").read_text(encoding="utf-8"))
    skipped = meta["imbalance"]["generation"]["skipped_windows"]
    assert meta["imbalance"]["generation"]["skipped_window_count"] == 2 == len(skipped)
    assert [s["window_index"] for s in skipped] == [0, 1] and all("did not converge" in s["reason"] for s in skipped)


def test_dispatch_mode_is_validated():
    with pytest.raises(ValueError, match="dispatch_mode"):
        ghi._scan_operating_points(np.random.default_rng(1), scan_count=1, mode="diverse", event_load_scale=1.0,
                                   load_log_std=0.08, dispatch_fraction=0.2, voltage_std=0.008, dispatch_mode="economic")
    with pytest.raises(ValueError, match="dispatch_mode"):
        gi.generate_dataset(out_dir="unused", n_imbalance=0, n_no_error=0, seed=1, load_scale_min=0.8, load_scale_max=1.25,
                            dirichlet_alpha=3.0, dispatch_mode="economic")
