"""Fresh OpenDSS HIF/unbalance experiments, with separately audited telemetry.

This is an engineering validation corpus, not a learned-policy evaluation.
Weak/undetected disturbances and failed solves are retained. Instrument noise
is added only after physical solves. Diagnostic functions receive no labels,
fault-node telemetry, altered device settings, or injection receipts.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from psse_env.providers.matpower import MatpowerDeploymentProviders, _render_matpower_case
from psse_env.systems import resolve_system
from three_phase_model.exporter import export_model, load_assumptions, write_json
from three_phase_model.measurements import extract_measurements
from three_phase_model.runtime import compile_model, redistribute_load
from three_phase_model.validation import validate_model, _element, _yprim


PROFILES = {
    "nominal": {"voltage_sigma_pu": 1e-4, "current_sigma_pu": 1e-3},
    "precision_sensitivity": {"voltage_sigma_pu": 1e-5, "current_sigma_pu": 1e-4},
}
IMPLEMENTATION_PATHS = [
    "scripts/validate_ieee57_disturbances.py", "three_phase_model/disturbances.py",
    "three_phase_model/diagnostics.py", "three_phase_model/measurements.py",
    "three_phase_model/runtime.py", "three_phase_model/exporter.py",
    "three_phase_model/validation.py", "psse_env/providers/matpower.py",
    "tools/lagrangian_port.py", "mcp_server/matpower_server.py", "mcp_server/case57.m",
]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def instrument_observations(measurements, *, seed, profile=None):
    """Allowlist the measured channels; independent SCADA and phasor sensors."""
    seeds = np.random.SeedSequence(seed).spawn(2)
    z = np.asarray(measurements["measurement_vector"], dtype=float)
    nbus = len(measurements["three_phase_voltages"])
    nbranch = len(measurements["three_phase_branch_currents"])
    if nbus == 0 or z.shape != (3 * nbus + 4 * nbranch,) or not np.all(np.isfinite(z)):
        raise ValueError("Expected a finite canonical 3*nbus+4*nbranch measurement vector")
    if profile is not None:
        if set(profile) != {"voltage_sigma_pu", "current_sigma_pu"} or any(
            not np.isfinite(v) or v <= 0 for v in profile.values()
        ):
            raise ValueError("Both phasor noise sigmas must be finite and positive")
    sigma = np.r_[np.full(nbus, .001), np.full(len(z) - nbus, .01)]
    if profile is not None:
        z = z + np.random.default_rng(seeds[0]).normal(0, sigma)
    phase_rng = np.random.default_rng(seeds[1])
    result = {"measurement_vector": z.tolist(), "three_phase_voltages": [],
              "three_phase_branch_currents": []}
    for family, keys, fields, std in (
        ("three_phase_voltages", ("external_bus", "bus", "row0"), ("vln_pu_rect",),
         0 if profile is None else profile["voltage_sigma_pu"]),
        ("three_phase_branch_currents", ("asset_id", "branch_row0", "from_bus", "to_bus"),
         ("i_from_pu_rect", "i_to_pu_rect"), 0 if profile is None else profile["current_sigma_pu"]),
    ):
        for original in measurements[family]:
            row = {key: original[key] for key in keys}
            for field in fields:
                value = np.asarray(original[field], dtype=float)
                if value.shape != (3, 2) or not np.all(np.isfinite(value)):
                    raise ValueError(f"Expected finite ABC real/imaginary phasors for {field}")
                if std:
                    value = value + phase_rng.normal(0, std, size=value.shape)
                row[field] = value.tolist()
            result[family].append(row)
    return result


def make_flat_case(case, directory):
    """Keep nominal parameters while explicitly removing solved-state seeds."""
    flat = deepcopy(case)
    flat["bus"][:, 7] = 1.0
    flat["bus"][:, 8] = 0.0
    path = Path(directory) / "flat_wls_case.m"
    path.write_text(_render_matpower_case(flat, "flat_wls_case"), encoding="utf-8")
    return path


def screen_wls(provider, case_path, observations):
    result = provider.run_wls({"case": str(case_path),
                               "measurements": observations["measurement_vector"]})
    keys = ("converged", "chi_square_statistic", "chi_square_threshold", "chi_square_dof",
            "max_normalized_residual", "normalized_residual_threshold", "chi_square_alarm",
            "normalized_residual_alarm", "anomaly_detection_rule", "error_code", "error_detail")
    result = {key: result.get(key) for key in keys}
    result["alarm"] = (bool(result["chi_square_alarm"] or result["normalized_residual_alarm"])
                       if result["converged"] else None)
    return result


def circuit_physics(dss, registry, assumptions):
    """Offline all-element audit, including hidden nodes and actual device PQ."""
    ibase = assumptions["base_mva"] * 1000 / (np.sqrt(3) * assumptions["base_kv_ll"])
    sbase_kva = assumptions["base_mva"] * 1000
    kcl = defaultdict(complex)
    passive_error = power_error = pq_error = fault_error = 0.0
    total_power = 0j
    passive_count = fault_count = 0
    for name in dss.Circuit.AllElementNames():
        element = _element(dss, name)
        if not element["enabled"]:
            continue
        current, voltage = element["currents"], element["volts"]
        total_power += sum(element["powers_kva"]) / sbase_kva
        power_error = max(power_error, float(np.max(np.abs(
            voltage * current.conj() / (sbase_kva * 1000) - element["powers_kva"] / sbase_kva))))
        for terminal, bus in enumerate(element["buses"]):
            for conductor in range(element["ncond"]):
                i = terminal * element["ncond"] + conductor
                node = int(element["nodes"][i])
                if node:
                    kcl[(bus.split(".")[0].lower(), node)] += current[i] / ibase
        if name.lower().split(".")[0] in {"line", "transformer", "capacitor", "reactor", "fault"}:
            passive_count += 1
            passive_error = max(passive_error, float(np.max(np.abs(
                (_yprim(dss, element) @ voltage - current) / ibase))))
        if name.lower().startswith("fault."):
            fault_count += 1
            dss.Circuit.SetActiveElement(name)
            resistance = float(dss.Properties.Value("R"))
            # This corpus has one selected phase and a grounded second terminal.
            fault_error = max(fault_error, float(abs(current[0] - (voltage[0]-voltage[1])/resistance)/ibase))
    for family, sign in (("loads", 1), ("generators", -1)):
        collection = dss.Loads if family == "loads" else dss.Generators
        for row in registry[family]:
            element = _element(dss, row["element"])
            collection.Name(row["element"].split(".", 1)[1])
            expected = sign * complex(collection.kW(), collection.kvar()) / sbase_kva
            pq_error = max(pq_error, abs(sum(element["powers_kva"]) / sbase_kva - expected))
    checks = {
        "phase_kcl_pu": (max(map(abs, kcl.values()), default=0), 1e-7),
        "passive_current_equation_pu": (passive_error, 1e-8),
        "reported_power_equation_pu": (power_error, 1e-8),
        "network_power_balance_pu": (abs(total_power), 1e-7),
        "constant_pq_device_power_pu": (pq_error, 1e-5),
        "fault_ohms_law_current_pu": (fault_error, 1e-8),
    }
    checked = {name: {"max_error": float(error), "limit": limit,
                      "passed": bool(np.isfinite(error) and error <= limit)}
               for name, (error, limit) in checks.items()}
    return {"passed": bool(dss.Solution.Converged() and all(v["passed"] for v in checked.values())),
            "converged": bool(dss.Solution.Converged()), "checks": checked,
            "phase_node_count": len(kcl), "passive_element_count": passive_count,
            "enabled_fault_count": fault_count}


def compare_snapshot(measurements, baseline):
    before = np.asarray([r["vln_pu_rect"] for r in baseline["three_phase_voltages"]])
    after = np.asarray([r["vln_pu_rect"] for r in measurements["three_phase_voltages"]])
    difference = after - before
    voltage = float(np.max(np.linalg.norm(difference, axis=-1)))
    operator = float(np.max(np.abs(np.asarray(measurements["measurement_vector"])
                                   - baseline["measurement_vector"])))
    currents = max(float(np.max(np.linalg.norm(
        np.asarray([r[f"i_{end}_pu_rect"] for r in measurements["three_phase_branch_currents"]])
        - np.asarray([r[f"i_{end}_pu_rect"] for r in baseline["three_phase_branch_currents"]]), axis=-1)))
                   for end in ("from", "to"))
    return {"passed": voltage <= 1e-6 and operator <= 1e-5 and currents <= 1e-7,
            "max_complex_voltage_error_pu": voltage, "max_operator_channel_error_pu": operator,
            "max_complex_current_error_pu": currents}


def scenario_design(registry, *, smoke=False):
    buses = sorted({row["bus"] for row in registry["loads"]})
    lines = [row for row in registry["branches"] if row["status"] and row["dss_element"].startswith("Line.")]
    if smoke:
        buses = [b for b in (1, 12, 57) if b in buses]
        lines = [lines[i] for i in (0, len(lines)//2, len(lines)-1)]
    result = [{"family": "unbalance", "bus": bus, "delta": delta}
              for bus in buses for delta in (.05, .2)]
    for index, branch in enumerate(lines):
        for phase in (1, 2, 3):
            result.append({"family": "hif", "branch_row0": branch["branch_row0"],
                           "asset_id": branch["asset_id"], "phase": phase,
                           "alpha": (.2, .5, .8)[(index//3 + phase-1) % 3],
                           "resistance_pu": (10., 100., 1000.)[(index + phase-1) % 3]})
    return result


def _save_observations(path, payload):
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(payload, stream, allow_nan=False, separators=(",", ":"))


def localization_correct(row, diagnostic):
    family = row["family"]
    label = diagnostic.get("classification")
    if family == "healthy":
        return label == "no_detectable_anomaly"
    if family == "hif":
        candidate = diagnostic.get("hif_candidate") or {}
        return (label == "hif_like_branch_mismatch" and candidate.get("asset_id") == row["truth"]["asset_id"]
                and candidate.get("phase") == row["truth"]["phase"])
    return (label == "load_unbalance" and
            (diagnostic.get("unbalance_candidate") or {}).get("bus") == row["truth"]["bus"])


def summarize_group(rows):
    count = len(rows)
    result = {"count": count, "physical_roots": len({r["physical_root_fingerprint"] for r in rows}),
              "execution_failures": sum("execution_failure" in r for r in rows),
              "physics_passed": sum(r.get("physics", {}).get("passed", False) for r in rows),
              "wls": {}, "phase": {}}
    for profile in ("exact", "noisy"):
        metrics = [r.get("wls", {}).get(profile, {}) for r in rows]
        result["wls"][profile] = {
            "converged": sum(bool(m.get("converged")) for m in metrics),
            "alarms": sum(m.get("alarm") is True for m in metrics),
            "chi_square_alarms": sum(m.get("chi_square_alarm") is True for m in metrics),
            "normalized_residual_alarms": sum(m.get("normalized_residual_alarm") is True for m in metrics),
            "normalized_residual_only_alarms": sum(m.get("normalized_residual_alarm") is True
                                                   and m.get("chi_square_alarm") is False for m in metrics),
        }
    for profile in ("exact", "noisy_nominal", "noisy_precision_sensitivity"):
        diagnostics = [r.get("phase_diagnostics", {}).get(profile, {}) for r in rows]
        correct = [localization_correct(r, d) for r, d in zip(rows, diagnostics)]
        summary = {"classification_counts": dict(Counter(d.get("classification", "execution_failed") for d in diagnostics)),
                   "alarms": sum(bool(d.get("anomaly_detected")) for d in diagnostics),
                   "correct": sum(correct),
                   "wrong_localization_or_family": sum(d.get("classification") not in
                       (None, "no_detectable_anomaly", "ambiguous") and not ok for d, ok in zip(diagnostics, correct)),
                   "correct_with_wls_alarm": sum(ok and r.get("wls", {}).get("exact" if profile == "exact" else "noisy", {}).get("alarm") is True
                                                   for r, ok in zip(rows, correct) if r["family"] != "healthy"),
                   "correct_but_wls_no_alarm": sum(ok and r.get("wls", {}).get("exact" if profile == "exact" else "noisy", {}).get("alarm") is False
                                                   for r, ok in zip(rows, correct) if r["family"] != "healthy")}
        parameters = [(r, d["hif_candidate"]) for r, d, ok in zip(rows, diagnostics, correct)
                      if ok and r["family"] == "hif" and d["hif_candidate"].get("alpha_estimate") is not None]
        if parameters:
            alpha = [abs(d["alpha_estimate"] - r["truth"]["alpha"]) for r, d in parameters]
            resistance = [abs(d["resistance_pu_estimate"] / r["truth"]["resistance_pu"] - 1) for r, d in parameters]
            summary["hif_parameter_errors_given_correct_branch_and_phase"] = {
                "count": len(parameters), "alpha_absolute_error_median": float(np.median(alpha)),
                "alpha_absolute_error_p95": float(np.quantile(alpha, .95)), "alpha_absolute_error_max": max(alpha),
                "resistance_relative_error_median": float(np.median(resistance)),
                "resistance_relative_error_p95": float(np.quantile(resistance, .95)),
                "distance_observable_linearized_count": sum(d.get("distance_observable", False) for _, d in parameters),
                "not_an_empirically_calibrated_confidence_statement": True,
            }
        result["phase"][profile] = summary
    return result


def summarize_experiment(output, config, receipts):
    rows = [row for receipt in receipts for row in json.loads((output / receipt["results_path"]).read_text())]
    groups = {family: summarize_group([r for r in rows if r["family"] == family])
              for family in ("healthy", "hif", "unbalance")}
    physical_checks = {}
    for key in ("physics", "split_null", "split_null_physics", "split_primitive_audit", "fault_physics",
                "fault_removed", "fault_removed_physics", "restoration", "restoration_physics", "restored_primitive_audit"):
        evidence = [r[key] for r in rows if key in r]
        physical_checks[key] = {"count": len(evidence), "passed": sum(e.get("passed", False) for e in evidence)}
    all_executed = not any("execution_failure" in r for r in rows)
    complete = all_executed and all(g["count"] == g["passed"] for g in physical_checks.values())
    result = {"contract": "ieee57_disturbance_validation_summary_v1", "groups": groups,
              "physical_validation_complete_and_passed": complete, "physical_checks": physical_checks,
              "by_model_and_family": {receipt["model_id"]: {f: summarize_group([r for r in rows if r["model_id"] == receipt["model_id"] and r["family"] == f])
                  for f in ("healthy", "hif", "unbalance")} for receipt in receipts},
              "hif_by_resistance_pu": {str(res): summarize_group([r for r in rows if r["family"] == "hif" and r["truth"]["resistance_pu"] == res])
                                      for res in (10., 100., 1000.)},
              "unbalance_by_delta": {str(delta): summarize_group([r for r in rows if r["family"] == "unbalance" and r["truth"]["delta"] == delta])
                                     for delta in (.05, .2)},
              "scope": config["scope"], "healthy_control_noise_replicates_are_not_new_physical_roots": True}
    failures = [{"scenario_id": r["scenario_id"], "family": r["family"], "truth": r.get("truth"),
                 "execution_failure": r.get("execution_failure"),
                 "wls_noisy": r.get("wls", {}).get("noisy"),
                 "phase_outcomes": {p: {"classification": d.get("classification"),
                     "correct": localization_correct(r, d), "hif_candidate": d.get("hif_candidate"),
                     "unbalance_candidate": d.get("unbalance_candidate")}
                     for p,d in r.get("phase_diagnostics", {}).items()}}
                for r in rows if "execution_failure" in r or any(not localization_correct(r, d)
                    for d in r.get("phase_diagnostics", {}).values())]
    write_json(output / "summary.json", result)
    write_json(output / "non_detections_and_ambiguities.json", failures)
    lines = ["# IEEE57 fresh HIF and unbalance validation", "", config["scope"] + ".", "",
             "The four models cover diagonal/coupled assumptions at 0.8 and 1.0 load. Each was freshly built, solved, and checked against its balanced reference before faults were applied.", "",
             "## Executed coverage", "",
             f"- {groups['hif']['count']} physical line/phase resistive HIF roots; all 63 lines and all three phases per model.",
             f"- {groups['unbalance']['count']} physical phase-load redistribution roots; all 42 load buses, at delta 0.05 and 0.20 per model.",
             f"- {groups['healthy']['count']} independent healthy instrument-noise realizations on {groups['healthy']['physical_roots']} physical healthy roots.",
             "- HIF resistance 10/100/1000 pu (0.1/1/10 ohm on the normalized 1 kV, 100 MVA base); alpha 0.2/0.5/0.8. These values are cycled across assets/phases, not a full factorial.", "",
             "## Numerical circuit checks", "", f"All required physical checks completed and passed: **{complete}**.", "",
             "Every HIF has a paired no-fault split, fault-enabled solve, fault-removal solve and original-line restoration. Full matrix endpoint charging is retained. Checks include all-node KCL, passive I=YV, source/device/network power, constant-PQ behavior, actual resistor I=V/R and P=|V|²/R, hidden-node mapping, and recovery of external voltages/currents and the 491-channel vector.", "",
             "## Detection and localization", "",
             "WLS uses the existing provider, flat VM=1/VA=0, sigma(Vm)=0.001 pu and sigma(P/Q)=0.01 pu. The alarm is chi-square >=424.334166 (378 DOF, alpha=0.05) OR max normalized residual >=4. Both statistics are saved for every scan.", "",
             "Phase screening uses pristine full ABC terminal admittances, nominal phase powers and external voltage/current observations. The detector receives no injection labels, hidden-node channels, or altered device settings. Its fixed threshold is 6 times the propagated per-component noise scale. Nominal phasor sigma(V)=1e-4 pu, sigma(I)=1e-3 pu; the separate precision sensitivity profile reduces both by 10. These are assumed sensor models, not field calibration.", "",
             "| Family | Cases | Noisy WLS alarms | Phase correct, exact telemetry | Phase correct, noisy nominal | Phase correct, precision sensitivity |",
             "|---|---:|---:|---:|---:|---:|"]
    for family,g in groups.items():
        lines.append(f"| {family} | {g['count']} | {g['wls']['noisy']['alarms']} | {g['phase']['exact']['correct']} | {g['phase']['noisy_nominal']['correct']} | {g['phase']['noisy_precision_sensitivity']['correct']} |")
    lines += ["", "Healthy 'correct' means no phase alarm. HIF 'correct' requires the correct branch and phase; unbalance requires the correct bus and family. Exact telemetry is still screened using the nominal sensor weights and fixed thresholds, so weak disturbances may remain undetected.", "",
              "| HIF R (pu) | Cases | Correct, nominal noise | Correct, precision sensitivity |", "|---|---:|---:|---:|"]
    for resistance,g in result["hif_by_resistance_pu"].items():
        lines.append(f"| {resistance} | {g['count']} | {g['phase']['noisy_nominal']['correct']} | {g['phase']['noisy_precision_sensitivity']['correct']} |")
    for family in ("hif", "unbalance"):
        phase = groups[family]["phase"]["noisy_nominal"]
        lines += ["", f"For {family}, {phase['correct_but_wls_no_alarm']} cases localize correctly from nominal-noise phase telemetry while noisy WLS has no alarm; {phase['correct_with_wls_alarm']} have both a WLS alarm and correct phase localization. This is an observability comparison with phase telemetry available, not a completed acquisition-policy evaluation."]
    lines += ["", "## Claim boundaries", "",
              "This tests the review's steady-state resistive HIF surrogate. It does not test nonlinear arcing, harmonic emissions, transient waveforms, sparse or missing PMUs, unknown grounding/sequence parameters, or the old IEEE14 NLM and learned policy on IEEE57. No thresholds were tuned to accept these samples. Non-detections, ambiguities and failed executions remain in the result files. Distance and resistance fits carry first-order uncertainty; correct branch detection does not guarantee precise distance estimation. Source-bus load redistribution is particularly weakly observable because of the stiff source boundary.", "",
              "## Reproduce and inspect", "", "```powershell",
              f"python scripts/validate_ieee57_disturbances.py --output-dir output/ieee57_disturbances_new --preset {config['preset']} --workers 4 --seed {config['seed']}",
              "```", "", "- [Machine-readable summary](summary.json)",
              "- [Retained non-detections and ambiguities](non_detections_and_ambiguities.json)",
              "- [Frozen experiment settings](experiment_config.json)",
              "- Per-model `results.json` contains every physical audit, both WLS statistics, candidate ranks and parameter fits. `observations/*.json.gz` contains truth-free exact/noisy acquisition channels; `scenarios/*.dss` replays each physical disturbance from the generated model.",
              "- OpenDSS component references: [Fault](https://dss-extensions.org/dss-format/Fault.html), [Capacitor matrix units](https://dss-extensions.org/dss-format/Capacitor.html).", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def run_model(job):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        return _run_model(job)


def _run_model(job):
    from three_phase_model.diagnostics import capture_nominal_model, screen_measurements, DiagnosticConfig
    from three_phase_model.disturbances import (
        inject_midspan_hif, set_hif_enabled, restore_midspan_hif, audit_disturbed_circuit,
    )

    start = time.perf_counter()
    output, model_id = Path(job["output_dir"]), job["model_id"]
    directory = output / model_id
    directory.mkdir()
    (directory / "observations").mkdir()
    (directory / "scenarios").mkdir()
    spec = resolve_system("case57")
    case = spec.load_case()
    case["bus"][:, 2:4] *= job["load_scale"]
    build = export_model(case, directory / "model", case_id="case57",
                         assumptions=load_assumptions(job["assumptions"]),
                         source_provenance={"system": spec.to_manifest(), "load_scale": job["load_scale"]})
    registry, assumptions = build["registry"], build["assumptions"]
    master = directory / "model" / "Master.dss"
    dss = compile_model(master)
    baseline_check = validate_model(dss, build["reference"], registry, assumptions)
    write_json(directory / "balanced_validation.json", baseline_check)
    if not baseline_check["passed"]:
        raise RuntimeError(f"{model_id} balanced validation failed; no disturbances generated")
    baseline = extract_measurements(dss, registry, assumptions)
    nominal = capture_nominal_model(dss, registry, assumptions)
    write_json(directory / "nominal_diagnostic_model.json", nominal)
    write_json(directory / "balanced_measurements.json", baseline)
    flat_case = make_flat_case(case, directory)
    provider = MatpowerDeploymentProviders(chi2_alpha=.05, normalized_residual_threshold=4.0)
    configs = {key: DiagnosticConfig(**profile) for key, profile in PROFILES.items()}
    model_seed = [job["seed"], job["model_index"]]
    rows = []

    def fingerprint(truth):
        return hashlib.sha256(json.dumps({"case_hash": registry["base_case_hash"],
            "assumptions": assumptions, "disturbance": truth}, sort_keys=True).encode()).hexdigest()

    def evaluate(measurements, row, number):
        exact = instrument_observations(measurements, seed=model_seed + [number, 0])
        nominal_obs = instrument_observations(measurements, seed=model_seed + [number, 1], profile=PROFILES["nominal"])
        precise = instrument_observations(measurements, seed=model_seed + [number, 1], profile=PROFILES["precision_sensitivity"])
        row["wls"] = {"exact": screen_wls(provider, flat_case, exact),
                      "noisy": screen_wls(provider, flat_case, nominal_obs)}
        row["phase_diagnostics"] = {
            "exact": screen_measurements(exact, nominal, config=configs["nominal"]),
            "noisy_nominal": screen_measurements(nominal_obs, nominal, config=configs["nominal"]),
            "noisy_precision_sensitivity": screen_measurements(precise, nominal, config=configs["precision_sensitivity"]),
        }
        row["maximum_vuf_percent"] = 100 * max(r["vln_sequence_pu"][2] / r["vln_sequence_pu"][1]
                                                 for r in measurements["three_phase_voltages"])
        observation_path = directory / "observations" / f"{row['scenario_id']}.json.gz"
        _save_observations(observation_path, {"exact": exact, "noisy_nominal": nominal_obs,
                                            "noisy_precision_sensitivity": precise})
        row["observations_path"] = str(observation_path.relative_to(output))
        row["observations_sha256"] = sha256(observation_path)

    # Independent instrument realizations share one physical healthy root/model.
    for index in range(job["healthy_controls"]):
        row = {"scenario_id": f"{model_id}_clean_{index:03d}", "family": "healthy",
               "model_id": model_id, "physical_root": f"{model_id}_healthy",
               "physical_root_fingerprint": fingerprint({"family": "healthy"}),
               "noise_replicate": index, "physics": circuit_physics(dss, registry, assumptions)}
        evaluate(baseline, row, 10000 + index)
        rows.append(row)
    design = scenario_design(registry, smoke=job["smoke"])
    write_json(directory / "scenario_design.json", design)
    for index, truth in enumerate(design):
        scenario_id = f"{model_id}_{truth['family']}_{index:04d}"
        row = {"scenario_id": scenario_id, "physical_root": scenario_id,
               "physical_root_fingerprint": fingerprint(truth),
               "model_id": model_id, "family": truth["family"], "truth": truth}
        try:
            dss = compile_model(master)
            if truth["family"] == "unbalance":
                injection = redistribute_load(dss, registry, bus=truth["bus"], delta=truth["delta"])
                measurements = extract_measurements(dss, registry, assumptions)
            else:
                injection = inject_midspan_hif(dss, registry, assumptions,
                    branch_row0=truth["branch_row0"], alpha=truth["alpha"], phase=truth["phase"],
                    resistance_pu=truth["resistance_pu"], enabled=False)
                split = extract_measurements(dss, registry, assumptions, branch_overrides=injection["branch_overrides"])
                row["split_null"] = compare_snapshot(split, baseline)
                row["split_null_physics"] = circuit_physics(dss, registry, assumptions)
                row["split_primitive_audit"] = audit_disturbed_circuit(dss, injection, registry, assumptions)
                row["split_null_diagnostic"] = screen_measurements(
                    instrument_observations(split, seed=model_seed+[index, 0]), nominal, config=configs["nominal"])
                if not all(row[key]["passed"] for key in ("split_null", "split_null_physics", "split_primitive_audit")):
                    raise RuntimeError("No-fault split failed healthy-operating-point or constant-PQ equivalence")
                set_hif_enabled(dss, injection, True)
                measurements = extract_measurements(dss, registry, assumptions, branch_overrides=injection["branch_overrides"])
                row["fault_physics"] = audit_disturbed_circuit(dss, injection, registry, assumptions)
            row["physics"] = circuit_physics(dss, registry, assumptions)
            evaluate(measurements, row, index)
            row["injection"] = deepcopy(injection)
            if truth["family"] == "unbalance":
                redistribute_load(dss, registry, bus=truth["bus"], delta=0)
            else:
                set_hif_enabled(dss, injection, False)
                removed = extract_measurements(dss, registry, assumptions, branch_overrides=injection["branch_overrides"])
                row["fault_removed"] = compare_snapshot(removed, baseline)
                row["fault_removed_physics"] = audit_disturbed_circuit(dss, injection, registry, assumptions)
                restore_midspan_hif(dss, injection)
                row["restored_primitive_audit"] = audit_disturbed_circuit(dss, injection, registry, assumptions)
            restored = extract_measurements(dss, registry, assumptions)
            row["restoration"] = compare_snapshot(restored, baseline)
            row["restoration_physics"] = circuit_physics(dss, registry, assumptions)
            # Standalone replay uses the immutable generated base plus exact edits.
            commands = injection.get("commands", [])
            replay = directory / "scenarios" / f"{scenario_id}.dss"
            replay.write_text('! Fresh IEEE57 fundamental-frequency research scenario\n'
                f'Redirect "{master.as_posix()}"\n' + "\n".join(commands)
                + (f"\nEdit {injection['fault_element']} Enabled=yes" if truth["family"] == "hif" else "")
                + "\nSolve\n", encoding="utf-8")
            row["replay_path"] = str(replay.relative_to(output))
        except Exception as exc:
            row["execution_failure"] = {"type": type(exc).__name__, "message": str(exc),
                                         "traceback": traceback.format_exc()}
        rows.append(row)
        with (directory / "results.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
        if (index+1) % 25 == 0 or index+1 == len(design):
            print(f"{model_id}: {index+1}/{len(design)} physical disturbances", flush=True)
    write_json(directory / "results.json", rows)
    receipt = {"model_id": model_id, "scenario_count": len(design), "row_count": len(rows),
               "elapsed_seconds": time.perf_counter()-start,
               "results_path": str((directory / "results.json").relative_to(output))}
    write_json(directory / "run_receipt.json", receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preset", choices=("smoke", "full"), default="full")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("workers must be between 1 and 4")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = {"contract": "ieee57_fresh_resistive_hif_unbalance_v1", "seed": args.seed,
        "preset": args.preset, "created_utc": datetime.now(timezone.utc).isoformat(),
        "wls": {"chi_square_alpha": .05, "normalized_residual_threshold": 4.,
                "sigma_voltage_pu": .001, "sigma_power_pu": .01, "initialization": "flat_VM1_VA0"},
        "phase_profiles": PROFILES, "phase_residual_threshold_sigmas": 6,
        "load_scales": [.8, 1.0], "unbalance_deltas": [.05, .2],
        "hif_resistance_pu": [10., 100., 1000.], "hif_alpha": [.2, .5, .8],
        "design": "All eligible lines, each phase once per model; resistance and alpha cycled, not a full factorial. All load buses at both deltas.",
        "noise": "Independent Gaussian real/imaginary phasor noise; separate independent SCADA channels. Sensitivity profile reuses standardized noise draws.",
        "scope": "Physical and diagnostic engineering validation; not policy/NLM/DAgger evaluation or arcing/harmonic validation",
        "source_before": {p: sha256(REPO/p) for p in IMPLEMENTATION_PATHS}}
    write_json(output / "experiment_config.json", config)
    for name in config["source_before"]:
        destination = output / "implementation_snapshot" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO/name, destination)
        if sha256(destination) != config["source_before"][name]:
            raise RuntimeError(f"Source changed during the experiment snapshot: {name}")
    jobs = []
    for assumptions in ("normalized_diagonal", "coupled_sensitivity"):
        for scale in (.8, 1.):
            jobs.append({"output_dir": str(output), "assumptions": assumptions, "load_scale": scale,
                         "model_id": f"{assumptions}_{round(scale*100):03d}", "model_index": len(jobs),
                         "seed": args.seed, "smoke": args.preset == "smoke",
                         "healthy_controls": 3 if args.preset == "smoke" else 100})
    receipts = []
    if args.workers == 1:
        receipts = [run_model(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(run_model, job): job for job in jobs}
            for future in as_completed(futures):
                receipts.append(future.result())
    summary = summarize_experiment(output, config, receipts)
    write_json(output / "run_receipt.json", {"models": receipts,
        "source_after": {p: sha256(REPO/p) for p in config["source_before"]},
        "all_sources_unchanged_during_run": all(sha256(REPO/p) == h for p,h in config["source_before"].items())})
    print(json.dumps({"output_dir": str(output), "models_completed": len(receipts)}), flush=True)
    return 0 if summary["physical_validation_complete_and_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
