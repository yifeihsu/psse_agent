"""Build and independently validate a declared-voltage OpenDSS snapshot."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from psse_env.systems import resolve_system
from three_phase_model.exporter import export_model, load_assumptions, write_json
from three_phase_model.runtime import compile_model, redistribute_load
from three_phase_model.measurements import extract_measurements, write_layout
from three_phase_model.validation import validate_model


def _reference_vector(reference):
    bus, branch, gen = reference["bus"], reference["branch"], reference["gen"]
    base = reference["baseMVA"]
    injection = -(bus[:, 2] + 1j * bus[:, 3]) / base
    bus_rows = {int(row[0]): i for i, row in enumerate(bus)}
    for row in gen:
        if row[7] > 0:
            injection[bus_rows[int(row[0])]] += complex(row[1], row[2]) / base
    return np.concatenate((bus[:, 7], injection.real, injection.imag,
                           branch[:, 13]/base, branch[:, 14]/base,
                           branch[:, 15]/base, branch[:, 16]/base))


def compare_matpower_reference(directory, build, measurements, load_scale):
    """Compare actual DSS telemetry with the separately executed MATPOWER 3p solve."""
    root = Path(directory).resolve(strict=True)
    report = json.loads((root / "reference_report.json").read_text())
    result = json.loads((root / "reference_results.json").read_text())
    if not (report["executed"] and report["passed"] and report["load_scale"] == load_scale):
        raise ValueError("MATPOWER reference must be executed, passing, and at the same load scale")
    if (report["base_mva_3phase"] != build["assumptions"]["base_mva"]
        or report["normalized_base_kv_ll"] != build["assumptions"]["base_kv_ll"]
        or report["frequency_hz"] != build["assumptions"]["frequency_hz"]):
        raise ValueError("MATPOWER reference bases/frequency do not match the exported model")
    source_path = REPO / "mcp_server" / f"{build['registry']['case_id']}.m"
    if report["source_case_sha256"] != hashlib.sha256(source_path.read_bytes()).hexdigest():
        raise ValueError("MATPOWER reference source hash differs from the canonical source")
    ids = [row["external_bus"] for row in build["registry"]["buses"]]
    if result["bus_ids"] != ids:
        raise ValueError("MATPOWER reference bus order differs")
    vm = np.array([row["vln_pu"] for row in measurements["three_phase_voltages"]])
    va = np.array([row["vln_ang_deg"] for row in measurements["three_phase_voltages"]])
    vm_error = float(np.max(np.abs(vm - result["vm3_pu"])))
    angle_error = float(np.max(np.abs(np.rad2deg(np.angle(np.exp(1j*np.deg2rad(va - result["va3_deg"])))))))
    branch_error = 0.0
    for side in ("from", "to"):
        short = "f" if side == "from" else "t"
        for component, unit in (("p", "kw"), ("q", "kvar")):
            actual = np.array([row[f"{component}_{side}_pu"] for row in measurements["branch_powers"]])
            expected = np.sum(np.array(result[f"branch_{component}{short}_{unit}_3p"]), axis=1) / (build["assumptions"]["base_mva"]*1000)
            branch_error = max(branch_error, float(np.max(np.abs(actual-expected))))
    cross = {"scope": "OpenDSS snapshot versus independently executed MATPOWER8.1 three-phase conversion",
             "passed": vm_error < 1e-6 and angle_error < 1e-4 and branch_error < 1e-5,
             "maximum_voltage_magnitude_error_pu": vm_error, "maximum_voltage_angle_error_deg": angle_error,
             "maximum_branch_power_error_pu": branch_error, "original_reference_directory": str(root)}
    destination = Path(build["output_dir"]) / "matpower_reference"
    destination.mkdir()
    for name in ("reference_report.json", "reference_results.json", "case57_balanced_3p.m", "case57_normalized_1p.m"):
        shutil.copyfile(root / name, destination / name)
    write_json(destination / "opendss_cross_reference_report.json", cross)
    return cross


def build_and_validate(output_dir, *, system="case57", assumptions="normalized_diagonal",
                       load_scale=1.0, unbalance_bus=12, unbalance_delta=0.2,
                       matpower_reference_dir=None, voltage_profile=None):
    if not np.isfinite(load_scale) or load_scale <= 0:
        raise ValueError("load_scale must be finite and positive")
    spec = resolve_system(system)
    case = spec.load_case()
    case["bus"][:, 2:4] *= load_scale
    build = export_model(case, output_dir, case_id=spec.case_id,
                         assumptions=load_assumptions(assumptions),
                         **({"voltage_profile": voltage_profile} if voltage_profile is not None else {}),
                         source_provenance={"system": spec.to_manifest(), "load_scale": load_scale})
    out = Path(build["output_dir"])
    dss = compile_model(out / "Master.dss")
    balanced = validate_model(dss, build["reference"], build["registry"], build["assumptions"])
    measurements = extract_measurements(dss, build["registry"], build["assumptions"])
    layout = write_layout(build["registry"], build["assumptions"])
    vector_error = float(np.max(np.abs(np.array(measurements["measurement_vector"])-_reference_vector(build["reference"]))))
    balanced["checks"]["operator_measurement_vector"] = {"passed": vector_error <= 1e-5,
        "max_error_pu": vector_error, "threshold_pu": 1e-5, "channel_count": len(measurements["measurement_vector"])}
    if not balanced["checks"]["operator_measurement_vector"]["passed"]:
        balanced["passed"] = False
        balanced["failed_checks"].append("operator_measurement_vector")
    write_json(out / "validation_report.json", balanced)
    write_json(out / "measurements.json", measurements)
    write_json(out / "measurement_layout.json", layout)
    reports = {"balanced": balanced}
    if balanced["passed"] and matpower_reference_dir is not None:
        reports["matpower_cross_reference"] = compare_matpower_reference(matpower_reference_dir, build, measurements, load_scale)
    # Every supplied balanced reference must pass before disturbance generation.
    if all(report["passed"] for report in reports.values()):
        before = np.array([row["vln_pu_rect"] for row in measurements["three_phase_voltages"]])
        disturbance = redistribute_load(dss, build["registry"], bus=unbalance_bus, delta=unbalance_delta)
        unbalanced = validate_model(dss, build["reference"], build["registry"], build["assumptions"], balanced=False)
        unbalanced_measurements = extract_measurements(dss, build["registry"], build["assumptions"])
        vuf = max(row["vln_sequence_pu"][2] / row["vln_sequence_pu"][1]
                  for row in unbalanced_measurements["three_phase_voltages"])
        unbalanced.update(disturbance=disturbance, maximum_negative_sequence_voltage_ratio=vuf)
        write_json(out / "unbalance_validation_report.json", unbalanced)
        write_json(out / "unbalance_measurements.json", unbalanced_measurements)
        (out / "UnbalanceExample.dss").write_text(
            "! Physical phase-load redistribution; total bus demand unchanged.\nRedirect Master.dss\n"
            + "\n".join(disturbance["commands"]) + "\nSolve\n", encoding="utf-8")
        reports["unbalance"] = unbalanced
        redistribute_load(dss, build["registry"], bus=unbalance_bus, delta=0.0)
        restored = extract_measurements(dss, build["registry"], build["assumptions"])
        after = np.array([row["vln_pu_rect"] for row in restored["three_phase_voltages"]])
        restoration_error = float(np.max(np.abs(after-before)))
        reports["restoration"] = {"passed": restoration_error < 1e-8,
                                   "maximum_voltage_rectangular_error_pu": restoration_error,
                                   "threshold_pu": 1e-8}
        write_json(out / "restoration_report.json", reports["restoration"])
    all_passed = all(report["passed"] for report in reports.values())
    manifest = build["manifest"]
    manifest.update(validation_performed=True, validation_passed=all_passed,
                    validation_files=[str(path.relative_to(out)) for path in out.rglob("*report.json")],
                    runtime={"engine": dss.Basic.Version()},
                    implementation_sha256={str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in [Path(__file__), *sorted((REPO / "three_phase_model").glob("*.py"))]},
                    files_sha256={str(path.relative_to(out)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in sorted(out.rglob("*")) if path.is_file() and path.name != "build_manifest.json"})
    write_json(out / "build_manifest.json", manifest)
    return {"output_dir": str(out), "passed": all_passed, "counts": {key: manifest[key] for key in
        ("external_bus_count", "external_phase_node_count", "physical_branch_count", "line_count", "transformer_count")},
        "reports": reports}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=("case14", "case57"), default="case57")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--assumptions", default="normalized_diagonal")
    parser.add_argument("--voltage-profile", help="Explicit named voltage reconstruction; omission preserves normalized 1-kV behavior")
    parser.add_argument("--load-scale", type=float, default=1.0)
    parser.add_argument("--unbalance-bus", type=int, default=12)
    parser.add_argument("--unbalance-delta", type=float, default=0.2)
    parser.add_argument("--matpower-reference-dir", type=Path)
    args = parser.parse_args(argv)
    result = build_and_validate(args.output_dir, system=args.system, assumptions=args.assumptions,
                               load_scale=args.load_scale, unbalance_bus=args.unbalance_bus,
                               unbalance_delta=args.unbalance_delta, matpower_reference_dir=args.matpower_reference_dir,
                               voltage_profile=args.voltage_profile)
    print(json.dumps({key: value for key, value in result.items() if key != "reports"}, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
