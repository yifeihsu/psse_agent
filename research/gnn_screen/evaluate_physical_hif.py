"""Frozen GNN transfer on an unfiltered, saved physical HIF/WLS sweep.

Replays saved noisy vectors exactly. Never calibrates or trains on sweep data.
The legacy-type sensitivity changes graph flags only, not the physical cases.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

for _variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np
from scipy.stats import chi2
import torch
from threadpoolctl import threadpool_limits

from .feature_schema import FAMILY_NAMES
from .graph_builder import build_graph
from .model import collate_graphs
from .protocol_adapter import FrozenScreen
from .train import load_trained_model

REPO = Path(__file__).resolve().parents[2]
SOURCES = ["research/gnn_screen/" + name for name in (
    "evaluate_physical_hif.py", "graph_builder.py", "wls_features.py", "feature_schema.py",
    "model.py", "train.py", "protocol_adapter.py")] + ["tools/lagrangian_port.py"]
DETECTORS = ("gnn_phase", "gnn_anomaly", "gnn_hif_family", "wls_chi_square",
             "wls_normalized", "wls_dual", "gnn_or_wls", "wls_source_calibrated",
             "legacy_type_gnn_phase")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numeric_hash(values):
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def reconstruct_observation(row, physical_records, noise_groups, sigma_by_profile):
    physical = physical_records[row["case_id"]]
    if not physical["physical_success"]:
        raise ValueError("Cannot score a failed physical case")
    unit = np.asarray(noise_groups[row["noise_group_id"]]["unit_noise"], dtype=float)
    sigma = np.asarray(sigma_by_profile[row["noise_profile"]], dtype=float)
    observed = np.asarray(physical["mean_measurement_vector"], dtype=float) + unit * sigma
    if numeric_hash(observed) != row["observed_sha256"]:
        raise ValueError(f"Noisy measurement reconstruction mismatch: {row['case_id']}")
    return observed, sigma


def legacy_type_graph(graph, case):
    """Diagnostic ablation: restore old native tap/shift-only type flags."""
    result = deepcopy(graph)
    branch = np.asarray(case["branch"])
    transformer = (branch[:, 8] != 0) | (branch[:, 9] != 0)
    result["edge_attr"][:, 35] = np.repeat(~transformer, 2)
    result["edge_attr"][:, 36] = np.repeat(transformer, 2)
    return result


def saved_wls_fields(saved, calibration, local_threshold):
    """Keep the verified source WLS comparator available if graph building fails."""
    names = ("wls_chi_square", "wls_normalized", "wls_dual", "wls_source_calibrated")
    if not saved["success"]:
        return dict.fromkeys(names)
    return {"wls_chi_square": bool(saved["chi_square_alarm"]),
            "wls_normalized": bool(saved["normalized_residual_alarm"]),
            "wls_dual": bool(saved["alarm"]),
            "wls_source_calibrated": bool(max(saved["J"] / saved["chi_square_threshold"],
                saved["max_normalized_residual"] / local_threshold) > calibration["matched_wls_threshold"])}


def attach_paired_controls(rows):
    controls = {(r["noise_group_id"], r["noise_profile"]): r for r in rows if r["kind"] == "healthy"}
    if len(controls) != sum(r["kind"] == "healthy" for r in rows):
        raise ValueError("Duplicate healthy control pairing key")
    for row in rows:
        if row["kind"] == "hif":
            control = controls[(row["noise_group_id"], row["noise_profile"])]
            for detector in DETECTORS:
                fault, healthy = row[detector], control[detector]
                row[f"{detector}_new_vs_healthy"] = (bool(fault and not healthy)
                    if fault is not None and healthy is not None else None)
                row[f"{detector}_lost_vs_healthy"] = (bool(healthy and not fault)
                    if fault is not None and healthy is not None else None)


def summarize(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in keys)].append(row)
    result = []
    for values, members in sorted(groups.items(), key=lambda item: str(item[0])):
        item = dict(zip(keys, values))
        item.update(observations=len(members), physical_cases=len({r["case_id"] for r in members}),
                    operating_parents=len({r["parent_id"] for r in members}),
                    noise_groups=len({r["noise_group_id"] for r in members}),
                    gnn_unavailable=sum(r["gnn_phase"] is None for r in members))
        for detector in DETECTORS:
            available = [r for r in members if r[detector] is not None]
            count = sum(r[detector] for r in available)
            item[detector + "_available"] = len(available)
            item[detector + "_count"] = count
            item[detector + "_rate"] = count / len(available) if available else None
            if members[0]["kind"] == "hif":
                paired = [r for r in members if r.get(detector + "_new_vs_healthy") is not None]
                item[detector + "_new_vs_healthy_count"] = sum(r[detector + "_new_vs_healthy"] for r in paired)
                item[detector + "_lost_vs_healthy_count"] = sum(r.get(detector + "_lost_vs_healthy", False) for r in paired)
                item[detector + "_paired_available"] = len(paired)
        both = [r for r in members if r["gnn_phase"] is not None and r["wls_dual"] is not None]
        misses = [r for r in both if not r["wls_dual"]]
        item["gnn_additional_to_wls"] = sum(r["gnn_phase"] for r in misses)
        item["wls_misses_with_gnn_available"] = len(misses)
        item["gnn_phase_score_median"] = float(np.median([r["phase_score"] for r in both])) if both else None
        item["legacy_type_decision_flips"] = sum(r["gnn_phase"] != r["legacy_type_gnn_phase"] for r in members
            if r["gnn_phase"] is not None and r["legacy_type_gnn_phase"] is not None)
        result.append(item)
    return result


def run(sweep, checkpoint_path, calibration_path, output, *, batch_size=64, device="cpu"):
    started = time.perf_counter()
    sweep, output = Path(sweep).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Fresh output directory required: {output}")
    config = json.loads((sweep / "experiment_config.json").read_text(encoding="utf-8"))
    source_summary = json.loads((sweep / "summary.json").read_text(encoding="utf-8"))
    if config.get("selection") != "none; retain quiet and failed cases":
        raise ValueError("Requires unfiltered physical sweep, not training-admitted rows")
    calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
    if calibration["checkpoint_sha256"] != sha256(checkpoint_path):
        raise ValueError("Calibration is not bound to this checkpoint")
    comparator = calibration["wls_comparator"]
    if (comparator["chi2_alpha"] != config["wls"]["chi_square_alpha"]
        or comparator["normalized_residual_threshold"] != config["wls"]["normalized_residual_threshold"]):
        raise ValueError("Source-calibrated WLS requires the same comparator normalization")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    torch.set_num_threads(1)
    model, scaler, checkpoint = load_trained_model(checkpoint_path, device=device)
    FrozenScreen(model, scaler, checkpoint, calibration)  # Validate frozen head/threshold contract.
    output.mkdir(parents=True)
    input_files = [sweep / name for name in ("experiment_config.json", "summary.json", "cases.jsonl",
        "controls.jsonl", "noise_groups.jsonl", "wls_observations.jsonl")]
    input_files += sorted((sweep / "parents").glob("*/positive_sequence_reference.json"))
    source_hashes = {name: sha256(REPO / name) for name in SOURCES}
    plan = {"created_utc": datetime.now(timezone.utc).isoformat(), "sweep": str(sweep),
        "checkpoint": str(Path(checkpoint_path).resolve()), "checkpoint_sha256": sha256(checkpoint_path),
        "calibration": str(Path(calibration_path).resolve()), "calibration_sha256": sha256(calibration_path),
        "model_id": checkpoint["model_id"], "weights_scaler_thresholds_frozen": True,
        "source_input_sha256": {str(path.relative_to(sweep)): sha256(path) for path in input_files},
        "implementation_sha256": source_hashes, "device": device, "batch_size": batch_size,
        "primary_graph": "current physical-voltage-aware graph; 7-8 is transformer",
        "sensitivity_graph": "same observations and graph except native tap/shift-only type flags",
        "calibration_policy": "original V2 source calibration, no target or test recalibration",
        "population": "all saved observations; no D, WLS, GNN or training-admission filter",
        "uncertainty": "descriptive paired counts only; two operating parents cannot support population intervals"}
    write_json(output / "run_plan.json", plan)
    shutil.copyfile(calibration_path, output / "frozen_calibration.json")
    for name in SOURCES:
        target = output / "implementation_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / name, target)
    physical_rows = read_jsonl(sweep / "cases.jsonl") + read_jsonl(sweep / "controls.jsonl")
    physical = {r["case_id"]: r for r in physical_rows}
    noise_rows = read_jsonl(sweep / "noise_groups.jsonl")
    noise = {r["noise_group_id"]: r for r in noise_rows}
    if len(noise) != len(noise_rows):
        raise ValueError("Duplicate noise group IDs")
    if len(physical) != len(physical_rows):
        raise ValueError("Duplicate physical case IDs")
    for row in noise.values():
        if numeric_hash(row["unit_noise"]) != row["unit_noise_sha256"]:
            raise ValueError("Corrupt saved unit noise")
    references = {path.parent.name: json.loads(path.read_text(encoding="utf-8"))
                  for path in input_files if path.name == "positive_sequence_reference.json"}
    source_rows = read_jsonl(sweep / "wls_observations.jsonl")
    rows, pending_graphs, pending_legacy, pending_rows = [], [], [], []
    audit = {"reconstructed_hash_matches": 0, "wls_comparisons": 0, "max_J_difference": 0.,
        "max_normalized_residual_difference": 0., "wls_alarm_mismatches": 0, "unavailable": []}
    alpha = config["wls"]["chi_square_alpha"]
    local_threshold = config["wls"]["normalized_residual_threshold"]

    def flush(handle):
        if not pending_graphs:
            return
        with torch.inference_mode():
            logits = model(collate_graphs(pending_graphs, device=device))
            scores = {key: torch.sigmoid(value).cpu().numpy() for key, value in logits.items()}
            legacy = torch.sigmoid(model(collate_graphs(pending_legacy, device=device))["phase_screen_logit"]).cpu().numpy().ravel()
        if any(not np.all(np.isfinite(value)) for value in scores.values()) or not np.all(np.isfinite(legacy)):
            raise ValueError("Nonfinite neural output")
        for i, row in enumerate(pending_rows):
            phase = float(scores["phase_screen_logit"].ravel()[i])
            anomaly = float(scores["anomaly_logit"].ravel()[i])
            families = {name: float(scores["family_logits"][i, j]) for j, name in enumerate(FAMILY_NAMES)}
            row.update(phase_score=phase, anomaly_score=anomaly, family_scores=families,
                gnn_phase=phase > calibration["phase_threshold"],
                gnn_anomaly=anomaly > calibration["anomaly_threshold"],
                gnn_hif_family=families["hif"] > calibration["family_thresholds"]["hif"],
                legacy_type_phase_score=float(legacy[i]), legacy_type_gnn_phase=bool(legacy[i] > calibration["phase_threshold"]))
            row["gnn_or_wls"] = bool(row["gnn_phase"] or row["wls_dual"]) if row["wls_dual"] is not None else None
            handle.write(json.dumps(row, allow_nan=False) + "\n")
        handle.flush()
        pending_graphs.clear(); pending_legacy.clear(); pending_rows.clear()

    with threadpool_limits(limits=1), (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for index, source in enumerate(source_rows):
            observed, sigma = reconstruct_observation(source, physical, noise, config["noise"]["sigma_z"])
            audit["reconstructed_hash_matches"] += 1
            row = {key: value for key, value in source.items() if key != "wls"}
            row.update({name: None for name in DETECTORS})
            row["saved_wls"] = source["wls"]
            row.update(saved_wls_fields(source["wls"], calibration, local_threshold))
            for name in ("resistance_pu", "fault_current_a", "fault_power_mw", "from_bus", "to_bus"):
                row[name] = physical[row["case_id"]].get(name)
            rows.append(row)
            case = references[row["parent_id"]]
            try:
                graph = build_graph(case, observed, measurement_sigma=sigma, scaler=scaler, max_it=30, tol=1e-8)
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
                audit["unavailable"].append({"case_id": row["case_id"], "error": row["error"]})
                handle.write(json.dumps(row, allow_nan=False) + "\n")
                continue
            objective, local = graph["metadata"]["wls_objective"], float(graph["u"][1])
            threshold = float(chi2.ppf(1 - alpha, graph["metadata"]["dof"]))
            row.update(wls_J=objective, wls_max_normalized_residual=local, wls_chi_square_threshold=threshold,
                wls_dof=graph["metadata"]["dof"], wls_chi_square=objective >= threshold,
                wls_normalized=local >= local_threshold)
            row["wls_dual"] = bool(row["wls_chi_square"] or row["wls_normalized"])
            row["wls_source_calibrated"] = bool(max(objective / threshold, local / local_threshold) > calibration["matched_wls_threshold"])
            old = source["wls"]
            if old["success"]:
                audit["wls_comparisons"] += 1
                audit["max_J_difference"] = max(audit["max_J_difference"], abs(objective - old["J"]))
                audit["max_normalized_residual_difference"] = max(audit["max_normalized_residual_difference"], abs(local - old["max_normalized_residual"]))
                audit["wls_alarm_mismatches"] += row["wls_dual"] != old["alarm"]
                if not np.allclose([objective, local], [old["J"], old["max_normalized_residual"]], rtol=1e-8, atol=1e-8) or row["wls_dual"] != old["alarm"]:
                    raise ValueError(f"WLS replay mismatch: {row['case_id']}")
            row["configured_model_hash"] = graph["metadata"]["configured_model_hash"]
            pending_graphs.append(graph)
            pending_legacy.append(legacy_type_graph(graph, case))
            pending_rows.append(row)
            if len(pending_graphs) >= batch_size:
                flush(handle)
            if (index + 1) % 256 == 0:
                print(f"replayed {index+1}/{len(source_rows)} observations in {time.perf_counter()-started:.1f}s", flush=True)
        flush(handle)
    attach_paired_controls(rows)
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")
    groups = {
        "by_voltage_resistance_noise": ["kind", "local_base_kv_ll", "resistance_ohm", "noise_profile"],
        "by_phase": ["kind", "local_base_kv_ll", "resistance_ohm", "noise_profile", "phase"],
        "by_parent": ["kind", "local_base_kv_ll", "resistance_ohm", "noise_profile", "parent_id"],
        "controls": ["kind", "noise_profile"]}
    tables = {name: summarize(rows if name != "controls" else [r for r in rows if r["kind"] != "hif"], keys)
              for name, keys in groups.items()}
    for name, table in tables.items():
        write_csv(output / (name + ".csv"), table)
    audit["implementation_unchanged_during_run"] = all(sha256(REPO / name) == digest for name, digest in source_hashes.items())
    audit["input_files_unchanged_during_run"] = all(sha256(sweep / name) == digest for name, digest in plan["source_input_sha256"].items())
    summary = {"plan": plan, "source_physical_coverage": {key: source_summary[key] for key in
        ("expected_physical_hif_cases", "physical_hif_succeeded", "physical_hif_failed", "operating_parent_count")},
        "observations": len(rows), "audit": audit, "tables": tables,
        "thresholds": {key: calibration[key] for key in ("phase_threshold", "anomaly_threshold", "family_thresholds", "matched_wls_threshold")},
        "wls_rule": {"chi_square_alpha": alpha, "max_abs_normalized_residual_threshold": local_threshold,
                     "comparison": ">=", "combination": "OR"},
        "elapsed_seconds": time.perf_counter()-started}
    write_json(output / "summary.json", summary)
    print(json.dumps({"output": str(output), "observations": len(rows), "audit": audit,
                      "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)
    run(args.sweep, args.checkpoint, args.calibration, args.output_dir, batch_size=args.batch_size, device=args.device)


if __name__ == "__main__":
    main()
