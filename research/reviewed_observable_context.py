"""Generate ordinary auxiliary sensors from saved reviewed physical scenarios.

This is an offline observation-construction boundary, never a routing oracle.
Generation parameters and clean replay checks stay in the separate receipt.
The returned metadata has identical sensor availability for every family and
contains only noisy external phasors and their declared uncertainties. Existing
SCADA observations/means are never overwritten, corrected, or re-noised here.
"""
from __future__ import annotations

import copy
import json
from numbers import Integral
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.gnn_screen.dataset import content_hash, file_sha256
from three_phase_nlm.branch_current_analysis import add_branch_current_noise
from three_phase_nlm.measurement_noise import add_voltage_phasor_noise, generated_noise_contract


VOLTAGE_COMPONENT_SIGMA = 0.005
CURRENT_COMPONENT_SIGMA = 0.001
SCADA_REPLAY_TOLERANCE = 1e-8


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _paths(manifest_path: Path, row: Mapping[str, Any]) -> tuple[Path, Path, Path, dict]:
    audit_reference = (row.get("offline_metadata") or {}).get("physical_audit_path")
    if not isinstance(audit_reference, str) or not audit_reference:
        raise ValueError("reviewed row requires its saved physical_audit_path")
    audit_path = (manifest_path.parent / audit_reference.replace("\\", "/")).resolve(strict=True)
    parent_id = str(row.get("parent_id") or "")
    parent_directory = next((p for p in audit_path.parents if p.name == parent_id and p.parent.name == "parents"), None)
    if parent_directory is None:
        raise ValueError("physical audit must belong to the row's original core parent directory")
    core = parent_directory.parent.parent
    audit = _json(audit_path)
    model_reference = audit.get("actual_physical_model_path")
    model = ((core / str(model_reference).replace("\\", "/")).resolve(strict=True)
             if model_reference else parent_directory / "model")
    if not model.is_relative_to(core):
        raise ValueError("saved actual physical model must remain within the original core")
    for name in ("Master.dss", "asset_registry.json", "assumptions.json"):
        if not (model / name).is_file():
            raise ValueError(f"saved physical model is missing {name}: {model}")
    return core, audit_path, model, audit


def _measurement_free_mean(row: Mapping[str, Any], audit: Mapping[str, Any], core: Path) -> tuple[np.ndarray, dict]:
    if row.get("measurement_kind") != "noiseless_mean":
        raise ValueError("pass the original noiseless_mean row; never infer a clean mean from noisy SCADA")
    mean = np.array(row["z"], dtype=float, copy=True)
    if mean.ndim != 1 or not np.isfinite(mean).all():
        raise ValueError("the original SCADA mean must be a finite vector")
    offline = row.get("offline_metadata") or {}
    settings = offline.get("settings") or {}
    corruption = audit.get("measurement_corruption")
    if corruption is None:
        corruption = settings.get("measurement")
    if not corruption:
        return mean, {"indices0": [], "offsets_pu": [], "sigma_reference": None}
    if not isinstance(corruption, Mapping):
        raise ValueError("saved measurement corruption must be an object")
    indices = corruption.get("channel_indices0")
    if (not isinstance(indices, list) or not indices or any(
        isinstance(i, bool) or not isinstance(i, Integral) or not 0 <= int(i) < mean.size for i in indices
    ) or len(set(indices)) != len(indices)):
        raise ValueError("saved measurement offsets require unique valid channel indices")
    indices = [int(i) for i in indices]
    offsets = corruption.get("additive_offsets_pu")
    sigma_reference = None
    if offsets is None:
        multiples = np.asarray(corruption.get("sigma_multiples"), dtype=float)
        # Accuracy views freeze the injected physical means. Multiples refer
        # to the ORIGINAL core sensor scale, not the view's smaller sigma.
        sigma_path = core / "measurement_sigma.json"
        baseline_sigma = np.asarray(json.loads(sigma_path.read_text()), dtype=float)
        if baseline_sigma.shape != mean.shape or not np.isfinite(baseline_sigma).all() or np.any(baseline_sigma <= 0):
            raise ValueError("original core measurement_sigma does not match the stored mean")
        if multiples.shape != (len(indices),) or not np.isfinite(multiples).all():
            raise ValueError("saved sigma multiples must match measurement indices")
        offsets = multiples * baseline_sigma[indices]
        sigma_reference = {"path": str(sigma_path), "sha256": file_sha256(sigma_path),
                           "scope": "original core injection covariance; accuracy views do not rescale gross errors"}
    offsets = np.asarray(offsets, dtype=float)
    if offsets.shape != (len(indices),) or not np.isfinite(offsets).all():
        raise ValueError("saved additive offsets must match measurement indices")
    mean[indices] -= offsets
    return mean, {"indices0": indices, "offsets_pu": offsets.tolist(), "sigma_reference": sigma_reference}


def _generation_parameters(row: Mapping[str, Any], audit: Mapping[str, Any]) -> tuple[dict | None, dict | None]:
    settings = (row.get("offline_metadata") or {}).get("settings") or {}
    disturbance = audit.get("disturbance") or {}
    if not isinstance(settings, Mapping) or not isinstance(disturbance, Mapping):
        raise ValueError("saved physical settings must be objects")
    hif = settings.get("hif")
    unbalance = settings.get("unbalance")
    if disturbance.get("kind") == "hif":
        hif = disturbance.get("settings")
    elif disturbance.get("kind") == "unbalance":
        unbalance = {key: disturbance[key] for key in ("bus", "fractions")}
    elif disturbance.get("kind") not in {None, ""}:
        raise ValueError("unsupported saved physical disturbance")
    if hif is not None:
        if not isinstance(hif, Mapping):
            raise ValueError("HIF generation parameters must be an object")
        hif = {key: copy.deepcopy(hif[key]) for key in
               ("branch_row0", "alpha", "phase", "resistance_pu", "resistance_ohm", "enabled") if key in hif}
    if unbalance is not None:
        if not isinstance(unbalance, Mapping):
            raise ValueError("unbalance generation parameters must be an object")
        unbalance = {key: copy.deepcopy(unbalance[key]) for key in ("bus", "fractions")}
    return hif, unbalance


def _replay(model: Path, hif: dict | None, unbalance: dict | None) -> tuple[dict, dict]:
    from three_phase_model.disturbances import inject_midspan_hif, audit_disturbed_circuit
    from three_phase_model.measurements import extract_measurements
    from three_phase_model.runtime import compile_model, solve

    registry, assumptions = _json(model / "asset_registry.json"), _json(model / "assumptions.json")
    dss = compile_model(model / "Master.dss")
    checks: dict[str, Any] = {}
    if unbalance is not None:
        fractions = np.asarray(unbalance["fractions"], dtype=float)
        if (fractions.shape != (3,) or not np.isfinite(fractions).all() or np.any(fractions <= 0)
                or not np.isclose(fractions.sum(), 1.0, rtol=0, atol=1e-12)):
            raise ValueError("three positive phase fractions must sum to one")
        rows = [row for row in registry["loads"] if row["bus"] == unbalance["bus"]]
        if len(rows) != 3 or {row["phase"] for row in rows} != {1, 2, 3}:
            raise ValueError("unbalance requires three saved phase-load elements")
        before = np.sum([[row["kw"], row["kvar"]] for row in rows], axis=0)
        after = np.zeros(2)
        for row in rows:
            factor = float(3 * fractions[row["phase"] - 1])
            dss.Text.Command(f"Edit {row['element']} kW={row['kw'] * factor:.16g} kvar={row['kvar'] * factor:.16g}")
            after += np.asarray([row["kw"], row["kvar"]]) * factor
        if not np.allclose(before, after, atol=1e-8, rtol=1e-12):
            raise ValueError("unbalance replay would change total P/Q")
        solve(dss)
        checks["load_totals_before_kw_kvar"] = before.tolist()
        checks["load_totals_after_kw_kvar"] = after.tolist()
    injection = None
    if hif is not None:
        injection = inject_midspan_hif(dss, registry, assumptions, **hif)
        check = audit_disturbed_circuit(dss, injection, registry, assumptions)
        if not check["passed"]:
            raise ValueError(f"physical HIF replay audit failed: {check['failed_checks']}")
        checks["hif_physical_audit_passed"] = True
    telemetry = extract_measurements(dss, registry, assumptions,
        branch_overrides=injection["branch_overrides"] if injection else None)
    kcl = float(telemetry["max_kcl_mismatch_pu"])
    if not np.isfinite(kcl) or kcl > 1e-7:
        raise ValueError("replayed external phase KCL is inconsistent")
    checks.update(converged=bool(dss.Solution.Converged()), external_kcl_max_mismatch_pu=kcl,
                  external_bus_count=len(registry["buses"]), external_branch_count=len(registry["branches"]))
    return telemetry, checks


def build_observable_context(manifest_path, row: Mapping[str, Any], *, noise_seed: int) -> tuple[dict, dict]:
    """Return noisy phase sensors and a separate, private replay receipt.

    ``row`` must be an original row returned by ``load_manifest`` before its
    SCADA noise materialization. Callers attach the metadata to their existing
    noisy SCADA snapshot; no SCADA vector is returned in metadata. Noise seeds
    must be chosen independently of diagnostic outcomes. Every supported row,
    including healthy, measurement, parameter and topology cases, acquires the
    same external voltage/current sensor types.
    """
    if isinstance(noise_seed, bool) or not isinstance(noise_seed, Integral) or noise_seed < 0:
        raise ValueError("noise_seed must be a nonnegative integer")
    manifest_path = Path(manifest_path).resolve(strict=True)
    before_row_hash = content_hash(row)
    core, audit_path, model, audit = _paths(manifest_path, row)
    expected_mean, removed_offsets = _measurement_free_mean(row, audit, core)
    sigma = np.asarray(row.get("measurement_sigma"), dtype=float)
    if sigma.shape != expected_mean.shape or not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise ValueError("loaded row measurement_sigma must match its positive SCADA covariance")
    hif, unbalance = _generation_parameters(row, audit)
    files = sorted(path for path in model.iterdir() if path.is_file() and
                   (path.suffix.lower() == ".dss" or path.name in {"asset_registry.json", "assumptions.json"}))
    before_files = {path.name: file_sha256(path) for path in files}
    telemetry, physics = _replay(model, hif, unbalance)
    replayed = np.asarray(telemetry["measurement_vector"], dtype=float)
    if replayed.shape != expected_mean.shape or not np.isfinite(replayed).all():
        raise ValueError("replayed SCADA layout differs from the source mean")
    maximum_error = float(np.max(np.abs(replayed - expected_mean)))
    if maximum_error > SCADA_REPLAY_TOLERANCE:
        raise ValueError(f"replayed clean SCADA differs from source mean after known meter offsets: {maximum_error:.12g}")

    # The general extractor also returns noiseless rectangular and sequence
    # aliases. Whitelist BEFORE perturbation so none survive as a clean copy.
    voltages = [{key: copy.deepcopy(item[key]) for key in
                 ("bus", "external_bus", "row0", "kvbase_ln", "vln_pu", "ang_deg")}
                for item in telemetry["three_phase_voltages"]]
    currents = [{key: copy.deepcopy(item[key]) for key in
                 ("asset_id", "branch", "branch_row0", "from_bus", "to_bus",
                  "i_from_pu", "ang_from_deg", "i_to_pu", "ang_to_deg", "ibase_from_a", "ibase_to_a")}
                for item in telemetry["three_phase_branch_currents"]]
    voltage_seed, current_seed = np.random.SeedSequence(int(noise_seed)).spawn(2)
    noisy_voltages = add_voltage_phasor_noise(voltages, np.random.default_rng(voltage_seed), VOLTAGE_COMPONENT_SIGMA)
    noisy_currents = add_branch_current_noise(currents, np.random.default_rng(current_seed), CURRENT_COMPONENT_SIGMA)
    contract = generated_noise_contract(sigma.tolist(), noise_scale=1.0,
        three_phase_sigma=VOLTAGE_COMPONENT_SIGMA, branch_current_sigma_pu=CURRENT_COMPONENT_SIGMA)
    contract["channels"].pop("scada")
    contract["generation_scope"] = "auxiliary_phase_sensors_only"
    contract["scada_noise_drawn_here"] = False
    metadata = {"three_phase_voltages": noisy_voltages, "three_phase_sigma": VOLTAGE_COMPONENT_SIGMA,
                "three_phase_branch_currents": noisy_currents, "branch_current_sigma_pu": CURRENT_COMPONENT_SIGMA,
                "sigma_z": sigma.tolist(), "noise_contract": contract}
    after_files = {path.name: file_sha256(path) for path in files}
    if before_files != after_files or content_hash(row) != before_row_hash:
        raise RuntimeError("observation construction modified its source row or saved physical model")
    receipt = {"schema": "reviewed_auxiliary_observation_replay_v1", "offline_only": True,
        "manifest_path": str(manifest_path), "physical_audit_path": str(audit_path),
        "original_core_path": str(core), "physical_model_path": str(model),
        "source_window_id": row.get("window_id"), "noise_seed": int(noise_seed),
        "generation_parameters": {"hif": hif, "unbalance": unbalance},
        "removed_measurement_offsets": removed_offsets, "physics": physics,
        "replayed_clean_scada": replayed.tolist(), "reference_mean_sha256": content_hash(expected_mean),
        "maximum_clean_scada_error": maximum_error, "clean_scada_tolerance": SCADA_REPLAY_TOLERANCE,
        "model_files_sha256": before_files, "source_row_sha256": before_row_hash,
        "source_unchanged": True, "metadata_sha256": content_hash(metadata),
        "scada_snapshot_modified": False, "routing_used_fault_labels": False,
        "sensor_noise_independence": "separate spawned voltage/current RNG streams; caller SCADA draw preserved"}
    return metadata, receipt


__all__ = ["build_observable_context"]
