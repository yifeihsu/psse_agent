"""Observable HIF conditioning for the ordinary transactional meter workflow.

Subtract a paired forward-model effect, never zero a residual or exclude a
sensor. The physical observations remain in the state store. Prediction spread
is a sensitivity check, not an estimated covariance or a calibrated interval.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from IEEE_14_OpenDSS.measurement_convention import resolve_shunt_convention
from three_phase_nlm.conditioned_meter_recovery import diagnose_conditioned_meter_errors
from three_phase_nlm.hif_conditioned_recovery import replay_hif_measurement_effect


def current_scan(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Bind old frozen windows by auxiliary acquisition identity, never SCADA.

    SCADA can have been corrupted or repaired since acquisition. A matching
    original SCADA copy must therefore never select the operating point.
    """
    runtime = metadata.get("hif_runtime") or {}
    scans = (metadata.get("hif_scan_window") or {}).get("scans") or []
    matches = []
    seen_indices: set[int] = set()
    for scan in scans:
        if not isinstance(scan, Mapping):
            raise ValueError("HIF scan metadata must contain observable scan objects")
        index = scan.get("scan_index")
        if type(index) is not int or index < 0 or index in seen_indices:
            raise ValueError("HIF scans require unique nonnegative integer scan_index values")
        seen_indices.add(index)
        if runtime.get("scan_index") is not None and (type(runtime["scan_index"]) is not int or index != runtime["scan_index"]):
            continue
        channels = ("three_phase_voltages", "three_phase_branch_currents")
        shared = [key for key in channels if runtime.get(key)]
        if shared and all(scan.get(key) == runtime[key] for key in shared):
            matches.append(scan)
    if len(matches) != 1 or not isinstance(matches[0].get("op_point"), Mapping):
        raise ValueError("current HIF acquisition requires one auxiliary-bound scan and an explicit operating point")
    if runtime.get("op_point") is not None and runtime["op_point"] != matches[0]["op_point"]:
        raise ValueError("current HIF operating point conflicts with the bound acquisition")
    scan = matches[0]
    return {key: copy.deepcopy(scan[key]) for key in ("scan_index", "op_point", "time_tag") if key in scan}


def fit_receipt(payload: Mapping[str, Any], case_path: str, *, independent: bool,
                metadata: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Small model receipt: no clean vectors, injected labels, or meter targets."""
    return {
        **{key: copy.deepcopy(payload[key]) for key in (
            "success", "candidate_branch_row0", "estimated", "uncertainty", "search",
            "measurement_convention", "selected_scan_indices",
        ) if key in payload},
        "case_sha256": case_fingerprint(case_path),
        "independent_of_current_scada": independent,
        "acquisition_sha256": acquisition_fingerprint(metadata) if independent and metadata is not None else None,
        "model_sha256": model_fingerprint((metadata.get("hif_scan_window") or {}).get("pristine_model_dir")) if independent and metadata is not None else None,
    }


def acquisition_fingerprint(metadata: Mapping[str, Any]) -> str:
    runtime = metadata.get("hif_runtime") or {}
    observable = {"binding": current_scan(metadata), **{key: runtime.get(key) for key in (
        "three_phase_voltages", "three_phase_branch_currents", "sigma_z", "three_phase_sigma",
        "branch_current_sigma_pu", "measurement_convention",
    )}}
    return hashlib.sha256(json.dumps(observable, sort_keys=True, allow_nan=False).encode()).hexdigest()


def model_fingerprint(model_dir: str | None) -> str:
    from three_phase_nlm.hif_parameter_estimator import _resolve_model_dir
    root = Path(_resolve_model_dir(model_dir, "case14")).resolve()
    # Glob case matching differs between Windows and Linux. Hash every DSS
    # circuit file, including the uppercase .DSS assets used by this model,
    # and order by the same relative-path strings on both platforms.
    files = sorted(
        (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".dss"),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not files:
        raise ValueError("HIF forward model has no DSS circuit files")
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def case_fingerprint(case_path: str) -> str:
    from mcp_server.matpower_server import _load_python_case
    case = _load_python_case(case_path)
    data = {key: np.asarray(case[key]).tolist() for key in ("baseMVA", "bus", "branch", "gen")}
    return hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


def accepted_fit(state: Mapping[str, Any]) -> Mapping[str, Any] | None:
    observation = state.get("policy_observation") or {}
    for record in reversed(observation.get("explained_anomalies") or []):
        if isinstance(record, Mapping) and record.get("family") == "hif":
            detail = record.get("detail")
            fit = detail.get("conditioning_fit") if isinstance(detail, Mapping) else None
            return fit if isinstance(fit, Mapping) else {}
    return None


def conditioned_prediction(state: Mapping[str, Any], cache: dict[str, Any]) -> dict[str, Any] | None:
    fit = accepted_fit(state)
    if fit is None:
        return None
    if fit.get("independent_of_current_scada") is not True:
        raise ValueError("HIF conditioning requires a fit independent of the current SCADA acquisition")
    case = state["case"]
    case_path = case.get("case_path") if isinstance(case, Mapping) else case
    if case_fingerprint(case_path) != fit.get("case_sha256"):
        raise ValueError("HIF fit is stale after a network model change")
    metadata = state.get("metadata") or {}
    target = current_scan(metadata)
    if fit.get("acquisition_sha256") != acquisition_fingerprint(metadata):
        raise ValueError("HIF fit is stale after an auxiliary acquisition change")
    convention = resolve_shunt_convention(None, metadata)
    # The ordinary provider WLS uses shunts in Ybus. Legacy corpora require a
    # separately adapted observation model, never a silently mismatched solve.
    if convention != "ybus":
        raise ValueError("HIF continuation requires the declared ybus measurement convention")
    model_dir = (metadata.get("hif_scan_window") or {}).get("pristine_model_dir")
    if fit.get("model_sha256") != model_fingerprint(model_dir):
        raise ValueError("HIF fit is stale after a physical forward-model change")
    key = json.dumps([fit, target, convention, model_dir], sort_keys=True, allow_nan=False)
    if key not in cache:
        kwargs = {**target, "snapshot_id": "bound_auxiliary_acquisition",
                  "pristine_model_dir": model_dir, "shunt_convention": convention}
        replay = replay_hif_measurement_effect(fit, **kwargs)
        estimated = fit["estimated"]
        if estimated.get("resistance_model", "shared") not in {"shared", "single_snapshot"}:
            raise ValueError("scan-specific HIF prediction uncertainty is not supported for meter continuation")
        uncertainty = fit.get("uncertainty") or {}
        alphas = uncertainty.get("near_best_alpha_interval")
        resistances = uncertainty.get("near_best_r_hif_pu_interval")
        if not alphas or not resistances:
            raise ValueError("HIF fit lacks the sensitivity profile required for meter continuation")
        vectors = [np.asarray(replay["predicted_hif_measurements"], dtype=float)]
        for alpha in sorted(set(alphas)):
            for resistance in sorted(set(resistances)):
                varied = copy.deepcopy(fit)
                varied["estimated"].update(alpha_from_from_bus=float(alpha), r_hif_pu=float(resistance))
                vectors.append(np.asarray(replay_hif_measurement_effect(varied, **kwargs)["predicted_hif_measurements"], dtype=float))
        replay["prediction_lower"] = np.min(vectors, axis=0).tolist()
        replay["prediction_upper"] = np.max(vectors, axis=0).tolist()
        # Bounded per-provider cache: repeated WLS/candidate checks reuse only
        # the model prediction, never a discrepancy computed on an old state.
        if len(cache) >= 32:
            cache.clear()
        cache[key] = replay
    return copy.deepcopy(cache[key])


def diagnose(state: Mapping[str, Any], prediction: Mapping[str, Any], sigma: Any) -> dict[str, Any]:
    result = diagnose_conditioned_meter_errors(
        state["measurements"], prediction["predicted_hif_measurements"], sigma,
        prediction_lower=prediction["prediction_lower"], prediction_upper=prediction["prediction_upper"],
        event_effect=prediction["measurement_effect"], detection_sigma=5.0,
        max_envelope_width_sigma=2.0,
    )
    reasons = list(result["failure_reasons"])
    # Wide predictions can hide a fault even when it is not a candidate.
    if result["wide_envelope_indices"]:
        reasons.append("prediction_envelope_too_wide_for_closure")
    result["conditioning"] = {
        "status": "unavailable" if reasons else "ready",
        "state_id": str(state.get("state_id") or ""),
        "state_hash": str(state.get("state_hash") or ""),
        "method": "paired_opendss_effect_compensation",
        "remaining_meter_candidate_indices": result["candidate_indices"],
        "failure_reasons": reasons,
        "overlapping_candidate_indices": result["candidate_event_overlap_indices"],
        "physical_fault_still_present": True,
        "channels_excluded": [],
        "effect_covariance_available": False,
        "statistical_interpretation": "conditional WLS diagnostic; nominal false-alarm calibration not established",
    }
    return result
