"""Measure an observable expert's executable investigation prefix.

Inputs are an operator model, an already observed sensor snapshot, its declared
noise, and optional acquired-sensor metadata. No family, clean mean, fault label,
offline residual energy, or source action is accepted. A successful prefix is
not a fault-identification, correction, physical-recovery, or release claim.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from psse_env.actions import (
    ASK_FOR_MORE_EVIDENCE, CONTEXT_TOOLS, CORRECTION_TOOLS, FINALIZE_DIAGNOSIS,
    GET_HARMONIC_CONTEXT, GET_THREE_PHASE_CONTEXT, RUN_WLS,
)
from psse_env.dagger.dataset_builder import validate_policy_payload
from psse_env.dagger.release_factories import select_observable_expert_actions
from psse_env.episode_budget import DEFAULT_EPISODE_ACTION_LIMIT, bind_env_action_limit
from psse_env.oracle import ExpertPolicyOracle
from psse_env.noise_contract import validate_noise_channel
from scripts.run_dagger_research import research_diagnostic_environment_factory
from Transmission.generate_measurements import write_ppc_as_matpower_m


CONTRACT = "wls_observable_expert_prefix_v1"
_ALLOWED_METADATA = {
    "measurement_kind", "sigma_z", "structural_zero_indices", "measurement_ids", "noise_contract",
    "three_phase_voltages", "three_phase_branch_currents", "three_phase_sigma",
    "branch_current_sigma_pu", "harmonic_measurements", "harmonic_orders",
    "parameter_scans", "substation_telemetry", "operator_layout", "operator_noise",
    "operator_voltage_meter_nodes", "reported_breaker_status", "topology_model_id",
    "topology_model_fingerprint",
}
_PRIVATE_KEYS = {
    "family", "families", "scenariofamily", "cohort", "severity", "label", "labels",
    "offlineaudit", "offlinemetadata", "jexact", "observablestrength", "physicalseverity",
    "residualvisibleenergybin", "zclean", "ztrue", "zexact", "vcomplextrue",
    "threephasevoltagesclean", "threephasebranchcurrentsclean", "initialstates",
    "unresolvedsignatures", "nlmdiagnostic", "hifruntime", "hifscanwindow",
}
_ACQUISITIONS = set(CONTEXT_TOOLS) | {GET_THREE_PHASE_CONTEXT, GET_HARMONIC_CONTEXT}


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None  # Unavailable numeric evidence is never converted to zero.
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _check_no_private(value: Any, path: str = "metadata") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            canonical = re.sub(r"[^a-z0-9]", "", str(key).lower())
            if canonical in _PRIVATE_KEYS:
                raise ValueError(f"Private or externally diagnosed input is forbidden: {path}.{key}")
            _check_no_private(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _check_no_private(item, f"{path}[{index}]")


def _inputs(case: Mapping[str, Any], z: Sequence[float], sigma: Sequence[float], metadata: Mapping[str, Any] | None):
    if not isinstance(case, Mapping) or not {"baseMVA", "bus", "branch", "gen"} <= set(case):
        raise ValueError("Probe requires a full public operator case with bus, branch, gen and baseMVA; a WLS-only whitelist is insufficient")
    configured = {key: deepcopy(case[key]) for key in ("version", "baseMVA", "bus", "branch", "gen", "gencost") if key in case}
    for key in ("bus", "branch", "gen", "gencost"):
        if key not in configured:
            continue
        configured[key] = np.array(configured[key], dtype=float, copy=True)
        if configured[key].ndim != 2 or not np.isfinite(configured[key]).all():
            raise ValueError(f"Operator {key} must be a finite matrix")
    bus, branch = configured["bus"], configured["branch"]
    if len(bus) != 14 or len(branch) != 20 or bus.shape[1] < 13 or branch.shape[1] < 13:
        raise ValueError("This reviewed prefix probe requires the fixed IEEE14 122-channel operator layout")
    if np.any(bus[:, 12] <= 0) or np.any(bus[:, 11] <= bus[:, 12]):
        raise ValueError("Full public operator voltage bounds are required; zeroed WLS-feature bounds are not valid execution limits")
    base_mva = float(configured["baseMVA"])
    if not math.isfinite(base_mva) or base_mva <= 0:
        raise ValueError("Operator baseMVA must be finite and positive")
    observed, weights = np.array(z, dtype=float, copy=True), np.array(sigma, dtype=float, copy=True)
    if observed.shape != (122,) or weights.shape != observed.shape or not np.isfinite(observed).all():
        raise ValueError("Observed z and sigma must match the fixed 122-channel layout")
    if not np.isfinite(weights).all() or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("sigma must be finite nonnegative, with stochastic channels")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise ValueError("observable_metadata must be a mapping")
    metadata = deepcopy(dict(metadata or {}))
    if set(metadata) - _ALLOWED_METADATA:
        raise ValueError(f"Unsupported observable metadata keys: {sorted(set(metadata) - _ALLOWED_METADATA)}")
    _check_no_private(metadata)
    validate_policy_payload(metadata)
    if metadata.get("measurement_kind", "observed") != "observed":
        raise ValueError("Probe consumes observed sensor data, never a noiseless mean as measurements")
    for key in ("three_phase_sigma", "branch_current_sigma_pu"):
        if key in metadata:
            value = metadata[key]
            if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not math.isfinite(float(value)) or float(value) <= 0:
                raise ValueError(f"{key} must be a finite positive declared sensor sigma")
    if metadata.get("sigma_z") is not None:
        declared = np.asarray(metadata["sigma_z"], dtype=float)
        if declared.shape != weights.shape or not np.allclose(declared, weights, rtol=1e-12, atol=0):
            raise ValueError("Metadata sigma_z conflicts with the supplied sensor covariance")
    if "noise_contract" in metadata:
        contract = metadata["noise_contract"]
        if not isinstance(contract, Mapping) or contract.get("schema") != "generated_sensor_noise_v1":
            raise ValueError("Unsupported acquired-sensor noise contract")
        channels = contract.get("channels")
        expected = {"scada": weights, "three_phase_voltages": metadata.get("three_phase_sigma"),
                    "three_phase_branch_currents": metadata.get("branch_current_sigma_pu")}
        if not isinstance(channels, Mapping) or set(channels) - set(expected):
            raise ValueError("Noise contract has unsupported sensor channels")
        for channel in ("three_phase_voltages", "three_phase_branch_currents"):
            if metadata.get(channel) and channel not in channels:
                raise ValueError(f"Noise contract is missing acquired channel {channel}")
        # An auxiliary-only contract need not claim to have drawn SCADA noise.
        # The caller separately supplies the authoritative SCADA sigma vector.
        for channel, declaration in channels.items():
            if not isinstance(declaration, Mapping) or expected[channel] is None:
                raise ValueError(f"Noise contract lacks matching declared {channel} sigma")
            actual = np.asarray(expected[channel], dtype=float)
            declared = np.asarray(declaration.get("estimator_sigma_per_component"), dtype=float)
            if declared.shape != actual.shape or not np.allclose(declared, actual, rtol=1e-12, atol=0):
                raise ValueError(f"Noise contract {channel} estimator sigma disagrees with acquired metadata")
            validate_noise_channel(channel=channel, role=declaration.get("role"),
                distribution=declaration.get("distribution"),
                applied_sigma=declaration.get("applied_sigma_per_component"), estimator_sigma=expected[channel],
                representation="scalar" if channel == "scada" else "complex_rectangular",
                require_matched_gaussian=True)
    exact = metadata.get("structural_zero_indices", [])
    if (not isinstance(exact, (list, tuple)) or any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in exact)
            or len(set(exact)) != len(exact) or set(np.flatnonzero(weights == 0).tolist()) != set(exact)):
        raise ValueError("Zero sigmas require exactly the declared structural measurement constraints")
    metadata["sigma_z"] = weights.tolist()
    metadata["measurement_kind"] = "observed"
    # Stored PF/OPF state is never estimator initialization. Model R/X/status,
    # public limits, external bus IDs and physical measurement ordering remain.
    configured["bus"][:, 7] = 1.0
    configured["bus"][:, 8] = 0.0
    return configured, observed, weights, _jsonable(metadata)


def _availability(action: Mapping[str, Any], output: Mapping[str, Any], observation: Mapping[str, Any]):
    tool = str(action.get("tool") or "")
    metrics = output.get("tool_metrics") or {}
    succeeded = output.get("execution_status") == "success"
    same_state = metrics.get("state_id") == observation.get("active_state_id")
    supported = [deepcopy(item) for item in metrics.get("supported_corrections") or []
                 if isinstance(item, Mapping) and item.get("tool") in CORRECTION_TOOLS
                 and isinstance(item.get("arguments"), Mapping)
                 and item["arguments"].get("state_id") == observation.get("active_state_id")]
    if tool == GET_THREE_PHASE_CONTEXT:
        available = succeeded and same_state and metrics.get("three_phase_context_status") == "available" and bool(metrics.get("available_evidence_channels"))
        return bool(available), bool(available), supported
    if tool == GET_HARMONIC_CONTEXT:
        available = succeeded and same_state and metrics.get("harmonic_context_status") == "available" and bool(metrics.get("available_evidence_channels"))
        return bool(available), bool(available), supported
    if tool in CONTEXT_TOOLS:
        available = succeeded and same_state and (bool(supported) or int(metrics.get("finding_count") or 0) > 0)
        # Empty or unsupported recommendations are not an executable recovery
        # prefix merely because the current WLS supplied a ranked finding.
        actionable = available and bool(supported)
        return bool(available), bool(actionable), supported
    if tool in CORRECTION_TOOLS:
        actionable = succeeded and bool(output.get("state_mutated"))
        return False, actionable, supported
    return False, False, supported


def probe_expert_action(
    case: Mapping[str, Any], z: Sequence[float], sigma: Sequence[float], *,
    observable_metadata: Mapping[str, Any] | None = None, seed: int = 20260917,
    max_actions: int = DEFAULT_EPISODE_ACTION_LIMIT,
) -> dict[str, Any]:
    """Execute the existing observable expert to its first evidenced prefix.

    Fresh WLS is selected and executed first. Successful negative acquisition
    replies remain in history but do not qualify. The probe then stops at an
    available acquired channel, a state-bound supported context, or an executed
    correction. A quiet snapshot instead executes observable finalization.
    All actions, including WLS, share the forty-action default horizon.

    The caller must supply already noisy observations. No random noise, clean
    means, latent initialization, external family flags or GNN decisions enter
    this function. Returned events carry actual evidence for a separate audited
    SFT exporter; this helper never manufactures production-label eligibility.
    """
    configured, observed, weights, metadata = _inputs(case, z, sigma, observable_metadata)
    env = research_diagnostic_environment_factory(seed=seed)
    limit = bind_env_action_limit(env, max_actions)
    owner = getattr(env.wls_runner, "__self__", None)
    if getattr(owner, "screen_checkpoint", None) or getattr(owner, "screen_calibration", None):
        raise ValueError("Expert admission probe requires GNN screening disabled")
    expert = ExpertPolicyOracle(process_oracle=env.process_oracle, candidate_oracle=env.candidate_quality_oracle)
    receipt = {"contract": CONTRACT, "scope": "evidenced_expert_prefix", "action_limit": limit,
        "gnn_enabled": False, "external_family_flags_used": False,
        "fault_actionable": False, "training_eligible": False, "healthy_completion_valid": False,
        "wls_success": False, "measured_alarm": None, "initial_wls": None,
        "events": [], "actionable_event": None, "actionable_event_index": None,
        "healthy_completion_event_index": None, "reasons": [], "acquisition_limitations": [],
        "full_repair_validated": False, "production_label_eligibility_manufactured": False,
        "initialization": "flat_vm1_va0; no latent PF/OPF initialization",
        "input_binding": {"case_sha256": _digest(configured), "observations_sha256": _digest(observed),
                          "sigma_sha256": _digest(weights), "metadata_sha256": _digest(metadata), "measurement_count": len(observed)},
        "runtime_settings": {"chi_square_alpha": getattr(owner, "chi2_alpha", None),
                             "normalized_residual_threshold": getattr(owner, "normalized_residual_threshold", None)},
        "export_requirement": "use actual event evidence, canonical rebinding and alias_before_compaction; quarantine export/audit failures",
    }
    history = []
    with tempfile.TemporaryDirectory(prefix="reviewed_expert_probe_") as temporary:
        path = write_ppc_as_matpower_m(configured, Path(temporary) / "configured_case.m", "configured_case")
        env.reset({"scenario_id": f"reviewed_probe_{receipt['input_binding']['observations_sha256'][:16]}",
                   "case": str(path), "measurements": observed.tolist(), "metadata": metadata})
        for step in range(limit):
            observation = env.get_policy_observation(history).as_dict()
            validate_policy_payload(observation)
            try:
                selection = select_observable_expert_actions(policy_observation=observation, expert_oracle=expert)
            except (ValueError, TypeError, RuntimeError) as exc:
                receipt["selection_error"] = f"{type(exc).__name__}: {exc}"
                receipt["reasons"].append("observable_expert_selection_failed")
                break
            action = deepcopy(selection.preferred_action)
            if action is None:
                receipt["reasons"].append("observable_expert_returned_no_action")
                break
            event = {"step": step, "phase": "initial_wls" if step == 0 else "expert",
                "policy_observation": observation, "preferred_action": action,
                "ordered_actions": deepcopy(list(selection.actions)), "selection_basis": selection.selection_basis,
                "process_evidence_valid": False, "training_decision_evidence_verified": False,
                "action_executed": False, "execution_success": False, "action_attempted": False,
                "available": False, "supported_recommendations": [], "qualifying_action": False,
                "tool_output": None}
            if step == 0 and action.get("tool") != RUN_WLS:
                receipt["events"].append(event)
                receipt["reasons"].append("observable_expert_did_not_select_initial_wls")
                break
            try:
                env.assert_training_decision_evidence(action)
                event["process_evidence_valid"] = event["training_decision_evidence_verified"] = True
            except (ValueError, TypeError) as exc:
                event["evidence_error"] = f"{type(exc).__name__}: {exc}"
                receipt["events"].append(event)
                receipt["reasons"].append("selected_action_lacks_observable_training_evidence")
                break
            event["action_attempted"] = True
            try:
                _state, output = env.step(action)
            except Exception as exc:
                event["execution_exception"] = f"{type(exc).__name__}: {exc}"
                receipt["events"].append(event)
                receipt["reasons"].append("environment_execution_exception")
                break
            event["tool_output"] = deepcopy(output)
            event["action_executed"] = event["execution_success"] = output.get("execution_status") == "success"
            event["available"], candidate, event["supported_recommendations"] = _availability(action, output, observation)
            event["qualifying_action"] = bool(candidate and event["process_evidence_valid"])
            receipt["events"].append(event)
            history.append({"action": action, "tool_output": output})
            metrics = output.get("tool_metrics") or {}
            if step == 0:
                converged = metrics.get("converged", metrics.get("state_estimation_converged", False))
                success = event["execution_success"] and bool(converged)
                chi, residual = metrics.get("chi_square_alarm"), metrics.get("normalized_residual_alarm")
                alarm = bool(chi or residual) if success and isinstance(chi, bool) and isinstance(residual, bool) else None
                receipt["wls_success"] = bool(success)
                receipt["measured_alarm"] = alarm
                receipt["initial_wls"] = {"success": bool(success), "J": metrics.get("chi_square_statistic"),
                    "threshold": metrics.get("chi_square_threshold"), "max_normalized_residual": metrics.get("max_normalized_residual"),
                    "normalized_residual_threshold": metrics.get("normalized_residual_threshold"),
                    "chi_square_alarm": chi, "normalized_residual_alarm": residual, "alarm": alarm,
                    "tool_output": deepcopy(output)}
                if not success or alarm is None:
                    receipt["reasons"].append("initial_wls_failed_or_missing_measured_alarm")
                    break
                continue
            if not event["execution_success"]:
                receipt["reasons"].append("selected_action_failed_or_invalid")
                break
            if action["tool"] in _ACQUISITIONS and not event["available"]:
                receipt["acquisition_limitations"].append(f"{action['tool']}:unavailable_or_empty")
            if action["tool"] == FINALIZE_DIAGNOSIS:
                healthy = receipt["measured_alarm"] is False and bool(env.terminal) and env.terminal_outcome == "resolved"
                receipt["healthy_completion_valid"] = healthy
                receipt["healthy_completion_event_index"] = len(receipt["events"]) - 1 if healthy else None
                receipt["reasons"].append("quiet_observable_finalization" if healthy else "finalized_without_fault_investigation")
                break
            if action["tool"] == ASK_FOR_MORE_EVIDENCE and str(action.get("arguments", {}).get("request", "")).startswith("operator_escalation:"):
                receipt["reasons"].append("operator_handoff_not_actionable_training")
                break
            if receipt["measured_alarm"] is True and event["qualifying_action"]:
                receipt["fault_actionable"] = receipt["training_eligible"] = True
                receipt["actionable_event"] = deepcopy(event)
                receipt["actionable_event_index"] = len(receipt["events"]) - 1
                receipt["reasons"].append("evidenced_expert_prefix")
                break
            if env.terminal:
                receipt["reasons"].append("terminal_without_evidenced_investigation")
                break
        else:
            receipt["reasons"].append("episode_action_budget_exhausted_before_evidenced_prefix")

    last = receipt["actionable_event"] or (receipt["events"][-1] if receipt["events"] else {})
    receipt.update(policy_observation=last.get("policy_observation"), selected_action=last.get("preferred_action"),
        step_result=last.get("tool_output"), action_executed=bool(last.get("action_executed")),
        process_evidence_valid=bool(last.get("process_evidence_valid")), acquired_evidence_available=bool(last.get("available")),
        available_context_tool=last.get("preferred_action", {}).get("tool") if last.get("available") else None,
        supported_recommendations=last.get("supported_recommendations", []), executed_action_count=sum(e["action_attempted"] for e in receipt["events"]))
    requests = [event for event in receipt["events"] if event["preferred_action"]["tool"] in _ACQUISITIONS]
    receipt["request_counts"] = {"selected": len(requests), "attempted": sum(e["action_attempted"] for e in requests),
        "process_evidence_valid": sum(e["process_evidence_valid"] for e in requests),
        "executed_successfully": sum(e["execution_success"] for e in requests),
        "available": sum(e["available"] for e in requests), "unavailable_or_empty": sum(e["execution_success"] and not e["available"] for e in requests),
        "by_tool": dict(Counter(e["preferred_action"]["tool"] for e in requests))}
    receipt.update(wls_alarm=receipt["measured_alarm"], expert_valid=receipt["process_evidence_valid"],
                   execution_success=receipt["action_executed"], safe_finalize=receipt["healthy_completion_valid"],
                   reason=receipt["reasons"][-1] if receipt["reasons"] else "no_evidenced_result")
    return _jsonable(receipt)


__all__ = ["CONTRACT", "probe_expert_action"]
