"""Instrument capabilities and the strict balanced-SCADA execution boundary.

Profiles are runtime configuration, never inferred from a scenario's family or
truth. Sanitizers copy observable execution data; callers retain offline audit
truth separately and never merge it into a provider payload.
"""
from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Mapping


#: Balanced SCADA and its WLS only; every auxiliary tool is refused.
SCADA_ONLY_PROFILE = "scada_only"
#: Research default (2026-09-23). Detection uses only balanced SCADA and WLS:
#: no fault flags, hints or precomputed diagnoses reach the agent. After a
#: current balanced WLS alarm on the active state the agent may request the
#: auxiliary streams that the ground truth generated for that root (three-phase
#: PMU phasors on HIF and unbalance roots, spectra on harmonic roots, breaker
#: telemetry on topology roots) and run the matching diagnostics.
WLS_GATED_PROFILE = "wls_gated_diagnostics"
#: Historical reproduction: flagged roots, seeded signatures and hints allowed.
AUXILIARY_EVIDENCE_PROFILE = "auxiliary_diagnostics"
DEFAULT_EVIDENCE_PROFILE = WLS_GATED_PROFILE
EVIDENCE_PROFILES = (SCADA_ONLY_PROFILE, WLS_GATED_PROFILE, AUXILIARY_EVIDENCE_PROFILE)
#: Profiles whose truth boundary is strict: no seeded fault signatures, no
#: private family or correction hints, no precomputed diagnoses, WLS first,
#: no synthetic terminal closure, scenario identity kept out of store metadata.
STRICT_BOUNDARY_PROFILES = frozenset({SCADA_ONLY_PROFILE, WLS_GATED_PROFILE})
#: Auxiliary diagnostics that wls_gated_diagnostics permits only after a
#: current balanced WLS alarm (chi-square or normalized residual) on the
#: active state.
GATED_DIAGNOSTIC_TOOLS = frozenset({
    "get_three_phase_context", "get_harmonic_context", "run_three_phase_nlm_from_path",
    "run_hse_from_path", "estimate_hif_location_magnitude_from_path",
    "estimate_hif_location_magnitude_multiscan_from_path",
})
#: Never available under wls_gated_diagnostics: the legacy alternative test has
#: no provider, and learned screens are an additional signal by construction.
WLS_GATED_DISABLED_TOOLS = frozenset({"run_alternative_test"})
WLS_GATED_DISABLED_REQUESTS: frozenset[str] = frozenset()
#: Precomputed diagnoses and truth-side model handles that a strict root must
#: not carry into execution: the tools recompute from measurements.
PRECOMPUTED_DIAGNOSIS_FIELDS = frozenset({
    "nlm_diagnostic", "faulted_model_dir", "hif_fit", "hif_estimate", "hse_summary",
    "nlm_summary", "three_phase_summary", "harmonic_summary", "anomaly_explanation",
    "diagnostic_acceptance",
})
SCADA_ALLOWED_TOOLS = frozenset({
    "run_wls", "verify_candidate", "get_measurement_context", "get_parameter_context",
    "get_topology_context", "correct_measurements", "correct_parameters", "correct_topology",
    "commit_state", "rollback_state", "finalize_diagnosis", "ask_for_more_evidence",
})
SCADA_DISABLED_TOOLS = frozenset({
    "run_alternative_test",
    "get_three_phase_context", "get_harmonic_context", "run_three_phase_nlm_from_path",
    "run_hse_from_path", "estimate_hif_location_magnitude_from_path",
    "estimate_hif_location_magnitude_multiscan_from_path",
})
SCADA_DISABLED_REQUESTS = frozenset({
    "operator_escalation:hif_diagnostics_exhausted",
    "operator_escalation:hif_conditioning_unavailable",
})
_WAVEFORM_SIGNATURE = re.compile(
    r"(?<![a-z0-9])(?:hif|harmonic|unbalance|imbalance|three[_ -]?phase|zero_sequence|negative_sequence)(?![a-z0-9])",
    re.IGNORECASE,
)
_NOISE_FIELDS = frozenset({
    "schema", "role", "distribution", "noise_scale", "sigma_semantics",
    "applied_sigmas_match_estimator", "population_calibration_verified",
})
_NOISE_CHANNEL_FIELDS = frozenset({
    "schema", "channel", "role", "distribution", "representation", "sigma_semantics",
    "applied_sigma", "estimator_sigma", "applied_variance", "estimator_variance",
    "applied_sigma_per_component", "estimator_sigma_per_component",
    "applied_covariance", "estimator_covariance", "sigmas_match", "variances_match",
})
_OPERATOR_NOISE_FIELDS = frozenset({
    "schema", "measurement_sigma", "measurement_covariance", "covariance",
    "structural_zero_indices", "measurement_ids", "covariance_model",
})
_PARAMETER_SCAN_FIELDS = frozenset({
    "z_scans", "sigma_z", "scan_indices", "time_tags", "scan_index", "time_tag",
})
_SCADA_METADATA_FIELDS = frozenset({
    "sigma_z", "measurement_covariance", "structural_zero_indices",
})
_EXECUTION_FIELDS = frozenset({
    "scenario_id", "id", "episode_id", "case", "case_path", "measurements", "z_obs",
    "state_id", "state_hash", "parent_state_id", "status", "candidate_status",
    "candidate_lifecycle", "source_action", "created_at_step", "depth", "active",
    "sigma_z", "structural_zero_indices", "measurement_covariance",
    "evidence_request",
})
_AUXILIARY_FIELDS = frozenset({
    "three_phase_voltages", "three_phase_branch_currents", "three_phase_sigma",
    "branch_current_sigma_pu", "harmonic_measurements", "harmonic_phasors", "harmonic_sigma",
    "harmonic_summary", "harmonic_context_status", "harmonic_distortion_detected",
    "harmonic_screening", "hse_summary", "nlm_summary", "nlm_diagnostic",
    "three_phase_summary", "three_phase_context_status", "anomaly_explanation",
    "diagnostic_acceptance", "physical_fault_still_present", "conditional_meter_scores",
    "op_point", "initial_states", "load_profile", "load_profiles", "label", "labels",
})
#: Execution metadata a wls_gated_diagnostics root may carry: balanced SCADA
#: declarations, the auxiliary measurement streams with their declared sigmas,
#: node/breaker telemetry, and the shared forward-model handle.  Precomputed
#: diagnoses, faulted models, labels and operating-point truth are absent.
_GATED_METADATA_FIELDS = frozenset({
    "sigma_z", "measurement_covariance", "structural_zero_indices", "slack_bus",
    "three_phase_voltages", "three_phase_branch_currents", "three_phase_sigma",
    "branch_current_sigma_pu", "harmonic_measurements", "harmonic_orders",
    "substation_telemetry", "reported_breaker_status", "operator_layout",
    "operator_voltage_meter_nodes", "topology_model_id", "topology_model_fingerprint",
    "last_topology_correction", "pristine_model_dir",
})
_GATED_STRUCTURED_METADATA_FIELDS = frozenset({
    "parameter_scans", "operator_noise", "noise_contract", "measurement_convention",
    "hif_runtime", "hif_scan_window",
})
_GATED_RUNTIME_FIELDS = frozenset({
    "scan_index", "time_tag", "op_point", "load_scale", "z_obs", "sigma_z",
    "three_phase_voltages", "three_phase_branch_currents", "three_phase_sigma",
    "branch_current_sigma_pu", "pristine_model_dir", "measurement_convention",
    "noise_contract", "topology_id",
})
_GATED_SCAN_FIELDS = frozenset({
    "scan_index", "time_tag", "z_obs", "z", "sigma_z", "op_point", "topology_id",
    "three_phase_voltages", "three_phase_branch_currents", "three_phase_sigma",
    "branch_current_sigma_pu", "noise_contract", "measurement_convention",
})
_GATED_WINDOW_FIELDS = frozenset({
    "scan_window_path", "sigma_z", "three_phase_sigma", "branch_current_sigma_pu",
    "pristine_model_dir", "measurement_convention", "noise_contract", "op_point",
    "topology_id",
})
_MEASUREMENT_CONVENTION_FIELDS = frozenset({"schema", "shunt_convention", "wls_model_convention"})
#: Truth, labels and precomputed diagnoses that never enter a
#: wls_gated_diagnostics observation or provider payload.  Diagnostic outputs
#: the agent earned through an admitted tool (nlm_summary, explanations,
#: conditioning ledgers, minted signatures) are deliberately kept.
_GATED_OBSERVATION_DENYLIST = frozenset({
    "hidden_truth", "label", "labels", "scenario_family", "family_hint",
    "oracle_action_hints", "suggested_actions", "nlm_diagnostic", "faulted_model_dir",
    "initial_states", "load_profile", "load_profiles", "op_point", "load_scale",
    "gnn_screen", "hif_runtime", "hif_scan_window", "window_metadata", "release_audit",
    "topology_ranking", "clean_case", "clean_measurements", "clean_parameter_values",
    "z_true", "z_clean", "three_phase_voltages_clean", "three_phase_branch_currents_clean",
})


def validate_evidence_profile(value: Any = DEFAULT_EVIDENCE_PROFILE) -> str:
    if not isinstance(value, str) or value not in EVIDENCE_PROFILES:
        raise ValueError(f"evidence_profile must be one of {EVIDENCE_PROFILES}")
    return value


def resolve_evidence_profile(value: Any = None) -> str:
    """Profile named by a string, a mapping/object with ``evidence_profile``, or None."""
    if isinstance(value, Mapping):
        profile = value.get("evidence_profile", DEFAULT_EVIDENCE_PROFILE)
    elif value is None:
        profile = DEFAULT_EVIDENCE_PROFILE
    elif isinstance(value, str):
        profile = value
    else:
        profile = getattr(value, "evidence_profile", DEFAULT_EVIDENCE_PROFILE)
    if profile is None:
        profile = DEFAULT_EVIDENCE_PROFILE
    return validate_evidence_profile(profile)


def is_scada_only(value: Any = None) -> bool:
    """Every auxiliary stream and tool is refused (the literal scada_only profile)."""
    return resolve_evidence_profile(value) == SCADA_ONLY_PROFILE


def is_strict_boundary(value: Any = None) -> bool:
    """No seeded fault signatures, hints, precomputed diagnoses or synthetic closure."""
    return resolve_evidence_profile(value) in STRICT_BOUNDARY_PROFILES


def is_wls_gated(value: Any = None) -> bool:
    return resolve_evidence_profile(value) == WLS_GATED_PROFILE


def allows_diagnostic_tools(value: Any = None) -> bool:
    """Auxiliary diagnostics exist in this profile (gated or not)."""
    return resolve_evidence_profile(value) != SCADA_ONLY_PROFILE


def requires_wls_alarm_for_diagnostics(value: Any = None) -> bool:
    """Auxiliary requests need a current balanced WLS alarm on the active state."""
    return resolve_evidence_profile(value) == WLS_GATED_PROFILE


def disabled_tools(value: Any = None) -> frozenset[str]:
    profile = resolve_evidence_profile(value)
    if profile == SCADA_ONLY_PROFILE:
        return SCADA_DISABLED_TOOLS
    if profile == WLS_GATED_PROFILE:
        return WLS_GATED_DISABLED_TOOLS
    return frozenset()


def disabled_requests(value: Any = None) -> frozenset[str]:
    profile = resolve_evidence_profile(value)
    if profile == SCADA_ONLY_PROFILE:
        return SCADA_DISABLED_REQUESTS
    if profile == WLS_GATED_PROFILE:
        return WLS_GATED_DISABLED_REQUESTS
    return frozenset()


scada_only = is_scada_only


def scada_signatures(values: Any) -> list[str]:
    if isinstance(values, str):
        values = [values]
    return [str(value) for value in (values or []) if not _WAVEFORM_SIGNATURE.search(str(value))]


def _selected(value: Any, fields: frozenset[str]) -> Any:
    if not isinstance(value, Mapping):
        return deepcopy(value)  # Retain malformed allowed input for normal validation.
    return {key: deepcopy(item) for key, item in value.items() if key in fields}


def sanitize_scada_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    metadata = metadata if isinstance(metadata, Mapping) else {}
    result = _selected(metadata, _SCADA_METADATA_FIELDS)
    if "parameter_scans" in metadata:
        result["parameter_scans"] = _selected(metadata["parameter_scans"], _PARAMETER_SCAN_FIELDS)
    if "operator_noise" in metadata:
        result["operator_noise"] = _selected(metadata["operator_noise"], _OPERATOR_NOISE_FIELDS)
    if "noise_contract" in metadata:
        raw = metadata["noise_contract"]
        contract = _selected(raw, _NOISE_FIELDS)
        if isinstance(raw, Mapping):
            channels = raw.get("channels")
            if isinstance(channels, Mapping) and "scada" in channels:
                contract["channels"] = {"scada": _selected(channels["scada"], _NOISE_CHANNEL_FIELDS)}
        result["noise_contract"] = contract
    if "measurement_convention" in metadata:
        raw = metadata["measurement_convention"]
        result["measurement_convention"] = _selected(raw, frozenset({"schema", "shunt_convention", "wls_model_convention"}))
    result["evidence_profile"] = SCADA_ONLY_PROFILE
    return result


def sanitize_scada_observation(value: Any) -> Any:
    """Drop stale auxiliary conclusions while preserving fresh balanced WLS."""
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            text = str(key)
            if text in _AUXILIARY_FIELDS or text.startswith("hif_"):
                continue
            if text in {"semantic_field_provenance", "policy_field_provenance", "policy_provenance"}:
                # These maps describe each field's source. Their entries are
                # provenance strings, not the field values they are named for.
                result[text] = deepcopy(item)
            elif text == "explained_anomalies":
                result[text] = []
            elif text == "unresolved_signatures":
                result[text] = scada_signatures(item)
            elif text == "fresh_context_evidence":
                result[text] = {
                    family: sanitize_scada_observation(record)
                    for family, record in (item.items() if isinstance(item, Mapping) else [])
                    if family in {"wls", "measurement", "parameter", "topology"}
                }
            else:
                result[text] = sanitize_scada_observation(item)
        return result
    if isinstance(value, (list, tuple)):
        return [sanitize_scada_observation(item) for item in value]
    return deepcopy(value)


def sanitize_scada_execution(scenario: Mapping[str, Any]) -> dict[str, Any]:
    """Return execution-only data; private audit fields are deliberately absent."""
    result = _selected(scenario, _EXECUTION_FIELDS)
    result["metadata"] = sanitize_scada_metadata(scenario.get("metadata"))
    for key in ("noise_contract", "operator_noise", "parameter_scans", "measurement_convention"):
        if key in scenario:
            result[key] = sanitize_scada_metadata({key: scenario[key]})[key]
    if isinstance(scenario.get("policy_observation"), Mapping):
        result["policy_observation"] = sanitize_scada_observation(scenario["policy_observation"])
        result["policy_observation"]["evidence_profile"] = SCADA_ONLY_PROFILE
    result["evidence_profile"] = SCADA_ONLY_PROFILE
    return result


def _sanitize_noise_contract(raw: Any) -> Any:
    contract = _selected(raw, _NOISE_FIELDS)
    if isinstance(raw, Mapping):
        channels = raw.get("channels")
        if isinstance(channels, Mapping):
            contract["channels"] = {
                str(name): _selected(channel, _NOISE_CHANNEL_FIELDS)
                for name, channel in channels.items()
            }
    return contract


def _sanitize_scan_window(raw: Any) -> Any:
    if not isinstance(raw, Mapping):
        return deepcopy(raw)
    window = _selected(raw, _GATED_WINDOW_FIELDS)
    if "noise_contract" in raw:
        window["noise_contract"] = _sanitize_noise_contract(raw["noise_contract"])
    if "measurement_convention" in raw:
        window["measurement_convention"] = _selected(raw["measurement_convention"], _MEASUREMENT_CONVENTION_FIELDS)
    scans = raw.get("scans")
    if isinstance(scans, (list, tuple)):
        cleaned = []
        for scan in scans:
            item = _selected(scan, _GATED_SCAN_FIELDS)
            if isinstance(scan, Mapping):
                if "noise_contract" in scan:
                    item["noise_contract"] = _sanitize_noise_contract(scan["noise_contract"])
                if "measurement_convention" in scan:
                    item["measurement_convention"] = _selected(scan["measurement_convention"], _MEASUREMENT_CONVENTION_FIELDS)
            cleaned.append(item)
        window["scans"] = cleaned
    elif "scans" in raw:
        window["scans"] = deepcopy(scans)
    return window


def sanitize_gated_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Execution metadata for wls_gated_diagnostics.

    Keeps the balanced SCADA declarations and the full auxiliary measurement
    streams (three-phase phasors with their declared sigmas, spectra, breaker
    telemetry, repeated scans, the HIF acquisition and its scan window) so the
    admitted diagnostics can compute from measurements.  Drops seeded
    signatures, hidden truth, labels, family and scenario hints and every
    precomputed diagnosis or truth-side model handle.
    """
    metadata = metadata if isinstance(metadata, Mapping) else {}
    result = _selected(metadata, _GATED_METADATA_FIELDS)
    if "parameter_scans" in metadata:
        result["parameter_scans"] = _selected(metadata["parameter_scans"], _PARAMETER_SCAN_FIELDS)
    if "operator_noise" in metadata:
        result["operator_noise"] = _selected(metadata["operator_noise"], _OPERATOR_NOISE_FIELDS)
    if "noise_contract" in metadata:
        result["noise_contract"] = _sanitize_noise_contract(metadata["noise_contract"])
    if "measurement_convention" in metadata:
        result["measurement_convention"] = _selected(metadata["measurement_convention"], _MEASUREMENT_CONVENTION_FIELDS)
    if "hif_runtime" in metadata:
        raw = metadata["hif_runtime"]
        runtime = _selected(raw, _GATED_RUNTIME_FIELDS)
        if isinstance(raw, Mapping):
            if "noise_contract" in raw:
                runtime["noise_contract"] = _sanitize_noise_contract(raw["noise_contract"])
            if "measurement_convention" in raw:
                runtime["measurement_convention"] = _selected(raw["measurement_convention"], _MEASUREMENT_CONVENTION_FIELDS)
        result["hif_runtime"] = runtime
    if "hif_scan_window" in metadata:
        result["hif_scan_window"] = _sanitize_scan_window(metadata["hif_scan_window"])
    for key in PRECOMPUTED_DIAGNOSIS_FIELDS:
        result.pop(key, None)
    result["evidence_profile"] = WLS_GATED_PROFILE
    return result


def sanitize_gated_observation(value: Any) -> Any:
    """Drop truth, labels and precomputed diagnoses; keep earned diagnostics.

    Unlike the SCADA sanitizer this keeps every unresolved signature (the
    environment mints waveform signatures only from admitted diagnostics),
    the explained-anomaly ledger, the three-phase, harmonic and conditioning
    request ledgers and the diagnostic summaries the tools returned.
    """
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            text = str(key)
            if text in _GATED_OBSERVATION_DENYLIST or text.startswith("true_"):
                continue
            if text in {"semantic_field_provenance", "policy_field_provenance", "policy_provenance"}:
                result[text] = deepcopy(item)
            else:
                result[text] = sanitize_gated_observation(item)
        return result
    if isinstance(value, (list, tuple)):
        return [sanitize_gated_observation(item) for item in value]
    return deepcopy(value)


def sanitize_gated_execution(scenario: Mapping[str, Any]) -> dict[str, Any]:
    """Return execution data for wls_gated_diagnostics; audit truth is absent.

    Seeded ``unresolved_signatures`` and their waveform provenance, anomaly
    scores, hints and hidden truth are not execution fields and are dropped;
    the metadata keeps the auxiliary streams through ``sanitize_gated_metadata``.
    """
    result = _selected(scenario, _EXECUTION_FIELDS)
    result["metadata"] = sanitize_gated_metadata(scenario.get("metadata"))
    for key in ("noise_contract", "operator_noise", "parameter_scans", "measurement_convention"):
        if key in scenario:
            result[key] = sanitize_gated_metadata({key: scenario[key]})[key]
    if isinstance(scenario.get("policy_observation"), Mapping):
        result["policy_observation"] = sanitize_gated_observation(scenario["policy_observation"])
        result["policy_observation"]["evidence_profile"] = WLS_GATED_PROFILE
    result["evidence_profile"] = WLS_GATED_PROFILE
    return result


def sanitize_execution_for_profile(scenario: Mapping[str, Any], profile: Any = None) -> dict[str, Any]:
    """Dispatch the execution sanitizer on the strict profile; permissive profiles copy."""
    resolved = resolve_evidence_profile(profile)
    if resolved == SCADA_ONLY_PROFILE:
        return sanitize_scada_execution(scenario)
    if resolved == WLS_GATED_PROFILE:
        return sanitize_gated_execution(scenario)
    return deepcopy(dict(scenario))


def sanitize_observation_for_profile(value: Any, profile: Any = None) -> Any:
    """Dispatch the observation sanitizer on the strict profile; permissive profiles copy."""
    resolved = resolve_evidence_profile(profile)
    if resolved == SCADA_ONLY_PROFILE:
        return sanitize_scada_observation(value)
    if resolved == WLS_GATED_PROFILE:
        return sanitize_gated_observation(value)
    return deepcopy(value)
