"""Instrument capabilities and the strict balanced-SCADA execution boundary.

Profiles are runtime configuration, never inferred from a scenario's family or
truth. Sanitizers copy observable execution data; callers retain offline audit
truth separately and never merge it into a provider payload.
"""
from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Mapping


DEFAULT_EVIDENCE_PROFILE = "scada_only"
AUXILIARY_EVIDENCE_PROFILE = "auxiliary_diagnostics"
EVIDENCE_PROFILES = (DEFAULT_EVIDENCE_PROFILE, AUXILIARY_EVIDENCE_PROFILE)
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


def validate_evidence_profile(value: Any = DEFAULT_EVIDENCE_PROFILE) -> str:
    if not isinstance(value, str) or value not in EVIDENCE_PROFILES:
        raise ValueError(f"evidence_profile must be one of {EVIDENCE_PROFILES}")
    return value


def is_scada_only(value: Any = None) -> bool:
    if isinstance(value, Mapping):
        profile = value.get("evidence_profile", DEFAULT_EVIDENCE_PROFILE)
    elif value is None:
        profile = DEFAULT_EVIDENCE_PROFILE
    elif isinstance(value, str):
        profile = value
    else:
        profile = getattr(value, "evidence_profile", DEFAULT_EVIDENCE_PROFILE)
    return validate_evidence_profile(profile) == DEFAULT_EVIDENCE_PROFILE


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
    result["evidence_profile"] = DEFAULT_EVIDENCE_PROFILE
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
        result["policy_observation"]["evidence_profile"] = DEFAULT_EVIDENCE_PROFILE
    result["evidence_profile"] = DEFAULT_EVIDENCE_PROFILE
    return result
