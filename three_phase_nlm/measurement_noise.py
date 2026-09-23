"""Matched Gaussian noise and declarations for generated sensor observations.

The forward simulators remain deterministic. Dataset generators use this module
to add noise and publish the same effective standard deviations to estimators.
Zero-noise output is deliberately outside this sensor-observation interface.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

import numpy as np

from psse_env.noise_contract import validate_noise_channel


DEFAULT_THREE_PHASE_SIGMA_PU = 5e-3


def positive_sigma(value: float, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be finite and strictly positive")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be finite and strictly positive") from exc
    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{field} must be finite and strictly positive")
    return number


def scaled_sensor_sigma(base_sigma: float, noise_scale: float, *, field: str) -> float:
    sigma = positive_sigma(base_sigma, field=field)
    scale = positive_sigma(noise_scale, field="noise_scale")
    return positive_sigma(sigma * scale, field=f"effective {field}")


def scada_noise_sigma(noise_scale: float = 1.0, *, nb: int = 14, nl: int = 20) -> np.ndarray:
    scale = positive_sigma(noise_scale, field="noise_scale")
    return np.r_[np.full(nb, 1e-3 * scale), np.full(2 * nb + 4 * nl, 1e-2 * scale)]


def add_scada_noise(
    clean: Sequence[float], rng: np.random.Generator, sigma_z: Sequence[float]
) -> list[float]:
    values = np.asarray(clean, dtype=float)
    sigma = np.asarray(sigma_z, dtype=float)
    if values.ndim != 1 or values.shape != sigma.shape or not np.all(np.isfinite(values)):
        raise ValueError("clean SCADA values and sigma_z must be matching finite vectors")
    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError("sigma_z must be finite and strictly positive")
    return (values + rng.normal(0.0, sigma)).tolist()


def add_voltage_phasor_noise(
    rows: Sequence[Mapping[str, Any]], rng: np.random.Generator, sigma_pu: float
) -> list[dict[str, Any]]:
    """Add independent Gaussian noise to each voltage's real/imaginary part."""
    sigma = positive_sigma(sigma_pu, field="three_phase_sigma")
    # Copy rows individually so aliased input dictionaries still receive one
    # independent sensor draw each rather than repeated noise on one object.
    noisy = [deepcopy(dict(row)) for row in rows]
    for row in noisy:
        magnitude = np.asarray(row["vln_pu"], dtype=float)
        angle = np.asarray(row["ang_deg"], dtype=float)
        if magnitude.shape != (3,) or angle.shape != (3,):
            raise ValueError("phase voltage telemetry must contain three phase phasors per bus")
        if not np.all(np.isfinite(magnitude)) or not np.all(np.isfinite(angle)):
            raise ValueError("phase voltage telemetry must be finite")
        phasors = magnitude * np.exp(1j * np.deg2rad(angle))
        phasors += rng.normal(0.0, sigma, 3) + 1j * rng.normal(0.0, sigma, 3)
        row["vln_pu"] = np.abs(phasors).tolist()
        row["ang_deg"] = np.rad2deg(np.angle(phasors)).tolist()
    return noisy


def generated_noise_contract(
    sigma_z: Sequence[float], *, noise_scale: float,
    three_phase_sigma: float | None = None,
    branch_current_sigma_pu: float | None = None,
) -> dict[str, Any]:
    """Declare actual applied sigmas equal to downstream estimator weights."""
    channels = {}
    for channel, sigma, representation in (
        ("scada", sigma_z, "scalar"),
        ("three_phase_voltages", three_phase_sigma, "complex_rectangular"),
        ("three_phase_branch_currents", branch_current_sigma_pu, "complex_rectangular"),
    ):
        if sigma is not None:
            channels[channel] = validate_noise_channel(
                channel=channel, role="sensor_observation", distribution="gaussian",
                applied_sigma=sigma, estimator_sigma=sigma, representation=representation,
                require_matched_gaussian=True,
            )
    return {
        "schema": "generated_sensor_noise_v1",
        "role": "sensor_observation",
        "distribution": "independent_gaussian",
        "noise_scale": positive_sigma(noise_scale, field="noise_scale"),
        "sigma_semantics": "per_scalar_or_per_real_imaginary_component",
        "channels": channels,
        "reference_fields_role": "noiseless_reference_for_offline_audit_only",
        "applied_sigmas_match_estimator": True,
        "population_calibration_verified": False,
    }


def _validate_declared_waveform_row(row: dict[str, Any]) -> dict[str, Any]:
    sigma = np.asarray(row.get("sigma_z"), dtype=float)
    observed = np.asarray(row.get("z_obs"), dtype=float)
    if sigma.shape != (122,) or observed.shape != sigma.shape or not np.all(np.isfinite(observed)):
        raise ValueError("waveform row requires 122 finite z_obs values and explicit sigma_z")
    if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
        raise ValueError("waveform row sigma_z must be finite and positive")
    voltage_sigma = positive_sigma(row.get("three_phase_sigma"), field="three_phase_sigma")
    has_currents = bool(row.get("three_phase_branch_currents"))
    current_sigma = (positive_sigma(row.get("branch_current_sigma_pu"), field="branch_current_sigma_pu")
                     if has_currents else None)
    current_sigmas = {
        "scada": sigma, "three_phase_voltages": voltage_sigma,
    }
    if has_currents:
        current_sigmas["three_phase_branch_currents"] = current_sigma
    contract = row.get("noise_contract")
    if contract is not None:
        if not isinstance(contract, Mapping) or contract.get("schema") != "generated_sensor_noise_v1":
            raise ValueError("unsupported waveform noise_contract schema")
        channels = contract.get("channels", {})
        for name, estimator_sigma in current_sigmas.items():
            declaration = channels.get(name)
            if not isinstance(declaration, Mapping):
                raise ValueError(f"noise_contract has no declaration for {name}")
            validate_noise_channel(
                channel=name, role=declaration.get("role"), distribution=declaration.get("distribution"),
                applied_sigma=declaration.get("applied_sigma_per_component"),
                estimator_sigma=estimator_sigma,
                representation="scalar" if name == "scada" else "complex_rectangular",
                require_matched_gaussian=True,
            )
    for scan in row.get("scans", []):
        scan_sigmas = [("sigma_z", sigma.tolist()), ("three_phase_sigma", voltage_sigma)]
        if has_currents:
            scan_sigmas.append(("branch_current_sigma_pu", current_sigma))
        for name, value in scan_sigmas:
            if name not in scan:
                scan[name] = deepcopy(value)
            if not np.allclose(np.asarray(scan[name]), np.asarray(value), rtol=1e-12, atol=0):
                raise ValueError(f"scan {name} differs from its root noise declaration")
        _validate_declared_waveform_row(scan)
    return row


def align_legacy_waveform_row(
    row: Mapping[str, Any], family: str, rng: np.random.Generator,
    *, legacy_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Add only noise missing from known historical generator schemas.

    This is dataset/scenario preparation, not an estimator operation. Existing
    SCADA/current observations are never denoised or resampled. Historical file
    metadata is required to establish their noise convention; unknown applied
    noise is rejected. Already explicit rows are copied and validated without
    drawing additional noise. Clean copies remain offline-audit fields.
    """
    aligned = deepcopy(dict(row))
    if "noise_contract" in aligned or "three_phase_sigma" in aligned:
        return _validate_declared_waveform_row(aligned)
    if not isinstance(legacy_metadata, Mapping):
        raise ValueError("legacy waveform noise alignment requires source meta.json evidence")
    normalized = str(family).lower()
    if normalized == "hif":
        hif_meta = legacy_metadata.get("hif")
        if not isinstance(hif_meta, Mapping):
            raise ValueError("legacy HIF source metadata must contain hif noise evidence")
        generation = hif_meta.get("generation", {})
        scale = positive_sigma(generation.get("noise_scale"), field="legacy HIF noise_scale")
        current_meta = hif_meta.get("branch_current_measurements", {})
        has_currents = bool(aligned.get("three_phase_branch_currents")) or any(
            scan.get("three_phase_branch_currents") for scan in aligned.get("scans", []))
        nominal_current = (positive_sigma(current_meta.get("branch_current_sigma_pu"), field="legacy current sigma")
                           if has_currents else None)
        nominal_sigma = np.asarray(aligned.get("sigma_z"), dtype=float)
        if nominal_sigma.shape != (122,) or not np.allclose(nominal_sigma, scada_noise_sigma(), rtol=1e-12, atol=0):
            raise ValueError("legacy HIF row does not match the known nominal SCADA-noise schema")
        if has_currents and not np.isclose(float(aligned.get("branch_current_sigma_pu", -1)), nominal_current, rtol=1e-12, atol=0):
            raise ValueError("legacy HIF current declaration differs from source metadata")
        scans = aligned.get("scans")
        if not isinstance(scans, list) or not scans:
            raise ValueError("known legacy HIF alignment requires its explicit scan window")
        sigma_z = scada_noise_sigma(scale).tolist()
        voltage_sigma = scaled_sensor_sigma(DEFAULT_THREE_PHASE_SIGMA_PU, scale, field="three_phase_sigma")
        current_sigma = (scaled_sensor_sigma(nominal_current, scale, field="branch_current_sigma_pu")
                         if has_currents else None)
        contract = generated_noise_contract(sigma_z, noise_scale=scale, three_phase_sigma=voltage_sigma,
                                            branch_current_sigma_pu=current_sigma)
        for scan in scans:
            if "three_phase_sigma" in scan or "noise_contract" in scan:
                raise ValueError("mixed explicit and legacy HIF scan noise declarations are unsupported")
            scan["three_phase_voltages_clean"] = deepcopy(scan["three_phase_voltages"])
            scan["three_phase_voltages"] = add_voltage_phasor_noise(scan["three_phase_voltages"], rng, voltage_sigma)
            scan.update(sigma_z=deepcopy(sigma_z), three_phase_sigma=voltage_sigma,
                        noise_contract=deepcopy(contract))
            if has_currents:
                if not scan.get("three_phase_branch_currents"):
                    raise ValueError("legacy HIF current channel must be present consistently across scans")
                scan["branch_current_sigma_pu"] = current_sigma
        first = scans[0]
        for key in ("z_obs", "z_clean", "three_phase_voltages", "three_phase_voltages_clean",
                    "three_phase_branch_currents", "three_phase_branch_currents_clean"):
            if key in first:
                aligned[key] = deepcopy(first[key])
        aligned.update(sigma_z=sigma_z, three_phase_sigma=voltage_sigma, noise_contract=contract)
        if has_currents:
            aligned["branch_current_sigma_pu"] = current_sigma
        aligned["noise_alignment"] = {
            "source": "legacy_HIF_generator_meta",
            "preserved_existing_noise": ["scada"] + (["three_phase_branch_currents"] if has_currents else []),
            "added_noise": ["three_phase_voltages"],
            "original_noise_scale": scale,
        }
    elif normalized in {"unbalance", "three_phase_unbalance", "three_phase_imbalance"}:
        meta = legacy_metadata.get("imbalance", {})
        current_meta = meta.get("three_phase_branch_current_measurements", {})
        applied = positive_sigma(current_meta.get("applied_noise_sigma_pu"), field="legacy applied current sigma")
        declared = positive_sigma(current_meta.get("branch_current_sigma_pu"), field="legacy declared current sigma")
        if not np.isclose(applied, declared, rtol=1e-12, atol=0):
            raise ValueError("legacy unbalance applied and declared current sigmas differ")
        if not np.isclose(float(aligned.get("branch_current_sigma_pu", -1)), declared, rtol=1e-12, atol=0):
            raise ValueError("legacy unbalance current declaration differs from source metadata")
        sigma_z = scada_noise_sigma().tolist()
        aligned["z_clean"] = deepcopy(aligned["z_obs"])
        aligned["z_obs"] = add_scada_noise(aligned["z_clean"], rng, sigma_z)
        aligned["three_phase_voltages_clean"] = deepcopy(aligned["three_phase_voltages"])
        aligned["three_phase_voltages"] = add_voltage_phasor_noise(
            aligned["three_phase_voltages_clean"], rng, DEFAULT_THREE_PHASE_SIGMA_PU)
        aligned.update(sigma_z=sigma_z, three_phase_sigma=DEFAULT_THREE_PHASE_SIGMA_PU,
                       noise_contract=generated_noise_contract(
                           sigma_z, noise_scale=1.0, three_phase_sigma=DEFAULT_THREE_PHASE_SIGMA_PU,
                           branch_current_sigma_pu=declared))
        aligned["noise_alignment"] = {
            "source": "legacy_unbalance_generator_meta",
            "preserved_existing_noise": ["three_phase_branch_currents"],
            "added_noise": ["scada", "three_phase_voltages"],
        }
    else:
        raise ValueError(f"unsupported legacy waveform family: {family!r}")
    return _validate_declared_waveform_row(aligned)


__all__ = [
    "DEFAULT_THREE_PHASE_SIGMA_PU", "positive_sigma", "scaled_sensor_sigma",
    "scada_noise_sigma", "add_scada_noise", "add_voltage_phasor_noise",
    "generated_noise_contract", "align_legacy_waveform_row",
]
