"""Adapter layer for IEEE-14 three-phase HIF data generation and tracing."""

from .dss_hif_injector import (
    HIFInjectionResult,
    copy_ieee14_model,
    hif_ohms_from_pu,
    inject_midspan_hif_ieee14,
    write_balanced_ieee14_load_override,
)
from .hif_parameter_estimator import estimate_hif_location_magnitude, simulate_hif_candidate
from .hif_multiscan_estimator import estimate_hif_location_magnitude_multiscan
from .nlm_runner import run_ieee14_hif_nlm

__all__ = [
    "HIFInjectionResult",
    "copy_ieee14_model",
    "estimate_hif_location_magnitude",
    "estimate_hif_location_magnitude_multiscan",
    "hif_ohms_from_pu",
    "inject_midspan_hif_ieee14",
    "run_ieee14_hif_nlm",
    "simulate_hif_candidate",
    "write_balanced_ieee14_load_override",
]
