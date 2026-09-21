"""Adapter layer for IEEE-14 three-phase HIF data generation and tracing.

Units: the OpenDSS model is normalized (1 kV, 0.01 ohm base); ``r_hif_ohm``
passed to the injector and the legacy NLM is always normalized-model ohms.
Physical ohms on the faulted line's local 69 / 13.8 / 18 kV base are handled
by :mod:`three_phase_nlm.hif_units`, whose public helpers are re-exported here.
"""

from .dss_hif_injector import (
    HIFInjectionResult,
    copy_ieee14_model,
    hif_ohms_from_pu,
    inject_midspan_hif_ieee14,
    write_balanced_ieee14_load_override,
)
from .hif_parameter_estimator import estimate_hif_location_magnitude, simulate_hif_candidate
from .hif_multiscan_estimator import estimate_hif_location_magnitude_multiscan
from .branch_current_analysis import (
    terminal_current_hif_localization,
    two_terminal_hif_estimate,
    unbalance_source_localization,
)
from .hif_units import (
    PHYSICAL_ELIGIBLE_HIF_BRANCHES,
    hif_resistance_record,
    label_model_ohm,
    label_physical_ohm,
    line_kv_ll_for_row0,
    resolve_line_kv_ll,
    resolve_resistance_search_box,
)
from .nlm_runner import run_ieee14_hif_nlm

__all__ = [
    "HIFInjectionResult",
    "PHYSICAL_ELIGIBLE_HIF_BRANCHES",
    "copy_ieee14_model",
    "estimate_hif_location_magnitude",
    "estimate_hif_location_magnitude_multiscan",
    "hif_ohms_from_pu",
    "hif_resistance_record",
    "inject_midspan_hif_ieee14",
    "label_model_ohm",
    "label_physical_ohm",
    "line_kv_ll_for_row0",
    "resolve_line_kv_ll",
    "resolve_resistance_search_box",
    "run_ieee14_hif_nlm",
    "simulate_hif_candidate",
    "terminal_current_hif_localization",
    "two_terminal_hif_estimate",
    "unbalance_source_localization",
    "write_balanced_ieee14_load_override",
]
