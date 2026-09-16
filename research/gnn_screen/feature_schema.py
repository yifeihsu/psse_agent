"""Versioned, asset-independent WLS-only feature contract.

Powers use system-base per unit and the terminal convention is power entering
the branch. Vm retains the exporter's phase-A magnitude convention; it is not
a measured positive-sequence voltage. Bus/branch IDs are metadata only.
"""
from __future__ import annotations

SCHEMA_VERSION = "wls_screen_v1"
MEASUREMENT_CONVENTION = "phase_a_vm_total_3ph_power_net_injection_excludes_shunts"
FAMILY_NAMES = ("hif", "unbalance", "measurement", "parameter", "topology")
MEASUREMENT_TYPES = ("vm", "pinj", "qinj", "pf", "qf", "pt", "qt")
PACKET_FIELDS = ("observed", "fitted", "signed_normalized_residual",
                 "log_relative_sigma", "leverage", "available", "residual_usable")
NODE_FEATURES = tuple(f"{kind}.{field}" for kind in MEASUREMENT_TYPES[:3]
                      for field in PACKET_FIELDS) + (
    "bus_type_pq", "bus_type_pv", "bus_type_reference", "g_shunt_pu",
    "b_shunt_pu", "log1p_registered_degree",
)
EDGE_FEATURES = tuple(f"{kind}.{field}" for kind in
                      ("p_source", "q_source", "p_destination", "q_destination")
                      for field in PACKET_FIELDS) + (
    "r_pu", "x_pu", "b_pu", "native_tap_ratio", "cos_native_shift",
    "sin_native_shift", "configured_status", "is_line", "is_transformer",
    "native_orientation", "cos_estimated_angle_difference",
    "sin_estimated_angle_difference",
)
GLOBAL_FEATURES = ("objective_per_dof", "max_abs_normalized_residual",
                   "fraction_abs_normalized_residual_gt3",
                   "fraction_abs_normalized_residual_gt4")
NODE_DIM, EDGE_DIM, GLOBAL_DIM = len(NODE_FEATURES), len(EDGE_FEATURES), len(GLOBAL_FEATURES)


class ScreenInputError(ValueError):
    """An unavailable screen is never converted into a negative screen."""

    def __init__(self, message: str, status: str = "unsupported_input") -> None:
        if status not in {"unsupported_input", "wls_failure", "inadequate_observability"}:
            raise ValueError(f"Unknown screen status: {status}")
        super().__init__(message)
        self.status = status
