"""Strict, certificate-bound logical switch action contract (no row aliases)."""
from __future__ import annotations

from collections.abc import Mapping
import re


def validate_atomic_arguments(arguments, *, state_key="state_id"):
    required = {state_key, "candidate_id", "certificate_hash", "desired_statuses"}
    if not isinstance(arguments, Mapping) or set(arguments) != required:
        raise ValueError("logical correction requires state, candidate, certificate hash, and desired statuses only")
    for key in (state_key, "candidate_id"):
        if not isinstance(arguments[key], str) or not arguments[key]:
            raise ValueError(f"{key} must be a nonempty string")
    if not isinstance(arguments["certificate_hash"], str) or not re.fullmatch(r"[0-9a-f]{64}", arguments["certificate_hash"]):
        raise ValueError("certificate_hash must be a SHA-256 hexadecimal digest")
    changes = arguments["desired_statuses"]
    if not isinstance(changes, Mapping) or not changes:
        raise ValueError("desired_statuses must be a nonempty logical-device map")
    for device, status in changes.items():
        if not isinstance(device, str) or not device or type(status) is not int or status not in (0, 1):
            raise ValueError("logical desired statuses require stable string IDs and integer 0/1 values")

