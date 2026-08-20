"""One reader for every artifact shape that names physical roots.

The DAgger-1 artifacts spell their root sets three different ways: scenario
generation emits an envelope *list*, evaluation and development suites emit a
*mapping* of suite name to rows, and aggregates emit *JSONL*.  Each consumer
that grew its own reader has been a latent failure: the probe stage accepted a
list and would have raised on the real development holdout, which is a suite
mapping, before any disjointness check could run.

This module fails closed.  An unrecognised structure, a missing fingerprint, or
an unversioned fingerprint is an error, never an empty set -- an empty set would
silently disable a disjointness guard.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

#: Physical roots are versioned; an unversioned value is a different identity
#: space and must never be compared against these sets.  A prefix check alone
#: would admit a truncated or differently-hashed value, so the whole identity
#: format is required: ``physical_v<version>_<64 hex>``.
PHYSICAL_ROOT_PREFIX = "physical_v"
PHYSICAL_ROOT_PATTERN = re.compile(r"^physical_v[0-9]+_[0-9a-f]{64}$")

ROOT_SET_CONTRACT = "dagger1_physical_root_set_reader_v1"


def _fingerprint(value: Any, *, source: str) -> str:
    root = str(value or "").strip()
    if not root:
        raise ValueError(f"{source}: row is missing a physical root fingerprint")
    if not PHYSICAL_ROOT_PATTERN.match(root):
        raise ValueError(
            f"{source}: physical root {root!r} is not a valid versioned "
            "fingerprint (expected physical_v<version>_<64 hex>)"
        )
    return root


def _row_root(row: Any, *, source: str) -> str:
    if not isinstance(row, Mapping):
        raise ValueError(f"{source}: expected an object row")
    grouping = row.get("grouping")
    holder = grouping if isinstance(grouping, Mapping) else row
    return _fingerprint(holder.get("physical_root_fingerprint"), source=source)


def _roots_from_rows(rows: Any, *, source: str) -> set[str]:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ValueError(f"{source}: expected a list of rows")
    return {_row_root(row, source=source) for row in rows}


def physical_roots_from_artifact(path: str | Path) -> set[str]:
    """Physical roots named by a scenario list, suite mapping, or JSONL file.

    Raises rather than returning an empty set: every caller uses the result to
    exclude roots, so a silent empty set would widen the candidate pool instead
    of narrowing it.
    """

    path = Path(path)
    source = str(path)
    if not path.is_file():
        raise FileNotFoundError(f"{source}: root-set artifact does not exist")

    if path.suffix == ".jsonl":
        roots: set[str] = set()
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                roots.add(
                    _row_root(json.loads(line), source=f"{source}:{number}")
                )
        if not roots:
            raise ValueError(f"{source}: JSONL artifact names no physical roots")
        return roots

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        roots = _roots_from_rows(payload, source=source)
    elif isinstance(payload, Mapping):
        # Suite mapping: suite name -> rows.  Every value must be a row list, so
        # a metadata-carrying mapping cannot masquerade as a suite.
        if not payload:
            raise ValueError(f"{source}: suite mapping is empty")
        roots = set()
        for name, rows in payload.items():
            roots |= _roots_from_rows(rows, source=f"{source}[{name}]")
    else:
        raise ValueError(
            f"{source}: unsupported root-set structure {type(payload).__name__}"
        )
    if not roots:
        raise ValueError(f"{source}: artifact names no physical roots")
    return roots


def root_set_digest(roots: set[str]) -> str:
    """Stable digest of a root set, for provenance binding."""

    import hashlib

    payload = json.dumps(sorted(roots), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
