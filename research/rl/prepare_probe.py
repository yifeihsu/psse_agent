"""Freeze a 32-root *training* probe and check its source suite against held-out roots.

This selects existing data, never generates new scenarios or submits a job.
All stratum/truth metadata stays in the offline envelope, as in the source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from psse_env.rl.train import load_suite
from research.rl.hard_draw_report import root_stratum

QUOTAS = {"misranked": 8, "same_sign": 8, "opposite_sign": 8, "controls": 8}


def identities(row: Mapping[str, Any]) -> set[tuple[str, str]]:
    grouping = row.get("grouping") or {}
    result = {("root", str(grouping.get("root_scenario_id") or row["execution"]["scenario_id"]))}
    for key in ("physical_root_fingerprint", "source_realization_id"):
        if grouping.get(key):
            result.add((key, str(grouping[key])))
    return result


def bucket(row: Mapping[str, Any]) -> str:
    label = root_stratum(row)
    if label.endswith(":misranked"):
        return "misranked"
    if label.endswith("flow_pair_same_sign"):
        return "same_sign"
    if label.endswith("flow_pair_opposite_sign"):
        return "opposite_sign"
    return "controls"


def select_probe(training: Sequence[Mapping[str, Any]], held_out: Sequence[Mapping[str, Any]], *,
                 seed: int = 20261016, quotas: Mapping[str, int] = QUOTAS) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    train_ids = set().union(*(identities(row) for row in training))
    protected_ids = set().union(*(identities(row) for row in held_out))
    overlap = train_ids & protected_ids
    if overlap:
        raise ValueError(f"training/held-out overlap: {sorted(overlap)[:8]}")
    roots = [str(row["execution"]["scenario_id"]) for row in training]
    if len(set(roots)) != len(roots):
        raise ValueError("duplicate training scenario IDs")
    rng = random.Random(seed)
    chosen = []
    for name, required in quotas.items():
        candidates = [row for row in training if bucket(row) == name]
        rng.shuffle(candidates)
        if len(candidates) < required:
            raise ValueError(f"{name}: need {required} training roots, found {len(candidates)}; do not borrow development roots")
        chosen.extend(dict(row) for row in candidates[:required])
    rng.shuffle(chosen)
    coverage = {key: sum(bool((row.get("grouping") or {}).get(key)) for row in [*training, *held_out])
                for key in ("physical_root_fingerprint", "source_realization_id")}
    total = len(training) + len(held_out)
    return chosen, {
        "seed": seed, "quotas": dict(quotas), "training_roots": len(training), "held_out_roots": len(held_out),
        "selected_roots": [row["execution"]["scenario_id"] for row in chosen],
        "selected_strata": dict(Counter(root_stratum(row) for row in chosen)),
        "identity_coverage": coverage,
        "parent_disjointness_fully_checked": coverage["source_realization_id"] == total and total > 0,
        "note": "No overlap among available identifiers. Missing source parents require a separate lineage audit.",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-suite", required=True)
    parser.add_argument("--held-out-suite", required=True, action="append", help="Repeat for hard development and protected old suites")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise SystemExit("probe already exists; use its frozen selection or a new output path")
    training = load_suite(args.training_suite)
    held_out = [row for path in args.held_out_suite for row in load_suite(path)]
    chosen, manifest = select_probe(training, held_out, seed=args.seed)
    manifest["source_sha256"] = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                 for path in [args.training_suite, *args.held_out_suite]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(chosen, indent=1, sort_keys=True) + "\n")
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
