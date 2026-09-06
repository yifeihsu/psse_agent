#!/usr/bin/env python3
"""Draw the DAgger suite: per-round training roots and one shared development set.

The suite is drawn once, after the expert aggregate exists, with the research
generator (train partition, branch-current corpora, discovered harmonic and
unbalance signatures) and excludes every D0 root and every protected root.
Training roots are split across the DAgger rounds family by family so the
rounds roll out on disjoint roots while the development set stays the same
for every evaluation.

    build_suite.py --source-root SRC --d0-raw D0/aggregate.raw.jsonl \
        --protected-suite SUITE [--protected-suite ...] \
        --round-train-plan JSON --development-plan JSON --rounds 2 \
        --seed N --candidate-multiplier 3 --output-dir OUT/suite
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


def load_research_script(source_root: Path):
    path = source_root / "scripts" / "run_dagger_research.py"
    spec = importlib.util.spec_from_file_location("run_dagger_research", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.path.insert(0, str(source_root))
    spec.loader.exec_module(module)
    return module


def split_rounds(
    training: Sequence[Mapping[str, Any]],
    *,
    rounds: int,
    family_of,
    root_of,
) -> list[list[dict[str, Any]]]:
    """Deal each family's roots across the rounds in sorted-root order."""

    if rounds < 1:
        raise ValueError("rounds must be positive")
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in training:
        by_family[family_of(row)].append(dict(row))
    dealt: list[list[dict[str, Any]]] = [[] for _ in range(rounds)]
    for family in sorted(by_family):
        rows = sorted(by_family[family], key=root_of)
        for index, row in enumerate(rows):
            dealt[index % rounds].append(row)
    for bucket in dealt:
        bucket.sort(key=lambda row: (family_of(row), root_of(row)))
    return dealt


def _plan(value: str) -> dict[str, int]:
    candidate = Path(value)
    try:
        is_file = candidate.is_file()
    except (OSError, ValueError):
        is_file = False
    payload = json.loads(candidate.read_text(encoding="utf-8")) if is_file else json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("plan must be a JSON object")
    return {str(k): int(v) for k, v in sorted(payload.items())}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--d0-raw", required=True, type=Path)
    parser.add_argument("--protected-suite", action="append", default=[], type=Path)
    parser.add_argument("--round-train-plan", required=True)
    parser.add_argument("--development-plan", required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--candidate-multiplier", type=int, default=3)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    research = load_research_script(args.source_root.resolve())
    round_plan = _plan(args.round_train_plan)
    development_plan = _plan(args.development_plan)
    train_plan = {family: count * args.rounds for family, count in round_plan.items()}
    families = set(train_plan) | set(development_plan)

    d0_roots = research.load_d0_roots(args.d0_raw)
    protected = research.load_protected_suite_roots(args.protected_suite)
    protected_roots = set(protected["physical_roots"])
    sources = research.resolve_scenario_sources(plan_families=families)
    profile = {
        "plan_preset": "full_pipeline_suite",
        "hif_search_profile": "research",
        "scenario_sources": sources,
    }
    requested = {
        family: (train_plan.get(family, 0) + development_plan.get(family, 0)) * args.candidate_multiplier
        for family in families
    }
    generator = research.research_scenario_generator(seed=args.seed, research_profile=profile)
    candidates = [
        research.partition_release_scenario_v1(row, split="dagger_train")
        for row in generator.build(requested)
    ]
    training, development = research.allocate_scenarios(
        candidates,
        d0_roots=d0_roots,
        train_plan=train_plan,
        development_plan=development_plan,
        seed=args.seed,
        protected_roots=protected_roots,
    )
    per_round = split_rounds(
        training,
        rounds=args.rounds,
        family_of=research._scenario_family,
        root_of=research._row_root,
    )

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    written: dict[str, dict[str, Any]] = {}
    for index, rows in enumerate(per_round, start=1):
        path = output / f"r{index}_training.json"
        path.write_text(json.dumps(rows, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        written[f"r{index}_training"] = {
            "path": str(path),
            "rows": len(rows),
            "sha256": _sha256(path),
            "families": dict(sorted(Counter(research._scenario_family(r) for r in rows).items())),
        }
    development_path = output / "development.json"
    development_path.write_text(
        json.dumps(development, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    written["development"] = {
        "path": str(development_path),
        "rows": len(development),
        "sha256": _sha256(development_path),
        "families": dict(sorted(Counter(research._scenario_family(r) for r in development).items())),
    }
    all_roots = [research._row_root(r) for bucket in per_round for r in bucket] + [
        research._row_root(r) for r in development
    ]
    if len(set(all_roots)) != len(all_roots):
        raise RuntimeError("suite roots are not unique across rounds and development")
    if set(all_roots) & (d0_roots | protected_roots):
        raise RuntimeError("suite roots overlap D0 or protected roots")
    source = research.git_source_state(args.source_root.resolve())
    manifest = {
        "contract": "research_full_pipeline_suite_v1",
        "seed": int(args.seed),
        "rounds": int(args.rounds),
        "round_train_plan": round_plan,
        "development_plan": development_plan,
        "candidate_multiplier": int(args.candidate_multiplier),
        "candidates_built": len(candidates),
        "generator_report": generator.report(),
        "research_profile": profile,
        "d0_raw": {"path": str(args.d0_raw.resolve()), "roots": len(d0_roots)},
        "protected_roots": len(protected_roots),
        "files": written,
        "physical_root_count": len(all_roots),
        "physical_roots_sha256": hashlib.sha256(
            "\n".join(sorted(all_roots)).encode("utf-8")
        ).hexdigest(),
        "source_commit": source.get("source_commit"),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v["rows"] for k, v in written.items()} | {"families": {k: v["families"] for k, v in written.items()}}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
