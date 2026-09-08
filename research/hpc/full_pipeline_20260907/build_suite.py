#!/usr/bin/env python3
"""Draw the DAgger suite: per-round training roots and one shared development set.

The suite is drawn once, after the expert aggregate exists, with the research
generator (train partition, branch-current corpora, discovered harmonic and
unbalance signatures) and excludes every D0 root and every protected root.

Training roots are drawn at the production parameter-ranking threshold, so
every root the students learn from is one the teacher can correct on the
first pass, and are dealt across the DAgger rounds family by family so the
rounds roll out on disjoint roots.  The development set is drawn at the
detection threshold with a rank allowance, so it keeps the adjacent-line
ambiguity the network really has; each development root records its
parameter-ranking stratum (``dominant``, ``ambiguous``, or
``not_applicable``) so results can be read per stratum against the teacher's
own ceiling.

    build_suite.py --source-root SRC --d0-raw D0/aggregate.raw.jsonl \\
        --protected-suite SUITE [--protected-suite ...] \\
        --round-train-plan JSON --development-plan JSON --rounds 2 \\
        --seed N --candidate-multiplier 3 --output-dir OUT/suite \\
        [--training-threshold 1.2] [--development-threshold 1.0] \\
        [--development-rank-allowance 2]
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

PARAMETER_FAMILIES = ("parameter", "measurement+parameter")
DEFAULT_TRAINING_THRESHOLD = 1.2
DEFAULT_DEVELOPMENT_THRESHOLD = 1.0
DEFAULT_DEVELOPMENT_RANK_ALLOWANCE = 2


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


def parameter_ranking_stratum(
    row: Mapping[str, Any], *, family: str, dominance_threshold: float
) -> dict[str, Any]:
    """Classify a root by what the deployed parameter context saw on it.

    ``dominant``: the true line ranks first and clears the dominance
    threshold, so the teacher corrects it on the first pass.  ``ambiguous``:
    the true line is among the ranked candidates but the ranking does not
    clear the threshold or the true line is not first, so the teacher tests
    the candidates in rank order and may hand off bounded to them.
    ``not_applicable``: the family carries no parameter fault.
    """

    if family not in PARAMETER_FAMILIES:
        return {"stratum": "not_applicable", "dominance_ratio": None, "true_line_rank": None}
    ranking = row.get("parameter_ranking")
    if ranking is None and isinstance(row.get("audit"), Mapping):
        ranking = row["audit"].get("parameter_ranking")
    ranking = ranking if isinstance(ranking, Mapping) else {}
    ratio = ranking.get("parameter_ranking_dominance_ratio")
    rank = ranking.get("true_line_rank")
    singleton = ranking.get("parameter_ranking_singleton") is True
    try:
        ratio_value = float(ratio) if ratio is not None else None
    except (TypeError, ValueError):
        ratio_value = None
    dominant = bool(
        (rank is None or int(rank) == 1)
        and (singleton or (ratio_value is not None and ratio_value >= dominance_threshold))
    )
    return {
        "stratum": "dominant" if dominant else "ambiguous",
        "dominance_ratio": ratio_value,
        "true_line_rank": rank,
    }


def stratified_envelope(research, row: Mapping[str, Any], *, dominance_threshold: float) -> dict[str, Any]:
    """Partition a generator row and record its stratum under ``audit``."""

    envelope = research.partition_release_scenario_v1(row, split="dagger_train")
    family = str(row.get("scenario_family") or "")
    info = parameter_ranking_stratum(row, family=family, dominance_threshold=dominance_threshold)
    ranking = row.get("parameter_ranking")
    envelope["audit"]["parameter_ranking"] = {
        **info,
        "generation": dict(ranking) if isinstance(ranking, Mapping) else None,
    }
    return envelope


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
    parser.add_argument("--training-threshold", type=float, default=DEFAULT_TRAINING_THRESHOLD)
    parser.add_argument("--development-threshold", type=float, default=DEFAULT_DEVELOPMENT_THRESHOLD)
    parser.add_argument(
        "--development-rank-allowance", type=int, default=DEFAULT_DEVELOPMENT_RANK_ALLOWANCE
    )
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

    # Training: the teacher-solvable population at the production threshold.
    train_generator = research.research_scenario_generator(
        seed=args.seed,
        research_profile=profile,
        parameter_ranking_dominance_threshold=args.training_threshold,
    )
    train_requested = {family: count * args.candidate_multiplier for family, count in train_plan.items()}
    train_candidates = [
        stratified_envelope(research, row, dominance_threshold=args.training_threshold)
        for row in train_generator.build(train_requested)
    ]
    training, _ = research.allocate_scenarios(
        train_candidates,
        d0_roots=d0_roots,
        train_plan=train_plan,
        development_plan={},
        seed=args.seed,
        protected_roots=protected_roots,
    )
    training_roots = {research._row_root(row) for row in training}

    # Development: the realistic population at the detection threshold, with
    # the true line allowed anywhere within the ambiguity allowance.
    dev_generator = research.research_scenario_generator(
        seed=args.seed + 1,
        research_profile=profile,
        parameter_ranking_dominance_threshold=args.development_threshold,
        parameter_target_rank_allowance=args.development_rank_allowance,
    )
    dev_requested = {
        family: count * args.candidate_multiplier for family, count in development_plan.items()
    }
    dev_candidates = [
        stratified_envelope(research, row, dominance_threshold=args.training_threshold)
        for row in dev_generator.build(dev_requested)
    ]
    _, development = research.allocate_scenarios(
        dev_candidates,
        d0_roots=d0_roots | training_roots,
        train_plan={},
        development_plan=development_plan,
        seed=args.seed + 1,
        protected_roots=protected_roots,
    )
    strata: dict[str, dict[str, Any]] = {}
    for row in development:
        info = dict(row["audit"]["parameter_ranking"])
        info.pop("generation", None)
        strata[research._row_root(row)] = {"family": research._scenario_family(row), **info}

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
    stratum_counts: dict[str, Counter] = defaultdict(Counter)
    for info in strata.values():
        stratum_counts[info["family"]][info["stratum"]] += 1
    written["development"] = {
        "path": str(development_path),
        "rows": len(development),
        "sha256": _sha256(development_path),
        "families": dict(sorted(Counter(research._scenario_family(r) for r in development).items())),
        "strata": {family: dict(counts) for family, counts in sorted(stratum_counts.items())},
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
        "contract": "research_full_pipeline_suite_v2",
        "seed": int(args.seed),
        "rounds": int(args.rounds),
        "round_train_plan": round_plan,
        "development_plan": development_plan,
        "candidate_multiplier": int(args.candidate_multiplier),
        "training_threshold": float(args.training_threshold),
        "development_threshold": float(args.development_threshold),
        "development_rank_allowance": int(args.development_rank_allowance),
        "training_candidates_built": len(train_candidates),
        "development_candidates_built": len(dev_candidates),
        "training_generator_report": train_generator.report(),
        "development_generator_report": dev_generator.report(),
        "research_profile": profile,
        "d0_raw": {"path": str(args.d0_raw.resolve()), "roots": len(d0_roots)},
        "protected_roots": len(protected_roots),
        "files": written,
        "development_strata": strata,
        "physical_root_count": len(all_roots),
        "physical_roots_sha256": hashlib.sha256(
            "\n".join(sorted(all_roots)).encode("utf-8")
        ).hexdigest(),
        "source_commit": source.get("source_commit"),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v["rows"] for k, v in written.items()}
            | {"strata": written["development"]["strata"]},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
