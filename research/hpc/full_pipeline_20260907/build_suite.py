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
parameter-ranking stratum (``dominant``, ``ambiguous``, ``misranked``, or
``not_applicable``) so results can be read per stratum against the teacher's
own ceiling.

    build_suite.py --source-root SRC --d0-raw D0/aggregate.raw.jsonl \\
        --protected-suite SUITE [--protected-suite ...] \\
        --round-train-plan JSON --development-plan JSON --rounds 2 \\
        --seed N --candidate-multiplier 3 --output-dir OUT/suite \\
        [--training-threshold 1.2] [--development-threshold 1.0] \\
        [--development-rank-allowance 2]

The hard draw of docs/rl_hard_cases_plan_20261013.md (step 1) reverses the
admission filters: ``--admission-mode physical``, a large
``--training-rank-allowance`` / ``--development-rank-allowance`` so misranked
parameter roots are kept, ``--effect-band low,high`` in place of the
10-sigma meter floor, and a ``--development-stratum-plan`` /
``--training-stratum-plan`` (JSON ``{family: {stratum: count}}``) that draws
the parameter families by ranking stratum instead of by family count.
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
    # The script defines dataclasses under ``from __future__ import annotations``;
    # dataclasses resolves their string annotations through sys.modules, so the
    # module must be registered before it executes (the 2026-10-01 cell's
    # stage 0 died here on the ParallelEvaluation dataclass added 2026-09-27).
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(spec.name, None)
        raise
    return module


def parse_stratum_plan(value: str | None) -> dict[str, dict[str, int]]:
    """``{"parameter": {"misranked": 24, "ambiguous": 24}}`` or an empty plan."""

    if not value:
        return {}
    payload = json.loads(value)
    if not isinstance(payload, Mapping):
        raise ValueError("a stratum plan is a JSON object of family -> {stratum: count}")
    plan: dict[str, dict[str, int]] = {}
    for family, strata in payload.items():
        if not isinstance(strata, Mapping):
            raise ValueError(f"stratum plan for {family!r} must map stratum -> count")
        plan[str(family)] = {str(stratum): int(count) for stratum, count in strata.items() if int(count) > 0}
    return {family: strata for family, strata in plan.items() if strata}


def _row_stratum(row: Mapping[str, Any]) -> str | None:
    audit = row.get("audit") if isinstance(row.get("audit"), Mapping) else {}
    ranking = audit.get("parameter_ranking") if isinstance(audit.get("parameter_ranking"), Mapping) else {}
    value = ranking.get("stratum")
    return str(value) if value is not None else None


def allocate_with_strata(
    research,
    candidates: Sequence[Mapping[str, Any]],
    *,
    role: str,
    family_plan: Mapping[str, int],
    stratum_plan: Mapping[str, Mapping[str, int]],
    seed: int,
    d0_roots: set[str],
    protected_roots: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Allocate roots by family, and by ranking stratum for the families a stratum plan names.

    ``role`` is ``training`` or ``development``.  A family in the stratum plan
    is drawn stratum by stratum from the candidates (sorted by root, then
    shuffled with ``seed``); the other families go through
    ``research.allocate_scenarios`` as before.  Shortfalls are reported, not
    silently filled from other strata.
    """

    import random

    if role not in {"training", "development"}:
        raise ValueError("role must be training or development")
    excluded = set(d0_roots) | set(protected_roots)
    chosen: list[dict[str, Any]] = []
    report: dict[str, Any] = {"stratum_counts": {}, "stratum_shortfalls": {}}
    for family, strata in sorted(stratum_plan.items()):
        for stratum, count in sorted(strata.items()):
            pool = [
                dict(row)
                for row in candidates
                if research._scenario_family(row) == family
                and _row_stratum(row) == stratum
                and research._row_root(row) not in excluded
            ]
            pool.sort(key=research._row_root)
            random.Random(f"{seed}:{family}:{stratum}").shuffle(pool)
            taken = pool[: int(count)]
            report["stratum_counts"][f"{family}:{stratum}"] = len(taken)
            if len(taken) < int(count):
                report["stratum_shortfalls"][f"{family}:{stratum}"] = int(count) - len(taken)
            for row in taken:
                excluded.add(research._row_root(row))
                chosen.append(row)
    remaining_plan = {family: count for family, count in family_plan.items() if family not in stratum_plan}
    remaining_candidates = [row for row in candidates if research._scenario_family(row) not in stratum_plan]
    if remaining_plan:
        training, development = research.allocate_scenarios(
            remaining_candidates,
            d0_roots=set(excluded),
            train_plan=remaining_plan if role == "training" else {},
            development_plan=remaining_plan if role == "development" else {},
            seed=seed,
            protected_roots=protected_roots,
        )
        chosen.extend(dict(row) for row in (training if role == "training" else development))
    return chosen, report


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
    the true line ranks first but the ranking does not clear the threshold,
    so the context offers the top candidates and the teacher tests them in
    rank order.  ``misranked``: a neighbouring line outranks the true one;
    the teacher tests the neighbour first and, when that correction also
    fits the measurements, commits it and is wrong (the rule-based ceiling).
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
    first = rank is None or int(rank) == 1
    clears = singleton or (ratio_value is not None and ratio_value >= dominance_threshold)
    if not first:
        stratum = "misranked"
    elif clears:
        stratum = "dominant"
    else:
        stratum = "ambiguous"
    return {
        "stratum": stratum,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    # Literal choices: psse_env is importable only after --source-root is known.
    # They mirror psse_env.evidence_profile.EVIDENCE_PROFILES and its default.
    parser.add_argument("--evidence-profile",
                        choices=("scada_only", "wls_gated_diagnostics", "suspicion_gated_diagnostics",
                                 "classifier_gated_diagnostics", "auxiliary_diagnostics"),
                        default="suspicion_gated_diagnostics")
    parser.add_argument("--hif-signature-mode", choices=("discovered", "flagged"), default="discovered")
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
    parser.add_argument(
        "--training-rank-allowance",
        type=int,
        default=None,
        help="Admit training roots whose true line ranks within this many candidates (hard draw); omitted keeps rank one",
    )
    parser.add_argument(
        "--training-stratum-plan",
        default=None,
        help='JSON {family: {stratum: count}} drawing those training families by parameter-ranking stratum',
    )
    parser.add_argument(
        "--development-stratum-plan",
        default=None,
        help='JSON {family: {stratum: count}} drawing those development families by parameter-ranking stratum',
    )
    parser.add_argument(
        "--stratum-dominance-threshold",
        type=float,
        default=None,
        help=(
            "Dominance ratio below which a true-first parameter root is 'ambiguous' when strata are "
            "recorded and drawn; omitted uses --training-threshold (the deployed context's threshold "
            "in the standard pipeline).  A hard draw that admits training roots at 1.0 passes 1.2 here"
        ),
    )
    parser.add_argument(
        "--stratum-candidates",
        type=int,
        default=0,
        help="Generator candidates requested for each stratum-plan family (large enough to walk the whole pool)",
    )
    parser.add_argument(
        "--effect-band",
        default=None,
        help=(
            "Draw the synthesized harmonic roots' distortion by WLS effect ratio, 'low,high' (both "
            "draws); corpus-drawn families keep their corpus magnitudes"
        ),
    )
    parser.add_argument(
        "--min-measurement-error-sigma",
        type=float,
        default=None,
        help="Lift injected meter errors to at least this many noise sigmas (both draws)",
    )
    parser.add_argument(
        "--topology-effects",
        type=lambda value: tuple(item.strip() for item in str(value).split(",") if item.strip()),
        default=None,
        help="Comma-separated breaker-error classes of the topology family (both draws)",
    )
    parser.add_argument(
        "--system",
        default="case14",
        help="Registered system for both draws (case14, case57 or case118)",
    )
    parser.add_argument(
        "--measurement-corpus",
        type=Path,
        default=None,
        help="Fresh balanced corpus for the generator (required for a non-IEEE14 system)",
    )
    parser.add_argument(
        "--balanced-artifact-dir",
        type=Path,
        default=None,
        help="Artifact directory of the fresh balanced corpus",
    )
    parser.add_argument(
        "--admission-mode",
        choices=("recoverable", "physical"),
        default=None,
        help="Generator admission for both draws; omitted keeps the teacher-solvable default",
    )
    parser.add_argument(
        "--normalized-residual-threshold",
        type=float,
        default=None,
        help=(
            "Maximum-normalized-residual alarm paired with the chi-square test in "
            "admission (both draws); omitted uses the research runner's default"
        ),
    )
    parser.add_argument(
        "--chi-square-only",
        action="store_true",
        help="Disable the normalized-residual test in admission (pre-2026-09-14 rule)",
    )
    parser.add_argument(
        "--progress-basis",
        choices=("chi_square_statistic", "chi_square_excess"),
        default=None,
        help=(
            "Basis of a candidate's global progress in the environment the aggregate ran under "
            "(stage 0 passes the same system arguments to both draws); recorded in the manifest, "
            "the draws themselves run no episodes"
        ),
    )
    parser.add_argument("--hif-sample-paths", nargs="+", type=Path)
    parser.add_argument("--imbalance-sample-path", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    research = load_research_script(args.source_root.resolve())
    if args.chi_square_only:
        normalized_residual_threshold = None
    elif args.normalized_residual_threshold is None:
        normalized_residual_threshold = research.DEFAULT_NORMALIZED_RESIDUAL_THRESHOLD
    else:
        normalized_residual_threshold = float(args.normalized_residual_threshold)
    round_plan = _plan(args.round_train_plan)
    development_plan = _plan(args.development_plan)
    train_plan = {family: count * args.rounds for family, count in round_plan.items()}
    families = set(train_plan) | set(development_plan)

    d0_roots = research.load_d0_roots(args.d0_raw)
    protected = research.load_protected_suite_roots(args.protected_suite)
    protected_roots = set(protected["physical_roots"])
    sources = research.resolve_scenario_sources(
        plan_families=families,
        system=args.system,
        measurement_corpus=args.measurement_corpus,
        hif_sample_paths=args.hif_sample_paths,
        imbalance_sample_path=args.imbalance_sample_path,
        balanced_artifact_dir=args.balanced_artifact_dir,
        admission_mode=args.admission_mode,
        evidence_profile=args.evidence_profile,
        signature_modes={"hif": args.hif_signature_mode},
    )
    profile = {
        "plan_preset": "full_pipeline_suite",
        "hif_search_profile": "research",
        "evidence_profile": args.evidence_profile,
        "scenario_sources": sources,
        # The suite draw does not depend on the teacher, but the receipt
        # declares the variant the stage ran under (pipeline.env checks it
        # before reusing a stage, defaulting absent declarations to baseline).
        "expert_variant": research.current_expert_variant(),
    }

    # Training: the teacher-solvable population at the production threshold.
    stratum_threshold = float(
        args.training_threshold if args.stratum_dominance_threshold is None else args.stratum_dominance_threshold
    )
    training_stratum_plan = parse_stratum_plan(args.training_stratum_plan)
    development_stratum_plan = parse_stratum_plan(args.development_stratum_plan)
    train_generator = research.research_scenario_generator(
        seed=args.seed,
        research_profile=profile,
        parameter_ranking_dominance_threshold=args.training_threshold,
        parameter_target_rank_allowance=args.training_rank_allowance,
        min_measurement_error_sigma=args.min_measurement_error_sigma,
        topology_effects=args.topology_effects,
        normalized_residual_threshold=normalized_residual_threshold,
        effect_band=args.effect_band,
    )
    train_requested = {family: count * args.candidate_multiplier for family, count in train_plan.items()}
    for family, strata in training_stratum_plan.items():
        # A stratum draw needs the whole candidate pool of that family.
        train_requested[family] = max(train_requested.get(family, 0), sum(strata.values()) * args.candidate_multiplier,
                                      int(args.stratum_candidates))
    train_candidates = [
        stratified_envelope(research, row, dominance_threshold=stratum_threshold)
        for row in train_generator.build(train_requested)
    ]
    training, training_allocation = allocate_with_strata(
        research,
        train_candidates,
        role="training",
        family_plan=train_plan,
        stratum_plan=training_stratum_plan,
        seed=args.seed,
        d0_roots=d0_roots,
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
        min_measurement_error_sigma=args.min_measurement_error_sigma,
        topology_effects=args.topology_effects,
        normalized_residual_threshold=normalized_residual_threshold,
        effect_band=args.effect_band,
    )
    dev_requested = {
        family: count * args.candidate_multiplier for family, count in development_plan.items()
    }
    for family, strata in development_stratum_plan.items():
        dev_requested[family] = max(dev_requested.get(family, 0), sum(strata.values()) * args.candidate_multiplier,
                                    int(args.stratum_candidates))
    dev_candidates = [
        stratified_envelope(research, row, dominance_threshold=stratum_threshold)
        for row in dev_generator.build(dev_requested)
    ]
    development, development_allocation = allocate_with_strata(
        research,
        dev_candidates,
        role="development",
        family_plan=development_plan,
        stratum_plan=development_stratum_plan,
        seed=args.seed + 1,
        d0_roots=d0_roots | training_roots,
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
        "expert_variant": research.current_expert_variant(),
        "seed": int(args.seed),
        "rounds": int(args.rounds),
        "round_train_plan": round_plan,
        "development_plan": development_plan,
        "candidate_multiplier": int(args.candidate_multiplier),
        "training_threshold": float(args.training_threshold),
        "development_threshold": float(args.development_threshold),
        "development_rank_allowance": int(args.development_rank_allowance),
        "training_rank_allowance": args.training_rank_allowance,
        "training_stratum_plan": training_stratum_plan,
        "development_stratum_plan": development_stratum_plan,
        "training_allocation": training_allocation,
        "development_allocation": development_allocation,
        "effect_band": args.effect_band,
        "stratum_dominance_threshold": stratum_threshold,
        "stratum_candidates": int(args.stratum_candidates),
        "min_measurement_error_sigma": args.min_measurement_error_sigma,
        "topology_effects": list(args.topology_effects) if args.topology_effects else None,
        "system": str(args.system),
        "normalized_residual_threshold": normalized_residual_threshold,
        "progress_basis": args.progress_basis,
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
