#!/usr/bin/env python3
"""Roll fixed adapters (and the teacher) out on a saved suite, with no collection.

    evaluate_suite.py --cell-dir PIPE --suite SUITE.json --output-dir OUT --name NAME \\
        --policy LABEL=ADAPTER [--policy LABEL=ADAPTER] [--expert] --seed N \\
        [--max-steps 40] [--workers-per-policy N]

The DAgger runner evaluates a round's student and candidate on the round's
development suite after resuming its collection.  The conventional-only cell's
follow-ups (2026-10-23) evaluate other adapters on other suites: the matched
cloning control on the test roots, every iterate on the hard draw, and the
iterates on a validation set for the round selection.  This entry point calls
the runner's evaluation core (``evaluate_paired_adapters``) under the
environment options, model and teacher that the cell's round-1 report
recorded, so an episode here is the episode a round evaluation would run.  The
evaluator has two adapter slots (``bc0`` and ``r1``) plus the expert;
``policies.json`` maps each slot to its label.  A requeued job resumes from the
parallel workers' per-episode checkpoints.  Writes OUT/NAME/{slot}_eval.json,
comparison.json, policies.json, episodes.jsonl (one compact record per root and
policy) and summary.json (per-family outcomes and the policy_table metrics by
label).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
SLOTS = ("bc0", "r1")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(Path(name).stem, HERE / name)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_policies(values: Sequence[str]) -> list[tuple[str, Path]]:
    """``label=path`` pairs, one or two, distinct labels."""
    policies = []
    for value in values:
        label, sep, path = str(value).partition("=")
        if not sep or not label.strip() or not path.strip():
            raise ValueError(f"--policy takes LABEL=ADAPTER, got {value!r}")
        policies.append((label.strip(), Path(path.strip())))
    if not 1 <= len(policies) <= len(SLOTS):
        raise ValueError("one or two --policy values (the evaluator's two adapter slots)")
    if len({label for label, _ in policies}) != len(policies) or "expert" in {label for label, _ in policies}:
        raise ValueError("policy labels must be distinct and not 'expert'")
    return policies


def _root_record(episode: Mapping[str, Any], row: Mapping[str, Any] | None, label: str) -> dict[str, Any]:
    audit = (row or {}).get("audit") or {}
    ranking = audit.get("parameter_ranking") if isinstance(audit.get("parameter_ranking"), Mapping) else {}
    mimic = audit.get("mimic") if isinstance(audit.get("mimic"), Mapping) else {}
    grouping = (row or {}).get("grouping") or {}
    return {
        "policy": label,
        "scenario_id": episode.get("scenario_id"),
        "physical_root": episode.get("physical_root"),
        "family": grouping.get("scenario_family") or episode.get("family"),
        "stratum": ranking.get("stratum"),
        "mimic": mimic.get("variant"),
        "success": bool(episode.get("truth_audited_task_success")),
        "terminal_outcome": episode.get("terminal_outcome"),
        "steps": episode.get("steps"),
        "invalid_action_count": int(episode.get("invalid_action_count") or 0),
        "false_commit_count": int(episode.get("false_commit_count") or 0),
        "loop_detected": bool(episode.get("loop_detected")),
        "wls_calls": episode.get("wls_calls"),
        "comparisons": int((episode.get("tool_counts") or {}).get("compare_parameter_candidates") or 0),
    }


def summarize(evaluation_dir: Path, rows: Sequence[Mapping[str, Any]], labels: Mapping[str, str]) -> dict[str, Any]:
    """Per-family outcomes and the policy_table metrics of every evaluated slot, by label."""
    summarize_module = _load("summarize.py")
    policy_table = _load("policy_table.py")
    families = summarize_module.family_by_root(rows)
    strata = summarize_module.stratum_by_root(rows)
    present = sorted(set(families.values()))
    by_root = {summarize_module._row_root(row): row for row in rows}
    summary: dict[str, Any] = {"families": present, "roots": len(rows), "policies": {}}
    records = []
    for slot, label in labels.items():
        path = evaluation_dir / f"{slot}_eval.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        block = summarize_module.per_family_outcomes(payload, families, strata)
        summary["policies"][label] = {
            "slot": slot,
            "per_family": block,
            "success_by_family": summarize_module.success_table(block),
            "success_by_stratum": summarize_module.stratum_table(block),
            "metrics": policy_table.policy_metrics(block["outcomes"], present),
        }
        for episode in summarize_module._episodes(payload):
            records.append(_root_record(episode, by_root.get(str(episode.get("physical_root") or "")), label))
    (evaluation_dir / "episodes.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell-dir", required=True, type=Path, help="the cell whose round-1 report fixes the environment")
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--policy", action="append", default=[], help="LABEL=ADAPTER, once or twice")
    parser.add_argument("--expert", action="store_true", help="also roll the cell's teacher out")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--workers-per-policy", type=int, default=0, help="0: sequential in this process")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    policies = parse_policies(args.policy)
    cell = args.cell_dir.resolve()
    source = cell / "source"
    sys.path.insert(0, str(source if (source / "scripts" / "run_dagger_research.py").is_file() else HERE.parents[2]))
    import scripts.run_dagger_research as research
    from psse_env.dagger import research_policy_factory as loaders
    from psse_env.dagger.evaluator import evaluate_rollout_suites
    from psse_env.dagger.release_factories import checkpoint_tree_sha256
    from psse_env.oracle.expert_variants import current_expert_variant

    report = json.loads((cell / "out" / "r1" / "collection" / "research_run_report.json").read_text(encoding="utf-8"))
    profile = report["research_profile"]
    variant = current_expert_variant()
    if variant != profile.get("expert_variant"):
        raise SystemExit(f"PSSE_EXPERT_VARIANT={variant} differs from the cell's teacher {profile.get('expert_variant')}")
    research.RESEARCH_ENVIRONMENT_OPTIONS.update(dict(report["environment_options"]))
    research.RESEARCH_EXPERT_OPTIONS["variant"] = variant
    factory = research.resolve_environment_factory(profile["hif_search_profile"], profile["evidence_profile"])
    model = report["model_selection"]
    rows, suite_info = research.load_scenario_suite(args.suite)
    adapters = [(label, path.expanduser().resolve(strict=True)) for label, path in policies]
    labels = {slot: label for slot, (label, _) in zip(SLOTS, adapters)}
    if args.expert:
        labels["expert"] = "expert"
    out = args.output_dir.resolve()
    evaluation_dir = out / args.name
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "contract": "conventional_only_suite_evaluation_v1",
        "suite": suite_info,
        "seed": int(args.seed),
        "max_steps": int(args.max_steps),
        "policies": {
            slot: {"label": label, "adapter": str(path), "tree_sha256": checkpoint_tree_sha256(path)}
            for slot, (label, path) in zip(SLOTS, adapters)
        },
        "expert": variant if args.expert else None,
        "research_profile": profile,
        "environment_options": dict(research.RESEARCH_ENVIRONMENT_OPTIONS),
        "model_selection": model,
        "cell": str(cell),
    }
    (evaluation_dir / "policies.json").write_text(json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    expert_factory = (lambda: research.research_expert_policy(factory)) if args.expert else None
    parallel = None
    if args.workers_per_policy > 0:
        parallel = research.ParallelEvaluation(
            workers_per_policy=int(args.workers_per_policy),
            environment={
                "hif_search_profile": profile["hif_search_profile"],
                "evidence_profile": profile["evidence_profile"],
                "options": dict(research.RESEARCH_ENVIRONMENT_OPTIONS),
            },
        )
    research.evaluate_paired_adapters(
        development_scenarios=rows,
        bc0_adapter=adapters[0][1],
        r1_adapter=adapters[1][1] if len(adapters) > 1 else None,
        base_model=model["model_id"],
        base_revision=model["revision"],
        output_dir=out,
        seed=int(args.seed),
        max_steps=int(args.max_steps),
        # Looked up at call time, so a test can substitute a light loader.
        policy_loader=lambda adapter, **kwargs: loaders.research_gemma_policy_factory(adapter, **kwargs),
        environment_factory=factory,
        evaluator=evaluate_rollout_suites,
        prompt_profile=model.get("prompt_profile"),
        architecture=model.get("architecture"),
        policy_cache_clear=loaders.clear_research_policy_cache,
        expert_policy_factory=expert_factory,
        evaluation_dirname=args.name,
        parallel=parallel,
    )
    summary = summarize(evaluation_dir, rows, labels)
    summary["evaluation"] = identity
    (evaluation_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({label: {k: block["metrics"][k] for k in ("episodes", "successes", "invalid_action_episodes",
                                                              "false_commit_episodes", "loop_episodes")}
                      for label, block in summary["policies"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
