"""Run one full DAgger iteration end to end.

    BC0            train on the round-0 aggregate
    collect        roll BC0 out on fresh scenarios, expert-label every state
    aggregate      concatenate round-0 and round-1 rows
    round1         retrain on the aggregate
    evaluate       compare BC0 and round-1 on held-out scenarios

Each stage writes its artifacts and can be skipped when they already exist, so
a stage that fails does not cost the stages before it.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Sequence

from .train import read_jsonl, write_jsonl

STAGES = ("bc0", "collect", "aggregate", "round1", "evaluate")


def build_aggregate(
    *,
    round0_train: str | Path,
    round0_validation: str | Path,
    round1_rows: str | Path,
    output_dir: str | Path,
    validation_fraction: float = 0.2,
    inclusion: str = "selective",
    train_stratum_upweight: dict[str, int] | None = None,
    seed: int = 20260823,
) -> dict[str, Any]:
    """Concatenate round-0 and round-1 rows into a new training view.

    Round-1 rows are split by *episode* rather than by row: rows from one
    episode share a scenario, so splitting rows independently would leak the
    same physical situation across the train and validation sides.
    """

    from psse_env.dagger.dataset_builder import examples_to_chat_sft

    base_train = read_jsonl(round0_train)
    base_validation = read_jsonl(round0_validation)

    # Collector rows carry the raw observation and expert target.  The round-0
    # corpus is chat-format SFT rows.  Export the new rows through the same
    # converter the release aggregate uses so both sides share one schema and
    # one protocol; concatenating the two shapes directly would produce rows
    # that no trainer can render.
    raw_rows = read_jsonl(round1_rows)
    # Supervision retention must never depend on whether the LEARNER's action
    # was well-formed: a state the learner visited and failed in is the
    # canonical DAgger state.  The earlier filter here censored exactly those
    # rows (41 across both iterations, several carrying commit/rollback
    # targets); the learner action stays in the row as diagnostic metadata but
    # plays no role in retention.
    supervisable = [row for row in raw_rows if row.get("preferred_action")]
    audit_failed = [
        row
        for row in supervisable
        if not (row.get("offline_teacher_target_audit") or {}).get("passed")
    ]
    audited = [
        row
        for row in supervisable
        if (row.get("offline_teacher_target_audit") or {}).get("passed")
    ]
    ineligibility: dict[str, int] = {}
    for row in audited:
        if not row.get("production_label_eligible"):
            reason = str(row.get("production_label_ineligibility_reason") or "unknown")
            ineligibility[reason] = ineligibility.get(reason, 0) + 1
    if inclusion == "selective":
        # Recovery-selective, truth-audited DAgger: the production pipeline's
        # own eligibility flag decides, uncensored.
        selected = [row for row in audited if row.get("production_label_eligible")]
    elif inclusion == "full_occupancy":
        # Standard-DAgger occupancy: every state visited under the mixture
        # policy that carries an audited expert target, including
        # expert-visited and non-recovery states.
        selected = audited
    else:
        raise ValueError(
            f"inclusion must be 'selective' or 'full_occupancy', got {inclusion!r}"
        )
    # Audit-failed states are recorded as a teacher-abstention stratum rather
    # than silently disappearing: retention conditioned on hidden truth is
    # selection leakage and must at least be measurable.
    new_rows = examples_to_chat_sft(
        selected,
        protocol="canonical",
        require_derived_provenance=False,
        allow_ineligible_auxiliary=inclusion == "full_occupancy",
    )

    by_episode: dict[str, list[dict[str, Any]]] = {}
    for row in new_rows:
        # The exported schema keeps scenario identity rather than episode id,
        # so fall through the available grouping keys.  Grouping everything
        # under one fallback key would defeat the held-out split entirely.
        group = (
            row.get("episode_id")
            or row.get("root_scenario_id")
            or row.get("scenario_id")
            or "ungrouped"
        )
        by_episode.setdefault(str(group), []).append(row)
    episodes = sorted(by_episode)
    random.Random(seed).shuffle(episodes)
    held_out = set(episodes[: max(1, int(len(episodes) * validation_fraction))])

    added_train = [
        row
        for episode in episodes
        if episode not in held_out
        for row in by_episode[episode]
    ]
    added_validation = [
        row for episode in held_out for row in by_episode[episode]
    ]

    # Duplicate discipline-stratum rows on the train side only: the failure
    # analysis showed the model ignoring observable escalation preconditions
    # and closed-candidate references, and the rows teaching those rules are
    # outnumbered roughly 20:1 in a uniform aggregate.  Validation stays
    # unweighted so eval losses remain comparable across runs.
    upweight_counts: dict[str, int] = {}
    if train_stratum_upweight:
        weighted: list[dict[str, Any]] = []
        for row in added_train:
            weighted.append(row)
            stratum = str(row.get("recovery_stratum") or "")
            factor = int(train_stratum_upweight.get(stratum, 1))
            if factor > 1:
                upweight_counts[stratum] = upweight_counts.get(stratum, 0) + 1
                weighted.extend(dict(row) for _ in range(factor - 1))
        added_train = weighted

    destination = Path(output_dir)
    train_path = destination / "aggregate.train.jsonl"
    validation_path = destination / "aggregate.validation.jsonl"
    write_jsonl(train_path, [*base_train, *added_train])
    write_jsonl(validation_path, [*base_validation, *added_validation])
    if audit_failed:
        write_jsonl(destination / "teacher_abstentions.jsonl", audit_failed)

    return {
        "train": str(train_path),
        "validation": str(validation_path),
        "round0_train_rows": len(base_train),
        "round0_validation_rows": len(base_validation),
        "round1_rows_collected": len(raw_rows),
        "inclusion": inclusion,
        "supervisable_rows": len(supervisable),
        "teacher_abstention_rows": len(audit_failed),
        "censored_by_old_rule": sum(
            1
            for row in selected
            if (row.get("executed_action") or {}).get("tool") == "__invalid_action__"
        ),
        "round1_ineligible_breakdown": ineligibility,
        "round1_rows_available": len(new_rows),
        "train_stratum_upweight": dict(train_stratum_upweight or {}),
        "upweighted_source_rows_by_stratum": upweight_counts,
        "round1_rows_added_to_train": len(added_train),
        "round1_rows_added_to_validation": len(added_validation),
        "round1_episodes": len(episodes),
        "round1_validation_episodes": sorted(held_out),
        "train_rows": len(base_train) + len(added_train),
        "validation_rows": len(base_validation) + len(added_validation),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round0-dir", required=True, help="D0 aggregate directory")
    parser.add_argument("--scenarios", required=True, help="Round-1 scenarios")
    parser.add_argument("--eval-scenarios", help="Held-out scenarios for evaluation")
    parser.add_argument(
        "--evaluation-suite",
        help="Frozen evaluation suite contributing protected physical roots",
    )
    parser.add_argument(
        "--development-holdout",
        help="Development holdout contributing protected physical roots",
    )
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--model-id")
    parser.add_argument("--revision")
    parser.add_argument("--bc0-adapter", help="Reuse an existing BC0 adapter")
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument(
        "--stages",
        default=",".join(STAGES),
        help=f"Comma-separated subset of {','.join(STAGES)}",
    )
    args = parser.parse_args(argv)

    stages = [stage.strip() for stage in args.stages.split(",") if stage.strip()]
    unknown = [stage for stage in stages if stage not in STAGES]
    if unknown:
        parser.error(f"unknown stages: {', '.join(unknown)}")

    round0 = Path(args.round0_dir)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    bc0_dir = Path(args.bc0_adapter) if args.bc0_adapter else work / "bc0"
    round1_rows = work / "round1_rows.jsonl"
    aggregate_dir = work / "aggregate"
    round1_dir = work / "round1"
    report: dict[str, Any] = {"work_dir": str(work), "stages": stages}

    if "bc0" in stages:
        from .train import train

        report["bc0"] = train(
            train_path=round0 / "aggregate.train_view.jsonl",
            validation_path=round0 / "aggregate.validation.jsonl",
            output_dir=bc0_dir,
            model_id=args.model_id,
            revision=args.revision,
            max_length=args.max_length,
            epochs=args.epochs,
            seed=args.seed,
        )

    if "collect" in stages:
        from .collect import collect

        report["collect"] = collect(
            scenarios_path=args.scenarios,
            output_path=round1_rows,
            adapter_path=bc0_dir,
            beta=args.beta,
            round0_dir=round0,
            evaluation_suite=args.evaluation_suite,
            development_holdout=args.development_holdout,
            max_steps=args.max_steps,
            model_id=args.model_id,
            revision=args.revision,
            seed=args.seed,
        )

    if "aggregate" in stages:
        report["aggregate"] = build_aggregate(
            round0_train=round0 / "aggregate.train_view.jsonl",
            round0_validation=round0 / "aggregate.validation.jsonl",
            round1_rows=round1_rows,
            output_dir=aggregate_dir,
            seed=args.seed,
        )

    if "round1" in stages:
        from .train import train

        report["round1"] = train(
            train_path=aggregate_dir / "aggregate.train.jsonl",
            validation_path=aggregate_dir / "aggregate.validation.jsonl",
            output_dir=round1_dir,
            model_id=args.model_id,
            revision=args.revision,
            max_length=args.max_length,
            epochs=args.epochs,
            seed=args.seed,
        )

    if "evaluate" in stages:
        from .evaluate import evaluate

        target = args.eval_scenarios or args.scenarios
        report["evaluate"] = {
            "scenarios": str(target),
            "bc0": evaluate(
                scenarios_path=target,
                adapter_path=bc0_dir,
                label="bc0",
                max_steps=args.max_steps,
                model_id=args.model_id,
                revision=args.revision,
            ),
            "round1": evaluate(
                scenarios_path=target,
                adapter_path=round1_dir,
                label="round1",
                max_steps=args.max_steps,
                model_id=args.model_id,
                revision=args.revision,
            ),
        }

    report["release_evidence"] = False
    rendered = json.dumps(report, indent=2, sort_keys=True)
    (work / "dagger_report.json").write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
