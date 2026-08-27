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

from .train import file_sha256, read_jsonl, write_jsonl

STAGES = ("bc0", "collect", "aggregate", "round1", "evaluate")
INCLUSION_MODES = ("selective", "learner_full", "full_occupancy")


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

    def state_origin_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in rows:
            origin = str(row.get("state_origin") or "unknown")
            counts[origin] = counts.get(origin, 0) + 1
        return dict(sorted(counts.items()))

    audited_origins = state_origin_counts(audited)
    allowed_origins = {"initial", "learner_policy", "expert_policy"}
    unexpected_origins = sorted(set(audited_origins) - allowed_origins)
    if unexpected_origins:
        raise ValueError(
            "audited rows contain missing or unexpected state_origin values: "
            + ", ".join(unexpected_origins)
        )

    ineligibility: dict[str, int] = {}
    for row in audited:
        if not row.get("production_label_eligible"):
            reason = str(row.get("production_label_ineligibility_reason") or "unknown")
            ineligibility[reason] = ineligibility.get(reason, 0) + 1
    if inclusion == "selective":
        # Recovery-selective, truth-audited DAgger: the production pipeline's
        # own eligibility flag decides, uncensored.
        selected = [row for row in audited if row.get("production_label_eligible")]
    elif inclusion == "learner_full":
        # Learner-full occupancy: retain every audited state reached by a
        # learner action, whether or not it is a recovery state.  In
        # particular, learner action validity is not a selection condition;
        # invalid actions are exactly the failures DAgger must supervise.
        # Initial and expert-visited states remain outside this arm.
        selected = [
            row for row in audited if row.get("state_origin") == "learner_policy"
        ]
    elif inclusion == "full_occupancy":
        # Standard-DAgger occupancy: every state visited under the mixture
        # policy that carries an audited expert target, including
        # expert-visited and non-recovery states.
        selected = audited
    else:
        raise ValueError(
            f"inclusion must be one of {INCLUSION_MODES}, got {inclusion!r}"
        )
    # Audit-failed states are recorded as a teacher-abstention stratum rather
    # than silently disappearing: retention conditioned on hidden truth is
    # selection leakage and must at least be measurable.
    new_rows = examples_to_chat_sft(
        selected,
        protocol="canonical",
        require_derived_provenance=False,
        allow_ineligible_auxiliary=inclusion in {"learner_full", "full_occupancy"},
    )
    selected_ids = {id(row) for row in selected}
    excluded = [row for row in audited if id(row) not in selected_ids]
    retained_invalid_learner_actions = sum(
        1
        for row in selected
        if (row.get("executed_action") or {}).get("tool")
        == "__invalid_action__"
    )

    def episode_key(row: dict[str, Any]) -> str:
        # The exported schema keeps scenario identity rather than episode id,
        # so prefer the stable physical-root keys shared by raw and exported
        # rows.  A collection-only episode id may disappear during conversion
        # and therefore cannot define the cross-arm split universe.
        group = (
            row.get("root_scenario_id")
            or row.get("scenario_id")
            or row.get("episode_id")
        )
        if group is None or not str(group).strip():
            raise ValueError(
                "Round-1 row has no root_scenario_id, scenario_id, or episode_id; "
                "refusing an ambiguous train/validation split"
            )
        return str(group)

    # Freeze the episode assignment on the complete audited occupancy before
    # applying an inclusion rule.  Deriving it from ``selected`` made each arm
    # shuffle a different episode universe, confounding occupancy with which
    # physical roots entered training.
    split_universe_episodes = sorted({episode_key(row) for row in audited})
    random.Random(seed).shuffle(split_universe_episodes)
    held_out = set(
        split_universe_episodes[
            : max(1, int(len(split_universe_episodes) * validation_fraction))
        ]
    )

    by_episode: dict[str, list[dict[str, Any]]] = {}
    for row in new_rows:
        by_episode.setdefault(episode_key(row), []).append(row)
    selected_episodes = sorted(by_episode)

    added_train = [
        row
        for episode in selected_episodes
        if episode not in held_out
        for row in by_episode[episode]
    ]
    added_validation = [
        row
        for episode in selected_episodes
        if episode in held_out
        for row in by_episode[episode]
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
    abstentions_path = destination / "teacher_abstentions.jsonl"
    write_jsonl(train_path, [*base_train, *added_train])
    write_jsonl(validation_path, [*base_validation, *added_validation])
    write_jsonl(abstentions_path, audit_failed)

    return {
        "train": str(train_path),
        "train_sha256": file_sha256(train_path),
        "validation": str(validation_path),
        "validation_sha256": file_sha256(validation_path),
        "teacher_abstentions": str(abstentions_path),
        "teacher_abstentions_sha256": file_sha256(abstentions_path),
        "round0_train": str(round0_train),
        "round0_train_sha256": file_sha256(round0_train),
        "round0_validation": str(round0_validation),
        "round0_validation_sha256": file_sha256(round0_validation),
        "round1_rows": str(round1_rows),
        "round1_rows_sha256": file_sha256(round1_rows),
        "round0_train_rows": len(base_train),
        "round0_validation_rows": len(base_validation),
        "round1_rows_collected": len(raw_rows),
        "inclusion": inclusion,
        "supervisable_rows": len(supervisable),
        "teacher_abstention_rows": len(audit_failed),
        "round1_audited_rows": len(audited),
        "round1_rows_selected": len(selected),
        "round1_audited_state_origin_breakdown": audited_origins,
        "round1_selected_state_origin_breakdown": state_origin_counts(selected),
        "round1_excluded_state_origin_breakdown": state_origin_counts(excluded),
        "invalid_learner_action_rows_retained": retained_invalid_learner_actions,
        # Compatibility with the first uncensoring receipts.  The value counts
        # rows the old rule would have censored; current code does not censor
        # them.
        "censored_by_old_rule": retained_invalid_learner_actions,
        "round1_ineligible_breakdown": ineligibility,
        "round1_rows_available": len(new_rows),
        "train_stratum_upweight": dict(train_stratum_upweight or {}),
        "upweighted_source_rows_by_stratum": upweight_counts,
        "round1_rows_added_to_train": len(added_train),
        "round1_rows_added_to_validation": len(added_validation),
        "round1_episodes": len(selected_episodes),
        "round1_split_universe": "all_audited_rows_before_inclusion",
        "round1_split_seed": seed,
        "round1_validation_fraction": validation_fraction,
        "round1_split_universe_episodes": len(split_universe_episodes),
        "round1_split_assignment_validation_episodes": sorted(held_out),
        "round1_validation_episodes": sorted(set(selected_episodes) & held_out),
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
    parser.add_argument(
        "--inclusion",
        choices=INCLUSION_MODES,
        default="selective",
        help="Round-1 occupancy included in the aggregate",
    )
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=24,
        help="Collection/evaluation episode horizon (not optimizer updates)",
    )
    parser.add_argument(
        "--train-max-steps",
        type=int,
        default=-1,
        help="Optimizer-update budget for BC0 and Round-1; -1 trains by epochs",
    )
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
            max_steps=args.train_max_steps,
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
            inclusion=args.inclusion,
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
            max_steps=args.train_max_steps,
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
