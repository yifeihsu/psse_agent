"""DAgger collection without provenance binding.

``psse_env.dagger.collect_dagger1`` wraps ``DaggerRolloutCollector`` in roughly
three thousand lines of provenance validation, holdout binding, root-support
floors and stopping contracts.  The collector itself is the science: it rolls
out the learner, queries the expert at every visited state, and labels the
recovery stratum.

This module calls the collector directly.  It still assembles the protected
root set, because that is contamination control rather than release
bookkeeping: collecting training data on a physical root reserved for
evaluation would silently invalidate every later comparison.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from psse_env.dagger.release_factories import production_environment_factory
from psse_env.dagger.rollout_collector import (
    DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION,
    DaggerRolloutCollector,
)
from psse_env.oracle import ExpertPolicyOracle

from .model import load_policy
from .train import write_jsonl

DEFAULT_MAX_STEPS = 24
DEFAULT_EVALUATION_SUITE = (
    Path(__file__).resolve().parents[1]
    / "psse_env"
    / "dagger"
    / "suites"
    / "bc0_eval_suite_v1.json"
)
ROOT_KEY = "physical_root_fingerprint"


def load_scenarios(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        for key in ("scenarios", "items", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return list(value)
        raise ValueError(f"{path} has no scenario list")
    if not isinstance(payload, list):
        raise ValueError(f"{path} is neither a list nor a scenario envelope")
    return list(payload)


def _harvest_roots(payload: Any, sink: set[str]) -> None:
    """Collect every physical root fingerprint anywhere in a payload.

    The three protected sources -- the round-0 aggregate, the frozen
    evaluation suite and the development holdout -- nest the fingerprint at
    different depths, so this walks rather than assuming a shape.
    """

    if isinstance(payload, Mapping):
        value = payload.get(ROOT_KEY)
        if isinstance(value, str) and value.strip():
            sink.add(value.strip())
        for item in payload.values():
            _harvest_roots(item, sink)
    elif isinstance(payload, list):
        for item in payload:
            _harvest_roots(item, sink)


def _roots_from_file(path: str | Path) -> set[str]:
    target = Path(path)
    roots: set[str] = set()
    if not target.is_file():
        return roots
    if target.suffix == ".jsonl":
        with open(target, encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if text:
                    _harvest_roots(json.loads(text), roots)
        return roots
    _harvest_roots(json.loads(target.read_text(encoding="utf-8")), roots)
    return roots


def protected_physical_roots(
    *,
    round0_dir: str | Path | None = None,
    evaluation_suite: str | Path | None = None,
    development_holdout: str | Path | None = None,
    extra: Iterable[str] = (),
) -> tuple[frozenset[str], dict[str, int]]:
    """Assemble the roots collection must not visit, with a per-source count."""

    breakdown: dict[str, int] = {}
    roots: set[str] = set()

    if round0_dir is not None:
        base = Path(round0_dir)
        round0: set[str] = set()
        for name in (
            "aggregate.raw.jsonl",
            "aggregate.train_view.jsonl",
            "aggregate.validation.jsonl",
            "aggregate.test.jsonl",
        ):
            round0 |= _roots_from_file(base / name)
        breakdown["round0"] = len(round0)
        roots |= round0

    suite = evaluation_suite if evaluation_suite is not None else DEFAULT_EVALUATION_SUITE
    suite_roots = _roots_from_file(suite)
    breakdown["evaluation_suite"] = len(suite_roots)
    roots |= suite_roots

    if development_holdout is not None:
        holdout_roots = _roots_from_file(development_holdout)
        breakdown["development_holdout"] = len(holdout_roots)
        roots |= holdout_roots

    additional = {str(root).strip() for root in extra if str(root).strip()}
    if additional:
        breakdown["extra"] = len(additional)
        roots |= additional

    breakdown["total"] = len(roots)
    return frozenset(roots), breakdown


def collect(
    *,
    scenarios_path: str | Path,
    output_path: str | Path,
    adapter_path: str | Path | None,
    beta: float,
    round0_dir: str | Path | None = None,
    evaluation_suite: str | Path | None = None,
    development_holdout: str | Path | None = None,
    iteration: int = 1,
    max_steps: int = DEFAULT_MAX_STEPS,
    model_id: str | None = None,
    revision: str | None = None,
    seed: int = 20260823,
) -> dict[str, Any]:
    """Roll out the learner and write expert-labelled rows.

    ``beta`` is the probability of deferring to the expert.  ``beta=0`` is a
    pure on-policy diagnostic pass; the DAgger-1 training regime mixes in the
    expert somewhere in ``[0.25, 0.5]``.
    """

    scenarios = load_scenarios(scenarios_path)
    forbidden, root_breakdown = protected_physical_roots(
        round0_dir=round0_dir,
        evaluation_suite=evaluation_suite,
        development_holdout=development_holdout,
    )
    if not forbidden:
        raise RuntimeError(
            "No protected physical roots were found; refusing to collect "
            "without contamination control."
        )

    policy = load_policy(
        model_id=model_id, revision=revision, adapter_path=adapter_path
    )
    env = production_environment_factory()
    # The expert must share the environment's own process and candidate
    # oracles.  A bare ExpertPolicyOracle() builds its own defaults, and the
    # default process oracle has executor_hydrated_corrections=False: it then
    # rejects every context-supported ``correct_parameters`` target as an
    # empty payload, leaving the expert nothing to propose and driving it into
    # a premature operator escalation the environment refuses to accept.
    expert = ExpertPolicyOracle(
        process_oracle=env.process_oracle,
        candidate_oracle=env.candidate_quality_oracle,
    )
    collector = DaggerRolloutCollector(
        env=env,
        policy=policy,
        expert_oracle=expert,
        rng=random.Random(seed),
        supervision_policy=DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION,
        forbidden_physical_roots=forbidden,
        stop_episode_on_unverified_expert_label=True,
    )
    role = "diagnostic" if float(beta) == 0.0 else "training"

    # Scenarios are collected one at a time so a single unrecoverable state
    # costs that scenario rather than the whole run.  Expert labels that the
    # environment's evidence contract rejects surface here; they are recorded
    # rather than hidden, because a silently shortened corpus would look like
    # a successful collection.
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, scenario in enumerate(scenarios):
        grouping = scenario.get("grouping")
        scenario_id = None
        if isinstance(grouping, Mapping):
            scenario_id = grouping.get("scenario_id")
        scenario_id = scenario_id or scenario.get("scenario_id") or f"index{index}"
        try:
            rows.extend(
                collector.collect_iteration(
                    scenarios=[scenario],
                    iteration=iteration,
                    beta=beta,
                    max_steps=max_steps,
                    collection_role=role,
                )
            )
        except Exception as exc:  # noqa: BLE001 - one scenario must not abort the run
            failures.append(
                {
                    "scenario_index": index,
                    "scenario_id": str(scenario_id),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    write_jsonl(output_path, rows)
    summary = summarize(rows, output_path=output_path, beta=beta)
    summary["protected_roots"] = root_breakdown
    summary["scenarios_attempted"] = len(scenarios)
    summary["scenarios_collected"] = len(scenarios) - len(failures)
    summary["scenario_failures"] = failures
    # Episodes cut short because the expert's label could not be justified from
    # observable evidence.  This is a property of the expert, not of the run,
    # and belongs in the report rather than in a log nobody reads.
    summary["unverified_expert_labels"] = list(collector.unverified_expert_labels)
    return summary


def summarize(
    rows: Sequence[dict[str, Any]],
    *,
    output_path: str | Path,
    beta: float,
) -> dict[str, Any]:
    """Report what the rollout actually did, not whether it is releasable."""

    from collections import Counter

    executed_tools: Counter[str] = Counter()
    strata: Counter[str] = Counter()
    classes: Counter[str] = Counter()
    episodes: set[str] = set()
    roots: set[str] = set()
    invalid_rows = 0
    for row in rows:
        action = row.get("executed_action") or {}
        tool = str(action.get("tool") or "unknown")
        executed_tools[tool] += 1
        if tool == "__invalid_action__":
            invalid_rows += 1
        strata[str(row.get("recovery_stratum") or "unclassified")] += 1
        classes[str(row.get("state_class") or "unknown")] += 1
        episode = row.get("episode_id")
        if episode:
            episodes.add(str(episode))
        root = row.get(ROOT_KEY)
        if isinstance(root, str) and root.strip():
            roots.add(root.strip())
    total = len(rows)
    return {
        "output": str(output_path),
        "beta": float(beta),
        "rows": total,
        "episodes": len(episodes),
        "distinct_physical_roots": len(roots),
        "invalid_action_rows": invalid_rows,
        "invalid_action_rate": (invalid_rows / total) if total else None,
        "executed_tools": dict(executed_tools.most_common()),
        "recovery_strata": dict(strata.most_common()),
        "state_classes": dict(classes.most_common()),
        "release_evidence": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--beta", type=float, default=0.3)
    parser.add_argument("--round0-dir", help="D0 aggregate directory (protected roots)")
    parser.add_argument("--evaluation-suite", help="Frozen suite (protected roots)")
    parser.add_argument("--development-holdout", help="Holdout (protected roots)")
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--model-id")
    parser.add_argument("--revision")
    parser.add_argument("--seed", type=int, default=20260823)
    args = parser.parse_args(argv)

    summary = collect(
        scenarios_path=args.scenarios,
        output_path=args.output,
        adapter_path=args.adapter,
        beta=args.beta,
        round0_dir=args.round0_dir,
        evaluation_suite=args.evaluation_suite,
        development_holdout=args.development_holdout,
        iteration=args.iteration,
        max_steps=args.max_steps,
        model_id=args.model_id,
        revision=args.revision,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
