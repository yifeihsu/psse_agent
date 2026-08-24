#!/usr/bin/env python3
"""Minimal, resumable research DAgger collection and mixture builder.

The release scenario/collection/aggregate CLIs intentionally remain outside
this path.  Scientific boundaries (observable teacher, root isolation, target
audit, canonical export) stay fail-closed while release hashes and quotas do
not participate in execution.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psse_env.actions import INVALID_ACTION  # noqa: E402
from psse_env.dagger.dataset_builder import (  # noqa: E402
    examples_to_chat_sft,
    load_jsonl,
    validate_policy_payload,
)
from psse_env.dagger.rollout_collector import (  # noqa: E402
    DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION,
    DaggerRolloutCollector,
)
from psse_env.dagger.suite_builder import partition_release_scenario_v1  # noqa: E402
from psse_env.oracle.expert_policy import ExpertPolicyOracle  # noqa: E402
from psse_env.providers.scenario_generator import Round0ScenarioGenerator  # noqa: E402
from psse_env.research_models import (  # noqa: E402
    DEFAULT_RESEARCH_MODEL,
    RESEARCH_MODEL_SPECS,
    SUPPORTED_RESEARCH_PROMPT_PROFILES,
    get_research_model_spec,
    resolve_research_model_spec,
)
from psse_env.sft.provenance import git_source_state  # noqa: E402


RESEARCH_CONTRACT = "research_dagger_minimal_v1"
RESEARCH_LABEL_CONTRACT = "observable_learner_state_research_v1"
DEFAULT_TRAIN_PLAN = {
    "measurement+parameter": 12,
    "multi_measurement": 12,
    "parameter": 6,
}
DEFAULT_DEVELOPMENT_PLAN = {
    "measurement+parameter": 6,
    "multi_measurement": 6,
    "parameter": 3,
}


def _stable_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    text = "".join(
        json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
        for row in materialized
    )
    _atomic_text(path, text)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _row_root(row: Mapping[str, Any]) -> str:
    for container in (
        row,
        row.get("grouping") if isinstance(row.get("grouping"), Mapping) else {},
        row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {},
    ):
        root = str(container.get("physical_root_fingerprint") or "").strip()
        if root:
            return root
    return ""


def _scenario_family(scenario: Mapping[str, Any]) -> str:
    grouping = scenario.get("grouping")
    grouping = grouping if isinstance(grouping, Mapping) else scenario
    return str(grouping.get("scenario_family") or "").strip()


def _root_seed(seed: int, root: str) -> int:
    digest = hashlib.sha256(f"{int(seed)}:{root}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _eligibility_reasons(row: Mapping[str, Any]) -> list[str]:
    labels = row.get("labels")
    labels = labels if isinstance(labels, Mapping) else {}
    rank_one = row.get("observable_rank_one_target_proof")
    rank_one = rank_one if isinstance(rank_one, Mapping) else {}
    offline_audit = row.get("offline_teacher_target_audit")
    offline_audit = offline_audit if isinstance(offline_audit, Mapping) else {}
    reasons: list[str] = []
    if row.get("state_origin") != "learner_policy":
        reasons.append("not_learner_policy_state")
    if row.get("preferred_action") is None:
        reasons.append("missing_expert_target")
    if labels.get("training_decision_evidence_verified") is not True:
        reasons.append("training_decision_evidence_not_verified")
    if rank_one.get("passed") is not True:
        reasons.append("observable_rank_one_proof_failed")
    if offline_audit.get("passed") is not True:
        reasons.append("offline_teacher_target_audit_failed")
    return reasons


def is_research_dagger_row(row: Mapping[str, Any]) -> bool:
    """Admit safe learner-state labels without requiring a named stratum."""

    return not _eligibility_reasons(row)


def mark_research_label_eligibility(
    row: Mapping[str, Any], *, extra_reason: str | None = None
) -> dict[str, Any]:
    stamped = copy.deepcopy(dict(row))
    reasons = _eligibility_reasons(stamped)
    if extra_reason:
        reasons.append(str(extra_reason))
    reasons = list(dict.fromkeys(reasons))
    eligible = not reasons
    stamped["research_label_eligible"] = eligible
    stamped["research_label_contract"] = RESEARCH_LABEL_CONTRACT
    labels = stamped.get("labels")
    labels = copy.deepcopy(dict(labels)) if isinstance(labels, Mapping) else {}
    labels["research_label_eligible"] = eligible
    labels["research_label_contract"] = RESEARCH_LABEL_CONTRACT
    stamped["labels"] = labels
    stamped["training_decision_evidence_verified"] = labels.get(
        "training_decision_evidence_verified"
    )
    if reasons:
        stamped["research_label_ineligibility_reasons"] = reasons
        labels["research_label_ineligibility_reasons"] = list(reasons)
    else:
        stamped.pop("research_label_ineligibility_reasons", None)
        labels.pop("research_label_ineligibility_reasons", None)
    return stamped


def load_d0_roots(path: str | Path) -> set[str]:
    rows = load_jsonl(path)
    rootless = [
        str(row.get("example_id") or index)
        for index, row in enumerate(rows)
        if not _row_root(row)
    ]
    if rootless:
        raise ValueError(
            "Every D0 raw row must retain physical_root_fingerprint for root "
            f"isolation; missing examples: {rootless[:8]}"
        )
    roots = {_row_root(row) for row in rows}
    if not roots:
        raise ValueError(f"D0 raw dataset is empty: {path}")
    return roots


def validate_d0_training_roots(
    rows: Sequence[Mapping[str, Any]], *, raw_roots: set[str]
) -> set[str]:
    rootless = [
        str(row.get("example_id") or index)
        for index, row in enumerate(rows)
        if not _row_root(row)
    ]
    if rootless:
        raise ValueError(
            "Every D0 training row must retain physical_root_fingerprint; "
            f"missing examples: {rootless[:8]}"
        )
    roots = {_row_root(row) for row in rows}
    unexpected = roots - set(raw_roots)
    if unexpected:
        raise ValueError(
            "D0 training rows contain physical roots absent from --d0-raw: "
            f"{sorted(unexpected)[:8]}"
        )
    return roots


def refresh_d0_training_view(
    raw_rows: Sequence[Mapping[str, Any]],
    selected_chat_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Re-export the selected D0 IDs with the current canonical prompt view.

    This prevents old pre-rendered D0 rows and freshly collected D1 rows from
    silently training on different observable-state compaction contracts.
    """

    by_id: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(raw_rows):
        example_id = str(row.get("example_id") or "").strip()
        if not example_id:
            raise ValueError(f"D0 raw row {index} has no example_id")
        if example_id in by_id:
            raise ValueError(f"D0 raw dataset repeats example_id {example_id!r}")
        by_id[example_id] = row

    refreshed: list[dict[str, Any]] = []
    changed = 0
    for index, prior in enumerate(selected_chat_rows):
        example_id = str(prior.get("example_id") or "").strip()
        if not example_id or example_id not in by_id:
            raise ValueError(
                f"D0 training row {index} cannot be rebound to --d0-raw by example_id"
            )
        exported = examples_to_chat_sft(
            [by_id[example_id]],
            protocol="canonical",
            allow_ineligible_auxiliary=True,
        )
        if len(exported) != 1:
            raise RuntimeError(f"D0 raw example {example_id!r} did not export exactly once")
        current = exported[0]
        if _row_root(current) != _row_root(prior):
            raise ValueError(f"D0 root changed while refreshing example {example_id!r}")
        prior_view = _stable_json([prior.get("messages"), prior.get("tools")])
        current_view = _stable_json([current.get("messages"), current.get("tools")])
        changed += prior_view != current_view
        refreshed.append(current)
    return refreshed, {
        "contract": "canonical_current_source_observation_view_v1",
        "selected_rows": len(selected_chat_rows),
        "rerendered_rows": len(refreshed),
        "rows_changed_from_input_view": changed,
    }


def allocate_scenarios(
    candidates: Sequence[Mapping[str, Any]],
    *,
    d0_roots: set[str],
    train_plan: Mapping[str, int],
    development_plan: Mapping[str, int],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_family: dict[str, list[dict[str, Any]]] = {
        family: [] for family in set(train_plan) | set(development_plan)
    }
    root_owner: dict[str, str] = {}
    for candidate in candidates:
        envelope = copy.deepcopy(dict(candidate))
        root = _row_root(envelope)
        family = _scenario_family(envelope)
        if not root or family not in by_family or root in d0_roots:
            continue
        previous = root_owner.get(root)
        if previous is not None:
            if previous != family:
                raise ValueError(
                    f"Generated physical root {root!r} appears in two families"
                )
            continue
        root_owner[root] = family
        by_family[family].append(envelope)

    training: list[dict[str, Any]] = []
    development: list[dict[str, Any]] = []
    for family in sorted(by_family):
        pool = by_family[family]
        random.Random(_root_seed(seed, family)).shuffle(pool)
        train_count = int(train_plan.get(family, 0))
        dev_count = int(development_plan.get(family, 0))
        if train_count < 0 or dev_count < 0:
            raise ValueError("Scenario allocation counts cannot be negative")
        if len(pool) < train_count + dev_count:
            raise ValueError(
                f"Scenario family {family!r} has {len(pool)} fresh roots; "
                f"needs {train_count + dev_count}. Increase --candidate-multiplier."
            )
        family_train = pool[:train_count]
        family_dev = pool[train_count : train_count + dev_count]
        for scenario in family_train:
            scenario["grouping"]["split"] = "dagger_train"
        for scenario in family_dev:
            scenario["grouping"]["split"] = "development"
        training.extend(family_train)
        development.extend(family_dev)

    training.sort(key=lambda row: (_scenario_family(row), _row_root(row)))
    development.sort(key=lambda row: (_scenario_family(row), _row_root(row)))
    train_roots = {_row_root(row) for row in training}
    development_roots = {_row_root(row) for row in development}
    if train_roots & development_roots:
        raise ValueError("D1 training and development physical roots overlap")
    if (train_roots | development_roots) & d0_roots:
        raise ValueError("D1 scenarios overlap D0 physical roots")
    return training, development


def _normalize_plan(value: Mapping[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for family, count in value.items():
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("Scenario plans require non-negative integer counts")
        if count:
            result[str(family)] = count
    if not result:
        raise ValueError("Scenario plan must allocate at least one root")
    return dict(sorted(result.items()))


def _parse_plan(value: str, default: Mapping[str, int]) -> dict[str, int]:
    if not value:
        return dict(default)
    candidate = Path(value)
    payload = (
        json.loads(candidate.read_text(encoding="utf-8"))
        if candidate.is_file()
        else json.loads(value)
    )
    if not isinstance(payload, Mapping):
        raise ValueError("Scenario plan must be a JSON object or JSON file")
    return _normalize_plan(payload)


def prepare_scenario_split(
    *,
    output_dir: Path,
    d0_raw_path: Path,
    d0_roots: set[str],
    train_plan: Mapping[str, int],
    development_plan: Mapping[str, int],
    candidate_multiplier: int,
    seed: int,
    run_descriptor: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_path = output_dir / "training_scenarios.json"
    development_path = output_dir / "development_scenarios.json"
    config_path = output_dir / "config.json"
    existing = [path.exists() for path in (train_path, development_path, config_path)]
    if any(existing) and not all(existing):
        raise RuntimeError("Research scenario/config files are only partially present")
    if all(existing):
        config = _read_json(config_path)
        expected = {
            "contract": RESEARCH_CONTRACT,
            "seed": int(seed),
            "train_plan": dict(train_plan),
            "development_plan": dict(development_plan),
            "d0_raw_path": str(d0_raw_path.resolve()),
            "run_descriptor": dict(run_descriptor),
        }
        mismatches = [key for key, value in expected.items() if config.get(key) != value]
        if mismatches:
            raise RuntimeError(
                "Existing research run configuration differs on: "
                + ", ".join(mismatches)
            )
        training = _read_json(train_path)
        development = _read_json(development_path)
    else:
        if candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be positive")
        requested = {
            family: (int(train_plan.get(family, 0)) + int(development_plan.get(family, 0)))
            * int(candidate_multiplier)
            for family in set(train_plan) | set(development_plan)
        }
        generator = Round0ScenarioGenerator(
            seed=int(seed),
            source_partition="train",
            parameter_ranking_dominance_threshold=1.0,
        )
        candidates = [
            partition_release_scenario_v1(row, split="dagger_train")
            for row in generator.build(requested)
        ]
        training, development = allocate_scenarios(
            candidates,
            d0_roots=d0_roots,
            train_plan=train_plan,
            development_plan=development_plan,
            seed=seed,
        )
        source = git_source_state(Path(__file__).resolve().parents[1])
        config = {
            "contract": RESEARCH_CONTRACT,
            "seed": int(seed),
            "train_plan": dict(train_plan),
            "development_plan": dict(development_plan),
            "candidate_multiplier": int(candidate_multiplier),
            "d0_raw_path": str(d0_raw_path.resolve()),
            "run_descriptor": dict(run_descriptor),
            "source_commit": source.get("source_commit"),
            "source_worktree_dirty": source.get("source_worktree_dirty"),
            "release_eligible": False,
            "training_roots": [_row_root(row) for row in training],
            "development_roots": [_row_root(row) for row in development],
        }
        _write_json(train_path, training)
        _write_json(development_path, development)
        _write_json(config_path, config)

    if not isinstance(training, list) or not isinstance(development, list):
        raise RuntimeError("Stored research scenario files must contain JSON arrays")
    train_roots = {_row_root(row) for row in training}
    development_roots = {_row_root(row) for row in development}
    if "" in train_roots | development_roots:
        raise RuntimeError("Stored research scenario is missing a physical root")
    if len(train_roots) != len(training) or len(development_roots) != len(development):
        raise RuntimeError("Stored research scenario arrays repeat a physical root")
    if train_roots & development_roots or (train_roots | development_roots) & d0_roots:
        raise RuntimeError("Stored research scenario split violates physical-root isolation")
    return training, development


def _episode_filename(root: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,180}", root):
        raise ValueError(f"Physical root is not safe as an episode filename: {root!r}")
    return f"{root}.jsonl"


def _completed_roots(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    payload = _read_json(path)
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, Mapping):
        values = payload.get("completed_roots") or []
    else:
        raise RuntimeError("completed_roots.json is malformed")
    return {str(root) for root in values if str(root).strip()}


def _action_tool(action: Any) -> str:
    return str(action.get("tool") or "") if isinstance(action, Mapping) else ""


def collection_metrics(
    rows: Sequence[Mapping[str, Any]], *, episodes_completed: int, episodes_total: int
) -> dict[str, Any]:
    learner_states = [row for row in rows if row.get("state_origin") == "learner_policy"]
    target_rows = [row for row in learner_states if row.get("preferred_action") is not None]
    safe = [row for row in rows if row.get("research_label_eligible") is True]
    quarantined = [
        row
        for row in target_rows
        if row.get("research_label_eligible") is not True
    ]
    comparable = [
        row
        for row in target_rows
        if isinstance(row.get("model_action"), Mapping)
        and isinstance(row.get("preferred_action"), Mapping)
    ]
    tool_disagreements = sum(
        _action_tool(row["model_action"]) != _action_tool(row["preferred_action"])
        for row in comparable
    )
    full_disagreements = sum(
        _stable_json(row["model_action"]) != _stable_json(row["preferred_action"])
        for row in comparable
    )
    model_actions = [row.get("model_action") for row in rows]
    invalid = sum(_action_tool(action) == INVALID_ACTION for action in model_actions)
    return {
        "contract": RESEARCH_CONTRACT,
        "episodes_completed": int(episodes_completed),
        "episodes_total": int(episodes_total),
        "states_visited": len(rows),
        "learner_actions_executed": sum(row.get("executed_by") == "model" for row in rows),
        "learner_induced_states": len(learner_states),
        "states_with_expert_target": len(target_rows),
        "safe_research_eligible_rows": len(safe),
        "quarantined_learner_target_rows": len(quarantined),
        "label_yield": len(safe) / len(learner_states) if learner_states else 0.0,
        "learner_expert_comparable_states": len(comparable),
        "learner_expert_tool_disagreement_rate": (
            tool_disagreements / len(comparable) if comparable else 0.0
        ),
        "learner_expert_full_action_disagreement_rate": (
            full_disagreements / len(comparable) if comparable else 0.0
        ),
        "invalid_learner_action_rate": invalid / len(model_actions) if model_actions else 0.0,
        "eligible_rows_by_scenario_family": dict(
            sorted(Counter(str(row.get("scenario_family") or "unknown") for row in safe).items())
        ),
        "eligible_rows_by_recovery_stratum": dict(
            sorted(Counter(str(row.get("recovery_stratum") or "unclassified") for row in safe).items())
        ),
        "research_ineligibility_reasons": dict(
            sorted(
                Counter(
                    reason
                    for row in rows
                    for reason in (row.get("research_label_ineligibility_reasons") or [])
                ).items()
            )
        ),
    }


def _load_episode_rows(
    training_scenarios: Sequence[Mapping[str, Any]],
    output_dir: Path,
    completed: set[str],
    *,
    beta: float,
    learner_adapter_path: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in training_scenarios:
        root = _row_root(scenario)
        if root not in completed:
            continue
        path = output_dir / "rollouts" / _episode_filename(root)
        if not path.is_file():
            raise RuntimeError(f"Completed root {root} has no episode JSONL")
        episode_rows = load_jsonl(path)
        _validate_episode_contract(
            episode_rows,
            root=root,
            beta=beta,
            learner_adapter_path=learner_adapter_path,
            path=path,
        )
        rows.extend(mark_research_label_eligibility(row) for row in episode_rows)
    return rows


def _validate_episode_contract(
    rows: Sequence[Mapping[str, Any]],
    *,
    root: str,
    beta: float,
    learner_adapter_path: str,
    path: Path | None = None,
) -> None:
    location = str(path) if path is not None else f"collected root {root}"
    if not rows:
        raise RuntimeError(f"Research rollout is empty: {location}")
    for index, row in enumerate(rows):
        labels = row.get("labels")
        labels = labels if isinstance(labels, Mapping) else {}
        problems: list[str] = []
        if _row_root(row) != root:
            problems.append("physical_root_mismatch")
        if int(row.get("iteration", -1)) != 1:
            problems.append("iteration_not_one")
        if row.get("collection_role", labels.get("collection_role")) != "training":
            problems.append("collection_role_not_training")
        if row.get("supervision_policy", labels.get("supervision_policy")) != (
            DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION
        ):
            problems.append("observable_supervision_contract_missing")
        try:
            observed_beta = float(
                row.get("collection_beta", labels.get("collection_beta"))
            )
        except (TypeError, ValueError):
            observed_beta = math.nan
        if not math.isclose(observed_beta, float(beta), rel_tol=0.0, abs_tol=1e-12):
            problems.append("collection_beta_mismatch")
        if row.get("research_learner_adapter_path") != learner_adapter_path:
            problems.append("learner_adapter_path_mismatch")
        observation = row.get("policy_observation")
        if not isinstance(observation, Mapping):
            problems.append("policy_observation_missing")
        else:
            try:
                validate_policy_payload(observation)
            except (TypeError, ValueError) as exc:
                problems.append(f"policy_observation_invalid:{exc}")
        if problems:
            raise RuntimeError(
                f"Research rollout row {index} violates the episode contract at "
                f"{location}: {', '.join(problems)}"
            )


def _publish_collection(
    *,
    output_dir: Path,
    training_scenarios: Sequence[Mapping[str, Any]],
    completed: set[str],
    beta: float,
    learner_adapter_path: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _load_episode_rows(
        training_scenarios,
        output_dir,
        completed,
        beta=beta,
        learner_adapter_path=learner_adapter_path,
    )
    safe = [row for row in rows if row.get("research_label_eligible") is True]
    metrics = collection_metrics(
        rows,
        episodes_completed=len(completed),
        episodes_total=len(training_scenarios),
    )
    _write_jsonl(output_dir / "d1.all.jsonl", rows)
    _write_jsonl(output_dir / "d1.safe.jsonl", safe)
    _write_json(output_dir / "collection_metrics.json", metrics)
    return rows, metrics


def collect_resumable(
    *,
    training_scenarios: Sequence[Mapping[str, Any]],
    development_roots: set[str],
    d0_roots: set[str],
    output_dir: Path,
    seed: int,
    beta: float,
    max_steps: int,
    policy_factory: Callable[[], Any],
    environment_factory: Callable[..., Any],
    learner_adapter_path: str,
    collector_class: type[DaggerRolloutCollector] = DaggerRolloutCollector,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not 0.25 <= float(beta) <= 0.5:
        raise ValueError("Research training beta must be in [0.25, 0.5]")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    completed_path = output_dir / "completed_roots.json"
    completed = _completed_roots(completed_path)
    expected_roots = {_row_root(row) for row in training_scenarios}
    unexpected = completed - expected_roots
    if unexpected:
        raise RuntimeError(f"completed_roots.json contains unknown roots: {sorted(unexpected)}")
    ledger_changed = False
    for root in sorted(expected_roots):
        episode_path = output_dir / "rollouts" / _episode_filename(root)
        if episode_path.is_file():
            episode_rows = load_jsonl(episode_path)
            _validate_episode_contract(
                episode_rows,
                root=root,
                beta=beta,
                learner_adapter_path=learner_adapter_path,
                path=episode_path,
            )
            if root not in completed:
                completed.add(root)
                ledger_changed = True
        elif root in completed:
            completed.remove(root)
            ledger_changed = True
    if ledger_changed:
        _write_json(
            completed_path,
            {"contract": RESEARCH_CONTRACT, "completed_roots": sorted(completed)},
        )
    pending_roots = expected_roots - completed
    policy: Any | None = policy_factory() if pending_roots else None
    forbidden = set(d0_roots) | set(development_roots)
    for scenario in training_scenarios:
        root = _row_root(scenario)
        episode_path = output_dir / "rollouts" / _episode_filename(root)
        if root in completed:
            if not episode_path.is_file():
                raise RuntimeError(f"Completed root {root} has no rollout file")
            continue
        try:
            episode_seed = _root_seed(seed, root)
            env = environment_factory(seed=episode_seed)
            expert = ExpertPolicyOracle(
                process_oracle=env.process_oracle,
                candidate_oracle=env.candidate_quality_oracle,
            )
            collector = collector_class(
                env=env,
                policy=policy,
                expert_oracle=expert,
                rng=random.Random(episode_seed),
                supervision_policy=DAGGER1_OBSERVABLE_RECOVERY_SUPERVISION,
                forbidden_physical_roots=forbidden,
            )
            rows = collector.collect_iteration(
                scenarios=[scenario],
                iteration=1,
                beta=float(beta),
                max_steps=int(max_steps),
                collection_role="training",
            )
        except Exception as exc:
            failure = {
                "physical_root_fingerprint": root,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            _write_json(output_dir / "failures" / f"{root}.json", failure)
            _publish_collection(
                output_dir=output_dir,
                training_scenarios=training_scenarios,
                completed=completed,
                beta=beta,
                learner_adapter_path=learner_adapter_path,
            )
            raise RuntimeError(
                f"Research collection stopped at root {root}; completed episodes "
                "remain resumable. First error: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        stamped = []
        for row in rows:
            stamped_row = copy.deepcopy(dict(row))
            stamped_row["research_collection_contract"] = RESEARCH_CONTRACT
            stamped_row["research_learner_adapter_path"] = learner_adapter_path
            stamped.append(mark_research_label_eligibility(stamped_row))
        _validate_episode_contract(
            stamped,
            root=root,
            beta=beta,
            learner_adapter_path=learner_adapter_path,
        )
        _write_jsonl(episode_path, stamped)
        completed.add(root)
        _write_json(
            completed_path,
            {
                "contract": RESEARCH_CONTRACT,
                "completed_roots": sorted(completed),
            },
        )
        _publish_collection(
            output_dir=output_dir,
            training_scenarios=training_scenarios,
            completed=completed,
            beta=beta,
            learner_adapter_path=learner_adapter_path,
        )
    return _publish_collection(
        output_dir=output_dir,
        training_scenarios=training_scenarios,
        completed=completed,
        beta=beta,
        learner_adapter_path=learner_adapter_path,
    )


def export_research_rows(
    rows: Sequence[Mapping[str, Any]], *, output_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    updated = [mark_research_label_eligibility(row) for row in rows]
    deduplicated: list[tuple[int, dict[str, Any]]] = []
    seen: set[str] = set()
    for index, row in enumerate(updated):
        if row.get("research_label_eligible") is not True:
            continue
        key = _stable_json(
            [row.get("policy_observation"), row.get("preferred_action")]
        )
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append((index, row))

    chat_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row_index, row in deduplicated:
        try:
            exported = examples_to_chat_sft(
                [row], protocol="canonical", allow_ineligible_auxiliary=True
            )
        except ValueError as exc:
            example_id = str(row.get("example_id"))
            reason = f"canonical_sft_export_failed:{exc}"
            updated[row_index] = mark_research_label_eligibility(
                row, extra_reason=reason
            )
            failures.append({"example_id": example_id, "error": str(exc)})
            continue
        if len(exported) != 1:
            raise RuntimeError("One eligible D1 row did not export exactly once")
        chat_rows.append(exported[0])

    _write_jsonl(output_dir / "d1.all.jsonl", updated)
    _write_jsonl(
        output_dir / "d1.safe.jsonl",
        [row for row in updated if row.get("research_label_eligible") is True],
    )
    _write_jsonl(output_dir / "d1.chat.jsonl", chat_rows)
    _write_json(output_dir / "export_failures.json", failures)
    return updated, chat_rows, failures


def build_research_mixture(
    d0_rows: Sequence[Mapping[str, Any]],
    d1_rows: Sequence[Mapping[str, Any]],
    *,
    d1_share: float,
    d1_cap: int | None,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not 0.0 < float(d1_share) <= 1.0:
        raise ValueError("d1_share must be in (0, 1]")
    if d1_cap is not None and d1_cap <= 0:
        raise ValueError("d1_cap must be positive when supplied")
    rootless_d0 = [
        str(row.get("example_id") or index)
        for index, row in enumerate(d0_rows)
        if not _row_root(row)
    ]
    if rootless_d0:
        raise ValueError(
            "D0 chat rows must retain physical_root_fingerprint for root isolation; "
            f"missing examples: {rootless_d0[:8]}"
        )
    rootless_d1 = [
        str(row.get("example_id") or index)
        for index, row in enumerate(d1_rows)
        if not _row_root(row)
    ]
    if rootless_d1:
        raise ValueError(
            "D1 chat rows must retain physical_root_fingerprint for paired root "
            f"isolation; missing examples: {rootless_d1[:8]}"
        )
    overlapping_roots = (
        {_row_root(row) for row in d0_rows}
        & {_row_root(row) for row in d1_rows}
    )
    if overlapping_roots:
        raise ValueError(
            "D0 and D1 training rows overlap on physical roots: "
            f"{sorted(overlapping_roots)[:8]}"
        )
    rng = random.Random(int(seed))
    selected_d1 = list(copy.deepcopy(list(d1_rows)))
    if d1_cap is not None and len(selected_d1) > d1_cap:
        selected_d1 = rng.sample(selected_d1, d1_cap)
    if not selected_d1:
        raise ValueError("No safe exported D1 rows are available for research training")
    target_d0 = (
        0
        if math.isclose(float(d1_share), 1.0)
        else math.floor(len(selected_d1) * (1.0 - float(d1_share)) / float(d1_share))
    )
    selected_d0 = rng.sample(list(d0_rows), min(len(d0_rows), target_d0))
    mixture: list[dict[str, Any]] = []
    for source, source_rows in (("d0", selected_d0), ("d1", selected_d1)):
        for source_row in source_rows:
            row = copy.deepcopy(dict(source_row))
            row["research_mixture_source"] = source
            metadata = row.get("metadata")
            metadata = copy.deepcopy(dict(metadata)) if isinstance(metadata, Mapping) else {}
            metadata["research_mixture_source"] = source
            row["metadata"] = metadata
            mixture.append(row)
    rng.shuffle(mixture)
    report = {
        "contract": RESEARCH_CONTRACT,
        "requested_d1_share": float(d1_share),
        "d0_available": len(d0_rows),
        "d1_available": len(d1_rows),
        "d0_selected": len(selected_d0),
        "d1_selected": len(selected_d1),
        "actual_d1_share": len(selected_d1) / len(mixture),
        "d1_cap": d1_cap,
        "seed": int(seed),
    }
    return mixture, report


def evaluate_paired_adapters(
    *,
    development_scenarios: Sequence[Mapping[str, Any]],
    bc0_adapter: Path,
    r1_adapter: Path,
    base_model: str,
    base_revision: str,
    output_dir: Path,
    seed: int,
    max_steps: int,
    policy_loader: Callable[..., Any],
    environment_factory: Callable[..., Any],
    evaluator: Callable[..., Any],
    load_in_4bit: bool = True,
    local_files_only: bool = True,
    trust_remote_code: bool = False,
    prompt_profile: str | None = None,
    architecture: str | None = None,
    policy_cache_clear: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Run BC0 and R1 on the exact same saved development scenarios."""

    roots = [_row_root(row) for row in development_scenarios]
    if not roots or "" in roots or len(roots) != len(set(roots)):
        raise ValueError("Paired evaluation requires unique development physical roots")
    suite = {"standard_success": list(development_scenarios)}
    payloads: dict[str, dict[str, Any]] = {}
    for label, adapter in (("bc0", bc0_adapter), ("r1", r1_adapter)):
        policy = policy_loader(
            adapter,
            base_model=base_model,
            base_revision=base_revision,
            load_in_4bit=load_in_4bit,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            prompt_profile=prompt_profile,
            architecture=architecture,
        )
        result = evaluator(
            suite,
            env_factory=environment_factory,
            policy_factory=lambda _policy=policy, **_kwargs: _policy,
            max_steps=max_steps,
            seed=seed,
            required_suites=["standard_success"],
            minimum_suites=1,
            minimum_episodes_per_suite=1,
            minimum_roots_per_suite=1,
            require_release_environment=False,
            require_policy_identity=False,
        )
        payload = result.as_dict()
        payloads[label] = payload
        _write_json(output_dir / "evaluation" / f"{label}_eval.json", payload)
        del result, policy
        if policy_cache_clear is not None:
            policy_cache_clear()
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    bc0_overall = payloads["bc0"]["suite_metrics"]["overall"]
    r1_overall = payloads["r1"]["suite_metrics"]["overall"]
    shared_numeric = sorted(
        key
        for key in set(bc0_overall) & set(r1_overall)
        if isinstance(bc0_overall[key], (int, float))
        and not isinstance(bc0_overall[key], bool)
        and isinstance(r1_overall[key], (int, float))
        and not isinstance(r1_overall[key], bool)
    )
    comparison = {
        "contract": RESEARCH_CONTRACT,
        "paired_physical_roots": roots,
        "seed": seed,
        "max_steps": max_steps,
        "bc0_adapter": str(bc0_adapter),
        "r1_adapter": str(r1_adapter),
        "bc0_overall": bc0_overall,
        "r1_overall": r1_overall,
        "r1_minus_bc0": {
            key: float(r1_overall[key]) - float(bc0_overall[key])
            for key in shared_numeric
        },
    }
    _write_json(output_dir / "evaluation" / "comparison.json", comparison)
    return comparison


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description="Collect a minimal observable-teacher research DAgger iteration"
    )
    result.add_argument("--d0-raw", required=True, type=Path)
    result.add_argument("--d0-train", required=True, type=Path)
    result.add_argument("--adapter-path", required=True, type=Path)
    result.add_argument("--output-dir", required=True, type=Path)
    result.add_argument(
        "--model-choice",
        choices=tuple(RESEARCH_MODEL_SPECS),
        default=DEFAULT_RESEARCH_MODEL.key,
        help="Atomic pinned learner preset; 12b is default and e4b is the fast smoke",
    )
    result.add_argument(
        "--base-model",
        help="Advanced override for the preset model id or a local snapshot",
    )
    result.add_argument(
        "--base-revision",
        help="Advanced revision override; registered presets are already pinned",
    )
    result.add_argument(
        "--architecture",
        choices=("gemma4", "gemma4_unified"),
        help="Required only for an unregistered custom base",
    )
    result.add_argument("--seed", type=int, default=20260720)
    result.add_argument("--beta", type=float, default=0.25)
    result.add_argument("--max-steps", type=int, default=16)
    result.add_argument("--d1-cap", type=int, default=150)
    result.add_argument("--d1-share", type=float, default=0.25)
    result.add_argument("--candidate-multiplier", type=int, default=3)
    result.add_argument(
        "--train-plan",
        default="",
        help="JSON object or JSON file; default is 12/12/6 roots",
    )
    result.add_argument(
        "--development-plan",
        default="",
        help="JSON object or JSON file; default is 6/6/3 roots",
    )
    result.add_argument("--allow-download", action="store_true")
    result.add_argument("--trust-remote-code", action="store_true")
    result.add_argument("--no-load-in-4bit", action="store_true")
    result.add_argument(
        "--prompt-profile",
        choices=tuple(sorted(SUPPORTED_RESEARCH_PROMPT_PROFILES)),
        help="Advanced override; known presets select the matching prompt contract",
    )
    result.add_argument(
        "--eval-r1-adapter",
        type=Path,
        help="After collection/training, run paired BC0/R1 evaluation on the saved development roots",
    )
    result.add_argument("--eval-max-steps", type=int, default=24)
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    from psse_env.dagger.release_factories import production_environment_factory
    from psse_env.dagger.research_policy_factory import (
        clear_research_policy_cache,
        research_gemma_policy_factory,
    )

    selected = get_research_model_spec(args.model_choice)
    model_spec = resolve_research_model_spec(
        model=args.base_model or selected.model_id,
        revision=args.base_revision,
        architecture=args.architecture,
        prompt_profile=args.prompt_profile,
        default=selected,
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    d0_raw = args.d0_raw.expanduser().resolve(strict=True)
    d0_train = args.d0_train.expanduser().resolve(strict=True)
    adapter = args.adapter_path.expanduser().resolve(strict=True)
    d0_raw_rows = load_jsonl(d0_raw)
    d0_roots = load_d0_roots(d0_raw)
    d0_train_rows, d0_view_report = refresh_d0_training_view(
        d0_raw_rows, load_jsonl(d0_train)
    )
    d0_train_roots = validate_d0_training_roots(
        d0_train_rows, raw_roots=d0_roots
    )
    d0_view_fingerprint = hashlib.sha256(
        _stable_json(d0_train_rows).encode("utf-8")
    ).hexdigest()
    train_plan = _parse_plan(args.train_plan, DEFAULT_TRAIN_PLAN)
    development_plan = _parse_plan(
        args.development_plan, DEFAULT_DEVELOPMENT_PLAN
    )
    run_descriptor = {
        "adapter_path": str(adapter),
        "model_choice": model_spec.key,
        "base_model": model_spec.model_id,
        "base_revision": model_spec.revision,
        "architecture": model_spec.architecture,
        "prompt_profile": model_spec.prompt_profile,
        "d0_train_path": str(d0_train),
        "d0_train_roots": sorted(d0_train_roots),
        "d0_observation_view": d0_view_report,
        "beta": float(args.beta),
        "max_steps": int(args.max_steps),
    }
    training, development = prepare_scenario_split(
        output_dir=output_dir,
        d0_raw_path=d0_raw,
        d0_roots=d0_roots,
        train_plan=train_plan,
        development_plan=development_plan,
        candidate_multiplier=args.candidate_multiplier,
        seed=args.seed,
        run_descriptor=run_descriptor,
    )
    _write_jsonl(output_dir / "d0.train.current.jsonl", d0_train_rows)
    _write_json(
        output_dir / "d0_view_report.json",
        {**d0_view_report, "sha256": d0_view_fingerprint},
    )
    development_roots = {_row_root(row) for row in development}
    rows, metrics = collect_resumable(
        training_scenarios=training,
        development_roots=development_roots,
        d0_roots=d0_roots,
        output_dir=output_dir,
        seed=args.seed,
        beta=args.beta,
        max_steps=args.max_steps,
        policy_factory=lambda: research_gemma_policy_factory(
            adapter,
            base_model=model_spec.model_id,
            base_revision=model_spec.revision,
            load_in_4bit=not args.no_load_in_4bit,
            local_files_only=not args.allow_download,
            trust_remote_code=args.trust_remote_code,
            prompt_profile=model_spec.prompt_profile,
            architecture=model_spec.architecture,
        ),
        environment_factory=production_environment_factory,
        learner_adapter_path=str(adapter),
    )
    updated, chat_rows, export_failures = export_research_rows(
        rows, output_dir=output_dir
    )
    metrics = collection_metrics(
        updated,
        episodes_completed=len(training),
        episodes_total=len(training),
    )
    metrics["canonical_export_rows"] = len(chat_rows)
    metrics["canonical_export_failures"] = len(export_failures)
    _write_json(output_dir / "collection_metrics.json", metrics)
    training_roots = {_row_root(row) for row in training}
    exported_d1_roots = {_row_root(row) for row in chat_rows}
    if "" in exported_d1_roots:
        raise RuntimeError("An exported D1 row lost its physical root")
    unexpected_d1_roots = exported_d1_roots - training_roots
    if unexpected_d1_roots:
        raise RuntimeError(
            "Exported D1 rows contain roots outside the saved training closure: "
            f"{sorted(unexpected_d1_roots)[:8]}"
        )
    forbidden_d1_roots = exported_d1_roots & (d0_roots | development_roots)
    if forbidden_d1_roots:
        raise RuntimeError(
            "Exported D1 rows overlap D0 or development roots: "
            f"{sorted(forbidden_d1_roots)[:8]}"
        )
    mixture, mixture_report = build_research_mixture(
        d0_train_rows,
        chat_rows,
        d1_share=args.d1_share,
        d1_cap=args.d1_cap,
        seed=args.seed,
    )
    _write_jsonl(output_dir / "round1.train.jsonl", mixture)
    _write_json(output_dir / "mixture_report.json", mixture_report)
    comparison = None
    if args.eval_r1_adapter is not None:
        from psse_env.dagger.evaluator import evaluate_rollout_suites

        comparison = evaluate_paired_adapters(
            development_scenarios=development,
            bc0_adapter=adapter,
            r1_adapter=args.eval_r1_adapter.expanduser().resolve(strict=True),
            base_model=model_spec.model_id,
            base_revision=model_spec.revision,
            output_dir=output_dir,
            seed=args.seed,
            max_steps=args.eval_max_steps,
            policy_loader=research_gemma_policy_factory,
            environment_factory=production_environment_factory,
            evaluator=evaluate_rollout_suites,
            load_in_4bit=not args.no_load_in_4bit,
            local_files_only=not args.allow_download,
            trust_remote_code=args.trust_remote_code,
            prompt_profile=model_spec.prompt_profile,
            architecture=model_spec.architecture,
            policy_cache_clear=clear_research_policy_cache,
        )
    report = {
        "passed": True,
        "contract": RESEARCH_CONTRACT,
        "release_eligible": False,
        "model_selection": {
            "key": model_spec.key,
            "model_id": model_spec.model_id,
            "revision": model_spec.revision,
            "architecture": model_spec.architecture,
            "prompt_profile": model_spec.prompt_profile,
            "purpose": model_spec.purpose,
        },
        "collection_metrics": metrics,
        "mixture": mixture_report,
        "paired_evaluation": comparison,
        "paths": {
            "training_scenarios": str(output_dir / "training_scenarios.json"),
            "development_scenarios": str(output_dir / "development_scenarios.json"),
            "d1_safe": str(output_dir / "d1.safe.jsonl"),
            "d1_chat": str(output_dir / "d1.chat.jsonl"),
            "d0_train_current": str(output_dir / "d0.train.current.jsonl"),
            "round1_train": str(output_dir / "round1.train.jsonl"),
        },
    }
    _write_json(output_dir / "research_run_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    try:
        report = run(parser().parse_args(argv))
    except (OSError, RuntimeError, ValueError) as exc:
        print(
            json.dumps({"passed": False, "error": str(exc)}, indent=2),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
