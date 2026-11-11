"""The fork probe: what commit and decline lead to from states the learner reached (workplan of 2026-11-10, stage 2).

    # learner episodes on roots that have none yet (the reserved validation roots); --policy expert is a CPU stand-in
    python -m research.rl.fork_probe rollout --config arm_a.json --suite ieee14_train.json --roots-file reserved_roots.json \\
        --store out/probe/validation_trajectories --samples 2
    # freeze the sample: one accepted-candidate fork per root, by kind, with the shortfalls written down
    python -m research.rl.fork_probe select --store out/ieee14_v1/iteration_000/trajectories --validation-store out/probe/validation_trajectories \\
        --suite ieee14_train.json --expert-reference teacher_episodes.jsonl --output out/probe/forks.json
    # what the learner is shown at each fork in each observation mode, by replay (CPU), then its choice there (GPU):
    # 48 states x 2 views
    python -m research.rl.fork_probe views --manifest out/probe/forks.json --suite ieee14_train.json --config arm_a.json \\
        --output out/probe/views.json
    python -m research.rl.fork_probe decide --manifest out/probe/forks.json --views out/probe/views.json --config arm_a.json \\
        --output out/probe/decisions.json
    # both choices, both observation modes, two learner continuations each (48 x 2 x 2 x 2 = 384 suffix episodes)
    python -m research.rl.fork_probe branch --manifest out/probe/forks.json --suite ieee14_train.json --config arm_a.json \\
        --output-dir out/probe/branches [--shard 0/2] [--continuations 4]
    python -m research.rl.fork_probe report --manifest out/probe/forks.json --branches out/probe/branches \\
        --decisions out/probe/decisions.json --output out/probe/report
    # the same forks as a fixed decision panel: the policy's choices before and after an update (stage 0)
    python -m research.rl.fork_probe panel --manifest out/probe/forks.json --before decisions_s0.json --after decisions_s1.json \\
        --output out/probe/panel
    # which of the probe's roots the rule expert fails: its own episodes on them (rollout --policy expert), as a map
    python -m research.rl.fork_probe reference --store out/probe/expert_training --store out/probe/expert_validation \\
        --output out/probe/expert_reference.json
    # how often an accepted line-parameter candidate is a true target, by class (the workplan's candidate audit)
    python -m research.rl.fork_probe candidates --episodes cloning_ieee14_v2/episodes.jsonl --output out/probe/candidates

A fork is a decision the learner sampled on a candidate the verifier
accepted, where the environment allows both a commit and a decline
(``lifecycle_choice``).  ``select`` takes one per root, by the kinds the
workplan names: a wrong accepted candidate; a true first candidate; a true
candidate after an earlier decline; a wrong line-parameter candidate in a
class whose faults are breakers alone; a true line-parameter candidate in a
class with a breaker and a line fault.  The kinds, the roots the rule expert
fails and the split come from the audit's truth and are used only to choose
and to report.  The rules and the seed are in the manifest; a kind that the
stored states cannot fill is a recorded shortfall, not a silent substitution.

``branch`` replays each fork's prefix through the real environment, forces
the choice and lets the policy continue under the budget the episode had
left (:mod:`psse_env.rl.forks`).  A branch counts only when the replay
showed the policy what the recorded episode showed it at every step and the
environment executed the forced choice.  The forced step is recorded as an
intervention; the continuation's steps are never training rows.

``report`` joins the forks, the branches and the decision probabilities:
the value of each choice by kind, split and observation mode; what followed
a forced choice (the next correction's target, its verdict, the outcome);
how the learner's own probability of declining moves with the complete
history; and the workplan's screening count, the distinct expert-failed
roots on which a learner continuation solved the case.

The same commands with ``--policy expert`` / ``--continuation expert`` run on
a CPU with the rule expert in the learner's place.  That checks the wiring,
and its table is the expert's own continuation value of each choice; it is
not the learner's.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psse_env.actions import COMMIT_STATE, CORRECT_PARAMETERS, ROLLBACK_STATE  # noqa: E402
from psse_env.dagger.observation_modes import BOUNDED_HISTORY, COMPLETE_HISTORY, OBSERVATION_MODES  # noqa: E402
from psse_env.rl.forks import CHOICES, DecisionState, branch_seed, capture_fork_view, decision_states, run_branch  # noqa: E402
from psse_env.rl.lifecycle import CORRECTION_TOOLS, scenario_truth  # noqa: E402
from psse_env.rl.rollout import TrajectoryStore, scenario_root  # noqa: E402

PROBE_CONTRACT = "rl_fork_probe_v1"
#: Classes whose injected faults are breaker errors alone: there a line-parameter candidate is never a true target.
BREAKER_ONLY_CLASSES = frozenset({"cb2_dangling", "cb2_mixed", "cb2_split", "control_cb"})
#: Classes that hold a breaker error and a line-parameter error.
MIXED_PREFIX = "rx_cb"
#: The kinds of fork, in the order a root's fork is chosen.
STRATA = ("breaker_only_false_parameter", "wrong_target", "true_after_decline", "mixed_true_parameter", "true_first", "true_later")
#: Forks per kind: 32 training roots and 16 validation roots (the workplan's sample).
DEFAULT_QUOTAS = {
    "training": {"breaker_only_false_parameter": 6, "wrong_target": 8, "true_after_decline": 4, "mixed_true_parameter": 6,
                 "true_first": 8, "true_later": 0},
    "validation": {"breaker_only_false_parameter": 3, "wrong_target": 4, "true_after_decline": 2, "mixed_true_parameter": 3,
                   "true_first": 4, "true_later": 0},
}
#: How many of a split's roots should be ones the rule expert fails (a target; a shortfall is recorded).
DEFAULT_EXPERT_FAILED = {"training": 12, "validation": 6}
#: The workplan's screening rule: learner continuations solve this many distinct expert-failed roots, one of them
#: at least in the validation split.
GATE_EXPERT_FAILED_ROOTS = 5
#: What a branch's ``continuation_policy`` is when the learner continued it (the driver's ``policy_source`` of a run
#: on an adapter).  The rule is about these branches alone: a run whose continuations the rule expert made checks
#: the wiring, and what it solves is not a count of the rule.
LEARNER_POLICY = "adapter"


# --------------------------------------------------------------------------- inputs


def load_suite(path: str | Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload if isinstance(payload, list) else (payload.get("scenarios") or payload.get("rows") or [])
    return {scenario_root(row): dict(row) for row in rows}


def compound_class(scenario: Mapping[str, Any]) -> str | None:
    compound = ((scenario.get("audit") or {}).get("compound") or {}) if isinstance(scenario.get("audit"), Mapping) else {}
    return str(compound["class"]) if compound.get("class") else None


def relation_key(scenario: Mapping[str, Any]) -> str:
    """What related scenarios share: the physical root they are noise variants of."""

    grouping = scenario.get("grouping") if isinstance(scenario.get("grouping"), Mapping) else {}
    return str(grouping.get("physical_root_fingerprint") or grouping.get("root_scenario_id") or scenario_root(scenario))


def load_expert_reference(path: str | Path | None) -> dict[str, bool]:
    """Root to whether the rule expert solved it: a JSON map, or the ``episodes.jsonl`` of an expert run."""

    if path is None:
        return {}
    text = Path(path).read_text(encoding="utf-8")
    reference: dict[str, bool] = {}
    try:
        whole = json.loads(text)
    except json.JSONDecodeError:
        whole = None  # several lines, one episode each
    if isinstance(whole, Mapping) and "scenario_id" not in whole:
        for root, value in whole.items():
            reference[str(root)] = bool(value.get("success") if isinstance(value, Mapping) else value)
        return reference
    for line in text.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        root = row.get("scenario_id")
        if root is not None:
            reference[str(root)] = bool(row.get("success", row.get("truth_audited_task_success")))
    return reference


def expert_reference_from_stores(stores: Iterable[str | Path]) -> dict[str, dict[str, Any]]:
    """Root to the rule expert's own result, from the stores of its rollouts (``rollout --policy expert``).

    The expert is deterministic, so a root's first stored episode is its result.
    """

    reference: dict[str, dict[str, Any]] = {}
    for directory in stores:
        for root, group in TrajectoryStore(directory).iter_groups():
            if root in reference:
                raise ValueError(f"root {root} is in more than one of the expert's stores")
            evaluation = group[0].evaluation
            reference[root] = {"success": bool(evaluation.get("truth_audited_task_success")),
                               "steps": int(evaluation.get("policy_steps") or evaluation.get("steps") or 0)}
    return reference


# --------------------------------------------------------------------------- the candidate audit


def candidate_records(*, episodes: Iterable[str | Path] = (), stores: Iterable[str | Path] = (),
                      suite: Mapping[str, Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Every decision on an accepted candidate in the given records, as ``{class, tool, target_is_true, declined}``.

    ``episodes`` are ``episodes.jsonl`` files of the compound study (the
    rule expert or the audit deciding): a correction the verifier accepted
    and the policy then kept or declined.  ``stores`` are rollout stores of
    the driver, read with ``suite`` for each root's class.
    """

    records: list[dict[str, Any]] = []
    for path in episodes:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            episode = json.loads(line)
            for correction in episode.get("corrections") or []:
                if not str(correction.get("verdict") or "").startswith("ACCEPT") or correction.get("lifecycle") is None:
                    continue
                records.append({"class": episode.get("class"), "tool": correction.get("tool"),
                                "target_is_true": correction.get("target_is_true"),
                                "declined": correction.get("lifecycle") == ROLLBACK_STATE})
    for directory in stores:
        for root, group in TrajectoryStore(directory).iter_groups():
            name = compound_class(suite[root]) if suite is not None and root in suite else None
            for trajectory in group:
                for decision in trajectory.lifecycle:
                    records.append({"class": name, "tool": decision.get("correction_tool"),
                                    "target_is_true": decision.get("target_is_true"), "declined": bool(decision.get("declined"))})
    return records


def candidate_audit(records: Sequence[Mapping[str, Any]], *, tool: str = CORRECT_PARAMETERS) -> dict[str, Any]:
    """How often an accepted candidate of ``tool`` is a true target, by class and by the workplan's slices.

    These are counts of candidates, conditional on the policy that proposed
    them; they are not rates of episodes.
    """

    by_class: dict[str, dict[str, int]] = defaultdict(lambda: {"candidates": 0, "true": 0, "declined": 0})
    labelled = [record for record in records if record.get("tool") == tool and record.get("target_is_true") is not None]
    for record in labelled:
        entry = by_class[str(record.get("class"))]
        entry["candidates"] += 1
        entry["true"] += int(bool(record["target_is_true"]))
        entry["declined"] += int(bool(record.get("declined")))

    def total(names: Iterable[str]) -> dict[str, Any]:
        chosen = [by_class[name] for name in names if name in by_class]
        candidates, true = sum(entry["candidates"] for entry in chosen), sum(entry["true"] for entry in chosen)
        return {"candidates": candidates, "true": true, "true_share": (true / candidates) if candidates else None}

    names = sorted(by_class)
    return {
        "decisions_on_accepted_candidates": len(records),
        "tool": tool,
        "candidates": len(labelled),
        "by_class": {name: dict(by_class[name]) for name in names},
        "breaker_only_classes": total(name for name in names if name in BREAKER_ONLY_CLASSES),
        "other_classes": total(name for name in names if name not in BREAKER_ONLY_CLASSES),
        # A subset of the other classes: a breaker error and a line-parameter error in one case.
        "mixed_breaker_and_line_classes": total(name for name in names if name.startswith(MIXED_PREFIX)),
    }


def candidate_markdown(audit: Mapping[str, Any], *, title: str = "Candidate audit") -> str:
    def row(label: str, entry: Mapping[str, Any]) -> str:
        share = entry.get("true_share")
        return f"| {label} | {entry['true']} of {entry['candidates']} | {'n/a' if share is None else f'{100 * share:.1f}%'} |"

    lines = [f"# {title}", "",
             f"{audit['decisions_on_accepted_candidates']} decisions on accepted candidates; {audit['candidates']} of them on a "
             f"`{audit['tool']}` candidate with a label. Counts of candidates, conditional on the policy that proposed them.", "",
             "| Slice | True targets | Share |", "| --- | --- | --- |",
             row("Classes whose faults are breakers alone", audit["breaker_only_classes"]),
             row("Other classes", audit["other_classes"]),
             row("Classes with a breaker and a line fault (part of the other classes)", audit["mixed_breaker_and_line_classes"]),
             "", "| Class | True targets | Declined |", "| --- | --- | --- |"]
    lines += [f"| {name} | {entry['true']} of {entry['candidates']} | {entry['declined']} |" for name, entry in audit["by_class"].items()]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- the forks of a store


def fork_stratum(state: DecisionState, name: str | None) -> str:
    """The kind of a fork, from the audit's label of its decision and the root's compound class."""

    decision = state.decision
    parameter = decision.get("correction_tool") == CORRECT_PARAMETERS
    if decision.get("target_is_true") is False:
        return "breaker_only_false_parameter" if parameter and name in BREAKER_ONLY_CLASSES else "wrong_target"
    if state.earlier_declines:
        return "true_after_decline"
    if parameter and str(name or "").startswith(MIXED_PREFIX):
        return "mixed_true_parameter"
    return "true_first" if state.ordinal == 0 else "true_later"


def edit_audit(state: DecisionState, scenario: Mapping[str, Any]) -> dict[str, Any]:
    """What the candidate's correction wrote, against the truth: every line a parameter correction changed.

    The membership label of a decision names the requested target alone.  A
    parameter correction with the pair refit also rewrites the line of the
    last committed parameter correction, so a candidate on a true line can
    still change a healthy one.  Read from the correction's own edit record
    (``detailed_evidence.edit``) in the fork's transitions.
    """

    truth = scenario_truth(scenario)
    correction = next((event for event in reversed(state.transitions)
                       if str((event.get("action") or {}).get("tool")) in CORRECTION_TOOLS
                       and (event.get("tool_output") or {}).get("execution_status") == "success"), None)
    if correction is None:
        return {"known": False}
    output = correction.get("tool_output") or {}
    metrics = output.get("tool_metrics") if isinstance(output.get("tool_metrics"), Mapping) else {}
    record = metrics.get("detailed_evidence", output.get("detailed_evidence"))
    edit = record.get("edit") if isinstance(record, Mapping) else None
    tool = str((correction.get("action") or {}).get("tool"))
    if tool != CORRECT_PARAMETERS or not isinstance(edit, list):
        return {"known": isinstance(record, Mapping), "tool": tool}
    lines = sorted({int(item["line"]) for item in edit if isinstance(item, Mapping) and item.get("line") is not None})
    healthy = [line for line in lines if line not in set(truth.get("lines") or ())]
    refit = metrics.get("pair_refit", output.get("pair_refit"))
    return {
        "known": True, "tool": tool, "lines_changed": lines, "healthy_lines_changed": healthy,
        "pair_refit": bool(isinstance(refit, Mapping) and refit.get("status") == "refit"),
        # The label says "true target" while the edit also moved a line the truth does not name.
        "label_true_but_healthy_line_changed": bool(state.decision.get("target_is_true") is True and healthy),
    }


def store_forks(stores: Iterable[str | Path], suite: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, list[DecisionState]], dict[str, int]]:
    """Every fork of the stores' episodes by root, and how many labelled decisions of each kind the stores hold."""

    by_root: dict[str, list[DecisionState]] = defaultdict(list)
    natural: dict[str, int] = defaultdict(int)
    for directory in stores:
        for root, group in TrajectoryStore(directory).iter_groups():
            if root not in suite:
                raise ValueError(f"root {root} of store {directory} is not in the suite")
            for trajectory in group:
                for state in decision_states(trajectory):
                    by_root[root].append(state)
                    natural[fork_stratum(state, compound_class(suite[root]))] += 1
    return dict(by_root), dict(natural)


def select_forks(
    by_root: Mapping[str, Sequence[DecisionState]],
    suite: Mapping[str, Mapping[str, Any]],
    expert: Mapping[str, bool],
    *,
    split: str,
    quotas: Mapping[str, int],
    expert_failed_target: int,
    seed: int,
    exclude_relations: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """One fork per root for a split, by kind, with the coverage and the shortfalls.

    A root gives the fork of the first kind (:data:`STRATA`) it has a
    candidate for and the split still wants; roots the rule expert fails go
    first until the split has its share of them.  Roots whose physical root
    is already in the other split are left out.
    """

    rng = random.Random(f"{int(seed)}:{split}")
    order = sorted(by_root)
    rng.shuffle(order)
    excluded = exclude_relations or set()
    order = [root for root in order if relation_key(suite[root]) not in excluded]
    kinds: dict[str, dict[str, DecisionState]] = {}
    for root in order:
        first: dict[str, DecisionState] = {}
        for state in sorted(by_root[root], key=lambda item: (item.sample_index, item.position)):
            first.setdefault(fork_stratum(state, compound_class(suite[root])), state)
        kinds[root] = first
    chosen: dict[str, tuple[str, DecisionState]] = {}
    failed_taken = 0
    relations: set[str] = set()

    def wanted_first(roots: list[str]) -> list[str]:
        # Stable: the seeded order holds within "expert fails" and within "expert solves".
        return sorted(roots, key=lambda root: 0 if (failed_taken < expert_failed_target and expert.get(root) is False) else 1)

    def take(stratum: str, limit: int) -> None:
        nonlocal failed_taken
        while limit > 0:
            eligible = [root for root in order if root not in chosen and stratum in kinds[root]
                        and relation_key(suite[root]) not in relations]
            if not eligible:
                return
            root = wanted_first(eligible)[0]
            chosen[root] = (stratum, kinds[root][stratum])
            relations.add(relation_key(suite[root]))
            failed_taken += int(expert.get(root) is False)
            limit -= 1

    total = sum(int(quotas.get(name, 0)) for name in STRATA)
    for name in STRATA:
        take(name, int(quotas.get(name, 0)))
    for name in STRATA:  # what a kind could not fill goes to the kinds that can
        take(name, total - len(chosen))
    forks = []
    for root in sorted(chosen):
        stratum, state = chosen[root]
        scenario = suite[root]
        forks.append({
            "split": split, "stratum": stratum, "class": compound_class(scenario), "relation": relation_key(scenario),
            "expert_success": expert.get(root), "edit_audit": edit_audit(state, scenario), "state": state.as_dict(),
        })
    selected = {name: sum(1 for fork in forks if fork["stratum"] == name) for name in STRATA}
    available = {name: sum(1 for root in order if name in kinds[root]) for name in STRATA}
    coverage = {
        "roots_with_a_fork": len(order),
        "selected": len(forks),
        "wanted": total,
        "by_stratum": {name: {"quota": int(quotas.get(name, 0)), "selected": selected[name], "roots_available": available[name],
                              "shortfall": max(0, int(quotas.get(name, 0)) - selected[name])} for name in STRATA},
        "expert_failed": {"target": int(expert_failed_target), "selected": sum(1 for fork in forks if fork["expert_success"] is False),
                          "available": sum(1 for root in order if expert.get(root) is False)},
        "expert_unknown": sum(1 for fork in forks if fork["expert_success"] is None),
        "relations": sorted(relations),
    }
    coverage["shortfalls"] = (
        [f"{name}: {entry['selected']} of {entry['quota']}" for name, entry in coverage["by_stratum"].items() if entry["shortfall"]]
        + ([f"expert-failed roots: {coverage['expert_failed']['selected']} of {coverage['expert_failed']['target']}"]
           if coverage["expert_failed"]["selected"] < coverage["expert_failed"]["target"] else [])
        + ([f"forks: {len(forks)} of {total}"] if len(forks) < total else [])
    )
    return forks, coverage


def build_manifest(
    *, training_stores: Sequence[str | Path], validation_stores: Sequence[str | Path], suite_path: str | Path,
    expert_reference: str | Path | None, seed: int, quotas: Mapping[str, Mapping[str, int]] | None = None,
    expert_failed: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """The frozen fork sample: the rules, the forks of both splits and what could not be covered."""

    suite = load_suite(suite_path)
    expert = load_expert_reference(expert_reference)
    quotas = {**DEFAULT_QUOTAS, **{name: dict(value) for name, value in (quotas or {}).items()}}
    expert_failed = {**DEFAULT_EXPERT_FAILED, **dict(expert_failed or {})}
    training_forks_by_root, training_natural = store_forks(training_stores, suite)
    validation_forks_by_root, validation_natural = store_forks(validation_stores, suite)
    overlap = sorted(set(training_forks_by_root) & set(validation_forks_by_root))
    if overlap:
        raise ValueError(f"{len(overlap)} roots are in a training store and a validation store (first: {overlap[0]})")
    validation, validation_coverage = select_forks(
        validation_forks_by_root, suite, expert, split="validation", quotas=quotas["validation"],
        expert_failed_target=expert_failed["validation"], seed=seed)
    training, training_coverage = select_forks(
        training_forks_by_root, suite, expert, split="training", quotas=quotas["training"],
        expert_failed_target=expert_failed["training"], seed=seed, exclude_relations=set(validation_coverage["relations"]))
    return {
        "contract": PROBE_CONTRACT,
        "seed": int(seed),
        "suite": str(suite_path),
        "expert_reference": str(expert_reference) if expert_reference else None,
        "stores": {"training": [str(path) for path in training_stores], "validation": [str(path) for path in validation_stores]},
        "rules": {
            "one_fork_per_root": True, "strata_in_order": list(STRATA), "quotas": quotas, "expert_failed_target": expert_failed,
            "breaker_only_classes": sorted(BREAKER_ONLY_CLASSES), "mixed_class_prefix": MIXED_PREFIX,
            "related_roots": "roots of one physical root stay in one split (validation is chosen first)",
            "fork": "a sampled decision on a candidate the verifier accepted, with the audit's label, in an episode whose "
                    "trace holds every observation up to it",
        },
        "natural_decisions": {"training": training_natural, "validation": validation_natural},
        "coverage": {"training": training_coverage, "validation": validation_coverage},
        "forks": training + validation,
    }


# --------------------------------------------------------------------------- branches


def branch_path(directory: Path, state: DecisionState, mode: str, choice: str, index: int) -> Path:
    return Path(directory) / f"{state.name}__{mode}__{choice}__{int(index)}.json"


def branch_jobs(manifest: Mapping[str, Any], modes: Sequence[str], continuations: int) -> list[tuple[int, str, str, int]]:
    """``(fork index, observation mode, forced choice, continuation index)`` of every branch of the probe."""

    return [(index, mode, choice, k) for index in range(len(manifest["forks"])) for mode in modes for choice in CHOICES
            for k in range(int(continuations))]


def run_branches(
    manifest: Mapping[str, Any],
    suite: Mapping[str, Mapping[str, Any]],
    *,
    output_dir: Path,
    env_factory: Callable[..., Any],
    continuation_factory: Callable[[int], Any],
    max_steps: int,
    case_loader: Callable[[Any], Any] | None,
    weights: Any,
    modes: Sequence[str] = OBSERVATION_MODES,
    continuations: int = 2,
    shard: tuple[int, int] | None = None,
    evaluator: Callable[..., Any] | None = None,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
    label: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the branches that are not there yet (a shard: every n-th fork's)."""

    output_dir.mkdir(parents=True, exist_ok=True)
    states = [DecisionState.from_dict(fork["state"]) for fork in manifest["forks"]]
    jobs = branch_jobs(manifest, modes, continuations)
    if shard is not None:
        jobs = [job for job in jobs if job[0] % shard[1] == shard[0]]
    ran = skipped = 0
    for index, mode, choice, k in jobs:
        state = states[index]
        path = branch_path(output_dir, state, mode, choice, k)
        if path.is_file():
            skipped += 1
            continue
        seed = branch_seed(int(manifest["seed"]), state, k)
        record = run_branch(suite[state.root], state, choice, continuation=continuation_factory(seed), continuation_index=k,
                            seed=seed, env_factory=env_factory, max_steps=max_steps, case_loader=case_loader, weights=weights,
                            observation_mode=mode, evaluator=evaluator)
        fork = manifest["forks"][index]
        record.update(fork_index=index, split=fork["split"], stratum=fork["stratum"], compound_class=fork["class"],
                      expert_success=fork["expert_success"], **dict(label or {}))
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, sort_keys=True, default=str) + "\n", encoding="utf-8", newline="\n")
        temporary.replace(path)
        ran += 1
        if progress is not None:
            progress({"fork": state.name, "mode": mode, "forced": choice, "continuation": k, "valid": record["valid"],
                      "success": record["success"], "ran": ran, "left": len(jobs) - ran - skipped})
    return {"jobs": len(jobs), "ran": ran, "already_there": skipped}


def load_branches(directory: str | Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(Path(directory).glob("*.json"))]


# --------------------------------------------------------------------------- the learner's own choice at a fork


def capture_views(
    manifest: Mapping[str, Any],
    suite: Mapping[str, Mapping[str, Any]],
    *,
    env_factory: Callable[..., Any],
    max_steps: int,
    case_loader: Callable[[Any], Any] | None,
    modes: Sequence[str] = OBSERVATION_MODES,
    evaluator: Callable[..., Any] | None = None,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """What a policy is shown at every fork in every observation mode, read from a replay through the environment."""

    views = []
    for index, fork in enumerate(manifest["forks"]):
        state = DecisionState.from_dict(fork["state"])
        for mode in modes:
            record = capture_fork_view(suite[state.root], state, observation_mode=mode, env_factory=env_factory, max_steps=max_steps,
                                       case_loader=case_loader, evaluator=evaluator)
            record.update(fork_index=index, root=state.root)
            views.append(record)
            if progress is not None:
                progress({"fork": state.name, "mode": mode, "matched": record["matched"],
                          "equals_sampled_view": record["equals_sampled_view"]})
    return views


def decide_forks(
    manifest: Mapping[str, Any],
    views: Sequence[Mapping[str, Any]],
    *,
    render_prompt: Callable[[Mapping[str, Any]], str],
    prompt_token_ids: Callable[[str], tuple[list[int], int]],
    render_target: Callable[[Mapping[str, Any], str, Mapping[str, Any]], Sequence[int]],
    logprob: Callable[[Sequence[int], Sequence[int]], float],
) -> list[dict[str, Any]]:
    """The probability a policy gives each choice at each fork under each captured view.

    ``views`` are :func:`capture_views`' records: the view a policy in that
    mode decodes from at the fork.  ``logprob(prompt ids, completion ids)``
    is the summed log-probability of a completion after a prompt under the
    policy.  The two completions are the same tool call but for the tool's
    name, so their probabilities, renormalized over the two, are the
    policy's commit and decline probabilities at the fork.  A fork whose
    episode recorded no generated text (an expert's) has nothing to render
    and is skipped, as is a view whose replay did not match.
    """

    decisions = []
    states = [DecisionState.from_dict(fork["state"]) for fork in manifest["forks"]]
    for record in views:
        index, mode = int(record["fork_index"]), str(record["observation_mode"])
        state = states[index]
        if not state.sampled_text or not record.get("matched") or record.get("view") is None:
            continue
        for view in (record["view"],):
            prompt_ids, original = prompt_token_ids(render_prompt(view))
            scores = {}
            for choice in CHOICES:
                target = [int(value) for value in render_target(view, state.sampled_text, {"label": choice, "tool": state.sampled_tool})]
                scores[choice] = float(logprob(prompt_ids, target))
            top = max(scores.values())
            mass = sum(math.exp(value - top) for value in scores.values())
            decisions.append({
                "fork_index": index, "root": state.root, "observation_mode": mode, "prompt_tokens": int(original),
                "prompt_cut": int(original) > len(prompt_ids), "equals_sampled_view": record.get("equals_sampled_view"),
                "logprob": scores,
                "p_decline": math.exp(scores[ROLLBACK_STATE] - top) / mass,
                # How much of the probability at this state is on the two choices at all.
                "probability_on_the_two_choices": sum(math.exp(value) for value in scores.values()),
            })
    return decisions


# --------------------------------------------------------------------------- the report


def _mean(values: Sequence[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _choice_values(branches: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [branch for branch in branches if branch.get("valid")]
    out: dict[str, Any] = {"valid": len(valid), "invalid": len(branches) - len(valid)}
    for choice in CHOICES:
        own = [branch for branch in valid if branch["forced_tool"] == choice]
        out[choice] = {"continuations": len(own), "solved": sum(1 for branch in own if branch["success"]),
                       "mean_reward": _mean([float(branch["reward"]["total"]) for branch in own])}
    commit, decline = out[COMMIT_STATE], out[ROLLBACK_STATE]
    known = commit["continuations"] and decline["continuations"]
    out["gap_commit_minus_decline"] = (commit["mean_reward"] - decline["mean_reward"]) if known else None
    out["better"] = None if not known else ("commit" if out["gap_commit_minus_decline"] > 1e-9 else
                                            "decline" if out["gap_commit_minus_decline"] < -1e-9 else "equal")
    return out


def _chain_counts(branches: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """The recovery chain over a set of forced branches, as counts of continuations that got that far."""

    valid = [branch for branch in branches if branch.get("valid")]
    chains = [branch.get("chain") or {} for branch in valid]
    return {
        "continuations": len(valid),
        "productive_target_offered": sum(1 for chain in chains if chain.get("productive_target_offered")),
        "next_correction_made": sum(1 for chain in chains if chain.get("next_correction")),
        "next_target_true": sum(1 for chain in chains if chain.get("next_target_is_true") is True),
        "next_correction_accepted": sum(1 for chain in chains if chain.get("next_target_is_true") is True and chain.get("next_correction_accepted")),
        "next_correction_committed": sum(1 for chain in chains if chain.get("next_target_is_true") is True
                                         and chain.get("next_correction_accepted") and chain.get("next_correction_committed")),
        "solved": sum(1 for branch in valid if branch.get("success")),
        # Counted apart from the outcome: what the continuation kept and what it declined.
        "continuations_keeping_a_wrong_target": sum(1 for branch in valid if any(
            decision.get("target_is_true") is False and not decision.get("declined") for decision in branch.get("lifecycle") or [])),
        "continuations_declining_a_true_target": sum(1 for branch in valid if any(
            decision.get("target_is_true") is True and decision.get("declined") for decision in branch.get("lifecycle") or [])),
    }


def build_report(manifest: Mapping[str, Any], branches: Sequence[Mapping[str, Any]],
                 decisions: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """The paired decision and continuation report of a fork probe."""

    forks = manifest["forks"]
    modes = sorted({str(branch["observation_mode"]) for branch in branches}) or list(OBSERVATION_MODES)
    by_fork: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for branch in branches:
        by_fork[(int(branch["fork_index"]), str(branch["observation_mode"]))].append(branch)
    decided = {(int(item["fork_index"]), str(item["observation_mode"])): item for item in decisions or []}
    selected = {name: sum(1 for fork in forks if fork["stratum"] == name) for name in STRATA}
    natural = defaultdict(int)
    for counts in (manifest.get("natural_decisions") or {}).values():
        for name, count in counts.items():
            natural[name] += int(count)
    rows = []
    for index, fork in enumerate(forks):
        state = fork["state"]
        label = (state.get("decision") or {}).get("label")
        entry: dict[str, Any] = {
            "fork_index": index, "root": state["root"], "split": fork["split"], "stratum": fork["stratum"], "class": fork["class"],
            "expert_success": fork["expert_success"], "sampled_tool": state["sampled_tool"],
            "target_is_true": (state.get("decision") or {}).get("target_is_true"), "membership_label": label,
            "source_success": state.get("source_success"), "edit_audit": fork.get("edit_audit"),
            # One fork stands for this many of the decisions the stores hold of its kind.
            "decision_weight": (natural[fork["stratum"]] / selected[fork["stratum"]]) if selected[fork["stratum"]] else None,
            "modes": {},
        }
        for mode in modes:
            own = by_fork.get((index, mode), [])
            values = _choice_values(own)
            values["chain_after_decline"] = _chain_counts([branch for branch in own if branch["forced_tool"] == ROLLBACK_STATE])
            values["chain_after_commit"] = _chain_counts([branch for branch in own if branch["forced_tool"] == COMMIT_STATE])
            choice = decided.get((index, mode))
            values["p_decline"] = None if choice is None else choice["p_decline"]
            values["label_against_value"] = (None if values["better"] in (None, "equal") or label not in CHOICES
                                             else (values["better"] == "commit") != (label == COMMIT_STATE))
            entry["modes"][mode] = values
        rows.append(entry)

    def table(items: Sequence[Mapping[str, Any]], mode: str, *, weighted: bool) -> dict[str, Any]:
        measured = [item for item in items if item["modes"][mode]["better"] is not None]
        weight = [float(item["decision_weight"] or 0.0) if weighted else 1.0 for item in measured]
        total = sum(weight)

        def share(test: Callable[[Mapping[str, Any]], bool]) -> float | None:
            return (sum(w for item, w in zip(measured, weight) if test(item)) / total) if total else None

        def average(read: Callable[[Mapping[str, Any]], float | None]) -> float | None:
            pairs = [(read(item), w) for item, w in zip(measured, weight) if read(item) is not None]
            mass = sum(w for _value, w in pairs)
            return (sum(value * w for value, w in pairs) / mass) if mass else None

        return {
            "forks": len(items), "forks_measured": len(measured),
            "decline_better": share(lambda item: item["modes"][mode]["better"] == "decline"),
            "commit_better": share(lambda item: item["modes"][mode]["better"] == "commit"),
            "equal": share(lambda item: item["modes"][mode]["better"] == "equal"),
            "mean_reward_commit": average(lambda item: item["modes"][mode][COMMIT_STATE]["mean_reward"]),
            "mean_reward_decline": average(lambda item: item["modes"][mode][ROLLBACK_STATE]["mean_reward"]),
            "mean_p_decline": average(lambda item: item["modes"][mode]["p_decline"]),
            "continuations_solved": {choice: sum(int(item["modes"][mode][choice]["solved"]) for item in measured) for choice in CHOICES},
            "continuations": {choice: sum(int(item["modes"][mode][choice]["continuations"]) for item in measured) for choice in CHOICES},
        }

    summary: dict[str, Any] = {}
    for mode in modes:
        summary[mode] = {
            "root_weighted": {"all": table(rows, mode, weighted=False),
                              **{name: table([row for row in rows if row["stratum"] == name], mode, weighted=False) for name in STRATA},
                              **{f"split:{split}": table([row for row in rows if row["split"] == split], mode, weighted=False)
                                 for split in ("training", "validation")}},
            "decision_weighted": {"all": table(rows, mode, weighted=True)},
            "chain_after_decline_of_a_wrong_target": _chain_counts([
                branch for branch in branches if branch.get("observation_mode") == mode and branch.get("forced_tool") == ROLLBACK_STATE
                and forks[int(branch["fork_index"])]["state"]["decision"].get("target_is_true") is False]),
            "chain_after_commit_of_a_true_target": _chain_counts([
                branch for branch in branches if branch.get("observation_mode") == mode and branch.get("forced_tool") == COMMIT_STATE
                and forks[int(branch["fork_index"])]["state"]["decision"].get("target_is_true") is True]),
            "label_against_value": {
                "forks_with_a_better_choice": sum(1 for row in rows if row["modes"][mode]["label_against_value"] is not None),
                "label_points_the_other_way": sum(1 for row in rows if row["modes"][mode]["label_against_value"]),
                "roots": sorted(row["root"] for row in rows if row["modes"][mode]["label_against_value"]),
            },
        }
    # The screening count: distinct expert-failed roots on which a continuation of the learner solved the case.
    # A branch another policy continued (the rule expert, in a wiring check) is counted beside it, never in it.
    policies = sorted({str(branch.get("continuation_policy")) for branch in branches})
    recovered: dict[str, set[str]] = {"training": set(), "validation": set()}
    recovered_by_mode: dict[str, set[str]] = {mode: set() for mode in modes}
    recovered_by_others: set[str] = set()
    for branch in branches:
        fork = forks[int(branch["fork_index"])]
        if branch.get("valid") and branch.get("success") and fork["expert_success"] is False:
            if str(branch.get("continuation_policy")) != LEARNER_POLICY:
                recovered_by_others.add(fork["state"]["root"])
                continue
            recovered[fork["split"]].add(fork["state"]["root"])
            recovered_by_mode[str(branch["observation_mode"])].add(fork["state"]["root"])
    distinct = recovered["training"] | recovered["validation"]
    applies = LEARNER_POLICY in policies
    others = [policy for policy in policies if policy != LEARNER_POLICY]
    gate = {
        "rule": f"learner continuations solve at least {GATE_EXPERT_FAILED_ROOTS} distinct expert-failed roots, one or more in the validation split",
        # Whether any branch was continued by the learner; without one the rule is not evaluated (``passed`` is None).
        "applies": applies,
        "continuation_policies": policies,
        "expert_failed_roots_probed": sum(1 for fork in forks if fork["expert_success"] is False),
        "expert_failed_roots_solved": len(distinct),
        "in_training": sorted(recovered["training"]), "in_validation": sorted(recovered["validation"]),
        "by_observation_mode": {mode: len(roots) for mode, roots in recovered_by_mode.items()},
        "solved_by_other_continuations": {"policies": others, "expert_failed_roots_solved": len(recovered_by_others),
                                          "roots": sorted(recovered_by_others)},
        "passed": (len(distinct) >= GATE_EXPERT_FAILED_ROOTS and bool(recovered["validation"])) if applies else None,
    }
    history = None
    if decisions and {BOUNDED_HISTORY, COMPLETE_HISTORY} <= set(modes):
        both = [row for row in rows if row["modes"][BOUNDED_HISTORY]["p_decline"] is not None
                and row["modes"][COMPLETE_HISTORY]["p_decline"] is not None]

        def moved(test: Callable[[Mapping[str, Any]], bool]) -> dict[str, Any]:
            items = [row for row in both if test(row)]
            before = [row["modes"][BOUNDED_HISTORY]["p_decline"] for row in items]
            after = [row["modes"][COMPLETE_HISTORY]["p_decline"] for row in items]
            return {"forks": len(items), "mean_p_decline_bounded": _mean(before), "mean_p_decline_complete": _mean(after),
                    "greedy_choice_changed": sum(1 for b, a in zip(before, after) if (b > 0.5) != (a > 0.5))}

        history = {"forks": len(both), "all_true_targets": moved(lambda row: row["target_is_true"] is True),
                   "all_wrong_targets": moved(lambda row: row["target_is_true"] is False),
                   **{name: moved(lambda row, name=name: row["stratum"] == name) for name in STRATA}}
    invalid = [branch for branch in branches if not branch.get("valid")]
    return {
        "contract": PROBE_CONTRACT,
        "forks": len(forks),
        "branches": {"total": len(branches), "valid": len(branches) - len(invalid),
                     "mismatched_prefix": sum(1 for branch in invalid if not branch.get("prefix_matched", True)),
                     "forced_choice_refused": sum(1 for branch in invalid if branch.get("prefix_matched", True) and not branch.get("forced_executed"))},
        "continuation_policy": sorted({str(branch.get("continuation_policy")) for branch in branches}),
        "coverage": manifest.get("coverage"),
        "summary": summary,
        "history_effect_on_the_decision": history,
        "screening_gate": gate,
        "edit_audit": {
            "parameter_candidates": sum(1 for fork in forks if (fork.get("edit_audit") or {}).get("lines_changed") is not None),
            "with_pair_refit": sum(1 for fork in forks if (fork.get("edit_audit") or {}).get("pair_refit")),
            "label_true_but_healthy_line_changed": sorted(fork["state"]["root"] for fork in forks
                                                         if (fork.get("edit_audit") or {}).get("label_true_but_healthy_line_changed")),
        },
        "forks_table": rows,
    }


def decision_panel(manifest: Mapping[str, Any], before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The policy's choices on the same forks before and after an update, by kind of fork and observation mode.

    The workplan's check of an update schedule: the intended choices move
    (wrong accepted candidates toward a decline) without the decline rate
    rising on every candidate alike.  ``separation`` is the chance that a
    wrong target gets a higher decline probability than a true one, which a
    shift of all probabilities leaves where it was.
    """

    forks = manifest["forks"]

    def read(decisions: Sequence[Mapping[str, Any]]) -> dict[tuple[int, str], float]:
        return {(int(item["fork_index"]), str(item["observation_mode"])): float(item["p_decline"]) for item in decisions}

    first, second = read(before), read(after)
    shared = sorted(set(first) & set(second))
    panel: dict[str, Any] = {"forks_scored_in_both": len({index for index, _mode in shared}), "modes": {}}
    for mode in sorted({mode for _index, mode in shared}):
        keys = [key for key in shared if key[1] == mode]

        def group(test: Callable[[Mapping[str, Any]], bool]) -> dict[str, Any]:
            own = [key for key in keys if test(forks[key[0]])]
            a, b = [first[key] for key in own], [second[key] for key in own]
            return {"forks": len(own), "mean_p_decline_before": _mean(a), "mean_p_decline_after": _mean(b),
                    "declined_before": sum(1 for value in a if value > 0.5), "declined_after": sum(1 for value in b if value > 0.5)}

        def separation(values: Mapping[tuple[int, str], float]) -> float | None:
            wrong = [values[key] for key in keys if forks[key[0]]["state"]["decision"].get("target_is_true") is False]
            true = [values[key] for key in keys if forks[key[0]]["state"]["decision"].get("target_is_true") is True]
            if not wrong or not true:
                return None
            return sum(1.0 if w > t else 0.5 if w == t else 0.0 for w in wrong for t in true) / (len(wrong) * len(true))

        panel["modes"][mode] = {
            "all_true_targets": group(lambda fork: fork["state"]["decision"].get("target_is_true") is True),
            "all_wrong_targets": group(lambda fork: fork["state"]["decision"].get("target_is_true") is False),
            **{name: group(lambda fork, name=name: fork["stratum"] == name) for name in STRATA},
            "separation_before": separation(first), "separation_after": separation(second),
        }
    return panel


def panel_markdown(panel: Mapping[str, Any], *, title: str = "Decision panel") -> str:
    lines = [f"# {title}", "", f"{panel['forks_scored_in_both']} forks scored before and after.", ""]
    for mode, entry in panel["modes"].items():
        lines += [f"## Observation mode {mode}", "",
                  f"Chance that a wrong target gets a higher decline probability than a true one: "
                  f"{_number(entry['separation_before'], 3)} before, {_number(entry['separation_after'], 3)} after.", "",
                  "| forks | n | p(decline) before | p(decline) after | declined before | declined after |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for name in ("all_true_targets", "all_wrong_targets", *STRATA):
            group = entry[name]
            if group["forks"]:
                lines.append(f"| {name} | {group['forks']} | {_number(group['mean_p_decline_before'], 3)} | "
                             f"{_number(group['mean_p_decline_after'], 3)} | {group['declined_before']} | {group['declined_after']} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def _number(value: Any, digits: int = 2) -> str:
    return "" if value is None else f"{float(value):.{digits}f}"


def markdown(report: Mapping[str, Any], *, title: str = "Fork probe") -> str:
    lines = [f"# {title}", "",
             f"{report['forks']} forks, {report['branches']['total']} branches ({report['branches']['valid']} valid; "
             f"{report['branches']['mismatched_prefix']} with a replay that did not match, "
             f"{report['branches']['forced_choice_refused']} with a refused forced choice). "
             f"Continuation policy: {', '.join(report['continuation_policy']) or 'not recorded'}.", ""]
    for split, coverage in (report.get("coverage") or {}).items():
        shortfalls = "; ".join(coverage.get("shortfalls") or []) or "none"
        lines.append(f"Coverage, {split}: {coverage['selected']} of {coverage['wanted']} forks; expert-failed roots "
                     f"{coverage['expert_failed']['selected']} (target {coverage['expert_failed']['target']}); shortfalls: {shortfalls}.")
    gate = report["screening_gate"]
    others = gate.get("solved_by_other_continuations") or {}
    beside = (f" Continuations by {', '.join(others['policies'])} solved {others['expert_failed_roots_solved']} of those roots; "
              "they are not the learner's and are not in the count." if others.get("policies") else "")
    if gate.get("applies", True):
        lines += ["", f"Screening count: {gate['expert_failed_roots_solved']} distinct expert-failed roots solved by a learner continuation, "
                      f"of {gate['expert_failed_roots_probed']} probed ({len(gate['in_training'])} training, "
                      f"{len(gate['in_validation'])} validation). Rule: {gate['rule']}. {'Met' if gate['passed'] else 'Not met'}.{beside}", ""]
    else:
        lines += ["", f"Screening count: not evaluated. No branch was continued by the learner, so this run checks the wiring and says "
                      f"nothing about the rule ({gate['rule']}).{beside}", ""]
    for mode, summary in report["summary"].items():
        lines += [f"## Observation mode {mode}", "",
                  "| kind of fork | forks | decline better | commit better | equal | reward if committed | reward if declined | "
                  "solved after commit | solved after decline | policy p(decline) |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |"]
        for name, entry in summary["root_weighted"].items():
            if not entry["forks"]:
                continue
            solved, runs = entry["continuations_solved"], entry["continuations"]
            lines.append(f"| {name} | {entry['forks_measured']} | {_number(entry['decline_better'])} | {_number(entry['commit_better'])} | "
                         f"{_number(entry['equal'])} | {_number(entry['mean_reward_commit'])} | {_number(entry['mean_reward_decline'])} | "
                         f"{solved[COMMIT_STATE]} of {runs[COMMIT_STATE]} | {solved[ROLLBACK_STATE]} of {runs[ROLLBACK_STATE]} | "
                         f"{_number(entry['mean_p_decline'], 3)} |")
        weighted = summary["decision_weighted"]["all"]
        lines += ["", f"Weighted by how many decisions of each kind the stores hold: decline better {_number(weighted['decline_better'])}, "
                      f"commit better {_number(weighted['commit_better'])}, equal {_number(weighted['equal'])}.", ""]
        for name, key in (("After a forced decline of a wrong target", "chain_after_decline_of_a_wrong_target"),
                          ("After a forced commit of a true target", "chain_after_commit_of_a_true_target")):
            chain = summary[key]
            lines.append(f"{name}: {chain['continuations']} continuations; a true target was offered in {chain['productive_target_offered']}, "
                         f"the next correction was on a true target in {chain['next_target_true']}, accepted in "
                         f"{chain['next_correction_accepted']}, committed in {chain['next_correction_committed']}; solved "
                         f"{chain['solved']}. Continuations that kept a wrong target {chain['continuations_keeping_a_wrong_target']}, "
                         f"that declined a true one {chain['continuations_declining_a_true_target']}.")
        conflict = summary["label_against_value"]
        lines += ["", f"Membership label against the measured values: {conflict['label_points_the_other_way']} of "
                      f"{conflict['forks_with_a_better_choice']} forks with a better choice have the label on the other one.", ""]
    history = report.get("history_effect_on_the_decision")
    if history:
        lines += ["## The policy's decline probability, bounded history against complete history", "",
                  "| forks | n | bounded | complete | greedy choice changed |", "| --- | ---: | ---: | ---: | ---: |"]
        for name in ("all_true_targets", "all_wrong_targets", *STRATA):
            entry = history[name]
            if entry["forks"]:
                lines.append(f"| {name} | {entry['forks']} | {_number(entry['mean_p_decline_bounded'], 3)} | "
                             f"{_number(entry['mean_p_decline_complete'], 3)} | {entry['greedy_choice_changed']} |")
        lines.append("")
    audit = report["edit_audit"]
    lines.append(f"Edit audit: {audit['parameter_candidates']} line-parameter candidates, {audit['with_pair_refit']} with a pair refit; "
                 f"labelled true while a healthy line was also changed: {len(audit['label_true_but_healthy_line_changed'])}.")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- commands


def _print(record: Mapping[str, Any]) -> None:
    print(json.dumps(dict(record), default=str), flush=True)


def _config(path: str, *, policy: str | None = None, mode: str | None = None) -> Any:
    from psse_env.rl.train import RLConfig

    from psse_env.rl.train import apply_prompt_limits

    config = RLConfig.load(path)
    changes: dict[str, Any] = {}
    if policy:
        changes["policy_source"] = policy
    if mode:
        changes["observation_mode"] = mode
    config = replace(config, **changes) if changes else config
    # The probe's rollouts, branches and scoring run under the config's prompt limit, as the driver's phases do.
    apply_prompt_limits(config)
    return config


def command_rollout(args: argparse.Namespace) -> int:
    from psse_env.dagger.release_factories import deterministic_case_loader
    from psse_env.rl.reward import RewardWeights
    from psse_env.rl.rollout import rollout_roots, store_summary
    from psse_env.rl.train import configure_research_environment, sampling_policy_factory

    config = _config(args.config, policy=args.policy, mode=args.observation_mode)
    suite = load_suite(args.suite)
    wanted = json.loads(Path(args.roots_file).read_text(encoding="utf-8"))
    wanted = wanted.get("roots") if isinstance(wanted, Mapping) else wanted
    missing = [root for root in wanted if root not in suite]
    if missing:
        raise SystemExit(f"{len(missing)} roots of {args.roots_file} are not in the suite (first: {missing[0]})")
    env_factory = configure_research_environment(config)
    store = TrajectoryStore(args.store)
    summary = rollout_roots(
        [suite[root] for root in wanted], store=store, samples=args.samples, base_seed=args.seed, env_factory=env_factory,
        policy_factory=sampling_policy_factory(config, args.adapter or config.initial_adapter, env_factory),
        max_steps=config.max_steps, case_loader=deterministic_case_loader, weights=RewardWeights.from_mapping(config.reward),
        progress=_print, evaluator_options=config.evaluator_options())
    summary.update(store=store_summary(store), policy=config.policy_source, adapter=args.adapter or config.initial_adapter,
                   observation_mode=config.observation_mode, roots=len(wanted))
    (Path(args.store) / "rollout_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str) + "\n",
                                                           encoding="utf-8", newline="\n")
    _print({"rollout": summary["completed_groups"], "skipped": summary["skipped_complete_groups"]})
    return 0


def command_select(args: argparse.Namespace) -> int:
    manifest = build_manifest(training_stores=args.store, validation_stores=args.validation_store or [], suite_path=args.suite,
                              expert_reference=args.expert_reference, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, sort_keys=True, default=str) + "\n", encoding="utf-8", newline="\n")
    _print({"forks": len(manifest["forks"]), "coverage": {split: {"selected": entry["selected"], "shortfalls": entry["shortfalls"]}
                                                          for split, entry in manifest["coverage"].items()}})
    return 0


def command_branch(args: argparse.Namespace) -> int:
    from psse_env.dagger.release_factories import deterministic_case_loader
    from psse_env.rl.reward import RewardWeights
    from psse_env.rl.train import configure_research_environment, parse_shard, sampling_policy_factory

    config = _config(args.config, policy=args.continuation)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    env_factory = configure_research_environment(config)
    adapter = args.adapter or config.initial_adapter
    result = run_branches(
        manifest, load_suite(args.suite), output_dir=args.output_dir, env_factory=env_factory,
        continuation_factory=sampling_policy_factory(config, adapter, env_factory), max_steps=config.max_steps,
        case_loader=deterministic_case_loader, weights=RewardWeights.from_mapping(config.reward), modes=args.modes,
        continuations=args.continuations, shard=parse_shard(args.shard), progress=_print,
        label={"continuation_policy": config.policy_source, "adapter": adapter if config.policy_source == "adapter" else None})
    _print(result)
    return 0


def command_views(args: argparse.Namespace) -> int:
    from psse_env.dagger.release_factories import deterministic_case_loader
    from psse_env.rl.train import configure_research_environment

    config = _config(args.config)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    views = capture_views(manifest, load_suite(args.suite), env_factory=configure_research_environment(config),
                          max_steps=config.max_steps, case_loader=deterministic_case_loader, modes=args.modes, progress=_print)
    payload = {"contract": PROBE_CONTRACT, "views": views,
               "summary": {"views": len(views), "matched": sum(1 for view in views if view["matched"]),
                           "equal_to_the_rollouts_own_view": sum(1 for view in views if view["equals_sampled_view"]),
                           "different_from_the_rollouts_own_view": sum(1 for view in views if view["equals_sampled_view"] is False)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, sort_keys=False, default=str) + "\n", encoding="utf-8", newline="\n")
    _print(payload["summary"])
    return 0


def command_panel(args: argparse.Namespace) -> int:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    before = json.loads(Path(args.before).read_text(encoding="utf-8"))
    after = json.loads(Path(args.after).read_text(encoding="utf-8"))
    panel = decision_panel(manifest, before["decisions"], after["decisions"])
    panel.update(before={"adapter": before.get("adapter"), "precision": before.get("precision")},
                 after={"adapter": after.get("adapter"), "precision": after.get("precision")})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(panel, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    text = panel_markdown(panel, title=args.title)
    args.output.with_suffix(".md").write_text(text, encoding="utf-8", newline="\n")
    print(text, flush=True)
    return 0


def command_decide(args: argparse.Namespace) -> int:
    import torch

    from psse_env.rl import train
    from psse_env.rl.rows import build_step_row
    from psse_env.rl.update import completion_logprobs

    config = _config(args.config)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    adapter = args.adapter or config.initial_adapter
    model, processor, side_inputs = train._load_trainable_model(config, adapter)
    render, prompt_ids, render_target = train._prompt_functions(config, adapter, processor)
    model.eval()

    def logprob(prompt: Sequence[int], completion: Sequence[int]) -> float:
        row = build_step_row(root="fork", sample_index=0, step=0, advantage=0.0, prompt_ids=prompt, completion_ids=completion,
                             max_length=config.max_length, required_side_inputs=side_inputs, lifecycle_term=1.0, supervised=True)
        with torch.no_grad():
            return float(completion_logprobs(model, row, temperature=config.temperature).sum())

    views = [view for view in json.loads(Path(args.views).read_text(encoding="utf-8"))["views"] if view["observation_mode"] in args.modes]
    decisions = decide_forks(manifest, views, render_prompt=render, prompt_token_ids=prompt_ids, render_target=render_target,
                             logprob=logprob)
    payload = {"contract": PROBE_CONTRACT, "adapter": adapter, "precision": "4bit" if config.load_in_4bit else "bf16",
               "temperature": config.temperature, "decisions": decisions}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    _print({"decisions": len(decisions)})
    return 0


def command_reference(args: argparse.Namespace) -> int:
    reference = expert_reference_from_stores(args.store)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reference, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    _print({"roots": len(reference), "solved": sum(1 for entry in reference.values() if entry["success"])})
    return 0


def command_candidates(args: argparse.Namespace) -> int:
    if not args.episodes and not args.store:
        raise SystemExit("name --episodes files or --store directories")
    if args.store and not args.suite:
        raise SystemExit("--store needs --suite for each root's class")
    records = candidate_records(episodes=args.episodes or [], stores=args.store or [], suite=load_suite(args.suite) if args.suite else None)
    audit = candidate_audit(records)
    audit["sources"] = {"episodes": [str(path) for path in args.episodes or []], "stores": [str(path) for path in args.store or []]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(audit, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    text = candidate_markdown(audit, title=args.title)
    args.output.with_suffix(".md").write_text(text, encoding="utf-8", newline="\n")
    print(text, flush=True)
    return 0


def command_report(args: argparse.Namespace) -> int:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    decisions = json.loads(Path(args.decisions).read_text(encoding="utf-8"))["decisions"] if args.decisions else None
    report = build_report(manifest, load_branches(args.branches), decisions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8", newline="\n")
    text = markdown(report, title=args.title)
    args.output.with_suffix(".md").write_text(text, encoding="utf-8", newline="\n")
    print(text, flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    rollout = commands.add_parser("rollout", help="episodes of the policy on a list of roots, into a trajectory store")
    rollout.add_argument("--config", required=True)
    rollout.add_argument("--suite", required=True, type=Path)
    rollout.add_argument("--roots-file", required=True, type=Path)
    rollout.add_argument("--store", required=True, type=Path)
    rollout.add_argument("--samples", type=int, default=2)
    rollout.add_argument("--seed", type=int, default=20261110)
    rollout.add_argument("--adapter", default=None)
    rollout.add_argument("--policy", choices=("adapter", "expert"), default=None)
    rollout.add_argument("--observation-mode", choices=OBSERVATION_MODES, default=None)
    select = commands.add_parser("select", help="freeze the fork sample")
    select.add_argument("--store", action="append", required=True, type=Path, help="a rollout store of training roots (repeatable)")
    select.add_argument("--validation-store", action="append", default=None, type=Path, help="a rollout store of the reserved roots")
    select.add_argument("--suite", required=True, type=Path)
    select.add_argument("--expert-reference", default=None, type=Path)
    select.add_argument("--seed", type=int, default=20261110)
    select.add_argument("--output", required=True, type=Path)
    branch = commands.add_parser("branch", help="run the forced branches")
    branch.add_argument("--manifest", required=True, type=Path)
    branch.add_argument("--suite", required=True, type=Path)
    branch.add_argument("--config", required=True)
    branch.add_argument("--output-dir", required=True, type=Path)
    branch.add_argument("--adapter", default=None)
    branch.add_argument("--continuation", choices=("adapter", "expert"), default=None)
    branch.add_argument("--modes", nargs="+", choices=OBSERVATION_MODES, default=list(OBSERVATION_MODES))
    branch.add_argument("--continuations", type=int, default=2)
    branch.add_argument("--shard", default=None)
    views = commands.add_parser("views", help="what the policy is shown at every fork in every observation mode (a replay; CPU)")
    views.add_argument("--manifest", required=True, type=Path)
    views.add_argument("--suite", required=True, type=Path)
    views.add_argument("--config", required=True)
    views.add_argument("--modes", nargs="+", choices=OBSERVATION_MODES, default=list(OBSERVATION_MODES))
    views.add_argument("--output", required=True, type=Path)
    panel = commands.add_parser("panel", help="the policy's choices on the same forks before and after an update")
    panel.add_argument("--manifest", required=True, type=Path)
    panel.add_argument("--before", required=True, type=Path)
    panel.add_argument("--after", required=True, type=Path)
    panel.add_argument("--output", required=True, type=Path)
    panel.add_argument("--title", default="Decision panel")
    decide = commands.add_parser("decide", help="the policy's commit and decline probabilities at every fork, per captured view")
    decide.add_argument("--manifest", required=True, type=Path)
    decide.add_argument("--views", required=True, type=Path)
    decide.add_argument("--config", required=True)
    decide.add_argument("--adapter", default=None)
    decide.add_argument("--modes", nargs="+", choices=OBSERVATION_MODES, default=list(OBSERVATION_MODES))
    decide.add_argument("--output", required=True, type=Path)
    report = commands.add_parser("report", help="the paired decision and continuation report")
    report.add_argument("--manifest", required=True, type=Path)
    report.add_argument("--branches", required=True, type=Path)
    report.add_argument("--decisions", default=None, type=Path)
    report.add_argument("--output", required=True, type=Path)
    report.add_argument("--title", default="Fork probe")
    reference = commands.add_parser("reference", help="the rule expert's result per root, from the stores of its own rollouts")
    reference.add_argument("--store", action="append", required=True, type=Path, help="a store of rollout --policy expert (repeatable)")
    reference.add_argument("--output", required=True, type=Path)
    candidates = commands.add_parser("candidates", help="how often an accepted line-parameter candidate is a true target, by class")
    candidates.add_argument("--episodes", action="append", default=None, type=Path, help="an episodes.jsonl of the compound study (repeatable)")
    candidates.add_argument("--store", action="append", default=None, type=Path, help="a rollout store of the driver (repeatable)")
    candidates.add_argument("--suite", default=None, type=Path, help="the suite of the stores' roots, for their class")
    candidates.add_argument("--output", required=True, type=Path)
    candidates.add_argument("--title", default="Candidate audit")
    args = parser.parse_args(argv)
    return {"rollout": command_rollout, "select": command_select, "branch": command_branch, "views": command_views,
            "decide": command_decide, "panel": command_panel, "report": command_report, "reference": command_reference,
            "candidates": command_candidates}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
