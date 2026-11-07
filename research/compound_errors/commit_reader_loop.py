"""Commit-record check, in the loop: the compound teacher with flagged corrections declined.

    python -m research.compound_errors.commit_reader_loop train \\
        --rows output/commit_record_check_20261105/check/rows.jsonl \\
        --suite output/compound_errors_20261027/suite.json \\
        --output output/commit_record_check_20261105/loop/readers.pkl
    python -m research.compound_errors.commit_reader_loop run \\
        --readers output/commit_record_check_20261105/loop/readers.pkl --level summary --budget 0.01 \\
        --suite output/compound_errors_20261027/suite.json --output-dir output/commit_record_check_20261105/loop/summary_1pct \\
        --workers 14 --env-option pair_refit_adjacent=1 --expert-option abstain_isolated_lines=1 \\
        --expert-option state_ambiguity=1

The offline check (``commit_record_check``) found that a learner tells a
wrong kept correction from a true one at the moment of the commit.  That is a
ranking; it does not say how many failed cases turn into solved ones when the
flagged corrections are declined.  This module measures it.

``train`` fits one reader per fold of roots on the at-commit rows of the
other folds (cross-fitting: every root is later run with a reader that never
saw it), for each information level of the offline check, and sets each
reader's threshold on its own training folds only, by inner cross-fitting, so
that a given share of the true corrections there would be declined.

``run`` rolls the teacher out again.  The teacher decides everything; one
wrapper stands between it and the environment: when the teacher commits a
verified candidate, the wrapper builds the same commit-time row as the
offline check from the episode so far, scores it with the root's reader and,
above the threshold, sends ``rollback_state`` instead.  The rollback goes
through the environment's ordinary path, so the teacher sees a rejected
hypothesis and moves to its next one.  Nothing else changes: same roots,
options, budget and truth audit.

The decline needs an environment built with the lifecycle choice
(``psse_env.actions.lifecycle_choice_of``, ``--env-option lifecycle_choice=1``;
``run`` turns it on unless ``--observe-only``).  Everywhere else the verdict
fixes the step after a verification and a rollback of an accepted candidate is
refused.  The first measurement of 2026-11-05 lifted that refusal by wrapping
the process gate and the state store at run time; with the option built the
same run reproduces it action for action.

``--decider audit`` replaces the reader by the truth audit: the wrapper declines
exactly the accepted candidates whose target is not a truly faulted component.
That is a privileged behaviour policy, never a deployable one.  It serves two
purposes: a labelled upper bound on what the lifecycle step can give under the
teacher's other rules, and the behaviour policy whose rollouts the cloning
pass of the RL plan copies, so that the states after a decline are the
teacher's own next steps.

``--observe-only`` never declines; it reproduces the stored trajectories and
writes the rows the wrapper would have scored, which must equal the offline
rows (the parity test of the wrapper's bookkeeping).
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from psse_env.oracle.expert_variants import compound_study_expert_variant  # noqa: E402
from research.compound_errors import cloning_rows, commit_record, rollout  # noqa: E402
from research.compound_errors import commit_record_check as check  # noqa: E402
from research.rl import decision_audit as audit  # noqa: E402
from research.rl.decision_audit import CORRECTION_TOOLS, parse_run_options  # noqa: E402

LOOP_CONTRACT = "compound_commit_reader_loop_v1"
COMMIT, ROLLBACK, RUN_WLS = "commit_state", "rollback_state", "run_wls"
LEVELS = {"expert": check.LEVEL_EXPERT, "summary": check.LEVEL_SUMMARY, "full": check.LEVEL_FULL}
BUDGETS = (0.005, 0.01, 0.02, 0.05)


# --------------------------------------------------------------------------- readers


def _model(seed: int) -> Any:
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_depth=4, min_samples_leaf=15,
                                          l2_regularization=1.0, class_weight="balanced", random_state=seed)


def train(rows_path: Path, suite_path: Path, output: Path, *, folds: int, seed: int) -> dict[str, Any]:
    rows = [row for row in check.load(rows_path) if row["at_commit"]]
    names, x_all = check._matrix(rows)
    y = np.asarray([int(row["wrong"]) for row in rows])
    roots = [str(row["scenario_id"]) for row in rows]
    suite_ids = [str(root["execution"]["scenario_id"]) for root in rollout.load_suite(suite_path)]
    assignment = check.fold_of(suite_ids, folds, seed)
    fold = np.asarray([assignment[root] for root in roots])
    position = sorted(check.POSITION_FEATURES)
    columns = check.level_columns(names)
    bundle: dict[str, Any] = {"contract": LOOP_CONTRACT, "folds": folds, "seed": seed, "fold_of": assignment, "levels": {}}
    for key, label in LEVELS.items():
        feature_names = columns[label] + position
        index = [names.index(name) for name in feature_names]
        entry: dict[str, Any] = {"names": feature_names, "models": [], "thresholds": [], "training": []}
        for held in range(folds):
            train_mask = fold != held
            x, target = x_all[train_mask][:, index], y[train_mask]
            train_roots = [root for root, keep in zip(roots, train_mask) if keep]
            # The threshold comes from the training folds alone: scores each training row gets from a model that did not see its root.
            inner = check.out_of_fold(x, target, train_roots, folds=max(2, folds - 1), seed=seed + 100 + held)
            true_scores = np.sort(inner[target == 0])[::-1]
            thresholds = {}
            for budget in BUDGETS:
                allowed = int(np.floor(budget * true_scores.size))
                thresholds[budget] = float(true_scores[allowed]) if allowed < true_scores.size else 0.0
            model = _model(seed)
            model.fit(x, target)
            entry["models"].append(model)
            entry["thresholds"].append(thresholds)
            entry["training"].append({"rows": int(train_mask.sum()), "wrong": int(target.sum()),
                                      "inner_wrong_above": {b: int((inner[target == 1] > t).sum()) for b, t in thresholds.items()}})
        bundle["levels"][key] = entry
        print(f"[train] {key}: {len(feature_names)} features; thresholds at 1% "
              + ", ".join(f"{t[0.01]:.3f}" for t in entry["thresholds"]), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        pickle.dump(bundle, stream)
    return bundle


# --------------------------------------------------------------------------- the wrapper


class DecliningPolicy:
    """The teacher, with one change: a commit the reader flags is sent as a rollback."""

    def __init__(self, expert: Any, *, scenario_id: str, model: Any, names: Sequence[str], threshold: float, observe_only: bool,
                 decider: str = "reader", truth: Mapping[str, Any] | None = None) -> None:
        self.expert = expert
        self.scenario_id = scenario_id
        # "reader": decline above the reader's threshold.  "audit": decline exactly the candidates on a
        # target the hidden truth does not name (privileged; a bound and the cloning pass's behaviour policy).
        self.decider = str(decider)
        self.truth = truth
        self.model = model
        self.names = list(names)
        self.threshold = float(threshold)
        self.observe_only = bool(observe_only)
        self.steps: list[dict[str, Any]] = []
        self.previous: dict[str, Any] | None = None
        self.previous_disposition: str | None = None
        self.decisions: list[dict[str, Any]] = []

    @property
    def release_policy_identity(self) -> Any:
        return getattr(self.expert, "release_policy_identity", None)

    def _record_previous(self, observation: Mapping[str, Any]) -> None:
        """The step the environment just executed, in the collector's form (the same form the offline rows were built from)."""
        if self.previous is None:
            return
        action = self.previous
        tool = str(action.get("tool"))
        output = observation.get("last_tool_output") if isinstance(observation.get("last_tool_output"), Mapping) else {}
        metrics = output.get("tool_metrics") if isinstance(output.get("tool_metrics"), Mapping) else {}
        record: dict[str, Any] = {
            "step": len(self.steps), "tool": tool, "arguments": dict(action.get("arguments") or {}),
            "status": output.get("execution_status"), "error_code": output.get("error_code"),
            "disposition": self.previous_disposition if tool in (COMMIT, ROLLBACK) else None,
            "active_state_id": output.get("active_state_id"), "candidate_state_id": output.get("candidate_state_id"),
        }
        if tool == RUN_WLS:
            record["metrics"] = {key: value for key, value in metrics.items() if key not in commit_record._DROPPED_WLS_KEYS}
            record["solve"] = copy.deepcopy(commit_record._CAPTURE.get(str(metrics.get("state_hash"))))
        elif tool in CORRECTION_TOOLS:
            record["metrics"] = dict(metrics)
        self.steps.append(record)
        self.previous = None
        self.previous_disposition = None

    def _commit_row(self, action: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str]:
        candidate = str((action.get("arguments") or {}).get("candidate_state_id"))
        verification = next((step for step in reversed(self.steps) if step["tool"] == RUN_WLS and step["status"] == "success"
                             and str(step["arguments"].get("state_id")) == candidate), None)
        resolved = bool(((verification or {}).get("metrics") or {}).get("no_material_anomaly_remaining"))
        disposition = "ACCEPT_FINAL" if resolved else "ACCEPT_PARTIAL"
        hypothetical = self.steps + [{"step": len(self.steps), "tool": COMMIT, "arguments": dict(action.get("arguments") or {}),
                                      "status": "success", "disposition": disposition}]
        rows, _account = check.episode_rows({"scenario_id": self.scenario_id, "class": None, "success": False, "corrections": [],
                                             "steps_full": hypothetical})
        own = [row for row in rows if row["at_commit"] and row["event_step"] == len(self.steps)]
        return (own[-1] if own else None), disposition

    def act(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        self._record_previous(observation)
        action = self.expert.act(observation)
        disposition = None
        if str(action.get("tool")) == COMMIT:
            try:
                row, disposition = self._commit_row(action)
            except Exception:  # under the audit decider the reader's row is only a record
                if self.decider != "audit":
                    raise
                row, disposition = {"correction_step": -1, "kind": None, "final": None, "features": {}}, None
            if row is not None:
                try:
                    vector = np.asarray([[check._f(row["features"].get(name)) for name in self.names]], dtype=float)
                    score = float(self.model.predict_proba(vector)[0, 1])
                except Exception:
                    if self.decider != "audit":
                        raise
                    score = float("nan")
                if self.decider == "audit":
                    correction = next((step for step in reversed(self.steps)
                                       if step["tool"] in CORRECTION_TOOLS and step["status"] == "success"), None)
                    wrong = correction is not None and self.truth is not None and rollout.target_is_true(
                        {"tool": correction["tool"], "arguments": {k: v for k, v in correction["arguments"].items() if k != "state_id"}},
                        self.truth) is False
                    declined = bool(wrong and not self.observe_only)
                else:
                    declined = bool(score > self.threshold and not self.observe_only)
                self.decisions.append({"step": len(self.steps), "correction_step": int(row["correction_step"]), "kind": row["kind"],
                                       "final": bool(row["final"]), "score": score, "threshold": self.threshold, "declined": declined,
                                       "features": row["features"] if self.observe_only else None})
                if declined:
                    candidate = (action.get("arguments") or {}).get("candidate_state_id")
                    action = {"tool": ROLLBACK, "arguments": {"candidate_state_id": candidate}}
                    disposition = "DECLINED"
        elif str(action.get("tool")) == ROLLBACK:
            disposition = "REJECT"
        self.previous = copy.deepcopy(dict(action))
        self.previous_disposition = disposition
        return action


# --------------------------------------------------------------------------- the rollout

_READERS: dict[str, Any] = {}


def _init_worker(settings: Mapping[str, Any], reader_path: str, level: str, budget: float, observe_only: bool,
                 decider: str = "reader", export_rows: bool = False, behaviour: str = "teacher") -> None:
    commit_record.install_capture()
    audit._init_worker(settings)
    with open(reader_path, "rb") as stream:
        bundle = pickle.load(stream)
    _READERS.update(bundle=bundle, level=level, budget=float(budget), observe_only=bool(observe_only), decider=str(decider),
                    export_rows=bool(export_rows), behaviour=str(behaviour))


def _run_root(payload: Mapping[str, Any]) -> dict[str, Any]:
    harness = audit._HARNESS
    root = payload["root"]
    scenario_id = str(root["execution"]["scenario_id"])
    bundle, level = _READERS["bundle"], _READERS["level"]
    # A case outside the reader's suite (a fresh draw under the audit decider) is scored by fold 0's reader, for the record only.
    fold = int(bundle["fold_of"].get(scenario_id, 0))
    entry = bundle["levels"][level]
    started = time.perf_counter()
    commit_record._CAPTURE.clear()
    policy = DecliningPolicy(harness.expert_policy(), scenario_id=scenario_id, model=entry["models"][fold], names=entry["names"],
                             threshold=entry["thresholds"][fold][_READERS["budget"]], observe_only=_READERS["observe_only"],
                             decider=_READERS.get("decider", "reader"), truth=rollout.truth_of(root))
    try:
        episode = harness.run(root, policy)
        record = rollout.compact_episode(episode, root)
        truth = rollout.truth_of(root)
        by_step = {int(c["step"]): c for c in record["corrections"]}
        for decision in policy.decisions:
            correction = by_step.get(decision["correction_step"]) or {}
            decision["tool"] = correction.get("tool")
            decision["arguments"] = correction.get("arguments")
            decision["target_is_true"] = rollout.target_is_true(correction, truth) if correction else None
        record["gate_decisions"] = policy.decisions
        record["false_rollback_count"] = int(episode.get("false_rollback_count") or 0)
        record["decline_accounting"] = dict(episode.get("decline_accounting") or {})
        record["declines"] = sum(1 for decision in policy.decisions if decision.get("declined"))
        if _READERS.get("export_rows"):
            # The cloning rows of this episode (the parent writes them and drops them from the episode record).
            rows, refused = cloning_rows.episode_chat_rows(episode, root, behaviour=_READERS["behaviour"])
            record["_rows"] = rows
            record["rows_exported"] = len(rows)
            record["rows_refused"] = refused
    except Exception as exc:  # keep the sweep going; the record names the failure
        compound = (root.get("audit") or {}).get("compound") or {}
        record = {"scenario_id": scenario_id, "class": compound.get("class"), "success": False, "error": f"{type(exc).__name__}: {exc}",
                  "steps": 0, "corrections": [], "tools": [], "trace": [], "gate_decisions": []}
    record["contract"] = LOOP_CONTRACT
    record["fold"] = fold
    record["seconds"] = round(time.perf_counter() - started, 1)
    return record


def run(args: argparse.Namespace) -> int:
    roots = rollout.load_suite(args.suite)
    if args.roots:
        wanted = set(args.roots)
        roots = [root for root in roots if str(root["execution"]["scenario_id"]) in wanted]
    if args.per_class:
        taken: Counter = Counter()
        kept = []
        for root in roots:
            name = ((root.get("audit") or {}).get("compound") or {}).get("class")
            if taken[name] < args.per_class:
                taken[name] += 1
                kept.append(root)
        roots = kept
    system = rollout.suite_system(roots, args.system)
    args.progress_basis = rollout.system_progress_basis(system, args.progress_basis)
    args.expert = args.expert or compound_study_expert_variant(system)
    settings = dict(profile=args.profile, expert_variant=args.expert, triage_model=None, progress_basis=args.progress_basis,
                    max_steps=args.max_steps, seed=args.seed, reward=None,
                    env_options=parse_run_options(args.env_option), expert_options=parse_run_options(args.expert_option))
    if not args.observe_only:
        # The policy may decline only in an environment built with the lifecycle choice.
        settings["env_options"]["lifecycle_choice"] = True
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    print(f"[reader loop] {len(roots)} roots, decider {args.decider}, level {args.level}, budget {args.budget}, observe only {args.observe_only}, "
          f"workers {args.workers}", flush=True)
    started = time.perf_counter()
    payloads = [{"root": root} for root in roots]
    behaviour = args.behaviour or f"{args.expert}+{args.decider}_decides"
    initargs = (settings, str(args.readers), args.level, args.budget, args.observe_only, args.decider, args.export_rows, behaviour)
    done = solved = 0
    row_total = 0
    row_stream = (out / "rows.jsonl").open("w", encoding="utf-8") if args.export_rows else None

    def write_rows(record: dict[str, Any]) -> None:
        nonlocal row_total
        rows = record.pop("_rows", None) or []
        if row_stream is not None:
            for row in rows:
                row_stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            row_total += len(rows)

    with (out / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=initargs) as pool:
                for record in pool.map(_run_root, payloads, chunksize=1):
                    write_rows(record)
                    stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")
                    stream.flush()
                    done += 1
                    solved += bool(record.get("success"))
                    if done % 25 == 0 or done == len(payloads):
                        print(json.dumps({"done": done, "solved": solved, "elapsed": round(time.perf_counter() - started, 1)}), flush=True)
        else:
            _init_worker(*initargs)
            for payload in payloads:
                record = _run_root(payload)
                write_rows(record)
                stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")
                stream.flush()
                done += 1
                solved += bool(record.get("success"))
                print(json.dumps({"root": record["scenario_id"], "success": record.get("success"), "steps": record.get("steps"),
                                  "declined": sum(d["declined"] for d in record.get("gate_decisions") or []),
                                  "error": record.get("error")}), flush=True)
    if row_stream is not None:
        row_stream.close()
    manifest = dict(contract=LOOP_CONTRACT, suite=str(args.suite), readers=str(args.readers), level=args.level, budget=args.budget,
                    decider=args.decider, system=system, progress_basis=args.progress_basis, rows=row_total, behaviour=behaviour,
                    observe_only=args.observe_only, profile=args.profile, expert=args.expert, max_steps=args.max_steps, seed=args.seed,
                    roots=len(roots), solved=solved, env_options=settings["env_options"], expert_options=settings["expert_options"],
                    workers=args.workers, total_seconds=round(time.perf_counter() - started, 1))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest), flush=True)
    return 0


# --------------------------------------------------------------------------- comparison with the teacher


def _kept(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [c for c in record.get("corrections") or [] if c.get("lifecycle") == COMMIT]


def _exact_two_sided(won: int, lost: int) -> float:
    """Exact sign test on the discordant pairs (McNemar)."""
    from math import comb

    n = won + lost
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(0, min(won, lost) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def compare(reference_path: Path, runs: Sequence[tuple[str, Path]]) -> tuple[dict[str, Any], str]:
    reference = {str(r["scenario_id"]): r for r in check.load(reference_path)}
    report: dict[str, Any] = {"reference": str(reference_path), "runs": {}}
    base_solved = sum(bool(r["success"]) for r in reference.values())
    base_wrong = sum(1 for r in reference.values() for c in _kept(r) if c.get("target_is_true") is False)
    base_wrong_episodes = sum(1 for r in reference.values() if any(c.get("target_is_true") is False for c in _kept(r)))
    base_steps = sum(int(r["steps"]) for r in reference.values()) / len(reference)
    base_steps_solved = sum(int(r["steps"]) for r in reference.values() if r["success"]) / max(1, base_solved)
    report["teacher"] = {"solved": base_solved, "episodes": len(reference), "wrong_kept": base_wrong,
                         "episodes_keeping_wrong": base_wrong_episodes, "mean_actions": base_steps, "mean_actions_solved": base_steps_solved}
    lines = ["# The teacher with flagged corrections declined", "",
             f"Teacher as stored: solved {base_solved} of {len(reference)}, wrong corrections kept {base_wrong} in {base_wrong_episodes} episodes, "
             f"mean actions {base_steps:.2f} ({base_steps_solved:.2f} on solved cases).", "",
             "| reader reads | solved | won / lost | exact p | declined: wrong / true target | wrong corrections kept | episodes keeping one | "
             "mean actions | mean actions, solved | untouched episodes identical |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    classes = sorted({str(r.get("class")) for r in reference.values()})
    per_run = []
    for label, path in runs:
        records = check.load(path)
        by_id = {str(r["scenario_id"]): r for r in records}
        won = [i for i, r in by_id.items() if r["success"] and not reference[i]["success"]]
        lost = [i for i, r in by_id.items() if not r["success"] and reference[i]["success"]]
        declines = [d for r in records for d in r.get("gate_decisions") or [] if d.get("declined")]
        wrong_declined = sum(1 for d in declines if d.get("target_is_true") is False)
        true_declined = sum(1 for d in declines if d.get("target_is_true") is True)
        wrong_kept = sum(1 for r in records for c in _kept(r) if c.get("target_is_true") is False)
        wrong_episodes = sum(1 for r in records if any(c.get("target_is_true") is False for c in _kept(r)))
        untouched = [r for r in records if not any(d.get("declined") for d in r.get("gate_decisions") or [])]
        identical = sum(1 for r in untouched if r.get("tools") == reference[str(r["scenario_id"])].get("tools"))
        solved = sum(bool(r["success"]) for r in records)
        solved_records = [r for r in records if r["success"]]
        subset_reference = sum(bool(reference[i]["success"]) for i in by_id)
        entry = {
            "episodes": len(records), "errors": sum(bool(r.get("error")) for r in records), "solved": solved,
            "teacher_solved_same_cases": subset_reference, "won": len(won), "lost": len(lost),
            "p_exact": _exact_two_sided(len(won), len(lost)), "declined": len(declines), "declined_wrong": wrong_declined,
            "declined_true": true_declined, "episodes_with_decline": len(records) - len(untouched), "wrong_kept": wrong_kept,
            "episodes_keeping_wrong": wrong_episodes, "mean_actions": sum(int(r["steps"]) for r in records) / max(1, len(records)),
            "mean_actions_solved": sum(int(r["steps"]) for r in solved_records) / max(1, len(solved_records)),
            "untouched": len(untouched), "untouched_identical": identical, "loops": sum(bool(r.get("loop_detected")) for r in records),
            "won_ids": sorted(won), "lost_ids": sorted(lost),
            "lost_with_true_declined": sum(1 for i in lost if any(d.get("declined") and d.get("target_is_true") is True
                                                                  for d in by_id[i].get("gate_decisions") or [])),
            "by_class": {name: {"reference": sum(bool(reference[i]["success"]) for i in by_id if str(reference[i].get("class")) == name),
                                "run": sum(bool(r["success"]) for r in records if str(r.get("class")) == name),
                                "n": sum(1 for r in records if str(r.get("class")) == name)} for name in classes},
            "won_by_teacher_outcome": dict(Counter(
                "teacher kept a wrong correction" if any(c.get("target_is_true") is False for c in _kept(reference[i]))
                else "teacher kept no wrong correction" for i in won)),
        }
        report["runs"][label] = entry
        lines.append(f"| {label} | {solved} (teacher {subset_reference}) | +{len(won)} / -{len(lost)} | {entry['p_exact']:.2g} | "
                     f"{wrong_declined} / {true_declined} | {wrong_kept} | {wrong_episodes} | {entry['mean_actions']:.2f} | "
                     f"{entry['mean_actions_solved']:.2f} | {identical} of {len(untouched)} |")
        per_run.append((label, entry))
    lines += ["", "Solved by class (teacher as stored, then each run):", "",
              "| class (cases) | teacher | " + " | ".join(label for label, _entry in per_run) + " |",
              "| --- | --- | " + " | ".join("---" for _ in per_run) + " |"]
    for name in classes:
        cells = [str(entry["by_class"][name]["run"]) for _label, entry in per_run]
        first = per_run[0][1]["by_class"][name] if per_run else {"n": 0, "reference": 0}
        lines.append(f"| {name} ({first['n']}) | {first['reference']} | " + " | ".join(cells) + " |")
    for label, entry in per_run:
        lines += ["", f"{label}: {entry['episodes_with_decline']} episodes with a decline, {entry['declined']} declines; lost cases with a true "
                      f"correction declined {entry['lost_with_true_declined']} of {entry['lost']}; loops {entry['loops']}; errors {entry['errors']}; "
                      f"won cases, by what the teacher did there: {entry['won_by_teacher_outcome']}."]
    return report, "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    comparer = commands.add_parser("compare")
    comparer.add_argument("--reference", required=True, type=Path)
    comparer.add_argument("--run", action="append", required=True, metavar="LABEL=EPISODES_JSONL")
    comparer.add_argument("--output", required=True, type=Path)
    trainer = commands.add_parser("train")
    trainer.add_argument("--rows", required=True, type=Path)
    trainer.add_argument("--suite", required=True, type=Path)
    trainer.add_argument("--output", required=True, type=Path)
    trainer.add_argument("--folds", type=int, default=5)
    trainer.add_argument("--seed", type=int, default=0)
    runner = commands.add_parser("run")
    runner.add_argument("--readers", required=True, type=Path)
    runner.add_argument("--level", choices=sorted(LEVELS), default="summary")
    runner.add_argument("--budget", type=float, choices=BUDGETS, default=0.01)
    runner.add_argument("--observe-only", action="store_true")
    runner.add_argument("--decider", choices=("reader", "audit"), default="reader",
                        help="who decides the lifecycle step: the cross-fitted reader, or the truth audit (privileged)")
    runner.add_argument("--suite", required=True, type=Path)
    runner.add_argument("--output-dir", required=True, type=Path)
    runner.add_argument("--profile", default="scada_only")
    runner.add_argument("--expert", default=None, help="unnamed: the compound study's teacher for the suite's system")
    runner.add_argument("--system", default=None, help="the suite's registered system (default: read from the roots)")
    runner.add_argument("--progress-basis", default=None)
    runner.add_argument("--export-rows", action="store_true",
                        help="write the cloning rows of every episode to rows.jsonl (research.compound_errors.cloning_rows)")
    runner.add_argument("--behaviour", default=None, help="label of the behaviour policy in the rows")
    runner.add_argument("--max-steps", type=int, default=40)
    runner.add_argument("--seed", type=int, default=20261006)
    runner.add_argument("--roots", nargs="*", default=None)
    runner.add_argument("--per-class", type=int, default=None)
    runner.add_argument("--workers", type=int, default=1)
    runner.add_argument("--env-option", action="append", default=None, metavar="KEY=VALUE")
    runner.add_argument("--expert-option", action="append", default=None, metavar="KEY=VALUE")
    args = parser.parse_args(argv)
    if args.command == "train":
        train(args.rows, args.suite, args.output, folds=args.folds, seed=args.seed)
        return 0
    if args.command == "compare":
        runs = [(item.partition("=")[0], Path(item.partition("=")[2])) for item in args.run]
        report, text = compare(args.reference, runs)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix(".json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        args.output.with_suffix(".md").write_text(text + "\n", encoding="utf-8")
        print(text, flush=True)
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
