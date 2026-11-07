"""Commit-record check, collection: the compound teacher's rollout with every solve kept in full.

    python -m research.compound_errors.commit_record --suite output/compound_errors_20261027/suite.json \\
        --output-dir output/commit_record_check_20261105/collect --workers 20 \\
        --env-option pair_refit_adjacent=1 --expert-option abstain_isolated_lines=1 \\
        --expert-option state_ambiguity=1

The question the check asks (2026-11-05): an accepted correction's ledger
entry holds only its action, while a rejected one keeps its verification
numbers.  Would a fuller record of a kept correction (its verification at
commit, the estimate it wrote, its own target metric re-read on every later
solve) let an agent tell a wrong kept correction from a true one?

The stored rollouts keep a slim trace, which cannot answer that.  This module
re-runs the same teacher on the same roots under the same options and keeps,
for every policy step, the tool's own metrics, and for every WLS solve the
full signed normalized residual vector, the normalized multipliers of every
branch, the measurement vector with its sigma and the branch table of the
solved case.  Everything kept is a quantity of the deployed solve on the
agent's own model; nothing reads the hidden truth.  The capture wraps
``MatpowerDeploymentProviders.run_wls`` and only copies what that call
computed, so the trajectories must reproduce the stored ones action for
action; ``research.compound_errors.commit_record_check`` verifies that
before it reads a number.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from psse_env.actions import RUN_WLS  # noqa: E402
from psse_env.oracle.expert_variants import COMPOUND_STUDY_EXPERT_VARIANT  # noqa: E402
from research.compound_errors import rollout  # noqa: E402
from research.rl import decision_audit as audit  # noqa: E402
from research.rl.decision_audit import CORRECTION_TOOLS, parse_run_options  # noqa: E402

COLLECT_CONTRACT = "compound_commit_record_collect_v1"
#: Solve captures of the episode in progress, by the solved state's hash.
_CAPTURE: dict[str, dict[str, Any]] = {}
_DROPPED_WLS_KEYS = {"steady_state_physical_evidence", "evidence_sufficiency"}
F_BUS, T_BUS, BR_R, BR_X, TAP, BR_STATUS = 0, 1, 2, 3, 8, 10


def _rounded(values: Any, digits: int) -> list[float]:
    return [round(float(value), digits) for value in (values if values is not None else [])]


def solve_capture(solved: Mapping[str, Any], metrics: Mapping[str, Any]) -> dict[str, Any]:
    """What one deployed solve computed, in full, on the agent's own model."""
    payload = solved["payload"]
    bus = np.asarray(solved["ppc"]["bus"], dtype=float)
    branch = np.asarray(solved["ppc"]["branch"], dtype=float)
    options = solved.get("noise_options") or {}
    return {
        "state_id": metrics.get("state_id"),
        "state_hash": metrics.get("state_hash"),
        "chi_square_statistic": metrics.get("chi_square_statistic"),
        "nb": int(solved["nb"]),
        "nl": int(solved["nl"]),
        "bus": [int(value) for value in bus[:, 0]],
        "branch": [[int(row[F_BUS]), int(row[T_BUS]), float(row[BR_R]), float(row[BR_X]), float(row[TAP]), int(row[BR_STATUS])]
                   for row in branch],
        "z": _rounded(solved.get("z"), 7),
        "sigma": _rounded(options.get("measurement_sigma"), 7),
        "exact": [int(index) for index in options.get("exact_measurement_indices") or []],
        "r": _rounded(payload.get("r"), 4),
        "lam": _rounded(payload.get("lambdaN"), 4),
    }


def install_capture() -> None:
    """Wrap ``run_wls`` so each call leaves its solve in ``_CAPTURE``; the call itself is untouched."""
    from psse_env.providers.matpower import MatpowerDeploymentProviders as Providers

    if getattr(Providers, "_commit_record_capture", False):
        return
    original = Providers.run_wls

    def run_wls(self: Any, state: Mapping[str, Any]) -> dict[str, Any]:
        box: dict[str, Any] = {}
        inner = self._solve

        def solve(inner_state: Mapping[str, Any]) -> dict[str, Any]:
            solved = inner(inner_state)
            box["solved"] = solved
            return solved

        self._solve = solve
        try:
            metrics = original(self, state)
        finally:
            self.__dict__.pop("_solve", None)
        solved = box.get("solved")
        if solved is not None and isinstance(metrics, Mapping) and metrics.get("state_hash") is not None \
                and (solved.get("payload") or {}).get("success"):
            _CAPTURE[str(metrics["state_hash"])] = solve_capture(solved, metrics)
        return metrics

    Providers.run_wls = run_wls
    Providers._commit_record_capture = True


def full_steps(steps: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Every policy step with the state it acted on; solves and corrections keep their own metrics in full."""
    slim = rollout.slim_trace(steps)
    out = []
    for index, step in enumerate(steps):
        action = step.get("action") or {}
        tool = str(action.get("tool"))
        output = step.get("policy_tool_output") if isinstance(step.get("policy_tool_output"), Mapping) else {}
        metrics = output.get("tool_metrics") if isinstance(output.get("tool_metrics"), Mapping) else {}
        record: dict[str, Any] = {
            "step": index, "tool": tool, "arguments": dict(action.get("arguments") or {}),
            "status": step.get("execution_status"), "error_code": step.get("error_code"),
            "disposition": step.get("candidate_disposition_offline"),
            "active_state_id": output.get("active_state_id"), "candidate_state_id": output.get("candidate_state_id"),
        }
        if tool == RUN_WLS:
            record["metrics"] = {key: value for key, value in metrics.items() if key not in _DROPPED_WLS_KEYS}
            record["solve"] = _CAPTURE.get(str(metrics.get("state_hash")))
        elif tool in CORRECTION_TOOLS:
            record["metrics"] = dict(metrics)
        else:
            for key in ("ranking", "comparison"):
                if key in slim[index]:
                    record[key] = slim[index][key]
        out.append(record)
    return out


def _init_worker(settings: Mapping[str, Any]) -> None:
    install_capture()
    audit._init_worker(settings)


def _run_root(payload: Mapping[str, Any]) -> dict[str, Any]:
    harness = audit._HARNESS
    root = payload["root"]
    started = time.perf_counter()
    _CAPTURE.clear()
    try:
        episode = harness.run(root, harness.expert_policy())
        record = rollout.compact_episode(episode, root)
        record["steps_full"] = full_steps(audit.policy_steps(episode.get("trace") or []))
    except Exception as exc:  # keep the sweep going; the record names the failure
        compound = (root.get("audit") or {}).get("compound") or {}
        record = {"scenario_id": str(root["execution"]["scenario_id"]), "class": compound.get("class"), "success": False,
                  "error": f"{type(exc).__name__}: {exc}", "steps": 0, "corrections": [], "tools": [], "trace": [], "steps_full": []}
    record["contract"] = COLLECT_CONTRACT
    record["seconds"] = round(time.perf_counter() - started, 1)
    return record


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--profile", default="scada_only")
    parser.add_argument("--expert", default=COMPOUND_STUDY_EXPERT_VARIANT)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--roots", nargs="*", default=None)
    parser.add_argument("--per-class", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--env-option", action="append", default=None, metavar="KEY=VALUE")
    parser.add_argument("--expert-option", action="append", default=None, metavar="KEY=VALUE")
    args = parser.parse_args(argv)

    roots = rollout.load_suite(args.suite)
    if args.roots:
        wanted = set(args.roots)
        roots = [root for root in roots if str(root["execution"]["scenario_id"]) in wanted]
    if args.per_class:
        taken: dict[Any, int] = {}
        kept = []
        for root in roots:
            name = ((root.get("audit") or {}).get("compound") or {}).get("class")
            if taken.get(name, 0) < args.per_class:
                taken[name] = taken.get(name, 0) + 1
                kept.append(root)
        roots = kept
    settings = dict(profile=args.profile, expert_variant=args.expert, triage_model=None, progress_basis=None,
                    max_steps=args.max_steps, seed=args.seed, reward=None,
                    env_options=parse_run_options(args.env_option), expert_options=parse_run_options(args.expert_option))
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    print(f"[commit-record collect] {len(roots)} roots from {args.suite} ({args.expert}), workers {args.workers}", flush=True)
    started = time.perf_counter()
    payloads = [{"root": root} for root in roots]
    done = solved = 0
    with (out / "episodes_full.jsonl").open("w", encoding="utf-8") as stream:
        if args.workers > 1:
            with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(settings,)) as pool:
                iterator = pool.map(_run_root, payloads, chunksize=1)
                for record in iterator:
                    stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")
                    stream.flush()
                    done += 1
                    solved += bool(record.get("success"))
                    if done % 25 == 0 or done == len(payloads):
                        print(json.dumps({"done": done, "solved": solved, "elapsed": round(time.perf_counter() - started, 1)}), flush=True)
        else:
            _init_worker(settings)
            for payload in payloads:
                record = _run_root(payload)
                stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")
                stream.flush()
                done += 1
                solved += bool(record.get("success"))
                print(json.dumps({"root": record["scenario_id"], "success": record.get("success"), "steps": record.get("steps"),
                                  "seconds": record.get("seconds"), "error": record.get("error")}), flush=True)
    manifest = dict(contract=COLLECT_CONTRACT, suite=str(args.suite), profile=args.profile, expert=args.expert,
                    max_steps=args.max_steps, seed=args.seed, roots=len(roots), solved=solved,
                    env_options=settings["env_options"], expert_options=settings["expert_options"],
                    workers=args.workers, total_seconds=round(time.perf_counter() - started, 1))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
