"""Truth-built demonstrations in the minimal environment, and their training rows.

    python -m research.minimal_env.build_demonstrations \\
        --suite output/compound_rl_20261106/train_ieee14/suite.json \\
        --output-dir output/minimal_env_20261115/demonstrations_train_ieee14 --workers 20

For every case of the suite: the success path (``psse_env.minimal.truth.build_success``)
and, with ``--recoveries N``, N recovery paths (a correction on a healthy component, then
``undo`` and the rest of the true corrections).  Every sequence is executed here, through
``MinimalDiagnosisEnv``; nothing is taken from an older trace.

Outputs under ``--output-dir``:

``episodes.jsonl``
    one record per sequence: its actions and results, its score against the truth
    (object-level and the strict audit), the tool cost.  No observations.
``rows.jsonl``
    one supervised row per target step (omitted with ``--no-rows``).  Each row passed the
    schema half of the trainer's row gate when it was written.
``manifest.json``
    settings, counts by class, prompt sizes, the code's commit.

A held-out suite may be run with ``--no-rows`` for reference numbers; its rows must never
be trained on.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psse_env.minimal import truth  # noqa: E402
from psse_env.minimal.env import DEFAULT_MAX_ACTIONS, DEFAULT_TABLE_VERSIONS, MinimalDiagnosisEnv  # noqa: E402
from psse_env.minimal.episode import compact  # noqa: E402
from psse_env.minimal.kernel import DEFAULT_PROVIDER_OPTIONS, NumericalKernel  # noqa: E402
from psse_env.minimal.rows import demonstration_rows, validate_row  # noqa: E402

BUILD_CONTRACT = "minimal_demonstration_build_v1"
_ENV: MinimalDiagnosisEnv | None = None
_SETTINGS: dict[str, Any] = {}


def load_suite(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        payload = payload.get("scenarios") or payload.get("roots") or []
    return [dict(item) for item in payload]


def class_of(root: Mapping[str, Any]) -> str:
    return str(((root.get("audit") or {}).get("compound") or {}).get("class") or (root.get("grouping") or {}).get("scenario_family") or "")


def _init_worker(settings: Mapping[str, Any]) -> None:
    global _ENV, _SETTINGS
    _SETTINGS = dict(settings)
    _ENV = MinimalDiagnosisEnv(NumericalKernel(**settings["provider_options"]), max_actions=int(settings["max_actions"]),
                               table_versions=settings["table_versions"])


def _prompt_chars(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    return [sum(len(str(message.get("content") or "")) for message in row["messages"][:2]) for row in rows]


def _run_root(root: Mapping[str, Any]) -> dict[str, Any]:
    assert _ENV is not None
    started = time.perf_counter()
    success = truth.build_success(root, _ENV, max_orders=int(_SETTINGS["max_orders"]),
                                  opening_inspection=bool(_SETTINGS["opening_inspection"]))
    sequences = [("success", success)]
    if success["solved"]:
        plan = truth.recovery_plan(root, success["order"], seed=int(_SETTINGS["seed"]), count=int(_SETTINGS["recoveries"]))
        for number, item in enumerate(plan):
            built = truth.build_recovery(root, _ENV, success["order"], after=item["after"], family=item["family"],
                                         opening_inspection=bool(_SETTINGS["opening_inspection"]))
            if built is not None:
                sequences.append((f"recovery{number}", built))
    episodes, rows = [], []
    for tag, built in sequences:
        if _SETTINGS["export_rows"]:
            made = demonstration_rows(root, built, tag=tag)
            for row in made:
                validate_row(row)
            rows.extend(made)
        record = compact(built)
        record.update(tag=tag, compound_class=class_of(root))
        episodes.append(record)
    return {"episodes": episodes, "rows": rows, "seconds": round(time.perf_counter() - started, 2)}


def summarize(episodes: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    by_class: dict[str, Counter] = defaultdict(Counter)
    totals: Counter = Counter()
    actions: list[int] = []
    for record in episodes:
        kind = str(record["demonstration"])
        score = record["score"]
        cell = by_class[str(record.get("compound_class"))]
        cell[f"{kind}_sequences"] += 1
        totals[f"{kind}_sequences"] += 1
        if record["solved"]:
            cell[f"{kind}_solved"] += 1
            totals[f"{kind}_solved"] += 1
        if kind == "success":
            actions.append(len(record["steps"]))
            totals["strict_solved"] += int(bool(score["strict"]["solved"]))
            totals["solved_and_strict_agree"] += int(bool(score["strict"]["solved"]) == bool(record["solved"]))
            totals["estimator_passes_at_end"] += int(bool(score["estimator_passes"]))
            totals["solved_with_estimator_alarm"] += int(bool(record["solved"]) and not score["estimator_passes"])
            totals["unsolved_with_estimator_passing"] += int(not record["solved"] and bool(score["estimator_passes"]))
            totals["healthy_objects_damaged"] += int(score["physical"]["damaged"] > 0)
            totals["orders_beyond_first"] += int(record.get("orders_tried", 1) > 1 and bool(record["solved"]))
            for key in ("wls_solves", "substation_solves", "parameter_fits", "measurement_repairs"):
                totals[key] += int(score["cost"].get(key) or 0)
            totals["tool_seconds_x1000"] += int(1000 * float(score["cost"].get("tool_seconds") or 0.0))
    return {
        "totals": dict(sorted(totals.items())),
        "mean_actions_success_path": round(sum(actions) / len(actions), 2) if actions else None,
        "by_class": {name: dict(sorted(cell.items())) for name, cell in sorted(by_class.items())},
    }


def _git_head() -> dict[str, Any]:
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO_ROOT, capture_output=True,
                               text=True, check=True).stdout.strip()
        return {"head": head, "uncommitted_paths": len(dirty.splitlines()) if dirty else 0}
    except (OSError, subprocess.CalledProcessError):
        return {"head": None, "uncommitted_paths": None}


def _percentiles(values: Sequence[int]) -> dict[str, int] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {"median": ordered[len(ordered) // 2], "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
            "max": ordered[-1]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-orders", type=int, default=6)
    parser.add_argument("--recoveries", type=int, default=1, help="recovery paths per case that has a solved success path")
    parser.add_argument("--max-actions", type=int, default=DEFAULT_MAX_ACTIONS)
    parser.add_argument("--table-versions", type=int, default=DEFAULT_TABLE_VERSIONS,
                        help="versions whose tables the prompt shows, most recent first (0: every version)")
    parser.add_argument("--per-class", type=int, default=None, help="run the first N cases of every class only")
    parser.add_argument("--no-opening-inspection", action="store_true",
                        help="inspect the breakers only before a breaker correction, not at the start of every case")
    parser.add_argument("--no-rows", action="store_true")
    parser.add_argument("--seed", type=int, default=20261115)
    parser.add_argument("--roots", nargs="*", help="scenario ids to run (default: all)")
    parser.add_argument("--limit", type=int, default=None, help="run the first N cases only")
    args = parser.parse_args(argv)

    roots = load_suite(args.suite)
    if args.roots:
        wanted = set(args.roots)
        roots = [root for root in roots if str((root.get("execution") or {}).get("scenario_id")) in wanted]
    if args.per_class is not None:
        seen: Counter = Counter()
        kept = []
        for root in roots:
            seen[class_of(root)] += 1
            if seen[class_of(root)] <= int(args.per_class):
                kept.append(root)
        roots = kept
    if args.limit is not None:
        roots = roots[: int(args.limit)]
    if not roots:
        raise SystemExit("no case to run")
    settings = {
        "provider_options": dict(DEFAULT_PROVIDER_OPTIONS), "max_actions": int(args.max_actions),
        "table_versions": (int(args.table_versions) or None), "max_orders": int(args.max_orders), "recoveries": int(args.recoveries),
        "opening_inspection": not args.no_opening_inspection, "export_rows": not args.no_rows, "seed": int(args.seed),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    episodes: list[dict[str, Any]] = []
    row_count = 0
    prompt_chars: list[int] = []
    row_kinds: Counter = Counter()
    episodes_path, rows_path = args.output_dir / "episodes.jsonl", args.output_dir / "rows.jsonl"
    with episodes_path.open("w", encoding="utf-8", newline="\n") as episode_file, \
            (rows_path.open("w", encoding="utf-8", newline="\n") if not args.no_rows else open(  # noqa: SIM115
                __import__("os").devnull, "w", encoding="utf-8")) as row_file, \
            ProcessPoolExecutor(max_workers=max(1, int(args.workers)), initializer=_init_worker, initargs=(settings,)) as pool:
        for done, result in enumerate(pool.map(_run_root, roots, chunksize=1), start=1):
            for record in result["episodes"]:
                episode_file.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
                episodes.append({key: record[key] for key in ("demonstration", "solved", "score", "compound_class", "steps")
                                 } | {"orders_tried": record.get("orders_tried", 1)})
                episodes[-1]["steps"] = [None] * len(record["steps"])
            for row in result["rows"]:
                row_file.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                row_kinds[f"{row['minimal']['demonstration']}:{row['minimal']['kind']}"] += 1
            prompt_chars.extend(_prompt_chars(result["rows"]))
            row_count += len(result["rows"])
            if done % 25 == 0 or done == len(roots):
                print(f"[demonstrations] {done}/{len(roots)} cases, {row_count} rows, {time.time() - started:.0f} s", flush=True)
    summary = summarize(episodes)
    manifest = {
        "contract": BUILD_CONTRACT, "suite": str(args.suite.resolve()), "cases": len(roots), "settings": settings,
        "rows": row_count, "rows_by_kind": dict(sorted(row_kinds.items())),
        "prompt_characters": _percentiles(prompt_chars), "summary": summary, "code": _git_head(),
        "seconds": round(time.time() - started, 1),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
                                                   newline="\n")
    print(json.dumps({"cases": len(roots), "rows": row_count, **summary["totals"],
                      "mean_actions_success_path": summary["mean_actions_success_path"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
