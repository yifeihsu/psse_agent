"""The training roots kept apart for the fork probe (workplan of 2026-11-10, stage 2).

    python -m research.hpc.rl_improvement_20261110.reserve_roots --suite output/compound_rl_20261106/train_ieee14/suite.json

Two files, both drawn by the series' own root schedule (the stratified draw
of the arms' base config: same seed, same class counts), so they have the
training mix:

* ``reserved_roots.json``: the draw of a far iteration (``--iteration``,
  default 30).  These are the probe's diagnostic-validation roots and its
  decision panel.  No arm trains on them: a config that names the file
  refuses any iteration whose draw holds one (``reserved_roots`` of the
  driver), so the reservation does not rest on an arm stopping early.
* ``probe_training_roots.json``: the draws of the iterations just before it
  (``--pool-iterations``, default 28 and 29).  These are the probe's
  training-split roots: training data like any other, only not yet used by
  an arm that runs fewer than 28 iterations.

Each file records the suite's digest, the seed and the counts; the reserved
file also records that the draws of every earlier iteration are disjoint
from it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psse_env.rl.rollout import scenario_root  # noqa: E402
from psse_env.rl.sampling import root_stratum, stratified_roots  # noqa: E402
from psse_env.rl.train import load_suite  # noqa: E402

CELL = Path(__file__).resolve().parent


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _draw(scenarios, iteration: int, *, seed: int, counts: dict[str, int], key: str) -> list[dict]:
    return stratified_roots(scenarios, iteration=iteration, counts=counts, key=key, seed=seed)


def _by_class(drawn, key: str) -> dict[str, int]:
    by_class: dict[str, int] = {}
    for scenario in drawn:
        name = root_stratum(scenario, key)
        by_class[name] = by_class.get(name, 0) + 1
    return dict(sorted(by_class.items()))


def reserve(suite_path: Path, *, iteration: int, seed: int, counts: dict[str, int], key: str) -> dict:
    scenarios = load_suite(suite_path)
    reserved = _draw(scenarios, iteration, seed=seed, counts=counts, key=key)
    roots = sorted(scenario_root(scenario) for scenario in reserved)
    overlap = {}
    for earlier in range(iteration):
        drawn = {scenario_root(scenario) for scenario in _draw(scenarios, earlier, seed=seed, counts=counts, key=key)}
        shared = sorted(drawn & set(roots))
        if shared:
            overlap[str(earlier)] = shared
    first_shared = min((int(name) for name in overlap), default=None)
    return {
        "purpose": "training roots no arm of the series may draw: the fork probe's diagnostic-validation split and its decision panel",
        "suite": suite_path.name, "suite_sha256": file_sha256(suite_path), "suite_roots": len(scenarios),
        "seed": int(seed), "key": key, "reserved_iteration": int(iteration), "counts": dict(sorted(counts.items())),
        "by_class": _by_class(reserved, key),
        # The iterations before the reserved one whose draw is free of reserved roots: every arm's planned iterations must be among them.
        "first_iteration_sharing_a_root": first_shared,
        "iterations_free_of_reserved_roots": first_shared if first_shared is not None else int(iteration),
        "roots": roots,
    }


def probe_pool(suite_path: Path, *, iterations: Sequence[int], reserved: Sequence[str], seed: int, counts: dict[str, int], key: str) -> dict:
    """The probe's training-split roots: the draws of ``iterations``, checked against the reserved roots."""

    scenarios = load_suite(suite_path)
    drawn = [scenario for iteration in iterations for scenario in _draw(scenarios, iteration, seed=seed, counts=counts, key=key)]
    roots = sorted(scenario_root(scenario) for scenario in drawn)
    if len(set(roots)) != len(roots):
        raise SystemExit(f"the draws of iterations {list(iterations)} share roots")
    shared = sorted(set(roots) & set(reserved))
    if shared:
        raise SystemExit(f"{len(shared)} roots of the probe's training pool are reserved (first: {shared[0]})")
    return {
        "purpose": "the fork probe's training-split roots: training roots the arms' first iterations do not draw",
        "suite": suite_path.name, "suite_sha256": file_sha256(suite_path),
        "seed": int(seed), "key": key, "iterations": [int(iteration) for iteration in iterations],
        "first_arm_iteration_that_draws_one": min(int(iteration) for iteration in iterations),
        "by_class": _by_class(drawn, key),
        "roots": roots,
    }


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", required=True, type=Path)
    parser.add_argument("--series", default=CELL / "series.json", type=Path)
    parser.add_argument("--iteration", type=int, default=30)
    parser.add_argument("--pool-iterations", type=int, nargs="+", default=[28, 29])
    parser.add_argument("--output", default=CELL / "reserved_roots.json", type=Path)
    parser.add_argument("--pool-output", default=CELL / "probe_training_roots.json", type=Path)
    args = parser.parse_args(argv)
    base = json.loads(args.series.read_text(encoding="utf-8"))["base"]
    sampling = base["root_sampling"]
    common = dict(seed=int(base["seed"]), counts=dict(sampling["counts"]), key=sampling["key"])
    reserved = reserve(args.suite, iteration=args.iteration, **common)
    if any(iteration >= args.iteration for iteration in args.pool_iterations):
        raise SystemExit("the probe's training pool is drawn from iterations before the reserved one")
    pool = probe_pool(args.suite, iterations=args.pool_iterations, reserved=reserved["roots"], **common)
    _write(args.output, reserved)
    _write(args.pool_output, pool)
    print(json.dumps({"reserved": {key: value for key, value in reserved.items() if key != "roots"},
                      "probe_training_pool": {key: value for key, value in pool.items() if key != "roots"}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
