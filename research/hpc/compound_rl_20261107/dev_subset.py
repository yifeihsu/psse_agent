"""A fixed stratified subset of a compound suite for the per-iteration greedy evaluation (RL plan of 2026-11-05, section 10).

    python -m research.hpc.compound_rl_20261107.dev_subset --suite output/compound_errors_20261027/suite.json \\
        --size 300 --seed 20261107 --output output/compound_rl_20261107/ieee14_dev300.json

The subset keeps the suite's class mix (quotas by largest remainder over
``audit.compound.class``), draws within a class by a seed and the class name
over the roots sorted by id (so the suite's row order does not matter), and
writes the roots unchanged, so the driver loads it like any suite.  The full
suite is for the start and the end of a run; this subset is for every
iteration in between.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psse_env.rl.sampling import root_stratum  # noqa: E402
from psse_env.rl.train import load_suite  # noqa: E402
from psse_env.rl.rollout import scenario_root  # noqa: E402


def class_quotas(sizes: Mapping[str, int], size: int) -> dict[str, int]:
    """``size`` split over the classes in proportion to their sizes, largest remainders first, every class at least one."""

    total = sum(sizes.values())
    if size <= 0 or total <= 0:
        raise ValueError("the subset size and the suite must be positive")
    if size > total:
        raise ValueError(f"subset size {size} exceeds the suite's {total} roots")
    if size < len(sizes):
        raise ValueError(f"subset size {size} cannot hold every one of the {len(sizes)} classes")
    exact = {name: size * count / total for name, count in sizes.items()}
    quotas = {name: max(1, int(value)) for name, value in exact.items()}
    remaining = size - sum(quotas.values())
    order = sorted(sizes, key=lambda name: (-(exact[name] - int(exact[name])), name))
    for name in order:
        if remaining <= 0:
            break
        if quotas[name] < sizes[name]:
            quotas[name] += 1
            remaining -= 1
    while remaining < 0:
        name = max((n for n in sizes if quotas[n] > 1), key=lambda n: (quotas[n] - exact[n], n))
        quotas[name] -= 1
        remaining += 1
    return quotas


def stratified_subset(scenarios: Sequence[Mapping[str, Any]], *, size: int, seed: int) -> list[dict[str, Any]]:
    pools: dict[str, list[Mapping[str, Any]]] = {}
    for scenario in scenarios:
        pools.setdefault(root_stratum(scenario, "compound_class"), []).append(scenario)
    quotas = class_quotas({name: len(pool) for name, pool in pools.items()}, size)
    chosen: list[dict[str, Any]] = []
    for name in sorted(pools):
        pool = sorted(pools[name], key=scenario_root)
        picked = random.Random(f"{seed}:{name}").sample(range(len(pool)), quotas[name])
        chosen.extend(dict(pool[index]) for index in sorted(picked))
    return chosen


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", required=True)
    parser.add_argument("--size", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    scenarios = load_suite(args.suite)
    subset = stratified_subset(scenarios, size=args.size, seed=args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # LF whatever the platform: the file goes to the cluster byte for byte (write_text would write CRLF on Windows).
    output.write_text(json.dumps(subset, indent=1) + "\n", encoding="utf-8", newline="\n")
    counts = Counter(root_stratum(scenario, "compound_class") for scenario in subset)
    print(json.dumps({"suite": args.suite, "roots": len(scenarios), "subset": len(subset), "seed": args.seed,
                      "by_class": dict(sorted(counts.items())), "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
