"""Which training roots an iteration samples.

``permutation`` (the default) walks one seeded permutation of the training
suite in disjoint slices of ``roots_per_iteration``, so every root is used
once before any is reused (:func:`psse_env.rl.train.iteration_roots`).

``stratified`` fixes how many roots each stratum contributes to every
iteration.  Within a stratum the roots are walked the same way, one seeded
permutation in slices, so all of a stratum's roots are used before any
repeats, and a stratum whose count equals its size contributes all of its
roots to every iteration.  The counts must name every stratum of the suite
and add up to ``roots_per_iteration``, so a typo or a changed suite fails
loudly instead of silently shifting the mix.

Strata are the scenario family (``family``) or the hard draw's stratum
(``hard_draw_stratum``): the family, with the parameter families split by
the ranking stratum the suite builder recorded (dominant, ambiguous,
misranked) and the flow-meter mimics by variant, the same labels as
research/rl/hard_draw_report.py.  These labels come from the audit, so they
shape the training mix only; the policy never sees them.

Why (2026-10-16): the screen arm's second update broke its harmonic roots
(12 to 5 of 16 on the hard development draw) after 18 of its 20 informative
groups came from measurement-type roots.  Harmonic roots already made up 7
of every 64 roots, so a proportional mix would not change their weight;
the counts are what moves it (docs/rl_experiments_20261014.md, section 5).
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from psse_env.rl.rollout import scenario_family, scenario_root

SAMPLING_MODES = ("permutation", "stratified")
STRATUM_KEYS = ("family", "hard_draw_stratum")
PARAMETER_FAMILIES = frozenset({"parameter", "measurement+parameter"})


def root_stratum(scenario: Mapping[str, Any], key: str = "hard_draw_stratum") -> str:
    """The stratum label of one root under ``key``."""

    family = scenario_family(scenario)
    if key == "family":
        return family
    if key != "hard_draw_stratum":
        raise ValueError(f"the stratum key must be one of {STRATUM_KEYS}, got {key!r}")
    audit = scenario.get("audit") if isinstance(scenario.get("audit"), Mapping) else {}
    mimic = audit.get("mimic")
    if isinstance(mimic, Mapping) and mimic.get("variant"):
        return f"mimic:{str(mimic['variant']).removeprefix('mimic_')}"
    if family in PARAMETER_FAMILIES:
        ranking = audit.get("parameter_ranking") if isinstance(audit.get("parameter_ranking"), Mapping) else {}
        return f"{family}:{ranking.get('stratum') or 'unknown'}"
    return family


def stratum_sizes(scenarios: Sequence[Mapping[str, Any]], key: str = "hard_draw_stratum") -> dict[str, int]:
    return dict(sorted(Counter(root_stratum(scenario, key) for scenario in scenarios).items()))


@dataclass(frozen=True)
class RootSampling:
    """The ``root_sampling`` block of an RL config; empty means ``permutation``."""

    mode: str = "permutation"
    key: str = "hard_draw_stratum"
    counts: Mapping[str, int] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "RootSampling":
        payload = dict(payload or {})
        unknown = sorted(set(payload) - {"mode", "key", "counts"})
        if unknown:
            raise ValueError(f"unknown root_sampling fields: {unknown}")
        counts = payload.get("counts") or {}
        if not isinstance(counts, Mapping):
            raise ValueError("root_sampling.counts must map stratum names to root counts")
        return cls(
            mode=str(payload.get("mode") or "permutation"),
            key=str(payload.get("key") or "hard_draw_stratum"),
            counts={str(name): int(value) for name, value in counts.items()},
        )

    def validate(self, roots_per_iteration: int | None) -> None:
        if self.mode not in SAMPLING_MODES:
            raise ValueError(f"root_sampling.mode must be one of {SAMPLING_MODES}, got {self.mode!r}")
        if self.key not in STRATUM_KEYS:
            raise ValueError(f"root_sampling.key must be one of {STRATUM_KEYS}, got {self.key!r}")
        if self.mode == "permutation":
            if self.counts:
                raise ValueError("root_sampling.counts apply to stratified sampling only")
            return
        if not self.counts:
            raise ValueError("stratified sampling needs root_sampling.counts")
        negative = sorted(name for name, value in self.counts.items() if value < 0)
        if negative:
            raise ValueError(f"root_sampling.counts must be nonnegative: {negative}")
        if roots_per_iteration is None:
            raise ValueError("stratified sampling needs roots_per_iteration")
        total = sum(self.counts.values())
        if total != int(roots_per_iteration):
            raise ValueError(f"root_sampling.counts add up to {total}, not roots_per_iteration {roots_per_iteration}")


def stratified_roots(
    scenarios: Sequence[Mapping[str, Any]],
    *,
    iteration: int,
    counts: Mapping[str, int],
    key: str = "hard_draw_stratum",
    seed: int,
) -> list[dict[str, Any]]:
    """The roots of one iteration with ``counts[stratum]`` roots from every stratum.

    Each stratum's roots are sorted by root id (so the suite's row order does
    not matter) and walked through one permutation seeded by ``seed`` and the
    stratum name; iteration ``i`` takes the ``i``-th slice, wrapping around.
    The chosen roots are then shuffled (seeded by ``seed`` and ``i``) so the
    rollout shards each get a mix of strata.
    """

    pools: dict[str, list[Mapping[str, Any]]] = {}
    for scenario in scenarios:
        pools.setdefault(root_stratum(scenario, key), []).append(scenario)
    missing = sorted(set(pools) - set(counts))
    unknown = sorted(set(counts) - set(pools))
    if missing or unknown:
        raise ValueError(
            "root_sampling.counts must name exactly the suite's strata: "
            f"missing {missing}, not in the suite {unknown} (suite strata {stratum_sizes(scenarios, key)})"
        )
    too_many = {name: (int(counts[name]), len(pools[name])) for name in pools if int(counts[name]) > len(pools[name])}
    if too_many:
        raise ValueError(f"root_sampling.counts exceed the stratum sizes (count, size): {too_many}")
    chosen: list[Mapping[str, Any]] = []
    for name in sorted(pools):
        count = int(counts[name])
        if count == 0:
            continue
        pool = sorted(pools[name], key=scenario_root)
        order = list(range(len(pool)))
        random.Random(f"{seed}:{name}").shuffle(order)
        start = (int(iteration) * count) % len(order)
        chosen.extend(pool[order[(start + offset) % len(order)]] for offset in range(count))
    roots = [scenario_root(scenario) for scenario in chosen]
    if len(set(roots)) != len(roots):
        raise ValueError("the training suite holds duplicate root ids")
    random.Random(f"{seed}:{int(iteration)}:order").shuffle(chosen)
    return [dict(scenario) for scenario in chosen]


__all__ = [
    "PARAMETER_FAMILIES",
    "RootSampling",
    "SAMPLING_MODES",
    "STRATUM_KEYS",
    "root_stratum",
    "stratified_roots",
    "stratum_sizes",
]
