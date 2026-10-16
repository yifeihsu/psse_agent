from __future__ import annotations

import json
import random
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from psse_env.rl.sampling import RootSampling, root_stratum, stratified_roots, stratum_sizes
from psse_env.rl.train import RLConfig, iteration_roots, select_iteration_roots


def _root(index: int, family: str, *, ranking: str | None = None, mimic: str | None = None) -> dict:
    audit: dict = {}
    if ranking is not None:
        audit["parameter_ranking"] = {"stratum": ranking}
    if mimic is not None:
        audit["mimic"] = {"variant": mimic}
    return {"execution": {"scenario_id": f"r{index:03d}"}, "grouping": {"scenario_family": family}, "audit": audit}


def _suite() -> list[dict]:
    rows = []
    rows += [_root(i, "harmonic") for i in range(0, 6)]
    rows += [_root(i, "measurement") for i in range(6, 16)]
    rows += [_root(i, "parameter", ranking="misranked") for i in range(16, 18)]
    rows += [_root(i, "parameter", ranking="ambiguous") for i in range(18, 26)]
    rows += [_root(i, "multi_measurement", mimic="mimic_flow_pair_same_sign") for i in range(26, 30)]
    random.Random(7).shuffle(rows)  # the suite's row order must not matter
    return rows


COUNTS = {"harmonic": 3, "measurement": 2, "parameter:misranked": 2, "parameter:ambiguous": 1, "mimic:flow_pair_same_sign": 2}


def _ids(rows) -> list[str]:
    return [row["execution"]["scenario_id"] for row in rows]


def _reference_permutation(scenarios, *, iteration, roots_per_iteration, seed):
    """The permutation sampler as it stood before stratified sampling (2026-10-15)."""

    if roots_per_iteration is None or roots_per_iteration >= len(scenarios):
        return [dict(row) for row in scenarios]
    order = list(range(len(scenarios)))
    random.Random(seed).shuffle(order)
    start = (iteration * roots_per_iteration) % len(order)
    chosen = [order[(start + offset) % len(order)] for offset in range(roots_per_iteration)]
    return [dict(scenarios[index]) for index in chosen]


def _config(directory: str, **overrides) -> RLConfig:
    payload = {
        "output_dir": directory, "training_suite": str(Path(directory) / "train.json"), "development_suite": None,
        "initial_adapter": str(Path(directory) / "adapter0"), "base_model": "google/gemma-4-12B-it",
        "base_revision": "0" * 40, "samples_per_root": 2, "seed": 11,
    }
    payload.update(overrides)
    return RLConfig(**payload)


class RootStratumTests(unittest.TestCase):
    def test_labels_follow_the_hard_draw_report(self) -> None:
        self.assertEqual(root_stratum(_root(0, "harmonic")), "harmonic")
        self.assertEqual(root_stratum(_root(0, "parameter", ranking="misranked")), "parameter:misranked")
        self.assertEqual(root_stratum(_root(0, "measurement+parameter")), "measurement+parameter:unknown")
        self.assertEqual(root_stratum(_root(0, "multi_measurement", mimic="mimic_flow_pair_opposite_sign")),
                         "mimic:flow_pair_opposite_sign")
        self.assertEqual(root_stratum(_root(0, "parameter", ranking="dominant"), "family"), "parameter")
        with self.assertRaises(ValueError):
            root_stratum(_root(0, "harmonic"), "bogus")

    def test_stratum_sizes(self) -> None:
        self.assertEqual(stratum_sizes(_suite()), {
            "harmonic": 6, "measurement": 10, "mimic:flow_pair_same_sign": 4,
            "parameter:ambiguous": 8, "parameter:misranked": 2,
        })


class StratifiedRootsTests(unittest.TestCase):
    def test_every_iteration_has_the_configured_mix_and_no_duplicates(self) -> None:
        suite = _suite()
        for iteration in range(5):
            rows = stratified_roots(suite, iteration=iteration, counts=COUNTS, seed=3)
            self.assertEqual(len(rows), sum(COUNTS.values()))
            self.assertEqual(len(set(_ids(rows))), len(rows))
            self.assertEqual(Counter(root_stratum(row) for row in rows), Counter(COUNTS))

    def test_a_stratum_is_used_up_before_any_root_repeats(self) -> None:
        suite = _suite()
        harmonic = []
        for iteration in range(2):  # 6 harmonic roots, 3 per iteration
            harmonic += [i for i in _ids(stratified_roots(suite, iteration=iteration, counts=COUNTS, seed=3))
                         if root_stratum(next(r for r in suite if r["execution"]["scenario_id"] == i)) == "harmonic"]
        self.assertEqual(sorted(harmonic), [f"r{i:03d}" for i in range(6)])
        misranked = {
            iteration: sorted(i for i in _ids(stratified_roots(suite, iteration=iteration, counts=COUNTS, seed=3))
                              if i in {"r016", "r017"})
            for iteration in range(3)
        }
        self.assertEqual(misranked, {0: ["r016", "r017"], 1: ["r016", "r017"], 2: ["r016", "r017"]})

    def test_deterministic_and_independent_of_row_order(self) -> None:
        suite = _suite()
        first = _ids(stratified_roots(suite, iteration=1, counts=COUNTS, seed=3))
        self.assertEqual(first, _ids(stratified_roots(list(reversed(suite)), iteration=1, counts=COUNTS, seed=3)))
        self.assertNotEqual(sorted(first), sorted(_ids(stratified_roots(suite, iteration=0, counts=COUNTS, seed=3))))

    def test_counts_must_match_the_suite(self) -> None:
        suite = _suite()
        with self.assertRaisesRegex(ValueError, "missing"):
            stratified_roots(suite, iteration=0, counts={**{k: v for k, v in COUNTS.items() if k != "harmonic"}}, seed=3)
        with self.assertRaisesRegex(ValueError, "not in the suite"):
            stratified_roots(suite, iteration=0, counts={**COUNTS, "topology": 1}, seed=3)
        with self.assertRaisesRegex(ValueError, "exceed"):
            stratified_roots(suite, iteration=0, counts={**COUNTS, "parameter:misranked": 3}, seed=3)
        zero = stratified_roots(suite, iteration=0, counts={**COUNTS, "measurement": 0}, seed=3)
        self.assertNotIn("measurement", {root_stratum(row) for row in zero})


class RootSamplingConfigTests(unittest.TestCase):
    def test_validation(self) -> None:
        RootSampling.from_mapping({}).validate(None)
        RootSampling.from_mapping({"mode": "stratified", "counts": COUNTS}).validate(10)
        bad = [
            ({"mode": "bogus"}, 10, "mode"),
            ({"mode": "stratified", "key": "bogus", "counts": COUNTS}, 10, "key"),
            ({"counts": COUNTS}, 10, "stratified sampling only"),
            ({"mode": "stratified"}, 10, "needs root_sampling.counts"),
            ({"mode": "stratified", "counts": COUNTS}, 12, "add up to 10"),
            ({"mode": "stratified", "counts": COUNTS}, None, "needs roots_per_iteration"),
            ({"mode": "stratified", "counts": {**COUNTS, "harmonic": -1, "measurement": 6}}, 10, "nonnegative"),
        ]
        for payload, roots, message in bad:
            with self.assertRaisesRegex(ValueError, message):
                RootSampling.from_mapping(payload).validate(roots)
        with self.assertRaisesRegex(ValueError, "unknown root_sampling fields"):
            RootSampling.from_mapping({"mode": "stratified", "count": COUNTS})

    def test_config_round_trip_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = _config(directory, roots_per_iteration=10, root_sampling={"mode": "stratified", "counts": COUNTS})
            config.validate()
            path = Path(directory) / "rl.json"
            path.write_text(json.dumps(config.as_dict()), encoding="utf-8")
            self.assertEqual(RLConfig.load(path).root_sampling["counts"], COUNTS)
            with self.assertRaisesRegex(ValueError, "add up to"):
                _config(directory, roots_per_iteration=9, root_sampling={"mode": "stratified", "counts": COUNTS}).validate()

    def test_default_selection_is_the_unchanged_permutation(self) -> None:
        """Runs started before stratified sampling must keep drawing the same roots."""

        suite = _suite()
        with tempfile.TemporaryDirectory() as directory:
            for seed in (11, 20261016):
                for per_iteration in (None, 4, 7, 30, 64):
                    config = _config(directory, seed=seed, roots_per_iteration=per_iteration)
                    for iteration in range(6):
                        expected = _ids(_reference_permutation(suite, iteration=iteration, roots_per_iteration=per_iteration, seed=seed))
                        self.assertEqual(_ids(select_iteration_roots(config, suite, iteration)), expected)
                        self.assertEqual(_ids(iteration_roots(suite, iteration=iteration, roots_per_iteration=per_iteration, seed=seed)), expected)

    def test_stratified_selection_through_the_config(self) -> None:
        suite = _suite()
        with tempfile.TemporaryDirectory() as directory:
            config = _config(directory, roots_per_iteration=10, root_sampling={"mode": "stratified", "counts": COUNTS})
            rows = select_iteration_roots(config, suite, 2)
            self.assertEqual(Counter(root_stratum(row) for row in rows), Counter(COUNTS))
            self.assertEqual(_ids(rows), _ids(stratified_roots(suite, iteration=2, counts=COUNTS, seed=11)))


if __name__ == "__main__":
    unittest.main()
