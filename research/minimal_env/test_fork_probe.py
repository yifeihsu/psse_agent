"""The fork probe on freshly generated cases, with the stand-in model of ``psse_env.minimal.test_forks``."""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from psse_env.minimal import test_forks as fixtures
from psse_env.minimal.env import MinimalDiagnosisEnv
from psse_env.rl.rollout import scenario_root
from research.minimal_env import fork_probe


def setUpModule() -> None:
    if not fixtures.ROOTS:
        fixtures.setUpModule()


def _roots() -> list[dict]:
    return [fixtures.ROOTS[name] for name in fixtures.CLASSES]


def _run(arguments: list[str], holder: dict) -> dict:
    """The probe with the model's side replaced; returns what its last printed line says."""

    def greedy(adapter, **options):
        holder.setdefault("greedy", []).append((adapter, options))
        return fixtures._Dispatch

    def sampling(adapter, **options):
        def factory(seed):
            holder.setdefault("seeds", []).append(seed)
            return fixtures._Dispatch()

        return factory

    printed = io.StringIO()
    with mock.patch.multiple(fork_probe, greedy_policy_factory=greedy, sampling_policy_factory=sampling,
                             prefix_check_factory=lambda adapter, **options: (lambda policy: None),
                             MinimalDiagnosisEnv=lambda **options: MinimalDiagnosisEnv(fixtures.KERNEL, **options)), \
            contextlib.redirect_stdout(printed):
        code = fork_probe.main(arguments)
    assert code == 0
    return json.loads(printed.getvalue().strip().splitlines()[-1])


class SelectionTests(unittest.TestCase):
    def test_the_cases_are_those_an_evaluation_left_unrepaired(self) -> None:
        roots = _roots()
        ids = [scenario_root(root) for root in roots]
        with tempfile.TemporaryDirectory() as folder:
            evaluation = Path(folder) / "evaluation"
            evaluation.mkdir()
            with self.assertRaisesRegex(ValueError, "holds no finished evaluation"):
                fork_probe.select_cases(roots, unrepaired_in=evaluation)
            (evaluation / "evaluation.json").write_text(json.dumps({"episodes": [
                {"scenario_id": ids[0], "solved": True}, {"scenario_id": ids[1], "solved": False}]}), encoding="utf-8")
            self.assertEqual([scenario_root(root) for root in fork_probe.select_cases(roots, unrepaired_in=evaluation)], [ids[1]])
            # With a reference, only what a truth-built sequence repairs.
            reference = {ids[1]: {"expert": False, "truth": False}}
            self.assertEqual(fork_probe.select_cases(roots, unrepaired_in=evaluation, reference=reference), [])
            reference[ids[1]]["truth"] = True
            self.assertEqual(len(fork_probe.select_cases(roots, unrepaired_in=evaluation, reference=reference)), 1)
            named = Path(folder) / "cases.json"
            named.write_text(json.dumps([ids[0]]), encoding="utf-8")
            self.assertEqual([scenario_root(root) for root in fork_probe.select_cases(roots, cases=named)], [ids[0]])
        # A sample is seeded and keeps the suite's order.
        many = [{"execution": {"scenario_id": f"case{index:02d}"}} for index in range(20)]
        first = [scenario_root(root) for root in fork_probe.select_cases(many, sample=5, seed=3)]
        self.assertEqual((len(first), first), (5, sorted(first)))
        self.assertEqual(first, [scenario_root(root) for root in fork_probe.select_cases(many, sample=5, seed=3)])
        self.assertNotEqual(first, [scenario_root(root) for root in fork_probe.select_cases(many, sample=5, seed=4)])
        self.assertEqual(len(fork_probe.select_cases(many, sample=50)), 20)


class ProbeTests(unittest.TestCase):
    def test_the_probe_counts_the_cases_one_forced_step_repairs(self) -> None:
        roots = _roots()
        ids = [scenario_root(root) for root in roots]
        holder: dict = {}
        with tempfile.TemporaryDirectory() as folder:
            suite = Path(folder) / "suite.json"
            suite.write_text(json.dumps(roots), encoding="utf-8")
            out = Path(folder) / "probe"
            base = ["--suite", str(suite), "--adapter", str(Path(folder) / "adapter"), "--output-dir", str(out)]
            part = _run(base + ["--shard", "0/2"], holder)
            self.assertEqual((part["shard"], part["forked"]), ([0, 2], 1))
            self.assertFalse((out / "summary.json").exists())  # a shard writes no summary
            closing = _run(base, holder)
            # The stand-in claims every case without acting.  The meter case is one forced correction from repaired.
            # The two-breaker case is two: after one forced step the stand-in claims again, so one step does not
            # repair it, and the probe says so.
            self.assertEqual((closing["cases"], closing["own_episode_unrepaired"]), (2, 2))
            self.assertEqual(closing["unrepaired_cases_one_forced_step_repairs"], 1)
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual((summary["contract"], summary["selection"]["selected"], summary["settings"]["argument_samples"]),
                             (fork_probe.PROBE_CONTRACT, 2, 0))
            lines = {line["scenario_id"]: line for line in summary["cases"]}
            self.assertEqual(sorted(lines), sorted(ids))
            meter = lines[ids[0]]
            self.assertEqual((meter["own_solved"], meter["own_actions"], meter["repaired_by_one_forced_step"]), (False, ["finish"], True))
            self.assertEqual((meter["first_repairing_branch"]["tool"], meter["first_repairing_branch"]["at_the_last_state"],
                              meter["first_repairing_branch"]["continuation"]), ("correct_measurements", True, ["finish"]))
            self.assertEqual(meter["better"][0]["tool"], "correct_measurements")
            breakers = lines[ids[1]]
            self.assertEqual((breakers["own_solved"], breakers["repaired_by_one_forced_step"]), (False, False))
            self.assertNotIn("first_repairing_branch", breakers)
            # A second closing call forks nothing and writes the same summary.
            holder.clear()
            again = _run(base, holder)
            self.assertEqual(again["cases"], 2)
            self.assertEqual(json.loads((out / "summary.json").read_text(encoding="utf-8"))["cases"], summary["cases"])

    def test_sampled_arguments_are_asked_of_a_seeded_policy(self) -> None:
        roots = _roots()
        holder: dict = {}
        with tempfile.TemporaryDirectory() as folder:
            suite = Path(folder) / "suite.json"
            suite.write_text(json.dumps(roots[:1]), encoding="utf-8")
            out = Path(folder) / "probe"
            closing = _run(["--suite", str(suite), "--adapter", "adapter", "--output-dir", str(out), "--argument-samples", "1",
                            "--seed", "7"], holder)
            self.assertEqual(len(holder["seeds"]), 1)
            summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual((summary["settings"]["argument_samples"], summary["settings"]["seed"]), (1, 7))
            self.assertEqual(closing["branches"], 5 + 6)  # the five other tools, then every tool once with sampled arguments
            self.assertEqual(summary["summary"]["totals"]["branches_duplicate"], 6)  # the sampler here writes what was tried

    def test_a_reference_without_an_evaluation_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            suite = Path(folder) / "suite.json"
            suite.write_text(json.dumps(_roots()[:1]), encoding="utf-8")
            reference = Path(folder) / "reference.json"
            reference.write_text("{}", encoding="utf-8")
            with self.assertRaises(SystemExit):
                fork_probe.main(["--suite", str(suite), "--adapter", "adapter", "--output-dir", str(Path(folder) / "probe"),
                                 "--reference", str(reference)])


if __name__ == "__main__":
    unittest.main()
