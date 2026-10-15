from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]


def _build_suite_module():
    path = REPO_ROOT / "research" / "hpc" / "full_pipeline_20260907" / "build_suite.py"
    spec = importlib.util.spec_from_file_location("build_suite_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _row(root: str, family: str, stratum: str | None) -> dict:
    audit = {"parameter_ranking": {"stratum": stratum}} if stratum else {}
    return {"grouping": {"scenario_family": family, "root_scenario_id": root}, "audit": audit, "execution": {"scenario_id": root}}


class _FakeResearch:
    """The two accessors and the family allocator the suite builder calls."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    @staticmethod
    def _scenario_family(row):
        return row["grouping"]["scenario_family"]

    @staticmethod
    def _row_root(row):
        return row["grouping"]["root_scenario_id"]

    def allocate_scenarios(self, candidates, *, d0_roots, train_plan, development_plan, seed, protected_roots):
        self.calls.append({"d0_roots": set(d0_roots), "train_plan": dict(train_plan), "development_plan": dict(development_plan)})
        plan = train_plan or development_plan
        chosen = []
        for family, count in plan.items():
            rows = [r for r in candidates if self._scenario_family(r) == family and self._row_root(r) not in d0_roots | protected_roots]
            chosen.extend(rows[:count])
        return (chosen, []) if train_plan else ([], chosen)


class HardDrawAllocationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _build_suite_module()

    def test_strata_follow_the_stratum_threshold_not_the_admission_threshold(self) -> None:
        # A hard draw admits training roots at 1.0 but must still call a
        # true-first root with ratio 1.06 ambiguous, as the deployed context does at 1.2.
        row = {"audit": {"parameter_ranking": {"parameter_ranking_dominance_ratio": 1.06, "true_line_rank": 1}}}
        stratum = self.module.parameter_ranking_stratum
        self.assertEqual(stratum(row, family="parameter", dominance_threshold=1.2)["stratum"], "ambiguous")
        self.assertEqual(stratum(row, family="parameter", dominance_threshold=1.0)["stratum"], "dominant")
        misranked = {"audit": {"parameter_ranking": {"parameter_ranking_dominance_ratio": 1.4, "true_line_rank": 2}}}
        self.assertEqual(stratum(misranked, family="measurement+parameter", dominance_threshold=1.2)["stratum"], "misranked")
        self.assertEqual(stratum(row, family="measurement", dominance_threshold=1.2)["stratum"], "not_applicable")

    def test_stratum_plan_parsing(self) -> None:
        self.assertEqual(self.module.parse_stratum_plan(None), {})
        plan = self.module.parse_stratum_plan('{"parameter": {"misranked": 2, "ambiguous": 1, "dominant": 0}}')
        self.assertEqual(plan, {"parameter": {"misranked": 2, "ambiguous": 1}})
        with self.assertRaises(ValueError):
            self.module.parse_stratum_plan('["parameter"]')

    def test_families_in_the_stratum_plan_are_drawn_by_stratum_and_the_rest_by_family(self) -> None:
        research = _FakeResearch()
        candidates = [_row(f"p_mis_{i}", "parameter", "misranked") for i in range(3)]
        candidates += [_row(f"p_amb_{i}", "parameter", "ambiguous") for i in range(2)]
        candidates += [_row(f"p_dom_{i}", "parameter", "dominant") for i in range(4)]
        candidates += [_row(f"m_{i}", "measurement", None) for i in range(3)]
        chosen, report = self.module.allocate_with_strata(
            research, candidates, role="development",
            family_plan={"parameter": 4, "measurement": 2},
            stratum_plan={"parameter": {"misranked": 2, "ambiguous": 3}},
            seed=5, d0_roots={"p_mis_0"}, protected_roots=set(),
        )
        roots = [research._row_root(r) for r in chosen]
        self.assertEqual(sum(1 for r in roots if r.startswith("p_mis_")), 2)
        self.assertNotIn("p_mis_0", roots)
        self.assertEqual(sum(1 for r in roots if r.startswith("p_amb_")), 2)
        self.assertEqual(sum(1 for r in roots if r.startswith("p_dom_")), 0)
        self.assertEqual(sum(1 for r in roots if r.startswith("m_")), 2)
        self.assertEqual(report["stratum_counts"], {"parameter:ambiguous": 2, "parameter:misranked": 2})
        self.assertEqual(report["stratum_shortfalls"], {"parameter:ambiguous": 1})
        self.assertEqual(research.calls[0]["development_plan"], {"measurement": 2})
        self.assertTrue({"p_mis_0"} <= research.calls[0]["d0_roots"])

    def test_the_draw_is_deterministic_in_the_seed(self) -> None:
        research = _FakeResearch()
        candidates = [_row(f"p_mis_{i}", "parameter", "misranked") for i in range(6)]
        first, _ = self.module.allocate_with_strata(research, candidates, role="training", family_plan={},
                                                    stratum_plan={"parameter": {"misranked": 3}}, seed=1,
                                                    d0_roots=set(), protected_roots=set())
        second, _ = self.module.allocate_with_strata(research, candidates, role="training", family_plan={},
                                                     stratum_plan={"parameter": {"misranked": 3}}, seed=1,
                                                     d0_roots=set(), protected_roots=set())
        other, _ = self.module.allocate_with_strata(research, candidates, role="training", family_plan={},
                                                    stratum_plan={"parameter": {"misranked": 3}}, seed=2,
                                                    d0_roots=set(), protected_roots=set())
        ids = lambda rows: [research._row_root(r) for r in rows]
        self.assertEqual(ids(first), ids(second))
        self.assertEqual(len(ids(other)), 3)
        self.assertEqual(research.calls, [])


if __name__ == "__main__":
    unittest.main()
