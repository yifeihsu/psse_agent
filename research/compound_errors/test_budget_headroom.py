"""The budget-headroom tabulation and the budget audit's root selection (2026-10-29)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from research.compound_errors.audit import select_roots
from research.compound_errors.budget_headroom import post_commit_cue, summarize


def _episode(sid: str, cls: str, success: bool, *, steps: int = 12, tools: list[str] | None = None) -> dict:
    return {"scenario_id": sid, "class": cls, "success": success, "terminal": True, "steps": steps, "tools": tools or [],
            "escalation_request": None, "trace": [], "corrections": []}


def _audited(sid: str, *, alternative: bool, truth: bool) -> dict:
    return {"root": sid, "decision_states": [{
        "alternatives": [{"action": {"tool": "correct_measurements"}, "outcome": {"success": alternative}}],
        "truth": {"outcome": {"success": truth, "steps": 12}},
    }]}


class BudgetHeadroomTests(unittest.TestCase):
    def test_lost_roots_are_binned_by_what_recovers_them(self) -> None:
        reference = [_episode(f"r{i}", "control_meter", True, steps=20, tools=["rollback_state"] * 2) for i in range(3)]
        reference.append(_episode("r3", "control_meter", False))
        tight = [_episode("r0", "control_meter", False), _episode("r1", "control_meter", False),
                 _episode("r2", "control_meter", False), _episode("r3", "control_meter", True)]
        audited = [_audited("r0", alternative=True, truth=True), _audited("r1", alternative=False, truth=True),
                   _audited("r2", alternative=False, truth=False)]
        with tempfile.TemporaryDirectory() as tmp:
            paths = {}
            for name, rows in {"ref": reference, "tight": tight, "audit": audited}.items():
                paths[name] = Path(tmp) / f"{name}.jsonl"
                paths[name].write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            summary = summarize({"s": {40: paths["ref"], 16: paths["tight"]}}, {"s": {16: paths["audit"]}})
        cell = summary["s"]["classes"]["control_meter"]["16"]
        self.assertEqual((cell["solved"], cell["lost_against_reference"], cell["gained_against_reference"]), (1, 3, 1))
        self.assertEqual(cell["lost_roots_audited"], {"legal_alternative": 1, "truth_probe_only": 1, "neither": 1})
        self.assertEqual(cell["solving_alternative_tools"], {"correct_measurements": 1})
        self.assertEqual(cell["lost_paths"]["mean_reference_actions"], 20)
        self.assertEqual(cell["lost_paths"]["mean_reference_rollbacks"], 2)
        self.assertEqual(cell["lost_paths"]["truth_probe_solves"], 2)
        self.assertNotIn("lost_roots_audited", summary["s"]["classes"]["control_meter"]["40"])

    def test_the_cue_is_the_verifying_wls_before_the_first_commit(self) -> None:
        episode = {"trace": [
            {"tool": "run_wls", "metrics": {"chi_square_ratio": 1.8, "max_normalized_residual": 25.0}},
            {"tool": "correct_measurements"},
            {"tool": "run_wls", "metrics": {"chi_square_ratio": 1.2, "max_normalized_residual": 15.7, "globally_resolved": False}},
            {"tool": "commit_state"},
            {"tool": "get_measurement_context"},
        ], "corrections": [
            {"tool": "correct_measurements", "step": 1, "target_is_true": True, "lifecycle": "commit_state"},
            {"tool": "correct_parameters", "step": 6, "target_is_true": False, "lifecycle": "rollback_state"},
            {"tool": "correct_measurements", "step": 9, "target_is_true": True, "lifecycle": "commit_state"},
        ]}
        self.assertEqual(post_commit_cue(episode), {"committed": "correct_measurements", "next_true": "correct_measurements",
                                                    "max_normalized_residual": 15.7})
        episode["trace"][2]["metrics"]["chi_square_ratio"] = 0.9
        self.assertIsNone(post_commit_cue(episode))

    def test_a_reference_run_restricts_the_audit_to_the_roots_the_budget_lost(self) -> None:
        suite = [{"execution": {"scenario_id": f"r{i}"}, "audit": {"compound": {"class": "control_meter"}}} for i in range(4)]
        tight = {"r0": {"success": False}, "r1": {"success": False}, "r2": {"success": True}, "r3": {"success": False}}
        reference = {"r0": {"success": True}, "r1": {"success": False}, "r2": {"success": True}, "r3": {"success": True}}
        roots, report = select_roots(suite, tight, solved_per_class=0, failed_per_class=None, seed=1, classes=None, reference=reference)
        self.assertEqual(sorted(r["execution"]["scenario_id"] for r in roots), ["r0", "r3"])
        self.assertEqual(report["control_meter"]["failed"], 2)
        roots, _ = select_roots(suite, tight, solved_per_class=0, failed_per_class=None, seed=1, classes=None)
        self.assertEqual(len(roots), 3)


if __name__ == "__main__":
    unittest.main()
