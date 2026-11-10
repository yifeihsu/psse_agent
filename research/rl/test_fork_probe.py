"""The fork probe on the toy closed loop of ``psse_env.rl.test_forks``: selection, branches, decisions, report."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from psse_env.actions import COMMIT_STATE, CORRECT_PARAMETERS, ROLLBACK_STATE
from psse_env.dagger.observation_modes import BOUNDED_HISTORY, COMPLETE_HISTORY
from psse_env.rl.forks import DecisionState
from psse_env.rl.reward import RewardWeights
from psse_env.rl.rollout import TrajectoryStore
from psse_env.rl.test_forks import Scripted, ToyEvaluator, render_other_choice, rollouts, scenario
from research.rl.fork_probe import (
    STRATA,
    build_manifest,
    build_report,
    candidate_audit,
    candidate_markdown,
    candidate_records,
    capture_views,
    decide_forks,
    decision_panel,
    edit_audit,
    expert_reference_from_stores,
    fork_stratum,
    load_branches,
    load_expert_reference,
    markdown,
    panel_markdown,
    run_branches,
)

QUOTAS = {"training": {"breaker_only_false_parameter": 1, "wrong_target": 1, "true_after_decline": 1, "mixed_true_parameter": 1,
                       "true_first": 1, "true_later": 0},
          "validation": {"breaker_only_false_parameter": 1, "wrong_target": 1, "true_after_decline": 1, "mixed_true_parameter": 1,
                         "true_first": 1, "true_later": 0}}


def _probe(directory: Path):
    """Two stores, a suite and the expert's results, on the toy loop."""

    evaluator = ToyEvaluator()
    roots = {
        "a1": (scenario("a1", name="cb2_split"), Scripted(lines=(3,))),              # a wrong line kept where only breakers are faulted
        "a2": (scenario("a2", name="rx2_distant"), Scripted(lines=(3,))),            # a wrong line kept
        "a3": (scenario("a3", name="rx_cb_distant"), Scripted(lines=(7,))),          # the true line, a breaker fault beside it
        "a4": (scenario("a4", name="rx2_distant"), Scripted(lines=(7,))),            # the true line, first candidate
        "a5": (scenario("a5", name="rx2_distant"), Scripted(lines=(3, 7), decline=(3,))),  # a decline, then the true line
        "a6": (scenario("a6", name="rx2_distant", relation="physical_v1"), Scripted(lines=(3,))),  # a noise variant of v1
        "v1": (scenario("v1", name="cb2_mixed"), Scripted(lines=(3,))),
        "v2": (scenario("v2", name="rx2_distant"), Scripted(lines=(7,))),
    }
    training, validation = TrajectoryStore(directory / "training"), TrajectoryStore(directory / "validation")
    for name, (root, policy) in roots.items():
        (validation if name.startswith("v") else training).write_group(name, rollouts(evaluator, root, [policy]))
    suite = directory / "suite.json"
    suite.write_text(json.dumps([root for root, _policy in roots.values()]), encoding="utf-8")
    expert = directory / "expert.json"
    expert.write_text(json.dumps({"a1": {"success": False}, "a2": {"success": False}, "a3": {"success": True}, "a4": True,
                                  "a5": True, "a6": False, "v1": {"success": False}, "v2": {"success": True}}), encoding="utf-8")
    manifest = build_manifest(training_stores=[directory / "training"], validation_stores=[directory / "validation"], suite_path=suite,
                              expert_reference=expert, seed=5, quotas=QUOTAS, expert_failed={"training": 2, "validation": 1})
    return evaluator, {name: root for name, (root, _policy) in roots.items()}, manifest


class ProbeInputTests(unittest.TestCase):
    def test_the_experts_record_is_read_from_the_stores_of_its_own_rollouts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            evaluator = ToyEvaluator()
            solved, failed = TrajectoryStore(Path(directory) / "a"), TrajectoryStore(Path(directory) / "b")
            solved.write_group("s1", rollouts(evaluator, scenario("s1", name="rx2_distant"), [Scripted(lines=(7,))]))
            failed.write_group("f1", rollouts(evaluator, scenario("f1", name="rx2_distant"), [Scripted(lines=(3,))]))
            reference = expert_reference_from_stores([Path(directory) / "a", Path(directory) / "b"])
            self.assertEqual({root: entry["success"] for root, entry in reference.items()}, {"s1": True, "f1": False})
            self.assertTrue(all(entry["steps"] > 0 for entry in reference.values()))
            path = Path(directory) / "expert.json"
            path.write_text(json.dumps(reference), encoding="utf-8")
            self.assertEqual(load_expert_reference(path), {"s1": True, "f1": False})
            with self.assertRaisesRegex(ValueError, "more than one"):
                expert_reference_from_stores([Path(directory) / "a", Path(directory) / "a"])

    def test_the_candidate_audit_counts_accepted_candidates_by_class(self) -> None:
        def episode(name: str, *corrections) -> dict:
            return {"scenario_id": f"r_{name}", "class": name, "corrections": [
                {"tool": tool, "verdict": verdict, "lifecycle": lifecycle, "target_is_true": true}
                for tool, verdict, lifecycle, true in corrections]}

        episodes = [
            episode("cb2_split", (CORRECT_PARAMETERS, "ACCEPT_PARTIAL", ROLLBACK_STATE, False),
                    ("correct_topology", "ACCEPT_FINAL", COMMIT_STATE, True)),
            episode("rx_cb_meter", (CORRECT_PARAMETERS, "ACCEPT_PARTIAL", COMMIT_STATE, True),
                    (CORRECT_PARAMETERS, "REJECT", ROLLBACK_STATE, False),          # a rejected candidate is no decision
                    (CORRECT_PARAMETERS, "ACCEPT_FINAL", None, True)),              # an accepted one the episode never decided on
            episode("rx2_distant", (CORRECT_PARAMETERS, "ACCEPT_FINAL", COMMIT_STATE, True),
                    (CORRECT_PARAMETERS, "ACCEPT_PARTIAL", COMMIT_STATE, False)),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "episodes.jsonl"
            path.write_text("\n".join(json.dumps(item) for item in episodes) + "\n", encoding="utf-8")
            records = candidate_records(episodes=[path])
            # The driver's own stores give the same records, with the class from the suite.
            evaluator = ToyEvaluator()
            root = scenario("t1", name="cb2_mixed")
            TrajectoryStore(Path(directory) / "store").write_group("t1", rollouts(evaluator, root, [Scripted(lines=(3,))]))
            stored = candidate_records(stores=[Path(directory) / "store"], suite={"t1": root})
        self.assertEqual(len(records), 5)
        audit = candidate_audit(records)
        self.assertEqual((audit["decisions_on_accepted_candidates"], audit["candidates"]), (5, 4))
        self.assertEqual(audit["breaker_only_classes"], {"candidates": 1, "true": 0, "true_share": 0.0})
        self.assertEqual(audit["other_classes"], {"candidates": 3, "true": 2, "true_share": 2 / 3})
        self.assertEqual(audit["mixed_breaker_and_line_classes"], {"candidates": 1, "true": 1, "true_share": 1.0})
        self.assertEqual(audit["by_class"]["cb2_split"], {"candidates": 1, "true": 0, "declined": 1})
        self.assertIn("| Classes whose faults are breakers alone | 0 of 1 | 0.0% |", candidate_markdown(audit))
        self.assertEqual([(item["class"], item["tool"], item["target_is_true"]) for item in stored], [("cb2_mixed", CORRECT_PARAMETERS, False)])
        self.assertEqual(candidate_audit(stored)["breaker_only_classes"]["candidates"], 1)
        self.assertIsNone(candidate_audit([])["other_classes"]["true_share"])


class ForkProbeTests(unittest.TestCase):
    def test_the_sample_is_one_fork_per_root_by_kind_with_its_shortfalls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            _evaluator, _roots, manifest = _probe(Path(directory))
        forks = {fork["state"]["root"]: fork for fork in manifest["forks"]}
        self.assertEqual({root: fork["stratum"] for root, fork in forks.items()}, {
            "a1": "breaker_only_false_parameter", "a2": "wrong_target", "a3": "mixed_true_parameter", "a4": "true_first",
            "a5": "true_after_decline", "v1": "breaker_only_false_parameter", "v2": "true_first"})
        self.assertEqual({root: fork["split"] for root, fork in forks.items()},
                         {"a1": "training", "a2": "training", "a3": "training", "a4": "training", "a5": "training",
                          "v1": "validation", "v2": "validation"})
        # a6 is a variant of the physical root of v1, which is in the validation split.
        self.assertNotIn("a6", forks)
        self.assertEqual(forks["a5"]["state"]["position"], 5)
        self.assertEqual((forks["a5"]["state"]["earlier_declines"], forks["a5"]["state"]["sampled_tool"]), (1, COMMIT_STATE))
        coverage = manifest["coverage"]
        self.assertEqual((coverage["training"]["selected"], coverage["training"]["shortfalls"]), (5, []))
        self.assertEqual(coverage["training"]["expert_failed"], {"target": 2, "selected": 2, "available": 2})
        self.assertEqual(coverage["validation"]["selected"], 2)
        self.assertEqual(coverage["validation"]["shortfalls"],
                         ["wrong_target: 0 of 1", "true_after_decline: 0 of 1", "mixed_true_parameter: 0 of 1", "forks: 2 of 5"])
        self.assertEqual(manifest["natural_decisions"]["training"],
                         {"breaker_only_false_parameter": 1, "wrong_target": 3, "mixed_true_parameter": 1, "true_first": 1,
                          "true_after_decline": 1})
        self.assertEqual(manifest["rules"]["strata_in_order"], list(STRATA))
        self.assertEqual(manifest["seed"], 5)

    def test_kinds_and_the_edit_audit_read_the_audits_labels(self) -> None:
        def state(tool: str, *, true: bool, ordinal: int = 0, declines: int = 0, transitions=()) -> DecisionState:
            return DecisionState(root="r", sample_index=0, position=2, sampled_tool=COMMIT_STATE,
                                 decision={"correction_tool": tool, "target_is_true": true}, prefix_actions=[{}, {}],
                                 observations=[{}, {}, {}], ordinal=ordinal, earlier_declines=declines, transitions=list(transitions))

        self.assertEqual(fork_stratum(state(CORRECT_PARAMETERS, true=False), "control_cb"), "breaker_only_false_parameter")
        self.assertEqual(fork_stratum(state(CORRECT_PARAMETERS, true=False), "rx_cb_meter"), "wrong_target")
        self.assertEqual(fork_stratum(state("correct_measurements", true=False), "cb2_split"), "wrong_target")
        self.assertEqual(fork_stratum(state(CORRECT_PARAMETERS, true=True), "rx_cb_meter"), "mixed_true_parameter")
        self.assertEqual(fork_stratum(state(CORRECT_PARAMETERS, true=True, declines=1), "rx_cb_meter"), "true_after_decline")
        self.assertEqual(fork_stratum(state("correct_topology", true=True), "rx_cb_meter"), "true_first")
        self.assertEqual(fork_stratum(state(CORRECT_PARAMETERS, true=True, ordinal=1), "rx2_distant"), "true_later")
        # A correction on the true line 9 whose pair refit also rewrote line 4.
        edited = [{"action": {"tool": CORRECT_PARAMETERS, "arguments": {"line_index": 9}},
                   "tool_output": {"execution_status": "success", "tool_metrics": {
                       "pair_refit": {"status": "refit", "partner_line_index": 4},
                       "detailed_evidence": {"edit": [{"line": 9, "r": {}}, {"line": 4, "r": {}}]}}}}]
        audit = edit_audit(state(CORRECT_PARAMETERS, true=True, transitions=edited), scenario("r", (9,)))
        self.assertEqual((audit["lines_changed"], audit["healthy_lines_changed"], audit["pair_refit"],
                          audit["label_true_but_healthy_line_changed"]), ([4, 9], [4], True, True))
        clean = edit_audit(state(CORRECT_PARAMETERS, true=True, transitions=edited), scenario("r", (4, 9)))
        self.assertEqual((clean["healthy_lines_changed"], clean["label_true_but_healthy_line_changed"]), ([], False))
        self.assertEqual(edit_audit(state(CORRECT_PARAMETERS, true=True), scenario("r")), {"known": False})

    def test_the_expert_reference_is_read_from_a_map_or_from_episodes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            episodes = Path(directory) / "episodes.jsonl"
            episodes.write_text("\n".join(json.dumps(row) for row in (
                {"scenario_id": "a", "success": True}, {"scenario_id": "b", "truth_audited_task_success": False})) + "\n", encoding="utf-8")
            self.assertEqual(load_expert_reference(episodes), {"a": True, "b": False})
            one = Path(directory) / "one.jsonl"
            one.write_text(json.dumps({"scenario_id": "a", "success": False}) + "\n", encoding="utf-8")
            self.assertEqual(load_expert_reference(one), {"a": False})
        self.assertEqual(load_expert_reference(None), {})

    def test_branches_decisions_and_the_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            evaluator, roots, manifest = _probe(base)
            common = dict(output_dir=base / "branches", env_factory=lambda **_k: None,
                          continuation_factory=lambda seed: Scripted(lines=(7,)), max_steps=12, case_loader=None,
                          weights=RewardWeights(), continuations=1, evaluator=evaluator, label={"continuation_policy": "toy"})
            first = run_branches(manifest, roots, shard=(0, 2), **common)
            rest = run_branches(manifest, roots, **common)
            self.assertEqual((first["jobs"], first["ran"]), (16, 16))  # four of the seven forks, two modes, two choices
            self.assertEqual((rest["jobs"], rest["ran"], rest["already_there"]), (28, 12, 16))
            self.assertEqual(run_branches(manifest, roots, **common)["ran"], 0)
            branches = load_branches(base / "branches")
            self.assertEqual(len(branches), 28)
            self.assertTrue(all(branch["valid"] for branch in branches))
            self.assertEqual({branch["observation_mode"] for branch in branches}, {BOUNDED_HISTORY, COMPLETE_HISTORY})
            # The policy's own choice at each fork: a scorer that leans to declining when it is shown the whole history.
            seen = []

            def logprob(prompt, completion):
                seen.append((tuple(prompt), tuple(completion)))
                declines = completion[-1] == 10
                return -0.1 if declines == (len(prompt) > 4) else -2.4

            # What the policy is shown at each fork in each mode, by replay; in the mode the episodes were recorded
            # in, the replayed state is the recorded one.
            views = capture_views(manifest, roots, env_factory=lambda **_k: None, max_steps=12, case_loader=None, evaluator=evaluator)
            self.assertEqual((len(views), sum(view["matched"] for view in views)), (14, 14))
            self.assertEqual({view["observation_mode"] for view in views}, {BOUNDED_HISTORY, COMPLETE_HISTORY})
            complete = next(view for view in views if view["observation_mode"] == COMPLETE_HISTORY and view["root"] == "a5")
            self.assertEqual(len(complete["view"]["history_window"]), 5)
            self.assertEqual(complete["view"]["candidate_experiments"][0]["disposition"], "rolled_back")
            decisions = decide_forks(
                manifest, views, render_prompt=lambda view: json.dumps(sorted(view)),
                prompt_token_ids=lambda text: ([1] * (9 if "history_record" in text else 3), 9 if "history_record" in text else 3),
                render_target=render_other_choice, logprob=logprob)
            self.assertEqual(len(decisions), 14)
            # A view whose replay did not match is not scored.
            unmatched = [{**view, "matched": view["root"] != "a1"} for view in views]
            self.assertEqual(len(decide_forks(manifest, unmatched, render_prompt=lambda view: "", prompt_token_ids=lambda text: ([1], 1),
                                              render_target=render_other_choice, logprob=logprob)), 12)
            # The same forks as a panel: an update that raises the decline probability of every candidate alike
            # leaves the separation of wrong from true targets where it was.
            shifted = [{**item, "p_decline": min(0.99, item["p_decline"] + 0.05)} for item in decisions]
            panel = decision_panel(manifest, decisions, shifted)
            bounded = panel["modes"][BOUNDED_HISTORY]
            self.assertEqual((panel["forks_scored_in_both"], bounded["all_wrong_targets"]["forks"], bounded["all_true_targets"]["forks"]), (7, 3, 4))
            self.assertGreater(bounded["all_true_targets"]["mean_p_decline_after"], bounded["all_true_targets"]["mean_p_decline_before"])
            self.assertEqual(bounded["separation_before"], bounded["separation_after"])
            self.assertIn("p(decline) before", panel_markdown(panel))
            self.assertEqual({item["observation_mode"] for item in decisions}, {BOUNDED_HISTORY, COMPLETE_HISTORY})
            self.assertTrue(all(0.0 < item["p_decline"] < 1.0 for item in decisions))
            report = build_report(manifest, branches, decisions)
            text = markdown(report, title="Toy probe")
        self.assertEqual((report["forks"], report["branches"]["total"], report["branches"]["valid"]), (7, 28, 28))
        self.assertEqual(report["continuation_policy"], ["toy"])
        rows = {row["root"]: row for row in report["forks_table"]}
        for mode in (BOUNDED_HISTORY, COMPLETE_HISTORY):
            # Declining a wrong line and correcting the true one solves the case; declining a true line loses it.
            self.assertEqual([rows[root]["modes"][mode]["better"] for root in ("a1", "a2", "v1")], ["decline"] * 3)
            self.assertEqual([rows[root]["modes"][mode]["better"] for root in ("a3", "a4", "a5", "v2")], ["commit"] * 4)
            self.assertEqual(rows["a1"]["modes"][mode][ROLLBACK_STATE]["solved"], 1)
            self.assertEqual(rows["a1"]["modes"][mode]["label_against_value"], False)
            summary = report["summary"][mode]
            self.assertEqual(summary["root_weighted"]["all"]["forks_measured"], 7)
            self.assertAlmostEqual(summary["root_weighted"]["all"]["decline_better"], 3 / 7)
            self.assertEqual(summary["root_weighted"]["breaker_only_false_parameter"]["decline_better"], 1.0)
            self.assertEqual(summary["root_weighted"]["split:validation"]["forks"], 2)
            # Three of the stores' four wrong-target decisions stand behind one fork, so the weighted share moves.
            self.assertNotAlmostEqual(summary["decision_weighted"]["all"]["decline_better"], 3 / 7)
            chain = summary["chain_after_decline_of_a_wrong_target"]
            self.assertEqual((chain["continuations"], chain["next_target_true"], chain["next_correction_committed"], chain["solved"]),
                             (3, 3, 3, 3))
            self.assertEqual(summary["chain_after_commit_of_a_true_target"]["solved"], 4)
            self.assertEqual(summary["label_against_value"]["label_points_the_other_way"], 0)
        gate = report["screening_gate"]
        self.assertEqual((gate["expert_failed_roots_probed"], gate["expert_failed_roots_solved"], gate["in_validation"], gate["passed"]),
                         (3, 3, ["v1"], False))
        history = report["history_effect_on_the_decision"]
        self.assertEqual((history["forks"], history["all_wrong_targets"]["forks"], history["all_true_targets"]["forks"]), (7, 3, 4))
        self.assertEqual(history["wrong_target"]["forks"], 1)  # the kind, which leaves out the two breaker-only forks
        self.assertLess(history["all_wrong_targets"]["mean_p_decline_bounded"], history["all_wrong_targets"]["mean_p_decline_complete"])
        self.assertEqual(history["all_true_targets"]["greedy_choice_changed"], 4)
        self.assertEqual(rows["a2"]["decision_weight"], 3.0)
        for needle in ("# Toy probe", "Screening count: 3 distinct expert-failed roots", "Not met", "## Observation mode complete_history_v1",
                       "After a forced decline of a wrong target: 3 continuations", "bounded history against complete history",
                       "shortfalls: wrong_target: 0 of 1"):
            self.assertIn(needle, text)


if __name__ == "__main__":
    unittest.main()
