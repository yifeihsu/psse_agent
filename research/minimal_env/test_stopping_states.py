"""The stopping-state replay and the evidence probe's rows, on compound cases generated here (IEEE 14, node/breaker).

Two classes: a breaker, a line and a meter together; one faulted line.  The cases are built by the scenario
generator as the environment's own tests build them, so nothing depends on data outside the repository.
"""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from psse_env.minimal import truth
from psse_env.minimal.env import MinimalDiagnosisEnv
from psse_env.minimal.episode import ScriptedPolicy, compact, run_episode
from psse_env.minimal.kernel import NumericalKernel
from psse_env.minimal.scoring import audit_scenario, physical_comparison
from psse_env.providers.scenario_generator import DEFAULT_COMPOUND_CLASSES, Round0ScenarioGenerator
from research.minimal_env import evidence_probe, stopping_states

GENERATOR_OPTIONS = dict(evidence_profile="scada_only", admission_mode="physical", normalized_residual_threshold=4.0,
                         min_measurement_error_sigma=10.0)
ROOTS: dict[str, dict] = {}
KERNEL: NumericalKernel | None = None
PROBE_KERNEL: NumericalKernel | None = None
INSPECT = {"tool": "inspect", "arguments": {"view": "breakers"}}
RESOLVED = {"tool": "finish", "arguments": {"outcome": "resolved"}}


def setUpModule() -> None:
    global KERNEL, PROBE_KERNEL
    from research.compound_errors.build_suite import envelope_of

    generator = Round0ScenarioGenerator(seed=11, **GENERATOR_OPTIONS)
    KERNEL, PROBE_KERNEL = NumericalKernel(), NumericalKernel()
    for name in ("rx_cb_meter", "control_rx"):
        spec = next(item for item in DEFAULT_COMPOUND_CLASSES if item.name == name)
        # The first case the truth repairs stands for its class.
        for row in generator._build_compound_class(spec, 4):
            root = envelope_of(row)
            if truth.build_success(root, MinimalDiagnosisEnv(KERNEL))["solved"]:
                ROOTS[name] = root
                break
        assert name in ROOTS, (name, generator.skipped[-3:])


def _stored(root: dict, actions: list[dict]) -> dict:
    """An episode record as research.minimal_env.evaluate stores it."""
    episode = compact(run_episode(MinimalDiagnosisEnv(KERNEL), root, ScriptedPolicy(copy.deepcopy(actions)), keep_observations=False))
    episode["compound_class"] = (root.get("audit") or {}).get("compound", {}).get("class")
    return json.loads(json.dumps(episode))


def _analyse(root: dict, record: dict) -> dict:
    return stopping_states.analyse(root, record, split="test", kernel=KERNEL, probe_kernel=PROBE_KERNEL)


def _healthy_breaker_flip(root: dict) -> dict:
    """A correction of a healthy breaker that disconnects a line: damage the truth sees whatever the tool reports."""
    env = MinimalDiagnosisEnv(KERNEL)
    env.reset(root)
    observation, _ = env.step(INSPECT)
    breaker = next(item for item in truth.plausible_errors(root, observation) if item["tool"] == "correct_topology")
    return breaker


class CorrectionTests(unittest.TestCase):
    def test_the_corrections_left_at_the_first_state_are_the_truths_own(self) -> None:
        for name, root in ROOTS.items():
            scenario = audit_scenario(root)
            env = MinimalDiagnosisEnv(KERNEL)
            env.reset(root)
            left = stopping_states.corrections_left(scenario, physical_comparison(scenario, env.final_state()))
            self.assertEqual(left, truth.truth_corrections(root), name)

    def test_a_correction_undone_is_told_apart_from_one_in_effect_and_one_never_made(self) -> None:
        error = {"kind": "line", "line": 7, "repaired": False}
        refit = {"tool": "correct_parameters", "arguments": {"line": 7}, "changes": []}
        # A pair refit names one line and changes another: the changed line counts as corrected.
        pair = {"tool": "correct_parameters", "arguments": {"line": 6},
                "changes": [{"object": "line", "line": 7, "field": "x", "before": 0.2, "after": 0.1}]}
        self.assertEqual(stopping_states.error_standing(error, [{**refit, "in_effect": True}]), stopping_states.CORRECTED_NOT_REPAIRED)
        self.assertEqual(stopping_states.error_standing(error, [{**pair, "in_effect": True}]), stopping_states.CORRECTED_NOT_REPAIRED)
        self.assertEqual(stopping_states.error_standing(error, [{**refit, "in_effect": False}]), stopping_states.UNDONE)
        self.assertEqual(stopping_states.error_standing(error, []), stopping_states.NEVER_CORRECTED)
        meter = {"kind": "meter", "meter": 44, "repaired": False}
        group = {"tool": "correct_measurements", "arguments": {"meters": [3, 44]}, "changes": [], "in_effect": True}
        self.assertEqual(stopping_states.error_standing(meter, [group]), stopping_states.CORRECTED_NOT_REPAIRED)


class OffPathTests(unittest.TestCase):
    def test_each_step_is_read_against_what_is_left_at_its_state(self) -> None:
        scenario = {"true_topology_errors": [{"cb_name": "CB_1", "true_cb_closed": False}]}
        physical = {"errors": [{"kind": "line", "line": 7, "repaired": False}, {"kind": "breaker", "breaker": "CB_1", "repaired": False},
                               {"kind": "meter", "meter": 44, "repaired": False}, {"kind": "meter", "meter": 3, "repaired": True}],
                    "damage": [], "solved": False}
        off = stopping_states.off_path
        self.assertIsNone(off(INSPECT, physical, scenario))
        self.assertIsNone(off({"tool": "correct_parameters", "arguments": {"line": 7}}, physical, scenario))
        self.assertEqual(off({"tool": "correct_parameters", "arguments": {"line": 8}}, physical, scenario), stopping_states.OFF_HEALTHY)
        self.assertIsNone(off({"tool": "correct_measurements", "arguments": {"meters": [44]}}, physical, scenario))
        # A repaired meter is healthy now: naming it again is off the path.
        self.assertEqual(off({"tool": "correct_measurements", "arguments": {"meters": [3, 44]}}, physical, scenario), stopping_states.OFF_HEALTHY)
        self.assertIsNone(off({"tool": "correct_topology", "arguments": {"breaker": "CB_1", "status": "open"}}, physical, scenario))
        self.assertEqual(off({"tool": "correct_topology", "arguments": {"breaker": "CB_1", "status": "closed"}}, physical, scenario),
                         stopping_states.OFF_STATUS)
        self.assertEqual(off({"tool": "correct_topology", "arguments": {"breaker": "CB_2", "status": "open"}}, physical, scenario),
                         stopping_states.OFF_HEALTHY)
        self.assertEqual(off({"tool": "undo", "arguments": {}}, physical, scenario), stopping_states.OFF_UNDO)
        self.assertEqual(off(RESOLVED, physical, scenario), stopping_states.OFF_FINISH)
        self.assertIsNone(off({"tool": "undo", "arguments": {}}, {**physical, "damage": [{"kind": "line", "line": 9}]}, scenario))
        repaired = {"errors": [{**error, "repaired": True} for error in physical["errors"]], "damage": [], "solved": True}
        self.assertIsNone(off(RESOLVED, repaired, scenario))


class ReplayTests(unittest.TestCase):
    def test_a_stored_episode_replays_to_its_score_and_a_changed_record_is_caught(self) -> None:
        root = ROOTS["control_rx"]
        record = _stored(root, [INSPECT] + truth.truth_corrections(root) + [RESOLVED])
        result = _analyse(root, record)
        self.assertTrue(result["record"]["reproduced"], result["record"]["mismatches"])
        self.assertTrue(result["record"]["solved"])
        self.assertIsNone(result["scoring_input"])  # a repaired episode has no stopping state to score
        changed = copy.deepcopy(record)
        changed["score"]["declared"] = "unresolved"
        changed["steps"][1]["result"]["to_version"] = 9
        mismatches = _analyse(root, changed)["record"]["mismatches"]
        self.assertTrue(any(item.startswith("score declared") for item in mismatches), mismatches)
        self.assertTrue(any("to_version" in item for item in mismatches), mismatches)

    def test_an_episode_stopping_with_errors_left_gets_the_truths_way_on(self) -> None:
        root = ROOTS["rx_cb_meter"]
        corrections = truth.truth_corrections(root)
        breaker = corrections[0]
        self.assertEqual(breaker["tool"], "correct_topology")
        record = _stored(root, [INSPECT, breaker, RESOLVED])
        result = _analyse(root, record)
        out = result["record"]
        self.assertTrue(out["reproduced"], out["mismatches"])
        self.assertFalse(out["solved"])
        stop = out["stop"]
        self.assertEqual(stop["index"], 2)
        self.assertEqual(stop["action"], RESOLVED)
        self.assertEqual({error["standing"] for error in stop["errors_left"]}, {stopping_states.NEVER_CORRECTED})
        self.assertEqual({error["kind"] for error in stop["errors_left"]}, {"line", "meter"})
        continuation = out["continuation"]
        self.assertTrue(continuation["repairs"], continuation)
        self.assertFalse(continuation["undo_needed"])
        self.assertIn(corrections[1], continuation["first_actions"])
        scoring = result["scoring_input"]
        self.assertEqual(scoring["step"], 3)
        roles = [(item["role"], item["action"]["tool"]) for item in scoring["actions"]]
        self.assertEqual(roles[0], ("own", "finish"))
        self.assertIn(("repairing", corrections[1]["tool"]), roles)
        self.assertEqual(roles[-1], ("other_finish", "finish"))
        self.assertEqual(scoring["observation"]["budget"]["used"], 2)  # the observation the finish was chosen on

    def test_the_first_step_off_the_path_is_found_before_the_stop(self) -> None:
        root = ROOTS["rx_cb_meter"]
        corrections = truth.truth_corrections(root)
        env = MinimalDiagnosisEnv(KERNEL)
        env.reset(root)
        env.step(INSPECT)
        observation, _ = env.step(corrections[0])
        healthy = next(item for item in truth.plausible_errors(root, observation) if item["tool"] == "correct_parameters")
        record = _stored(root, [INSPECT, corrections[0], healthy, RESOLVED])
        self.assertEqual(record["steps"][2]["result"]["status"], "ok")
        result = stopping_states.divergence(root, record, split="test", kernel=KERNEL)
        found = result["record"]["divergence"]
        self.assertEqual((found["index"], found["kind"], found["is_stop"]), (2, stopping_states.OFF_HEALTHY, False))
        self.assertEqual(found["action"], healthy)
        way = result["record"]["continuation"]
        self.assertTrue(way["repairs"], way)
        self.assertFalse(way["undo_needed"])
        scoring = result["scoring_input"]
        self.assertEqual((scoring["point"], scoring["step"]), ("divergence", 3))
        self.assertEqual(scoring["actions"][0], {"role": "own", "action": healthy})
        self.assertIn(corrections[1], [item["action"] for item in scoring["actions"]])
        summary = stopping_states.summarize_divergence([result["record"]])
        self.assertEqual(summary["totals"]["before the stopping state"], 1)

    def test_a_kept_correction_of_a_healthy_breaker_is_undone_first(self) -> None:
        root = ROOTS["control_rx"]
        record = _stored(root, [INSPECT, _healthy_breaker_flip(root), RESOLVED])
        out = _analyse(root, record)["record"]
        self.assertTrue(out["reproduced"], out["mismatches"])
        self.assertTrue(out["stop"]["damage"])
        continuation = out["continuation"]
        self.assertTrue(continuation["repairs"], continuation)
        self.assertTrue(continuation["undo_needed"])
        self.assertEqual(continuation["base"], 0)
        self.assertIn({"tool": "undo", "arguments": {}}, continuation["first_actions"])
        self.assertIn({"tool": "undo", "arguments": {"version": 0}}, continuation["first_actions"])
        summary = stopping_states.summarize([out])
        self.assertEqual(summary["continuations"]["repairs after an undo"], 1)


class ProbeRowTests(unittest.TestCase):
    def test_rows_label_the_unrepaired_objects_and_no_column_names_an_object(self) -> None:
        root = ROOTS["rx_cb_meter"]
        corrections = truth.truth_corrections(root)
        result = _analyse(root, _stored(root, [INSPECT, corrections[0], RESOLVED]))
        packed = result["probe"]
        from psse_env.minimal.scoring import truth_targets

        targets = truth_targets(root)
        true_objects = ({("meter", str(index)) for index in targets["meters"]} | {("line", str(line)) for line in targets["lines"]}
                        | {("breaker", name) for name in targets["breakers"]})
        self.assertEqual(len(targets["breakers"]), 1)

        def labelled(state: int) -> set[tuple[str, str]]:
            rows = (packed["cand_state"] == state) & (packed["cand_label"] == 1)
            return {(evidence_probe.KINDS[int(kind)], str(item))
                    for kind, item in zip(packed["cand_kind"][rows], np.asarray(packed["cand_object"])[rows])}

        self.assertEqual(labelled(0), true_objects)
        self.assertEqual(labelled(int(packed["state_index"][-1])), true_objects - {("breaker", targets["breakers"][0])})
        first = packed["cand_state"] == 0
        kinds = packed["cand_kind"][first]
        self.assertEqual(int((kinds == evidence_probe.KINDS.index("line")).sum()), 20)
        self.assertEqual(int((kinds == evidence_probe.KINDS.index("breaker")).sum()), 73)
        for name in evidence_probe.CANDIDATE_COLUMNS + evidence_probe.STATE_COLUMNS:
            # No column is a table's identity field or holds an object's index, number or name.
            self.assertNotIn(name, ("line", "index", "cb", "bus", "at", "from", "to", "meter", "breaker"))
            self.assertFalse(set(name.split("_")) & {"index", "number", "object", "id", "cb"}, name)
        self.assertEqual(packed["state"].shape[1], len(evidence_probe.STATE_COLUMNS))
        self.assertEqual(packed["candidates"].shape[1], len(evidence_probe.CANDIDATE_COLUMNS))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "probe.npz"
            evidence_probe.save_packed([packed, packed | {"case": "r0_zz"}], path)
            data = evidence_probe.load(path)
            self.assertEqual(len(data["state_x"]), 2 * len(packed["state_index"]))
            self.assertEqual(int(data["cand_state"].max()), len(data["state_x"]) - 1)
            self.assertEqual(list(data["columns_candidate"]), list(evidence_probe.CANDIDATE_COLUMNS))

    def test_places_count_the_candidates_above_the_best_true_one(self) -> None:
        data = {"cand_state": np.asarray([0, 0, 0, 0, 1, 1]), "cand_label": np.asarray([0, 1, 0, 1, 0, 1]),
                "cand_kind": np.asarray([0, 1, 1, 2, 0, 0], dtype=np.int8),
                "cand_object": np.asarray(["3", "7", "8", "CB_1", "5", "6"]),
                "cand_x": np.zeros((6, len(evidence_probe.CANDIDATE_COLUMNS)), dtype=np.float32)}
        scores = np.asarray([0.9, 0.5, 0.7, 0.4, 0.2, 0.1])
        found = evidence_probe.places(data, scores, np.asarray([True, True]))
        self.assertEqual([(item["object"], item["place"], item["place_within_kind"]) for item in found], [("7", 3, 2), ("6", 2, 2)])


def _synthetic_episode(case: str, rng: np.random.Generator, *, solved: bool) -> dict:
    """Three states of ten candidates; the wrong object is the meter whose residual stands out, until it is repaired."""
    rows = []
    for index in range(3):
        left = 0 if solved and index == 2 else 1
        candidates = np.full((10, len(evidence_probe.CANDIDATE_COLUMNS)), np.nan, dtype=np.float32)
        candidates[:, evidence_probe.CANDIDATE_COLUMNS.index("kind_meter")] = 1.0
        residual = rng.uniform(0.0, 2.0, size=10).astype(np.float32)
        labels = np.zeros(10, dtype=np.int8)
        if left:
            residual[3] = rng.uniform(4.0, 9.0)
            labels[3] = 1
        candidates[:, evidence_probe.CANDIDATE_COLUMNS.index("mt_res")] = residual
        state = np.full(len(evidence_probe.STATE_COLUMNS), np.nan, dtype=np.float32)
        state[evidence_probe.STATE_COLUMNS.index("meter_r1")] = residual.max()
        state[evidence_probe.STATE_COLUMNS.index("passes")] = 0.0 if left else 1.0
        state[evidence_probe.STATE_COLUMNS.index("breakers_shown")] = 0.0
        rows.append({"state": state, "solved": not left, "errors_left": left, "damaged": 0, "candidates": candidates,
                     "kinds": np.zeros(10, dtype=np.int8), "objects": [str(number) for number in range(10)], "labels": labels})
    return evidence_probe.pack_states(case, "x", rows, stop_index=2, episode_solved=solved, reproduced=True)


class ProbeFitTests(unittest.TestCase):
    def test_the_probes_read_a_cue_the_rows_carry(self) -> None:
        rng = np.random.default_rng(3)
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            # Enough training states for the probe's leaves of 40 to isolate the repaired finishes.
            for split, count in (("train", 160), ("dev", 12)):
                episodes = [_synthetic_episode(f"r0_{split}{number:02d}", rng, solved=number % 2 == 0) for number in range(count)]
                evidence_probe.save_packed(episodes, out / f"probe_{split}.npz")
                records = [{"case": item["case"], "stop": {"errors_left": [] if item["episode_solved"] else
                                                           [{"kind": "meter", "object": "3", "standing": "never corrected"}]}}
                           for item in episodes]
                (out / f"episodes_{split}.jsonl").write_text("".join(json.dumps(item) + "\n" for item in records), encoding="utf-8")
            reference = out / "reference.json"
            reference.write_text(json.dumps({f"r0_dev{number:02d}": {"expert": False, "truth": number < 10} for number in range(12)}),
                                 encoding="utf-8")
            self.assertEqual(evidence_probe.main(["--input-dir", folder, "--train", "train", "--evaluate", "dev",
                                                  "--evaluate-cases", str(reference)]), 0)
            report = json.loads((out / "probe_report.json").read_text(encoding="utf-8"))
        state = next(item for item in report["state"] if item["reading"] == "inspect" and not item["counts"])
        self.assertEqual(state["evaluate"]["unrepaired_stops"], 5)  # odd numbers below 10: the two cases above are left out
        self.assertEqual(state["evaluate"]["repaired_finishes"], 5)
        self.assertGreater(state["evaluate"]["auc_stopping_vs_finished"], 0.9)
        candidate = next(item for item in report["candidate"] if item["reading"] == "inspect")
        self.assertEqual(candidate["dev"]["first"], "5 of 5")
        self.assertEqual(candidate["dev"]["by_standing"]["never corrected"]["stops"], 5)
        self.assertEqual(sum(1 for key in report["stopping_states"] if key.startswith("dev:")), 5)


if __name__ == "__main__":
    unittest.main()
