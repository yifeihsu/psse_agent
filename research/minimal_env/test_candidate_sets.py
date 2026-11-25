"""The candidate read-out without the model: the search against brute force, the candidates, the reader.

The model's side is the RL update's forward pass with one line changed (``position_logprobs``), checked here against
``completion_logprobs`` on a small torch model; the loader and the renderers are first exercised by the cell's job.
"""
from __future__ import annotations

import io
import json
import math
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from research.minimal_env import candidate_sets
from research.minimal_env.candidate_sets import CORRECTION, CallSearch

try:
    import torch
except Exception:  # pragma: no cover - the suite runs without torch on the login node
    torch = None

END = 0


class _HashModel:
    """A next-token distribution for any prefix over tokens ``0..vocab-1``, fixed by a seed."""

    def __init__(self, seed: int, vocab: int, sharp: float = 3.0) -> None:
        self.seed, self.vocab, self.sharp = seed, vocab, sharp
        self.passes: list[tuple[int, ...]] = []

    def row(self, prefix: tuple[int, ...]) -> list[float]:
        rng = random.Random(f"{self.seed}:{prefix}")
        logits = [rng.gauss(0.0, self.sharp) for _ in range(self.vocab)]
        peak = max(logits)
        total = peak + math.log(sum(math.exp(value - peak) for value in logits))
        return [value - total for value in logits]

    def forward(self, sequence, wanted):
        self.passes.append(tuple(sequence))
        return [[self.row(tuple(sequence[:position]))[token] for token in tokens] for position, tokens in enumerate(wanted)]

    def exact(self, sequence) -> float:
        return sum(self.row(tuple(sequence[:position]))[token] for position, token in enumerate(sequence))


def _random_calls(rng: random.Random, count: int, vocab: int) -> list[list[int]]:
    """Distinct calls over tokens ``1..vocab-1``, each closed by ``END``: none is the beginning of another."""
    found: set[tuple[int, ...]] = set()
    while len(found) < count:
        found.add(tuple(rng.randrange(1, vocab) for _ in range(rng.randint(1, 4))))
    return [list(call) + [END] for call in sorted(found)]


class _CallModel:
    """A distribution over whole calls; the next-token probabilities are its exact marginals."""

    def __init__(self, calls: dict[tuple[int, ...], float]) -> None:
        self.calls = dict(calls)
        assert sum(self.calls.values()) <= 1.0 + 1e-9

    def _mass(self, prefix: tuple[int, ...]) -> float:
        return sum(value for call, value in self.calls.items() if call[: len(prefix)] == prefix)

    def forward(self, sequence, wanted):
        rows = []
        for position, tokens in enumerate(wanted):
            prefix = tuple(sequence[:position])
            # The root's mass is 1: what the listed calls do not hold belongs to anything else the model could write.
            below = 1.0 if position == 0 else self._mass(prefix)
            rows.append([math.log(self._mass(prefix + (token,)) / below) for token in tokens])
        return rows


class SearchTests(unittest.TestCase):
    def test_the_most_probable_calls_are_the_brute_force_ones(self) -> None:
        for seed in range(12):
            rng = random.Random(seed)
            vocab = rng.choice([4, 6, 9])
            calls = _random_calls(rng, rng.randint(8, 60), vocab)
            model = _HashModel(seed, vocab)
            truth = [model.exact(call) for call in calls]
            for top in (1, 3, 5):
                model.passes.clear()
                search = CallSearch(calls, model.forward)
                found = search.top(counted=range(len(calls)), top=top, max_passes=10_000)
                self.assertTrue(found["complete"])
                self.assertEqual(found["groups"], min(top, len(calls)))
                wanted = sorted(truth, reverse=True)[:top]
                self.assertEqual(len(found["order"]), len(wanted))
                for index, value in zip(found["order"], wanted):
                    self.assertAlmostEqual(truth[index], value, places=9)
                    self.assertAlmostEqual(search.logprob(index), value, places=9)
                self.assertEqual(search.passes, len(model.passes))
                self.assertLessEqual(search.passes, len(calls))
                last = truth[found["order"][-1]]
                for index in range(len(calls)):
                    if search.logprob(index) is None:
                        # What was not scored is bounded, the bound holds, and it is below everything found.
                        self.assertGreaterEqual(search.bound(index), truth[index] - 1e-9)
                        self.assertLessEqual(search.bound(index), last + 1e-9)
                    else:
                        self.assertAlmostEqual(search.logprob(index), truth[index], places=9)

    def test_a_peaked_model_needs_far_fewer_passes_than_calls(self) -> None:
        rng = random.Random(7)
        calls = _random_calls(rng, 200, 12)
        search = CallSearch(calls, _HashModel(7, 12, sharp=4.0).forward)
        found = search.top(counted=range(len(calls)), top=5, max_passes=10_000)
        self.assertTrue(found["complete"])
        self.assertLess(search.passes, 40)

    def test_groups_count_distinct_objects_and_only_counted_calls_are_found(self) -> None:
        for seed in range(6):
            rng = random.Random(100 + seed)
            calls = _random_calls(rng, 40, 6)
            model = _HashModel(100 + seed, 6)
            truth = [model.exact(call) for call in calls]
            counted = [index for index in range(len(calls)) if index % 3]
            search = CallSearch(calls, model.forward)
            found = search.top(counted=counted, group=lambda index: index // 4, top=3, max_passes=10_000)
            self.assertTrue(set(found["order"]) <= set(counted))
            groups = []
            for index in sorted(counted, key=lambda index: -truth[index]):
                if index // 4 not in groups:
                    groups.append(index // 4)
            seen = []
            for index in found["order"]:
                if index // 4 not in seen:
                    seen.append(index // 4)
            self.assertEqual(seen, groups[:3])
            self.assertEqual(found["groups"], 3)

    def test_a_call_scored_beforehand_keeps_its_place(self) -> None:
        rng = random.Random(3)
        calls = _random_calls(rng, 30, 6)
        model = _HashModel(3, 6)
        truth = [model.exact(call) for call in calls]
        worst = min(range(len(calls)), key=lambda index: truth[index])
        search = CallSearch(calls, model.forward)
        self.assertAlmostEqual(search.score(worst), truth[worst], places=9)
        found = search.top(counted=range(len(calls)), top=3, max_passes=10_000)
        self.assertNotIn(worst, found["order"])
        self.assertEqual([round(truth[index], 9) for index in found["order"]], [round(value, 9) for value in sorted(truth, reverse=True)[:3]])
        self.assertAlmostEqual(search.logprob(worst), truth[worst], places=9)

    def test_the_limit_stops_the_search_and_what_was_found_is_still_the_top(self) -> None:
        rng = random.Random(11)
        calls = _random_calls(rng, 80, 9)
        model = _HashModel(11, 9, sharp=0.2)  # nearly flat: the search cannot prune
        truth = [model.exact(call) for call in calls]
        search = CallSearch(calls, model.forward)
        found = search.top(counted=range(len(calls)), top=40, max_passes=6)
        self.assertFalse(found["complete"])
        self.assertEqual(search.passes, 6)
        self.assertLess(found["groups"], 40)
        ranked = sorted(truth, reverse=True)
        for place, index in enumerate(found["order"]):
            self.assertAlmostEqual(truth[index], ranked[place], places=9)
        highest = max(search.bound(index) for index in range(len(calls)) if search.logprob(index) is None)
        for index in found["order"]:
            self.assertGreaterEqual(truth[index], highest - 1e-9)

    def test_calls_that_cannot_share_a_tree_are_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "beginning"):
            CallSearch([[1, 2], [1, 2, 3]], lambda sequence, wanted: [])
        with self.assertRaisesRegex(ValueError, "beginning"):
            CallSearch([[1, 2, 3], [1, 2]], lambda sequence, wanted: [])
        with self.assertRaisesRegex(ValueError, "same tokens"):
            CallSearch([[1, 2], [1, 2]], lambda sequence, wanted: [])
        with self.assertRaisesRegex(ValueError, "no tokens"):
            CallSearch([[]], lambda sequence, wanted: [])

    def test_a_forward_pass_that_returns_something_else_fails(self) -> None:
        with self.assertRaises(RuntimeError):
            CallSearch([[1, END], [2, END]], lambda sequence, wanted: [[0.0, 0.0]]).score(0)
        with self.assertRaises(RuntimeError):
            CallSearch([[1, END], [2, END]], lambda sequence, wanted: [[-0.1], [0.0]]).score(0)
        with self.assertRaises(RuntimeError):
            CallSearch([[1, END], [2, END]], lambda sequence, wanted: [[0.5, -1.0], [0.0]]).score(0)
        with self.assertRaises(RuntimeError):
            CallSearch([[1, END], [2, END]], lambda sequence, wanted: [[float("nan"), -1.0], [0.0]]).score(0)

    def test_a_labels_probability_is_that_of_the_branches_only_it_owns(self) -> None:
        # Tool 7 has two calls, tool 8 one; token 5 starts both of tool 7's calls and only those.
        calls = [[5, 1, END], [5, 2, END], [6, END]]
        model = _CallModel({(5, 1, END): 0.2, (5, 2, END): 0.1, (6, END): 0.6})
        search = CallSearch(calls, model.forward)
        self.assertEqual(search.label_logprobs(["a", "a", "b"]), {"a": None, "b": None})
        search.score(2)
        found = search.label_logprobs(["a", "a", "b"])
        self.assertAlmostEqual(math.exp(found["a"]), 0.3)
        self.assertAlmostEqual(math.exp(found["b"]), 0.6)
        # A label spread over two branches: their probabilities add.
        search.score(0)
        spread = search.label_logprobs(["x", "y", "x"])
        self.assertAlmostEqual(math.exp(spread["x"]), 0.2 + 0.6)
        self.assertAlmostEqual(math.exp(spread["y"]), 0.1)
        self.assertAlmostEqual(math.exp(search.label_logprobs(["z", "z", "z"])["z"]), 1.0)


# ------------------------------------------------------------------ a state


def _observation(*, breakers=("CB_1_A", "CB_2_B"), breaker_model: bool = True, current: int = 2) -> dict:
    def estimate(tables: bool) -> dict:
        out = {"converged": True, "chi_square": 10.0, "chi_square_threshold": 100.0}
        if tables:
            out["lines"] = [{"line": number} for number in (1, 2, 12)]
            out["meters"] = [{"index": number} for number in (0, 3, 14)]
        return out

    versions = [{"version": 0, "estimate": estimate(False)},
                {"version": 1, "estimate": estimate(True), "substation": {"breakers": [{"cb": name, "status": "closed"} for name in breakers]}},
                {"version": 2, "estimate": estimate(True)}]
    return {"breaker_model": breaker_model, "current_version": current, "versions": versions, "budget": {"used": 3},
            "history": [{"action": {"tool": "inspect", "arguments": {"view": "breakers", "flip": ["CB_9_Z"]}}, "result": {"status": "ok"}}],
            "modifications": [{"tool": "correct_topology", "arguments": {"breaker": "CB_7_Q", "status": "open"}, "in_effect": True}]}


TOOL_TOKENS = {"finish": 1, "undo": 2, "inspect": 3, "correct_parameters": 4, "correct_topology": 5, "correct_measurements": 6}


def _render(observation, action) -> list[int]:
    """A stand-in for the chat template: a tool token, the arguments character by character, a closing token."""
    if action["tool"] == "inspect" and action["arguments"].get("view") == "estimate" and observation.get("refuse_estimate"):
        raise ValueError("does not render")
    text = json.dumps(action["arguments"], sort_keys=True, separators=(",", ":"))
    return [TOOL_TOKENS[action["tool"]]] + [100 + ord(character) for character in text] + [END]


def _model_of(observation, probabilities: dict[str, float], floor: float = 1e-9) -> _CallModel:
    """A model that gives the named calls (by signature) their probability and every other listed call ``floor``."""
    calls = {}
    for candidate in candidate_sets.candidates(observation):
        try:
            tokens = tuple(_render(observation, candidate["action"]))
        except ValueError:
            continue
        calls[tokens] = probabilities.get(candidate_sets.signature(candidate["action"]), floor)
    for signature, value in probabilities.items():
        action = json.loads(signature)
        calls.setdefault(tuple(_render(observation, action)), value)
    return _CallModel(calls)


def _sig(tool: str, **arguments) -> str:
    return candidate_sets.signature({"tool": tool, "arguments": arguments})


FINISH_RESOLVED = {"tool": "finish", "arguments": {"outcome": "resolved"}}
FINISH_UNRESOLVED = {"tool": "finish", "arguments": {"outcome": "unresolved"}}


def _item(case: str = "r0_a", *, errors_left=None, actions=None, observation=None, split: str = "dev") -> dict:
    return {"contract": candidate_sets.INPUT_CONTRACT, "case": case, "split": split, "class": "rx2_adjacent", "step": 4,
            "point": "stop", "observation": observation if observation is not None else _observation(),
            "actions": actions if actions is not None else [{"role": "own", "action": FINISH_RESOLVED},
                                                             {"role": "other_finish", "action": FINISH_UNRESOLVED}],
            "truth": {"errors_left": errors_left if errors_left is not None else [{"kind": "line", "object": "12", "standing": "never corrected"}],
                      "damaged": 0, "declared": "resolved", "outcome": "resolved", "estimator_passes": True,
                      "continuation_repairs": True, "undo_needed": False}}


def _score(item: dict, probabilities: dict[str, float], **kwargs) -> dict:
    model = _model_of(item["observation"], probabilities)
    passes: list[int] = []

    def positions(prompt, completion, wanted):
        assert prompt == [7, 8, 9]
        passes.append(len(completion))
        return model.forward(completion, wanted)

    result = candidate_sets.score_state(item, view=lambda observation: {**observation, "viewed": True},
                                        render=lambda observation: "prompt", prompt_ids=lambda text: ([7, 8, 9], 3),
                                        render_target=_render, positions=positions,
                                        tokens=lambda ids: [f"t{value}" for value in ids], **kwargs)
    result["test_passes"] = len(passes)
    return result


class CandidateTests(unittest.TestCase):
    def test_the_candidates_are_what_the_observation_names(self) -> None:
        listed = candidate_sets.candidates(_observation())
        by_kind: dict[str, list] = {}
        for entry in listed:
            by_kind.setdefault(entry["kind"], []).append(entry)
        self.assertEqual([entry["action"]["arguments"] for entry in by_kind["finish"]], [{"outcome": "resolved"}, {"outcome": "unresolved"}])
        # The bare undo and one per saved version other than the current one.
        self.assertEqual([entry["action"]["arguments"] for entry in by_kind["undo"]], [{}, {"version": 0}, {"version": 1}])
        self.assertEqual([entry["action"]["arguments"]["view"] for entry in by_kind["inspect"]], ["estimate", "breakers"])
        objects = [tuple(entry["object"]) for entry in by_kind[CORRECTION]]
        self.assertEqual([item for kind, item in objects if kind == "line"], ["1", "2", "12"])
        self.assertEqual([item for kind, item in objects if kind == "meter"], ["0", "3", "14"])
        # Both statuses of every breaker a table shows or an earlier call of the episode named.
        breakers = [(entry["object"][1], entry["action"]["arguments"]["status"]) for entry in by_kind[CORRECTION] if entry["object"][0] == "breaker"]
        self.assertEqual(breakers, [(name, status) for name in ("CB_1_A", "CB_2_B", "CB_7_Q", "CB_9_Z") for status in ("open", "closed")])
        self.assertTrue(all(entry["object"] is None for entry in listed if entry["kind"] != CORRECTION))
        self.assertEqual(len({candidate_sets.signature(entry["action"]) for entry in listed}), len(listed))

    def test_without_a_breaker_model_the_topology_candidates_are_line_statuses(self) -> None:
        listed = candidate_sets.candidates(_observation(breaker_model=False))
        topology = [entry for entry in listed if entry["action"]["tool"] == "correct_topology"]
        self.assertEqual([(entry["object"], entry["action"]["arguments"]["status"]) for entry in topology],
                         [(["line_status", str(line)], status) for line in (1, 2, 12) for status in ("in", "out")])
        self.assertNotIn("breakers", [entry["action"]["arguments"].get("view") for entry in listed])

    def test_a_group_of_meters_is_one_object_of_its_own_kind(self) -> None:
        self.assertEqual(candidate_sets.call_object({"tool": "correct_measurements", "arguments": {"meters": [14, 3]}}), ["meters", "3,14"])
        self.assertEqual(candidate_sets.call_object({"tool": "correct_measurements", "arguments": {"meters": [3]}}), ["meter", "3"])
        self.assertIsNone(candidate_sets.call_object(FINISH_RESOLVED))


class ScoreStateTests(unittest.TestCase):
    PROBABILITIES = {
        _sig("finish", outcome="resolved"): 0.80,
        _sig("finish", outcome="unresolved"): 0.02,
        _sig("correct_parameters", line=2): 0.10,
        _sig("correct_topology", breaker="CB_1_A", status="open"): 0.04,
        _sig("correct_measurements", meters=[3]): 0.02,
        _sig("correct_parameters", line=12): 0.001,
    }

    def test_the_most_probable_objects_and_every_other_call_are_exact(self) -> None:
        result = _score(_item(), self.PROBABILITIES, top=3)
        self.assertEqual(result["contract"], candidate_sets.SCORES_CONTRACT)
        self.assertTrue(result["complete"])
        self.assertEqual(result["exact_objects"], 3)
        self.assertEqual(result["passes"], result["test_passes"])
        self.assertEqual(result["shown"], {"lines": 3, "meters": 3, "breakers": 4})
        by_signature = {candidate_sets.signature(call["action"]): call for call in result["calls"]}
        for signature, value in self.PROBABILITIES.items():
            self.assertAlmostEqual(math.exp(by_signature[signature]["logprob"]), value, places=6, msg=signature)
        # Every call that is not a correction is scored whatever its probability.
        kinds = {call["kind"] for call in result["calls"]}
        self.assertTrue({"finish", "undo", "inspect", CORRECTION} <= kinds)
        self.assertEqual(sum(1 for call in result["calls"] if call["kind"] == "undo"), 3)
        self.assertEqual(by_signature[_sig("finish", outcome="resolved")]["roles"], ["own"])
        self.assertEqual(by_signature[_sig("finish", outcome="unresolved")]["roles"], ["other_finish"])
        self.assertEqual(by_signature[_sig("finish", outcome="resolved")]["token_text"][0], "t1")
        # The calls are stored most probable first, and far fewer are scored than listed.
        self.assertEqual([call["logprob"] for call in result["calls"]], sorted((call["logprob"] for call in result["calls"]), reverse=True))
        self.assertLess(result["passes"], result["candidates"]["listed"])
        self.assertGreater(result["unscored"]["corrections"], 0)
        self.assertLessEqual(result["unscored"]["highest_bound"], by_signature[_sig("correct_measurements", meters=[3])]["logprob"] + 1e-9)

    def test_the_tools_probabilities_are_the_branches_of_their_names(self) -> None:
        result = _score(_item(), self.PROBABILITIES, top=3)
        tools = {tool: math.exp(value) for tool, value in result["tool_logprob"].items()}
        self.assertAlmostEqual(tools["finish"], 0.82, places=6)
        self.assertAlmostEqual(tools["correct_parameters"], 0.101, places=5)
        self.assertAlmostEqual(tools["correct_topology"], 0.04, places=5)
        self.assertAlmostEqual(tools["correct_measurements"], 0.02, places=5)

    def test_the_object_the_truth_names_is_scored_wherever_it_stands(self) -> None:
        # Line 12 has probability 0.001: far below the three most probable objects, and exact all the same.
        result = _score(_item(), self.PROBABILITIES, top=1)
        line = next(call for call in result["calls"] if call["object"] == ["line", "12"])
        self.assertAlmostEqual(math.exp(line["logprob"]), 0.001, places=7)
        self.assertEqual(result["exact_objects"], 1)
        # A breaker the truth names is scored in both statuses.
        breaker = _score(_item(errors_left=[{"kind": "breaker", "object": "CB_2_B", "standing": "never corrected"}]), self.PROBABILITIES, top=1)
        self.assertEqual(sorted(call["action"]["arguments"]["status"] for call in breaker["calls"] if call["object"] == ["breaker", "CB_2_B"]),
                         ["closed", "open"])
        # An object the prompt does not show has no candidate: nothing is scored for it.
        hidden = _score(_item(errors_left=[{"kind": "breaker", "object": "CB_HIDDEN", "standing": "never corrected"}]), self.PROBABILITIES, top=1)
        self.assertFalse([call for call in hidden["calls"] if call["object"] == ["breaker", "CB_HIDDEN"]])

    def test_a_call_that_is_no_listed_candidate_is_scored_and_never_counted(self) -> None:
        group = {"tool": "correct_measurements", "arguments": {"meters": [3, 14]}}
        probabilities = {**self.PROBABILITIES, _sig("finish", outcome="resolved"): 0.75, candidate_sets.signature(group): 0.03}
        item = _item(actions=[{"role": "own", "action": FINISH_RESOLVED}, {"role": "repairing", "action": group},
                              {"role": "repairing", "action": {"tool": "correct_parameters", "arguments": {"line": 12}}}])
        result = _score(item, probabilities, top=2)
        extra = next(call for call in result["calls"] if call["object"] == ["meters", "3,14"])
        self.assertFalse(extra["listed"])
        self.assertEqual(extra["roles"], ["repairing"])
        self.assertAlmostEqual(math.exp(extra["logprob"]), 0.03, places=6)
        listed = next(call for call in result["calls"] if call["object"] == ["line", "12"])
        self.assertTrue(listed["listed"])
        self.assertEqual(listed["roles"], ["repairing"])
        # The two most probable listed objects are line 2 and the breaker; the group of meters is not one of them.
        reading = candidate_sets.state_reading(result)
        self.assertEqual(reading["top_object"], ["line", "2"])
        self.assertEqual(result["exact_objects"], 2)

    def test_a_call_that_does_not_render_is_recorded_and_left_out(self) -> None:
        item = _item(observation={**_observation(), "refuse_estimate": True})
        result = _score(item, self.PROBABILITIES, top=1)
        self.assertEqual(result["candidates"]["unrenderable"], 1)
        self.assertEqual(result["unrenderable"][0]["action"], {"tool": "inspect", "arguments": {"view": "estimate"}})
        self.assertFalse([call for call in result["calls"] if call["action"]["arguments"].get("view") == "estimate"])

    def test_the_token_cache_is_used_and_checked(self) -> None:
        cache: dict[str, list[int]] = {}
        first = _score(_item(), self.PROBABILITIES, top=2, cache=cache)
        second = _score(_item("r0_b"), self.PROBABILITIES, top=2, cache=cache)
        self.assertEqual(first["rendered_fresh"], first["candidates"]["listed"])
        # At the next state only the calls that are always scored are rendered again.
        self.assertLess(second["rendered_fresh"], first["rendered_fresh"])
        self.assertEqual([call["logprob"] for call in first["calls"]], [call["logprob"] for call in second["calls"]])
        key = next(key for key in cache if '"finish"' in key)
        cache[key] = cache[key] + [5]
        with self.assertRaisesRegex(RuntimeError, "differ from one state to another"):
            _score(_item("r0_c"), self.PROBABILITIES, top=2, cache=cache)

    def test_the_limit_is_reported(self) -> None:
        flat = {candidate_sets.signature(entry["action"]): 0.01 for entry in candidate_sets.candidates(_observation())}
        result = _score(_item(), flat, top=5, max_passes=10)
        self.assertFalse(result["complete"])
        self.assertLess(result["exact_objects"], 5)
        self.assertEqual(result["passes"], 10)
        self.assertIsNotNone(result["unscored"]["highest_bound"])


# ------------------------------------------------------------------ the reader


def _call(tool: str, arguments: dict, logprob: float, *, roles=(), listed: bool = True) -> dict:
    action = {"tool": tool, "arguments": arguments}
    return {"action": action, "kind": candidate_sets.call_kind(action), "object": candidate_sets.call_object(action),
            "roles": list(roles), "listed": listed, "tokens": 5, "logprob": logprob}


def _result(case: str, *, errors_left, corrections, own=("finish", {"outcome": "resolved"}, math.log(0.8)), extra=(),
            highest=None, complete: bool = True, split: str = "dev", tools=None) -> dict:
    calls = [_call(own[0], own[1], own[2], roles=["own"])] + [_call(*item) if not isinstance(item, dict) else item for item in corrections]
    calls += list(extra)
    mass = {"correct_parameters": 0.0, "correct_topology": 0.0, "correct_measurements": 0.0}
    for call in calls:
        if call["kind"] == CORRECTION:
            mass[call["action"]["tool"]] += math.exp(call["logprob"])
    tool_logprob = {tool: math.log(max(value, 1e-12)) for tool, value in (tools or mass).items()}
    return {"contract": candidate_sets.SCORES_CONTRACT, "case": case, "split": split, "class": "x", "step": 4, "point": "stop",
            "passes": 12, "complete": complete, "exact_objects": 5, "top": 5,
            "unscored": {"corrections": 0 if highest is None else 9, "highest_bound": highest},
            "tool_logprob": tool_logprob, "calls": calls,
            "truth": {"errors_left": [{"kind": kind, "object": item, "standing": standing} for kind, item, standing in errors_left],
                      "damaged": 0, "declared": "resolved", "estimator_passes": True}}


LINE = "correct_parameters"
BREAKER = "correct_topology"
METER = "correct_measurements"


class ReadingTests(unittest.TestCase):
    def test_an_object_still_wrong_is_looked_up_among_the_most_probable_objects(self) -> None:
        result = _result("a", errors_left=[("line", "12", "never corrected")], corrections=[
            (LINE, {"line": 2}, math.log(0.10)), (BREAKER, {"breaker": "CB_1", "status": "open"}, math.log(0.04)),
            (BREAKER, {"breaker": "CB_1", "status": "closed"}, math.log(0.03)),   # the same object again: no new place
            (LINE, {"line": 12}, math.log(0.02)), (METER, {"meters": [3]}, math.log(0.01))])
        reading = candidate_sets.state_reading(result)
        self.assertEqual(reading["errors"][0]["place"], 3)
        self.assertEqual({size: reading["covered"][size]["any"] for size in ("1", "3", "5")}, {"1": False, "3": True, "5": True})
        self.assertEqual(reading["covered"]["3"]["all"], True)
        self.assertEqual((reading["errors_left"], reading["listed_left"]), (1, 1))
        self.assertEqual(reading["top_object"], ["line", "2"])
        self.assertFalse(reading["top_is_wrong_object"])
        self.assertAlmostEqual(reading["correction_mass"], 0.20)
        self.assertAlmostEqual(reading["top_share"], 0.5)
        self.assertAlmostEqual(reading["mass_on_objects_still_wrong"], 0.02)
        self.assertAlmostEqual(reading["share_on_objects_still_wrong"], 0.1)
        self.assertAlmostEqual(reading["own"], 0.8)
        self.assertEqual((reading["own_tool"], reading["own_outcome"]), ("finish", "resolved"))

    def test_any_and_all_differ_when_two_objects_are_left(self) -> None:
        result = _result("a", errors_left=[("line", "2", "never corrected"), ("breaker", "CB_9", "never corrected")], corrections=[
            (LINE, {"line": 2}, math.log(0.10)), (LINE, {"line": 5}, math.log(0.05))])
        reading = candidate_sets.state_reading(result)
        # The breaker is on no table of the prompt: it has no candidate and counts as not covered.
        self.assertEqual((reading["errors_left"], reading["listed_left"]), (2, 1))
        self.assertEqual(reading["covered"]["1"], {"any": True, "all": False})
        self.assertEqual(reading["covered"]["5"], {"any": True, "all": False})
        self.assertTrue(reading["top_is_wrong_object"])

    def test_a_state_with_nothing_left_has_no_coverage(self) -> None:
        reading = candidate_sets.state_reading(_result("a", errors_left=[], corrections=[(LINE, {"line": 2}, math.log(0.1))]))
        self.assertEqual(reading["covered"]["1"], {"any": None, "all": None})
        self.assertIsNone(reading["top_is_wrong_object"])

    def test_a_search_that_stopped_early_decides_only_what_it_can(self) -> None:
        corrections = [(LINE, {"line": 2}, math.log(0.10)), (LINE, {"line": 12}, math.log(0.02)), (LINE, {"line": 7}, math.log(0.001))]
        # Unscored calls may be as probable as 0.05: line 2 is above them (certain), line 12 is not.
        first = candidate_sets.state_reading(_result("a", errors_left=[("line", "2", "x")], corrections=corrections,
                                                     highest=math.log(0.05), complete=False))
        self.assertEqual(first["covered"]["1"]["any"], True)
        second = candidate_sets.state_reading(_result("a", errors_left=[("line", "12", "x")], corrections=corrections,
                                                      highest=math.log(0.05), complete=False))
        self.assertEqual(second["covered"]["1"]["any"], False)   # second among the scored: never first
        self.assertIsNone(second["covered"]["3"]["any"])         # second among the scored, but unscored calls may pass it
        third = candidate_sets.state_reading(_result("a", errors_left=[("line", "7", "x")], corrections=corrections,
                                                     highest=math.log(0.05), complete=False))
        self.assertEqual(third["covered"]["1"]["any"], False)
        self.assertIsNone(third["covered"]["5"]["any"])

    def test_the_kind_of_error_is_read_against_the_three_tools(self) -> None:
        result = _result("a", errors_left=[("breaker", "CB_1", "never corrected")], corrections=[
            (LINE, {"line": 2}, math.log(0.10)), (BREAKER, {"breaker": "CB_1", "status": "open"}, math.log(0.04)),
            (METER, {"meters": [3]}, math.log(0.01))])
        reading = candidate_sets.state_reading(result)
        self.assertEqual(reading["kinds"]["1"], {"any": False, "all": False})
        self.assertEqual(reading["kinds"]["2"], {"any": True, "all": True})

    def test_a_step_of_the_continuation_is_looked_up_among_the_listed_calls_that_go_on(self) -> None:
        undo = _call("undo", {}, math.log(0.06), roles=["repairing"])
        inspect = _call("inspect", {"view": "breakers"}, math.log(0.2))
        group = _call(METER, {"meters": [3, 14]}, math.log(0.5), roles=["repairing"], listed=False)
        result = _result("a", errors_left=[("line", "12", "x")], corrections=[(LINE, {"line": 2}, math.log(0.10))],
                         extra=[undo, inspect, group])
        reading = candidate_sets.state_reading(result)
        # inspect 0.2, line 2 0.10, undo 0.06: the undo is third; the unlisted group of meters is not ranked.
        self.assertEqual(reading["next_step"], {"1": False, "3": True, "5": True})
        self.assertTrue(reading["has_repairing"])
        self.assertAlmostEqual(reading["inspect_breakers"], 0.2)
        # The group is a correction on the way: its probability counts as mass on what is still wrong.
        self.assertAlmostEqual(reading["mass_on_objects_still_wrong"], 0.5)

    def test_the_summary_counts_per_split_and_group(self) -> None:
        results = [
            _result("a", errors_left=[("line", "2", "never corrected")], corrections=[(LINE, {"line": 2}, math.log(0.10)), (LINE, {"line": 5}, math.log(0.05))]),
            _result("b", errors_left=[("line", "5", "corrected, not within tolerance")],
                    corrections=[(LINE, {"line": 2}, math.log(0.10)), (LINE, {"line": 5}, math.log(0.05))]),
            _result("c", errors_left=[("breaker", "CB_9", "never corrected")], corrections=[(LINE, {"line": 2}, math.log(0.10))],
                    own=("finish", {"outcome": "unresolved"}, math.log(0.6))),
            _result("d", errors_left=[], corrections=[(LINE, {"line": 2}, math.log(0.10))]),
            _result("e", errors_left=[("line", "2", "never corrected")], corrections=[(LINE, {"line": 2}, math.log(0.10))], split="train"),
        ]
        reference = {"a": {"expert": True, "truth": True}, "b": {"expert": False, "truth": True}, "c": {"expert": False, "truth": False},
                     "d": {"expert": False, "truth": False}}
        probe = {"dev:a": 1, "dev:b": 1, "dev:c": 4}
        summary = candidate_sets.summarize(results, reference=reference, probe=probe)
        self.assertEqual(sorted(summary["by_split"]), ["dev:stop", "train:stop"])
        dev = summary["by_split"]["dev:stop"]
        self.assertEqual((dev["states"], dev["states_with_an_error_left"]), (4, 3))
        self.assertEqual((dev["every_object_still_wrong_can_be_named"], dev["no_object_still_wrong_can_be_named"]), (2, 1))
        one = dev["coverage"]["1"]["an object still wrong is in the set"]
        self.assertEqual(one, {"yes": 1, "no": 2, "undecided": 0})
        self.assertEqual(dev["coverage"]["3"]["an object still wrong is in the set"], {"yes": 2, "no": 1, "undecided": 0})
        self.assertEqual(dev["coverage_where_every_object_can_be_named"]["3"]["every object still wrong is in the set"],
                         {"yes": 2, "no": 0, "undecided": 0})
        self.assertEqual(dev["errors_left_by_kind"]["line"], {"errors": 2, "listed": 2, "in the 1 most probable": 1,
                                                               "in the 3 most probable": 2, "in the 5 most probable": 2})
        self.assertEqual(dev["errors_left_by_kind"]["breaker"]["listed"], 0)
        self.assertEqual(dev["errors_left_by_standing"]["corrected, not within tolerance"]["in the 1 most probable"], 0)
        self.assertEqual(sorted(dev["by_own_last_action"]), ["finish resolved", "finish unresolved"])
        self.assertEqual(dev["by_own_last_action"]["finish unresolved"]["states"], 1)
        self.assertEqual([dev["by_reference"][group]["states"] for group in dev["by_reference"]], [1, 1, 2])
        # Against the probe: at one object, a is named by both, b by the evidence only, c by neither.
        self.assertEqual(dev["against_the_evidence_probe"]["states"], 3)
        self.assertEqual(dev["against_the_evidence_probe"]["1"], {"evidence and model": 1, "evidence only": 1, "neither": 1})
        self.assertEqual(dev["against_the_evidence_probe"]["5"], {"evidence and model": 2, "evidence only": 1})
        bands = dev["most_probable_object_by_its_share_of_the_correction_mass"]
        self.assertEqual(bands["0.5 to 0.9: states"], 2)
        self.assertEqual(bands["0.9 and over: states"], 1)
        self.assertAlmostEqual(dev["probability"]["own action"]["median"], 0.8)
        json.dumps(summary, allow_nan=False)
        text = "\n".join(candidate_sets.lines_of(summary))
        self.assertIn("dev:stop: 4 states, 3 with an error left", text)
        self.assertIn("set of 3, all states with an error left: an object still wrong in 2 of 3", text)

    def test_the_probes_places_are_read_from_its_report(self) -> None:
        report = {"stopping_states": {"dev:a": {"candidate_inspect": {"place": 2}, "candidate_shown": {"place": 7}},
                                      "dev:b": {"state_inspect": {"score": 0.4}}}}
        self.assertEqual(candidate_sets.probe_places(report), {"dev:a": 2})
        self.assertEqual(candidate_sets.probe_places(report, "candidate_shown"), {"dev:a": 7})


# ------------------------------------------------------------------ the inputs and the commands


def _stop_record(case: str, *, step: int = 4, errors_left=None, split: str = "dev") -> dict:
    return {"case": case, "split": split, "class": "x", "solved": False, "declared": "resolved", "outcome": "resolved",
            "stop": {"index": step - 1, "estimator_passes": True, "damage": [{"kind": "line", "line": 4}],
                     "errors_left": errors_left or [{"kind": "line", "object": "12", "standing": "never corrected"}]},
            "continuation": {"repairs": True, "undo_needed": True}}


def _scoring_input(case: str, *, step: int = 4, split: str = "dev", point: str = "stop") -> dict:
    return {"contract": "minimal_scoring_inputs_v1", "case": case, "split": split, "class": "x", "step": step, "point": point,
            "observation": _observation(), "actions": [{"role": "own", "action": FINISH_RESOLVED}]}


class InputTests(unittest.TestCase):
    def test_a_stopping_state_is_joined_with_what_the_truth_leaves(self) -> None:
        items = candidate_sets.build_inputs([_scoring_input("a"), _scoring_input("b")],
                                            [_stop_record("a"), _stop_record("b"), {"case": "c", "split": "dev", "solved": True}])
        self.assertEqual([item["case"] for item in items], ["a", "b"])
        first = items[0]
        self.assertEqual(first["contract"], candidate_sets.INPUT_CONTRACT)
        self.assertEqual(first["truth"], {"errors_left": [{"kind": "line", "object": "12", "standing": "never corrected"}], "damaged": 1,
                                          "declared": "resolved", "outcome": "resolved", "estimator_passes": True,
                                          "continuation_repairs": True, "undo_needed": True})
        # The truth sits beside the observation, never in it.
        self.assertNotIn("truth", first["observation"])
        self.assertEqual(first["observation"], _observation())

    def test_inputs_that_do_not_match_are_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "no stopping state"):
            candidate_sets.build_inputs([_scoring_input("a")], [_stop_record("b")])
        with self.assertRaisesRegex(ValueError, "step"):
            candidate_sets.build_inputs([_scoring_input("a", step=5)], [_stop_record("a", step=4)])
        with self.assertRaisesRegex(ValueError, "only stopping states"):
            candidate_sets.build_inputs([_scoring_input("a", point="divergence")], [_stop_record("a")])


class CommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)

    def _write(self, name: str, lines) -> Path:
        path = self.root / name
        path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
        return path

    def _run(self, *arguments: str) -> str:
        printed = io.StringIO()
        with redirect_stdout(printed):
            self.assertEqual(candidate_sets.main([str(argument) for argument in arguments]), 0)
        return printed.getvalue()

    def test_build_score_and_read(self) -> None:
        cases = [f"r0_{index}" for index in range(5)]
        scoring = self._write("scoring_inputs_dev.jsonl", [_scoring_input(case) for case in cases])
        episodes = self._write("episodes_dev.jsonl", [_stop_record(case) for case in cases])
        inputs = self.root / "inputs" / "candidate_inputs_dev.jsonl"
        built = json.loads(self._run("build", "--scoring-inputs", scoring, "--episodes", episodes, "--output", inputs))
        self.assertEqual(built["states"], 5)
        self.assertEqual(built["states_by_errors_left"], {"1": 5})
        self.assertNotIn(b"\r", inputs.read_bytes())
        self.assertEqual(len(candidate_sets.read_inputs([inputs])), 5)
        with self.assertRaisesRegex(ValueError, "appears twice"):
            candidate_sets.read_inputs([inputs, inputs])
        with self.assertRaisesRegex(ValueError, "not an input"):
            candidate_sets.read_inputs([scoring])

        scored: list[str] = []

        def stub_scorer(adapter, output_dir, **settings):
            self.assertEqual((settings["top"], settings["max_passes"]), (3, 50))

            def scorer(item):
                scored.append(item["case"])
                return _score(item, ScoreStateTests.PROBABILITIES, top=settings["top"], max_passes=settings["max_passes"])

            return scorer

        out = self.root / "scores"
        with mock.patch.object(candidate_sets, "model_scorer", stub_scorer):
            shard = self._run("score", "--adapter", "adapter", "--inputs", inputs, "--output-dir", out, "--shard", "0/2",
                              "--top", "3", "--max-passes", "50")
            self.assertEqual(scored, ["r0_0", "r0_2", "r0_4"])
            self.assertIn('"shard": [0, 2]', shard)
            self.assertFalse((out / "summary.json").exists())
            closing = self._run("score", "--adapter", "adapter", "--inputs", inputs, "--output-dir", out, "--top", "3", "--max-passes", "50")
            self.assertEqual(scored[3:], ["r0_1", "r0_3"])
            # A third call finds everything written and loads no model.
            with mock.patch.object(candidate_sets, "model_scorer", side_effect=AssertionError("no model is needed")):
                self._run("score", "--adapter", "adapter", "--inputs", inputs, "--output-dir", out, "--top", "3", "--max-passes", "50")
        self.assertIn("dev:stop: 5 states, 5 with an error left", closing)
        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["contract"], candidate_sets.SUMMARY_CONTRACT)
        self.assertEqual(summary["settings"]["top"], 3)
        # Line 12 (0.001) is the fourth object: outside a set of three, exact all the same.
        dev = summary["by_split"]["dev:stop"]
        self.assertEqual(dev["coverage"]["3"]["an object still wrong is in the set"], {"yes": 0, "no": 5, "undecided": 0})
        self.assertEqual(dev["search_stopped_at_its_limit"], 0)
        state = json.loads((out / "states" / "dev_r0_0_4.json").read_text(encoding="utf-8"))
        self.assertIn("seconds", state)

        reference = self.root / "reference.json"
        reference.write_text(json.dumps({case: {"expert": False, "truth": True} for case in cases}), encoding="utf-8")
        report = self.root / "probe_report.json"
        report.write_text(json.dumps({"stopping_states": {f"dev:{case}": {"candidate_inspect": {"place": 1}} for case in cases}}), encoding="utf-8")
        output = self.root / "read.json"
        self._run("read", "--scores", out, "--reference", reference, "--probe-report", report, "--output", output)
        read = json.loads(output.read_text(encoding="utf-8"))["by_split"]["dev:stop"]
        self.assertEqual(read["by_reference"]["only the truth-built sequence repairs"]["states"], 5)
        self.assertEqual(read["against_the_evidence_probe"]["3"], {"evidence only": 5})


# ------------------------------------------------------------------ the model's side


@unittest.skipIf(torch is None, "torch is not installed")
class PositionLogprobTests(unittest.TestCase):
    def _model(self, *, keeps: bool = True):
        class _Outputs:
            def __init__(self, logits):
                self.logits = logits

        class _TinyLM(torch.nn.Module):
            """Logits that depend on the previous two tokens, so a position's row is its own."""

            vocab = 17

            def __init__(self) -> None:
                super().__init__()
                generator = torch.Generator().manual_seed(5)
                self.table = torch.nn.Parameter(torch.randn(self.vocab, self.vocab, generator=generator))
                self.second = torch.nn.Parameter(torch.randn(self.vocab, self.vocab, generator=generator))
                self.calls: list = []

            @property
            def device(self):
                return self.table.device

            def _logits(self, input_ids):
                before = torch.roll(input_ids, shifts=1, dims=1)
                return self.table[input_ids] + 0.5 * self.second[before]

            if keeps:
                def forward(self, input_ids, attention_mask=None, logits_to_keep=None, use_cache=False, **side):
                    self.calls.append(logits_to_keep)
                    logits = self._logits(input_ids)
                    return _Outputs(logits[:, -int(logits_to_keep):, :] if logits_to_keep else logits)
            else:
                def forward(self, input_ids, attention_mask=None, use_cache=False, **side):
                    self.calls.append(None)
                    return _Outputs(self._logits(input_ids))

        return _TinyLM()

    def _row(self, prompt, completion):
        from psse_env.rl.rows import build_step_row

        return build_step_row(root="score", sample_index=0, step=0, advantage=0.0, prompt_ids=prompt, completion_ids=completion,
                              max_length=64)

    def test_the_table_holds_the_updates_own_completion_logprobs(self) -> None:
        from psse_env.rl.update import completion_logprobs

        for keeps in (True, False):
            model = self._model(keeps=keeps)
            row = self._row([3, 1, 4, 1, 5], [9, 2, 6, 5])
            for temperature in (1.0, 0.7):
                with torch.no_grad():
                    table = candidate_sets.position_logprobs(model, row, temperature=temperature)
                    wanted = completion_logprobs(model, row, temperature=temperature)
                self.assertEqual(tuple(table.shape), (4, model.vocab))
                picked = table.gather(-1, torch.tensor([9, 2, 6, 5]).unsqueeze(-1)).squeeze(-1)
                self.assertTrue(torch.allclose(picked, wanted, atol=1e-6))
                # Each row is a distribution over the vocabulary.
                self.assertTrue(torch.allclose(torch.logsumexp(table, dim=-1), torch.zeros(4), atol=1e-5))
            self.assertEqual(model.calls[0], 5 if keeps else None)

    def test_a_search_on_the_table_finds_the_updates_own_numbers(self) -> None:
        from psse_env.rl.update import completion_logprobs

        model = self._model()
        prompt = [3, 1, 4, 1, 5]
        rng = random.Random(2)
        calls = [[rng.randrange(1, 17) for _ in range(rng.randint(1, 3))] + [END] for _ in range(60)]
        calls = [list(call) for call in sorted({tuple(call) for call in calls})]

        def positions(sequence, wanted):
            with torch.no_grad():
                table = candidate_sets.position_logprobs(model, self._row(prompt, sequence))
            return [[float(table[position, token]) for token in tokens] for position, tokens in enumerate(wanted)]

        search = CallSearch(calls, positions)
        found = search.top(counted=range(len(calls)), top=5, max_passes=1000)
        with torch.no_grad():
            truth = [float(completion_logprobs(model, self._row(prompt, call)).sum()) for call in calls]
        self.assertTrue(found["complete"])
        for index, value in zip(found["order"], sorted(truth, reverse=True)[:5]):
            self.assertAlmostEqual(truth[index], value, places=4)
            self.assertAlmostEqual(search.logprob(index), value, places=4)
        self.assertLess(search.passes, len(calls))

    def test_the_helpers_of_the_update_this_module_reads_exist(self) -> None:
        from psse_env.rl import update

        self.assertTrue(callable(update._row_tensors))
        self.assertTrue(callable(update._model_device))


if __name__ == "__main__":
    unittest.main()
