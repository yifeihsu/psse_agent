"""One-step forks, on freshly generated cases with stand-ins for the model.

The environment and the scoring are the real ones.  The stand-in decides from the observation alone, like the
model: it makes the case's true corrections in order, and may stop early and claim the case.  Its forced calls
write the arguments a tool needs.  What only the model can show (how a forced tool name tokenizes, what
arguments the model writes after it) is first exercised by a cell's smoke.
"""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from psse_env.minimal import forks, truth
from psse_env.minimal.env import MinimalDiagnosisEnv
from psse_env.minimal.episode import ScriptedPolicy, run_episode
from psse_env.minimal.forks import (
    ARGUMENTS_GREEDY,
    ARGUMENTS_SAMPLED,
    STATUS_DRIFTED,
    STATUS_DUPLICATE,
    STATUS_INVALID,
    STATUS_NO_CALL,
    STATUS_RAN,
    ForkPolicy,
    ForkSettings,
    ForkStore,
    better_branches,
    fork_case,
    fork_roots,
    offered_tools,
    run_branch,
    state_digest,
    summarize,
)
from psse_env.minimal.kernel import NumericalKernel
from psse_env.providers.scenario_generator import DEFAULT_COMPOUND_CLASSES, Round0ScenarioGenerator
from psse_env.rl.rollout import scenario_root

GENERATOR_OPTIONS = dict(evidence_profile="scada_only", admission_mode="physical", normalized_residual_threshold=4.0,
                         min_measurement_error_sigma=10.0)
CLASSES = ("control_meter", "cb2_split")
ROOTS: dict[str, dict] = {}
ORDERS: dict[str, list[dict]] = {}
KERNEL: NumericalKernel | None = None
FINISH = {"tool": "finish", "arguments": {"outcome": "resolved"}}


def setUpModule() -> None:
    global KERNEL
    from research.compound_errors.build_suite import envelope_of

    generator = Round0ScenarioGenerator(seed=11, **GENERATOR_OPTIONS)
    KERNEL = NumericalKernel()
    for name in CLASSES:
        spec = next(item for item in DEFAULT_COMPOUND_CLASSES if item.name == name)
        root = envelope_of(generator._build_compound_class(spec, 1)[0])
        order = truth.orders(truth.truth_corrections(root), 1)[0]
        assert run_episode(_env(), root, ScriptedPolicy(order))["score"]["solved"], name
        ROOTS[name], ORDERS[name] = root, order
    assert len(ORDERS["cb2_split"]) == 2 and len(ORDERS["control_meter"]) == 1, ORDERS


def _env(**options) -> MinimalDiagnosisEnv:
    return MinimalDiagnosisEnv(KERNEL, **options)


def _same(left: dict, right: dict) -> bool:
    return forks.signature(left) == forks.signature(right)


class _Model:
    """The model's stand-in.  It reads the observation and nothing else.

    ``act``: the next true correction that is not in effect, then the claim.  ``stop_after`` makes it claim the
    case after that many corrections unless its last action was a look that worked; ``look_first`` makes it look
    at the estimate once before anything else.  ``act_forced``: a call of the named tool, with the next true
    correction's arguments where the tool is that correction's and fixed ones otherwise.
    """

    def __init__(self, order: list[dict], *, stop_after: int | None = None, look_first: bool = False) -> None:
        self.order = copy.deepcopy(order)
        self.stop_after = stop_after
        self.look_first = look_first
        self.calls = 0
        self.forced_calls: list[str] = []
        self.forced_answers: dict[str, object] = {}
        self.last_generation: dict | None = None
        self.last_action_metrics = {"prompt_tokens": 6}

    def _left(self, observation: dict) -> list[dict]:
        made = [{"tool": record["tool"], "arguments": record["arguments"]} for record in observation["modifications"] if record["in_effect"]]
        return [action for action in self.order if not any(_same(action, done) for done in made)]

    def act(self, observation: dict) -> dict:
        self.calls += 1
        history = observation["history"]
        left = self._left(observation)
        looked = bool(history) and history[-1]["action"]["tool"] == "inspect" and history[-1]["result"]["status"] == "ok"
        if self.look_first and not history:
            return {"tool": "inspect", "arguments": {"view": "estimate"}}
        if not left:
            return copy.deepcopy(FINISH)
        made = len(self.order) - len(left)
        if self.stop_after is not None and made >= self.stop_after and not looked:
            return copy.deepcopy(FINISH)
        return copy.deepcopy(left[0])

    def act_forced(self, observation: dict, tool: str):
        self.forced_calls.append(tool)
        if tool in self.forced_answers:
            answer = self.forced_answers[tool]
            if isinstance(answer, Exception):
                raise answer
            return copy.deepcopy(answer)
        self.last_generation = {"completion_token_ids": [7, 8, 9, 10], "forced_prefix_tokens": 2}
        nearest = next((action for action in self._left(observation) if action["tool"] == tool), None)
        if nearest is not None:
            return copy.deepcopy(nearest)
        return {"tool": tool, "arguments": {
            "inspect": {"view": "breakers" if observation["breaker_model"] else "estimate"},
            "correct_topology": {"breaker": "CB_NO_SUCH", "status": "open"},
            "correct_parameters": {"line": 1},
            "correct_measurements": {"meters": [0]},
            "undo": {},
            "finish": {"outcome": "resolved"},
        }[tool]}


class SettingsTests(unittest.TestCase):
    def test_the_settings_are_checked(self) -> None:
        self.assertFalse(ForkSettings.from_mapping(None).enabled)
        settings = ForkSettings.from_mapping({"enabled": True, "weight": 2.0, "min_gain": 0.05, "argument_samples": 2})
        self.assertEqual((settings.weight, settings.min_gain, settings.argument_samples), (2.0, 0.05, 2))
        for bad in ({"enabled": 1}, {"enabled": True, "weight": 0.0}, {"weight": -1.0}, {"min_gain": float("inf")},
                    {"argument_samples": -1}, {"argument_samples": 1.5}, {"states": "last"}):
            with self.assertRaises(ValueError, msg=str(bad)):
                ForkSettings.from_mapping(bad)

    def test_an_observation_offers_the_six_tools(self) -> None:
        observation = _env().reset(ROOTS["control_meter"])
        self.assertEqual(offered_tools(observation),
                         ["inspect", "correct_topology", "correct_parameters", "correct_measurements", "undo", "finish"])
        self.assertEqual(state_digest(observation), state_digest(copy.deepcopy(observation)))
        self.assertNotEqual(state_digest(observation), state_digest(_env().reset(ROOTS["cb2_split"])))


class ForkCaseTests(unittest.TestCase):
    def test_a_claim_without_a_correction_is_beaten_by_the_forced_correction(self) -> None:
        root, order = ROOTS["control_meter"], ORDERS["control_meter"]
        model = _Model(order, stop_after=0)
        payload = fork_case(root, env_factory=_env, policy=model)
        record = payload["record"]
        self.assertEqual((record["own"]["solved"], record["own"]["declared"], record["states"]), (False, "resolved", 1))
        # Every other tool is forced once at the one state: five branches, each a whole episode.
        self.assertEqual([branch["tool"] for branch in record["branches"]],
                         ["inspect", "correct_topology", "correct_parameters", "correct_measurements", "undo"])
        self.assertEqual({branch["status"] for branch in record["branches"]}, {STATUS_RAN})
        self.assertEqual({branch["arguments"] for branch in record["branches"]}, {ARGUMENTS_GREEDY})
        by_tool = {branch["tool"]: branch for branch in record["branches"]}
        repaired = by_tool["correct_measurements"]
        self.assertTrue(repaired["solved"] and repaired["executed"])
        self.assertTrue(_same(repaired["action"], order[0]))
        self.assertEqual(repaired["continuation"], [FINISH])
        # A branch's return is the whole episode's: the repair and the claim, less two actions.
        self.assertAlmostEqual(repaired["return"], 1.0 - 0.02)
        self.assertAlmostEqual(record["own"]["return"], -0.5 - 0.01)
        # The environment refused the breaker that does not exist and the undo with nothing to undo.
        self.assertEqual((by_tool["correct_topology"]["executed"], by_tool["correct_topology"]["error_code"]), (False, "unknown_object"))
        self.assertEqual((by_tool["undo"]["executed"], by_tool["undo"]["error_code"]), (False, "no_change"))
        # After a forced look the stand-in corrects: that branch repairs too, in one action more, so the direct
        # correction is the state's better branch.
        self.assertEqual([action["tool"] for action in by_tool["inspect"]["continuation"]], ["correct_measurements", "finish"])
        self.assertAlmostEqual(by_tool["inspect"]["return"], 1.0 - 0.03)
        # A fit of a healthy line repairs nothing: the claim after it is as false as the own episode's.
        self.assertFalse(by_tool["correct_parameters"]["solved"])
        self.assertLess(by_tool["correct_parameters"]["return"], record["own"]["return"])
        self.assertEqual(len(record["better"]), 1)
        better = record["better"][0]
        self.assertEqual((better["state"], better["tool"], better["solved"]), (0, "correct_measurements", True))
        self.assertAlmostEqual(better["gain"], 1.49)
        # The observation of that state is kept: it is what the training row is rendered from.
        self.assertEqual(list(payload["observations"]), ["0"])
        self.assertEqual(state_digest(payload["observations"]["0"]), state_digest(_env().reset(root)))
        json.dumps(payload, allow_nan=False)

    def test_the_second_breaker_is_reached_by_one_forced_look(self) -> None:
        root, order = ROOTS["cb2_split"], ORDERS["cb2_split"]
        model = _Model(order, stop_after=1)  # it corrects one breaker and claims the case
        record = fork_case(root, env_factory=_env, policy=model)["record"]
        self.assertEqual((record["own"]["solved"], record["states"], [step["action"]["tool"] for step in record["own"]["steps"]]),
                         (False, 2, ["correct_topology", "finish"]))
        at_the_claim = {branch["tool"]: branch for branch in record["branches"] if branch["state"] == 1}
        look = at_the_claim["inspect"]
        self.assertEqual(look["action"], {"tool": "inspect", "arguments": {"view": "breakers"}})
        self.assertEqual([action["tool"] for action in look["continuation"]], ["correct_topology", "finish"])
        self.assertTrue(look["solved"])
        self.assertAlmostEqual(look["return"], 1.0 - 0.04)
        # The forced correction of the second breaker also repairs, in one action less: it is the better branch.
        direct = at_the_claim["correct_topology"]
        self.assertTrue(direct["solved"] and _same(direct["action"], order[1]))
        chosen = {item["state"]: item for item in record["better"]}
        self.assertEqual(chosen[1]["tool"], "correct_topology")
        self.assertAlmostEqual(chosen[1]["gain"], (1.0 - 0.03) - record["own"]["return"])
        summary = summarize([record])
        self.assertEqual((summary["totals"]["own_episode_unrepaired"], summary["totals"]["unrepaired_cases_one_forced_step_repairs"]), (1, 1))
        self.assertEqual(summary["totals"]["unrepaired_cases_one_forced_step_repairs_in_the_policys_own_decoding"], 1)
        self.assertEqual(summary["by_class"]["cb2_split"]["unrepaired_cases_one_forced_step_repairs"], 1)

    def test_an_action_saved_is_a_gain_of_one_action_cost(self) -> None:
        root, order = ROOTS["control_meter"], ORDERS["control_meter"]
        model = _Model(order, look_first=True)  # look, correct, claim: repaired in three actions
        record = fork_case(root, env_factory=_env, policy=model)["record"]
        self.assertTrue(record["own"]["solved"])
        self.assertAlmostEqual(record["own"]["return"], 1.0 - 0.03)
        better = record["better"]
        self.assertEqual([(item["state"], item["tool"]) for item in better], [(0, "correct_measurements")])
        self.assertAlmostEqual(better[0]["gain"], 0.01)
        # A threshold above one action's cost leaves nothing to teach on a case already repaired.
        self.assertEqual(better_branches(record, min_gain=0.05), [])
        kept = fork_case(root, env_factory=_env, policy=model, settings=ForkSettings(enabled=True, min_gain=0.05))
        self.assertEqual((kept["record"]["better"], kept["observations"]), ([], {}))
        self.assertEqual(summarize([record])["gain"]["at_most_one_action_cost"], 1)

    def test_a_refused_call_is_never_the_better_branch(self) -> None:
        record = {"own": {"return": -0.5}, "branches": [
            {"state": 0, "tool": "undo", "status": STATUS_RAN, "executed": False, "return": 0.9, "solved": True, "actions": 2, "action": {}},
            {"state": 0, "tool": "inspect", "status": STATUS_DRIFTED},
            {"state": 0, "tool": "finish", "status": STATUS_RAN, "executed": True, "return": 0.2, "solved": False, "actions": 1, "action": {}},
            {"state": 0, "tool": "correct_parameters", "status": STATUS_RAN, "executed": True, "return": 0.2, "solved": False, "actions": 1, "action": {}},
        ]}
        better = better_branches(record)
        self.assertEqual([(item["tool"], item["branch"]) for item in better], [("finish", 2)])  # the earlier of two equal branches

    def test_sampled_calls_try_other_arguments_and_a_repeat_is_not_run_again(self) -> None:
        root, order = ROOTS["control_meter"], ORDERS["control_meter"]
        model = _Model(order, stop_after=0)
        sampler = _Model(order, stop_after=0)
        sampler.forced_answers = {"finish": {"tool": "finish", "arguments": {"outcome": "unresolved"}}}
        record = fork_case(root, env_factory=_env, policy=model, argument_policy=sampler,
                           settings=ForkSettings(enabled=True, argument_samples=1))["record"]
        sampled = {branch["tool"]: branch for branch in record["branches"] if branch["arguments"] == ARGUMENTS_SAMPLED}
        self.assertEqual(sorted(sampled), sorted(offered_tools(_env().reset(root))))  # every tool, the model's own included
        # The sampler wrote the calls the model's own decoding had already written: nothing new to run.
        for tool in ("inspect", "correct_topology", "correct_parameters", "correct_measurements", "undo"):
            self.assertEqual(sampled[tool]["status"], STATUS_DUPLICATE, tool)
            self.assertNotIn("return", sampled[tool])
        # The model's own tool with another argument is a new call: an honest declaration beats the false claim.
        self.assertEqual((sampled["finish"]["status"], sampled["finish"]["declared"], sampled["finish"]["sample"]), (STATUS_RAN, "unresolved", 0))
        self.assertAlmostEqual(sampled["finish"]["return"] - record["own"]["return"], 0.5)
        self.assertEqual(sampler.calls, 0)  # the sampler writes forced calls only; the model continues every branch
        totals = summarize([record])["totals"]
        self.assertEqual((totals["branches"], totals["branches_duplicate"], totals["branches_ran"]), (11, 5, 6))

    def test_what_is_not_a_call_of_the_forced_tool_is_marked_and_not_used(self) -> None:
        root, order = ROOTS["control_meter"], ORDERS["control_meter"]
        model = _Model(order, stop_after=0)

        class PromptLimitError(RuntimeError):
            pass

        model.forced_answers = {"inspect": "I would look at the breakers.", "undo": {"tool": "finish", "arguments": {"outcome": "resolved"}},
                                "correct_measurements": PromptLimitError("the prompt has too many tokens")}
        record = fork_case(root, env_factory=_env, policy=model)["record"]
        by_tool = {branch["tool"]: branch for branch in record["branches"]}
        self.assertEqual((by_tool["inspect"]["status"], by_tool["inspect"]["action"]), (STATUS_INVALID, "I would look at the breakers."))
        self.assertEqual(by_tool["undo"]["status"], STATUS_INVALID)  # a valid call, of another tool
        self.assertEqual((by_tool["correct_measurements"]["status"], by_tool["correct_measurements"]["action"]), (STATUS_NO_CALL, None))
        self.assertEqual(record["better"], [])
        totals = summarize([record])["totals"]
        self.assertEqual((totals["branches_invalid_call"], totals["branches_no_call"]), (2, 1))

    def test_a_replay_that_shows_another_state_is_marked_drifted(self) -> None:
        root, order = ROOTS["cb2_split"], ORDERS["cb2_split"]
        model = _Model(order)
        env = _env()
        first = state_digest(env.reset(root))
        second = state_digest(env.step(copy.deepcopy(order[0]))[0])
        good = run_branch(root, env=_env(), policy=model, prefix=[order[0]], digests=[first, second], tool="inspect")
        self.assertEqual((good["status"], good["state"]), (STATUS_RAN, 1))
        # The same branch against the digest of another second state.
        drifted = run_branch(root, env=_env(), policy=model, prefix=[order[0]], digests=[first, first], tool="inspect")
        self.assertEqual((drifted["status"], drifted["drifted_at"], drifted["action"]), (STATUS_DRIFTED, 1, None))
        self.assertNotIn("return", drifted)
        with self.assertRaisesRegex(ValueError, "one more than its prefix"):
            ForkPolicy(model, prefix=[order[0]], digests=[first], tool="inspect")

    def test_the_forced_prefix_is_checked_against_the_calls_own_rendering(self) -> None:
        root, order = ROOTS["control_meter"], ORDERS["control_meter"]
        asked: list[tuple[str, list[int]]] = []

        def canonical(observation, action, forced_ids):
            asked.append((action["tool"], list(forced_ids)))
            if action["tool"] == "undo":
                raise ValueError("the call does not render")
            return action["tool"] != "inspect"

        record = fork_case(root, env_factory=_env, policy=_Model(order, stop_after=0), canonical=canonical)["record"]
        self.assertEqual(len(asked), 5)
        self.assertEqual({ids for _tool, ids in map(lambda item: (item[0], tuple(item[1])), asked)}, {(7, 8)})  # the forced tokens alone
        by_tool = {branch["tool"]: branch for branch in record["branches"]}
        self.assertEqual((by_tool["inspect"]["prefix_canonical"], by_tool["undo"]["prefix_canonical"],
                          by_tool["correct_measurements"]["prefix_canonical"]), (False, False, True))
        self.assertEqual(by_tool["inspect"]["forced_prefix_tokens"], 2)
        totals = summarize([record])["totals"]
        self.assertEqual((totals["forced_prefixes_checked"], totals["forced_prefixes_canonical"]), (5, 3))


class ForcedCallTests(unittest.TestCase):
    """The model policy's forced call: the tool's name is given, the model writes the rest."""

    def test_the_policy_starts_the_answer_with_the_tools_name_and_checks_what_follows(self) -> None:
        from psse_env.dagger.release_factories import _ModelBundle
        from psse_env.minimal.policy import MinimalModelPolicy, forced_call_prefix

        self.assertEqual(forced_call_prefix("inspect"), "<|tool_call>call:inspect{")
        bundle = _ModelBundle(model=None, processor=object(), model_id="google/gemma-4-12B-it", model_revision="f" * 40,
                              adapter_snapshot_path="adapter")
        policy = MinimalModelPolicy(bundle, architecture="gemma4_unified")
        observation = _env().reset(ROOTS["cb2_split"])
        asked: list[tuple[dict, str | None]] = []
        answers = iter(['<|tool_call>call:inspect{view:<|"|>breakers<|"|>}<tool_call|>',
                        '<|tool_call>call:correct_parameters{line:<|"|>seven<|"|>}<tool_call|>'])

        def generate_text(seen, *, forced_prefix=None):
            asked.append((seen, forced_prefix))
            return next(answers)

        policy.generate_text = generate_text  # type: ignore[method-assign]
        self.assertEqual(policy.act_forced(observation, "inspect"), {"tool": "inspect", "arguments": {"view": "breakers"}})
        self.assertEqual(asked[0][1], "<|tool_call>call:inspect{")
        self.assertEqual(asked[0][0]["versions"], observation["versions"])  # the observation, marked as a whole history
        # Arguments the tool's schema refuses come back as the model's words, not as a call.
        self.assertIsInstance(policy.act_forced(observation, "correct_parameters"), str)
        with self.assertRaisesRegex(ValueError, "not a tool of this observation"):
            policy.act_forced(observation, "commit_state")
        self.assertEqual(len(asked), 2)


class StoreTests(unittest.TestCase):
    def test_cases_are_forked_once_and_a_later_start_loads_no_model(self) -> None:
        roots = [ROOTS[name] for name in CLASSES]
        built: list[int] = []
        seeds: list[int] = []
        progress: list[dict] = []

        def policy_factory():
            built.append(1)
            return _Dispatch()

        def sampler_factory(seed):
            seeds.append(seed)
            return _Dispatch()

        with tempfile.TemporaryDirectory() as folder:
            store = ForkStore(Path(folder) / "forks")
            self.assertEqual(list(store.iter()), [])
            ran = fork_roots(roots[:1], store=store, env_factory=_env, policy_factory=policy_factory, progress=progress.append)
            self.assertEqual((ran, len(built), seeds), (1, 1, []))
            self.assertTrue(store.has(scenario_root(roots[0])) and not store.has(scenario_root(roots[1])))
            ran = fork_roots(roots, store=store, env_factory=_env, policy_factory=policy_factory, base_seed=5,
                             settings=ForkSettings(enabled=True, argument_samples=1), argument_policy_factory=sampler_factory)
            self.assertEqual((ran, len(built), len(seeds)), (1, 2, 1))  # the second case only, with its own seed
            self.assertEqual(fork_roots(roots, store=store, env_factory=_env, policy_factory=policy_factory), 0)
            self.assertEqual(len(built), 2)
            stored = dict(store.iter())
            self.assertEqual(set(stored), {scenario_root(root) for root in roots})
            payload = store.read(scenario_root(roots[0]))
            self.assertEqual((payload["case"], payload["record"]["scenario_id"]), (scenario_root(roots[0]), scenario_root(roots[0])))
            # The stored observation renders to the prompt the model read: a training row is rendered from the file.
            self.assertEqual(state_digest(payload["observations"]["0"]), state_digest(_env().reset(roots[0])))
            self.assertEqual(progress[0]["fork"], scenario_root(roots[0]))
            self.assertEqual(progress[0]["better"][0][:2], (0, "correct_measurements"))
            self.assertEqual(list(Path(folder, "forks", "cases").glob(".*")), [])  # nothing half-written is left


class _Dispatch:
    """One stand-in for every case: the case is told by what its first state shows."""

    def __init__(self) -> None:
        self.models = {state_digest(_env().reset(ROOTS[name])): _Model(ORDERS[name], stop_after=0) for name in CLASSES}
        self.current: _Model | None = None
        self.last_action_metrics: dict = {}

    def _model(self, observation: dict) -> _Model:
        if not observation["history"]:
            self.current = self.models[state_digest(observation)]
        assert self.current is not None
        return self.current

    def act(self, observation: dict) -> dict:
        return self._model(observation).act(observation)

    def act_forced(self, observation: dict, tool: str):
        return self._model(observation).act_forced(observation, tool)


if __name__ == "__main__":
    unittest.main()
