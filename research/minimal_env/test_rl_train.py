"""The RL driver of the minimal environment, on freshly generated cases with stand-ins for the model.

The environment, the scoring, the store, the group advantages, the row builder, the update and the integrity
checks are the real ones.  What stands in: the sampling and the greedy policy (scripted, leaving the generation
record the research model leaves), the tokenizer (a digest of the whole rendered prompt), and the trained model
(the tiny bigram model of the update's own tests).  What needs the model itself is first exercised by a cell's
smoke.
"""
from __future__ import annotations

import contextlib
import copy
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from psse_env.minimal import observation as view
from psse_env.minimal import truth
from psse_env.minimal.env import MinimalDiagnosisEnv
from psse_env.minimal.episode import ScriptedPolicy, run_episode
from psse_env.minimal.kernel import NumericalKernel
from psse_env.minimal.policy import MinimalModelPolicy
from psse_env.providers.scenario_generator import DEFAULT_COMPOUND_CLASSES, Round0ScenarioGenerator
from psse_env.rl.integrity import MARKER_FILE, IntegrityError
from psse_env.rl.prompt_digest import prompt_token_digest
from psse_env.rl.rollout import TrajectoryStore, sample_seed
from psse_env.rl.train import OPTIMIZER_FILE, Manifest, iteration_dir
from research.minimal_env import evaluate, rl_train, truth_sequences
from research.minimal_env.rl_train import DRIVER_CONTRACT, EXAMPLE_SAMPLE, MinimalRLConfig

try:
    import torch

    from psse_env.rl.test_update import _Preempted, _TinyLM
except Exception:  # pragma: no cover - torch is optional for the pure-Python tests
    torch = None

GENERATOR_OPTIONS = dict(evidence_profile="scada_only", admission_mode="physical", normalized_residual_threshold=4.0,
                         min_measurement_error_sigma=10.0)
CLASSES = ("control_meter", "meter_on_rx")
ROOTS: list[dict] = []
#: Each case's true corrections, by what its first observation shows (the stand-in policies see nothing else).
REPAIRS: dict[tuple[int, float], list[dict]] = {}
#: Whether the scripted repair of each case ends repaired (a line fit can be beyond the tool).
REPAIRED: list[bool] = []
KERNEL: NumericalKernel | None = None


def _key(observation: dict) -> tuple[int, float]:
    first = observation["versions"][0]["estimate"]
    return len(first["meters"]), round(first["chi_square"], 3)


def setUpModule() -> None:
    global KERNEL
    from research.compound_errors.build_suite import envelope_of

    generator = Round0ScenarioGenerator(seed=11, **GENERATOR_OPTIONS)
    KERNEL = NumericalKernel()
    for name in CLASSES:
        spec = next(item for item in DEFAULT_COMPOUND_CLASSES if item.name == name)
        root = envelope_of(generator._build_compound_class(spec, 1)[0])
        order = truth.orders(truth.truth_corrections(root), 1)[0]
        env = MinimalDiagnosisEnv(KERNEL)
        REPAIRS[_key(env.reset(root))] = order
        REPAIRED.append(bool(run_episode(env, root, ScriptedPolicy(order))["score"]["solved"]))
        ROOTS.append(root)
    # The tests below read a repair against a false claim; the meter case is repaired by its true corrections.
    assert REPAIRED[0], REPAIRED


def _tokens(text: str) -> tuple[list[int], int]:
    """Six tokens that depend on every character of the prompt, inside the tiny model's vocabulary."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [1 + byte % 10 for byte in digest[:6]], 6


def _answer_tokens(action: dict) -> list[int]:
    """The two tokens the stand-ins give an action, as the model's answer and as a rendered target alike."""
    return [1 + sum(map(ord, json.dumps(action, sort_keys=True))) % 10, 11]


class _Scripted:
    """The model's stand-in: repairs the case it is shown by its true corrections, or claims it without acting.

    It leaves the generation record of the research policy: the observation it was given, the prompt's length
    and digest, the tokens of its answer and their probabilities.
    """

    def __init__(self, repair: bool) -> None:
        self.repair = bool(repair)
        self.queue: list[dict] = []
        self.calls = 0
        self.last_generation: dict | None = None
        self.last_action_metrics = {"prompt_tokens": 6, "original_prompt_tokens": 6, "generated_tokens": 2}
        self.research_policy_identity = {"adapter": "stand-in"}

    def act(self, observation: dict) -> dict:
        if not observation["history"]:
            self.queue = copy.deepcopy(REPAIRS[_key(observation)]) if self.repair else []
        action = self.queue.pop(0) if self.queue else {"tool": "finish", "arguments": {"outcome": "resolved"}}
        prompt, _ = _tokens(view.user_text(observation))
        wire = json.dumps(action, sort_keys=True)
        self.last_generation = {
            "model_observation": copy.deepcopy(observation), "prompt_tokens": len(prompt), "original_prompt_tokens": len(prompt),
            "completion_token_ids": _answer_tokens(action), "text": wire, "call_index": self.calls,
            "completion_token_probabilities": [0.5, 0.9], "prompt_token_digest": prompt_token_digest(prompt),
        }
        self.calls += 1
        return action


class _Stop(RuntimeError):
    pass


@contextlib.contextmanager
def _stand_ins(holder: dict):
    """The driver with the model's side replaced for the length of the block.

    ``holder`` steers the stand-ins: ``never_repair`` (every sample claims the case), ``fail_after`` (cut the
    trained model's forward passes, as a preemption cuts an update), ``other_prompt`` (the update renders another
    prompt than the sampled one), ``stop_at_load``; ``loads`` collects what the loader was asked for, and
    ``sampled_from`` and ``seeds`` the adapter and the seeds the sampling policies were built with.
    """

    def sampling(config, adapter):
        holder.setdefault("sampled_from", []).append(str(adapter))
        count = iter(range(10**6))

        def factory(seed):
            holder.setdefault("seeds", []).append(int(seed))
            return _Scripted(repair=not holder.get("never_repair") and next(count) % 2 == 0)

        return factory

    def greedy(config, adapter):
        # The starting adapter claims every case without acting; any later adapter repairs.
        return lambda: _Scripted(repair=Path(adapter) != Path(config.initial_adapter))

    def load(config, adapter, *, reference_adapter=None):
        holder.setdefault("loads", []).append({"adapter": str(adapter), "reference": reference_adapter})
        if holder.get("stop_at_load"):
            raise _Stop()
        model = _TinyLM(seed=3, fail_after=holder.get("fail_after"))
        weights = Path(adapter) / "weights.pt"
        if weights.is_file():
            model.load_state_dict(torch.load(weights))
        return model, None, ()

    def prompts(config, adapter, processor):
        return (lambda observation: view.user_text(observation) + ("!" if holder.get("other_prompt") else "")), _tokens

    def save(model, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), Path(directory) / "weights.pt")

    def targets(config, adapter, processor):
        def render_target(observation, action):
            holder.setdefault("targets", []).append((view.user_text(observation), copy.deepcopy(dict(action))))
            if holder.get("unrenderable"):
                raise ValueError("the action does not render")
            return _answer_tokens(action)

        return render_target

    with mock.patch.multiple(rl_train, sampling_policy_factory=sampling, greedy_policy_factory=greedy,
                             load_trainable_model=load, prompt_functions=prompts, save_trained_adapter=save,
                             target_function=targets,
                             environment=lambda config: MinimalDiagnosisEnv(
                                 KERNEL, max_actions=int(config.max_actions), table_versions=int(config.table_versions) or None)):
        yield rl_train


def _config(folder: str, **overrides) -> MinimalRLConfig:
    root = Path(folder)
    suite = root / "suite.json"
    if not suite.is_file():
        suite.write_text(json.dumps(ROOTS), encoding="utf-8")
    adapter = root / "adapter0"
    if torch is not None and not adapter.is_dir():
        adapter.mkdir()
        torch.save(_TinyLM(seed=3).state_dict(), adapter / "weights.pt")
    payload = {
        "output_dir": str(root / "run"), "training_suite": str(suite), "development_suite": str(suite),
        "initial_adapter": str(adapter), "samples_per_root": 2,
        "update": {"learning_rate": 0.01, "carry_optimizer": True, "chunk_by": "group"},
        "rows": {"require_complete_groups": True, "reject_truncated_prompts": True, "require_prompt_digest": True,
                 "require_sampling_records": True},
        "update_checkpoint_rows": 1,
    }
    payload.update(overrides)
    config = MinimalRLConfig(**payload)
    config.validate()
    return config


def _run(config: MinimalRLConfig, phase: str, **options):
    with contextlib.redirect_stdout(io.StringIO()):
        return rl_train.run_phase(config, phase, **options)


def _truth_order(root: dict) -> list[dict]:
    """A case's truth-built sequence: its true corrections, then the declaration."""
    order = copy.deepcopy(REPAIRS[_key(MinimalDiagnosisEnv(KERNEL).reset(root))])
    return order + [{"tool": "finish", "arguments": {"outcome": "resolved"}}]


def _focus(folder: str, *, repaired: tuple[bool, ...] = (False, False), sequences_for: tuple[int, ...] = (0, 1),
           orders: dict[int, list[dict]] | None = None, **settings) -> dict:
    """The ``focus`` block of a config, with the two files it names written under ``folder``.

    The file of truth-built sequences holds the cases of ``sequences_for`` (``orders`` replaces a case's sequence);
    the evaluation is the starting adapter's greedy pass, which repaired the cases flagged in ``repaired``.
    """
    root = Path(folder)
    plain = _config(folder)
    records = [{"demonstration": "success", "solved": True, "scenario_id": ROOTS[index]["execution"]["scenario_id"],
                "compound_class": CLASSES[index],
                "steps": [{"step": number, "action": action, "result": {"status": "ok"}}
                          for number, action in enumerate((orders or {}).get(index) or _truth_order(ROOTS[index]), start=1)]}
               for index in sequences_for]
    truth_sequences.dump(truth_sequences.build(records), root / "truth_sequences.json")
    limits = rl_train.apply_prompt_limits(plain)
    passed = root / "train_pass"
    passed.mkdir(exist_ok=True)
    (passed / "evaluation.json").write_text(json.dumps({
        "adapter": plain.initial_adapter,
        "settings": {"model_choice": "12b", "load_in_4bit": False, "decoding": "greedy", "max_actions": 16, "table_versions": 2,
                     "max_input_tokens": str(limits["max_input_tokens"]), "max_new_tokens": str(limits["max_new_tokens"])},
        "episodes": [{"scenario_id": root_["execution"]["scenario_id"], "solved": bool(flag)} for root_, flag in zip(ROOTS, repaired)],
    }), encoding="utf-8")
    return {"truth_sequences": str(root / "truth_sequences.json"), "evaluation": str(passed), "failed_per_iteration": 2, **settings}


#: The row checks of the tests' configs, with the sampled rows of uniform groups kept for the KL term.
FOCUS_ROWS = {"require_complete_groups": True, "reject_truncated_prompts": True, "require_prompt_digest": True,
              "require_sampling_records": True, "preservation": "one"}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class ConfigTests(unittest.TestCase):
    def test_the_config_is_checked_and_names_its_model(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            config = _config(folder)
            self.assertEqual((config.base_model, config.architecture), ("google/gemma-4-12B-it", "gemma4_unified"))
            self.assertEqual(len(config.base_revision), 40)
            self.assertEqual((config.advantage_scale, config.load_in_4bit, config.stop_on_integrity_failure), ("none", False, True))
            for bad in (dict(samples_per_root=1), dict(update={"learning_rate": -1.0}), dict(update={"temperature": 2.0}),
                        dict(reward={"lifecycle_credit": 0.25}), dict(rows={"preservation": "some"}),
                        dict(advantage_scale="batch"), dict(model_choice="none"), dict(table_versions=-1),
                        dict(max_input_tokens=40000), dict(development_suite=None, baseline_evaluation=True),
                        dict(root_sampling={"mode": "stratified", "key": "compound_class", "counts": {"a": 1}})):
                with self.assertRaises((ValueError, KeyError), msg=str(bad)):
                    _config(folder, **bad)
            settings = _config(folder, temperature=1.3, update={"beta_kl": 0.04}).update_settings(4)
            self.assertEqual((settings.temperature, settings.beta_kl, settings.seed), (1.3, 0.04, config.seed + 4))
            # A clipped update needs every row's sampling record, whatever the rows block says.
            self.assertTrue(_config(folder, update={"clip_ratio": 0.2}, rows={}).row_settings(0).require_sampling_records)
            path = Path(folder) / "rl.json"
            path.write_text(json.dumps(config.as_dict()), encoding="utf-8")
            self.assertEqual(MinimalRLConfig.load(path).samples_per_root, 2)
            path.write_text(json.dumps({**config.as_dict(), "lifecycle_choice": 1}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown config fields"):
                MinimalRLConfig.load(path)

    def test_the_example_config_loads_and_sets_what_the_earlier_arms_measured(self) -> None:
        config = MinimalRLConfig.load(Path(rl_train.__file__).with_name("rl_config_example.json"))
        settings = config.update_settings(0)
        self.assertEqual((settings.optimizer_steps, settings.carry_optimizer, settings.clip_ratio, settings.loss_normalization,
                          settings.chunk_by), (1, True, None, "dapo", "group"))
        self.assertGreater(settings.beta_kl, 0.0)
        self.assertEqual((config.advantage_scale, config.load_in_4bit, config.baseline_evaluation), ("none", False, True))
        rows = config.row_settings(0)
        self.assertTrue(rows.require_complete_groups and rows.require_prompt_digest and rows.reject_truncated_prompts
                        and rows.require_sampling_records)
        self.assertEqual(sum(config.root_sampling["counts"].values()), config.roots_per_iteration)
        self.assertTrue(Path(rl_train.REPO_ROOT, config.reference).is_file())

    def test_the_cases_of_an_iteration_are_drawn_by_class_or_in_slices(self) -> None:
        def case(index: int, name: str) -> dict:
            return {"execution": {"scenario_id": f"case{index:02d}"}, "audit": {"compound": {"class": name}}}

        cases = [case(index, "a" if index < 6 else "b") for index in range(10)]
        with tempfile.TemporaryDirectory() as folder:
            sliced = _config(folder, roots_per_iteration=4)
            drawn = [[item["execution"]["scenario_id"] for item in rl_train.select_iteration_roots(sliced, cases, iteration)]
                     for iteration in range(3)]
            self.assertEqual(len({name for names in drawn[:2] for name in names}), 8)  # disjoint until the suite is used up
            self.assertEqual(len(drawn[2]), 4)
            by_class = _config(folder, roots_per_iteration=4,
                               root_sampling={"mode": "stratified", "key": "compound_class", "counts": {"a": 3, "b": 1}})
            for iteration in range(3):
                names = [item["audit"]["compound"]["class"] for item in rl_train.select_iteration_roots(by_class, cases, iteration)]
                self.assertEqual((names.count("a"), names.count("b")), (3, 1))

    def test_the_check_reads_what_a_run_would_start_from_and_runs_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            train, development, reference, adapter = root / "train.json", root / "dev.json", root / "reference.json", root / "lora"
            train.write_text(json.dumps(ROOTS[:1]), encoding="utf-8")
            development.write_text(json.dumps(ROOTS[1:]), encoding="utf-8")
            reference.write_text(json.dumps({ROOTS[1]["execution"]["scenario_id"]: {"expert": True, "truth": True}}), encoding="utf-8")
            adapter.mkdir()
            (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
            (adapter / "adapter_model.safetensors").write_bytes(b"0")
            settings = dict(training_suite=str(train), development_suite=str(development), reference=str(reference),
                            initial_adapter=str(adapter), update={"beta_kl": 0.04})
            found = rl_train.check(_config(folder, **settings))
            self.assertEqual((found["ok"], found["problems"]), (True, []))
            self.assertEqual((found["training_cases"], found["cases_drawn_for_the_first_iteration"], found["episodes_per_iteration"],
                              found["development_cases"]), (1, 1, 2, 1))
            self.assertEqual((found["kl_reference"], found["run_is_at"]), (str(adapter), {"iteration": 0, "phase": "rollout"}))
            self.assertFalse((root / "run").exists())  # nothing was written
            # What a run would trip over: no weights, a case in both suites, a case the reference does not know.
            (adapter / "adapter_model.safetensors").unlink()
            reference.write_text("{}", encoding="utf-8")
            problems = rl_train.check(_config(folder, **{**settings, "development_suite": str(train)}))["problems"]
            self.assertEqual(len(problems), 3)
            self.assertRegex(" ".join(problems), "no adapter at .*1 cases are in the training suite.*1 development cases are not")
            path = root / "rl.json"
            path.write_text(json.dumps(_config(folder, **settings).as_dict()), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                self.assertEqual(rl_train.main(["--config", str(path), "--check"]), 2)
            self.assertFalse(json.loads(printed.getvalue())["ok"])
            # A class count that does not fit the suite is refused when the cases are drawn.
            by_class = _config(folder, **settings, roots_per_iteration=1,
                               root_sampling={"mode": "stratified", "key": "compound_class", "counts": {"no_such_class": 1}})
            with self.assertRaisesRegex(ValueError, "must name exactly the suite's strata"):
                rl_train.check(by_class)

    def test_a_run_folder_of_the_old_driver_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            config = _config(folder)
            self.assertEqual(rl_train.load_manifest(config).contract, DRIVER_CONTRACT)
            Path(config.output_dir).mkdir(parents=True)
            Manifest(current_adapter="x").save(Path(config.output_dir) / "manifest.json")
            with self.assertRaisesRegex(RuntimeError, "rl_iteration_driver"):
                rl_train.load_manifest(config)


@unittest.skipIf(torch is None, "torch is not installed")
class PromptTests(unittest.TestCase):
    """The update's prompt is the policy's: the minimal environment's messages and six tools, from the stored step."""

    def test_a_stored_step_renders_to_the_messages_and_tools_the_policy_was_given(self) -> None:
        from psse_env.minimal.policy import MinimalModelPolicy

        suffix = "<|turn>model\n<|channel>thought\n<channel|>"

        class Processor:
            def __init__(self) -> None:
                self.seen: list[tuple[list, dict]] = []

            def apply_chat_template(self, messages, **options):
                self.seen.append((copy.deepcopy(messages), options))
                return "\n".join(message["content"] for message in messages) + suffix

            def __call__(self, text=None, return_tensors=None):
                return {"input_ids": torch.tensor([[1 + ord(character) % 7 for character in text[-40:]]])}

        env = MinimalDiagnosisEnv(KERNEL)
        observation = env.reset(ROOTS[1])
        observation, _ = env.step(REPAIRS[_key(observation)][0])  # a later step: history, a modification, two versions
        stored = json.loads(json.dumps(MinimalModelPolicy._whole(observation), allow_nan=False))  # as the store keeps it
        processor = Processor()
        with tempfile.TemporaryDirectory() as folder:
            render, prompt_ids = rl_train.prompt_functions(_config(folder), "adapter", processor)
        text = render(stored)
        messages, options = processor.seen[-1]
        self.assertEqual(messages, view.messages(observation))
        self.assertEqual(options["tools"], view.tools(observation))
        self.assertEqual([tool["function"]["name"] for tool in options["tools"]],
                         ["inspect", "correct_topology", "correct_parameters", "correct_measurements", "undo", "finish"])
        self.assertEqual((options["enable_thinking"], options["add_generation_prompt"]), (False, True))
        self.assertTrue(text.endswith(suffix))
        self.assertIn("== Modifications ==\n1. step 1,", text)
        ids, count = prompt_ids(text)
        self.assertEqual((len(ids), count), (40, 40))

    def test_a_truth_built_action_is_rendered_as_the_assistant_turn_and_reads_back(self) -> None:
        """The target of a truth-built example: the action as the supervised rows' assistant turn, through the chat template.

        The stand-in template writes a tool call in the model's wire form and, like the 12B's, adds the empty thought
        channel to a generation prompt only.
        """
        turn, thought = "<|turn>model\n", "<|channel>thought\n<channel|>"

        def wire(value) -> str:
            if isinstance(value, dict):
                return "{" + ",".join(f"{key}:{wire(item)}" for key, item in sorted(value.items())) + "}"
            if isinstance(value, list):
                return "[" + ",".join(wire(item) for item in value) + "]"
            return f'<|"|>{value}<|"|>' if isinstance(value, str) else json.dumps(value)

        class Processor:
            def __init__(self, *, calls: str | None = None, lead: str = "") -> None:
                self.seen: list[tuple[list, dict]] = []
                self.calls, self.lead = calls, lead

            def apply_chat_template(self, messages, **options):
                self.seen.append((copy.deepcopy(messages), options))
                text = "\n".join(message["content"] for message in messages if message["role"] != "assistant")
                if options["add_generation_prompt"]:
                    return text + turn + thought
                call = messages[-1]["tool_calls"][0]["function"]
                return (self.lead + text + turn + f"<|tool_call>call:{self.calls or call['name']}{wire(call['arguments'])}<tool_call|>"
                        + "<|tool_response>")

            def __call__(self, text=None, return_tensors=None):
                return {"input_ids": torch.tensor([[ord(character) for character in text]])}

        env = MinimalDiagnosisEnv(KERNEL)
        observation = rl_train.model_view(env.reset(ROOTS[1]))
        actions = ({"tool": "inspect", "arguments": {"view": "breakers"}}, {"tool": "correct_parameters", "arguments": {"line": 3}},
                   {"tool": "correct_measurements", "arguments": {"meters": [84, 19]}}, {"tool": "undo", "arguments": {}},
                   {"tool": "correct_topology", "arguments": {"breaker": "CB_9_B1_N3", "status": "open"}},
                   {"tool": "finish", "arguments": {"outcome": "resolved"}})
        with tempfile.TemporaryDirectory() as folder:
            config = _config(folder)
            processor = Processor()
            render, prompt_ids = rl_train.prompt_functions(config, "adapter", processor)
            render_target = rl_train.target_function(config, "adapter", processor)
            prompt, _count = prompt_ids(render(observation))
            for action in actions:
                target = render_target(observation, action)
                self.assertEqual("".join(map(chr, target)),
                                 f"<|tool_call>call:{action['tool']}{wire(action['arguments'])}<tool_call|><|tool_response>", action)
                messages, options = processor.seen[-1]
                self.assertEqual(messages, view.messages(observation) + [view.target_call(action)])
                self.assertEqual((options["tools"], options["add_generation_prompt"], options["enable_thinking"]),
                                 (view.tools(observation), False, False))
            # The completion follows the prompt the policy was given, token for token.
            self.assertEqual("".join(map(chr, prompt)), render(observation))
            self.assertTrue(render(observation).endswith(turn + thought))
            # A render that does not start with the generation prompt, or that reads back as another action, is refused.
            for label, broken in (("another prefix", Processor(lead="x")), ("another tool", Processor(calls="undo"))):
                with self.assertRaises(ValueError, msg=label):
                    rl_train.target_function(config, "adapter", broken)(observation, actions[1])


@unittest.skipIf(torch is None, "torch is not installed")
class IterationTests(unittest.TestCase):
    def test_one_iteration_from_the_baseline_to_the_next_rollout(self) -> None:
        holder: dict = {}
        with tempfile.TemporaryDirectory() as folder, _stand_ins(holder) as rl:
            config = _config(folder, baseline_evaluation=True)
            out = iteration_dir(config, 0)
            run = Path(config.output_dir)

            # The baseline: sharded, then closed.  The starting adapter repairs nothing.
            self.assertEqual(_run(config, "baseline", shard=(1, 2)), {"shard": "1/2", "evaluated": 1})
            self.assertEqual(rl.load_manifest(config).phase, "baseline")
            baseline = _run(config, "baseline")
            self.assertEqual((baseline["cases"], baseline["solved"], baseline["false_resolved"]), (2, 0, 2))
            self.assertNotIn("against_baseline", baseline)
            with contextlib.redirect_stdout(io.StringIO()):  # run again, the baseline is still not compared with itself
                again = rl.greedy_evaluation(config, adapter=config.initial_adapter, out=run / "baseline", shard=None,
                                             head={"baseline": True})
            self.assertNotIn("against_baseline", again)
            manifest = rl.load_manifest(config)
            self.assertEqual((manifest.phase, manifest.iteration, manifest.baseline["adapter"]), ("rollout", 0, config.initial_adapter))
            with self.assertRaisesRegex(RuntimeError, "at phase 'rollout'"):
                _run(config, "update")

            # The rollout: one shard samples its case and closes nothing; the unsharded call samples the other.
            self.assertEqual(_run(config, "rollout", shard=(0, 2))["groups_sampled"], 1)
            self.assertFalse((out / "rollout_summary.json").exists())
            rollout = _run(config, "rollout")
            self.assertEqual(rollout["groups_sampled"], 1)
            totals = rollout["store"]["totals"]
            self.assertEqual((totals["groups"], totals["episodes"], totals["unpaired_trajectories"]), (2, 4, 0))
            # In every group the first sample acts on the truth and the second claims the case without acting.
            self.assertEqual(totals["solved"], sum(REPAIRED))
            self.assertEqual(totals["groups_with_repaired_and_unrepaired_samples"], sum(REPAIRED))
            self.assertEqual(totals["groups_with_reward_spread"], 2)
            self.assertEqual(totals["false_resolved"], 4 - sum(REPAIRED))
            self.assertEqual(rollout["sampling"], {"temperature": 1.0, "top_p": 1.0, "top_k": None})
            self.assertEqual(_json(out / "roots.json"), [root["execution"]["scenario_id"] for root in ROOTS])
            with self.assertRaises(ValueError):
                _run(config, "update", shard=(0, 2))

            # The update: every stored step is rendered again to the prompt it was sampled from, and trained once.
            sampled_steps = sum(len(trajectory.steps) for _root, group in TrajectoryStore(out / "trajectories").iter_groups()
                                for trajectory in group)
            report = _run(config, "update")
            rows = report["rows"]
            self.assertEqual((rows["rows"], rows["policy_rows"], rows["informative_groups"]), (sampled_steps, sampled_steps, 2))
            self.assertEqual((rows["prompt_length_mismatches"], rows["prompt_digest_mismatches"], rows["prompt_digest_missing"],
                              rows["rows_without_sampling_record_rejected"], rows["prompt_truncated_rows"]), (0, 0, 0, 0, 0))
            self.assertEqual((rows["lifecycle_rows"], rows["supervised_rows"], rows["decision_credit_rows"]), (0, 0, 0))
            self.assertEqual((report["update"]["optimizer_steps"], report["update"]["rows"]), (1, sampled_steps))
            self.assertEqual(report["integrity"]["failures"], [])
            self.assertEqual(report["optimizer"], {"carried_from": None, "steps_before": 0, "steps_after": 1,
                                                   "continuity_since_iteration": 0})
            self.assertIsNone(report["kl_reference"])
            self.assertEqual(report["scores"]["advantage"], {"method": "grpo", "scale": "none"})
            self.assertTrue((out / OPTIMIZER_FILE).is_file())
            self.assertFalse((out / "update_checkpoint").exists())
            before = torch.load(Path(config.initial_adapter) / "weights.pt")["table"]
            after = torch.load(out / "adapter" / "weights.pt")["table"]
            self.assertFalse(torch.equal(before, after))
            # The advantages are the rewards less their case's mean, unscaled.
            for entry in _json(out / "advantages.json")["by_root"].values():
                mean = sum(entry["rewards"]) / 2
                self.assertEqual([round(value, 9) for value in entry["advantages"]],
                                 [round(value - mean, 9) for value in entry["rewards"]])

            # The evaluation: the updated adapter, case by case against the baseline.
            evaluation = _run(config, "evaluate")
            self.assertEqual((evaluation["cases"], evaluation["solved"]), (2, sum(REPAIRED)))
            self.assertEqual(evaluation["against_baseline"],
                             {"cases_in_both": 2, "cases_in_one_only": 0, "solved_before": 0, "solved_now": sum(REPAIRED),
                              "gained": sum(REPAIRED), "lost": 0})
            written = _json(out / "evaluation.json")
            self.assertEqual((written["iteration"], written["settings"]["decoding"], written["settings"]["precision"]),
                             (0, "greedy", "bf16"))
            self.assertEqual(len(written["against_baseline"]["gained"]), sum(REPAIRED))
            manifest = rl.load_manifest(config)
            self.assertEqual((manifest.phase, manifest.iteration, manifest.current_adapter), ("rollout", 1, str(out / "adapter")))
            self.assertEqual(manifest.history[0]["evaluation"]["solved"], sum(REPAIRED))
            self.assertEqual(_json(run / "manifest.json")["contract"], DRIVER_CONTRACT)

            # The next iteration samples the same cases with other seeds, each the store's own for its sample.
            _run(config, "rollout")
            ids = [root["execution"]["scenario_id"] for root in ROOTS]
            for iteration in (0, 1):
                self.assertEqual(sorted(holder["seeds"][4 * iteration: 4 * iteration + 4]),
                                 sorted(sample_seed(config.seed + iteration, case, index) for case in ids for index in (0, 1)))
            self.assertEqual(len(set(holder["seeds"])), 8)
            # ... and with the adapter the first update left (iteration 0 was sampled in two calls).
            self.assertEqual(holder["sampled_from"], [config.initial_adapter, config.initial_adapter, str(out / "adapter")])
            # Its update continues the optimizer the first one left.
            second = _run(config, "update")
            self.assertEqual(second["optimizer"], {"carried_from": str(out / OPTIMIZER_FILE), "steps_before": 1,
                                                   "steps_after": 2, "continuity_since_iteration": 0})
            self.assertEqual([load["adapter"] for load in holder["loads"]], [config.initial_adapter, str(out / "adapter")])

    def test_the_whole_iteration_runs_in_one_call(self) -> None:
        with tempfile.TemporaryDirectory() as folder, _stand_ins({}) as rl:
            config = _config(folder, baseline_evaluation=True)
            with contextlib.redirect_stdout(io.StringIO()):
                rl.run_iterations(config, 1)
            manifest = rl.load_manifest(config)
            self.assertEqual((manifest.phase, manifest.iteration, len(manifest.history)), ("rollout", 1, 1))
            self.assertEqual(rl.status(config)["baseline"]["evaluation"]["solved"], 0)

    def test_a_finished_evaluation_of_the_starting_adapter_is_taken_as_the_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as folder, _stand_ins({}) as rl:
            plain = _config(folder, baseline_evaluation=True)
            earlier = Path(folder) / "supervised_evaluation"
            limits = rl.apply_prompt_limits(plain)

            def write(**changed) -> None:
                """The supervised cell's evaluation of the starting adapter, as research.minimal_env.evaluate writes it."""
                records, missing = evaluate.read_episodes(ROOTS, earlier / "episodes")
                self.assertEqual(missing, [])
                settings = {"model_choice": "12b", "load_in_4bit": False, "decoding": "greedy", "max_actions": 16, "table_versions": 2,
                            "max_input_tokens": str(limits["max_input_tokens"]), "max_new_tokens": str(limits["max_new_tokens"])}
                payload = evaluate.evaluation_payload(records, suite=plain.development_suite, adapter=plain.initial_adapter,
                                                      settings={**settings, **changed.pop("settings", {})})
                (earlier / "evaluation.json").write_text(json.dumps({**payload, **changed}), encoding="utf-8")

            def config_in(name: str) -> MinimalRLConfig:
                return _config(folder, baseline_evaluation=True, baseline_from=str(earlier), output_dir=str(Path(folder) / name))

            # No finished evaluation there yet: the baseline is evaluated here (the starting adapter repairs nothing).
            here = _run(config_in("run_before"), "baseline")
            self.assertEqual((here["solved"], here["baseline_source"]), (0, "evaluated by this run"))
            # The evaluation, by a policy that repairs: its episodes are the baseline's, and no policy is built.
            env = MinimalDiagnosisEnv(KERNEL)
            evaluate.evaluate_cases(ROOTS, policy_factory=lambda: _Scripted(repair=True), env_factory=lambda: env,
                                    episodes_dir=earlier / "episodes")
            write()

            def no_policy(config, adapter):
                def build():
                    raise AssertionError("the baseline was run again")
                return build

            with mock.patch.object(rl, "greedy_policy_factory", no_policy):
                config = config_in("run_adopting")
                self.assertEqual(_run(config, "baseline", shard=(0, 2)), {"shard": "0/2", "evaluated": 0})
                adopted = _run(config, "baseline")
            self.assertEqual((adopted["solved"], adopted["baseline_source"]), (sum(REPAIRED), str(earlier)))
            self.assertEqual(_json(Path(config.output_dir) / "baseline" / "evaluation.json")["baseline_source"], str(earlier))
            # An evaluation under another episode length, or of another adapter, is refused.
            for label, changed in (("another budget", {"settings": {"max_actions": 20}}),
                                   ("another adapter", {"adapter": str(Path(folder) / "another")})):
                write(**changed)
                with self.assertRaisesRegex(ValueError, "not an evaluation of this run's starting adapter", msg=label):
                    _run(config_in(f"run_{label.replace(' ', '_')}"), "baseline")
            with self.assertRaises(ValueError):
                _config(folder, baseline_from=str(earlier))  # without baseline_evaluation there is nothing to take it for

    def test_a_development_limit_keeps_the_first_cases_only(self) -> None:
        with tempfile.TemporaryDirectory() as folder, _stand_ins({}):
            config = _config(folder, baseline_evaluation=True, development_limit=1)
            self.assertEqual(_run(config, "baseline")["cases"], 1)
            written = _json(Path(config.output_dir) / "baseline" / "evaluation.json")
            self.assertEqual([episode["scenario_id"] for episode in written["episodes"]], [ROOTS[0]["execution"]["scenario_id"]])
            with self.assertRaises(ValueError):
                _config(folder, development_limit=0)

    def test_samples_that_all_score_the_same_leave_the_adapter_as_it_was(self) -> None:
        holder = {"never_repair": True}
        with tempfile.TemporaryDirectory() as folder, _stand_ins(holder) as rl:
            config = _config(folder)
            out = iteration_dir(config, 0)
            rollout = _run(config, "rollout")
            self.assertEqual((rollout["store"]["totals"]["groups_with_reward_spread"],
                              rollout["store"]["totals"]["groups_with_distinct_samples"]), (0, 0))
            report = _run(config, "update")
            self.assertIsNone(report["update"])
            self.assertTrue(report["integrity"]["ok"])
            self.assertNotIn("loads", holder)  # no model is loaded for an update that has nothing to train on
            self.assertTrue(torch.equal(torch.load(Path(config.initial_adapter) / "weights.pt")["table"],
                                        torch.load(out / "adapter" / "weights.pt")["table"]))
            self.assertIn("skipped", _run(config, "evaluate"))
            self.assertEqual(rl.load_manifest(config).iteration, 1)

    def test_a_prompt_that_is_not_the_sampled_one_stops_the_run(self) -> None:
        holder = {"other_prompt": True}
        with tempfile.TemporaryDirectory() as folder, _stand_ins(holder) as rl:
            config = _config(folder)
            out = iteration_dir(config, 0)
            _run(config, "rollout")
            with self.assertRaisesRegex(IntegrityError, "likelihood_provenance"):
                _run(config, "update")
            report = _json(out / "update_report.json")
            self.assertIsNone(report["update"])
            self.assertGreater(report["rows"]["prompt_digest_mismatches"], 0)
            self.assertEqual(report["rows"]["rows"], 0)
            self.assertTrue((out / MARKER_FILE).is_file())
            self.assertEqual(rl.load_manifest(config).phase, "update")
            holder["other_prompt"] = False
            with self.assertRaisesRegex(IntegrityError, "remove the marker"):  # until someone has looked
                _run(config, "update")

    def test_the_kl_reference_is_the_starting_adapter(self) -> None:
        holder: dict = {}
        with tempfile.TemporaryDirectory() as folder, _stand_ins(holder) as rl:
            config = _config(folder, update={"learning_rate": 0.01, "beta_kl": 0.04})
            _run(config, "rollout")
            holder["stop_at_load"] = True
            manifest = rl.load_manifest(config)
            manifest.current_adapter = str(Path(folder) / "a_later_adapter")
            with self.assertRaises(_Stop):
                rl.phase_update(config, manifest)
            self.assertEqual(holder["loads"], [{"adapter": manifest.current_adapter, "reference": config.initial_adapter}])

    def test_a_preempted_update_resumes_where_it_stopped(self) -> None:
        # Two optimizer steps, one case's rows in each (three or four rows a case): the first step lands, and the
        # second is cut one or two rows in.  The resumed update starts from the first step's weights.
        update = {"learning_rate": 0.01, "carry_optimizer": True, "chunk_by": "group", "optimizer_steps": 2}
        tables = {}
        for preempted in (False, True):
            holder: dict = {}
            with tempfile.TemporaryDirectory() as folder, _stand_ins(holder) as rl:
                config = _config(folder, update=update)
                out = iteration_dir(config, 0)
                _run(config, "rollout")
                if preempted:
                    holder["fail_after"] = 5
                    with self.assertRaises(_Preempted):
                        _run(config, "update")
                    holder["fail_after"] = None
                    state = _json(out / "update_checkpoint" / "state.json")
                    self.assertEqual(state["completed_chunks"], 1)
                    self.assertIn(state["chunk_rows_done"], (1, 2))
                    self.assertEqual(rl.load_manifest(config).phase, "update")
                    self.assertFalse((out / "adapter").exists())
                report = _run(config, "update")
                self.assertEqual(report["resumed_from_checkpoint"], str(out / "update_checkpoint" / "step_1") if preempted else None)
                self.assertEqual((report["update"]["resumed_from"] or {}).get("completed_chunks"), 1 if preempted else None)
                self.assertEqual(report["update"]["optimizer_steps"], 2)
                self.assertTrue(report["integrity"]["ok"])
                self.assertFalse((out / "update_checkpoint").exists())
                tables[preempted] = torch.load(out / "adapter" / "weights.pt")["table"]
                start = torch.load(Path(config.initial_adapter) / "weights.pt")["table"]
        self.assertTrue(torch.allclose(tables[True], tables[False], atol=1e-7))
        self.assertFalse(torch.allclose(tables[True], start, atol=1e-4))


class FocusDrawTests(unittest.TestCase):
    """A focused run draws from the cases with a truth-built sequence, split by what the starting adapter did on them."""

    def test_the_draw_is_the_unrepaired_cases_with_a_share_of_the_repaired_ones(self) -> None:
        cases = [{"execution": {"scenario_id": f"case{index:02d}"}, "audit": {"compound": {"class": "a"}}} for index in range(10)]
        finish = {"tool": "finish", "arguments": {"outcome": "resolved"}}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            plain = _config(folder)
            limits = rl_train.apply_prompt_limits(plain)
            # Cases 0 to 7 have a sequence; the starting adapter repaired 0 to 2.  Cases 8 and 9 have none.
            records = [{"demonstration": "success", "solved": True, "scenario_id": f"case{index:02d}", "compound_class": "a",
                        "steps": [{"step": 1, "action": finish, "result": {"status": "ok"}}]} for index in range(8)]
            truth_sequences.dump(truth_sequences.build(records), root / "sequences.json")
            passed = root / "pass"
            passed.mkdir()
            episodes = [{"scenario_id": f"case{index:02d}", "solved": index < 3} for index in range(8)]

            def write(**changed) -> None:
                payload = {"adapter": plain.initial_adapter, "episodes": episodes,
                           "settings": {"model_choice": "12b", "load_in_4bit": False, "decoding": "greedy", "max_actions": 16,
                                        "table_versions": 2, "max_input_tokens": str(limits["max_input_tokens"]),
                                        "max_new_tokens": str(limits["max_new_tokens"])}}
                (passed / "evaluation.json").write_text(json.dumps({**payload, **changed}), encoding="utf-8")

            write()
            focus = {"truth_sequences": str(root / "sequences.json"), "evaluation": str(passed), "failed_per_iteration": 2,
                     "solved_per_iteration": 1}
            config = _config(folder, focus=focus, rows=FOCUS_ROWS)
            found = rl_train.focus_cases(config, cases)
            names = lambda roots: [item["execution"]["scenario_id"] for item in roots]  # noqa: E731
            failed, solved = [f"case{index:02d}" for index in range(3, 8)], ["case00", "case01", "case02"]
            self.assertEqual((names(found["failed"]), names(found["solved"]), len(found["sequences"])), (failed, solved, 8))
            drawn = [names(rl_train.select_iteration_roots(config, cases, iteration)) for iteration in range(6)]
            for iteration, picked in enumerate(drawn):
                self.assertEqual(len(picked), 3, iteration)
                self.assertTrue(set(picked[:2]) <= set(failed) and picked[2] in solved, picked)  # never case08 or case09
            self.assertEqual(len({name for picked in drawn[:2] for name in picked[:2]}), 4)  # no case twice before all were drawn
            self.assertEqual({name for picked in drawn[:3] for name in picked[:2]}, set(failed))
            self.assertEqual({picked[2] for picked in drawn[:3]}, set(solved))
            self.assertEqual(drawn, [names(rl_train.select_iteration_roots(config, cases, iteration)) for iteration in range(6)])
            # No repaired case is drawn when none is asked for.
            alone = _config(folder, focus={**focus, "solved_per_iteration": 0}, rows=FOCUS_ROWS)
            self.assertEqual(len(rl_train.select_iteration_roots(alone, cases, 0)), 2)

            # What would make the split something else than what the starting adapter does on these cases.
            for label, changed, message in (
                    ("another adapter", {"adapter": str(root / "another")}, "not an evaluation of this run's starting adapter"),
                    ("another budget", {"settings": {"max_actions": 20}}, "not an evaluation of this run's starting adapter"),
                    ("a case not run", {"episodes": episodes[:-1]}, "holds no episode of case07"),
                    ("nothing unrepaired", {"episodes": [{**entry, "solved": True} for entry in episodes]}, "nothing to focus on")):
                write(**changed)
                with self.assertRaisesRegex(ValueError, message, msg=label):
                    rl_train.focus_cases(config, cases)
            write()
            with self.assertRaisesRegex(ValueError, "1 cases of .* are not in the training suite"):
                rl_train.focus_cases(config, cases[1:])
            (passed / "evaluation.json").unlink()
            with self.assertRaisesRegex(ValueError, "holds no finished evaluation"):
                rl_train.select_iteration_roots(config, cases, 0)

            # The config: the draw is focus's own, and its examples need sampled rows beside them.
            for bad in (dict(focus={**focus, "weight": 1.0}), dict(focus={"failed_per_iteration": 2}),
                        dict(focus={**focus, "evaluation": None}), dict(focus={**focus, "failed_per_iteration": 0}),
                        dict(focus={**focus, "demonstration_weight": -1.0}), dict(focus=focus, roots_per_iteration=4),
                        dict(focus=focus, rows={**FOCUS_ROWS, "preservation": "none"})):
                with self.assertRaises(ValueError, msg=str(bad)):
                    _config(folder, **{"rows": FOCUS_ROWS, **bad})
            settings = _config(folder, focus={**focus, "demonstration_weight": 0.0}, rows={}).focus_settings()
            self.assertEqual((settings.enabled, settings.examples), (True, False))
            self.assertFalse(plain.focus_settings().enabled)

    def test_the_focused_example_config_differs_from_the_plain_one_in_its_draw_alone(self) -> None:
        folder = Path(rl_train.__file__).parent
        focused = MinimalRLConfig.load(folder / "rl_focus_config_example.json")
        plain = MinimalRLConfig.load(folder / "rl_config_example.json")
        differing = {name for name in MinimalRLConfig.__dataclass_fields__ if getattr(focused, name) != getattr(plain, name)}
        self.assertEqual(differing, {"output_dir", "seed", "samples_per_root", "roots_per_iteration", "root_sampling", "focus"})
        settings = focused.focus_settings()
        self.assertTrue(settings.examples and settings.failed_per_iteration > settings.solved_per_iteration > 0)
        # As many sampled episodes an iteration as the plain run's.
        self.assertEqual((settings.failed_per_iteration + settings.solved_per_iteration) * focused.samples_per_root,
                         plain.roots_per_iteration * plain.samples_per_root)
        sequences = truth_sequences.load(Path(rl_train.REPO_ROOT, settings.truth_sequences))
        self.assertEqual(len(sequences), 2093)


@unittest.skipIf(torch is None, "torch is not installed")
class FocusIterationTests(unittest.TestCase):
    """A case none of whose samples is repaired gets its truth-built sequence, as rows the update trains beside the sampled ones."""

    def test_a_case_without_a_repaired_sample_is_taught_its_truth_built_sequence(self) -> None:
        holder = {"never_repair": True}
        ids = [root["execution"]["scenario_id"] for root in ROOTS]
        with tempfile.TemporaryDirectory() as folder, _stand_ins(holder) as rl:
            config = _config(folder, focus=_focus(folder), rows=FOCUS_ROWS)
            out = iteration_dir(config, 0)
            self.assertEqual(_run(config, "rollout", shard=(0, 2))["groups_sampled"], 1)
            self.assertFalse((out / "demonstrations").exists())  # a shard replays nothing
            rollout = _run(config, "rollout")
            focus = rollout["focus"]
            self.assertEqual(focus["available"], {"left_unrepaired_by_the_starting_adapter": 2, "repaired_by_the_starting_adapter": 0})
            self.assertEqual((focus["left_unrepaired_by_the_starting_adapter"]["cases"],
                              focus["left_unrepaired_by_the_starting_adapter"]["cases_with_no_sample_repaired"],
                              focus["repaired_by_the_starting_adapter"]["cases"]), (2, 2, 0))
            replayed = rollout["truth_built_examples"]
            self.assertEqual((replayed["cases_without_a_repaired_sample"], replayed["examples"], replayed["examples_replayed_now"],
                              replayed["without_a_sequence"], replayed["replay_did_not_repair"]), (2, 2, 2, 0, []))
            self.assertEqual(rollout["store"]["totals"]["episodes"], 4)  # the groups hold the sampled episodes alone
            kept = dict(TrajectoryStore(out / "demonstrations").iter_groups())
            self.assertEqual(sorted(kept), sorted(ids))
            for root in ROOTS:
                case = root["execution"]["scenario_id"]
                [example] = kept[case]
                self.assertEqual([step.action for step in example.steps], _truth_order(root))
                self.assertEqual((example.sample_index, example.contract, example.evaluation["score"]["solved"], example.pairing_ok),
                                 (EXAMPLE_SAMPLE, rl_train.EXAMPLE_CONTRACT, True, True))
                self.assertTrue(all(not step.completion_token_ids and not step.token_probabilities for step in example.steps))
                # Its first action is taken on what every sample of the case was shown first.
                sampled = TrajectoryStore(out / "trajectories").read_group(case)[0]
                self.assertEqual(view.user_text(example.steps[0].model_observation), view.user_text(sampled.steps[0].model_observation))
                self.assertEqual(len(example.steps[-1].model_observation["history"]), len(example.steps) - 1)
                # ... and is kept as the model policy keeps the observation it decoded from: marked as the whole history.
                self.assertEqual(example.steps[0].model_observation,
                                 json.loads(json.dumps(MinimalModelPolicy._whole(MinimalDiagnosisEnv(KERNEL).reset(root)))))

            # The update: the sampled rows of both cases for the KL term, and one supervised row per truth-built action.
            report = _run(config, "update")
            rows = report["rows"]
            taught_rows = sum(len(_truth_order(root)) for root in ROOTS)
            self.assertEqual((rows["policy_rows"], rows["preservation_rows"], rows["supervised_rows"], rows["rows"]),
                             (0, 2, taught_rows, 2 + taught_rows))
            taught = rows["truth_built_examples"]
            self.assertEqual((taught["examples"], taught["examples_with_rows"], taught["rows"], taught["unrenderable"],
                              taught["prompt_over_the_input_limit"]), (2, 2, taught_rows, 0, 0))
            # A row's weight is the example's reward above the mean of the case's samples (0.98 and 0.97 against -0.51).
            self.assertEqual((round(taught["row_weights"]["min"], 6), round(taught["row_weights"]["max"], 6)), (1.48, 1.49))
            update = report["update"]
            self.assertEqual((update["rows"], update["supervised_rows"], update["optimizer_steps"]), (2 + taught_rows, taught_rows, 1))
            self.assertEqual(update["row_accounting"]["rows_by_kind"]["supervised"], taught_rows)
            self.assertGreater(update["supervised_loss"][0], 0.0)
            self.assertEqual((report["integrity"]["ok"], report["integrity"]["failures"]), (True, []))
            self.assertEqual(report["integrity"]["checks"]["update_work"]["rows_built"], 2 + taught_rows)
            # Every action was rendered once, in its sequence's order, on the prompt it was taken on.
            expected = [action for case in sorted(ids) for action in _truth_order(ROOTS[ids.index(case)])]
            self.assertEqual([action for _prompt, action in holder["targets"]], expected)
            first_prompts = {view.user_text(kept[case][0].steps[0].model_observation) for case in ids}
            self.assertEqual({prompt for prompt, action in holder["targets"] if action == _truth_order(ROOTS[0])[0]
                              or action == _truth_order(ROOTS[1])[0]}, first_prompts)
            before = torch.load(Path(config.initial_adapter) / "weights.pt")["table"]
            self.assertFalse(torch.equal(before, torch.load(out / "adapter" / "weights.pt")["table"]))

            # A closing call that runs again replays nothing.
            manifest = rl.load_manifest(config)
            manifest.iteration, manifest.phase = 0, "rollout"
            with contextlib.redirect_stdout(io.StringIO()):
                again = rl.phase_rollout(config, manifest)["truth_built_examples"]
            self.assertEqual((again["examples"], again["examples_replayed_now"]), (2, 0))

    def test_a_case_with_a_repaired_sample_gets_no_example(self) -> None:
        # The first sample of every case acts on the truth: each group has its own contrast.
        with tempfile.TemporaryDirectory() as folder, _stand_ins({}) as rl:
            config = _config(folder, focus=_focus(folder), rows=FOCUS_ROWS)
            out = iteration_dir(config, 0)
            rollout = _run(config, "rollout")
            self.assertEqual(rollout["focus"]["left_unrepaired_by_the_starting_adapter"]["cases_with_a_repaired_and_an_unrepaired_sample"], 2)
            self.assertEqual((rollout["truth_built_examples"]["cases_without_a_repaired_sample"],
                              rollout["truth_built_examples"]["examples"]), (0, 0))
            # An example left beside such a group, by whatever means, is not trained on.
            example = rl.example_trajectory(config, ROOTS[0], _truth_order(ROOTS[0]), MinimalDiagnosisEnv(KERNEL),
                                            rl_train.RewardWeights.from_mapping(config.reward))
            TrajectoryStore(out / "demonstrations").write_group(example.root, [example])
            report = _run(config, "update")
            self.assertEqual((report["rows"]["supervised_rows"], report["update"]["supervised_rows"]), (0, 0))
            self.assertNotIn("truth_built_examples", report["rows"])
            self.assertGreater(report["rows"]["policy_rows"], 0)

    def test_a_sequence_that_does_not_repair_here_is_not_kept(self) -> None:
        holder = {"never_repair": True}
        claim = [{"tool": "finish", "arguments": {"outcome": "resolved"}}]
        with tempfile.TemporaryDirectory() as folder, _stand_ins(holder):
            # The first case's sequence only claims the case; the second case has none at all.
            config = _config(folder, focus=_focus(folder, sequences_for=(0,), orders={0: claim}, repaired=(False, False)),
                             rows=FOCUS_ROWS)
            out = iteration_dir(config, 0)
            rollout = _run(config, "rollout")
            self.assertEqual(rollout["store"]["totals"]["groups"], 1)  # the case without a sequence is never drawn
            replayed = rollout["truth_built_examples"]
            self.assertEqual((replayed["examples"], replayed["replay_did_not_repair"]), (0, [ROOTS[0]["execution"]["scenario_id"]]))
            self.assertEqual(list(TrajectoryStore(out / "demonstrations").iter_groups()), [])
            report = _run(config, "update")  # no contrast and no example: the adapter is carried over
            self.assertIsNone(report["update"])
            self.assertIn("no truth-built example", report["note"])
            self.assertNotIn("loads", holder)

    def test_the_configured_weight_scales_every_example_row(self) -> None:
        holder = {"never_repair": True}
        with tempfile.TemporaryDirectory() as folder, _stand_ins(holder):
            config = _config(folder, focus=_focus(folder, demonstration_weight=0.5), rows=FOCUS_ROWS)
            _run(config, "rollout")
            report = _run(config, "update")
            taught = report["rows"]["truth_built_examples"]
            # Half of the sequences' rewards above their samples' mean (0.98 and 0.97 against -0.51).
            self.assertEqual((taught["weight"], round(taught["row_weights"]["min"], 6), round(taught["row_weights"]["max"], 6)),
                             (0.5, 0.74, 0.745))
            self.assertTrue(report["integrity"]["ok"])

    def test_without_weight_no_example_is_replayed(self) -> None:
        holder = {"never_repair": True}
        with tempfile.TemporaryDirectory() as folder, _stand_ins(holder):
            config = _config(folder, focus=_focus(folder, demonstration_weight=0.0), rows=FOCUS_ROWS)
            rollout = _run(config, "rollout")
            self.assertIn("focus", rollout)
            self.assertNotIn("truth_built_examples", rollout)
            self.assertFalse((iteration_dir(config, 0) / "demonstrations").exists())
            self.assertIsNone(_run(config, "update")["update"])

    def test_an_action_that_does_not_render_stops_the_run(self) -> None:
        holder = {"never_repair": True, "unrenderable": True}
        with tempfile.TemporaryDirectory() as folder, _stand_ins(holder) as rl:
            config = _config(folder, focus=_focus(folder), rows=FOCUS_ROWS)
            out = iteration_dir(config, 0)
            _run(config, "rollout")
            with self.assertRaisesRegex(IntegrityError, "could not be rendered"):
                _run(config, "update")
            report = _json(out / "update_report.json")
            self.assertEqual(report["rows"]["supervised_rows_unrenderable"], sum(len(_truth_order(root)) for root in ROOTS))
            # The sampled rows alone were left to train on; the run does not go on from that update.
            self.assertEqual((report["update"]["rows"], report["update"]["supervised_rows"], report["rows"]["supervised_rows"]), (2, 0, 0))
            self.assertFalse(report["integrity"]["ok"])
            self.assertTrue((out / MARKER_FILE).is_file())
            self.assertEqual(rl.load_manifest(config).phase, "update")

    def test_a_preempted_update_with_examples_resumes_to_the_same_weights(self) -> None:
        tables = {}
        for preempted in (False, True):
            holder = {"never_repair": True}
            with tempfile.TemporaryDirectory() as folder, _stand_ins(holder):
                config = _config(folder, focus=_focus(folder), rows=FOCUS_ROWS)
                out = iteration_dir(config, 0)
                _run(config, "rollout")
                if preempted:
                    holder["fail_after"] = 4
                    with self.assertRaises(_Preempted):
                        _run(config, "update")
                    holder["fail_after"] = None
                    self.assertGreater(_json(out / "update_checkpoint" / "state.json")["chunk_rows_done"], 0)
                report = _run(config, "update")
                self.assertTrue(report["integrity"]["ok"])
                self.assertEqual(bool(report["update"]["resumed_from"]), preempted)
                tables[preempted] = torch.load(out / "adapter" / "weights.pt")["table"]
        self.assertTrue(torch.allclose(tables[True], tables[False], atol=1e-7))

    def test_the_check_names_what_a_focused_run_draws_from(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            config = _config(folder, focus=_focus(folder, repaired=(False, True), solved_per_iteration=1), rows=FOCUS_ROWS)
            found = rl_train.check(config)["focus"]
            self.assertEqual((found["cases_with_a_truth_built_sequence"], found["left_unrepaired_by_the_starting_adapter"],
                              found["repaired_by_the_starting_adapter"], found["drawn_for_the_first_iteration"],
                              found["truth_built_examples"]),
                             (2, 1, 1, {"left_unrepaired": 1, "repaired": 1}, True))
            self.assertFalse(Path(config.output_dir).exists())


if __name__ == "__main__":
    unittest.main()
