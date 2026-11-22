"""The RL iteration driver of the minimal environment: sample, score, update, evaluate.

    python -m research.minimal_env.rl_train --config rl.json --status
    python -m research.minimal_env.rl_train --config rl.json --check                       # read the inputs, run nothing
    python -m research.minimal_env.rl_train --config rl.json --phase rollout --shard 0/2   # one worker's slice
    python -m research.minimal_env.rl_train --config rl.json --phase rollout               # closes the phase
    python -m research.minimal_env.rl_train --config rl.json --iterations 1                # the next iteration, whole

One iteration:

``rollout``
    K sampled episodes of every drawn training case in ``MinimalDiagnosisEnv``, with the current adapter choosing
    every action (``psse_env.minimal.rollout``).  One file per case; a restart skips the cases that are done.
``update``
    One on-policy LoRA update on the sampled tokens.  An episode's advantage is its reward less the mean reward of
    its case's samples; the reward is ``psse_env.minimal.scoring.episode_reward``: the repair outcome, less the
    actual damage, a false claim of resolution and a small cost per action.  A KL penalty (``update.beta_kl``)
    holds the policy near the run's starting adapter, the supervised model.
``evaluate``
    The updated adapter, greedy, on the development suite, written as ``research.minimal_env.evaluate`` writes
    it, so a number here stands beside the supervised pass's own evaluation.

A run whose config sets ``baseline_evaluation`` first evaluates the starting adapter the same way (``baseline``).
``manifest.json`` holds the phase the run is at and advances only when a phase completes; every phase survives
preemption.  The rollout and the evaluations shard over worker processes (``--shard i/n``); the unsharded call of
the same phase finishes whatever is left and closes it.

What is this environment's own: the episode, the reward, the prompt and the tools.  What is the RL series'
(``psse_env.rl``), used as it is: the trajectory store, the group advantages, the row builder, the update with its
checkpoint and carried optimizer, and the integrity checks.  There is no lifecycle term, no decision credit and no
supervised target on a sampled step: a correction is kept or undone by the model's own later action and credited
by the episode's outcome.  Truth-built sequences are never part of a group.

``focus`` (2026-11-18) aims a run at the training cases a truth-built sequence repairs and the starting adapter
does not.  It names the file of those sequences (``research.minimal_env.truth_sequences``) and a greedy
evaluation of the starting adapter on their cases; an iteration then draws ``failed_per_iteration`` of the cases
that evaluation left unrepaired and ``solved_per_iteration`` of the repaired ones, and no case without a sequence.
For a drawn case none of whose samples is repaired, the closing call of the rollout replays the case's sequence
in the environment and keeps it beside the groups (``demonstrations/``).  The update trains the replayed actions
as supervised rows: each action on the prompt the model would have read there, rendered as the supervised pass
taught it, with weight ``demonstration_weight`` times how far the sequence's reward lies above the mean of the
case's samples.  Those rows are imitation of the truth-built sequence, not on-policy data: they have no sampling
record, no advantage and no part in the KL term or the on-policy check.

``forks`` (2026-11-22, the RL plan's stage B) lets an update learn an action the sampling never takes.  In the
rollout phase, beside the sampled episodes, the current adapter's greedy episode is run on every drawn case and
forked at every state: each other tool is forced once (the model writes the arguments), the model continues
greedily, and the branch is scored like any episode (``psse_env.minimal.forks``).  At a state where a branch
returned more than the greedy episode, the update trains that branch's action on that state's prompt as a
supervised row, with weight ``forks.weight`` times the difference of the returns.  Which states and tools matter
is nowhere written: the forks are everywhere, and the truth enters through the reward alone.  A forced step is
the harness's, so nothing after it is trained, and such a row has no sampling record, no advantage and no part
in the KL term, like a truth-built example's.  ``evaluate_every`` evaluates every N-th iteration's adapter only.

Settings that carry what the earlier RL arms measured (``rl_config_example.json`` beside this file sets them;
docs/minimal_env_20261115.md, section 12, has the reasons):

* ``advantage_scale`` ``none`` (the default here).  Samples of a case that all repair it still differ by an
  action or two, and the group scaling would turn that 0.01 of cost into an advantage as large as a repair's.
* ``update.optimizer_steps`` 1 with ``update.carry_optimizer``: two steps of a new AdamW follow whichever half of
  the batch came first.  With one step there is nothing for a clip to bound: ``update.clip_ratio`` null.
* ``update.loss_normalization`` ``dapo``: the only one under which what a case's samples share cancels exactly.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psse_env.minimal.env import DEFAULT_MAX_ACTIONS, DEFAULT_TABLE_VERSIONS, MinimalDiagnosisEnv  # noqa: E402
from psse_env.minimal import rollout as minimal_rollout  # noqa: E402
from psse_env.minimal.episode import OUTCOME_PROMPT_LIMIT, run_episode  # noqa: E402
from psse_env.minimal.forks import FORK_SAMPLE, ForkSettings, ForkStore, fork_roots  # noqa: E402
from psse_env.minimal.forks import summarize as summarize_forks  # noqa: E402
from psse_env.minimal.rollout import rollout_roots  # noqa: E402
from psse_env.minimal.scoring import RewardWeights  # noqa: E402
from psse_env.rl.advantages import GroupAdvantages, group_advantages, summarize_groups  # noqa: E402
from psse_env.rl.integrity import MARKER_FILE, IntegrityError, update_integrity  # noqa: E402
from psse_env.rl.rollout import (  # noqa: E402
    StepRecord,
    Trajectory,
    TrajectoryStore,
    distinct_trajectories,
    reward_spread,
    scenario_family,
    scenario_root,
)
from psse_env.rl.rows import RowSettings, TrainingRow, build_rows, build_step_row  # noqa: E402
from psse_env.rl.sampling import RootSampling, root_stratum, stratified_roots  # noqa: E402
from psse_env.rl.train import (  # noqa: E402
    OPTIMIZER_FILE,
    Manifest,
    _carried_optimizer_file,
    _carry_adapter_over,
    _continuity_start,
    _release_gpu,
    _write_json,
    apply_prompt_limits,
    iteration_dir,
    iteration_roots,
    parse_shard,
    shard_scenarios,
)
from psse_env.rl.train import _load_trainable_model as load_trainable_model  # noqa: E402
from psse_env.rl.train import _save_trained_adapter as save_trained_adapter  # noqa: E402
from psse_env.rl.update import (  # noqa: E402
    UpdateCheckpoint,
    UpdateSettings,
    build_optimizer,
    load_optimizer_state,
    optimizer_steps_taken,
    policy_gradient_update,
    save_optimizer_state,
)
from research.minimal_env import evaluate, truth_sequences  # noqa: E402

DRIVER_CONTRACT = "minimal_rl_driver_v1"
#: A truth-built sequence replayed for a case none of whose samples was repaired.
EXAMPLE_CONTRACT = "minimal_truth_built_example_v1"
#: ``sample_index`` of such an example: beside the case's sampled episodes, never one of them.
EXAMPLE_SAMPLE = 1000
PHASES = ("baseline", "rollout", "update", "evaluate")
ITERATION_PHASES = ("rollout", "update", "evaluate")
SHARDED_PHASES = ("baseline", "rollout", "evaluate")


@dataclass(frozen=True)
class FocusSettings:
    """A run aimed at the cases a truth-built sequence repairs and the starting adapter does not (the module docstring)."""

    #: A file of ``research.minimal_env.truth_sequences``; null leaves the run's draw as ``root_sampling`` says.
    truth_sequences: str | None = None
    #: A folder ``research.minimal_env.evaluate`` wrote for the starting adapter on the file's cases.
    evaluation: str | None = None
    #: Cases of an iteration that the evaluation left unrepaired, and cases it repaired.
    failed_per_iteration: int = 0
    solved_per_iteration: int = 0
    #: Weight of a truth-built example's rows, times the example's reward above the mean of its case's samples;
    #: 0 adds no example.
    demonstration_weight: float = 1.0

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "FocusSettings":
        payload = dict(payload or {})
        unknown = sorted(set(payload) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown focus fields: {unknown}")
        settings = cls(**payload)
        settings.validate()
        return settings

    @property
    def enabled(self) -> bool:
        return bool(self.truth_sequences)

    @property
    def examples(self) -> bool:
        return self.enabled and float(self.demonstration_weight) > 0.0

    def validate(self) -> None:
        for name in ("failed_per_iteration", "solved_per_iteration"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"focus.{name} must be a nonnegative integer")
        weight = self.demonstration_weight
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not 0.0 <= float(weight) < float("inf"):
            raise ValueError("focus.demonstration_weight must be finite and nonnegative")
        if not self.enabled:
            if self.evaluation or self.failed_per_iteration or self.solved_per_iteration:
                raise ValueError("focus needs truth_sequences: the cases it draws from are the ones that file names")
            return
        if not self.evaluation:
            raise ValueError("focus needs evaluation: the starting adapter's greedy pass over the cases of truth_sequences")
        if self.failed_per_iteration < 1:
            raise ValueError("focus.failed_per_iteration must be at least 1")


@dataclass
class MinimalRLConfig:
    output_dir: str
    training_suite: str
    #: The adapter the run starts from, and the reference of the KL penalty: the supervised model.
    initial_adapter: str
    development_suite: str | None = None
    #: The first N development cases only (a smoke); null is the whole suite.
    development_limit: int | None = None
    #: ``{case: {"expert": bool, "truth": bool}}``: the split every evaluation of the run is also read by
    #: (``research.minimal_env.evaluate``).
    reference: str | None = None
    model_choice: str = "12b"
    #: One precision for the rollouts, the update and the evaluations of a run (bf16 unless set).
    load_in_4bit: bool = False
    local_files_only: bool = True
    #: The episode: both must be what the starting adapter was trained on.
    max_actions: int = DEFAULT_MAX_ACTIONS
    #: Versions whose tables the prompt shows (0: every version).
    table_versions: int = DEFAULT_TABLE_VERSIONS
    samples_per_root: int = 4
    seed: int = 20261116
    #: The sampling temperature; the update takes its log-probabilities at the same one.  No nucleus or top-k
    #: cut is ever applied: the update could not optimize the distribution such a cut samples from.
    temperature: float = 1.0
    roots_per_iteration: int | None = None
    #: ``psse_env.rl.sampling``: empty walks one seeded permutation; ``{"mode": "stratified", "key":
    #: "compound_class", "counts": {class: n}}`` fixes each class's share of every iteration.
    root_sampling: dict[str, Any] = field(default_factory=dict)
    #: :class:`FocusSettings`; when it names ``truth_sequences`` it replaces ``roots_per_iteration`` and
    #: ``root_sampling`` as the draw.
    focus: dict[str, Any] = field(default_factory=dict)
    #: The longest training row, in tokens (prompt and completion).
    max_length: int = 32768
    #: The longest prompt the policy is given (null: the process's ``RESEARCH_MAX_INPUT_TOKENS``).  A longer prompt
    #: is refused, never cut, and ends its episode undeclared.
    max_input_tokens: int | None = None
    advantage_method: str = "grpo"
    advantage_scale: str = "none"
    #: ``psse_env.minimal.scoring.RewardWeights``; empty is the first version's reward.
    reward: dict[str, Any] = field(default_factory=dict)
    #: ``psse_env.rl.update.UpdateSettings``.
    update: dict[str, Any] = field(default_factory=dict)
    #: ``psse_env.rl.rows.RowSettings``: complete groups only, prompt digest and truncation checks, preservation
    #: rows for the KL term.
    rows: dict[str, Any] = field(default_factory=dict)
    update_checkpoint_rows: int = 32
    #: Evaluate the starting adapter on the development suite before the first rollout, by this driver's own
    #: loader and precision.
    baseline_evaluation: bool = False
    #: A folder ``research.minimal_env.evaluate`` wrote for the starting adapter (``evaluation.json`` and
    #: ``episodes/``).  When it holds a finished evaluation under this run's settings, the baseline takes its
    #: episodes instead of running them again, so the supervised model has one number; when it holds none yet, the
    #: baseline is evaluated here.  An evaluation of another adapter or under other settings is refused.
    baseline_from: str | None = None
    evaluate_every_iteration: bool = True
    #: Evaluate the adapter of every N-th iteration only (the N-th, the 2N-th, ...; 1: every one).  An evaluation
    #: is about as long as a rollout, and a run of many small updates is read every few of them.
    evaluate_every: int = 1
    #: ``psse_env.minimal.forks.ForkSettings``: one-step forks of the current adapter's greedy episode on every
    #: drawn case, made in the rollout phase, and the rows the update takes from them.  Empty: none.
    forks: dict[str, Any] = field(default_factory=dict)
    #: Do not advance past an update that shows a stop condition of ``psse_env.rl.integrity`` (a prompt that is
    #: not the sampled one, a row without its sampling record, a nonfinite loss, duplicated or missing work).
    stop_on_integrity_failure: bool = True

    @classmethod
    def load(cls, path: str | Path) -> "MinimalRLConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = sorted(set(payload) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown config fields: {unknown}")
        config = cls(**payload)
        config.validate()
        return config

    # The base model, as the shared loaders of the RL series read it from a config.
    def _spec(self) -> Any:
        from psse_env.research_models import get_research_model_spec

        return get_research_model_spec(self.model_choice)

    @property
    def base_model(self) -> str:
        return str(self._spec().model_id)

    @property
    def base_revision(self) -> str:
        return str(self._spec().revision)

    @property
    def architecture(self) -> str:
        return str(self._spec().architecture)

    def validate(self) -> None:
        self._spec()
        if isinstance(self.samples_per_root, bool) or int(self.samples_per_root) < 2:
            raise ValueError("samples_per_root must be at least 2: an advantage is relative to the case's other samples")
        if int(self.max_actions) < 1 or int(self.max_length) < 1 or int(self.table_versions) < 0:
            raise ValueError("max_actions and max_length must be positive, table_versions nonnegative (0: every version)")
        if self.max_input_tokens is not None:
            if isinstance(self.max_input_tokens, bool) or not isinstance(self.max_input_tokens, int) or self.max_input_tokens <= 0:
                raise ValueError("max_input_tokens must be a positive integer (null: the process's own input limit)")
            if self.max_length < self.max_input_tokens:
                raise ValueError(f"max_length ({self.max_length}) is the longest training row, so it must hold the longest "
                                 f"prompt the policy is given (max_input_tokens {self.max_input_tokens})")
        if not float(self.temperature) > 0.0:
            raise ValueError("temperature must be positive")
        if "temperature" in self.update:
            raise ValueError("the update's temperature is the sampling temperature; set it at the top level only")
        if int(self.update_checkpoint_rows) <= 0:
            raise ValueError("update_checkpoint_rows must be positive")
        if self.baseline_evaluation and not self.development_suite:
            raise ValueError("baseline_evaluation needs a development suite to evaluate the starting adapter on")
        if self.baseline_from and not self.baseline_evaluation:
            raise ValueError("baseline_from names where the baseline evaluation may be taken from; set baseline_evaluation")
        limit = self.development_limit
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 1):
            raise ValueError("development_limit must be a positive integer (null: the whole development suite)")
        every = self.evaluate_every
        if isinstance(every, bool) or not isinstance(every, int) or every < 1:
            raise ValueError("evaluate_every must be a positive integer (1: every iteration)")
        ForkSettings.from_mapping(self.forks)
        group_advantages([0.0, 1.0], method=self.advantage_method, scale=self.advantage_scale)
        RewardWeights.from_mapping(self.reward)
        UpdateSettings(**self.update).validate()
        RootSampling.from_mapping(self.root_sampling).validate(self.roots_per_iteration)
        RowSettings.from_mapping(self.rows)
        focus = FocusSettings.from_mapping(self.focus)
        if focus.enabled and (self.root_sampling or self.roots_per_iteration is not None):
            raise ValueError("focus draws an iteration's cases itself: leave root_sampling empty and roots_per_iteration null")
        if focus.examples and RowSettings.from_mapping(self.rows).preservation == "none":
            # An update of truth-built examples alone would have no sampled row: nothing for the KL term to hold,
            # and nothing for the update's on-policy check to read.
            raise ValueError("focus with truth-built examples needs rows.preservation 'one' or 'all': the sampled rows of the "
                             "cases whose samples all scored the same")

    def row_settings(self, iteration: int) -> RowSettings:
        payload = {**self.rows, "seed": self.seed + int(iteration)}
        if self.update.get("clip_ratio") is not None:
            payload["require_sampling_records"] = True  # a clipped update needs every row's behaviour probabilities
        return RowSettings.from_mapping(payload)

    def focus_settings(self) -> FocusSettings:
        return FocusSettings.from_mapping(self.focus)

    def fork_settings(self) -> ForkSettings:
        return ForkSettings.from_mapping(self.forks)

    def evaluated(self, iteration: int) -> bool:
        """Whether the adapter of ``iteration`` (from 0) is evaluated: every ``evaluate_every``-th one."""
        return bool(self.development_suite) and bool(self.evaluate_every_iteration) and (int(iteration) + 1) % int(self.evaluate_every) == 0

    def update_settings(self, iteration: int) -> UpdateSettings:
        return UpdateSettings(**{**self.update, "seed": self.seed + int(iteration), "temperature": float(self.temperature)})

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------ the run's state


def initial_phase(config: MinimalRLConfig) -> str:
    return "baseline" if config.baseline_evaluation else "rollout"


def load_manifest(config: MinimalRLConfig) -> Manifest:
    path = Path(config.output_dir) / "manifest.json"
    manifest = Manifest.load(path, initial_adapter=config.initial_adapter, initial_phase=initial_phase(config))
    if not path.is_file():
        manifest.contract = DRIVER_CONTRACT
    elif manifest.contract != DRIVER_CONTRACT:
        raise RuntimeError(f"{path} is a run of {manifest.contract}, not of {DRIVER_CONTRACT}")
    return manifest


def load_suite(path: str | Path) -> list[dict[str, Any]]:
    roots = evaluate.load_suite(Path(path))
    if not roots:
        raise ValueError(f"suite {path} holds no case")
    return roots


def development_roots(config: MinimalRLConfig) -> list[dict[str, Any]]:
    """The cases every evaluation of the run is made on."""
    roots = load_suite(config.development_suite)
    return roots if config.development_limit is None else roots[: int(config.development_limit)]


def evaluation_differences(config: MinimalRLConfig, found: Mapping[str, Any]) -> dict[str, Any]:
    """Where an evaluation written by ``research.minimal_env.evaluate`` is not one of this run's starting adapter under
    its model, precision, episode and prompt limits (empty: it is)."""
    settings = found.get("settings") if isinstance(found.get("settings"), Mapping) else {}
    limits = apply_prompt_limits(config)
    mine = {**{key: value for key, value in environment_record(config).items() if key != "precision"},
            "load_in_4bit": bool(config.load_in_4bit), "max_input_tokens": int(limits["max_input_tokens"]),
            "max_new_tokens": int(limits["max_new_tokens"])}
    theirs = {key: settings.get(key) for key in mine}
    for key in ("max_input_tokens", "max_new_tokens"):
        theirs[key] = int(theirs[key]) if theirs[key] is not None else None
    different = {key: {"evaluation": theirs[key], "run": mine[key]} for key in mine if theirs[key] != mine[key]}
    if Path(str(found.get("adapter") or "")).resolve() != Path(config.initial_adapter).resolve():
        different["adapter"] = {"evaluation": found.get("adapter"), "run": config.initial_adapter}
    return different


def focus_cases(config: MinimalRLConfig, scenarios: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The training cases a focused run draws from, in the suite's order.

    ``failed`` and ``solved``: the cases with a truth-built sequence that the starting adapter's greedy pass left
    unrepaired and repaired; ``sequences``: every case's sequence.  A pass of another adapter or under other
    settings, a case with a sequence that the pass did not run, and a sequence for a case the suite lacks are
    refused: each would make the split something other than what the starting adapter does on these cases.
    """
    focus = config.focus_settings()
    sequences = truth_sequences.load(focus.truth_sequences)
    source = Path(str(focus.evaluation))
    if not (source / "evaluation.json").is_file():
        raise ValueError(f"focus.evaluation {source} holds no finished evaluation (evaluation.json)")
    found = json.loads((source / "evaluation.json").read_text(encoding="utf-8"))
    different = evaluation_differences(config, found)
    if different:
        raise ValueError(f"{source} is not an evaluation of this run's starting adapter under its settings: {different}")
    repaired = {str(entry["scenario_id"]): bool(entry["solved"]) for entry in found.get("episodes") or []}
    failed: list[dict[str, Any]] = []
    solved: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in scenarios:
        case = scenario_root(root)
        if case not in sequences:
            continue
        seen.add(case)
        if case not in repaired:
            raise ValueError(f"{source} holds no episode of {case}, which {focus.truth_sequences} names")
        (solved if repaired[case] else failed).append(dict(root))
    absent = sorted(set(sequences) - seen)
    if absent:
        raise ValueError(f"{len(absent)} cases of {focus.truth_sequences} are not in the training suite (first: {absent[0]})")
    if not failed:
        raise ValueError(f"{source} leaves no case of {focus.truth_sequences} unrepaired: there is nothing to focus on")
    return {"failed": failed, "solved": solved, "sequences": sequences}


def focused_draw(config: MinimalRLConfig, cases: Mapping[str, Any], iteration: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The unrepaired and the repaired cases of one iteration: slices of two seeded permutations that cycle."""
    focus = config.focus_settings()
    failed = iteration_roots(cases["failed"], iteration=iteration, roots_per_iteration=int(focus.failed_per_iteration), seed=config.seed)
    solved = (iteration_roots(cases["solved"], iteration=iteration, roots_per_iteration=int(focus.solved_per_iteration),
                              seed=config.seed + 1)
              if focus.solved_per_iteration and cases["solved"] else [])
    return failed, solved


def select_iteration_roots(config: MinimalRLConfig, scenarios: Sequence[Mapping[str, Any]], iteration: int) -> list[dict[str, Any]]:
    """The cases one iteration samples: slices of one seeded permutation, fixed counts per class, or (``focus``) the
    cases the starting adapter leaves unrepaired with a share of those it repairs."""
    if config.focus_settings().enabled:
        failed, solved = focused_draw(config, focus_cases(config, scenarios), iteration)
        return failed + solved
    sampling = RootSampling.from_mapping(config.root_sampling)
    if sampling.mode == "stratified":
        return stratified_roots(scenarios, iteration=iteration, counts=sampling.counts, key=sampling.key, seed=config.seed)
    return iteration_roots(scenarios, iteration=iteration, roots_per_iteration=config.roots_per_iteration, seed=config.seed)


def environment(config: MinimalRLConfig) -> MinimalDiagnosisEnv:
    return MinimalDiagnosisEnv(max_actions=int(config.max_actions), table_versions=int(config.table_versions) or None)


def environment_record(config: MinimalRLConfig) -> dict[str, Any]:
    return {"max_actions": int(config.max_actions), "table_versions": int(config.table_versions) or None,
            "precision": "4bit" if config.load_in_4bit else "bf16", "model_choice": config.model_choice}


# ------------------------------------------------------------------ the model's side


def sampling_policy_factory(config: MinimalRLConfig, adapter: str) -> Callable[[int], Any]:
    """``factory(seed)``: the adapter as a sampling policy whose draws are seeded with ``seed`` (one model, loaded once)."""
    from psse_env.dagger.research_policy_factory import GenerationOptions
    from psse_env.minimal.policy import minimal_policy_factory

    def factory(seed: int) -> Any:
        return minimal_policy_factory(
            adapter, base_model=config.base_model, base_revision=config.base_revision, load_in_4bit=config.load_in_4bit,
            local_files_only=config.local_files_only, architecture=config.architecture,
            generation=GenerationOptions(do_sample=True, temperature=float(config.temperature), top_p=1.0, top_k=None,
                                         seed=int(seed)))

    return factory


def greedy_policy_factory(config: MinimalRLConfig, adapter: str) -> Callable[[], Any]:
    return evaluate.model_policy_factory(str(adapter), model_choice=config.model_choice, load_in_4bit=config.load_in_4bit)


def prompt_functions(config: MinimalRLConfig, adapter: str, processor: Any) -> tuple[Callable[..., str], Callable[..., Any]]:
    """The policy's own prompt renderer and tokenizer, bound to the training processor (no model is loaded).

    ``render(observation)`` is the prompt text a stored step was sampled from, and ``prompt_ids(text)`` its
    tokens and their count, as the policy feeds them to the model.
    """
    from psse_env.dagger.release_factories import _ModelBundle
    from psse_env.minimal.policy import MinimalModelPolicy

    bundle = _ModelBundle(model=None, processor=processor, model_id=config.base_model, model_revision=config.base_revision,
                          adapter_snapshot_path=str(adapter))
    policy = MinimalModelPolicy(bundle, architecture=config.architecture)

    def render(observation: Mapping[str, Any]) -> str:
        return policy.render_prompt(observation)[0]

    return render, policy.prompt_token_ids


def target_function(config: MinimalRLConfig, adapter: str, processor: Any) -> Callable[[Mapping[str, Any], Mapping[str, Any]], list[int]]:
    """``render_target(observation, action)``: the tokens the supervised pass taught for ``action`` after ``observation``.

    The action is appended to the observation's messages as the assistant turn of a supervised row
    (``psse_env.minimal.observation.target_call``) and rendered through the chat template; the completion is what
    that render adds to the generation prompt.  It is read back with the policy's own parser and schema check and
    must be the action again; one that is not raises ``ValueError``.
    """
    from psse_env.dagger.release_factories import _ModelBundle, _validated_generated_action
    from psse_env.minimal.policy import MinimalModelPolicy
    from psse_env.rl.targets import TargetError, render_target_completion
    from psse_env.sft.gates import GateError

    bundle = _ModelBundle(model=None, processor=processor, model_id=config.base_model, model_revision=config.base_revision,
                          adapter_snapshot_path=str(adapter))
    policy = MinimalModelPolicy(bundle, architecture=config.architecture)

    def render_target(observation: Mapping[str, Any], action: Mapping[str, Any]) -> list[int]:
        wanted = {"tool": str(action["tool"]), "arguments": dict(action.get("arguments") or {})}
        messages, tools, parameters = policy.render_messages(observation)
        rendered = policy.render_prompt(observation)[0]
        target = render_target_completion(processor, messages=messages, tools=tools, rendered_prompt=rendered, action=wanted)
        try:
            read_back = _validated_generated_action(target.text, parameters)
        except GateError as exc:
            raise TargetError(f"the rendered action does not parse back: {exc}") from exc
        if read_back != wanted:
            raise TargetError(f"the rendered action parses to {read_back}, not {wanted}")
        return [int(value) for value in target.completion_ids]

    return render_target


def fork_prefix_check(config: MinimalRLConfig, adapter: str, policy: Any) -> Callable[..., bool] | None:
    """``check(observation, action, forced_ids)``: whether the tokens a forced call was started with are the first
    tokens of that call as the supervised pass rendered it.  ``None`` for a policy that holds no processor.

    A forced call's name is given to the model as text (``psse_env.minimal.policy.forced_call_prefix``); were its
    tokens not the ones the model was trained to write the call with, the arguments would be written after a
    prefix the model never saw.  A rollout's fork summary counts how many forced calls pass.
    """
    processor = getattr(getattr(policy, "_bundle", None), "processor", None)
    if processor is None:
        return None
    render_target = target_function(config, adapter, processor)

    def check(observation: Mapping[str, Any], action: Mapping[str, Any], forced_ids: Sequence[int]) -> bool:
        target = render_target(model_view(observation), action)
        return [int(value) for value in target[: len(forced_ids)]] == [int(value) for value in forced_ids]

    return check


def model_view(observation: Mapping[str, Any]) -> dict[str, Any]:
    """An environment observation as the model policy records the one it decoded from (marked as a whole history)."""
    from psse_env.minimal.policy import MinimalModelPolicy

    return MinimalModelPolicy._whole(observation)


def _release_models() -> None:
    try:
        from psse_env.dagger.research_policy_factory import clear_research_policy_cache

        clear_research_policy_cache()
    except Exception:  # pragma: no cover - the model stack is absent in the pure-Python tests
        pass
    _release_gpu()


# ------------------------------------------------------------------ rollout


def rollout_summary(store: TrajectoryStore) -> dict[str, Any]:
    """What an iteration's sampled episodes hold: the signal for the update, the outcomes, and how peaked the sampling was."""
    totals: Counter = Counter()
    by_class: dict[str, Counter] = defaultdict(Counter)
    rewards: list[float] = []
    actions = steps = peaked = 0
    for _root, group in store.iter_groups():
        solved = [bool(trajectory.evaluation["score"]["solved"]) for trajectory in group]
        totals["groups"] += 1
        totals["groups_with_reward_spread"] += int(reward_spread(group) > 1e-9)
        totals["groups_with_distinct_samples"] += int(distinct_trajectories(group) > 1)
        # The contrast an update can learn a repair from: one case, some samples repaired and some not.
        totals["groups_with_repaired_and_unrepaired_samples"] += int(any(solved) and not all(solved))
        totals["groups_with_every_sample_repaired"] += int(all(solved))
        totals["groups_with_no_sample_repaired"] += int(not any(solved))
        for trajectory in group:
            score = trajectory.evaluation["score"]
            declared = score.get("declared")
            outcome = trajectory.evaluation.get("terminal_outcome")
            totals["episodes"] += 1
            totals["solved"] += int(bool(score["solved"]))
            totals["solved_and_declared_resolved"] += int(bool(score["success"]))
            totals["false_resolved"] += int(declared == "resolved" and not score["solved"])
            totals["declared_unresolved"] += int(declared == "unresolved")
            totals["budget_exhausted"] += int(outcome == "budget_exhausted")
            totals["prompt_limit"] += int(outcome == OUTCOME_PROMPT_LIMIT)
            totals["episodes_with_a_damaged_healthy_object"] += int(score["physical"]["damaged"] > 0)
            totals["tool_errors"] += int(score.get("tool_errors") or 0)
            totals["unpaired_trajectories"] += int(not trajectory.pairing_ok)
            actions += int(score.get("actions") or 0)
            rewards.append(trajectory.total_reward)
            for entry in trajectory.evaluation.get("trace") or []:
                if entry.get("error_code"):
                    totals[f"error:{entry['error_code']}"] += 1
            for step in trajectory.steps:
                if step.token_probabilities:
                    steps += 1
                    peaked += int(min(step.token_probabilities) >= 0.99)
            cell = by_class[str(score.get("class"))]
            cell["episodes"] += 1
            cell["solved"] += int(bool(score["solved"]))
    episodes = int(totals["episodes"])
    return {
        "totals": dict(sorted(totals.items())),
        "mean_reward": (sum(rewards) / episodes) if episodes else None,
        "mean_actions": (actions / episodes) if episodes else None,
        "sampled_steps_with_probabilities": steps,
        "fraction_of_steps_with_every_token_above_0_99": (peaked / steps) if steps else None,
        "by_class": {name: dict(sorted(cell.items())) for name, cell in sorted(by_class.items())},
    }


class _ObservedScript:
    """A fixed list of actions as a policy that keeps the observation each one was chosen on."""

    def __init__(self, actions: Sequence[Mapping[str, Any]]) -> None:
        self._actions = [{"tool": str(action["tool"]), "arguments": dict(action.get("arguments") or {})} for action in actions]
        self.observations: list[dict[str, Any]] = []

    def act(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        self.observations.append(model_view(observation))
        # A sequence that ran out before the episode ended has not repaired the case: say so.
        return self._actions.pop(0) if self._actions else {"tool": "finish", "arguments": {"outcome": "unresolved"}}


def example_trajectory(config: MinimalRLConfig, root: Mapping[str, Any], actions: Sequence[Mapping[str, Any]],
                       env: MinimalDiagnosisEnv, weights: RewardWeights) -> Trajectory:
    """A truth-built sequence replayed on ``root``: its actions, the observation each was taken on, its score and reward.

    The record is a trajectory with no sampled token: ``steps`` hold the observation and the action alone.
    """
    del config
    script = _ObservedScript(actions)
    episode = run_episode(env, root, script, weights=weights, keep_observations=False)
    trace = minimal_rollout._trace(episode)
    steps = [StepRecord(step=int(entry["step"]), model_observation=observation, prompt_tokens=0, original_prompt_tokens=0,
                        completion_token_ids=[], text="", action=dict(entry["action"]), tool=str(entry["action"].get("tool") or ""),
                        execution_status=entry["execution_status"], error_code=entry["error_code"])
             for entry, observation in zip(trace, script.observations)]
    score = episode["score"]
    return Trajectory(
        root=scenario_root(root), family=scenario_family(root), sample_index=EXAMPLE_SAMPLE, episode_seed=0, steps=steps,
        reward={**episode["reward"], "calls": int(score["actions"])},
        evaluation={"truth_audited_task_success": bool(score["success"]), "terminal_outcome": episode["outcome"], "trace": trace,
                    "score": score, "seconds": episode["seconds"]},
        pairing_ok=len(steps) == len(trace), pairing_note="a truth-built sequence, replayed",
        policy_identity={"policy": "truth_built_sequence"}, lifecycle=[], contract=EXAMPLE_CONTRACT)


def truth_built_examples(config: MinimalRLConfig, drawn: Sequence[Mapping[str, Any]], sequences: Mapping[str, Any],
                         store: TrajectoryStore, out: Path, weights: RewardWeights) -> dict[str, Any]:
    """For every drawn case none of whose samples is repaired: its truth-built sequence, replayed and kept under
    ``out/demonstrations``.  A replay that does not repair the case here is not kept (the count says how many).

    A case whose example is already there is not replayed again, so a closing call that runs twice writes once.
    """
    started = time.perf_counter()
    kept = TrajectoryStore(out / "demonstrations")
    counts: Counter = Counter()
    not_repaired: list[str] = []
    env: MinimalDiagnosisEnv | None = None
    for root in drawn:
        case = scenario_root(root)
        if any(bool(trajectory.evaluation["score"]["solved"]) for trajectory in store.read_group(case)):
            continue
        counts["cases_without_a_repaired_sample"] += 1
        if case not in sequences:
            counts["without_a_sequence"] += 1
            continue
        if kept.has_group(case, 1):
            counts["examples"] += 1
            continue
        env = env if env is not None else environment(config)
        example = example_trajectory(config, root, sequences[case], env, weights)
        if not example.evaluation["score"]["solved"] or not example.pairing_ok:
            not_repaired.append(case)
            continue
        kept.write_group(case, [example])
        counts["examples"] += 1
        counts["examples_replayed_now"] += 1
    return {**{name: int(counts[name]) for name in ("cases_without_a_repaired_sample", "examples", "examples_replayed_now",
                                                     "without_a_sequence")},
            "replay_did_not_repair": not_repaired, "seconds": round(time.perf_counter() - started, 1)}


def phase_rollout(config: MinimalRLConfig, manifest: Manifest, *, progress: Callable[[Mapping[str, Any]], None] | None = None,
                  shard: tuple[int, int] | None = None) -> dict[str, Any]:
    """Sample the iteration's cases.  A shard samples its slice and writes no summary; the unsharded call finds
    every finished group in the store, samples what is left and closes the phase.  In a focused run the closing
    call also replays the truth-built sequence of every drawn case none of whose samples is repaired."""
    started = time.perf_counter()
    out = iteration_dir(config, manifest.iteration)
    out.mkdir(parents=True, exist_ok=True)
    sampling = RootSampling.from_mapping(config.root_sampling)
    focus = config.focus_settings()
    suite = load_suite(config.training_suite)
    focused: dict[str, Any] | None = None
    if focus.enabled:
        cases = focus_cases(config, suite)
        drawn_failed, drawn_solved = focused_draw(config, cases, manifest.iteration)
        drawn = drawn_failed + drawn_solved
        focused = {"available": {"left_unrepaired_by_the_starting_adapter": len(cases["failed"]),
                                 "repaired_by_the_starting_adapter": len(cases["solved"])},
                   "sequences": cases["sequences"], "failed": drawn_failed, "solved": drawn_solved}
        del cases
    else:
        drawn = select_iteration_roots(config, suite, manifest.iteration)
    del suite  # the drawn cases are all the phase needs of it
    if shard is None:
        (out / "roots.json").write_text(json.dumps([scenario_root(root) for root in drawn], indent=1) + "\n", encoding="utf-8")
    mine = shard_scenarios(drawn, shard)
    limits = apply_prompt_limits(config)
    weights = RewardWeights.from_mapping(config.reward)
    store = TrajectoryStore(out / "trajectories")
    pending = [root for root in mine if not store.has_group(scenario_root(root), int(config.samples_per_root))]
    forks = config.fork_settings()
    fork_store = ForkStore(out / "forks")
    unforked = [root for root in mine if not fork_store.has(scenario_root(root))] if forks.enabled else []
    forked = 0
    if pending or unforked:
        env = environment(config)
        try:
            if pending:
                rollout_roots(pending, store_dir=out / "trajectories", samples=int(config.samples_per_root),
                              base_seed=config.seed + manifest.iteration, env_factory=lambda: env,
                              policy_factory=sampling_policy_factory(config, manifest.current_adapter), weights=weights,
                              progress=progress)
            if unforked:
                # The same model, decoding greedily: the episode an evaluation would run, and its one-step forks.
                forked = fork_roots(
                    unforked, store=fork_store, env_factory=lambda: env,
                    policy_factory=greedy_policy_factory(config, manifest.current_adapter), weights=weights, settings=forks,
                    argument_policy_factory=(sampling_policy_factory(config, manifest.current_adapter)
                                             if forks.argument_samples else None),
                    base_seed=config.seed + manifest.iteration,
                    canonical_factory=lambda policy: fork_prefix_check(config, manifest.current_adapter, policy),
                    progress=progress)
        finally:
            _release_models()
    result: dict[str, Any] = {"iteration": manifest.iteration, "adapter": manifest.current_adapter, "roots": len(mine),
                              "groups_sampled": len(pending), "samples_per_root": int(config.samples_per_root),
                              "seconds": round(time.perf_counter() - started, 1)}
    if forks.enabled:
        result["cases_forked"] = forked
    if shard is not None:
        return {**result, "shard": f"{shard[0]}/{shard[1]}"}
    result.update(
        store=rollout_summary(store), sampling={"temperature": float(config.temperature), "top_p": 1.0, "top_k": None},
        environment=environment_record(config), prompt_limits=limits, reward_weights=asdict(weights))
    if sampling.mode == "stratified":
        strata = Counter(root_stratum(root, sampling.key) for root in drawn)
        result["root_sampling"] = {"mode": sampling.mode, "key": sampling.key, "strata": dict(sorted(strata.items()))}
    if focused is not None:
        result["focus"] = focus_summary(store, focused["available"], focused["failed"], focused["solved"])
        if focus.examples:
            result["truth_built_examples"] = truth_built_examples(config, drawn, focused["sequences"], store, out, weights)
    if forks.enabled:
        result["forks"] = fork_summary(fork_store, drawn)
    _write_json(out / "rollout_summary.json", result)
    return result


def fork_summary(store: ForkStore, drawn: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """What the one-step forks of an iteration's drawn cases found (``psse_env.minimal.forks.summarize``)."""
    records = [store.read(scenario_root(root))["record"] for root in drawn if store.has(scenario_root(root))]
    return {"cases_drawn": len(drawn), "cases_forked": len(records), **summarize_forks(records)}


def focus_summary(store: TrajectoryStore, available: Mapping[str, Any], drawn_failed: Sequence[Mapping[str, Any]],
                  drawn_solved: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """What a focused iteration's samples did on the cases the starting adapter left unrepaired and on those it repaired."""

    def read(roots: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        solved = [[bool(trajectory.evaluation["score"]["solved"]) for trajectory in store.read_group(scenario_root(root))]
                  for root in roots]
        return {"cases": len(roots), "episodes": sum(len(group) for group in solved),
                "episodes_repaired": sum(sum(group) for group in solved),
                "cases_with_a_repaired_sample": sum(1 for group in solved if any(group)),
                "cases_with_every_sample_repaired": sum(1 for group in solved if group and all(group)),
                "cases_with_a_repaired_and_an_unrepaired_sample": sum(1 for group in solved if any(group) and not all(group)),
                "cases_with_no_sample_repaired": sum(1 for group in solved if not any(group))}

    return {"available": dict(available),
            "left_unrepaired_by_the_starting_adapter": read(drawn_failed),
            "repaired_by_the_starting_adapter": read(drawn_solved)}


# ------------------------------------------------------------------ update


def score_groups(config: MinimalRLConfig, store: TrajectoryStore) -> tuple[list[tuple[list[Trajectory], GroupAdvantages]], dict[str, Any]]:
    """Every stored group with its advantages, and how much signal the batch carries."""
    groups: list[tuple[list[Trajectory], GroupAdvantages]] = []
    by_root: dict[str, GroupAdvantages] = {}
    for root, trajectories in store.iter_groups():
        advantages = group_advantages([trajectory.total_reward for trajectory in trajectories],
                                      method=config.advantage_method, scale=config.advantage_scale, root=root)
        groups.append((trajectories, advantages))
        by_root[root] = advantages
    summary: dict[str, Any] = dict(summarize_groups(by_root))
    episodes = sum(len(trajectories) for trajectories, _ in groups)
    solved = sum(1 for trajectories, _ in groups for trajectory in trajectories if trajectory.evaluation["score"]["solved"])
    summary.update(
        episodes=episodes, solved_rate=(solved / episodes) if episodes else None,
        unpaired_trajectories=sum(1 for trajectories, _ in groups for trajectory in trajectories if not trajectory.pairing_ok),
        advantage={"method": config.advantage_method, "scale": config.advantage_scale})
    return groups, summary


def stored_examples(out: Path, groups: Sequence[tuple[Sequence[Trajectory], GroupAdvantages]]) -> list[tuple[Trajectory, float]]:
    """The truth-built examples of an iteration with how far each one's reward lies above the mean of its case's samples.

    Only the example of a case whose stored group holds no repaired sample: with a repaired sample the group has
    its own contrast, and the example is left out even if one was written.
    """
    if not (out / "demonstrations" / "groups").is_dir():
        return []
    by_root = {advantages.root: (trajectories, advantages) for trajectories, advantages in groups}
    found: list[tuple[Trajectory, float]] = []
    for root, stored in TrajectoryStore(out / "demonstrations").iter_groups():
        if root not in by_root or not stored:
            continue
        trajectories, advantages = by_root[root]
        if any(bool(trajectory.evaluation["score"]["solved"]) for trajectory in trajectories):
            continue
        rewards = [float(value) for value in advantages.rewards]
        found.append((stored[0], stored[0].total_reward - sum(rewards) / len(rewards)))
    return found


def example_rows(examples: Sequence[tuple[Trajectory, float]], *, weight: float, render_prompt: Callable[..., str],
                 prompt_token_ids: Callable[..., Any], render_target: Callable[..., Sequence[int]], max_length: int,
                 required_side_inputs: Sequence[str], settings: RowSettings) -> tuple[list[TrainingRow], dict[str, Any]]:
    """The supervised rows of the truth-built examples: one per action the environment carried out.

    A row is the action on the prompt the model would have read there, rendered as the supervised pass taught it,
    with weight ``weight`` times the example's reward above the mean of its case's samples.  An example that is
    not above that mean gives no row.  A prompt over the policy's input limit gives no row either (the model would
    have been refused that prompt); an action that cannot be rendered is counted for the integrity check.
    """
    rows: list[TrainingRow] = []
    report: dict[str, Any] = {"examples": len(examples), "examples_with_rows": 0, "examples_not_above_their_samples": 0,
                              "rows": 0, "tokens": 0, "unrenderable": 0, "prompt_over_the_input_limit": 0,
                              "prompt_truncated_rows": 0, "prompt_truncated_rows_rejected": 0, "weight": float(weight),
                              "row_weights": None}
    weights: list[float] = []
    for example, gain in examples:
        term = float(weight) * float(gain)
        if not term > 0.0:
            report["examples_not_above_their_samples"] += 1
            continue
        built: list[TrainingRow] = []
        for step in example.steps:
            if step.execution_status != "success":
                continue
            prompt_ids, original = prompt_token_ids(render_prompt(step.model_observation))
            if int(original) != len(prompt_ids):
                report["prompt_over_the_input_limit"] += 1
                continue
            try:
                target_ids = [int(value) for value in render_target(step.model_observation, step.action)]
            except ValueError:
                target_ids = []
            if not target_ids:
                report["unrenderable"] += 1
                continue
            row = build_step_row(root=example.root, sample_index=example.sample_index, step=step.step, advantage=0.0,
                                 prompt_ids=prompt_ids, completion_ids=target_ids, max_length=max_length,
                                 required_side_inputs=required_side_inputs, lifecycle_term=term, supervised=True)
            if row.prompt_truncated:
                report["prompt_truncated_rows"] += 1
                if settings.reject_truncated_prompts:
                    report["prompt_truncated_rows_rejected"] += 1
                    continue
            row.trajectory_tokens = row.completion_tokens
            built.append(row)
        if built:
            report["examples_with_rows"] += 1
            weights.append(term)
        rows.extend(built)
    report["rows"] = len(rows)
    report["tokens"] = sum(row.completion_tokens for row in rows)
    if weights:
        report["row_weights"] = {"min": min(weights), "mean": sum(weights) / len(weights), "max": max(weights)}
    return rows, report


def stored_forks(out: Path, settings: ForkSettings) -> list[dict[str, Any]]:
    """The forked cases of an iteration with a state where a branch returned more than the greedy episode by more
    than ``settings.min_gain``: each case's stored payload, its ``better`` cut to those states."""
    found: list[dict[str, Any]] = []
    for _case, payload in ForkStore(out / "forks").iter():
        better = [item for item in payload["record"].get("better") or [] if float(item["gain"]) > float(settings.min_gain)]
        if better:
            found.append({**payload, "better": better})
    return found


def fork_rows(payloads: Sequence[Mapping[str, Any]], *, weight: float, render_prompt: Callable[..., str],
              prompt_token_ids: Callable[..., Any], render_target: Callable[..., Sequence[int]], max_length: int,
              required_side_inputs: Sequence[str], settings: RowSettings) -> tuple[list[TrainingRow], dict[str, Any]]:
    """The supervised rows of an iteration's forks: one per state where a forced branch returned more.

    A row is the branch's forced call on the prompt the model read at that state, rendered as the supervised pass
    taught it, with weight ``weight`` times how far the branch's return lies above the greedy episode's.  A prompt
    over the policy's input limit gives no row.  A call that does not render as the student emits it gives no row
    and is counted (``unrenderable``): the model wrote the call itself, so the count should be zero, and it is
    read, not a stop condition, because nothing wrong is trained by leaving the row out.
    """
    rows: list[TrainingRow] = []
    report: dict[str, Any] = {"cases": len(payloads), "candidates": 0, "rows": 0, "tokens": 0, "unrenderable": 0,
                              "prompt_over_the_input_limit": 0, "prompt_truncated_rows": 0, "prompt_truncated_rows_rejected": 0,
                              "weight": float(weight), "row_weights": None, "rows_by_tool": {},
                              "rows_whose_branch_repairs": 0, "rows_within_one_action_cost": 0}
    weights: list[float] = []
    for payload in payloads:
        case = str(payload["case"])
        for item in payload["better"]:
            report["candidates"] += 1
            term = float(weight) * float(item["gain"])
            observation = model_view(payload["observations"][str(item["state"])])
            prompt_ids, original = prompt_token_ids(render_prompt(observation))
            if int(original) != len(prompt_ids):
                report["prompt_over_the_input_limit"] += 1
                continue
            try:
                target_ids = [int(value) for value in render_target(observation, item["action"])]
            except ValueError:
                target_ids = []
            if not target_ids:
                report["unrenderable"] += 1
                continue
            row = build_step_row(root=case, sample_index=FORK_SAMPLE, step=int(item["state"]) + 1, advantage=0.0,
                                 prompt_ids=prompt_ids, completion_ids=target_ids, max_length=max_length,
                                 required_side_inputs=required_side_inputs, lifecycle_term=term, supervised=True)
            if row.prompt_truncated:
                report["prompt_truncated_rows"] += 1
                if settings.reject_truncated_prompts:
                    report["prompt_truncated_rows_rejected"] += 1
                    continue
            row.trajectory_tokens = row.completion_tokens
            rows.append(row)
            weights.append(term)
            tool = str(item["tool"])
            report["rows_by_tool"][tool] = int(report["rows_by_tool"].get(tool, 0)) + 1
            report["rows_whose_branch_repairs"] += int(bool(item["solved"]) and not payload["record"]["own"]["solved"])
            report["rows_within_one_action_cost"] += int(float(item["gain"]) <= 0.05)
    report["rows"] = len(rows)
    report["tokens"] = sum(row.completion_tokens for row in rows)
    report["rows_by_tool"] = dict(sorted(report["rows_by_tool"].items()))
    if weights:
        report["row_weights"] = {"min": min(weights), "mean": sum(weights) / len(weights), "max": max(weights)}
    return rows, report


def _count_fork_rows(rows_report: dict[str, Any], report: Mapping[str, Any]) -> None:
    """The forks' rows in the counts the row builder keeps; a call that did not render is reported, not a stop."""
    rows_report["rows"] = int(rows_report.get("rows") or 0) + int(report["rows"])
    rows_report["completion_tokens"] = int(rows_report.get("completion_tokens") or 0) + int(report["tokens"])
    rows_report["supervised_rows"] = int(rows_report.get("supervised_rows") or 0) + int(report["rows"])
    rows_report["supervised_tokens"] = int(rows_report.get("supervised_tokens") or 0) + int(report["tokens"])
    rows_report["prompt_truncated_rows"] = int(rows_report.get("prompt_truncated_rows") or 0) + int(report["prompt_truncated_rows"])
    rows_report["prompt_truncated_rows_rejected"] = (int(rows_report.get("prompt_truncated_rows_rejected") or 0)
                                                     + int(report["prompt_truncated_rows_rejected"]))
    rows_report["one_step_forks"] = dict(report)


def _count_example_rows(rows_report: dict[str, Any], report: Mapping[str, Any]) -> None:
    """The examples' rows in the counts the row builder keeps and the integrity check reads."""
    rows_report["rows"] = int(rows_report.get("rows") or 0) + int(report["rows"])
    rows_report["completion_tokens"] = int(rows_report.get("completion_tokens") or 0) + int(report["tokens"])
    rows_report["supervised_rows"] = int(rows_report.get("supervised_rows") or 0) + int(report["rows"])
    rows_report["supervised_tokens"] = int(rows_report.get("supervised_tokens") or 0) + int(report["tokens"])
    # A truth-built action that does not render as the student emits it is a stop condition, like a labelled choice.
    rows_report["supervised_rows_unrenderable"] = int(rows_report.get("supervised_rows_unrenderable") or 0) + int(report["unrenderable"])
    rows_report["prompt_truncated_rows"] = int(rows_report.get("prompt_truncated_rows") or 0) + int(report["prompt_truncated_rows"])
    rows_report["prompt_truncated_rows_rejected"] = (int(rows_report.get("prompt_truncated_rows_rejected") or 0)
                                                     + int(report["prompt_truncated_rows_rejected"]))
    rows_report["truth_built_examples"] = dict(report)


def phase_update(config: MinimalRLConfig, manifest: Manifest, *, progress: Callable[[Mapping[str, Any]], None] | None = None) -> dict[str, Any]:
    out = iteration_dir(config, manifest.iteration)
    marker = out / MARKER_FILE
    if marker.is_file():
        raise IntegrityError(f"iteration {manifest.iteration}'s update failed its integrity checks ({marker}); read "
                             "update_report.json, and remove the marker only once the cause is understood")
    store = TrajectoryStore(out / "trajectories")
    groups, score_summary = score_groups(config, store)
    _write_json(out / "advantages.json", {
        **score_summary,
        "by_root": {adv.root: {"rewards": list(adv.rewards), "advantages": list(adv.advantages), "informative": adv.informative}
                    for _, adv in groups},
    })
    next_adapter = out / "adapter"
    report: dict[str, Any] = {"contract": DRIVER_CONTRACT, "iteration": manifest.iteration, "adapter_in": manifest.current_adapter,
                              "scores": score_summary, "environment": environment_record(config)}
    # The update renders every stored step again and takes the prompt the policy was given: under the same limit.
    report["prompt_limits"] = apply_prompt_limits(config)
    informative_paired = any(
        trajectory.pairing_ok for trajectories, adv in groups if adv.informative for trajectory in trajectories)
    focus = config.focus_settings()
    examples = stored_examples(out, groups) if focus.examples else []
    forks = config.fork_settings()
    forked = stored_forks(out, forks) if forks.enabled else []
    if not informative_paired and not examples and not forked:
        report["update"] = None
        report["note"] = ("no case whose paired samples scored differently" + (" and no truth-built example" if focus.examples else "")
                          + (" and no forced branch above the greedy episode" if forks.enabled else "")
                          + ": the adapter is carried over unchanged")
        _carry_adapter_over(manifest.current_adapter, next_adapter)
        report["adapter_out"] = str(next_adapter)
        report["integrity"] = update_integrity(report)
        _write_json(out / "update_report.json", report)
        return report
    settings = config.update_settings(manifest.iteration)
    checkpoint = UpdateCheckpoint(directory=out / "update_checkpoint", save_model=save_trained_adapter,
                                  every_rows=int(config.update_checkpoint_rows))
    resume_dir = checkpoint.model_dir()
    start_adapter = str(resume_dir) if resume_dir is not None else manifest.current_adapter
    # The KL reference is the run's starting adapter, never the adapter of the iteration before.
    reference = config.initial_adapter if settings.beta_kl > 0.0 else None
    report["kl_reference"] = reference
    model, processor, side_inputs = load_trainable_model(config, start_adapter, reference_adapter=reference)
    try:
        render, prompt_ids = prompt_functions(config, manifest.current_adapter, processor)
        row_settings = config.row_settings(manifest.iteration)
        rows, rows_report = build_rows(groups, render_prompt=render, prompt_token_ids=prompt_ids,
                                       max_length=int(config.max_length), required_side_inputs=side_inputs,
                                       settings=row_settings)
        render_target = target_function(config, manifest.current_adapter, processor) if examples or forked else None
        if examples:
            taught, taught_report = example_rows(
                examples, weight=float(focus.demonstration_weight), render_prompt=render, prompt_token_ids=prompt_ids,
                render_target=render_target, max_length=int(config.max_length),
                required_side_inputs=side_inputs, settings=row_settings)
            rows.extend(taught)
            _count_example_rows(rows_report, taught_report)
        if forked:
            taught, taught_report = fork_rows(
                forked, weight=float(forks.weight), render_prompt=render, prompt_token_ids=prompt_ids,
                render_target=render_target, max_length=int(config.max_length),
                required_side_inputs=side_inputs, settings=row_settings)
            rows.extend(taught)
            _count_fork_rows(rows_report, taught_report)
        report["rows"] = rows_report
        report["resumed_from_checkpoint"] = str(resume_dir) if resume_dir is not None else None
        if not rows:
            report["update"] = None
            report["note"] = "no training row after the pairing and rendering checks"
            _carry_adapter_over(manifest.current_adapter, next_adapter)
        elif all(row.supervised for row in rows):
            # Every sampled row was refused: the examples alone would be an update with nothing sampled in it.
            report["update"] = None
            report["note"] = ("no sampled row after the pairing and rendering checks: the truth-built examples and the "
                              "forks' rows alone are not applied")
            _carry_adapter_over(manifest.current_adapter, next_adapter)
        else:
            optimizer = None
            if settings.carry_optimizer:
                # The state before this update: a resumed update loads its checkpoint's state over it.
                carried = _carried_optimizer_file(config, manifest.iteration)
                carried_state = load_optimizer_state(carried) if carried is not None else None
                optimizer = build_optimizer(model, settings, carried_state)
                report["optimizer"] = {"carried_from": str(carried) if carried is not None else None,
                                       "steps_before": optimizer_steps_taken(carried_state),
                                       "continuity_since_iteration": _continuity_start(config, manifest.iteration)}
            update = policy_gradient_update(model, rows, settings, optimizer=optimizer, progress=progress, checkpoint=checkpoint)
            report["update"] = update.as_dict()
            temporary = out / "adapter.tmp"
            if temporary.exists():
                shutil.rmtree(temporary)
            save_trained_adapter(model, temporary)
            if next_adapter.exists():
                shutil.rmtree(next_adapter)
            temporary.rename(next_adapter)
            if optimizer is not None:
                # Before the checkpoint goes: until then a restart resumes from it and writes this again.
                save_optimizer_state(optimizer, out / OPTIMIZER_FILE)
                report["optimizer"]["steps_after"] = optimizer_steps_taken(optimizer.state_dict())
            shutil.rmtree(checkpoint.directory, ignore_errors=True)
    finally:
        del model
        _release_gpu()
    report["adapter_out"] = str(next_adapter)
    report["integrity"] = update_integrity(report)
    _write_json(out / "update_report.json", report)
    if config.stop_on_integrity_failure and not report["integrity"]["ok"]:
        _write_json(marker, {"iteration": manifest.iteration, "failures": report["integrity"]["failures"]})
        raise IntegrityError(
            f"iteration {manifest.iteration}'s update shows a stop condition: {'; '.join(report['integrity']['failures'])}")
    return report


# ------------------------------------------------------------------ evaluation


def paired_change(records: Sequence[Mapping[str, Any]], earlier: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Case by case against an earlier evaluation of the same suite: what is repaired now and was not, and the reverse."""
    before = {str(record["scenario_id"]): bool(record["score"]["solved"]) for record in earlier}
    now = {str(record["scenario_id"]): bool(record["score"]["solved"]) for record in records}
    shared = sorted(set(before) & set(now))
    return {
        "cases_in_both": len(shared), "cases_in_one_only": len(set(before) ^ set(now)),
        "solved_before": sum(1 for case in shared if before[case]), "solved_now": sum(1 for case in shared if now[case]),
        "gained": [case for case in shared if now[case] and not before[case]],
        "lost": [case for case in shared if before[case] and not now[case]],
    }


def _stored_records(directory: Path, roots: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]] | None:
    """A finished evaluation's episodes, or ``None`` where there is none."""
    if not (directory / "evaluation.json").is_file():
        return None
    records, missing = evaluate.read_episodes(roots, directory / "evaluation" / "episodes")
    return None if missing else records


def greedy_evaluation(config: MinimalRLConfig, *, adapter: str, out: Path, shard: tuple[int, int] | None,
                      head: Mapping[str, Any], progress: Callable[[Mapping[str, Any]], None] | None = None) -> dict[str, Any]:
    """``adapter``, greedy, on the development suite: one file per case under ``out/evaluation/episodes``.

    A shard evaluates its slice and returns its count; the unsharded call finishes what is left and writes
    ``out/evaluation.json`` in the form of ``research.minimal_env.evaluate``, with the case-by-case change against
    the run's baseline where the run has one.
    """
    roots = development_roots(config)
    limits = apply_prompt_limits(config)
    try:
        ran = evaluate.evaluate_cases(roots, policy_factory=greedy_policy_factory(config, adapter),
                                      env_factory=lambda: environment(config),
                                      episodes_dir=out / "evaluation" / "episodes", shard=shard, progress=progress)
    finally:
        _release_models()
    if shard is not None:
        return {"shard": f"{shard[0]}/{shard[1]}", "evaluated": ran}
    records, missing = evaluate.read_episodes(roots, out / "evaluation" / "episodes")
    if missing:
        raise RuntimeError(f"{len(missing)} cases have no episode after the evaluation (first: {missing[0]})")
    reference = json.loads(Path(config.reference).read_text(encoding="utf-8")) if config.reference else None
    payload = evaluate.evaluation_payload(
        records, suite=str(config.development_suite), adapter=str(adapter), reference=reference,
        settings={**environment_record(config), "decoding": "greedy", "prompt_limits": limits})
    payload.update(head)
    baseline = None if head.get("baseline") else _stored_records(Path(config.output_dir) / "baseline", roots)
    if baseline is not None:
        payload["against_baseline"] = paired_change(records, baseline)
    _write_json(out / "evaluation.json", payload)
    summary = payload["summary"]
    brief = {**head, "adapter": str(adapter), **summary["totals"], "mean_actions": summary["mean_actions"],
             "mean_reward": summary["mean_reward"]}
    if "by_reference" in summary:
        brief["by_reference"] = summary["by_reference"]
    if baseline is not None:
        brief["against_baseline"] = {key: (len(value) if isinstance(value, list) else value)
                                     for key, value in payload["against_baseline"].items()}
    return brief


def adopt_baseline(config: MinimalRLConfig, out: Path) -> dict[str, Any] | None:
    """Take a finished evaluation of the starting adapter (``baseline_from``) as the run's baseline.

    ``None`` when the config names none or the folder holds no finished evaluation.  The evaluation must be of
    the run's starting adapter under the run's model, precision, episode and prompt limits; anything else is
    refused rather than compared with.  Each episode file is copied whole and in one move, so shards that start
    together can both do it.
    """
    if not config.baseline_from:
        return None
    source = Path(config.baseline_from)
    if not (source / "evaluation.json").is_file():
        return None
    found = json.loads((source / "evaluation.json").read_text(encoding="utf-8"))
    different = evaluation_differences(config, found)
    if different:
        raise ValueError(f"{source} is not an evaluation of this run's starting adapter under its settings: {different}")
    episodes = out / "evaluation" / "episodes"
    episodes.mkdir(parents=True, exist_ok=True)
    copied = 0
    for root in development_roots(config):
        name = f"{evaluate._safe_name(evaluate.case_id(root))}.json"
        if (episodes / name).is_file():
            continue
        if not (source / "episodes" / name).is_file():
            raise ValueError(f"{source} has no episode of the development case {evaluate.case_id(root)}")
        temporary = episodes / f".{name}.tmp-{os.getpid()}"
        shutil.copyfile(source / "episodes" / name, temporary)
        os.replace(temporary, episodes / name)
        copied += 1
    return {"adopted_from": str(source), "episodes_copied": copied}


def phase_baseline(config: MinimalRLConfig, manifest: Manifest, *, shard: tuple[int, int] | None = None,
                   progress: Callable[[Mapping[str, Any]], None] | None = None) -> dict[str, Any]:
    """The starting adapter on the development suite: taken from ``baseline_from`` where that holds it, else run here."""
    del manifest
    if not config.development_suite:
        raise ValueError("the baseline evaluation needs a development suite")
    out = Path(config.output_dir) / "baseline"
    adopted = adopt_baseline(config, out)
    head = {"baseline": True, "baseline_source": adopted["adopted_from"] if adopted else "evaluated by this run"}
    return greedy_evaluation(config, adapter=config.initial_adapter, out=out, shard=shard, head=head, progress=progress)


def phase_evaluate(config: MinimalRLConfig, manifest: Manifest, *, shard: tuple[int, int] | None = None,
                   progress: Callable[[Mapping[str, Any]], None] | None = None) -> dict[str, Any] | None:
    """The iteration's adapter on the development suite; skipped when the update left the adapter unchanged, and in
    the iterations between two evaluations of a run that sets ``evaluate_every``."""
    if not config.development_suite or not config.evaluate_every_iteration:
        return None
    out = iteration_dir(config, manifest.iteration)
    if not config.evaluated(manifest.iteration):
        every = int(config.evaluate_every)
        summary = {"iteration": manifest.iteration, "adapter": str(out / "adapter"),
                   "skipped": f"the run evaluates every {every} iterations; the next evaluation is of iteration "
                              f"{(manifest.iteration // every + 1) * every - 1}"}
        if shard is None:
            _write_json(out / "evaluation.json", {"summary": summary})
        return summary
    update_path = out / "update_report.json"
    update_report = json.loads(update_path.read_text(encoding="utf-8")) if update_path.is_file() else None
    if update_report is not None and update_report.get("update") is None:
        summary = {"iteration": manifest.iteration, "skipped": "the update phase left the adapter unchanged",
                   "adapter": str(out / "adapter")}
        if shard is None:
            _write_json(out / "evaluation.json", {"summary": summary})
        return summary
    return greedy_evaluation(config, adapter=str(out / "adapter"), out=out, shard=shard,
                             head={"iteration": manifest.iteration}, progress=progress)


# ------------------------------------------------------------------ the loop


def run_phase(config: MinimalRLConfig, phase: str, *, progress: Callable[[Mapping[str, Any]], None] | None = None,
              shard: tuple[int, int] | None = None) -> dict[str, Any] | None:
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    manifest_path = Path(config.output_dir) / "manifest.json"
    manifest = load_manifest(config)
    if phase != manifest.phase:
        raise RuntimeError(f"the run is at phase {manifest.phase!r} of iteration {manifest.iteration}, not {phase!r}")
    if shard is not None and phase not in SHARDED_PHASES:
        raise ValueError(f"only the {', '.join(SHARDED_PHASES)} phases are sharded")
    started = time.perf_counter()
    if phase == "baseline":
        result = phase_baseline(config, manifest, shard=shard, progress=progress)
    elif phase == "rollout":
        result = phase_rollout(config, manifest, progress=progress, shard=shard)
    elif phase == "update":
        result = phase_update(config, manifest, progress=progress)
    elif phase == "evaluate":
        result = phase_evaluate(config, manifest, shard=shard, progress=progress)
    else:
        raise ValueError(f"unknown phase {phase!r}")
    if shard is None:
        if phase == "baseline":
            manifest.baseline = {"adapter": config.initial_adapter, "evaluation": result}
            manifest.phase = "rollout"
        elif phase == "rollout":
            manifest.phase = "update"
        elif phase == "update":
            manifest.phase = "evaluate"
        else:
            adapter = str(iteration_dir(config, manifest.iteration) / "adapter")
            manifest.history.append({"iteration": manifest.iteration, "adapter": adapter, "evaluation": result})
            manifest.current_adapter = adapter
            manifest.iteration += 1
            manifest.phase = "rollout"
        manifest.save(manifest_path)
    print(json.dumps({"phase": phase, "shard": None if shard is None else f"{shard[0]}/{shard[1]}",
                      "iteration": manifest.iteration, "seconds": round(time.perf_counter() - started, 1),
                      "result": result}, default=str), flush=True)
    return result


def run_iterations(config: MinimalRLConfig, iterations: int, *, progress: Callable[[Mapping[str, Any]], None] | None = None) -> None:
    for _ in range(int(iterations)):
        manifest = load_manifest(config)
        if manifest.phase == "baseline":
            run_phase(config, "baseline", progress=progress)  # before the first iteration, not one of them
            manifest = load_manifest(config)
        for phase in ITERATION_PHASES[ITERATION_PHASES.index(manifest.phase):]:
            run_phase(config, phase, progress=progress)


def status(config: MinimalRLConfig) -> dict[str, Any]:
    return asdict(load_manifest(config))


def check(config: MinimalRLConfig) -> dict[str, Any]:
    """What a run of this config would start from, read without running anything.

    The suites are loaded and the first iteration's cases drawn (a class count that does not fit the suite is
    refused there); ``problems`` names what a run would trip over later: an adapter folder that holds no adapter,
    cases in both suites, development cases the reference does not know.
    """
    training = load_suite(config.training_suite)
    focus = config.focus_settings()
    focused: dict[str, Any] | None = None
    if focus.enabled:
        cases = focus_cases(config, training)
        drawn_failed, drawn_solved = focused_draw(config, cases, 0)
        drawn = drawn_failed + drawn_solved
        focused = {"cases_with_a_truth_built_sequence": len(cases["sequences"]),
                   "left_unrepaired_by_the_starting_adapter": len(cases["failed"]),
                   "repaired_by_the_starting_adapter": len(cases["solved"]),
                   "drawn_for_the_first_iteration": {"left_unrepaired": len(drawn_failed), "repaired": len(drawn_solved)},
                   "truth_built_examples": focus.examples, "demonstration_weight": float(focus.demonstration_weight),
                   "evaluation": str(focus.evaluation), "truth_sequences": str(focus.truth_sequences)}
    else:
        drawn = select_iteration_roots(config, training, 0)
    development = development_roots(config) if config.development_suite else []
    adapter = Path(config.initial_adapter)
    adapter_found = (adapter / "adapter_config.json").is_file() and any(
        (adapter / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin"))
    shared = {scenario_root(root) for root in training} & {scenario_root(root) for root in development}
    unknown: list[str] = []
    if config.reference:
        reference = json.loads(Path(config.reference).read_text(encoding="utf-8"))
        unknown = [evaluate.case_id(root) for root in development if evaluate.case_id(root) not in reference]
    problems = []
    if not adapter_found:
        problems.append(f"no adapter at {adapter} (adapter_config.json and its weights)")
    if shared:
        problems.append(f"{len(shared)} cases are in the training suite and in the development suite")
    if unknown:
        problems.append(f"{len(unknown)} development cases are not in the reference (first: {unknown[0]})")
    manifest = load_manifest(config)
    return {
        "ok": not problems, "problems": problems,
        "training_cases": len(training), "cases_drawn_for_the_first_iteration": len(drawn),
        "episodes_per_iteration": len(drawn) * int(config.samples_per_root), "development_cases": len(development),
        "initial_adapter": str(adapter), "kl_reference": config.initial_adapter if config.update_settings(0).beta_kl > 0.0 else None,
        "environment": environment_record(config), "prompt_limits": apply_prompt_limits(config),
        "run_is_at": {"iteration": manifest.iteration, "phase": manifest.phase},
        "evaluate_every": int(config.evaluate_every),
        **({"focus": focused} if focused is not None else {}),
        **({"forks": asdict(config.fork_settings())} if config.fork_settings().enabled else {}),
    }


def _print_progress(record: Mapping[str, Any]) -> None:
    print(json.dumps(dict(record), default=str), flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=PHASES, default=None, help="run one phase of the current iteration")
    parser.add_argument("--shard", default=None,
                        help="'i/n': this worker's slice of the baseline, rollout or evaluation phase; the unsharded call closes it")
    parser.add_argument("--iterations", type=int, default=1, help="iterations to run when no phase is given")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--check", action="store_true",
                        help="load the suites, draw the first iteration's cases and look for the adapter; run nothing")
    args = parser.parse_args(argv)
    config = MinimalRLConfig.load(args.config)
    if args.status:
        print(json.dumps(status(config), indent=2))
        return 0
    if args.check:
        found = check(config)
        print(json.dumps(found, indent=2, default=str))
        return 0 if found["ok"] else 2
    if args.phase:
        run_phase(config, args.phase, progress=_print_progress, shard=parse_shard(args.shard))
        return 0
    if args.shard:
        raise SystemExit("--shard needs --phase")
    run_iterations(config, args.iterations, progress=_print_progress)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
