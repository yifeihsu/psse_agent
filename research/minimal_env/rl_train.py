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
supervised target here: a correction is kept or undone by the model's own later action and credited by the
episode's outcome.  Truth-built sequences are never part of a group.

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
from psse_env.minimal.episode import OUTCOME_PROMPT_LIMIT  # noqa: E402
from psse_env.minimal.rollout import rollout_roots  # noqa: E402
from psse_env.minimal.scoring import RewardWeights  # noqa: E402
from psse_env.rl.advantages import GroupAdvantages, group_advantages, summarize_groups  # noqa: E402
from psse_env.rl.integrity import MARKER_FILE, IntegrityError, update_integrity  # noqa: E402
from psse_env.rl.rollout import Trajectory, TrajectoryStore, distinct_trajectories, reward_spread, scenario_root  # noqa: E402
from psse_env.rl.rows import RowSettings, build_rows  # noqa: E402
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
from research.minimal_env import evaluate  # noqa: E402

DRIVER_CONTRACT = "minimal_rl_driver_v1"
PHASES = ("baseline", "rollout", "update", "evaluate")
ITERATION_PHASES = ("rollout", "update", "evaluate")
SHARDED_PHASES = ("baseline", "rollout", "evaluate")


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
        group_advantages([0.0, 1.0], method=self.advantage_method, scale=self.advantage_scale)
        RewardWeights.from_mapping(self.reward)
        UpdateSettings(**self.update).validate()
        RootSampling.from_mapping(self.root_sampling).validate(self.roots_per_iteration)
        RowSettings.from_mapping(self.rows)

    def row_settings(self, iteration: int) -> RowSettings:
        payload = {**self.rows, "seed": self.seed + int(iteration)}
        if self.update.get("clip_ratio") is not None:
            payload["require_sampling_records"] = True  # a clipped update needs every row's behaviour probabilities
        return RowSettings.from_mapping(payload)

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


def select_iteration_roots(config: MinimalRLConfig, scenarios: Sequence[Mapping[str, Any]], iteration: int) -> list[dict[str, Any]]:
    """The cases one iteration samples: slices of one seeded permutation, or fixed counts per class."""
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


def phase_rollout(config: MinimalRLConfig, manifest: Manifest, *, progress: Callable[[Mapping[str, Any]], None] | None = None,
                  shard: tuple[int, int] | None = None) -> dict[str, Any]:
    """Sample the iteration's cases.  A shard samples its slice and writes no summary; the unsharded call finds
    every finished group in the store, samples what is left and closes the phase."""
    started = time.perf_counter()
    out = iteration_dir(config, manifest.iteration)
    out.mkdir(parents=True, exist_ok=True)
    sampling = RootSampling.from_mapping(config.root_sampling)
    drawn = select_iteration_roots(config, load_suite(config.training_suite), manifest.iteration)
    if shard is None:
        (out / "roots.json").write_text(json.dumps([scenario_root(root) for root in drawn], indent=1) + "\n", encoding="utf-8")
    mine = shard_scenarios(drawn, shard)
    limits = apply_prompt_limits(config)
    weights = RewardWeights.from_mapping(config.reward)
    store = TrajectoryStore(out / "trajectories")
    pending = [root for root in mine if not store.has_group(scenario_root(root), int(config.samples_per_root))]
    if pending:
        env = environment(config)
        try:
            rollout_roots(pending, store_dir=out / "trajectories", samples=int(config.samples_per_root),
                          base_seed=config.seed + manifest.iteration, env_factory=lambda: env,
                          policy_factory=sampling_policy_factory(config, manifest.current_adapter), weights=weights,
                          progress=progress)
        finally:
            _release_models()
    result: dict[str, Any] = {"iteration": manifest.iteration, "adapter": manifest.current_adapter, "roots": len(mine),
                              "groups_sampled": len(pending), "samples_per_root": int(config.samples_per_root),
                              "seconds": round(time.perf_counter() - started, 1)}
    if shard is not None:
        return {**result, "shard": f"{shard[0]}/{shard[1]}"}
    result.update(
        store=rollout_summary(store), sampling={"temperature": float(config.temperature), "top_p": 1.0, "top_k": None},
        environment=environment_record(config), prompt_limits=limits, reward_weights=asdict(weights))
    if sampling.mode == "stratified":
        strata = Counter(root_stratum(root, sampling.key) for root in drawn)
        result["root_sampling"] = {"mode": sampling.mode, "key": sampling.key, "strata": dict(sorted(strata.items()))}
    _write_json(out / "rollout_summary.json", result)
    return result


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
    if not informative_paired:
        report["update"] = None
        report["note"] = "no case whose paired samples scored differently: the adapter is carried over unchanged"
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
        rows, rows_report = build_rows(groups, render_prompt=render, prompt_token_ids=prompt_ids,
                                       max_length=int(config.max_length), required_side_inputs=side_inputs,
                                       settings=config.row_settings(manifest.iteration))
        report["rows"] = rows_report
        report["resumed_from_checkpoint"] = str(resume_dir) if resume_dir is not None else None
        if not rows:
            report["update"] = None
            report["note"] = "no training row after the pairing and rendering checks"
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
    """The iteration's adapter on the development suite; skipped when the update left the adapter unchanged."""
    if not config.development_suite or not config.evaluate_every_iteration:
        return None
    out = iteration_dir(config, manifest.iteration)
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
