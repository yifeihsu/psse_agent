"""The RL iteration driver: sample, score, update, evaluate, checkpoint.

    python -m psse_env.rl.train --config rl.json --iterations 1              # run the next iterations
    python -m psse_env.rl.train --config rl.json --phase rollout             # one phase of the current iteration
    python -m psse_env.rl.train --config rl.json --phase rollout --shard 0/2 # one worker's slice of a phase
    python -m psse_env.rl.train --config rl.json --status

Every phase survives preemption.  Rollouts and the greedy evaluations are
stored per root, so a restarted process skips what is done; the update
commits its adapter and optimizer after every optimizer step and its
accumulated gradients every few rows (:class:`psse_env.rl.update.UpdateCheckpoint`).
``manifest.json`` records the phase the run is at and advances only when a
phase completes.  A run whose config sets ``baseline_evaluation`` starts
with a ``baseline`` phase, the starting adapter's greedy evaluation under the
driver's own loader, before its first rollout.  The baseline, rollout and
evaluation phases shard over worker processes (``--shard i/n``); the
unsharded call of the same phase finishes any root left over and closes the
phase.

With ``carry_optimizer`` in the update settings, an update leaves its
optimizer state in its iteration directory (``optimizer.pt``) and the next
update continues from it instead of starting a new optimizer (2026-11-09).

For the experiment series of 2026-11-10 (docs/rl_improvement_implementation_20261110.md):
``observation_mode`` chooses how much of an episode's tool history the
policy is shown, in every rollout, continuation and evaluation of the run
(the update re-renders the model-visible view each step stored); a config
with ``decision_credit`` runs a ``continuation`` phase between the rollout
and the update, in which the lifecycle decisions the policy sampled are
branched and both choices continued by the current policy
(:mod:`psse_env.rl.decision_credit`); every update report carries the
integrity checks of :mod:`psse_env.rl.integrity`, and a config with
``stop_on_integrity_failure`` does not advance past an update that fails one.
``max_input_tokens`` is the run's own prompt limit: every phase sets the
policy's input limit from it (:func:`apply_prompt_limits`) and records it.

The config (JSON) names the training and development suites, the starting
adapter and base model, the evidence profile and environment options the
pipeline cell used (``environment_options`` and ``expert_options`` carry the
named options of the research environment and the teacher, 2026-11-07), the
sampling, reward, advantage and update settings, and the action budgets the
greedy evaluation reports success within.  See docs/rl_experiments_20261014.md
and the RL plan of 2026-11-05 (docs/rl_beyond_teacher_plan_20261105.md).
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from psse_env.dagger.observation_modes import BOUNDED_HISTORY, validate_observation_mode
from psse_env.rl.advantages import GroupAdvantages, group_advantages, summarize_groups
from psse_env.rl.decision_credit import (
    DECISION_CREDIT_CONTRACT,
    DecisionCreditSettings,
    candidate_states,
    check_provenance,
    decision_token_probability,
    load_values,
    select_states,
    state_values,
    summarize_values,
    values_map,
)
from psse_env.rl.forks import CHOICES, branch_seed, run_branch
from psse_env.rl.integrity import MARKER_FILE, IntegrityError, update_integrity
from psse_env.rl.lifecycle import lifecycle_decisions, scenario_truth, summarize_decisions
from psse_env.rl.reward import RewardWeights, episode_basis, episode_reward, load_expert_reference
from psse_env.rl.rollout import (
    Trajectory,
    TrajectoryStore,
    prompt_limit_refusals,
    rollout_roots,
    scenario_family,
    scenario_root,
    store_summary,
)
from psse_env.rl.rows import RowSettings, build_rows
from psse_env.rl.sampling import RootSampling, root_stratum, stratified_roots
from psse_env.rl.update import (
    REFERENCE_ADAPTER,
    TRAINED_ADAPTER,
    UpdateCheckpoint,
    UpdateSettings,
    build_optimizer,
    load_optimizer_state,
    optimizer_steps_taken,
    policy_gradient_update,
    save_optimizer_state,
)

TRAIN_CONTRACT = "rl_iteration_driver_v2"
#: The phases of a run: a baseline evaluation of the starting adapter before
#: the first iteration when the config asks for one, then the iteration cycle.
PHASES = ("baseline", "rollout", "continuation", "update", "evaluate")
ITERATION_PHASES = ("rollout", "update", "evaluate")
#: The iteration cycle of a run with decision credit: the lifecycle decisions the rollouts sampled are
#: branched and continued between the rollout and the update.
CREDIT_ITERATION_PHASES = ("rollout", "continuation", "update", "evaluate")
SHARDED_PHASES = ("baseline", "rollout", "continuation", "evaluate")
#: The optimizer state an update leaves in its iteration directory when the
#: update settings say ``carry_optimizer``, for the next update to continue.
OPTIMIZER_FILE = "optimizer.pt"
#: The named options of the research environment a config may set
#: (``RESEARCH_ENVIRONMENT_OPTIONS`` of scripts/run_dagger_research.py, as
#: ``--env-option`` names them).  An option a config does not name is left
#: to the factory: off, or what its teacher variant needs
#: (``flow_at_noise_cue``, ``revert_fruitless_cycles``, ``route_comparison``).
ENVIRONMENT_OPTION_KEYS = (
    "lifecycle_choice",
    "detailed_evidence",
    "pair_refit_adjacent",
    "flow_at_noise_cue",
    "route_comparison",
    "revert_fruitless_cycles",
    "target_first_partial",
    "branch_first_partial",
)
#: Environment options that are config fields of their own.
TOP_LEVEL_ENVIRONMENT_FIELDS = ("evidence_profile", "normalized_residual_threshold", "triage_classifier", "progress_basis")


@dataclass
class RLConfig:
    output_dir: str
    training_suite: str
    development_suite: str | None
    initial_adapter: str
    base_model: str
    base_revision: str
    architecture: str | None = None
    prompt_profile: str | None = None
    evidence_profile: str = "classifier_gated_diagnostics"
    triage_classifier: str | None = None
    progress_basis: str | None = None
    normalized_residual_threshold: float | None = 4.0
    expert_variant: str = "baseline"
    #: Named options of the research environment beyond the fields above
    #: (:data:`ENVIRONMENT_OPTION_KEYS`; RL plan of 2026-11-05, section 5):
    #: ``{"lifecycle_choice": 1, "detailed_evidence": 1, ...}``.  Every
    #: phase and shard applies them before building an environment, and the
    #: rollout summary records them.
    environment_options: dict[str, Any] = field(default_factory=dict)
    #: Named options of the teacher beyond its variant
    #: (``RESEARCH_EXPERT_OPTIONS``: ``abstain_isolated_lines``,
    #: ``state_ambiguity``, ``learned_ranker``), for ``policy_source: expert``.
    expert_options: dict[str, Any] = field(default_factory=dict)
    hif_search_profile: str = "research"
    max_steps: int = 40
    #: Action budgets the greedy evaluation also reports success within: the
    #: solved cases that used at most this many actions under ``max_steps``
    #: (:func:`aggregate_evaluation`).
    evaluation_budgets: list[int] = field(default_factory=lambda: [16, 20])
    samples_per_root: int = 4
    seed: int = 20261014
    #: Sampling.  The update takes log-probabilities at this temperature, so
    #: any temperature is on-policy; a nucleus or top-k cut is not and is
    #: refused (``validate``).
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int | None = None
    roots_per_iteration: int | None = None
    #: How the iteration's roots are drawn (:mod:`psse_env.rl.sampling`):
    #: empty is the seeded permutation in disjoint slices; ``{"mode":
    #: "stratified", "key": ..., "counts": {stratum: n}}`` fixes each
    #: stratum's share of every iteration.
    root_sampling: dict[str, Any] = field(default_factory=dict)
    #: The longest training row, in tokens (prompt and completion).
    max_length: int = 32768
    #: The longest prompt the policy is given, in tokens: the research policy's input limit
    #: (``RESEARCH_MAX_INPUT_TOKENS``), which every phase's process sets from this field
    #: (:func:`apply_prompt_limits`).  ``None`` leaves the process's own value, as in every recorded
    #: cell.  The bounded mode cuts a longer prompt from the left; ``complete_history_v1`` refuses
    #: it.  The model's window is the only limit of the model itself (262,144 positions for
    #: gemma-4-12B-it); what a card holds is the practical one, and ``max_length`` must hold it too.
    max_input_tokens: int | None = None
    advantage_method: str = "grpo"
    advantage_scale: str = "group"
    expert_reference: str | None = None
    load_in_4bit: bool = True
    local_files_only: bool = True
    evaluate_every_iteration: bool = True
    #: Evaluate the starting adapter on the development suite before the
    #: first rollout, through the driver's own loader (``load_in_4bit`` as
    #: configured), so every iteration's score compares with a baseline
    #: measured the same way (2026-11-07: the cloning cell's arms ran on a
    #: 4-bit base and the RL cell runs in bf16, so an arm measured there is
    #: not this run's baseline).  Applies at the start of a run only.
    baseline_evaluation: bool = False
    evaluation_workers_per_policy: int = 1
    update_checkpoint_rows: int = 32
    #: ``adapter`` samples the LoRA policy; ``expert`` rolls the observable
    #: teacher out instead (a wiring check of the environment, the audit and
    #: the store on a CPU: its episodes carry no generation record, so the
    #: update phase has nothing to train on and carries the adapter over).
    policy_source: str = "adapter"
    reward: dict[str, Any] = field(default_factory=dict)
    update: dict[str, Any] = field(default_factory=dict)
    #: What the update trains on (:class:`psse_env.rl.rows.RowSettings`):
    #: complete groups only, prompt digest and truncation checks, sampling
    #: records, preservation rows for the KL term.  Empty keeps the
    #: pre-2026-10-17 behaviour.
    rows: dict[str, Any] = field(default_factory=dict)
    #: How much of an episode's tool history the policy is shown
    #: (:mod:`psse_env.dagger.observation_modes`): the last four events, as in
    #: every recorded cell, or every event (``complete_history_v1``).  One
    #: setting for the rollouts, the continuations and the evaluations of a
    #: run; the update re-renders what each step stored.
    observation_mode: str = BOUNDED_HISTORY
    #: Outcome credit for lifecycle decisions from learner continuations
    #: (:class:`psse_env.rl.decision_credit.DecisionCreditSettings`); empty is off.
    decision_credit: dict[str, Any] = field(default_factory=dict)
    #: A JSON list of root ids no iteration may draw (roots kept apart for a
    #: diagnostic or a test).  The draw is not changed to avoid them: an
    #: iteration that would draw one is refused, so the reservation holds or
    #: the run stops.
    reserved_roots: str | None = None
    #: Do not advance past an update that shows one of the stop conditions of
    #: :mod:`psse_env.rl.integrity` (off: the report carries the checks only).
    stop_on_integrity_failure: bool = False
    #: Where a run belongs: the series, the arm, what it is compared with.
    #: Recorded in the phase summaries and the baseline register, read by nothing.
    series: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "RLConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {name for name in cls.__dataclass_fields__}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"unknown config fields: {unknown}")
        config = cls(**payload)
        config.validate()
        return config

    def validate(self) -> None:
        if self.samples_per_root < 2:
            raise ValueError("samples_per_root must be at least 2 for a group-relative advantage")
        if self.max_steps <= 0 or self.max_length <= 0:
            raise ValueError("max_steps and max_length must be positive")
        if self.max_input_tokens is not None:
            if isinstance(self.max_input_tokens, bool) or not isinstance(self.max_input_tokens, int) or self.max_input_tokens <= 0:
                raise ValueError("max_input_tokens must be a positive integer (null: the process's own input limit)")
            if self.max_length < self.max_input_tokens:
                raise ValueError(
                    f"max_length ({self.max_length}) is the longest training row, so it must hold the longest prompt the "
                    f"policy is given (max_input_tokens {self.max_input_tokens})"
                )
        if not 0.0 < self.temperature:
            raise ValueError("temperature must be positive")
        if self.top_p < 1.0 or (self.top_k is not None and int(self.top_k) > 0):
            raise ValueError(
                "a nucleus or top-k cut samples from a distribution the update does not optimize, "
                "which biases the policy gradient; set top_p to 1.0 and top_k to null (use the "
                "temperature to control exploration)"
            )
        if self.policy_source not in {"adapter", "expert"}:
            raise ValueError("policy_source must be adapter or expert")
        if self.update_checkpoint_rows <= 0:
            raise ValueError("update_checkpoint_rows must be positive")
        if "temperature" in self.update:
            raise ValueError("the update's temperature is the sampling temperature; set it at the top level only")
        if not isinstance(self.environment_options, Mapping) or not isinstance(self.expert_options, Mapping):
            raise ValueError("environment_options and expert_options must be mappings")
        unknown = sorted(set(self.environment_options) - set(ENVIRONMENT_OPTION_KEYS))
        if unknown:
            top_level = sorted(set(unknown) & set(TOP_LEVEL_ENVIRONMENT_FIELDS))
            if top_level:
                raise ValueError(f"environment_options {top_level} are config fields of their own; set them at the top level")
            raise ValueError(f"unknown environment_options {unknown}; the named options are {list(ENVIRONMENT_OPTION_KEYS)}")
        if "variant" in self.expert_options:
            raise ValueError("expert_options.variant is the expert_variant field")
        weights = RewardWeights.from_mapping(self.reward)
        if weights.lifecycle_credit > 0.0 and not self.environment_options.get("lifecycle_choice"):
            raise ValueError(
                "reward.lifecycle_credit scores the lifecycle choice, which environment_options.lifecycle_choice must open"
            )
        for budget in self.evaluation_budgets:
            if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0 or budget > int(self.max_steps):
                raise ValueError(f"evaluation_budgets must be positive integers at most max_steps, got {self.evaluation_budgets!r}")
        if self.baseline_evaluation and not self.development_suite:
            raise ValueError("baseline_evaluation needs a development suite to evaluate the starting adapter on")
        validate_observation_mode(self.observation_mode)
        credit = DecisionCreditSettings.from_mapping(self.decision_credit)
        if credit.enabled and not self.environment_options.get("lifecycle_choice"):
            raise ValueError(
                "decision_credit branches the lifecycle choice, which environment_options.lifecycle_choice must open"
            )
        if not isinstance(self.series, Mapping):
            raise ValueError("series must be a mapping")
        UpdateSettings(**self.update).validate()
        RootSampling.from_mapping(self.root_sampling).validate(self.roots_per_iteration)
        RowSettings.from_mapping(self.rows)

    def row_settings(self, iteration: int) -> RowSettings:
        """The row settings of one iteration: a clipped update always requires sampling records."""

        payload = {**self.rows, "seed": self.seed + int(iteration)}
        if self.update.get("clip_ratio") is not None:
            payload["require_sampling_records"] = True
        return RowSettings.from_mapping(payload)

    def update_settings(self, iteration: int) -> UpdateSettings:
        return UpdateSettings(**{**self.update, "seed": self.seed + int(iteration), "temperature": self.temperature})

    def credit_settings(self) -> DecisionCreditSettings:
        return DecisionCreditSettings.from_mapping(self.decision_credit)

    def evaluator_options(self) -> dict[str, Any]:
        """Keyword arguments every closed-loop evaluation of the run adds: the observation mode when it is not the bounded one."""

        return {} if self.observation_mode == BOUNDED_HISTORY else {"observation_mode": self.observation_mode}

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Manifest:
    contract: str = TRAIN_CONTRACT
    iteration: int = 0
    phase: str = "rollout"
    current_adapter: str = ""
    history: list[dict[str, Any]] = field(default_factory=list)
    #: The baseline evaluation of the starting adapter (adapter and summary), once the phase has run.
    baseline: dict[str, Any] | None = None

    @classmethod
    def load(cls, path: Path, *, initial_adapter: str, initial_phase: str = "rollout") -> "Manifest":
        """The run's manifest, or a fresh one at ``initial_phase`` (``baseline`` when the config asks for one)."""

        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return cls(
                contract=str(payload.get("contract") or TRAIN_CONTRACT),
                iteration=int(payload.get("iteration") or 0),
                phase=str(payload.get("phase") or "rollout"),
                current_adapter=str(payload.get("current_adapter") or initial_adapter),
                history=list(payload.get("history") or []),
                baseline=dict(payload["baseline"]) if isinstance(payload.get("baseline"), Mapping) else None,
            )
        if initial_phase not in PHASES:
            raise ValueError(f"unknown initial phase {initial_phase!r}")
        return cls(current_adapter=initial_adapter, phase=initial_phase)

    def save(self, path: Path) -> None:
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _safe_name(root: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in root)


def load_suite(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        rows = payload.get("scenarios") or payload.get("rows") or payload.get("development") or []
    else:
        rows = payload
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"suite {path} holds no scenarios")
    return [dict(row) for row in rows]


def iteration_roots(scenarios: Sequence[Mapping[str, Any]], *, iteration: int, roots_per_iteration: int | None, seed: int) -> list[dict[str, Any]]:
    """The roots of one iteration: all of them, or a seeded slice that cycles through the suite."""

    if roots_per_iteration is None or roots_per_iteration >= len(scenarios):
        return [dict(row) for row in scenarios]
    import random

    order = list(range(len(scenarios)))
    random.Random(seed).shuffle(order)
    start = (iteration * roots_per_iteration) % len(order)
    chosen = [order[(start + offset) % len(order)] for offset in range(roots_per_iteration)]
    return [dict(scenarios[index]) for index in chosen]


def select_iteration_roots(config: RLConfig, scenarios: Sequence[Mapping[str, Any]], iteration: int) -> list[dict[str, Any]]:
    """The roots of one iteration under the config's ``root_sampling`` (the permutation when it is empty)."""

    sampling = RootSampling.from_mapping(config.root_sampling)
    if sampling.mode == "stratified":
        return stratified_roots(scenarios, iteration=iteration, counts=sampling.counts, key=sampling.key, seed=config.seed)
    return iteration_roots(scenarios, iteration=iteration, roots_per_iteration=config.roots_per_iteration, seed=config.seed)


def parse_shard(value: str | None) -> tuple[int, int] | None:
    """``"i/n"`` selects every n-th root starting at i (0-based); ``None`` is the whole draw."""

    if value is None:
        return None
    try:
        index_text, count_text = str(value).split("/", 1)
        index, count = int(index_text), int(count_text)
    except ValueError as exc:
        raise ValueError(f"a shard is 'i/n', got {value!r}") from exc
    if count <= 0 or not 0 <= index < count:
        raise ValueError(f"shard index must lie in [0, n), got {value!r}")
    return index, count


def shard_scenarios(scenarios: Sequence[Mapping[str, Any]], shard: tuple[int, int] | None) -> list[dict[str, Any]]:
    if shard is None:
        return [dict(row) for row in scenarios]
    index, count = shard
    return [dict(row) for row in scenarios[index::count]]


def configure_research_environment(config: RLConfig) -> Callable[..., Any]:
    """Point the research runner's environment and teacher options at the config's and return the environment factory.

    Every named option (:data:`ENVIRONMENT_OPTION_KEYS`) is set from the
    config or removed, and the teacher options are replaced, so a process
    that configured another cell earlier cannot leave an option on that this
    config does not name.
    """

    import scripts.run_dagger_research as research

    research.RESEARCH_ENVIRONMENT_OPTIONS["evidence_profile"] = config.evidence_profile
    research.RESEARCH_ENVIRONMENT_OPTIONS["normalized_residual_threshold"] = config.normalized_residual_threshold
    research.RESEARCH_ENVIRONMENT_OPTIONS["triage_classifier"] = (
        str(Path(config.triage_classifier).resolve()) if config.triage_classifier else None
    )
    if config.progress_basis:
        research.RESEARCH_ENVIRONMENT_OPTIONS["progress_basis"] = config.progress_basis
    for key in ENVIRONMENT_OPTION_KEYS:
        if key in config.environment_options:
            research.RESEARCH_ENVIRONMENT_OPTIONS[key] = config.environment_options[key]
        else:
            research.RESEARCH_ENVIRONMENT_OPTIONS.pop(key, None)
    research.RESEARCH_EXPERT_OPTIONS.clear()
    research.RESEARCH_EXPERT_OPTIONS["variant"] = config.expert_variant
    research.RESEARCH_EXPERT_OPTIONS.update({str(key): value for key, value in config.expert_options.items()})
    return research.resolve_environment_factory(config.hif_search_profile, config.evidence_profile)


@lru_cache(maxsize=8)
def model_context_window(base_model: str, base_revision: str | None = None) -> int | None:
    """The base model's context window: ``max_position_embeddings`` of its text tower, from its config file on this machine.

    ``None`` when the file is not here; nothing is downloaded for it.
    """

    path = Path(base_model) / "config.json"
    if not path.is_file():
        try:
            from huggingface_hub import hf_hub_download

            path = Path(hf_hub_download(base_model, "config.json", revision=base_revision, local_files_only=True))
        except Exception:
            return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    text = payload.get("text_config") if isinstance(payload.get("text_config"), Mapping) else payload
    value = text.get("max_position_embeddings") if isinstance(text, Mapping) else None
    return int(value) if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def apply_prompt_limits(config: RLConfig) -> dict[str, Any]:
    """Set the research policy's input limit to the config's, and return the limits this process runs under.

    A config that names no limit (``max_input_tokens`` null) leaves the
    process's own, the ``RESEARCH_MAX_INPUT_TOKENS`` it was started with.  A
    limit the base model cannot take, with its longest completion, is
    refused when the model's config is on this machine.  Every phase calls
    this before it renders or generates, and its summary carries the record.
    """

    import psse_env.dagger.preliminary_e2b_eval as policy_limits

    window = model_context_window(str(config.base_model), config.base_revision)
    source = "environment"
    if config.max_input_tokens is not None:
        longest = int(config.max_input_tokens) + int(policy_limits.MAX_NEW_TOKENS)
        if window is not None and longest > window:
            raise ValueError(
                f"max_input_tokens {config.max_input_tokens} and a completion of {policy_limits.MAX_NEW_TOKENS} tokens are more "
                f"than the {window} positions of {config.base_model}"
            )
        policy_limits.MAX_INPUT_TOKENS = int(config.max_input_tokens)
        source = "config"
    limit = int(policy_limits.MAX_INPUT_TOKENS)
    return {
        "max_input_tokens": limit,
        "max_new_tokens": int(policy_limits.MAX_NEW_TOKENS),
        # What a complete-history prompt may have: the limit less the longest completion.
        "usable_complete_history_tokens": limit - int(policy_limits.MAX_NEW_TOKENS),
        "max_length": int(config.max_length),
        "source": source,
        "model_context_window": window,
    }


def environment_record(config: RLConfig) -> dict[str, Any]:
    """What the run's environment is built with, for the phase summaries."""

    return {
        "evidence_profile": config.evidence_profile,
        "hif_search_profile": config.hif_search_profile,
        "normalized_residual_threshold": config.normalized_residual_threshold,
        "triage_classifier": config.triage_classifier,
        "progress_basis": config.progress_basis,
        "environment_options": dict(config.environment_options),
        "expert_variant": config.expert_variant,
        "expert_options": dict(config.expert_options),
        "max_steps": config.max_steps,
        "observation_mode": config.observation_mode,
    }


def sampling_policy_factory(config: RLConfig, adapter: str, env_factory: Callable[..., Any] | None = None) -> Callable[[int], Any]:
    if config.policy_source == "expert":
        import scripts.run_dagger_research as research

        if env_factory is None:
            raise ValueError("the expert policy source needs the environment factory")
        return lambda seed: research.research_expert_policy(env_factory)

    from psse_env.dagger.research_policy_factory import GenerationOptions, research_gemma_policy_factory

    def factory(seed: int) -> Any:
        return research_gemma_policy_factory(
            adapter,
            base_model=config.base_model,
            base_revision=config.base_revision,
            load_in_4bit=config.load_in_4bit,
            local_files_only=config.local_files_only,
            prompt_profile=config.prompt_profile,
            architecture=config.architecture,
            generation=GenerationOptions(
                do_sample=True,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                seed=int(seed),
            ),
        )

    return factory


def iteration_dir(config: RLConfig, iteration: int) -> Path:
    return Path(config.output_dir) / f"iteration_{iteration:03d}"


def initial_phase(config: RLConfig) -> str:
    """Where a fresh run starts: the baseline evaluation when the config asks for one, else the first rollout."""

    return "baseline" if config.baseline_evaluation else "rollout"


def iteration_phases(config: RLConfig) -> tuple[str, ...]:
    """The phases of one iteration of this run, in order."""

    return CREDIT_ITERATION_PHASES if config.credit_settings().enabled else ITERATION_PHASES


def reserved_roots(config: RLConfig) -> set[str]:
    if not config.reserved_roots:
        return set()
    payload = json.loads(Path(config.reserved_roots).read_text(encoding="utf-8"))
    roots = payload.get("roots") if isinstance(payload, Mapping) else payload
    if not isinstance(roots, list):
        raise ValueError(f"reserved_roots {config.reserved_roots} is not a list of root ids (or a mapping with one under 'roots')")
    return {str(root) for root in roots}


def load_manifest(config: RLConfig) -> Manifest:
    return Manifest.load(Path(config.output_dir) / "manifest.json", initial_adapter=config.initial_adapter,
                         initial_phase=initial_phase(config))


def phase_rollout(
    config: RLConfig,
    manifest: Manifest,
    *,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
    shard: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Sample the iteration's roots.  A shard samples its slice and writes no summary;
    the unsharded call afterwards finds every finished group in the store and closes the phase."""

    from psse_env.dagger.release_factories import deterministic_case_loader

    out = iteration_dir(config, manifest.iteration)
    out.mkdir(parents=True, exist_ok=True)
    sampling = RootSampling.from_mapping(config.root_sampling)
    scenarios = select_iteration_roots(config, load_suite(config.training_suite), manifest.iteration)
    strata: dict[str, int] = {}
    if sampling.mode == "stratified":
        for scenario in scenarios:
            name = root_stratum(scenario, sampling.key)
            strata[name] = strata.get(name, 0) + 1
    drawn_reserved = sorted({scenario_root(s) for s in scenarios} & reserved_roots(config))
    if drawn_reserved:
        raise RuntimeError(
            f"iteration {manifest.iteration} draws {len(drawn_reserved)} reserved roots (first: {drawn_reserved[0]}); "
            "they are kept apart from training (reserved_roots)"
        )
    if shard is None:
        (out / "roots.json").write_text(json.dumps([scenario_root(s) for s in scenarios], indent=1) + "\n", encoding="utf-8")
    scenarios = shard_scenarios(scenarios, shard)
    limits = apply_prompt_limits(config)
    env_factory = configure_research_environment(config)
    store = TrajectoryStore(out / "trajectories")
    weights = RewardWeights.from_mapping(config.reward)
    reference = load_expert_reference(config.expert_reference)
    summary = rollout_roots(
        scenarios,
        store=store,
        samples=config.samples_per_root,
        base_seed=config.seed + manifest.iteration,
        env_factory=env_factory,
        policy_factory=sampling_policy_factory(config, manifest.current_adapter, env_factory),
        max_steps=config.max_steps,
        case_loader=deterministic_case_loader,
        weights=weights,
        expert_reference=reference or None,
        progress=progress,
        evaluator_options=config.evaluator_options(),
    )
    summary.update(adapter=manifest.current_adapter, iteration=manifest.iteration, roots=len(scenarios),
                   shard=None if shard is None else f"{shard[0]}/{shard[1]}", observation_mode=config.observation_mode)
    if shard is None:
        summary["store"] = store_summary(store)
        summary["sampling"] = {"temperature": config.temperature, "top_p": config.top_p, "top_k": config.top_k}
        summary["environment"] = environment_record(config)
        summary["prompt_limits"] = limits
        if config.series:
            summary["series"] = dict(config.series)
        if sampling.mode == "stratified":
            summary["root_sampling"] = {"mode": sampling.mode, "key": sampling.key, "strata": dict(sorted(strata.items()))}
        _write_json(out / "rollout_summary.json", summary)
    return summary


def _alternative_renderer(config: RLConfig, adapter: str) -> Any:
    """The renderer of the other lifecycle choice at a sampled decision (the processor alone; no model is loaded).

    ``None`` for a run that rolls the expert out: its steps have no sampling record to read a probability from.
    """

    if config.policy_source != "adapter":
        return None
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        config.base_model, revision=config.base_revision, local_files_only=config.local_files_only
    )
    _render, _prompt_ids, renderer = _prompt_functions(config, adapter, processor)
    return renderer


def _branch_path(directory: Path, name: str, choice: str, index: int) -> Path:
    return directory / f"{name}__{choice}__{int(index)}.json"


def phase_continuation(
    config: RLConfig,
    manifest: Manifest,
    *,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
    shard: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Branch the lifecycle decisions this iteration's rollouts sampled and continue both choices with the current policy.

    The states are chosen from the stored rollouts by a seeded rule
    (:func:`psse_env.rl.decision_credit.select_states`), so every shard
    works on the same list and takes its slice of it.  Each branch is one
    file; a restarted phase skips the branches that are there.  The
    unsharded call finishes what is left and writes ``decision_values.json``:
    the values of the two choices at every state and the fork-local
    advantage of every sampled decision they cover, for this adapter and
    observation mode only.
    """

    from psse_env.dagger.release_factories import deterministic_case_loader

    settings = config.credit_settings()
    if not settings.enabled:
        raise RuntimeError("the continuation phase belongs to a run with decision_credit.enabled")
    out = iteration_dir(config, manifest.iteration)
    store = TrajectoryStore(out / "trajectories")
    branch_dir = out / "continuation" / "branches"
    branch_dir.mkdir(parents=True, exist_ok=True)
    groups = list(store.iter_groups())
    chosen, selection = select_states(candidate_states(groups), settings, seed=config.seed + manifest.iteration)
    by_root = {scenario_root(scenario): scenario
               for scenario in select_iteration_roots(config, load_suite(config.training_suite), manifest.iteration)}
    mine = chosen if shard is None else chosen[shard[0]::shard[1]]
    pending = [(group, choice, index) for group in mine for choice in CHOICES for index in range(settings.continuations)
               if not _branch_path(branch_dir, group.state.name, choice, index).is_file()]
    ran = 0
    limits = apply_prompt_limits(config)
    if pending:
        env_factory = configure_research_environment(config)
        weights = RewardWeights.from_mapping(config.reward)
        policy_factory = sampling_policy_factory(config, manifest.current_adapter, env_factory)
        try:
            for group, choice, index in pending:
                state = group.state
                seed = branch_seed(config.seed + manifest.iteration, state, index)
                record = run_branch(
                    by_root[state.root], state, choice,
                    continuation=policy_factory(seed), continuation_index=index, seed=seed,
                    env_factory=env_factory, max_steps=config.max_steps, case_loader=deterministic_case_loader,
                    weights=weights, observation_mode=config.observation_mode,
                )
                record.update(adapter=manifest.current_adapter, iteration=manifest.iteration, stratum=group.stratum)
                _write_json(_branch_path(branch_dir, state.name, choice, index), record)
                ran += 1
                if progress is not None:
                    progress({"branch": state.name, "forced": choice, "continuation": index, "valid": record["valid"],
                              "success": record["success"], "reward": round(float(record["reward"]["total"]), 3), "ran": ran,
                              "of": len(pending)})
        finally:
            if config.policy_source == "adapter":
                from psse_env.dagger.research_policy_factory import clear_research_policy_cache

                clear_research_policy_cache()
            _release_gpu()
    if shard is not None:
        return {"shard": f"{shard[0]}/{shard[1]}", "states": len(mine), "branches_run": ran, "iteration": manifest.iteration}

    # The behaviour probability of each covered sampled choice, from the rollout's own sampling record.
    renderer = _alternative_renderer(config, manifest.current_adapter)
    trajectories = {(trajectory.root, int(trajectory.sample_index)): trajectory for _root, group in groups for trajectory in group}
    sources: dict[str, int] = {}
    for group in chosen:
        for row in group.rows:
            trajectory = trajectories[(group.state.root, int(row["sample_index"]))]
            probability, source = decision_token_probability(trajectory.steps[int(row["position"])], row["tool"], renderer)
            row["probability"], row["probability_source"] = probability, source
            sources[source] = sources.get(source, 0) + 1
    states = []
    for group in chosen:
        branches = [json.loads(_branch_path(branch_dir, group.state.name, choice, index).read_text(encoding="utf-8"))
                    for choice in CHOICES for index in range(settings.continuations)]
        states.append(state_values(group, branches, settings))
    payload = {
        "contract": DECISION_CREDIT_CONTRACT,
        "iteration": manifest.iteration,
        "adapter": manifest.current_adapter,
        "observation_mode": config.observation_mode,
        "prompt_limits": limits,
        "settings": settings.as_dict(),
        "selection": selection,
        "probability_sources": sources,
        "summary": summarize_values(states),
        "states": states,
    }
    _write_json(out / "decision_values.json", payload)
    return {"iteration": manifest.iteration, "selection": selection, "summary": payload["summary"],
            "probability_sources": sources, "branches_run": ran}


def score_groups(config: RLConfig, store: TrajectoryStore) -> tuple[list[tuple[list[Trajectory], GroupAdvantages]], dict[str, Any]]:
    groups: list[tuple[list[Trajectory], GroupAdvantages]] = []
    by_root: dict[str, GroupAdvantages] = {}
    for root, trajectories in store.iter_groups():
        advantages = group_advantages(
            [trajectory.total_reward for trajectory in trajectories],
            method=config.advantage_method,
            scale=config.advantage_scale,
            root=root,
        )
        groups.append((trajectories, advantages))
        by_root[root] = advantages
    summary = summarize_groups(by_root)
    successes = sum(
        1 for trajectories, _ in groups for trajectory in trajectories if trajectory.evaluation.get("truth_audited_task_success")
    )
    episodes = sum(len(trajectories) for trajectories, _ in groups)
    summary.update(
        episodes=episodes,
        success_rate=(successes / episodes) if episodes else None,
        mean_calls=(sum(int(t.reward.get("calls") or 0) for ts, _ in groups for t in ts) / episodes) if episodes else None,
        unpaired_trajectories=sum(1 for ts, _ in groups for t in ts if not t.pairing_ok),
        lifecycle={
            **summarize_decisions([decision for ts, _ in groups for t in ts for decision in t.lifecycle]),
            "labelled_decisions_paired": sum(t.labelled_decisions for ts, _ in groups for t in ts if t.pairing_ok),
        },
    )
    return groups, summary


def _save_trained_adapter(model: Any, directory: Path) -> None:
    """Write only the trained adapter (never the frozen reference) as a loadable PEFT directory."""

    try:
        model.save_pretrained(str(directory), selected_adapters=[TRAINED_ADAPTER])
    except TypeError:
        model.save_pretrained(str(directory))


def _load_trainable_model(
    config: RLConfig, adapter: str, *, reference_adapter: str | None = None
) -> tuple[Any, Any, tuple[str, ...]]:
    """The base in 4-bit with ``adapter`` attached trainable, plus its processor and side inputs.

    With ``reference_adapter`` the run's starting adapter is loaded beside it,
    frozen, as the KL reference (PEFT adapter ``reference``).
    """

    from psse_env.sft.research_cli import ResearchTrainerSettings, _load_research_base, attach_research_adapter
    from psse_env.sft.training import LoraSettings, infer_required_side_input_names

    settings = ResearchTrainerSettings(
        model_name=config.base_model,
        revision=config.base_revision,
        output_dir=str(Path(config.output_dir) / "_trainer_scratch"),
        architecture=config.architecture,
        max_length=config.max_length,
        load_in_4bit=config.load_in_4bit,
        local_files_only=config.local_files_only,
        initial_adapter_path=adapter,
    )
    settings.validate()
    model = _load_research_base(settings)
    model, _attachment = attach_research_adapter(model, settings=settings, lora=LoraSettings())
    if reference_adapter:
        model.load_adapter(str(reference_adapter), adapter_name=REFERENCE_ADAPTER, is_trainable=False)
        model.set_adapter(TRAINED_ADAPTER)
        for name, parameter in model.named_parameters():
            if f".{REFERENCE_ADAPTER}." in name and parameter.requires_grad:
                parameter.requires_grad_(False)
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        config.base_model, revision=config.base_revision, local_files_only=config.local_files_only
    )
    side_inputs = infer_required_side_input_names(model, processor, config.base_model)
    return model, processor, side_inputs


def _prompt_functions(
    config: RLConfig, adapter: str, processor: Any
) -> tuple[Callable[[Mapping[str, Any]], str], Callable[[str], tuple[list[int], int]], Any]:
    """The policy's own prompt renderer and tokenizer, bound to the training processor, and the renderer of the
    audit's choice for the supervised lifecycle term (:class:`psse_env.rl.targets.TargetRenderer`)."""

    from psse_env.dagger.release_factories import _ModelBundle
    from psse_env.dagger.research_policy_factory import _CanonicalResearchNativePolicy
    from psse_env.rl.targets import TargetRenderer

    bundle = _ModelBundle(model=None, processor=processor, model_id=config.base_model,
                          model_revision=config.base_revision, adapter_snapshot_path=str(adapter))
    policy = _CanonicalResearchNativePolicy(bundle, architecture=str(config.architecture or "gemma4_unified"))

    def render(observation: Mapping[str, Any]) -> str:
        rendered, _tools, _parameters = policy.render_prompt(observation)
        return rendered

    return render, policy.prompt_token_ids, TargetRenderer(policy, processor)


def _carry_adapter_over(source: str, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def _release_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # pragma: no cover - torch optional in tests
        pass


def _carried_optimizer_file(config: RLConfig, iteration: int) -> Path | None:
    """The optimizer state the latest earlier update of this run left, or ``None`` before the first.

    An iteration whose adapter was carried over unchanged took no step and
    left no state, so the search walks back past it.
    """

    for earlier in range(int(iteration) - 1, -1, -1):
        candidate = iteration_dir(config, earlier) / OPTIMIZER_FILE
        if candidate.is_file():
            return candidate
    return None


def _continuity_start(config: RLConfig, iteration: int) -> int:
    """The first iteration whose update's optimizer moments reach the update of ``iteration`` unbroken.

    Walking back from the iteration before it: an iteration that took no
    step (its adapter was carried over) breaks nothing, one that left its
    optimizer state extends the chain, and one that stepped without leaving
    a state ends it.
    """

    start = int(iteration)
    for earlier in range(int(iteration) - 1, -1, -1):
        directory = iteration_dir(config, earlier)
        if (directory / OPTIMIZER_FILE).is_file():
            start = earlier
            continue
        report_path = directory / "update_report.json"
        stepped = True
        if report_path.is_file():
            stepped = json.loads(report_path.read_text(encoding="utf-8")).get("update") is not None
        if stepped:
            break
    return start


def phase_update(config: RLConfig, manifest: Manifest, *, progress: Callable[[Mapping[str, Any]], None] | None = None) -> dict[str, Any]:
    out = iteration_dir(config, manifest.iteration)
    marker = out / MARKER_FILE
    if marker.is_file():
        raise IntegrityError(
            f"iteration {manifest.iteration}'s update failed its integrity checks ({marker}); read update_report.json, "
            "and remove the marker only once the cause is understood"
        )
    store = TrajectoryStore(out / "trajectories")
    groups, score_summary = score_groups(config, store)
    _write_json(out / "advantages.json", {
        **score_summary,
        "by_root": {adv.root: {"rewards": list(adv.rewards), "advantages": list(adv.advantages), "informative": adv.informative}
                    for _, adv in groups},
    })
    next_adapter = out / "adapter"
    weights = RewardWeights.from_mapping(config.reward)
    report: dict[str, Any] = {"iteration": manifest.iteration, "adapter_in": manifest.current_adapter, "scores": score_summary,
                              "lifecycle_credit": weights.lifecycle_credit, "lifecycle_form": weights.lifecycle_form}
    # The update renders every stored step again and takes the prompt the policy was given: under the same limit.
    report["prompt_limits"] = apply_prompt_limits(config)
    informative_paired = any(adv.informative for _, adv in groups) and any(
        trajectory.pairing_ok for trajectories, adv in groups if adv.informative for trajectory in trajectories
    )
    # A labelled lifecycle decision is trained whether or not its group is informative.
    lifecycle_trainable = weights.lifecycle_credit > 0.0 and int(score_summary["lifecycle"]["labelled_decisions_paired"]) > 0
    credit = config.credit_settings()
    decision_values: dict[tuple[str, int, int], dict[str, Any]] = {}
    if credit.enabled:
        # Values of this iteration's policy under this run's observation mode, or none at all.
        values_payload = load_values(out / "decision_values.json")
        check_provenance(values_payload, iteration=manifest.iteration, adapter=manifest.current_adapter,
                         observation_mode=config.observation_mode)
        decision_values = values_map(values_payload)
        report["decision_credit"] = {"weight": credit.weight, "values": len(decision_values),
                                     "selection": values_payload.get("selection"), "summary": values_payload.get("summary")}
        lifecycle_trainable = lifecycle_trainable or bool(decision_values)
    if not informative_paired and not lifecycle_trainable:
        report["update"] = None
        report["note"] = "no informative, paired group and no labelled lifecycle decision: the adapter is carried over unchanged"
        _carry_adapter_over(manifest.current_adapter, next_adapter)
        _write_json(out / "update_report.json", report)
        return report
    settings = config.update_settings(manifest.iteration)
    checkpoint = UpdateCheckpoint(directory=out / "update_checkpoint", save_model=_save_trained_adapter,
                                  every_rows=config.update_checkpoint_rows)
    resume_dir = checkpoint.model_dir()
    start_adapter = str(resume_dir) if resume_dir is not None else manifest.current_adapter
    reference = config.initial_adapter if settings.beta_kl > 0.0 else None
    model, processor, side_inputs = _load_trainable_model(config, start_adapter, reference_adapter=reference)
    try:
        render, prompt_ids, render_target = _prompt_functions(config, manifest.current_adapter, processor)
        rows, rows_report = build_rows(
            groups, render_prompt=render, prompt_token_ids=prompt_ids,
            max_length=config.max_length, required_side_inputs=side_inputs,
            settings=config.row_settings(manifest.iteration),
            lifecycle_credit=weights.lifecycle_credit,
            lifecycle_form=weights.lifecycle_form,
            render_target=render_target,
            **({"decision_values": decision_values, "decision_credit_weight": credit.weight} if credit.enabled else {}),
        )
        rows_report["target_renderer"] = {"rendered": render_target.rendered, "refused": render_target.refused}
        report["rows"] = rows_report
        report["resumed_from_checkpoint"] = str(resume_dir) if resume_dir is not None else None
        if not rows:
            report["update"] = None
            report["note"] = "no training rows after pairing and rendering checks"
            _carry_adapter_over(manifest.current_adapter, next_adapter)
        else:
            optimizer = None
            if settings.carry_optimizer:
                # The state before this update: a resumed update loads its checkpoint's state over it,
                # and one that starts again starts from the same place.
                carried = _carried_optimizer_file(config, manifest.iteration)
                carried_state = load_optimizer_state(carried) if carried is not None else None
                optimizer = build_optimizer(model, settings, carried_state)
                report["optimizer"] = {"carried_from": str(carried) if carried is not None else None,
                                       "steps_before": optimizer_steps_taken(carried_state)}
                # The first iteration whose moments this update continues (itself when it starts a new optimizer).
                report["optimizer"]["continuity_since_iteration"] = _continuity_start(config, manifest.iteration)
            update = policy_gradient_update(model, rows, settings, optimizer=optimizer, progress=progress,
                                            checkpoint=checkpoint)
            report["update"] = update.as_dict()
            temporary = out / "adapter.tmp"
            if temporary.exists():
                shutil.rmtree(temporary)
            _save_trained_adapter(model, temporary)
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
    report["observation_mode"] = config.observation_mode
    if config.series:
        report["series"] = dict(config.series)
    report["integrity"] = update_integrity(report)
    _write_json(out / "update_report.json", report)
    if config.stop_on_integrity_failure and not report["integrity"]["ok"]:
        _write_json(marker, {"iteration": manifest.iteration, "failures": report["integrity"]["failures"]})
        raise IntegrityError(
            f"iteration {manifest.iteration}'s update shows a stop condition: {'; '.join(report['integrity']['failures'])}"
        )
    return report


class EpisodeTelemetry:
    """What an evaluated episode cost the policy and the tools, summed from the evaluator's progress events."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.input_tokens = self.output_tokens = self.policy_calls = 0
        self.policy_seconds = self.generation_seconds = self.tool_seconds = 0.0
        self.token_records = 0

    def __call__(self, record: Mapping[str, Any]) -> None:
        if record.get("event") == "policy_action":
            self.policy_calls += 1
            self.policy_seconds += float(record.get("policy_seconds") or 0.0)
            metrics = record.get("policy_metrics") if isinstance(record.get("policy_metrics"), Mapping) else {}
            if metrics.get("prompt_tokens") is not None:
                self.token_records += 1
                self.input_tokens += int(metrics.get("prompt_tokens") or 0)
                self.output_tokens += int(metrics.get("generated_tokens") or 0)
                self.generation_seconds += float(metrics.get("generation_seconds") or 0.0)
        elif record.get("event") == "step_complete":
            self.tool_seconds += float(record.get("tool_seconds") or 0.0)

    def as_dict(self) -> dict[str, Any]:
        known = self.token_records > 0
        return {
            "policy_calls": self.policy_calls,
            "input_tokens": self.input_tokens if known else None,
            "output_tokens": self.output_tokens if known else None,
            "generation_seconds": round(self.generation_seconds, 3) if known else None,
            "policy_seconds": round(self.policy_seconds, 3),
            "tool_seconds": round(self.tool_seconds, 3),
        }


def internal_work(trace: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Solves and corrections the tools ran inside one call, which the count of tool calls does not show:
    the comparison tools' internal corrections and WLS solves, and the pair refits of a parameter correction."""

    work = {"internal_wls_calls": 0, "internal_correction_calls": 0, "pair_refits": 0}
    for step in trace:
        output = step.get("policy_tool_output") if isinstance(step, Mapping) and isinstance(step.get("policy_tool_output"), Mapping) else {}
        metrics = output.get("tool_metrics") if isinstance(output.get("tool_metrics"), Mapping) else {}
        for name in ("internal_wls_calls", "internal_correction_calls"):
            value = metrics.get(name, output.get(name))
            work[name] += int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0
        refit = metrics.get("pair_refit", output.get("pair_refit"))
        work["pair_refits"] += int(isinstance(refit, Mapping) and refit.get("status") == "refit")
    return work


def _compound_class(scenario: Mapping[str, Any] | None) -> str | None:
    audit = scenario.get("audit") if isinstance(scenario, Mapping) and isinstance(scenario.get("audit"), Mapping) else {}
    compound = audit.get("compound") if isinstance(audit.get("compound"), Mapping) else {}
    return str(compound["class"]) if compound.get("class") else None


def _compact_evaluation(
    episode: Mapping[str, Any], *, root: str, family: str, reward: Mapping[str, Any],
    lifecycle: Sequence[Mapping[str, Any]] = (), scenario: Mapping[str, Any] | None = None,
    telemetry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    trace = [step for step in episode.get("trace") or [] if isinstance(step, Mapping)]
    return {
        "scenario_id": root,
        "family": family,
        # The audit's class of a compound root, for the reports' strata; the policy never sees it.
        "compound_class": _compound_class(scenario),
        "terminal": episode.get("terminal"),
        "healthy_preservation_known": episode.get("healthy_preservation_known"),
        "wls_calls": episode.get("wls_calls"),
        "prompt_limit_refusals": prompt_limit_refusals(trace),
        "cost": {**internal_work(trace), **dict(telemetry or {})},
        "truth_audited_task_success": bool(episode.get("truth_audited_task_success")),
        "basis": episode_basis(episode),
        "steps": episode.get("steps"),
        "policy_steps": episode.get("policy_steps"),
        "false_commit_count": episode.get("false_commit_count"),
        "false_finalization_count": episode.get("false_finalization_count"),
        "loop_detected": bool(episode.get("loop_detected")),
        "invalid_action_count": episode.get("invalid_action_count"),
        "healthy_components_preserved": episode.get("healthy_components_preserved"),
        "tools": [str((step.get("action") or {}).get("tool")) for step in episode.get("trace") or [] if isinstance(step, Mapping)],
        "reward": dict(reward),
        # The episode's decisions on accepted candidates with the audit's label (psse_env.rl.lifecycle).
        "lifecycle": [dict(decision) for decision in lifecycle],
    }


def _record_steps(record: Mapping[str, Any]) -> int:
    return int(record.get("policy_steps") or record.get("steps") or 0)


def aggregate_evaluation(records: Sequence[Mapping[str, Any]], budgets: Sequence[int] = ()) -> dict[str, Any]:
    """Success, cost and safety of a greedy evaluation, overall and per family.

    ``solved_within_actions[b]`` counts the solved cases that used at most
    ``b`` actions under the run's own budget.  The policy sees its remaining
    budget, so this is not the same as a run with ``max_steps`` ``b``; it is
    the efficiency measure of the RL plan (section 4), reported beside the
    solved count.
    """

    count = len(records)
    solved = [record for record in records if record.get("truth_audited_task_success")]
    by_family: dict[str, dict[str, int]] = {}
    for record in records:
        entry = by_family.setdefault(str(record.get("family")), {"roots": 0, "successes": 0})
        entry["roots"] += 1
        entry["successes"] += int(bool(record.get("truth_audited_task_success")))

    def total(key: str) -> int:
        return sum(int(record.get(key) or 0) for record in records)

    decisions = [decision for record in records for decision in record.get("lifecycle") or [] if isinstance(decision, Mapping)]
    costs = [record.get("cost") for record in records if isinstance(record.get("cost"), Mapping)]

    def cost_total(key: str) -> int | float | None:
        values = [cost.get(key) for cost in costs if cost.get(key) is not None]
        return sum(values) if values else None

    return {
        "roots": count,
        # Solved with no refused action, no refused commit and no loop.
        "clean_success": sum(1 for record in solved if not int(record.get("invalid_action_count") or 0)
                             and not int(record.get("false_commit_count") or 0) and not record.get("loop_detected")),
        "healthy_component_corrupted": sum(1 for record in records
                                           if (record.get("reward") or {}).get("healthy_component_corrupted")),
        "prompt_limit_refusals": total("prompt_limit_refusals"),
        "cost": {key: cost_total(key) for key in ("input_tokens", "output_tokens", "generation_seconds", "policy_seconds",
                                                  "tool_seconds", "internal_wls_calls", "internal_correction_calls", "pair_refits")},
        "truth_audited_task_success": len(solved),
        "mean_steps": (sum(int(record.get("steps") or 0) for record in records) / count) if count else None,
        "mean_steps_solved": (sum(_record_steps(record) for record in solved) / len(solved)) if solved else None,
        "solved_within_actions": {str(int(budget)): sum(1 for record in solved if _record_steps(record) <= int(budget))
                                  for budget in budgets},
        "mean_reward": (sum(float((record.get("reward") or {}).get("total") or 0.0) for record in records) / count) if count else None,
        "false_commit_count": total("false_commit_count"),
        "false_finalization_count": total("false_finalization_count"),
        "invalid_action_count": total("invalid_action_count"),
        "loop_episodes": sum(1 for record in records if record.get("loop_detected")),
        "lifecycle": {
            **summarize_decisions(decisions),
            "episodes_with_decline": sum(1 for record in records if any(
                decision.get("declined") for decision in record.get("lifecycle") or [] if isinstance(decision, Mapping))),
        },
        "by_family": dict(sorted(by_family.items())),
    }


def _greedy_evaluation(
    config: RLConfig, *, adapter: str, out: Path, shard: tuple[int, int] | None, summary_head: Mapping[str, Any]
) -> dict[str, Any]:
    """Greedy evaluation of ``adapter`` on the development suite, stored per root under ``out/evaluation/episodes``.

    A shard evaluates its slice and returns its counts; the unsharded call
    finishes any root left over and writes ``out/evaluation.json`` and
    ``out/evaluation_episodes.jsonl``.
    """

    from psse_env.dagger.evaluator import evaluate_rollout_suites
    from psse_env.dagger.release_factories import deterministic_case_loader
    from psse_env.dagger.research_policy_factory import clear_research_policy_cache, research_gemma_policy_factory
    from scripts.run_dagger_research import PAIRED_EVALUATION_CONTRACT

    all_scenarios = load_suite(config.development_suite)
    scenarios = shard_scenarios(all_scenarios, shard)
    limits = apply_prompt_limits(config)
    episodes_dir = out / "evaluation" / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    pending = [s for s in scenarios if not (episodes_dir / f"{_safe_name(scenario_root(s))}.json").is_file()]
    if pending:
        env_factory = configure_research_environment(config)
        weights = RewardWeights.from_mapping(config.reward)
        reference = load_expert_reference(config.expert_reference)
        policy = research_gemma_policy_factory(
            adapter, base_model=config.base_model, base_revision=config.base_revision,
            load_in_4bit=config.load_in_4bit, local_files_only=config.local_files_only,
            prompt_profile=config.prompt_profile, architecture=config.architecture,
        )
        telemetry = EpisodeTelemetry()
        try:
            for scenario in pending:
                root = scenario_root(scenario)
                family = scenario_family(scenario)
                telemetry.reset()
                result = evaluate_rollout_suites(
                    {"standard_success": [scenario]}, env_factory=env_factory,
                    policy_factory=lambda **_kwargs: policy, max_steps=config.max_steps, seed=config.seed,
                    case_loader=deterministic_case_loader, **PAIRED_EVALUATION_CONTRACT,
                    progress_callback=telemetry, **config.evaluator_options(),
                ).as_dict()
                episode = dict(result["suite_metrics"]["episodes"][0])
                decisions = lifecycle_decisions(episode.get("trace") or [], scenario_truth(scenario))
                reward = episode_reward(episode, family=family, expert_reference_calls=reference.get(root), weights=weights,
                                        kept_wrong_targets=summarize_decisions(decisions)["wrong_target_committed"])
                _write_json(episodes_dir / f"{_safe_name(root)}.json",
                            _compact_evaluation(episode, root=root, family=family, reward=reward.as_dict(), lifecycle=decisions,
                                                scenario=scenario, telemetry=telemetry.as_dict()))
                print(json.dumps({"evaluated": root, "family": family,
                                  "success": bool(episode.get("truth_audited_task_success")),
                                  "steps": episode.get("steps"), "reward": round(reward.total, 3),
                                  "declines": sum(1 for decision in decisions if decision.get("declined"))}), flush=True)
        finally:
            del policy
            clear_research_policy_cache()
            _release_gpu()
    if shard is not None:
        return {"shard": f"{shard[0]}/{shard[1]}", "evaluated": len(pending), "roots": len(scenarios)}
    records = []
    for scenario in all_scenarios:
        path = episodes_dir / f"{_safe_name(scenario_root(scenario))}.json"
        records.append(json.loads(path.read_text(encoding="utf-8")))
    summary = {**summary_head, "adapter": adapter, "observation_mode": config.observation_mode,
               "precision": "4bit" if config.load_in_4bit else "bf16", "max_steps": config.max_steps,
               "prompt_limits": limits,
               **({"series": dict(config.series)} if config.series else {}),
               **aggregate_evaluation(records, config.evaluation_budgets)}
    _write_json(out / "evaluation.json", {"summary": summary})
    with (out / "evaluation_episodes.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    return summary


def phase_baseline(config: RLConfig, manifest: Manifest, *, shard: tuple[int, int] | None = None) -> dict[str, Any]:
    """Greedy evaluation of the starting adapter on the development suite before the first rollout (``baseline_evaluation``).

    Measured by the driver's own loader, so every iteration's score compares
    with a baseline measured the same way; stored under ``<output_dir>/baseline``.
    """

    del manifest
    if not config.development_suite:
        raise ValueError("the baseline evaluation needs a development suite")
    return _greedy_evaluation(config, adapter=config.initial_adapter, out=Path(config.output_dir) / "baseline", shard=shard,
                              summary_head={"baseline": True})


def phase_evaluate(config: RLConfig, manifest: Manifest, *, shard: tuple[int, int] | None = None) -> dict[str, Any] | None:
    """Greedy evaluation of the iteration's adapter on the development suite, stored per root.

    Skipped when the update phase carried the adapter over unchanged: its
    score is the previous iteration's (or the baseline's).
    """

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
    return _greedy_evaluation(config, adapter=str(out / "adapter"), out=out, shard=shard,
                              summary_head={"iteration": manifest.iteration})


def run_phase(
    config: RLConfig,
    phase: str,
    *,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
    shard: tuple[int, int] | None = None,
) -> dict[str, Any] | None:
    manifest_path = Path(config.output_dir) / "manifest.json"
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(config)
    if phase != manifest.phase:
        raise RuntimeError(f"the run is at phase {manifest.phase!r} of iteration {manifest.iteration}, not {phase!r}")
    if shard is not None and phase not in SHARDED_PHASES:
        raise ValueError(f"only the {', '.join(SHARDED_PHASES)} phases are sharded")
    started = time.perf_counter()
    if shard is not None:
        if phase == "rollout":
            result = phase_rollout(config, manifest, progress=progress, shard=shard)
        elif phase == "continuation":
            result = phase_continuation(config, manifest, progress=progress, shard=shard)
        elif phase == "baseline":
            result = phase_baseline(config, manifest, shard=shard)
        else:
            result = phase_evaluate(config, manifest, shard=shard)
        print(json.dumps({"phase": phase, "shard": f"{shard[0]}/{shard[1]}", "iteration": manifest.iteration,
                          "seconds": round(time.perf_counter() - started, 1), "result": result}, default=str), flush=True)
        return result
    if phase == "baseline":
        result = phase_baseline(config, manifest)
        manifest.baseline = {"adapter": config.initial_adapter, "evaluation": result}
        manifest.phase = "rollout"
    elif phase == "rollout":
        result = phase_rollout(config, manifest, progress=progress)
        manifest.phase = iteration_phases(config)[1]
    elif phase == "continuation":
        result = phase_continuation(config, manifest, progress=progress)
        manifest.phase = "update"
    elif phase == "update":
        result = phase_update(config, manifest, progress=progress)
        manifest.phase = "evaluate"
    elif phase == "evaluate":
        result = phase_evaluate(config, manifest)
        manifest.history.append({
            "iteration": manifest.iteration,
            "adapter": str(iteration_dir(config, manifest.iteration) / "adapter"),
            "evaluation": result,
        })
        manifest.current_adapter = str(iteration_dir(config, manifest.iteration) / "adapter")
        manifest.iteration += 1
        manifest.phase = "rollout"
    else:
        raise ValueError(f"unknown phase {phase!r}")
    manifest.save(manifest_path)
    print(json.dumps({"phase": phase, "iteration": manifest.iteration, "seconds": round(time.perf_counter() - started, 1),
                      "result": result}, default=str), flush=True)
    return result


def run_iterations(config: RLConfig, iterations: int, *, progress: Callable[[Mapping[str, Any]], None] | None = None) -> None:
    for _ in range(int(iterations)):
        manifest = load_manifest(config)
        if manifest.phase == "baseline":
            # Before the first iteration, not one of them.
            run_phase(config, "baseline", progress=progress)
            manifest = load_manifest(config)
        phases = iteration_phases(config)
        for phase in phases[phases.index(manifest.phase):]:
            run_phase(config, phase, progress=progress)


def status(config: RLConfig) -> dict[str, Any]:
    return asdict(load_manifest(config))


def _print_progress(record: Mapping[str, Any]) -> None:
    print(json.dumps(dict(record), default=str), flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--phase", choices=PHASES, default=None,
                        help="run one phase of the current iteration (baseline: the starting adapter's evaluation before the first)")
    parser.add_argument("--shard", default=None,
                        help="'i/n': this worker's slice of the baseline, rollout, continuation or evaluation phase; "
                             "the unsharded call closes the phase")
    parser.add_argument("--iterations", type=int, default=1, help="iterations to run when no phase is given")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args(argv)
    config = RLConfig.load(args.config)
    if args.status:
        print(json.dumps(status(config), indent=2))
        return 0
    if args.phase:
        run_phase(config, args.phase, progress=_print_progress, shard=parse_shard(args.shard))
        return 0
    if args.shard:
        raise SystemExit("--shard needs --phase")
    run_iterations(config, args.iterations, progress=_print_progress)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
