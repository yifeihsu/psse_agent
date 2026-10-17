"""Training rows from recorded steps: the prompt the policy decoded from, the tokens it sampled.

A row is exact on-policy data: the prompt is re-rendered from the stored
model observation with the same renderer the policy used, checked against
the recorded prompt length (and, when the record carries one, the digest of
the prompt token ids), and the completion is the recorded token list.  No
target is re-rendered from the parsed action, so the trained tokens are the
sampled ones.  Rows carry their trajectory's advantage.  When the policy
recorded the probability of each token it sampled, the row also carries
those as log-probabilities, the reference the update checks itself against
(and clips against, when asked).

Two kinds of rows (:class:`RowSettings`, 2026-10-17):

- **policy rows** come from informative groups (samples that scored
  differently) and carry the group advantage;
- **preservation rows** come from groups whose samples all scored the same.
  They carry a zero advantage, so they add nothing to the policy gradient,
  and exist so the KL penalty to the starting adapter also covers the states
  the policy already handles the same way every time.  Without them a KL
  anchor sees only the contexts of the informative groups (the external
  review of 2026-10-17, item 2.3).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from psse_env.rl.advantages import GroupAdvantages
from psse_env.rl.prompt_digest import prompt_token_digest
from psse_env.rl.rollout import Trajectory

ROWS_CONTRACT = "rl_training_rows_v3"
#: A recorded sampling probability below this is clamped before the log, so a
#: token whose probability the sampler's float32 softmax rounded to zero still
#: gives a finite reference log-probability.
MIN_SAMPLING_PROBABILITY = 1e-12
PRESERVATION_MODES = ("none", "one", "all")

PromptRenderer = Callable[[Mapping[str, Any]], str]
PromptTokenizer = Callable[[str], tuple[list[int], int]]


@dataclass(frozen=True)
class RowSettings:
    """What :func:`build_rows` admits; the defaults keep the pre-2026-10-17 behaviour."""

    #: Skip an informative group when any of its trajectories is unpaired,
    #: instead of training on the paired ones with advantages computed over
    #: all of them.
    require_complete_groups: bool = False
    #: Refuse rows whose prompt would be truncated again on the update side
    #: (the trained context would differ from the sampled one).
    reject_truncated_prompts: bool = False
    #: Refuse rows whose record carries no prompt digest or whose re-rendered
    #: prompt has a different one.
    require_prompt_digest: bool = False
    #: Refuse rows without recorded sampling probabilities (a clipped update
    #: needs the behaviour log-probabilities of every row).
    require_sampling_records: bool = False
    #: ``none``: no preservation rows; ``one``: the first paired trajectory of
    #: each uninformative group; ``all``: every paired trajectory of them.
    preservation: str = "none"
    #: At most this many uninformative groups contribute preservation rows
    #: (a seeded draw); ``None`` is all of them.
    preservation_max_groups: int | None = None
    seed: int = 0

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "RowSettings":
        payload = dict(payload or {})
        unknown = sorted(set(payload) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown rows fields: {unknown}")
        settings = cls(**payload)
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.preservation not in PRESERVATION_MODES:
            raise ValueError(f"rows.preservation must be one of {PRESERVATION_MODES}, got {self.preservation!r}")
        if self.preservation_max_groups is not None and int(self.preservation_max_groups) < 0:
            raise ValueError("rows.preservation_max_groups must be nonnegative when set")


@dataclass
class TrainingRow:
    root: str
    sample_index: int
    step: int
    advantage: float
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    completion_mask: list[int]
    side_inputs: dict[str, list[int]] = field(default_factory=dict)
    prompt_truncated: bool = False
    #: Natural log of each completion token's probability under the
    #: distribution the policy sampled it from, as the policy recorded it;
    #: ``None`` when the step carried no record or the record did not match
    #: the completion token for token.
    sampling_logprobs: list[float] | None = None
    #: Completion tokens of the whole trajectory this row belongs to, set by
    #: :func:`build_rows`; the ``trajectory`` loss normalization divides by
    #: it so a trajectory weighs the same whatever its length.
    trajectory_tokens: int | None = None
    #: Whether the re-rendered prompt's digest equals the recorded one;
    #: ``None`` when the record carried no digest.
    prompt_digest_matches: bool | None = None
    #: A preservation row: zero advantage, present for the KL term only.
    preservation: bool = False

    @property
    def completion_tokens(self) -> int:
        return int(sum(self.completion_mask))

    @property
    def trajectory_key(self) -> tuple[str, int]:
        return (self.root, int(self.sample_index))

    def model_record(self) -> dict[str, list[int]]:
        record = {
            "input_ids": list(self.input_ids),
            "attention_mask": list(self.attention_mask),
            "labels": list(self.labels),
            "completion_mask": list(self.completion_mask),
        }
        record.update({key: list(value) for key, value in self.side_inputs.items()})
        return record


def sampling_logprobs_for(
    completion_ids: Sequence[int], probabilities: Sequence[float] | None
) -> list[float] | None:
    """The recorded sampling probabilities as log-probabilities, or ``None`` when they do not pair with the completion."""

    if probabilities is None:
        return None
    values = [float(value) for value in probabilities]
    if len(values) != len(completion_ids) or not values:
        return None
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 + 1e-6 for value in values):
        return None
    return [math.log(min(1.0, max(value, MIN_SAMPLING_PROBABILITY))) for value in values]


def build_step_row(
    *,
    root: str,
    sample_index: int,
    step: int,
    advantage: float,
    prompt_ids: Sequence[int],
    completion_ids: Sequence[int],
    max_length: int,
    required_side_inputs: Sequence[str] = (),
    sampling_probabilities: Sequence[float] | None = None,
    prompt_digest: str | None = None,
    preservation: bool = False,
) -> TrainingRow:
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    prompt = [int(value) for value in prompt_ids]
    completion = [int(value) for value in completion_ids]
    if not completion:
        raise ValueError("a training row needs at least one completion token")
    if len(completion) >= max_length:
        raise ValueError(f"completion of {len(completion)} tokens does not fit max_length={max_length}")
    full = prompt + completion
    slice_start = max(0, len(full) - int(max_length))
    used = full[slice_start:]
    prompt_kept = len(prompt) - slice_start
    completion_mask = [0] * prompt_kept + [1] * len(completion)
    labels = [token if keep else -100 for token, keep in zip(used, completion_mask)]
    side_inputs = {name: [0] * len(used) for name in dict.fromkeys(required_side_inputs)}
    sampling_logprobs = sampling_logprobs_for(completion, sampling_probabilities)
    return TrainingRow(
        root=root,
        sample_index=sample_index,
        step=step,
        advantage=float(advantage),
        input_ids=used,
        attention_mask=[1] * len(used),
        labels=labels,
        completion_mask=completion_mask,
        side_inputs=side_inputs,
        prompt_truncated=slice_start > 0,
        sampling_logprobs=sampling_logprobs,
        prompt_digest_matches=(prompt_token_digest(prompt) == prompt_digest) if prompt_digest else None,
        preservation=bool(preservation),
    )


def build_rows(
    groups: Sequence[tuple[Sequence[Trajectory], GroupAdvantages]],
    *,
    render_prompt: PromptRenderer,
    prompt_token_ids: PromptTokenizer,
    max_length: int,
    required_side_inputs: Sequence[str] = (),
    include_invalid_steps: bool = True,
    settings: RowSettings | None = None,
) -> tuple[list[TrainingRow], dict[str, Any]]:
    """Rows for every step of every informative, correctly paired trajectory, plus the preservation rows ``settings`` ask for."""

    settings = settings or RowSettings()
    settings.validate()
    rows: list[TrainingRow] = []
    report: dict[str, Any] = {
        "contract": ROWS_CONTRACT,
        "settings": {k: getattr(settings, k) for k in RowSettings.__dataclass_fields__},
        "groups": len(groups),
        "informative_groups": 0,
        "groups_incomplete": 0,
        "trajectories": 0,
        "trajectories_unpaired": 0,
        "steps": 0,
        "rows": 0,
        "policy_rows": 0,
        "preservation_groups": 0,
        "preservation_rows": 0,
        "preservation_tokens": 0,
        "prompt_length_mismatches": 0,
        "prompt_digest_missing": 0,
        "prompt_digest_mismatches": 0,
        "empty_completions": 0,
        "invalid_steps_excluded": 0,
        "prompt_truncated_rows": 0,
        "prompt_truncated_rows_rejected": 0,
        "rows_without_sampling_record_rejected": 0,
        "completion_tokens": 0,
        "rows_with_sampling_probabilities": 0,
    }

    def trajectory_rows(trajectory: Trajectory, advantage: float, *, preservation: bool) -> list[TrainingRow]:
        built: list[TrainingRow] = []
        for step in trajectory.steps:
            report["steps"] += 1
            if not include_invalid_steps and step.execution_status != "success":
                report["invalid_steps_excluded"] += 1
                continue
            if not step.completion_token_ids:
                report["empty_completions"] += 1
                continue
            rendered = render_prompt(step.model_observation)
            prompt_ids, _original = prompt_token_ids(rendered)
            if step.prompt_tokens and len(prompt_ids) != int(step.prompt_tokens):
                report["prompt_length_mismatches"] += 1
                continue
            if settings.require_prompt_digest and not step.prompt_digest:
                report["prompt_digest_missing"] += 1
                continue
            row = build_step_row(
                root=trajectory.root,
                sample_index=trajectory.sample_index,
                step=step.step,
                advantage=advantage,
                prompt_ids=prompt_ids,
                completion_ids=step.completion_token_ids,
                max_length=max_length,
                required_side_inputs=required_side_inputs,
                sampling_probabilities=step.token_probabilities or None,
                prompt_digest=step.prompt_digest,
                preservation=preservation,
            )
            if row.prompt_digest_matches is False:
                report["prompt_digest_mismatches"] += 1
                if settings.require_prompt_digest:
                    continue
            if row.prompt_truncated:
                report["prompt_truncated_rows"] += 1
                if settings.reject_truncated_prompts:
                    report["prompt_truncated_rows_rejected"] += 1
                    continue
            if settings.require_sampling_records and row.sampling_logprobs is None:
                report["rows_without_sampling_record_rejected"] += 1
                continue
            built.append(row)
        total_tokens = sum(row.completion_tokens for row in built)
        for row in built:
            row.trajectory_tokens = total_tokens
            report["rows"] += 1
            report["completion_tokens"] += row.completion_tokens
            if row.sampling_logprobs is not None:
                report["rows_with_sampling_probabilities"] += 1
        return built

    preservation_candidates: list[Sequence[Trajectory]] = []
    for trajectories, advantages in groups:
        if not advantages.informative:
            if settings.preservation != "none":
                preservation_candidates.append(trajectories)
            continue
        if len(advantages.advantages) != len(trajectories):
            raise ValueError(f"group {advantages.root} has {len(trajectories)} trajectories and {len(advantages.advantages)} advantages")
        if settings.require_complete_groups and not all(trajectory.pairing_ok for trajectory in trajectories):
            report["groups_incomplete"] += 1
            report["trajectories_unpaired"] += sum(1 for trajectory in trajectories if not trajectory.pairing_ok)
            continue
        report["informative_groups"] += 1
        for trajectory, advantage in zip(trajectories, advantages.advantages):
            report["trajectories"] += 1
            if not trajectory.pairing_ok:
                report["trajectories_unpaired"] += 1
                continue
            built = trajectory_rows(trajectory, advantage, preservation=False)
            rows.extend(built)
            report["policy_rows"] += len(built)

    if preservation_candidates:
        order = list(range(len(preservation_candidates)))
        random.Random(f"{settings.seed}:preservation").shuffle(order)
        if settings.preservation_max_groups is not None:
            order = order[: int(settings.preservation_max_groups)]
        for index in sorted(order):
            paired = [trajectory for trajectory in preservation_candidates[index] if trajectory.pairing_ok]
            if not paired:
                continue
            chosen = paired[:1] if settings.preservation == "one" else paired
            built: list[TrainingRow] = []
            for trajectory in chosen:
                report["trajectories"] += 1
                built.extend(trajectory_rows(trajectory, 0.0, preservation=True))
            if built:
                report["preservation_groups"] += 1
                report["preservation_rows"] += len(built)
                report["preservation_tokens"] += sum(row.completion_tokens for row in built)
                rows.extend(built)
    return rows, report


__all__ = [
    "MIN_SAMPLING_PROBABILITY",
    "PRESERVATION_MODES",
    "ROWS_CONTRACT",
    "PromptRenderer",
    "PromptTokenizer",
    "RowSettings",
    "TrainingRow",
    "build_rows",
    "build_step_row",
    "sampling_logprobs_for",
]
