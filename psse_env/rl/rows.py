"""Training rows from recorded steps: the prompt the policy decoded from, the tokens it sampled.

A row is exact on-policy data: the prompt is re-rendered from the stored
model observation with the same renderer the policy used, checked against
the recorded prompt length, and the completion is the recorded token list.
No target is re-rendered from the parsed action, so the trained tokens are
the sampled ones.  Rows carry their trajectory's advantage; a group with no
advantage contributes no rows.  When the policy recorded the probability of
each token it sampled, the row also carries those as log-probabilities, the
reference the update checks itself against (and clips against, when asked).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from psse_env.rl.advantages import GroupAdvantages
from psse_env.rl.rollout import Trajectory

ROWS_CONTRACT = "rl_training_rows_v2"
#: A recorded sampling probability below this is clamped before the log, so a
#: token whose probability the sampler's float32 softmax rounded to zero still
#: gives a finite reference log-probability.
MIN_SAMPLING_PROBABILITY = 1e-12

PromptRenderer = Callable[[Mapping[str, Any]], str]
PromptTokenizer = Callable[[str], tuple[list[int], int]]


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

    @property
    def completion_tokens(self) -> int:
        return int(sum(self.completion_mask))

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
    )


def build_rows(
    groups: Sequence[tuple[Sequence[Trajectory], GroupAdvantages]],
    *,
    render_prompt: PromptRenderer,
    prompt_token_ids: PromptTokenizer,
    max_length: int,
    required_side_inputs: Sequence[str] = (),
    include_invalid_steps: bool = True,
) -> tuple[list[TrainingRow], dict[str, Any]]:
    """Rows for every step of every informative, correctly paired trajectory."""

    rows: list[TrainingRow] = []
    report: dict[str, Any] = {
        "contract": ROWS_CONTRACT,
        "groups": len(groups),
        "informative_groups": 0,
        "trajectories": 0,
        "trajectories_unpaired": 0,
        "steps": 0,
        "rows": 0,
        "prompt_length_mismatches": 0,
        "empty_completions": 0,
        "invalid_steps_excluded": 0,
        "prompt_truncated_rows": 0,
        "completion_tokens": 0,
        "rows_with_sampling_probabilities": 0,
    }
    for trajectories, advantages in groups:
        if not advantages.informative:
            continue
        report["informative_groups"] += 1
        if len(advantages.advantages) != len(trajectories):
            raise ValueError(f"group {advantages.root} has {len(trajectories)} trajectories and {len(advantages.advantages)} advantages")
        for trajectory, advantage in zip(trajectories, advantages.advantages):
            report["trajectories"] += 1
            if not trajectory.pairing_ok:
                report["trajectories_unpaired"] += 1
                continue
            trajectory_rows: list[TrainingRow] = []
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
                )
                rows.append(row)
                trajectory_rows.append(row)
                report["rows"] += 1
                report["completion_tokens"] += row.completion_tokens
                if row.sampling_logprobs is not None:
                    report["rows_with_sampling_probabilities"] += 1
            total_tokens = sum(row.completion_tokens for row in trajectory_rows)
            for row in trajectory_rows:
                row.trajectory_tokens = total_tokens
                if row.prompt_truncated:
                    report["prompt_truncated_rows"] += 1
    return rows, report


__all__ = [
    "MIN_SAMPLING_PROBABILITY",
    "ROWS_CONTRACT",
    "PromptRenderer",
    "PromptTokenizer",
    "TrainingRow",
    "build_rows",
    "build_step_row",
    "sampling_logprobs_for",
]
