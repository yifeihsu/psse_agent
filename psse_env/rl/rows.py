"""Training rows from recorded steps: the prompt the policy decoded from, the tokens it sampled.

A row is exact on-policy data: the prompt is re-rendered from the stored
model observation with the same renderer the policy used, checked against
the recorded prompt length, and the completion is the recorded token list.
No target is re-rendered from the parsed action, so the trained tokens are
the sampled ones.  Rows carry their trajectory's advantage; a group with no
advantage contributes no rows.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from psse_env.rl.advantages import GroupAdvantages
from psse_env.rl.rollout import Trajectory

ROWS_CONTRACT = "rl_training_rows_v1"

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
    behavior_logprobs: list[float] = field(default_factory=list)

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
    token_probabilities: Sequence[float] = (),
) -> TrainingRow:
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    prompt = [int(value) for value in prompt_ids]
    completion = [int(value) for value in completion_ids]
    if not completion:
        raise ValueError("a training row needs at least one completion token")
    if len(completion) >= max_length:
        raise ValueError(f"completion of {len(completion)} tokens does not fit max_length={max_length}")
    probabilities = [float(value) for value in token_probabilities]
    if probabilities and (len(probabilities) != len(completion) or
                          any(not math.isfinite(p) or not 0 < p <= 1 for p in probabilities)):
        raise ValueError("sampling probabilities must match the completion and lie in (0, 1]")
    full = prompt + completion
    slice_start = max(0, len(full) - int(max_length))
    used = full[slice_start:]
    prompt_kept = len(prompt) - slice_start
    completion_mask = [0] * prompt_kept + [1] * len(completion)
    labels = [token if keep else -100 for token, keep in zip(used, completion_mask)]
    side_inputs = {name: [0] * len(used) for name in dict.fromkeys(required_side_inputs)}
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
        behavior_logprobs=[math.log(p) for p in probabilities],
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
        "invalid_probability_rows": 0,
        "completion_tokens": 0,
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
                if len(prompt_ids) + len(step.completion_token_ids) > max_length:
                    # A shorter training context is a different conditional policy.
                    report["prompt_truncated_rows"] += 1
                    continue
                try:
                    row = build_step_row(
                        root=trajectory.root,
                        sample_index=trajectory.sample_index,
                        step=step.step,
                        advantage=advantage,
                        prompt_ids=prompt_ids,
                        completion_ids=step.completion_token_ids,
                        max_length=max_length,
                        required_side_inputs=required_side_inputs,
                        token_probabilities=step.token_probabilities,
                    )
                except ValueError:
                    report["invalid_probability_rows"] += 1
                    continue
                rows.append(row)
                report["rows"] += 1
                report["completion_tokens"] += row.completion_tokens
    return rows, report


__all__ = ["ROWS_CONTRACT", "PromptRenderer", "PromptTokenizer", "TrainingRow", "build_rows", "build_step_row"]
