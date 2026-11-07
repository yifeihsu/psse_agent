"""The audit's choice at a lifecycle step, rendered as the completion the student was trained to emit.

This is the supervised form of the lifecycle term (RL plan of 2026-11-05,
section 6 and decision 5; the plan session's correction of 2026-11-07).  The
per-decision form of :mod:`psse_env.rl.lifecycle` scores the tokens the
policy sampled, so for a two-way choice it moves the choice in proportion to
p(1 - p), p being the probability of the audit's choice: a choice the policy
seldom samples barely moves, and two such choices hold the prize (declining a
wrong accepted candidate, committing a partial candidate after a decline).
The supervised form trains the audit's choice itself, in proportion to 1 - p.

A supervised row has the step's own prompt, the one the policy decoded from
(digest-checked like every row), and as its completion the labelled tool call
rendered the way the SFT rows of the cloning pass were
(``psse_env.sft.gates.prepare_example``): the assistant turn appended to the
prompt's messages through the chat template, the empty-thought-channel
normalization, the generation prompt as an exact prefix of the full render,
the completion as the token suffix, and a round trip of the rendered call
through the policy's own parser and schema check.  The labelled call keeps the
arguments the policy wrote (the candidate alias in ``case_path``) and swaps the
tool name: ``commit_state`` and ``rollback_state`` share one argument schema
(``psse_env.dagger.protocol_bridge``).

Nothing here reads the truth; the label comes with the decision record.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from psse_env.rl.lifecycle import DECISION_TOOLS

TARGETS_CONTRACT = "rl_supervised_lifecycle_targets_v1"
_MODEL_TURN_MARKER = "<|turn>model\n"
_EMPTY_THOUGHT = "<|channel>thought\n<channel|>"


class TargetError(ValueError):
    """The audit's choice could not be rendered as a completion the student would emit."""


@dataclass(frozen=True)
class TargetCompletion:
    text: str
    prompt_ids: list[int]
    completion_ids: list[int]
    empty_thought_injected: bool


def labelled_action(text: str, decision: Mapping[str, Any]) -> dict[str, Any]:
    """The audit's choice as a canonical action: the sampled call's arguments with the labelled tool name.

    ``text`` is the step's generated text (``StepRecord.text``), which must
    parse to the decision's own tool; ``decision`` is a record of
    :func:`psse_env.rl.lifecycle.lifecycle_decisions` with a label.
    """

    from psse_env.sft.gates import GateError, parse_tool_call

    label = decision.get("label")
    if label not in DECISION_TOOLS:
        raise TargetError(f"the decision carries no label to render, got {label!r}")
    try:
        parsed = parse_tool_call(text)
    except GateError as exc:
        raise TargetError(f"the sampled text of the decision does not parse to one tool call: {exc}") from exc
    if parsed.name != decision.get("tool"):
        raise TargetError(f"the sampled text calls {parsed.name!r}, not the decision's {decision.get('tool')!r}")
    if not isinstance(parsed.arguments, Mapping):
        raise TargetError("the sampled call carries no arguments object")
    return {"tool": str(label), "arguments": copy.deepcopy(dict(parsed.arguments))}


def normalize_thought_prefix(full: str, rendered_prompt: str) -> tuple[str, bool]:
    """Make a full render start with the generation prompt where the template adds the empty thought channel to a
    generation prompt only (the normalization of ``psse_env.sft.gates.prepare_example``)."""

    if full.startswith(rendered_prompt):
        return full, False
    marker_end = rendered_prompt.rfind(_MODEL_TURN_MARKER)
    if marker_end == -1:
        return full, False
    marker_end += len(_MODEL_TURN_MARKER)
    base = rendered_prompt[:marker_end]
    if rendered_prompt[marker_end:] == _EMPTY_THOUGHT and full.startswith(base):
        return base + _EMPTY_THOUGHT + full[len(base):], True
    return full, False


def _token_ids(encoded: Any) -> list[int]:
    ids = encoded["input_ids"] if hasattr(encoded, "__getitem__") else getattr(encoded, "input_ids")
    if hasattr(ids, "reshape"):
        return [int(value) for value in ids.reshape(-1).tolist()]
    if ids and isinstance(ids[0], (list, tuple)):
        if len(ids) != 1:
            raise TargetError(f"expected one encoded sample, got {len(ids)}")
        ids = ids[0]
    return [int(value) for value in ids]


def render_target_completion(
    processor: Any,
    *,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    rendered_prompt: str,
    action: Mapping[str, Any],
    enable_thinking: bool = False,
) -> TargetCompletion:
    """Render ``action`` as the assistant turn after ``messages`` and return its completion tokens.

    The prompt is tokenized the way the policy tokenizes its generation
    prompt (``psse_env.sft.gemma_text.tokenize_rendered_text``), so the
    completion ids follow the ids a sampled row's prompt would have before
    its left truncation.
    """

    from psse_env.sft.gemma_text import render_text_with_stringified_content_fallback, tokenize_rendered_text

    assistant = {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": "call_0",
            "type": "function",
            "function": {"name": str(action["tool"]), "arguments": copy.deepcopy(dict(action.get("arguments") or {}))},
        }],
    }
    conversation = [copy.deepcopy(dict(message)) for message in messages] + [assistant]
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": False, "enable_thinking": enable_thinking}
    if tools is not None:
        kwargs["tools"] = [copy.deepcopy(dict(tool)) for tool in tools]
    try:
        full = processor.apply_chat_template(conversation, **kwargs)
    except Exception as exc:
        try:
            full = render_text_with_stringified_content_fallback(processor, conversation, kwargs, exc)
        except Exception as fallback_exc:
            raise TargetError(f"the chat template refused the labelled turn: {type(fallback_exc).__name__}: {fallback_exc}") from exc
    if not isinstance(full, str) or not full:
        raise TargetError("the chat template rendered no text for the labelled turn")
    full, injected = normalize_thought_prefix(full, rendered_prompt)
    if not full.startswith(rendered_prompt):
        raise TargetError("the generation prompt is not an exact prefix of the labelled turn's render")
    prompt_ids = _token_ids(tokenize_rendered_text(processor, rendered_prompt))
    full_ids = _token_ids(tokenize_rendered_text(processor, full))
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise TargetError("the tokenized generation prompt is not an exact prefix of the labelled turn's tokens")
    completion_ids = full_ids[len(prompt_ids):]
    if not completion_ids:
        raise TargetError("the labelled turn adds no tokens to the prompt")
    return TargetCompletion(text=full[len(rendered_prompt):], prompt_ids=prompt_ids, completion_ids=completion_ids,
                            empty_thought_injected=injected)


class TargetRenderer:
    """``render_target(model_observation, sampled_text, decision) -> completion ids`` for :func:`psse_env.rl.rows.build_rows`.

    ``policy`` renders prompts (``render_messages``, ``render_prompt``) as the
    sampling policy did; ``processor`` is its tokenizer.  Every rendered
    completion is parsed back with the policy's own parser and schema check
    (``_validated_generated_action``) and must equal the labelled action.
    """

    def __init__(self, policy: Any, processor: Any) -> None:
        self.policy = policy
        self.processor = processor
        self.rendered = 0
        self.refused = 0

    def __call__(self, observation: Mapping[str, Any], text: str, decision: Mapping[str, Any]) -> list[int]:
        from psse_env.dagger.release_factories import GateError, _validated_generated_action

        try:
            action = labelled_action(text, decision)
            messages, tools, parameters = self.policy.render_messages(observation)
            rendered_prompt, _visible, _parameters = self.policy.render_prompt(observation)
            target = render_target_completion(self.processor, messages=messages, tools=tools, rendered_prompt=rendered_prompt,
                                              action=action)
            try:
                validated = _validated_generated_action(target.text, parameters)
            except GateError as exc:
                raise TargetError(f"the rendered labelled turn does not parse back: {exc}") from exc
            if validated != action:
                raise TargetError(f"the rendered labelled turn parses to {validated}, not {action}")
        except TargetError:
            self.refused += 1
            raise
        self.rendered += 1
        return target.completion_ids


__all__ = [
    "TARGETS_CONTRACT",
    "TargetCompletion",
    "TargetError",
    "TargetRenderer",
    "labelled_action",
    "normalize_thought_prefix",
    "render_target_completion",
]
