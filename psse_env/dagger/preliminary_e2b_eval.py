"""Prompt budget, tool schemas and policy wrapper for the small Gemma 4 models.

``research_policy_factory`` renders the E2B/E4B prompts with the canonical tool
schemas, token budgets and forced tool-call prefix defined here and wraps the
loaded model in ``_CanonicalE2BPolicy``.
"""

from __future__ import annotations

import copy
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Mapping

from psse_env.dagger.dataset_builder import (
    CANONICAL_DAGGER_SYSTEM_PROMPT,
    validate_policy_payload,
    tool_schemas_for_observation,
    system_prompt_for_observation,
)
from psse_env.dagger.protocol_bridge import unified_tool_schemas
from psse_env.dagger.release_factories import (
    _model_input_device,
    _validated_generated_action,
)
from psse_env.sft.gates import GateError
from psse_env.sft.gemma_text import (
    decode_generated_response,
    encode_text,
    get_stop_token_ids,
    render_eval_text,
    resolve_pad_token_id,
    sanitize_tool_schemas,
    tokenize_rendered_text,
)
from psse_env.sft.training import infer_required_side_input_names


def _positive_int_env(name: str, default: int) -> int:
    value = int(os.environ.get(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


# Keep the historical defaults for compatibility with the preliminary tool
# gate, while allowing research jobs to relax both limits without changing
# source code or rebuilding any dataset/checkpoint.
MAX_INPUT_TOKENS = _positive_int_env("RESEARCH_MAX_INPUT_TOKENS", 8192)
MAX_NEW_TOKENS = _positive_int_env("RESEARCH_MAX_NEW_TOKENS", 64)
FORCED_TOOL_PREFIX = "<|tool_call>call:"


def canonical_prompt_tool_schemas() -> list[dict[str, Any]]:
    """Return the exact sanitized registry rendered during SFT preprocessing."""

    return sanitize_tool_schemas(unified_tool_schemas())


def normalize_episode_state_reference(
    action: Mapping[str, Any],
    observation: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Rewrite only the visible episode id to the canonical active alias."""

    normalized = copy.deepcopy(dict(action))
    arguments = normalized.get("arguments")
    if not isinstance(arguments, Mapping):
        return normalized, []
    normalized_arguments = copy.deepcopy(dict(arguments))
    normalized["arguments"] = normalized_arguments
    episode_id = observation.get("episode_id")
    active_alias = observation.get("active_state_id")
    if not isinstance(episode_id, str) or not isinstance(active_alias, str):
        return normalized, []
    if not episode_id or not active_alias or episode_id == active_alias:
        return normalized, []
    rewrites: list[dict[str, str]] = []
    for field in ("case_path", "scan_window_path"):
        if normalized_arguments.get(field) != episode_id:
            continue
        normalized_arguments[field] = active_alias
        rewrites.append({"argument": field, "from": episode_id, "to": active_alias})
    return normalized, rewrites


@dataclass(frozen=True)
class _E2BBundle:
    model: Any
    processor: Any
    model_id: str
    model_revision: str
    base_model_path: str


class _CanonicalE2BPolicy:
    """Greedy canonical-tool generation with exact preliminary prompt parity."""

    def __init__(self, bundle: _E2BBundle) -> None:
        self._bundle = bundle
        self._tools = canonical_prompt_tool_schemas()
        self._parameter_schemas = {
            str(row["function"]["name"]): row["function"]["parameters"]
            for row in self._tools
        }
        self._last_action_metrics: dict[str, Any] = {}

    @property
    def last_action_metrics(self) -> dict[str, Any]:
        return copy.deepcopy(self._last_action_metrics)

    def generate_text(self, observation: Mapping[str, Any]) -> str:
        """Generate one raw response with the exact closed-loop prompt contract."""

        self._last_action_metrics = {}
        if not isinstance(observation, Mapping):
            raise TypeError("E2B policy requires a model-observation mapping")
        payload = {"state": copy.deepcopy(dict(observation))}
        validate_policy_payload(payload)
        visible_tools = tool_schemas_for_observation(self._tools, observation)
        messages = [
            {"role": "system", "content": system_prompt_for_observation(CANONICAL_DAGGER_SYSTEM_PROMPT, observation)},
            {
                "role": "user",
                "content": json.dumps(payload, sort_keys=True, allow_nan=False),
            },
        ]
        rendered = render_eval_text(
            self._bundle.processor,
            messages,
            visible_tools,
            enable_thinking=False,
            # The injection exists to mirror SFT formatting, but this pipeline's
            # rows are rendered by apply_chat_template with no thought channel,
            # so injecting one makes the model generate from four tokens it
            # never saw after <|turn>model.  Measured on a locally trained E2B
            # LoRA over 12 held-out D0 rows, greedy, identical adapter:
            #
            #     canonical      12/12 valid format, 11/12 correct tool
            #     +injection      0/12 valid format,  0/12 correct tool
            #
            # With the injection the model emits free-form prose to the token
            # limit -- the exact failure the preliminary study recorded as
            # 0/5 resolved with zero usable tool calls.  The global CLI default
            # stays True: adapters trained through a rendering that does include
            # the channel still need it.
            inject_empty_thought_channel=False,
        )
        encoded = tokenize_rendered_text(self._bundle.processor, rendered)
        input_ids = encoded.get("input_ids")
        if input_ids is None or not hasattr(input_ids, "shape"):
            raise GateError("E2B processor did not return tensor input_ids")
        original_prompt_length = int(input_ids.shape[-1])
        if original_prompt_length <= 0:
            raise GateError("E2B processor returned an empty prompt")
        prompt_length = min(original_prompt_length, MAX_INPUT_TOKENS)
        truncated_input_tokens = original_prompt_length - prompt_length

        try:
            import torch
        except Exception as exc:  # pragma: no cover - live optional dependency.
            raise GateError(f"torch is required for E2B evaluation: {exc}") from exc
        device = _model_input_device(self._bundle.model)
        model_inputs: dict[str, Any] = {}
        for key, value in encoded.items():
            if (
                not hasattr(value, "shape")
                or int(value.shape[-1]) != original_prompt_length
            ):
                continue
            # Research mode keeps the newest observable state/history when a
            # prompt exceeds the configured window instead of rejecting the
            # episode before inference.
            model_inputs[str(key)] = value[..., -prompt_length:].to(device)
        required = infer_required_side_input_names(
            self._bundle.model,
            self._bundle.processor,
            self._bundle.base_model_path,
        )
        for name in required:
            model_inputs.setdefault(name, torch.zeros_like(model_inputs["input_ids"]))

        forced_prefix_ids = encode_text(
            self._bundle.processor,
            FORCED_TOOL_PREFIX,
        )["input_ids"]
        if not forced_prefix_ids or len(forced_prefix_ids) >= MAX_NEW_TOKENS:
            raise GateError("E2B native tool-call prefix tokenization is invalid")
        forced_prefix = torch.tensor(
            [forced_prefix_ids],
            dtype=model_inputs["input_ids"].dtype,
            device=device,
        )
        for name, value in tuple(model_inputs.items()):
            if not hasattr(value, "shape") or int(value.shape[-1]) != prompt_length:
                continue
            if name == "input_ids":
                suffix = forced_prefix
            elif name == "attention_mask":
                suffix = torch.ones_like(forced_prefix, dtype=value.dtype)
            else:
                suffix = torch.zeros_like(forced_prefix, dtype=value.dtype)
            model_inputs[name] = torch.cat((value, suffix), dim=-1)
        conditioned_length = prompt_length + len(forced_prefix_ids)

        stop_ids = get_stop_token_ids(self._bundle.processor)
        pad_token_id = resolve_pad_token_id(self._bundle.processor)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = self._bundle.model.generate(
                **model_inputs,
                max_new_tokens=MAX_NEW_TOKENS - len(forced_prefix_ids),
                do_sample=False,
                temperature=0.0,
                use_cache=True,
                eos_token_id=stop_ids,
                pad_token_id=pad_token_id,
            )
        generation_seconds = time.perf_counter() - started
        sampled_ids = generated[0][conditioned_length:].detach().cpu()
        output_ids = torch.cat((forced_prefix.detach().cpu()[0], sampled_ids))
        text, generated_tokens, trimmed_pad_tokens = decode_generated_response(
            self._bundle.processor,
            output_ids,
            pad_token_id=pad_token_id,
        )
        self._last_action_metrics = {
            "prompt_tokens": prompt_length,
            "original_prompt_tokens": original_prompt_length,
            "truncated_input_tokens": truncated_input_tokens,
            "generated_tokens": int(generated_tokens),
            "generation_seconds": float(generation_seconds),
            "hit_max_new_tokens": int(generated_tokens) >= MAX_NEW_TOKENS,
            "forced_tool_prefix": FORCED_TOOL_PREFIX,
            "forced_tool_prefix_tokens": len(forced_prefix_ids),
            "trimmed_trailing_pad_tokens": int(trimmed_pad_tokens),
        }
        try:
            action = _validated_generated_action(text,
                {row["function"]["name"]: row["function"]["parameters"] for row in visible_tools})
        except GateError:
            return text
        normalized, rewrites = normalize_episode_state_reference(action, observation)
        self._last_action_metrics["state_reference_rewrites"] = rewrites
        return json.dumps(
            {"name": normalized["tool"], "arguments": normalized["arguments"]},
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )

    def act(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        return _validated_generated_action(
            self.generate_text(observation),
            {row["function"]["name"]: row["function"]["parameters"]
             for row in tool_schemas_for_observation(self._tools, observation)},
        )


__all__ = [
    "FORCED_TOOL_PREFIX",
    "MAX_INPUT_TOKENS",
    "MAX_NEW_TOKENS",
    "canonical_prompt_tool_schemas",
    "normalize_episode_state_reference",
]
