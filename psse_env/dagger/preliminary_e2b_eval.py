"""Diagnostic-only Gemma 4 E2B policy loader for small closed-loop replays.

This module intentionally is not part of the release policy.  It lets the
production closed-loop evaluator exercise the small local adapters created by
the preliminary pipeline while preserving the canonical SFT observation and
tool-controller bridge.
"""

from __future__ import annotations

import copy
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from eval_sft_agent_gemma_v4 import (
    decode_generated_response,
    get_stop_token_ids,
    render_eval_text,
    resolve_pad_token_id,
    tokenize_rendered_text,
)
from psse_env.dagger.dataset_builder import (
    CANONICAL_DAGGER_SYSTEM_PROMPT,
    validate_policy_payload,
)
from psse_env.dagger.policy_adapter import LocalAliasPolicyAdapter
from psse_env.dagger.protocol_bridge import unified_tool_schemas
from psse_env.dagger.release_factories import (
    _model_input_device,
    _validated_generated_action,
    checkpoint_tree_sha256,
)
from psse_env.sft.gates import GateError
from psse_env.sft.training import infer_required_side_input_names


BASE_MODEL_REVISION = "f0c5915f17ad6c66dbeb577fb06ff8925bf8d7ae"
BASE_MODEL_ID = "unsloth/gemma-4-E2B-it"
MAX_INPUT_TOKENS = 8192
MAX_NEW_TOKENS = 64


@dataclass(frozen=True)
class _E2BBundle:
    model: Any
    processor: Any
    model_id: str
    model_revision: str
    base_model_path: str


_BUNDLES: dict[tuple[str, str], _E2BBundle] = {}
_BUNDLE_LOCK = threading.Lock()


def _validate_adapter_identity(model_id: str, model_revision: str) -> tuple[Path, Path]:
    if re.fullmatch(r"[0-9a-f]{64}", model_revision) is None:
        raise GateError("E2B adapter revision must be a lowercase 64-hex tree digest")
    adapter = Path(model_id).expanduser()
    if not adapter.is_absolute():
        raise GateError("E2B adapter model_id must be an absolute path")
    adapter = adapter.resolve(strict=True)
    if checkpoint_tree_sha256(adapter) != model_revision:
        raise GateError("E2B adapter tree digest does not match model_revision")

    config_path = adapter / "adapter_config.json"
    weights_path = adapter / "adapter_model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise GateError("E2B adapter must contain adapter_config.json and safetensors weights")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise GateError(f"E2B adapter configuration is invalid: {exc}") from exc
    if str(config.get("peft_type") or "").upper() != "LORA":
        raise GateError("E2B diagnostic loader accepts only LoRA adapters")

    base_value = str(config.get("base_model_name_or_path") or "").strip()
    if not base_value:
        raise GateError("E2B adapter has no base_model_name_or_path")
    if base_value == BASE_MODEL_ID:
        hf_home = Path(
            os.environ.get("HF_HOME")
            or Path.home() / ".cache" / "huggingface"
        ).expanduser()
        base = (
            hf_home
            / "hub"
            / "models--unsloth--gemma-4-E2B-it"
            / "snapshots"
            / BASE_MODEL_REVISION
        ).resolve(strict=True)
    else:
        base = Path(base_value).expanduser().resolve(strict=True)
    if base.name.lower() != BASE_MODEL_REVISION:
        raise GateError(
            "E2B adapter does not bind the pinned preliminary base revision "
            f"{BASE_MODEL_REVISION}"
        )
    try:
        base_config = json.loads((base / "config.json").read_text(encoding="utf-8"))
    except Exception as exc:
        raise GateError(f"Pinned E2B base configuration is invalid: {exc}") from exc
    if base_config.get("model_type") != "gemma4":
        raise GateError("Pinned preliminary base is not a Gemma 4 model")
    return adapter, base


def _load_bundle(model_id: str, model_revision: str) -> _E2BBundle:
    adapter, base = _validate_adapter_identity(model_id, model_revision)
    try:
        from unsloth import FastModel
    except Exception as exc:  # pragma: no cover - depends on the live GPU environment.
        raise GateError(f"Unsloth is required for preliminary E2B evaluation: {exc}") from exc

    try:
        model, processor = FastModel.from_pretrained(
            model_name=str(adapter),
            tokenizer_name=str(adapter),
            max_seq_length=MAX_INPUT_TOKENS,
            load_in_4bit=True,
            load_in_16bit=False,
            full_finetuning=False,
        )
        FastModel.for_inference(model)
    except Exception as exc:  # pragma: no cover - depends on the live GPU environment.
        raise GateError(
            "Pinned local E2B adapter load failed; no raw-base fallback was used: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if checkpoint_tree_sha256(adapter) != model_revision:
        raise GateError("E2B adapter changed while it was being loaded")
    model.eval()
    return _E2BBundle(
        model=model,
        processor=processor,
        model_id=str(adapter),
        model_revision=model_revision,
        base_model_path=str(base),
    )


def _cached_bundle(model_id: str, model_revision: str) -> _E2BBundle:
    key = (str(Path(model_id).expanduser().resolve(strict=True)), model_revision)
    with _BUNDLE_LOCK:
        bundle = _BUNDLES.get(key)
        if bundle is None:
            bundle = _load_bundle(*key)
            _BUNDLES[key] = bundle
        return bundle


class _CanonicalE2BPolicy:
    """Greedy canonical-tool generation with exact preliminary prompt parity."""

    def __init__(self, bundle: _E2BBundle) -> None:
        self._bundle = bundle
        self._tools = unified_tool_schemas()
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
        messages = [
            {"role": "system", "content": CANONICAL_DAGGER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(payload, sort_keys=True, allow_nan=False),
            },
        ]
        rendered = render_eval_text(
            self._bundle.processor,
            messages,
            self._tools,
            enable_thinking=False,
            inject_empty_thought_channel=True,
        )
        encoded = tokenize_rendered_text(self._bundle.processor, rendered)
        input_ids = encoded.get("input_ids")
        if input_ids is None or not hasattr(input_ids, "shape"):
            raise GateError("E2B processor did not return tensor input_ids")
        prompt_length = int(input_ids.shape[-1])
        if prompt_length <= 0 or prompt_length > MAX_INPUT_TOKENS:
            raise GateError(
                f"E2B prompt length {prompt_length} is outside the trained 1..{MAX_INPUT_TOKENS} range"
            )

        try:
            import torch
        except Exception as exc:  # pragma: no cover - live optional dependency.
            raise GateError(f"torch is required for E2B evaluation: {exc}") from exc
        device = _model_input_device(self._bundle.model)
        model_inputs: dict[str, Any] = {}
        for key, value in encoded.items():
            if not hasattr(value, "shape") or int(value.shape[-1]) != prompt_length:
                continue
            model_inputs[str(key)] = value.to(device)
        required = infer_required_side_input_names(
            self._bundle.model,
            self._bundle.processor,
            self._bundle.base_model_path,
        )
        for name in required:
            model_inputs.setdefault(name, torch.zeros_like(model_inputs["input_ids"]))

        stop_ids = get_stop_token_ids(self._bundle.processor)
        pad_token_id = resolve_pad_token_id(self._bundle.processor)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = self._bundle.model.generate(
                **model_inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=0.0,
                use_cache=True,
                eos_token_id=stop_ids,
                pad_token_id=pad_token_id,
            )
        generation_seconds = time.perf_counter() - started
        output_ids = generated[0][prompt_length:].detach().cpu()
        text, generated_tokens, trimmed_pad_tokens = decode_generated_response(
            self._bundle.processor,
            output_ids,
            pad_token_id=pad_token_id,
        )
        self._last_action_metrics = {
            "prompt_tokens": prompt_length,
            "generated_tokens": int(generated_tokens),
            "generation_seconds": float(generation_seconds),
            "hit_max_new_tokens": int(generated_tokens) >= MAX_NEW_TOKENS,
            "trimmed_trailing_pad_tokens": int(trimmed_pad_tokens),
        }
        return text

    def act(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        return _validated_generated_action(
            self.generate_text(observation), self._parameter_schemas
        )


class PreliminaryE2BPolicy:
    """SFT-visible policy wrapper with evaluator identity attestation."""

    def __init__(self, bundle: _E2BBundle) -> None:
        self._canonical = _CanonicalE2BPolicy(bundle)
        self._adapter = LocalAliasPolicyAdapter(self._canonical, protocol="canonical")
        self._model_id = bundle.model_id
        self._model_revision = bundle.model_revision

    @property
    def release_policy_identity(self) -> dict[str, str | None]:
        return {
            "explicit_policy_identity": None,
            "model_id": self._model_id,
            "model_revision": self._model_revision,
        }

    @property
    def last_action_metrics(self) -> dict[str, Any]:
        return self._canonical.last_action_metrics

    def act(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        validate_policy_payload(observation)
        return self._adapter.act(copy.deepcopy(dict(observation)))


def preliminary_e2b_policy_factory(
    *,
    model_id: str | None = None,
    model_revision: str | None = None,
    seed: int | None = None,
    rng: Any | None = None,
) -> PreliminaryE2BPolicy:
    """Build the content-addressed local E2B policy for diagnostic replay."""

    del seed, rng
    normalized_id = str(model_id or "").strip()
    normalized_revision = str(model_revision or "").strip().lower()
    if not normalized_id or not normalized_revision:
        raise ValueError("Preliminary E2B policy requires model_id and model_revision")
    return PreliminaryE2BPolicy(_cached_bundle(normalized_id, normalized_revision))


__all__ = [
    "BASE_MODEL_REVISION",
    "MAX_INPUT_TOKENS",
    "MAX_NEW_TOKENS",
    "PreliminaryE2BPolicy",
    "preliminary_e2b_policy_factory",
]
