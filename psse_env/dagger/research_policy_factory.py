"""Research-only Gemma policy loading without release tree attestation.

This module deliberately reuses the canonical generation and controller-alias
path from :mod:`psse_env.dagger.release_factories`, but it does not claim a
release identity and it never computes or checks an adapter tree digest.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import threading
import time
from typing import Any, Mapping, Sequence

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
from psse_env.research_models import (
    GEMMA4_E2B_LEGACY,
    PROMPT_PROFILE_E2B_ALIAS,
    PROMPT_PROFILE_NATIVE,
    PROMPT_PROFILE_RELEASE,
    PROMPT_PROFILE_SMALL_FORCED,
    SUPPORTED_RESEARCH_ARCHITECTURES,
    SUPPORTED_RESEARCH_PROMPT_PROFILES,
    assert_adapter_model_compatible,
    known_research_model,
)
from psse_env.dagger.release_factories import (
    _CanonicalGemmaPolicy,
    _ModelBundle,
    _model_input_device,
    _validated_generated_action,
)
from psse_env.sft.gates import GateError
from psse_env.sft.training import infer_required_side_input_names


# Keep the old spelling as a CLI/API compatibility alias.  Internally E2B and
# E4B share the same measured small-model forced-prefix contract.
PROMPT_PROFILE_E2B = PROMPT_PROFILE_E2B_ALIAS
E2B_BASE_MODEL_ID = GEMMA4_E2B_LEGACY.model_id
E2B_BASE_MODEL_REVISION = GEMMA4_E2B_LEGACY.revision
_PROMPT_PROFILES = SUPPORTED_RESEARCH_PROMPT_PROFILES
_RESEARCH_BUNDLE_CACHE: dict[tuple[Any, ...], tuple[_ModelBundle, dict[str, Any]]] = {}
_RESEARCH_BUNDLE_LOCK = threading.Lock()


def _token_repetition_metrics(token_ids: Sequence[int]) -> dict[str, Any]:
    """Detect a sustained repeated token block, not ordinary JSON reuse.

    A block is considered runaway only when it repeats consecutively at least
    four times and covers at least 24 generated tokens.  The minimum span keeps
    repeated punctuation and schema keys from being mistaken for a loop while
    still catching both single-token and phrase-level degeneration.
    """

    tokens = tuple(int(token) for token in token_ids)
    best: tuple[int, int, int] | None = None
    for start in range(len(tokens)):
        maximum_width = min(32, (len(tokens) - start) // 4)
        for width in range(1, maximum_width + 1):
            block = tokens[start : start + width]
            repeats = 1
            cursor = start + width
            while tokens[cursor : cursor + width] == block:
                repeats += 1
                cursor += width
            span = width * repeats
            if repeats >= 4 and span >= 24:
                candidate = (span, width, repeats)
                if best is None or candidate > best:
                    best = candidate
    return {
        "repetition_loop_detected": best is not None,
        "repetition_span_tokens": best[0] if best else 0,
        "repetition_ngram_width": best[1] if best else 0,
        "repetition_consecutive_repeats": best[2] if best else 0,
    }


def _validate_adapter_directory(adapter_path: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(adapter_path).expanduser().resolve(strict=True)
    if not path.is_dir():
        raise GateError(f"Research adapter path is not a directory: {path}")
    config_path = path / "adapter_config.json"
    weights = (
        path / "adapter_model.safetensors",
        path / "adapter_model.bin",
    )
    if not config_path.is_file() or not any(item.is_file() for item in weights):
        raise GateError(
            "Research adapter must contain adapter_config.json and PEFT adapter weights"
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise GateError(f"Research adapter_config.json is invalid: {exc}") from exc
    if str(config.get("peft_type") or "").upper() != "LORA":
        raise GateError("Research policy factory accepts only a LoRA PEFT adapter")
    return path, config


def _known_base_details() -> tuple[str, str]:
    return E2B_BASE_MODEL_ID, E2B_BASE_MODEL_REVISION


def _prompt_profile_hints(*sources: str) -> set[str]:
    matches: set[str] = set()
    for source in sources:
        spec = known_research_model(source)
        if spec is not None:
            matches.add(spec.prompt_profile)
    return matches


def _infer_prompt_profile(*sources: str) -> str | None:
    """Infer from registered model families whose prompt contracts we own."""

    matches = _prompt_profile_hints(*sources)
    if len(matches) == 1:
        return next(iter(matches))
    return None


def _resolve_prompt_profile(
    requested: str | None, *, base_model: str, recorded_base: str, adapter: Path
) -> str:
    del adapter
    hints = _prompt_profile_hints(base_model, recorded_base)
    if len(hints) > 1:
        raise ValueError(
            "The requested and adapter-recorded base models identify conflicting "
            f"prompt families: {sorted(hints)}"
        )
    inferred = next(iter(hints)) if hints else None
    if requested is not None:
        profile = str(requested).strip().lower()
        if profile not in _PROMPT_PROFILES:
            raise ValueError(
                f"prompt_profile must be one of {sorted(_PROMPT_PROFILES)}, got {requested!r}"
            )
        if profile == PROMPT_PROFILE_E2B_ALIAS:
            profile = PROMPT_PROFILE_SMALL_FORCED
        if inferred is not None and profile != inferred:
            raise ValueError(
                f"prompt_profile={profile!r} conflicts with the known {inferred!r} model family"
            )
        return profile
    if inferred is None:
        raise ValueError(
            "prompt_profile is required when the adapter/base path does not identify "
            "a registered Gemma research prompt contract"
        )
    return inferred


def _local_model_type(source: str) -> str | None:
    candidate = Path(source).expanduser()
    if not candidate.is_dir():
        return None
    config_path = candidate / "config.json"
    if not config_path.is_file():
        return None
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Local research base config.json is invalid: {exc}") from exc
    model_type = str(payload.get("model_type") or "").strip()
    return model_type or None


def _resolve_architecture(requested: str | None, *, base_model: str) -> str:
    known = known_research_model(base_model)
    inferred = known.architecture if known is not None else _local_model_type(base_model)
    architecture = str(requested or inferred or "").strip()
    if not architecture:
        raise ValueError(
            "architecture is required when an opaque research base path is used"
        )
    if architecture not in SUPPORTED_RESEARCH_ARCHITECTURES:
        raise ValueError(
            "research architecture must be one of "
            f"{sorted(SUPPORTED_RESEARCH_ARCHITECTURES)}, got {architecture!r}"
        )
    if inferred is not None and architecture != inferred:
        raise ValueError(
            f"architecture={architecture!r} conflicts with the base model's "
            f"{inferred!r} architecture"
        )
    return architecture


def _adapter_has_processor_assets(adapter: Path) -> bool:
    # A chat template alone is insufficient for AutoProcessor.  A processor
    # config is conclusive; otherwise require tokenizer configuration plus at
    # least one tokenizer/template payload.
    if any((adapter / name).is_file() for name in ("processor_config.json", "preprocessor_config.json")):
        return True
    if not (adapter / "tokenizer_config.json").is_file():
        return False
    return any(
        (adapter / name).is_file()
        for name in ("tokenizer.json", "tokenizer.model", "chat_template.jinja")
    )


def _resolve_base(
    *, recorded_base: str, base_model: str | None, base_revision: str | None
) -> tuple[str, str | None, str | None]:
    """Return loader source, loader revision, and descriptive revision.

    Existing local snapshots are already concrete byte sources, so passing a
    Hub ``revision`` to Transformers would be both invalid and unnecessary.
    """

    requested = str(base_model or "").strip()
    source = requested or recorded_base
    if not source:
        raise ValueError(
            "base_model is required when adapter_config.json records no base model"
        )
    candidate = Path(source).expanduser()
    if candidate.is_dir():
        resolved = str(candidate.resolve(strict=True))
        descriptive_revision = str(base_revision or "").strip() or None
        return resolved, None, descriptive_revision

    revision = str(base_revision or "").strip()
    if not revision:
        known = known_research_model(source)
        if known is not None:
            revision = known.revision
    if not revision:
        raise ValueError(
            "base_revision is required for a non-local base model with no known pinned revision"
        )
    return source, revision, revision


def clear_research_policy_cache() -> None:
    """Release cached research model bundles between different large adapters."""

    with _RESEARCH_BUNDLE_LOCK:
        _RESEARCH_BUNDLE_CACHE.clear()


def _load_research_bundle(
    *,
    adapter_path: str | Path,
    base_model: str | None,
    base_revision: str | None,
    load_in_4bit: bool,
    local_files_only: bool,
    trust_remote_code: bool,
    prompt_profile: str | None,
    use_cache: bool,
    architecture: str | None = None,
) -> tuple[_ModelBundle, dict[str, Any]]:
    adapter, adapter_config = _validate_adapter_directory(adapter_path)
    recorded_base = str(adapter_config.get("base_model_name_or_path") or "").strip()
    resolved_base_model, loader_revision, descriptive_revision = _resolve_base(
        recorded_base=recorded_base,
        base_model=base_model,
        base_revision=base_revision,
    )
    try:
        assert_adapter_model_compatible(resolved_base_model, recorded_base)
    except ValueError as exc:
        raise GateError(str(exc)) from exc
    resolved_architecture = _resolve_architecture(
        architecture, base_model=resolved_base_model
    )
    resolved_profile = _resolve_prompt_profile(
        prompt_profile,
        base_model=resolved_base_model,
        recorded_base=recorded_base,
        adapter=adapter,
    )
    processor_source = str(adapter) if _adapter_has_processor_assets(adapter) else resolved_base_model
    processor_revision = None if processor_source == str(adapter) else loader_revision
    adapter_stamp = tuple(
        sorted(
            (item.name, item.stat().st_size, item.stat().st_mtime_ns)
            for item in adapter.iterdir()
            if item.is_file()
        )
    )
    cache_key = (
        str(adapter),
        adapter_stamp,
        resolved_base_model,
        loader_revision or "",
        processor_source,
        resolved_profile,
        resolved_architecture,
        bool(load_in_4bit),
        bool(local_files_only),
        bool(trust_remote_code),
    )
    if use_cache:
        with _RESEARCH_BUNDLE_LOCK:
            cached = _RESEARCH_BUNDLE_CACHE.get(cache_key)
            if cached is not None:
                bundle, identity = cached
                return bundle, copy.deepcopy(identity)
            # Keep at most one very large Gemma bundle resident through this
            # cache. Callers evaluating two adapters should also drop their
            # prior policy wrapper before requesting the next one.
            _RESEARCH_BUNDLE_CACHE.clear()
            bundle, identity = _load_research_bundle_uncached(
                adapter=adapter,
                adapter_config=adapter_config,
                recorded_base=recorded_base,
                resolved_base_model=resolved_base_model,
                loader_revision=loader_revision,
                descriptive_revision=descriptive_revision,
                processor_source=processor_source,
                processor_revision=processor_revision,
                prompt_profile=resolved_profile,
                architecture=resolved_architecture,
                load_in_4bit=load_in_4bit,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
            )
            _RESEARCH_BUNDLE_CACHE[cache_key] = (bundle, copy.deepcopy(identity))
            return bundle, identity
    return _load_research_bundle_uncached(
        adapter=adapter,
        adapter_config=adapter_config,
        recorded_base=recorded_base,
        resolved_base_model=resolved_base_model,
        loader_revision=loader_revision,
        descriptive_revision=descriptive_revision,
        processor_source=processor_source,
        processor_revision=processor_revision,
        prompt_profile=resolved_profile,
        architecture=resolved_architecture,
        load_in_4bit=load_in_4bit,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )


def _load_research_bundle_uncached(
    *,
    adapter: Path,
    adapter_config: Mapping[str, Any],
    recorded_base: str,
    resolved_base_model: str,
    loader_revision: str | None,
    descriptive_revision: str | None,
    processor_source: str,
    processor_revision: str | None,
    prompt_profile: str,
    architecture: str,
    load_in_4bit: bool,
    local_files_only: bool,
    trust_remote_code: bool,
) -> tuple[_ModelBundle, dict[str, Any]]:
    del adapter_config

    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForMultimodalLM, AutoProcessor
    except Exception as exc:  # pragma: no cover - optional live dependencies.
        raise GateError(
            "Research small-model policy dependencies are unavailable. Install "
            "the research dependency overlay (Transformers >=5.10): "
            f"{exc}"
        ) from exc

    model_kwargs: dict[str, Any] = {
        "local_files_only": bool(local_files_only),
        "trust_remote_code": bool(trust_remote_code),
        "device_map": "auto",
        "dtype": torch.bfloat16,
    }
    if loader_revision is not None:
        model_kwargs["revision"] = loader_revision
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except Exception as exc:  # pragma: no cover - optional live dependency.
            raise GateError(f"4-bit research policy loading is unavailable: {exc}") from exc
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    processor_kwargs: dict[str, Any] = {
        "local_files_only": bool(local_files_only),
        "trust_remote_code": bool(trust_remote_code),
    }
    if processor_revision is not None:
        processor_kwargs["revision"] = processor_revision
    try:
        processor = AutoProcessor.from_pretrained(processor_source, **processor_kwargs)
    except Exception as exc:  # pragma: no cover - live model/cache state.
        raise GateError(
            "Research Gemma processor failed to load: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not callable(getattr(processor, "apply_chat_template", None)):
        raise GateError("Research Gemma processor exposes no chat template")
    decoder = (
        processor
        if callable(getattr(processor, "decode", None))
        else getattr(processor, "tokenizer", None)
    )
    if decoder is None or not callable(getattr(decoder, "decode", None)):
        raise GateError("Research Gemma processor exposes no decoder")

    try:
        base = AutoModelForMultimodalLM.from_pretrained(
            resolved_base_model, **model_kwargs
        )
    except Exception as exc:  # pragma: no cover - live model/cache state.
        raise GateError(
            "Research Gemma base model failed to load: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    observed_architecture = getattr(getattr(base, "config", None), "model_type", None)
    if observed_architecture != architecture:
        raise GateError(
            "Research base-model architecture mismatch: expected "
            f"{architecture!r}, got {observed_architecture!r}"
        )
    missing = [
        name
        for name in ("generate", "get_input_embeddings")
        if not callable(getattr(base, name, None))
    ]
    if missing:
        raise GateError(
            "Research base model lacks required capabilities: " + ", ".join(missing)
        )

    try:
        model = PeftModel.from_pretrained(
            base,
            str(adapter),
            is_trainable=False,
            local_files_only=bool(local_files_only),
        )
    except Exception as exc:  # pragma: no cover - live adapter state.
        raise GateError(
            "Research LoRA adapter failed to load on the selected base: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    peft_config = getattr(model, "peft_config", None)
    if not isinstance(peft_config, Mapping) or not peft_config:
        raise GateError("PEFT returned a model with no active adapter configuration")
    model.eval()

    bundle = _ModelBundle(
        model=model,
        processor=processor,
        model_id=resolved_base_model,
        model_revision=descriptive_revision or "local-snapshot",
        adapter_snapshot_path=str(adapter),
    )
    identity = {
        "base_model": resolved_base_model,
        "base_revision": descriptive_revision,
        "base_loader_revision": loader_revision,
        "adapter_path": str(adapter),
        "adapter_recorded_base": recorded_base or None,
        "processor_source": processor_source,
        "prompt_profile": prompt_profile,
        "architecture": architecture,
        "tree_digest_enforced": False,
    }
    return bundle, identity


class _CanonicalResearchNativePolicy:
    """Greedy native-tool generation for the Unified 12B research model."""

    _UNIFIED_GENERATION_SUFFIX = (
        "<|turn>model\n<|channel>thought\n<channel|>"
    )

    def __init__(self, bundle: _ModelBundle, *, architecture: str) -> None:
        from psse_env.dagger.preliminary_e2b_eval import (
            canonical_prompt_tool_schemas,
        )

        self._bundle = bundle
        self._architecture = architecture
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
        from psse_env.dagger.preliminary_e2b_eval import (
            MAX_INPUT_TOKENS,
            MAX_NEW_TOKENS,
            normalize_episode_state_reference,
        )

        self._last_action_metrics = {}
        if not isinstance(observation, Mapping):
            raise TypeError("Research Gemma policy requires a model-observation mapping")
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
            # The pinned Unified processor inserts its own empty thought
            # channel.  A manual injection would risk duplicating the prefix.
            inject_empty_thought_channel=False,
        )
        if (
            self._architecture == "gemma4_unified"
            and not rendered.endswith(self._UNIFIED_GENERATION_SUFFIX)
        ):
            raise GateError(
                "Unified research processor produced an unexpected no-thinking "
                "generation prefix"
            )
        encoded = tokenize_rendered_text(self._bundle.processor, rendered)
        input_ids = encoded.get("input_ids")
        if input_ids is None or not hasattr(input_ids, "shape"):
            raise GateError("Research processor did not return tensor input_ids")
        original_prompt_length = int(input_ids.shape[-1])
        if original_prompt_length <= 0:
            raise GateError("Research processor returned an empty prompt")
        prompt_length = min(original_prompt_length, MAX_INPUT_TOKENS)
        truncated_input_tokens = original_prompt_length - prompt_length

        try:
            import torch
        except Exception as exc:  # pragma: no cover - optional live dependency.
            raise GateError(f"torch is required for research evaluation: {exc}") from exc
        device = _model_input_device(self._bundle.model)
        model_inputs: dict[str, Any] = {}
        for key, value in encoded.items():
            if (
                not hasattr(value, "shape")
                or int(value.shape[-1]) != original_prompt_length
            ):
                continue
            model_inputs[str(key)] = value[..., -prompt_length:].to(device)
        required = infer_required_side_input_names(
            self._bundle.model,
            self._bundle.processor,
            self._bundle.model_id,
        )
        for name in required:
            model_inputs.setdefault(name, torch.zeros_like(model_inputs["input_ids"]))

        stop_ids = get_stop_token_ids(self._bundle.processor)
        pad_token_id = resolve_pad_token_id(self._bundle.processor)
        started = time.perf_counter()
        self._bundle.model.eval()
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
        sampled_ids = generated[0][prompt_length:].detach().cpu()
        text, generated_tokens, trimmed_pad_tokens = decode_generated_response(
            self._bundle.processor,
            sampled_ids,
            pad_token_id=pad_token_id,
        )
        repetition = _token_repetition_metrics(
            sampled_ids[: int(generated_tokens)].tolist()
        )
        self._last_action_metrics = {
            "prompt_tokens": prompt_length,
            "original_prompt_tokens": original_prompt_length,
            "truncated_input_tokens": truncated_input_tokens,
            "generated_tokens": int(generated_tokens),
            "generation_seconds": float(generation_seconds),
            "hit_max_new_tokens": int(generated_tokens) >= MAX_NEW_TOKENS,
            "forced_tool_prefix": None,
            "trimmed_trailing_pad_tokens": int(trimmed_pad_tokens),
            **repetition,
        }
        try:
            action = _validated_generated_action(text, self._parameter_schemas)
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
            self.generate_text(observation), self._parameter_schemas
        )


def _canonical_research_policy(
    bundle: _ModelBundle, identity: Mapping[str, Any]
) -> Any:
    profile = str(identity.get("prompt_profile") or "")
    if profile == PROMPT_PROFILE_SMALL_FORCED:
        # E2B and E4B share the exact no-thinking template and use the measured
        # forced native-tool prefix that keeps small adapters in tool mode.
        from psse_env.dagger.preliminary_e2b_eval import (
            _CanonicalE2BPolicy,
            _E2BBundle,
        )

        return _CanonicalE2BPolicy(
            _E2BBundle(
                model=bundle.model,
                processor=bundle.processor,
                model_id=str(identity["adapter_path"]),
                model_revision=str(identity.get("base_revision") or "local-snapshot"),
                base_model_path=str(identity["base_model"]),
            )
        )
    if profile == PROMPT_PROFILE_NATIVE:
        return _CanonicalResearchNativePolicy(
            bundle, architecture=str(identity.get("architecture") or "")
        )
    if profile == PROMPT_PROFILE_RELEASE:
        return _CanonicalGemmaPolicy(bundle)
    raise GateError(f"Unknown research prompt profile: {profile!r}")


class ResearchGemmaPolicy:
    """Canonical Gemma tool policy with an explicit non-release identity."""

    def __init__(self, bundle: _ModelBundle, identity: Mapping[str, Any]) -> None:
        self._canonical_policy = _canonical_research_policy(bundle, identity)
        self._adapter = LocalAliasPolicyAdapter(
            self._canonical_policy, protocol="canonical"
        )
        self._identity = copy.deepcopy(dict(identity))

    @property
    def research_policy_identity(self) -> dict[str, Any]:
        return copy.deepcopy(self._identity)

    @property
    def last_action_metrics(self) -> dict[str, Any]:
        return self._canonical_policy.last_action_metrics

    def act_model_observation(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        """Act on an already-exported canonical SFT observation.

        Closed-loop callers must use :meth:`act`, which compacts a raw
        controller observation and binds local aliases on the returned action.
        Processor/generation canaries already start at the model-visible SFT
        boundary, so sending them through that adapter a second time would
        corrupt bounded history and compare different action protocols.
        """

        validate_policy_payload(observation)
        return self._canonical_policy.act(copy.deepcopy(dict(observation)))

    def act(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        validate_policy_payload(observation)
        return self._adapter.act(copy.deepcopy(dict(observation)))


def research_gemma_policy_factory(
    adapter_path: str | Path,
    *,
    base_model: str | None = None,
    base_revision: str | None = None,
    load_in_4bit: bool = True,
    local_files_only: bool = True,
    trust_remote_code: bool = False,
    prompt_profile: str | None = None,
    architecture: str | None = None,
    cache: bool = True,
    seed: int | None = None,
    rng: Any | None = None,
) -> ResearchGemmaPolicy:
    """Load a research LoRA directly, without source/tree/hash qualification."""

    del seed, rng
    bundle, identity = _load_research_bundle(
        adapter_path=adapter_path,
        base_model=base_model,
        base_revision=base_revision,
        load_in_4bit=load_in_4bit,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
        prompt_profile=prompt_profile,
        use_cache=cache,
        architecture=architecture,
    )
    return ResearchGemmaPolicy(bundle, identity)


__all__ = [
    "PROMPT_PROFILE_E2B",
    "PROMPT_PROFILE_NATIVE",
    "PROMPT_PROFILE_RELEASE",
    "PROMPT_PROFILE_SMALL_FORCED",
    "ResearchGemmaPolicy",
    "clear_research_policy_cache",
    "research_gemma_policy_factory",
]
