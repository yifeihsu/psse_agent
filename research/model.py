"""Load a base model, optionally with a LoRA adapter, without release gates.

``psse_env.dagger.release_factories`` verifies snapshot file manifests, pins
the base model by commit, and refuses substitutes.  For a research prototype
we want to point at whatever local model fits on the available GPU, so this
module loads the components directly.

The returned policy is the *same* ``GemmaReleasePolicy`` the release path
builds, so prompt rendering, canonical action validation and controller alias
binding are identical.  Only the gating around loading differs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from psse_env.dagger.release_factories import GemmaReleasePolicy, _ModelBundle

DEFAULT_MODEL_ID = "unsloth/gemma-4-E2B-it"
DEFAULT_MODEL_REVISION = "f0c5915f17ad6c66dbeb577fb06ff8925bf8d7ae"

# The language tower uses plain ``nn.Linear``; the vision and audio towers use
# clipped variants that PEFT cannot adapt.  Targeting by bare suffix would
# match those too, so callers resolve fully qualified names instead.
LORA_TARGET_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


def resolve_model_identity(
    model_id: str | None = None, revision: str | None = None
) -> tuple[str, str]:
    """Resolve the model to load from arguments, environment, then default."""

    resolved_id = (
        model_id
        or os.environ.get("PSSE_RESEARCH_MODEL_ID")
        or DEFAULT_MODEL_ID
    )
    resolved_revision = (
        revision
        or os.environ.get("PSSE_RESEARCH_MODEL_REVISION")
        or DEFAULT_MODEL_REVISION
    )
    return str(resolved_id), str(resolved_revision)


def resolve_snapshot(model_id: str, revision: str) -> Path:
    """Return the local Hub snapshot directory for an exact revision."""

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_files_only=True,
        )
    )


def load_processor(
    *,
    model_id: str | None = None,
    revision: str | None = None,
) -> tuple[Any, str, str]:
    """Load only the pinned local processor for CPU-side data accounting."""

    from transformers import AutoProcessor

    resolved_id, resolved_revision = resolve_model_identity(model_id, revision)
    snapshot = resolve_snapshot(resolved_id, resolved_revision)
    processor = AutoProcessor.from_pretrained(str(snapshot), local_files_only=True)
    return processor, resolved_id, resolved_revision


def load_model_and_processor(
    *,
    model_id: str | None = None,
    revision: str | None = None,
    adapter_path: str | Path | None = None,
    load_in_4bit: bool = True,
    for_training: bool = False,
) -> tuple[Any, Any, str, str]:
    """Load the model and processor, applying a LoRA adapter when given."""

    import torch
    from transformers import AutoModelForImageTextToText, BitsAndBytesConfig

    processor, resolved_id, resolved_revision = load_processor(
        model_id=model_id,
        revision=revision,
    )
    snapshot = resolve_snapshot(resolved_id, resolved_revision)

    kwargs: dict[str, Any] = {
        "local_files_only": True,
        "device_map": "auto",
        "dtype": torch.bfloat16,
    }
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    try:
        model = AutoModelForImageTextToText.from_pretrained(str(snapshot), **kwargs)
    except (ValueError, KeyError):
        # Text-only checkpoints in the same family register under the causal-LM
        # mapping instead of the multimodal one.
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(str(snapshot), **kwargs)

    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model, str(adapter_path), is_trainable=for_training
        )
    if not for_training:
        model.eval()
    return model, processor, resolved_id, resolved_revision


def lora_target_modules(model: Any) -> tuple[str, ...]:
    """Return fully qualified adaptable projection names.

    Delegates to the resolver the training path already uses: it scopes
    strictly to the ``language_model`` subtree and unwraps the clipped linear
    variants, which name-based exclusion of the vision and audio towers does
    not reliably reproduce.
    """

    from psse_env.sft.training import resolve_language_lora_targets

    return resolve_language_lora_targets(model, LORA_TARGET_SUFFIXES)


def load_policy(
    *,
    model_id: str | None = None,
    revision: str | None = None,
    adapter_path: str | Path | None = None,
    load_in_4bit: bool = True,
    guarded: bool = False,
) -> Any:
    """Build a canonical-protocol policy around a locally loaded model.

    ``guarded=True`` wraps the policy with the silence-retry and
    stale-reference-retry inference guards; interventions are recorded on the
    returned policy and drained via ``pop_events``.
    """

    model, processor, resolved_id, resolved_revision = load_model_and_processor(
        model_id=model_id,
        revision=revision,
        adapter_path=adapter_path,
        load_in_4bit=load_in_4bit,
    )
    bundle = _ModelBundle(
        model=model,
        processor=processor,
        model_id=resolved_id,
        model_revision=resolved_revision,
    )
    policy = GemmaReleasePolicy(bundle)
    if guarded:
        from .guards import GuardedPolicy

        return GuardedPolicy(policy, bundle)
    return policy
