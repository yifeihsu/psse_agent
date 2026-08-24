"""Small Gemma 4 model choices for the non-release research pipeline.

The strict release path owns its own frozen 31B identity.  This registry is
deliberately separate: it selects models by the capabilities needed for rapid
DAgger experiments and keeps the model id, revision, architecture, and prompt
contract together so they cannot drift independently on a launcher command.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path


PROMPT_PROFILE_SMALL_FORCED = "small_forced"
PROMPT_PROFILE_NATIVE = "native"
PROMPT_PROFILE_RELEASE = "release"
PROMPT_PROFILE_E2B_ALIAS = "e2b"


@dataclass(frozen=True)
class ResearchModelSpec:
    """Pinned research-model identity and the loader behavior it requires."""

    key: str
    model_id: str
    revision: str
    architecture: str
    prompt_profile: str
    purpose: str
    load_in_4bit: bool = True
    trust_remote_code: bool = False


GEMMA4_E4B = ResearchModelSpec(
    key="e4b",
    model_id="google/gemma-4-E4B-it",
    revision="ee0ef6023621cff504d758262d4e04895a5af4a2",
    architecture="gemma4",
    prompt_profile=PROMPT_PROFILE_SMALL_FORCED,
    purpose="fast live integration smoke",
)

GEMMA4_12B = ResearchModelSpec(
    key="12b",
    model_id="google/gemma-4-12B-it",
    revision="707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
    architecture="gemma4_unified",
    prompt_profile=PROMPT_PROFILE_NATIVE,
    purpose="first meaningful DAgger pilot",
)

# Backward-compatible research identities.  They remain loadable so existing
# artifacts can be inspected, but neither is the selected small-model default.
GEMMA4_E2B_LEGACY = ResearchModelSpec(
    key="e2b-legacy",
    model_id="unsloth/gemma-4-E2B-it",
    revision="f0c5915f17ad6c66dbeb577fb06ff8925bf8d7ae",
    architecture="gemma4",
    prompt_profile=PROMPT_PROFILE_SMALL_FORCED,
    purpose="legacy preliminary adapter compatibility",
)

GEMMA4_31B_LEGACY = ResearchModelSpec(
    key="31b-legacy",
    model_id="unsloth/gemma-4-31B-it",
    revision="8a796db4df380b178065ed910849477ff0e99c87",
    architecture="gemma4",
    prompt_profile=PROMPT_PROFILE_RELEASE,
    purpose="optional final scale confirmation",
)

DEFAULT_RESEARCH_MODEL = GEMMA4_12B
RESEARCH_MODEL_SPECS = {
    spec.key: spec
    for spec in (
        GEMMA4_E4B,
        GEMMA4_12B,
        GEMMA4_E2B_LEGACY,
        GEMMA4_31B_LEGACY,
    )
}
SUPPORTED_RESEARCH_ARCHITECTURES = frozenset({"gemma4", "gemma4_unified"})
SUPPORTED_RESEARCH_PROMPT_PROFILES = frozenset(
    {
        PROMPT_PROFILE_SMALL_FORCED,
        PROMPT_PROFILE_NATIVE,
        PROMPT_PROFILE_RELEASE,
        PROMPT_PROFILE_E2B_ALIAS,
    }
)


def _normalized_source(source: str) -> str:
    return str(source or "").replace("\\", "/").strip().lower()


def known_research_model(source: str | Path | None) -> ResearchModelSpec | None:
    """Recognize official ids and normal Hugging Face cache snapshot paths."""

    normalized = _normalized_source(str(source or ""))
    if not normalized:
        return None
    for spec in RESEARCH_MODEL_SPECS.values():
        model_id = spec.model_id.lower()
        cache_slug = "models--" + model_id.replace("/", "--")
        if normalized == model_id or model_id in normalized or cache_slug in normalized:
            return spec
    return None


def get_research_model_spec(key: str) -> ResearchModelSpec:
    normalized = str(key or "").strip().lower()
    try:
        return RESEARCH_MODEL_SPECS[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unknown research model {key!r}; choose one of {sorted(RESEARCH_MODEL_SPECS)}"
        ) from exc


def resolve_research_model_spec(
    *,
    model: str | Path | None = None,
    revision: str | None = None,
    architecture: str | None = None,
    prompt_profile: str | None = None,
    default: ResearchModelSpec = DEFAULT_RESEARCH_MODEL,
) -> ResearchModelSpec:
    """Resolve one atomic model choice, allowing explicit local snapshots.

    Known model ids receive their pinned revision, architecture, and prompt
    contract automatically.  An opaque/local custom Gemma 4 source must state
    the missing architecture and revision rather than inheriting unrelated
    defaults.
    """

    source = str(model or default.model_id).strip()
    known = known_research_model(source)
    base = known or default
    resolved_revision = str(revision or (known.revision if known else "")).strip()
    resolved_architecture = str(
        architecture or (known.architecture if known else "")
    ).strip()
    requested_profile = str(
        prompt_profile or (known.prompt_profile if known else PROMPT_PROFILE_NATIVE)
    ).strip().lower()
    if requested_profile == PROMPT_PROFILE_E2B_ALIAS:
        requested_profile = PROMPT_PROFILE_SMALL_FORCED
    if not resolved_revision:
        raise ValueError("revision is required for an unregistered research model")
    if not resolved_architecture:
        raise ValueError("architecture is required for an unregistered research model")
    if resolved_architecture not in SUPPORTED_RESEARCH_ARCHITECTURES:
        raise ValueError(
            "research architecture must be one of "
            f"{sorted(SUPPORTED_RESEARCH_ARCHITECTURES)}, got {resolved_architecture!r}"
        )
    if requested_profile not in SUPPORTED_RESEARCH_PROMPT_PROFILES:
        raise ValueError(
            "research prompt profile must be one of "
            f"{sorted(SUPPORTED_RESEARCH_PROMPT_PROFILES)}, got {requested_profile!r}"
        )
    if known is not None:
        if revision is not None and resolved_revision != known.revision:
            raise ValueError(
                f"revision={resolved_revision!r} conflicts with the pinned "
                f"{known.revision!r} revision for {known.model_id}"
            )
        if architecture is not None and resolved_architecture != known.architecture:
            raise ValueError(
                f"architecture={resolved_architecture!r} conflicts with the known "
                f"{known.architecture!r} architecture for {known.model_id}"
            )
        if prompt_profile is not None and requested_profile != known.prompt_profile:
            raise ValueError(
                f"prompt_profile={requested_profile!r} conflicts with the known "
                f"{known.prompt_profile!r} contract for {known.model_id}"
            )
    return replace(
        base,
        key=known.key if known else "custom",
        model_id=source,
        revision=resolved_revision,
        architecture=resolved_architecture,
        prompt_profile=requested_profile,
    )


def assert_adapter_model_compatible(
    selected_model: str | Path, recorded_base: str | Path | None
) -> None:
    """Reject known cross-family adapter reuse before allocating a base model."""

    recorded = str(recorded_base or "").strip()
    if not recorded:
        return
    selected_spec = known_research_model(selected_model)
    recorded_spec = known_research_model(recorded)
    if (
        selected_spec is not None
        and recorded_spec is not None
        and selected_spec.model_id != recorded_spec.model_id
    ):
        raise ValueError(
            "LoRA adapter/base mismatch: the selected model is "
            f"{selected_spec.model_id}, but the adapter records {recorded_spec.model_id}. "
            "Train a fresh BC0 adapter for the selected learner model."
        )


__all__ = [
    "DEFAULT_RESEARCH_MODEL",
    "GEMMA4_12B",
    "GEMMA4_E4B",
    "GEMMA4_E2B_LEGACY",
    "PROMPT_PROFILE_NATIVE",
    "PROMPT_PROFILE_SMALL_FORCED",
    "RESEARCH_MODEL_SPECS",
    "ResearchModelSpec",
    "SUPPORTED_RESEARCH_ARCHITECTURES",
    "assert_adapter_model_compatible",
    "get_research_model_spec",
    "known_research_model",
    "resolve_research_model_spec",
]
