from __future__ import annotations

import copy
from typing import Any, Mapping

from psse_env.dagger.dataset_builder import (
    SUPPORTED_EXPORT_PROTOCOLS,
    bind_controller_action,
    prepare_model_policy_observation,
)
from psse_env.dagger.protocol_bridge import canonical_to_internal_action


class LocalAliasPolicyAdapter:
    """Give a model the SFT-visible observation and rebind its tool call.

    Programmatic diagnostic policies may continue to consume raw controller
    observations.  A generated model policy should be wrapped by this adapter
    so rollout inference uses the same bounded history, provenance stripping,
    and local ID aliases as SFT export.  A model trained on the canonical
    power-tool protocol should be wrapped with ``protocol="canonical"`` so its
    ``wls_from_path``/``case_path`` calls are converted back to controller
    actions before alias binding.
    """

    def __init__(
        self,
        policy: Any,
        *,
        max_history_events: int = 8,
        max_history_chars: int = 4096,
        protocol: str = "canonical",
    ) -> None:
        if protocol not in SUPPORTED_EXPORT_PROTOCOLS:
            raise ValueError(
                f"protocol must be one of {SUPPORTED_EXPORT_PROTOCOLS}, got {protocol!r}."
            )
        self.policy = policy
        self.max_history_events = int(max_history_events)
        self.max_history_chars = int(max_history_chars)
        self.protocol = protocol

    def act(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        model_observation, controller = prepare_model_policy_observation(
            observation,
            max_history_events=self.max_history_events,
            max_history_chars=self.max_history_chars,
        )
        if hasattr(self.policy, "act"):
            generated = self.policy.act(copy.deepcopy(model_observation))
        elif callable(self.policy):
            generated = self.policy(copy.deepcopy(model_observation))
        else:
            raise TypeError("Wrapped model policy must be callable or expose .act(observation).")
        if self.protocol == "canonical":
            generated = canonical_to_internal_action(generated)
        return bind_controller_action(generated, controller["state_aliases"])


__all__ = ["LocalAliasPolicyAdapter"]
