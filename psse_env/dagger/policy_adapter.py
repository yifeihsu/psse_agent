from __future__ import annotations

import copy
from typing import Any, Mapping

from psse_env.dagger.dataset_builder import (
    bind_controller_action,
    prepare_model_policy_observation,
)


class LocalAliasPolicyAdapter:
    """Give a model the SFT-visible observation and rebind its tool call.

    Programmatic diagnostic policies may continue to consume raw controller
    observations.  A generated model policy should be wrapped by this adapter
    so rollout inference uses the same bounded history, provenance stripping,
    and local ID aliases as SFT export.
    """

    def __init__(
        self,
        policy: Any,
        *,
        max_history_events: int = 8,
        max_history_chars: int = 4096,
    ) -> None:
        self.policy = policy
        self.max_history_events = int(max_history_events)
        self.max_history_chars = int(max_history_chars)

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
        return bind_controller_action(generated, controller["state_aliases"])


__all__ = ["LocalAliasPolicyAdapter"]
