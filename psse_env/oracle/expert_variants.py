"""Named expert variants under study (hypothesis-ranking plan, steps 3 to 5).

The teacher a pipeline stage builds is chosen once, by name, from the
environment variable ``PSSE_EXPERT_VARIANT`` (the HPC cell exports it from
``EXPERT_VARIANT`` in pipeline.env), so the aggregate generator, the DAgger
collector, the training-decision audit and the evaluation's expert arm all
construct the same expert:

* ``baseline``: the rule expert as shipped (step 2 contract);
* ``ledger``: the baseline following the balanced screen's hypothesis ledger
  (``ExpertPolicyOracle(hypothesis_ledger=True)``, step 3);
* ``ledger_ranked``: the ledger expert with the learned ranker's acquisition
  deferral (``learned_ranker=DEFAULT_RANKER_MODEL``, step 4 decision).

Research only.  The default is the baseline, so nothing changes unless a
run names a variant; receipts record the variant beside the evidence
profile.
"""
from __future__ import annotations

import os
from typing import Any

from psse_env.oracle.learned_ranker import DEFAULT_RANKER_MODEL

EXPERT_VARIANT_ENV = "PSSE_EXPERT_VARIANT"
DEFAULT_EXPERT_VARIANT = "baseline"
EXPERT_VARIANTS = ("baseline", "ledger", "ledger_ranked")


def current_expert_variant(variant: str | None = None) -> str:
    """The variant named explicitly, else by the environment, else the baseline."""
    name = str(variant if variant is not None else os.environ.get(EXPERT_VARIANT_ENV, "")).strip() or DEFAULT_EXPERT_VARIANT
    if name not in EXPERT_VARIANTS:
        raise ValueError(f"unknown expert variant {name!r}; expected one of {', '.join(EXPERT_VARIANTS)}")
    return name


def expert_variant_options(variant: str | None = None, *, learned_ranker: Any = None) -> dict[str, Any]:
    """Keyword arguments for ``ExpertPolicyOracle`` that select the variant.

    ``learned_ranker`` overrides the model path of the ranked variant (a
    research harness pointing at a fresh export); it is ignored otherwise.
    """
    name = current_expert_variant(variant)
    if name == "baseline":
        return {}
    if name == "ledger":
        return {"hypothesis_ledger": True}
    return {"hypothesis_ledger": True, "learned_ranker": learned_ranker if learned_ranker is not None else str(DEFAULT_RANKER_MODEL)}


__all__ = ["DEFAULT_EXPERT_VARIANT", "EXPERT_VARIANTS", "EXPERT_VARIANT_ENV", "current_expert_variant", "expert_variant_options"]
