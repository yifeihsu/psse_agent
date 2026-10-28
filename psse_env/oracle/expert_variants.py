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
  deferral (``learned_ranker=DEFAULT_RANKER_MODEL``, step 4 decision);
* ``ledger_ranked_compare``: ``ledger_ranked`` that compares the ranked
  parameter candidates on isolated branches before correcting one when the
  ranking does not clearly dominate (``candidate_comparison=True``, the
  decision audit's interface repair of 2026-10-19);
* ``baseline_compare``: the baseline with the same bounded comparison and no
  ledger or ranker, the teacher for profiles without the balanced screen the
  ledger reads (scada_only; the conventional-only cell of 2026-10-22);
* ``baseline_compare_topology_first``: ``baseline_compare`` with the breaker
  route ahead of the parameter route inside the structural tier
  (``topology_first=True``; the compound-error study of 2026-10-27, which
  found the parameter route masking breaker errors on roots that carry both
  parameter scans and the node/breaker channel);
* ``baseline_compare_topology_guarded``: ``baseline_compare`` with the breaker
  route first only in a state whose WLS solve reports the open-terminal cue
  (``topology_first="flow_at_noise"``: a top branch candidate whose terminal
  flows read at noise while the stale model routes flow through it; stage 2
  of the compound-error study, 2026-10-28).  ``baseline_compare`` and the
  seven conventional families are untouched by either variant.

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
EXPERT_VARIANTS = ("baseline", "ledger", "ledger_ranked", "ledger_ranked_compare", "baseline_compare",
                   "baseline_compare_topology_first", "baseline_compare_topology_guarded")


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
    if name == "baseline_compare":
        return {"candidate_comparison": True}
    if name == "baseline_compare_topology_first":
        return {"candidate_comparison": True, "topology_first": True}
    if name == "baseline_compare_topology_guarded":
        return {"candidate_comparison": True, "topology_first": "flow_at_noise"}
    if name == "ledger":
        return {"hypothesis_ledger": True}
    options: dict[str, Any] = {
        "hypothesis_ledger": True,
        "learned_ranker": learned_ranker if learned_ranker is not None else str(DEFAULT_RANKER_MODEL),
    }
    if name == "ledger_ranked_compare":
        options["candidate_comparison"] = True
    return options


__all__ = ["DEFAULT_EXPERT_VARIANT", "EXPERT_VARIANTS", "EXPERT_VARIANT_ENV", "current_expert_variant", "expert_variant_options"]
