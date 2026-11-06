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
  of the compound-error study, 2026-10-28; the provider reports the cue only
  to this teacher, ``variant_provider_options``).  ``baseline_compare`` and
  the seven conventional families are untouched by either variant;
* ``baseline_compare_guarded_walk``: the guarded teacher whose balanced routes
  follow the ranking of the confidence over meter, parameter and topology,
  and which, where it would hand off after a hypothesis's commits went
  fruitless, reverts to the state before that hypothesis and walks the
  ranking to the next one (``hypothesis_walk=True``; item 7 step 6 of the
  compound-error study, the user's decision of 2026-10-28).  Its environment
  carries the gated revert (``variant_environment_options``:
  ``revert_fruitless_cycles``);
* ``baseline_compare_guarded_routes``: the guarded teacher that, where the
  solve's confidence says the impedance edit and the breaker flip compete,
  compares the two routes on isolated clones before correcting either and
  takes the best-verified route first (``route_comparison=True``; item 7
  step 7, the user's request of 2026-10-28).  Its environment carries the
  ``compare_routes`` tool (``variant_environment_options``);
* ``baseline_compare_meter_route``: ``baseline_compare`` with the meter-route
  rule (``meter_route_residual=METER_ROUTE_RESIDUAL``): after a committed
  meter correction, while the solve still reports a maximum normalized
  residual of at least 11, the meter route leads and the measurement
  expert's post-commit branch guard stands down (compound errors on IEEE
  118, 2026-10-29, docs/compound_errors_ieee118_20261028.md section 10).

Research only.  The default for new work, what a pipeline env or a research
script picks up when it names no variant, is ``baseline_compare`` (the user's
decision of 2026-10-28; until then the default was ``baseline``).  A teacher
decides when to look from observable evidence; exhaustive screening is never
adopted to raise the success rate, so the always-on topology-first order
stays a flagged research variant, never a default, and the clean-flip teacher
of the same day, which screened the node/breaker estimate on every anomalous
solve, was removed (the compound-error study, section 7.8).  Every recorded
cell that names its teacher keeps it; receipts record the variant beside the
evidence profile.
"""
from __future__ import annotations

import os
from typing import Any

from psse_env.oracle.learned_ranker import DEFAULT_RANKER_MODEL

EXPERT_VARIANT_ENV = "PSSE_EXPERT_VARIANT"
#: The teacher of new work when none is named (2026-10-28); recorded cells
#: name theirs.  Until then the default was ``baseline``.
DEFAULT_EXPERT_VARIANT = "baseline_compare"
#: The teacher of the compound-error study from here on (the user's decision
#: of 2026-10-29, in their words: "Use baseline_compare_guarded_routes as the
#: compound study's teacher from here on; baseline_compare stays the default
#: everywhere else and the paper results stay as they are").  The compound
#: scripts default to it; every other path keeps ``DEFAULT_EXPERT_VARIANT``.
COMPOUND_STUDY_EXPERT_VARIANT = "baseline_compare_guarded_routes"
EXPERT_VARIANTS = ("baseline", "ledger", "ledger_ranked", "ledger_ranked_compare", "baseline_compare",
                   "baseline_compare_topology_first", "baseline_compare_topology_guarded",
                   "baseline_compare_guarded_walk", "baseline_compare_guarded_routes", "baseline_compare_meter_route")
#: The meter-route rule's threshold: the separation of section 9.4 of the IEEE
#: 118 compound note (meter-next 32 of 32 and 37 of 39 at or above it,
#: line-next 4 of 19 and 4 of 36).
METER_ROUTE_RESIDUAL = 11.0


def compound_study_expert_variant(system: str | None = None) -> str:
    """The compound scripts' teacher for a system when a run names none.

    IEEE 14 keeps ``COMPOUND_STUDY_EXPERT_VARIANT`` (the user's decision of
    2026-10-29, taken on the IEEE 14 study).  Every other system keeps
    ``DEFAULT_EXPERT_VARIANT``, the teacher the IEEE 118 study recorded its
    runs with: the route teacher's cue and route comparison were set on IEEE
    14's node/breaker roots and have not been run on a bus-branch suite.
    Naming ``--expert`` overrides either.
    """
    from psse_env.systems import resolve_system

    case_id = resolve_system(system).case_id if system else "case14"
    return COMPOUND_STUDY_EXPERT_VARIANT if case_id == "case14" else DEFAULT_EXPERT_VARIANT


def current_expert_variant(variant: str | None = None) -> str:
    """The variant named explicitly, else by the environment, else the default."""
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
    if name == "baseline_compare_guarded_walk":
        return {"candidate_comparison": True, "topology_first": "flow_at_noise", "hypothesis_walk": True}
    if name == "baseline_compare_guarded_routes":
        return {"candidate_comparison": True, "topology_first": "flow_at_noise", "route_comparison": True}
    if name == "baseline_compare_meter_route":
        return {"candidate_comparison": True, "meter_route_residual": METER_ROUTE_RESIDUAL}
    if name == "ledger":
        return {"hypothesis_ledger": True}
    options: dict[str, Any] = {
        "hypothesis_ledger": True,
        "learned_ranker": learned_ranker if learned_ranker is not None else str(DEFAULT_RANKER_MODEL),
    }
    if name == "ledger_ranked_compare":
        options["candidate_comparison"] = True
    return options


#: Provider options a teacher variant needs in order to see its own routing
#: signal: the guarded teacher reads the open-terminal cue, which no other
#: teacher's environment reports (the stage-2 review of 2026-10-28).  The
#: environment factories merge these in unless a run sets the option itself.
VARIANT_PROVIDER_OPTIONS: dict[str, dict[str, Any]] = {
    "baseline_compare_topology_guarded": {"flow_at_noise_cue": True},
    "baseline_compare_guarded_walk": {"flow_at_noise_cue": True},
    "baseline_compare_guarded_routes": {"flow_at_noise_cue": True},
}

#: Item 7 step 6: the gated revert's fruitless-cycle count, chosen from the
#: guarded teacher's stage-2 trajectories: 96% of the committed states of its
#: solved compound roots moved on to their next commit within 2 fruitless
#: correction cycles (91% within none), so a hypothesis still fruitless after
#: 3 is past what a productive state needed (the compound-error study,
#: section 8.6).  A candidate for learning, with when to abandon a hypothesis.
REVERT_FRUITLESS_CYCLES = 3

#: Environment options a teacher variant needs: the walk teacher's gated revert.
VARIANT_ENVIRONMENT_OPTIONS: dict[str, dict[str, Any]] = {
    "baseline_compare_guarded_walk": {"revert_fruitless_cycles": REVERT_FRUITLESS_CYCLES},
    "baseline_compare_guarded_routes": {"route_comparison": True},
}


def variant_provider_options(variant: str | None = None) -> dict[str, Any]:
    """Provider keyword arguments the named (else current) teacher variant needs."""
    return dict(VARIANT_PROVIDER_OPTIONS.get(current_expert_variant(variant), {}))


def variant_environment_options(variant: str | None = None) -> dict[str, Any]:
    """Environment keyword arguments the named (else current) teacher variant needs."""
    return dict(VARIANT_ENVIRONMENT_OPTIONS.get(current_expert_variant(variant), {}))


__all__ = ["COMPOUND_STUDY_EXPERT_VARIANT", "DEFAULT_EXPERT_VARIANT", "EXPERT_VARIANTS", "EXPERT_VARIANT_ENV", "METER_ROUTE_RESIDUAL",
           "REVERT_FRUITLESS_CYCLES", "VARIANT_ENVIRONMENT_OPTIONS", "VARIANT_PROVIDER_OPTIONS", "compound_study_expert_variant",
           "current_expert_variant",
           "expert_variant_options", "variant_environment_options", "variant_provider_options"]
