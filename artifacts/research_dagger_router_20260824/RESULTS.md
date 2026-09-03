# Research-only DAgger router: preliminary result

Date: 2026-08-24

This is a CPU-only research prototype trained from the existing anchor corpus
and two learner-trace iterations. It does not replace the paused strict
production pipeline and it does not claim the preregistered 90% recovery goal.
No protected held-out root was used for fitting.

## Training and validation

- Training view: 954 unique observable-state rows, 202 features, 271 physical
  roots, with zero overlap against the four protected held-out roots.
- Frozen original validation: 27/27 exact actions.
- Iteration-1 downstream validation: 39/43 exact actions.
- Iteration-2 downstream validation: 23/25 exact actions.
- Schema-valid and state-bound actions: 100% on all three validation views.
- Targeted tests: 29 passed.
- Router artifact SHA-256:
  `67784f3579effbe03f7f2ac426f17724e67c59296c3c5386fbbe3ccc4a38766d`.

The router now reuses the canonical observable candidate-disposition rule from
`release_factories.py`. This removed the remaining false rollback caused by a
separate hard-coded 0.95 partial-closure threshold.

## Untouched paired 14-episode result

| Policy | Primary resolved | Terminal | Loops | Invalid actions | Audit-verified completion followed by safe handoff |
| --- | ---: | ---: | ---: | ---: | ---: |
| BC0 Gemma adapter | 0/14 | 0/14 | 14/14 | 146 | 0/14 |
| 32-step learner-trace Gemma adapter | 0/14 | 0/14 | 14/14 | 136 | 0/14 |
| CPU DAgger router, iteration 2 | 0/14 | 14/14 | 0/14 | 1 | 6/14 (42.9%) |

For the CPU router, false commits, false finalizations, false rollbacks, and
healthy-component corruption are all zero. The one invalid action was a
visible, supported parameter correction that failed inside the solver with
`parameter_correction_failure` (non-positive finite-difference denominator);
the policy recovered and completed that episode with an audit-verified safe
operator handoff.

The paired evaluator score improved from 0.0 for both Gemma policies to 2.0
for the router. The result artifact is
`router_eval_14_iter2_canonical.json`, content SHA-256
`8c811c9167dd1ef847122e24fc82e9caddfeaa4cca5cd5607451408e255cf926`.

## Honest limitation

The primary autonomous multi-error recovery rate remains 0/14. All 14 terminal
outcomes are operator escalations because the production controller requires an
additional observable post-correction confirmation. Offline truth audit shows
that six episodes were physically complete and preserved healthy components,
but the runtime correctly refused to relabel those handoffs as autonomous
resolution. The multi-measurement roots still expose only the dominant fault
after correction, so selecting all remaining true measurements cannot be done
safely from the current observable evidence.

The completed 32-step Gemma continuation did update its LoRA weights: at
checkpoint 32, 25,337,829 of 25,337,856 values differ from BC0 and the LoRA-B
delta is 12.89% of the BC0 B-norm. Nevertheless, all 27 greedy validation
actions stayed byte-identical across BC0 and checkpoints 8/16/24/32. This
supports greedy-margin/domain mismatch rather than failed training or stale
checkpoint selection, although a direct logit comparison would be needed to
exclude an inference-time adapter-loading bug conclusively.

## Next research step

The efficient next step is an observable confirmation/probe mechanism that can
reveal masked remaining measurement faults after a safe partial commit. It
should be CPU-validated on root-disjoint scenarios before any further GPU SFT.
Another short Gemma continuation is not justified by the current evidence.
