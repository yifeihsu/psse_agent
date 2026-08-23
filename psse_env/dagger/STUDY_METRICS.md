# DAgger study metrics gate

`psse_env.dagger.study_metrics` recomputes the preregistered model-level
metrics from schema-v3 or schema-v4 closed-loop artifacts. It does not accept
an artifact's aggregate score as evidence. A study report requires the complete
four-model matrix (`base`, `bc0`, `natural_dagger`, and
`natural_dagger_probes`) in three distinct scopes: the 30-root development
holdout, the untouched frozen suite, and the preregistered 70-episode
recovery-stress suite.

The comparison is paired by the exact physical-root identity within each
training seed. The 95% interval resamples physical roots and retains every
seed observation for a sampled root. The bootstrap seed and resample count are
written into the report.

Example:

```bash
python -m psse_env.dagger.study_metrics \
  --expected-source-commit REVIEWED_40_HEX_COMMIT \
  --base-development-run /artifacts/base-development.json \
  --base-frozen-run /artifacts/base-frozen.json \
  --base-recovery-stress-run /artifacts/base-recovery-stress.json \
  --bc0-development-run 3407=/artifacts/bc0-3407-development.json \
  --bc0-development-run 3408=/artifacts/bc0-3408-development.json \
  --bc0-development-run 3409=/artifacts/bc0-3409-development.json \
  --bc0-run 3407=/artifacts/bc0-3407.json \
  --bc0-run 3408=/artifacts/bc0-3408.json \
  --bc0-run 3409=/artifacts/bc0-3409.json \
  --bc0-recovery-stress-run 3407=/artifacts/bc0-3407-recovery-stress.json \
  --bc0-recovery-stress-run 3408=/artifacts/bc0-3408-recovery-stress.json \
  --bc0-recovery-stress-run 3409=/artifacts/bc0-3409-recovery-stress.json \
  --natural-development-run 3407=/artifacts/natural-3407-development.json \
  --natural-development-run 3408=/artifacts/natural-3408-development.json \
  --natural-development-run 3409=/artifacts/natural-3409-development.json \
  --natural-run 3407=/artifacts/natural-3407.json \
  --natural-run 3408=/artifacts/natural-3408.json \
  --natural-run 3409=/artifacts/natural-3409.json \
  --natural-recovery-stress-run 3407=/artifacts/natural-3407-recovery-stress.json \
  --natural-recovery-stress-run 3408=/artifacts/natural-3408-recovery-stress.json \
  --natural-recovery-stress-run 3409=/artifacts/natural-3409-recovery-stress.json \
  --full-development-run 3407=/artifacts/full-3407-development.json \
  --full-development-run 3408=/artifacts/full-3408-development.json \
  --full-development-run 3409=/artifacts/full-3409-development.json \
  --full-run 3407=/artifacts/full-3407.json \
  --full-run 3408=/artifacts/full-3408.json \
  --full-run 3409=/artifacts/full-3409.json \
  --full-recovery-stress-run 3407=/artifacts/full-3407-recovery-stress.json \
  --full-recovery-stress-run 3408=/artifacts/full-3408-recovery-stress.json \
  --full-recovery-stress-run 3409=/artifacts/full-3409-recovery-stress.json \
  --bc0-checkpoint 3407=/checkpoints/bc0-3407/checkpoint_receipt.json \
  --bc0-checkpoint 3408=/checkpoints/bc0-3408/checkpoint_receipt.json \
  --bc0-checkpoint 3409=/checkpoints/bc0-3409/checkpoint_receipt.json \
  --natural-checkpoint 3407=/checkpoints/natural-3407/checkpoint_receipt.json \
  --natural-checkpoint 3408=/checkpoints/natural-3408/checkpoint_receipt.json \
  --natural-checkpoint 3409=/checkpoints/natural-3409/checkpoint_receipt.json \
  --full-checkpoint 3407=/checkpoints/full-3407/checkpoint_receipt.json \
  --full-checkpoint 3408=/checkpoints/full-3408/checkpoint_receipt.json \
  --full-checkpoint 3409=/checkpoints/full-3409/checkpoint_receipt.json \
  --output /artifacts/dagger-study-decision.json
```

The default versioned manifest is
`psse_env/dagger/studies/dagger_multiseed_study_v1.json`. Each input must bind
that manifest, the reviewed source commit, its preregistered training seed,
its scope-specific suite/provenance/root-set contract, and its exact checkpoint
receipt and adapter-tree revision (null only for the base model). Natural and
full DAgger receipts must warm-start from the same-seed BC0 tree. No evaluation
scope can substitute for either of the other two.

Every checkpoint receipt also carries `production_d1_quarantine_binding`.
BC0 uses the exact not-applicable/null object. Natural and full DAgger carry
the complete zero-quarantine summary returned by the Round-1 source gate, its
generation-provenance ID, candidate/quarantine counts, and the source gate's
provenance-authenticated report SHA-256. Receipt validation recomputes the
summary arithmetic and report hash. Report construction then requires all six
natural/full receipts to agree exactly and counts that one production-D1
corpus once, not once per checkpoint.

The primary frozen comparison is `natural_dagger_probes - bc0`. Development
and frozen artifacts supply the preregistered recovery, stability, physical,
efficiency, and non-regression comparisons. The recovery-stress scope alone
supplies the seven exact recovery-action objectives, their zero-count safety
gates, and the targeted `natural_dagger_probes - natural_dagger` probe
ablation. This prevents incidental natural opportunities in either primary
scope from determining whether a recovery action was tested. Missing action-
opportunity denominators or any registry-required rate fails closed.

## Evidence boundary

The schema-v3 episode ledger authoritatively supports safe multi/single recovery,
exact accepted standard targets, target-level F1, standard-target
cardinality, mixed measurement/parameter correction sequencing, strict
physical correctness, healthy-component preservation, lifecycle safety,
tool validity, explicit state-reference validity, loops, horizons, and tool
counts.

Schema v4 adds the exact policy observation and policy-visible tool output at
every learner step. The evaluator, release gate, and study ingestion share one
pure validator: it rebuilds the four-transition history window from preceding
trace rows, binds active/candidate identifiers (and phase when the runtime
exposes it) to `state_before`, rebuilds last-tool fields, verifies the
observation hash, and recomputes the canonical-expert assessment. A coordinated
observation/hash/assessment rewrite therefore cannot create a recovery
opportunity that did not occur in the trace. The seven recovery-opportunity
denominators and autonomous-exhaustion operator-handoff denominator come only
from that reconstruction. An observed action is correct only when its full
canonical action object exactly matches the recomputed expected action.

The recovery-stress artifact is generated from deterministic states derived
from 20 roots in the protected 30-root development parent. It contains exactly
ten episodes in each of seven intervention strata (70 total), while its root
set has zero overlap with D0, natural D1, observable probe training, or the
frozen suite. The real production environment applies each pre-policy
intervention, and the observable expert independently derives the expected
canonical action from the resulting policy-safe observation; no target label
is stored in the model-visible scenario.

Each row's state SHA-256 is recomputed from the exact persisted canonical
lifecycle snapshot, and adjacent before/after snapshots must form one exact
chain. The first and last active-state identities must also match the
evaluator-owned initial/final identities in the privileged offline audit. Thus
rewriting every trace copy of an active state and recomputing all trace-local
hashes still fails closed at the independent episode endpoint anchors.

Every persisted policy observation and transition is scanned recursively for
forbidden policy fields and privileged provenance. Forbidden-key comparison is
case- and separator-insensitive, so forms such as `HiddenTruth`,
`GROUND_TRUTH`, and `TrueMeasurementErrors` are equivalent to their canonical
denylist names. Any detected path increments the hidden-truth leakage safety
count and is a release-gate evidence failure.

For successful WLS and candidate-verification calls, schema v4 carries a narrow
copy of the observable, state-bound tool certificate. The certificate is
recomputed from the persisted policy-visible tool output; its state ID must
equal the executed action target and its state hash and the trace runtime hash
must be identical, non-null lowercase SHA-256 values. Final residual acceptance
requires a successful certificate for the final active state, observable source
provenance, finite nonnegative residual/statistic values, a positive chi-square
threshold, and resolution flags that agree with the strict statistic-threshold
comparison. Post-commit feasibility requires the earlier verification of the
exact promoted candidate and the complete MATPOWER
`observed_snapshot_topology_vm_rate_a` certificate: connectivity, bus-voltage,
and active-branch RATE_A checks, with internally consistent input-error and
violation counts. It is not an unconstrained AC power-flow claim. A complete
certificate with any violation is a physically unsafe commit; a missing,
unrecognized, or incomplete certificate makes both the feasibility objective
and the zero-unsafe-commit gate unevaluable rather than silently safe.

Schema v3 remains ingestible for archival comparisons, but these newer
opportunity, residual, feasibility, handoff, and leakage fields are emitted as
`null` with `evidence_available: false`; every associated manifest rule fails
closed. Production-D1 quarantine remains separate source-gate evidence consumed
from the checkpoint receipts. None of these facts may be replaced by a recovery
outcome or another proxy.

Exit code `0` means every comparison and objective rule passed. Exit code `2`
means a rule failed or evidence was missing/ambiguous. The JSON report is
content-addressed and is still written for an ordinary threshold failure.
