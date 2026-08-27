# Research prototype

A minimal imitation-learning pipeline for academic use. The default method is
**recovery-selective, truth-audited DAgger**: supervision is retained only for
learner-visited recovery states whose expert target is observably rank-one and
passes a privileged offline truth audit. This is a *selected subset* of the
standard-DAgger occupancy distribution, so standard no-regret arguments do not
directly apply. Two broader controls partially disambiguate the source of
occupancy breadth:

| Inclusion | Learner recovery | Learner non-recovery | Expert-visited | Initial |
|---|---:|---:|---:|---:|
| `selective` | Yes | No | No | No |
| `learner_full` | Yes | Yes | No | No |
| `full_occupancy` | Yes | Yes | Yes | Yes |

All three modes require an expert target that passes the privileged offline
truth audit. Audit-failed states are written to a `teacher_abstentions.jsonl`
stratum rather than silently dropped, and retention never depends on whether
the learner's own action was well-formed. Thus `learner_full` retains invalid
learner-action states but excludes both expert-visited and initial states. It
reuses the scientific core of `psse_env` and skips the release scaffolding.
The aggregate report records `round1_selected_state_origin_breakdown`; use it
to verify the realized arm before training. Because `full_occupancy` also adds
initial states, its contrast with `learner_full` is not strictly an
expert-origin-only contrast unless initial-state exposure is controlled
separately. Likewise, `learner_full` adds both non-recovery states and recovery
states whose target did not pass the observable rank-one proof, so it is not a
pure non-recovery-only contrast.

The Round-1 train/validation assignment is frozen on the complete audited
occupancy *before* applying any inclusion rule. This keeps physical-root
assignment identical across the three arms. Historical aggregates made before
this rule must be identified as `legacy_split` rather than treated as a clean
three-level inclusion comparison.

Keep optimization controls distinct in reports: fixed optimizer updates,
fixed corpus passes, and fixed sampled supervised-token exposure answer
different questions. The completed legacy Exp1 runs used 666 updates for E2B
and 662 for 12B; they must not be described as one universal 662-update cell.
Training receipts record both prepared-corpus token totals and the rows/input
tokens/supervised tokens actually collated during training.

## Why this exists

The `psse_env` pipeline is built to produce a releasable artifact. Before it
will train or collect anything it verifies content-addressed dataset
provenance, exact source-commit equality, frozen predecessor root sets, a
byte-pinned study manifest, expert and base baselines measured on a frozen
suite, and an approved accelerator class. Those checks are appropriate for a
production release and are pure friction for a demo on a single local GPU.

This package keeps what determines the result and drops what determines
releasability.

## What is reused unchanged

These decide what the model sees and learns, so they are imported, never
reimplemented — a divergent copy would silently invalidate any comparison
against the release path.

| Component | Module |
|---|---|
| Environment | `psse_env.dagger.release_factories.production_environment_factory` |
| Expert oracle | `psse_env.oracle.ExpertPolicyOracle` |
| DAgger loop, expert labelling, recovery strata | `psse_env.dagger.rollout_collector.DaggerRolloutCollector` |
| Prompt render, tokenization, assistant-only masking | `psse_env.sft.gates.prepare_example` |
| Padding collator preserving `-100` | `psse_env.sft.collator.AssistantOnlyCollator` |
| Canonical action validation, alias binding | `psse_env.dagger.release_factories.GemmaReleasePolicy` |

## What is replaced

| Replaced | Lines | With |
|---|---|---|
| `psse_env/dagger/collect_dagger1.py` | ~3900 | `research/collect.py` |
| `psse_env/sft/{cli,training,gates}.py` gate paths | ~4000 | `research/train.py` |

## Usage

One full iteration:

```bash
python -m research.run_dagger --round0-dir DIR --scenarios FILE --work-dir OUT
```

Choose the Round-1 occupancy arm with `--inclusion selective`,
`--inclusion learner_full`, or `--inclusion full_occupancy`.

Stages run in order and each writes its artifacts, so a later failure does not
cost the earlier stages. Re-run a subset with `--stages`:

```bash
python -m research.run_dagger --round0-dir DIR --scenarios FILE --work-dir OUT --stages collect,aggregate,round1,evaluate --bc0-adapter OUT/bc0
```

Individual stages are also runnable on their own:

```bash
python -m research.train --train rows.jsonl --output-dir bc0
python -m research.collect --scenarios scenarios.json --output rows.jsonl --adapter bc0 --beta 0.3
python -m research.evaluate --scenarios holdout.json --adapter bc0 --label bc0
```

Audit one or more compact evaluation reports by deterministic replay against
the clean hidden state:

```bash
python -m research.physical_outcome_audit \
  --scenarios holdout.json \
  --evaluation checkpoint=checkpoint.evaluation.json \
  --output physical_audit.full.json \
  --summary-output physical_audit.summary.json
```

The primary exact-recovery metric is computed on the final active physical
state, while stable-terminal exact recovery additionally requires a stable
trajectory outcome. Mutable oracle-ledger counts are diagnostics only. The
versioned all-eight historical replay is retained in
`results/physical_audit_all8_full.json` with its compact companion
`results/physical_audit_all8_summary.json`. Their immutable scenario suite and
eight compact source evaluations are retained under
`results/physical_audit_all8_inputs/`. The old compact evaluations lacked both
the suite hash and per-episode identities, so their scenario binding is
explicitly recorded as ordered-index alignment; it is not identity-proven.
Schema-v2 evaluations bind both the suite hash and per-episode physical
identity.

Select the base model with `--model-id` / `--revision`, or the
`PSSE_RESEARCH_MODEL_ID` / `PSSE_RESEARCH_MODEL_REVISION` environment
variables. It defaults to Gemma 4 E2B, which fits a 16 GB card in 4-bit.

## Notes

`beta` is the probability of deferring to the expert. `beta=0` is a pure
on-policy diagnostic pass; DAgger-1 training normally mixes the expert in
around 0.25–0.5. Rolling out an untrained base model at `beta=0` yields no
usable data — the policy never emits a valid action — so collection is worth
running only against a trained adapter.

The aggregate splits round-1 rows by episode, not by row. Rows from one
episode share a scenario, so an independent row split would put the same
physical situation on both sides.

Every report carries `release_evidence: false`. The dated HPC screening cell
adds source, input, model, exposure, hardware, evaluation, and replay receipts
for research reproducibility, but those receipts deliberately do not satisfy
the production-release contract. No output of this package is release
evidence.
