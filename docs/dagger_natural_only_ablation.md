# Round-1 natural-D1-only ablation

The preregistered `natural_dagger` arm uses an exact derived view of the
canonical `natural_dagger_probes` placement. It does not run a second replay
sampler.

## Frozen construction

The canonical full placement remains unchanged at 1,880 rows:

- 1,317 `d0_bc0` rows;
- 525 `natural_dagger1` rows; and
- 38 `observable_recovery_probe` placements.

The natural-only artifact is the order-preserving filter of that exact full
placement. It retains the identical 1,317 D0 and 525 natural-D1 row objects and
removes all 38 probe placements, producing exactly 1,842 rows. Multiplicity,
root-cap decisions, and row order are inherited from the full placement; no
row is reselected or rebalanced for the ablation.

The Round-1 aggregate publishes both immutable representations:

- `aggregate.train_view.raw.jsonl` and `aggregate.train_view.jsonl` for the
  full method;
- `aggregate.natural_only.train_view.raw.jsonl` and
  `aggregate.natural_only.train_view.jsonl` for the ablation.

The aggregate descriptor binds the versioned natural-only policy, its parent
full-view policy digest, the deterministic derivation report, and
generation-ID-independent raw/chat content hashes. The outer provenance and
`SHA256SUMS` bind the final bytes.

## Fail-closed ingestion

The Round-1 source gate reconstructs the canonical full placement from the
three immutable source ledgers, derives the natural-only projection from that
reconstruction, and compares both raw and canonical-chat artifacts
byte-for-byte. The selected SFT view must be named explicitly as `full` or
`natural-only`; the train path must be the canonical artifact for that choice.

This prevents a probe ablation from silently changing the D0 or natural-D1
sample, accepting an extra or missing row, relabeling a probe as natural, or
training on the full view while recording the natural-only arm (and vice
versa).
