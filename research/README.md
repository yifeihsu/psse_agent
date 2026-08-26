# Research prototype

A minimal DAgger pipeline for academic use. It reuses the scientific core of
`psse_env` and skips the release scaffolding.

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

Every report carries `release_evidence: false`. Nothing here verifies
provenance, pins commits, or attests hardware, so no output of this package is
evidence about a release candidate.
