# Multi-Scan HIF Revision Report

Date: 2026-07-14

## Decision

The corrected multi-scan pipeline is physically consistent and replayable, but
the current measurement set does not support production point estimates of
along-line HIF position. Exact-alpha SFT generation is therefore a **no-go**.
The estimator remains useful for line confirmation, searched phase, HIF
resistance/current/power, and a near-best alpha interval with an explicit
weak-observability warning.

## Physics Correction

The original exporter segmented OpenDSS terminal arrays with phase count rather
than conductor count. Four-conductor transformer terminals therefore mixed the
neutral and phase entries at the receiving terminal. Terminal extraction now
segments by `NumConductors * NumTerminals` and selects phase nodes 1, 2, and 3
from `NodeOrder`.

| Check | Frozen baseline | Corrected data |
| --- | ---: | ---: |
| Median transformer `abs(Pt)/abs(Pf)` | 0.667371 | 1.000000 |
| Median bus 6/7/9 balance residual | 0.094642 pu | 0.0000036 pu |
| Maximum transformer loss mismatch | not authoritative | `8.32e-15` MW/Mvar |
| 122-entry ordering | unchanged | unchanged |
| Hidden HIF bus exposed | no | no |

The failing baseline is frozen at
`artifacts/measurements/hif_multiscan_v0_pre_transformer_fix_20260714/`.
Its reproducibility record identifies source commit
`8b3a29ce58d89195547848c894a5c16d3ba14056`.

## Dataset QA

Two matched 17-event by 20-scan datasets were generated with identical HIF
labels: one with diverse operating points and one with an identical operating
point plus independent noise. All 17 eligible lines are represented.

| Dataset | Clean replays | Control rank | Measurement rank | Strict QA |
| --- | ---: | ---: | ---: | --- |
| Identical-noise | 340/340 | 0 | 0 | pass |
| Diverse | 340/340 | 16-17 | 19 | pass |

Every scan uses the canonical explicit operating-point schema. Event-label,
operating-point, and measurement-noise random streams are independent. The
generator stores `z_clean` only as hidden QA data so deterministic replay can
be checked without using labels in estimator inputs.

## Controlled Benchmark

The benchmark searched phases A, B, and C. Synthetic phase and parameter labels
were used only for evaluation. NLM top-1 was correct for 16 of 17 events; the
miss was retained. The common comparative configuration was a 7 by 9 coarse
grid, two refinement seeds, and 20 local evaluations.

| Condition | Time | Phase acc. | Median alpha err. | P90 alpha err. | Median R err. | Top-k truth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A: single | 20.7 s | 87.5% | 0.380 | 0.666 | 32.5% | 25.0% |
| B: identical 20 | 38.5 s | 100% | 0.181 | 0.406 | 7.9% | 37.5% |
| C: diverse 5 | 114.1 s | 100% | 0.320 | 0.649 | 21.5% | 18.8% |
| D: diverse 10 | 214.4 s | 100% | 0.245 | 0.541 | 11.0% | 18.8% |
| E: diverse 20 | 394.1 s | 100% | 0.195 | 0.394 | 8.0% | 37.5% |

False precision was zero and near-best alpha interval coverage was 100%, but
the line, alpha, and top-k gates failed. More importantly, the identical-noise
control had lower median alpha error than the 20-scan diverse condition. The
completed multi-scan diagnostics were classified as `noise_averaging_only`;
isolated finite-difference failures were classified as `diagnostic_partial`.

The 7 by 9 search is a controlled comparison configuration, not a production
31 by 35 trace-generation grid. A denser grid cannot establish the missing
sensitivity diversity, so production trace generation was not started after
the observability gate failed.

## Runtime Choice

Retain up to 20 candidate scans and select 10 with information-greedy selection
for the default estimator call. Ten diverse scans took 214 seconds versus 394
seconds for all 20, while median resistance error changed from 11.0% to 8.0%.
Neither setting made alpha production-ready.

## Trace Policy

Point alpha is emitted only when a multi-scan result is non-ambiguous,
parameter-identifiable, and has status `full_rank_well_conditioned` or
`full_rank_weakly_conditioned`. For `noise_averaging_only`,
`diagnostic_partial`, or `rank_deficient`, final evidence omits point alpha and
reports `near_best_alpha_interval` instead. Label-backed mock estimates are
tagged `synthetic_oracle` and rejected by the production trace validator.

Because the exact-alpha gate failed, matched no-fault production windows and
messages-based production traces were intentionally not generated. They remain
downstream work after measurement diversity is improved and the benchmark is
rerun. Recommended additions are synchronized terminal voltage/current phasors
or per-phase terminal power/current measurements on the suspected line.

Machine-readable results are in
`artifacts/measurements/hif_multiscan_benchmark_fixed_results_17_20260714/`.
