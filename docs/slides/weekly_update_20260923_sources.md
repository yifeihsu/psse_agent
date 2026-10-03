# Weekly update: source index

Deck date: September 23, 2026. The source is `weekly_update_20260923.tex`.
The design follows `docs/slides/weekly_update.tex`: the original 4:3 default
Beamer theme, Latin Modern, blue headings and bullets, and white background.
The original template is preserved.

All reported results come from local saved artifacts; this slide task did not
run training, physical simulations or HPC evaluations. File hashes are captured
in the accompanying `source_manifest.json`. Paths below are relative to the
repository root unless linked otherwise.

## Evidence groups

**S1 — Physical corpora and admission.**

- [Revision history](../ieee14_hif_legacy_reconfiguration_20260919.md).
- [Latest HIF generation receipts](../../output/hif_physical_revision_20260923b/generation_receipts.json).
- [Latest HIF validation receipts](../../output/hif_physical_revision_20260923b/validation_receipts.json).
- [HIF detection audit](../../output/hif_physical_revision_20260923b/detection_audit/summary.json).
- [Filtered HIF re-audit](../../output/hif_physical_revision_20260923b/detectable_audit/summary.json).
- [Unbalance audit](../../output/hif_physical_revision_20260923b/unbalance_audit/summary.json).
- [Filtered unbalance re-audit](../../output/hif_physical_revision_20260923b/detectable_unbalance_audit/summary.json).
- [Corpus fact sheet](../../output/weekly_update_20260923/corpus_notes.md).

The `20260923b` generation and audits completed while this deck was prepared.
The main HIF admitted counts remain 27 + 8 + 77 + 19 = 131; unbalance remains
160/440. All six strict HIF physics replays and all six current-localization
checks pass. Two of 18 HIF validation processes return failure because the
legacy NLM ranking has five top-1 misses (four sweep rows and one training-extra
row). The deck does not characterize all validators as passing.

The main HIF source has 420 windows, and the six raw HIF cohorts including
evaluation-only sources have 777 windows / 7,770 scans plus 100 controls.
Admitted cases occupy 100–200 and 200–500 ohm bands; the actual maximum is
approximately 300.683 ohm. Three admitted first-scan alarms also occur in the
paired healthy observations. Admission alone therefore does not establish
fault attribution or localization.

**S2 — Actual fault and noise settings.**

- [Scalar source generator](../../Transmission/generate_measurements.py).
- [DAgger scenario generator](../../psse_env/providers/scenario_generator.py).
- [Unbalance generator](../../Transmission/generate_measurements_imbalance.py).
- [Harmonic source/transducer model](../../Transmission/generate_hse_traces.py).
- [Shared fault profiles](../../psse_env/fault_profiles.py).
- [Reviewed sensitivity profile](../fault_scenario_review_20260917.md).

Main-source meter errors start at signed 5–15 sigma. The current pipeline
enforces a 10-sigma floor by redrawing smaller corrupted readings at 10–15
sigma relative to the reference, retaining larger readings. The resulting
population is not exactly uniform 10–15 sigma. Mixed cases add one power-channel
offset of signed 0.10–0.30 pu. Parameter factors are **true physical / reported
canonical**, independently drawn for R and X in an RX fault.

Noise sigmas refer to physical sensor components. Topology projections propagate
covariance; structural-zero constraints are handled separately. Matching draws
and estimator covariance does not establish the combined detector's false-alarm
calibration. The runtime alarm uses greater-than-or-equal comparisons; admission
uses `J > 1.25 * threshold` or `max_normalized_residual >= 5`.

Separate moderate/sensitivity populations are not credited as the completed
DAgger training population: parameter factors 0.70–0.95 / 1.05–1.30, harmonic
THD 1–5%, power-noise sigma 0.005 / 0.002 pu, and weak HIF evaluation sweeps.

**S3 — Completed original DAgger results.**

- [Results report](../../output/r2_status_20260922/completed/RESULTS.md).
- [Paired comparison](../../output/r2_status_20260922/completed/comparison.json).
- [Pipeline summary](../../output/r2_status_20260922/completed/pipeline_summary.json).
- [Action-counter semantics](../../output/r2_status_20260922/completed/failure_semantics_check.json).

Original BC0/R1/R2/expert successes: 132/140/141/142 of 160. R1 to R2 has
two gains and one regression. R2's 85 truth-correct handoffs must not be confused
with its 73 handoffs meeting the stricter stored completion certificate.
Original R2's 16 false-finalization counts were completed but audit-failing
finalizations, including eight healthy-control reference errors.

**S4 — Revised frozen R2 evaluation.**

- [Results](../../output/revised_eval_20260922/r2_completed/RESULTS.md).
- [Metric definitions and counts](../../output/revised_eval_20260922/r2_completed/performance_summary.json).
- [Comparison](../../output/revised_eval_20260922/r2_completed/comparison.json).
- [Run configuration](../../output/revised_eval_20260922/r2_completed/run_configuration.json).
- [Completed receipt](../../output/revised_eval_20260922/r2_completed/completed.json).

157 final-state truth passes = 63 resolved + 90 truth-correct handoffs + four
nonterminal states. Correct and terminal = 153. Stored audited completion =
136 (63 resolved + 73 qualified handoffs). All 26 revised false-finalization
attempts and 15 rejected commit attempts were nonmutating rejections. Accepted
wrong-target edits are counted separately by the truth audit.

**S5 — Revised expert evaluation.**

- [Results](../../output/revised_eval_20260922/expert_completed/RESULTS.md).
- [Comparison](../../output/revised_eval_20260922/expert_completed/comparison.json).
- [Completed receipt](../../output/revised_eval_20260922/expert_completed/completed.json).

158 truth-correct terminal outcomes = 62 resolved + 96 truth-correct handoffs.
The final sentence in the historical expert report says R2 was still running;
that status sentence is superseded by S4's completed receipt.

**S6 — HIF-conditioned meter recovery.**

- [Method and limitations](../hif_conditioned_meter_continuation_20260922.md).
- [24-episode replay metrics](../../output/hif_continuation_fix_20260922/final_verified/report_metrics.json).
- [Worked overlap episode](../../output/hif_continuation_fix_20260922/final_verified/r0_5fb25759717c_controlled_overlap.json).
- [Physical-refusal example](../../output/hif_continuation_fix_20260922/final_verified/r0_de51c28ced3e_controlled_overlap.json).
- [Detailed mathematical notes](../../output/weekly_update_20260923/hif_method_notes.md).

These are September 22 auxiliary-assisted results on the frozen **pre-fix**
physical model. The current reactive-limit fix changes the replay model; old
measurements must not be rescored with it as though the physical input were
unchanged. Conditioning preserves all channels and R but omits HIF-estimation
uncertainty from R. Sampled uncertainty rectangles are not confidence intervals.

The overlap example adds a second meter fault to a root already containing one.
Its bus-1 injection HIF effect is +0.152163011507 pu, and its added meter bias is
-0.1 pu. Both faulty targets are repaired. Both reported before/after WLS
statistics are **conditioned** statistics, isolating the meter-repair stage.

**S7 — New SCADA-only protocol.**

- [Protocol](../scada_only_discovery_20260922.md).
- [Boundary/test receipt](../../output/scada_only_protocol_20260922/verification_summary.json).
- [Eight-root initial-WLS check](../../output/scada_only_protocol_20260922/frozen_mixed_smoke.json).

This profile blocks auxiliary evidence and HIF replay; the earlier reported
repair scores are not SCADA-only results. The eight-root smoke checks only
initial alarm/action invariance on the historical frozen inputs, not completed
repair, fault identification or new-corpus population performance.

**S8 — Training/data provenance.**

- [D0/BC0 receipts](../../output/r2_status_20260922/completed/pipeline_summary.json).
- [R1 round summary](../../output/r1_failure_analysis_20260922/candidate_artifacts/round_summary.json).
- [R2 round summary](../../output/r2_status_20260922/completed/round_summary.json).
- [Consolidated results notes](../../output/weekly_update_20260923/results_notes.md).

BC0: 3,784 train rows, 128-row training-validation subset, 946 total steps.
R1: 688 replay + 688 new = 1,376 rows, 344 steps. R2: 660 replay + 660 new =
1,320 rows, 330 steps. R2's prior pool contains 3,784 D0 + 688 R1 rows.
D0 lacked eligible mixed-HIF and healthy-telemetry training families; each
DAgger round added only six eligible mixed-HIF rows. The conditioned repair
workflow was enabled at reevaluation without training new weights.

**S10 — Voltage bases and physical-unit interpretation.**

- [IEEE-14 physical HIF profile](../ieee14_physical_hif_20260918.md).
- [IEEE-57 declared reconstruction](../ieee57_physical_hif_20260919.md).

The standalone registry exporter uses local physical kV. Main legacy-stack
DAgger corpora retain a normalized 1-kV DSS equivalent and convert physical
ohms through the branch's declared local per-unit base. The deck does not claim
that a normalized-engine ohm equals a physical transmission-system ohm.

## Slide-to-source map

| Slide | Main evidence |
| --- | --- |
| 1 | Original weekly-update template |
| 2 | S1, S3–S5, S7 |
| 3 | S2, S6, S8 |
| 4 | S2, S8 |
| 5 | S1, S10; arithmetic conversion |
| 6–7 | S1, latest 20260923b receipts |
| 8 | S8, S3 |
| 9 | S3 |
| 10 | S4, S5 |
| 11–14 | S6 |
| 15 | S3, S4 |
| 16 | S7 |
| 17 | Proposed follow-on experiments, not completed results |
| 18 | S3–S5, independently summed family totals |
| 19 | S3–S7 metric definitions and limitations |

## Frozen evaluation identity

- Original runtime commit: `1cf17e63c7de161304f87afb6684b351e6488ee4`.
- Revised overlay: `7dafdc9f0f7c91cd0daabbe14b61617b0683ab0447e64ce0638ad9db3164a21c`.
- Frozen 160-root input SHA256: `3a088ebc9dd7b847511d70ecd2fabd945979c66f0303cd5f964001fc958f5262`.
- Revised physical-model SHA256: `e248ac81797ad1b58c06ad1d7a2a73d5098d4768d2043555763c6411b216c667`.
- Revised expert and R2 retained matching roots/seeds and the same frozen R2
  checkpoint where applicable. None of these identities describes the new
  September 23b physical corpus.
