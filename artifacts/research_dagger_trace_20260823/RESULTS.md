# Research learner-trace DAgger prototype result

Date: 2026-08-23 (America/New_York)

## Outcome

This is a valid negative preliminary result. The 32-step Gemma-4-E2B learner-trace continuation did not improve the primary closed-loop recovery outcome over BC0 on the matched recovery-stress closure.

| Metric | BC0 | Learner-trace DAgger | DAgger minus BC0 |
|---|---:|---:|---:|
| Exact first recovery action | 0/14 (0.0%) | 0/14 (0.0%) | 0.0 pp |
| Exact and successfully executed first recovery action | 0/14 (0.0%) | 0/14 (0.0%) | 0.0 pp |
| Physically resolved episodes | 0/14 (0.0%) | 0/14 (0.0%) | 0.0 pp |
| Terminal episodes | 0/14 (0.0%) | 0/14 (0.0%) | 0.0 pp |
| Loop episodes | 14/14 (100.0%) | 14/14 (100.0%) | 0.0 pp |
| Broad invalid-action count | 146 | 136 | -10 |
| Successfully executable action rate | 22/168 (13.095%) | 21/157 (13.376%) | +0.281 pp |
| Normalized non-sentinel action rate | 165/168 (98.214%) | 153/157 (97.452%) | -0.762 pp |
| Mean policy steps | 12.000 | 11.214 | -0.786 |
| Control or audit quarantined episodes | 0 | 2 | +2 |
| Evaluator-error episodes | 0 | 0 | 0 |
| False commit/finalize/rollback counts | 0/0/0 | 0/0/0 | 0/0/0 |

All seven recovery suites were 0/2 for exact first recovery action and 0/2 for physical resolution under both models. Every paired episode was a tie on those two core outcomes.

Zero recorded healthy-component corruption is not affirmative safety evidence here: healthy-component preservation was unknown in all 14 episodes for both models because neither model reached an auditable terminal recovery.

## Failure pattern

The first-action results show tool collapse rather than an evaluator or adapter-loading failure.

Expected expert tools across the 14 post-intervention opportunities were:

- `run_wls`: 6
- `correct_parameters`: 2
- `rollback_state`: 2
- `get_measurement_context`: 2
- `get_topology_context`: 2

The learner-trace adapter instead selected:

- `get_measurement_context`: 10
- `correct_measurements`: 2
- `get_parameter_context`: 2

The adapter weights differ from both its initialization and BC0, the saved safetensors file is loadable, and both evaluation artifacts pass their content-address checks. The zero result is therefore a genuine generalization failure of this short update.

## Data and run scope

- Collection closure: 56 episodes, 16 physical roots, seven suites.
- Training view: 555 rows = 185 sampled D0 rows + 185 unique learner-state labels repeated twice.
- Validation view: 27 learner-state labels on two roots excluded from training.
- Held-out closure: 14 intervention episodes, seven suites x two episodes, four independent physical roots, zero overlap with collection/training/validation roots.
- Model: Gemma-4-E2B LoRA, 32 optimizer steps, maximum sequence length 16,384; no examples were dropped or prompt-trimmed.
- Final reported training loss was 0.01418 while validation loss remained 0.99887. Together with 0/14 exact held-out recovery actions, this is direct evidence of memorization without useful root-level generalization.
- Training job `16257850` completed all 32 steps and saved the adapter, then stopped on a launcher-only `trainer_state.json` path error.
- Continuation job `16258358` used the fixed completion check, skipped collection/export/training, ran both evaluations, and completed with exit code 0.

The DAgger adapter was continued from an earlier preliminary DAgger adapter, not directly from BC0. This is therefore a cumulative DAgger-vs-BC0 descriptive comparison; it does not isolate the incremental causal effect of only the final 32 steps.

## Interpretation and next experiment

This run does not demonstrate DAgger benefit. The small reduction in broad invalid actions is secondary and does not compensate for zero exact recovery actions, zero resolution, universal loops, and two quarantined episodes.

Do not repeat this configuration unchanged. The efficient next experiment should reuse the already-created canonical D0/natural-D1/probe material, rebalance explicit recovery-tool targets, and require a cheap preflight to pass before another closed-loop GPU evaluation: (1) exact action accuracy on the root-disjoint validation labels, (2) per-tool recall for `run_wls`, `rollback_state`, `correct_parameters`, and `get_topology_context`, and (3) a tiny overfit/memorization canary proving that the training/rendering path can change greedy actions. Only a checkpoint that clears that preflight should consume another paired closed-loop run.

## Artifact integrity

SHA-256 values below are the ordinary file digests and match the remote copies:

- `trace_view_report.json`: `eaeaf218e19e96fa87abe4934b758d2f4c58546dcd34db3841847df54e7421e1`
- `dagger_trace_eval_14.json`: `b1792a5693dd3fb9147835c239b030c9f1fb8a61bff519c7745ba9321e317cd2`
- `bc0_eval_14.json`: `b9762cb8741757febcc47933d002584404b72f99b7dd356b5c7fd9be3b36d2ca`
- `comparison_trace_vs_bc0.json`: `cd3d7e7be3a3ddfe04f784f304d6bf9d0e13d8516dd5681c2224664c4f217199`
