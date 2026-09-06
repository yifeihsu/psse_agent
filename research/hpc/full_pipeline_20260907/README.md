# Full research pipeline (2026-09-07)

Expanded IEEE-14 dataset over every error family designed so far and their
existing combinations, a BC0 student trained from the base model on an
expert aggregate, then two DAgger rounds, all as one Slurm chain. Research
only; nothing here is release evidence.

| Item | Value |
| --- | --- |
| Families | no_error, measurement, multi_measurement, parameter, topology, harmonic, hif, measurement+parameter, measurement+topology, measurement+hif, three_phase_unbalance, telemetry_no_disturbance |
| Expert aggregate (D0) | 438 roots (plan in `pipeline.env`), 24-step expert episodes, three counterfactual recovery branches per root, truth-audited; split 75/15/10 by root into train, validation, test |
| DAgger suite | 122 training roots per round, two rounds on disjoint roots, 160 shared development roots; drawn after D0 and excluding every D0 and protected root |
| Student | `google/gemma-4-12B-it` 4-bit base plus rank-16 LoRA |
| BC0 | one pass over the D0 train view at lr 1e-4, micro-batch 1 x 4 accumulation, best validation-loss checkpoint on a 128-row family-stratified subset of the D0 validation split |
| Rounds | beta 0.25 collection in 12-step episodes; the safe rows mixed 1:1 with the pool (round 1: D0; round 2: D0 plus round 1's rows); one pass at lr 3e-5 continuing the student; final adapter |
| Evaluation | student versus candidate on the same 160 development roots, 24-step budget; BC0 vs R1 in round 1, R1 vs R2 in round 2 |
| Signatures | harmonic and unbalance discovered from the WLS anomaly; HIF keeps its relay flag; research HIF budget 7 x 9 x 10 |
| Scheduler | stage 0 on the `cs` CPU partition (16 CPUs); GPU stages on the `a100\|h100\|h200\|rtx6000` union with preemption opt-in and requeue |

Capacity that bounds the plans: 102 HIF windows serve `hif` and
`measurement+hif` separately, 220 unbalance rows serve `three_phase_unbalance`
and the balanced control separately, and the train partition of the tabular
corpus (four fifths of 304 single-meter, 226 multi-meter, 620 parameter, and
500 harmonic rows) serves the rest. Total roots: 438 + 244 + 160 = 842.

## Stages

| Stage | Job | Partition | Reads | Writes |
| --- | --- | --- | --- | --- |
| 0 | `fp-d0` | cs | corpora | `out/d0/` (aggregate raw, train view, validation, test), `out/suite/` (`r1_training.json`, `r2_training.json`, `development.json`, `manifest.json`), `out/d0.done`, `out/suite.done` |
| 1 | `fp-bc0` | GPU | D0 train view, validation subset | `out/bc0/lora`, `out/bc0.done` |
| 2 | `fp-r1c` | GPU | BC0, `r1_training`, `development`, D0 pool | `out/r1/collection/` (D1 rows, mixture, ledger), `out/r1/collection.done` |
| 3 | `fp-r1t` | GPU | mixture, BC0 | `out/r1/training/lora`, `out/r1/training.done` |
| 4 | `fp-r1e` | GPU | BC0, R1, development | `out/r1/collection/evaluation/`, `out/r1/round_summary.json` |
| 5 | `fp-r2c` | GPU | R1, `r2_training`, pool D0 + D1 | `out/r2/pool/`, `out/r2/collection/`, `out/r2/collection.done` |
| 6 | `fp-r2t` | GPU | mixture, R1 | `out/r2/training/lora`, `out/r2/training.done` |
| 7 | `fp-r2e` | GPU | R1, R2, development | `out/r2/round_summary.json`, `out/pipeline_summary.json` |

Every stage is output-guarded and the collector keeps a completed-root
ledger, so a preempted job resumes on requeue and a resubmitted chain skips
finished stages (`FROM=<stage> submit_pipeline.sh` starts later). The expert
aggregate itself is not resumable and restarts if stage 0 is requeued.

The round-0 generator is the release aggregate builder run with `--research`:
the release verdict and its reasons are recorded in the generation
provenance instead of failing the stage (the research environment does not
carry the release numerical pins, and `PSSE_LOCAL_DIAGNOSTIC_BUILD=1` records
that deviation). Composed measurement+HIF episodes that finalize after the
fault explanation are quarantined out of D0 by the truth audit, as in the
diagnostic rounds; that family contributes its estimator steps only.

## Deploy, submit, watch

From the `local/relaxed-current` worktree:

```bash
git bundle create dw_pipeline.bundle local/relaxed-current
wsl -- scp dw_pipeline.bundle torch:/scratch/yx3882/research_full_pipeline_20260907/dw_pipeline.bundle
wsl -- ssh torch bash -s -- /scratch/yx3882/research_full_pipeline_20260907/dw_pipeline.bundle local/relaxed-current "$(git rev-parse HEAD)" \
  < research/hpc/full_pipeline_20260907/deploy_remote.sh
wsl -- ssh torch bash /scratch/yx3882/research_full_pipeline_20260907/submit_pipeline.sh
wsl -- ssh torch bash /scratch/yx3882/research_full_pipeline_20260907/status_pipeline.sh
```

`deploy_remote.sh BUNDLE BRANCH COMMIT [PIPE_DIR] [OVERRIDES]` stages the cell
into a fresh directory and optionally a file from `overrides/` as
`pipeline.overrides.env`, which `pipeline.env` sources last.

## Reading the results

`out/pipeline_summary.json` carries per-family truth-audited success for
BC0, R1, and R2 on the same development roots, the overall rates, and a
check that R1 scores identically as candidate (round 1) and as student
(round 2). Each `out/<round>/round_summary.json` carries the paired
comparison, the collection metrics, and the mixture report for that round.
Measurement+HIF is scored as fault-identified rather than resolved, so the
ceiling on the 160 development roots is 152.
