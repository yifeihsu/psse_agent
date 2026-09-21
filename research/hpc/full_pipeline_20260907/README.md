# Full research pipeline (2026-09-07; re-runs 2026-09-08, 2026-09-09 and 2026-09-12)

Expanded IEEE-14 dataset over every error family designed so far and their
existing combinations, a BC0 student trained from the base model on an
expert aggregate, then two DAgger rounds, all as one Slurm chain. Research
only; nothing here is release evidence.

The 2026-09-08 re-run kept the first run's expert aggregate and BC0 (linked
from `PREVIOUS_PIPE`) and redrew the suite under the ambiguity design
below. The 2026-09-09 run regenerates everything from a larger 548-root
expert aggregate with every injected meter error lifted to at least ten
noise sigmas (`MEASUREMENT_ERROR_MIN_SIGMA`) and the HIF localization
tolerance widened to 0.15 of line length.
The 2026-09-12 run keeps every setting of the 2026-09-09 run and changes
only the topology family: its roots are breaker-status errors in the full
IEEE-14 node/breaker model with substation telemetry (`TOPOLOGY_EFFECTS`,
isolated line terminals and bus splits), the agent requests the substation
measurements and identifies the breaker with the node/breaker
normalized-multiplier estimator, and the correction names the breaker
(`docs/ieee14_node_breaker_nlm_20260912.md`). Everything is regenerated
because the expert aggregate contains topology roots. Cluster directory
`research_full_pipeline_20260912`, seeds unchanged.
Outcome: R2 and the expert at 147/160, the same 147 roots and the same
thirteen misses as the 2026-09-09 run; topology 16/16 and
measurement+topology 12/12 for R2 and the expert, every accepted breaker
the true one (`docs/node_breaker_pipeline_run_20260912.md`; three stage-0
attempts failed on integration gaps fixed in 5955cfe, bd59df4, ae99f03).

| Item | Value |
| --- | --- |
| Families | no_error, measurement, multi_measurement, parameter, topology, harmonic, hif, measurement+parameter, measurement+topology, measurement+hif, three_phase_unbalance, telemetry_no_disturbance |
| Expert aggregate (D0) | 548 roots planned (plan in `pipeline.env`; 438 in the first run), expert episodes, three counterfactual recovery branches per root, truth-audited; split 75/15/10 by root into train, validation, test; parameter roots at the production ranking threshold 1.2 |
| Meter errors | every injected meter error (corpus rows and composed overlays) lifted to at least 10 noise sigmas; the tracked corpus centres near 10 sigma and reaches down to 5 |
| DAgger suite | 122 training roots per round, two rounds on disjoint roots, 160 shared development roots; drawn after D0 and excluding every D0 and protected root |
| Training draw | parameter-ranking threshold 1.2 (teacher corrects the true line on the first pass) |
| Development draw | detection threshold 1.0 with the true line admitted anywhere in the top 2 of the deployed ranking; each root records its stratum |
| Student | `google/gemma-4-12B-it` 4-bit base plus rank-16 LoRA |
| BC0 | one pass over the D0 train view at lr 1e-4, micro-batch 1 x 4 accumulation, best validation-loss checkpoint on a 128-row family-stratified subset of the D0 validation split |
| Rounds | beta 0.25 collection in 12-step episodes; the safe rows mixed 1:1 with the pool (round 1: D0; round 2: D0 plus round 1's rows); one pass at lr 3e-5 continuing the student; final adapter |
| Evaluation | student versus candidate on the same 160 development roots, 40-step budget, production case loader; BC0 vs R1 in round 1 (plus the expert itself), R1 vs R2 in round 2 |
| Signatures | harmonic and unbalance discovered from the WLS anomaly; HIF keeps its relay flag; research HIF budget 7 x 9 x 10 |
| Scheduler | stage 0 on the `cs` CPU partition (16 CPUs); GPU stages on the `a100\|h100\|h200\|rtx6000` union with preemption opt-in and requeue |

`SYSTEM` (pipeline.env) selects the registered system for stage 0. `case14`
is everything above. `case57` runs only the five balanced families of the
system registry (no_error, measurement, multi_measurement, parameter,
measurement+parameter) from a fresh balanced corpus named by
`MEASUREMENT_CORPUS` and `BALANCED_ARTIFACT_DIR`, with every plan restricted
to those families and no waveform corpora; `ADMISSION_MODE` picks the
generator admission for both the aggregate and the suite draws. The later
stages read the suites the same way for either system.

**WLS anomaly detector (both systems, from 2026-09-14).** Scenario admission
in stage 0 and the collection/evaluation environment use one rule: the
chi-square test at alpha 0.01 *or* a maximum absolute normalized residual at
`NORMALIZED_RESIDUAL_THRESHOLD` (4.0). An anomaly is present when either
fires; a corrected configuration is clean only when neither does. The
threshold was chosen on fresh clean windows of both networks: 3 sigma flagged
38 of 60 healthy IEEE 57 windows and 178 of 600 healthy IEEE 14 windows, 3.5
sigma 13 of 60 and 36 of 600, 4 sigma 3 of 60 and 12 of 600, while every
meter fault of ten sigma or more was caught at any of the three (the
chi-square test alone missed 3 of 25 such faults on IEEE 57 because one
channel's error is diluted over 378 degrees of freedom). The runs above,
through 2026-09-12, used the chi-square-only detector
(`--chi-square-only` reproduces it); their scores are not comparable with
runs under the combined rule without a rerun. The IEEE 57 pilot's own
runtime pin (alpha 0.05 with the same residual test) stays with the pilot
scripts.

## IEEE 57 transfer run (2026-09-14)

`overrides/ieee57_transfer_20260914.env` generates the balanced IEEE 57 data
and reads the frozen IEEE-14 checkpoint on it, without retraining anything on
IEEE 14 (only the detector changed since the 2026-09-12 run):

1. **Stage 0 (CPU)** builds a fresh balanced corpus under `out/corpus`
   (`CORPUS_COUNTS`, `scripts/build_balanced_corpus.py`), the expert aggregate
   D0 and the suites for the five balanced families, then links
   `FROZEN_STUDENT_ADAPTER` (the 2026-09-12 run's R2 adapter) as this run's
   `bc0` with a `bc0.done` receipt, so stage 1 has nothing to train.
2. **Round-1 collection (GPU)** rolls the frozen adapter out on the IEEE 57
   round-1 training roots with the expert labelling (the usual beta 0.25
   collection), which is also the D1 a later adaptation round trains on.
3. **Zero-shot evaluation (GPU, `stage_zeroshot.sbatch`)** evaluates the frozen
   adapter (`--eval-student-only`) and the expert on the IEEE 57 development
   roots into `out/r1/collection/zeroshot/` and summarizes them per family
   and stratum in `out/r1/zeroshot_summary.json` (the student is labelled
   `bc0` there, as the round-1 student).

Deploy to its own cell directory and submit the three-stage chain:

```bash
git bundle create ieee57.bundle <deployed_commit>..feature/ieee14-full-topology
wsl -- ssh -o BatchMode=yes torch "cat > /scratch/yx3882/research_full_pipeline_20260914_ieee57/ieee57.bundle" < ieee57.bundle
wsl -- ssh torch bash -s -- /scratch/yx3882/research_full_pipeline_20260914_ieee57/ieee57.bundle feature/ieee14-full-topology "$(git rev-parse HEAD)" \
  /scratch/yx3882/research_full_pipeline_20260914_ieee57 ieee57_transfer_20260914.env < research/hpc/full_pipeline_20260907/deploy_remote.sh
wsl -- ssh torch 'CHAIN="d0 r1c zs" bash /scratch/yx3882/research_full_pipeline_20260914_ieee57/submit_pipeline.sh'
```

Outcome: the frozen adapter scores 67/68 truth-audited task success, the
same as the expert, with five order errors on mixed roots
(`docs/ieee57_transfer_zero_shot_20260914.md`).

Adaptation on IEEE 57 is then `FROM=r1t CHAIN="r1t r1e" bash submit_pipeline.sh`
(round-1 training on the collected mixture and the paired evaluation against
the frozen student). The three-phase, harmonic and topology families have no
IEEE 57 route yet, so the plans name only the five balanced families.

Capacity that bounds the plans: 102 HIF windows serve `hif` and
`measurement+hif` separately, 220 unbalance rows serve `three_phase_unbalance`
and the balanced control separately, and the train partition of the tabular
corpus (four fifths of 304 single-meter, 226 multi-meter, 620 parameter, and
500 harmonic rows) serves the rest. Total roots: 548 + 244 + 160 = 952.

## Ambiguity design (ranked testing with escalation fallback)

A parameter fault on one line often raises the Lagrange multiplier of an
adjacent line almost as much. The deployed context ranks lines by
multiplier; when the top line does not clear the runner-up by the dominance
ratio (1.2) the ranking is *ambiguous*. Such roots exist in real networks,
so the development set keeps them and the students are read against the
teacher on them:

- **Context.** Under ambiguity `get_parameter_context` offers the top two
  ranked lines in rank order (`parameter_ranking_ambiguous`,
  `parameter_ranking_candidate_lines`) instead of standing down.
- **Teacher.** The expert tests the candidates in rank order under
  verification. Once every candidate has a verification-rejected
  hypothesis at the active state it escalates with
  `operator_escalation:ambiguous_branch_candidates`, naming the candidate
  lines, rather than the generic exhaustion request.
- **Guard.** While the anomaly is branch-dominant the meter route is
  blocked (`measurement_route_blocked_by_branch_dominance`) until both
  branch families have a rejected hypothesis, so a student cannot mask a
  branch fault with a meter edit; a context fetched while three-phase
  screening is pending no longer counts as a completed screen.
- **Audit.** A bounded handoff is credited as success when the true branch
  is among at most two named candidates and no healthy component was
  modified (`bounded_localization_handoff`); everything else is scored as
  before.
- **Labels.** A non-correction preferred action that is the ladder's first
  proposal is a deterministic research label; the rank-one proof no longer
  quarantines context fetches at states with two admissible proposals.
- **Reporting.** `summarize.py` splits every family by stratum and carries
  the expert's own score on the development roots as the ceiling. Strata:
  `dominant` (true line first, ratio >= 1.2), `ambiguous` (true line first,
  ratio < 1.2: the candidates are tested in rank order), `misranked` (a
  neighbour outranks the true line: the teacher tests the neighbour first
  and, when that correction fits the measurements too, commits it and is
  wrong; this is the rule-based ceiling), `not_applicable`.

## Stages

| Stage | Job | Partition | Reads | Writes |
| --- | --- | --- | --- | --- |
| 0 | `fp-d0` | cs | corpora, or `PREVIOUS_PIPE/out/d0` | `out/d0/` (aggregate raw, train view, validation, test), `out/suite/` (`r1_training.json`, `r2_training.json`, `development.json`, `manifest.json` with per-root strata), `out/d0.done`, `out/suite.done` |
| 1 | `fp-bc0` | GPU | D0 train view, validation subset, or `PREVIOUS_PIPE/out/bc0` | `out/bc0/lora`, `out/bc0.done` |
| 2 | `fp-r1c` | GPU | BC0, `r1_training`, `development`, D0 pool | `out/r1/collection/` (D1 rows, mixture, ledger), `out/r1/collection.done` |
| 3 | `fp-r1t` | GPU | mixture, BC0 | `out/r1/training/lora`, `out/r1/training.done` |
| 4 | `fp-r1e` | GPU | BC0, R1, expert, development | `out/r1/collection/evaluation/` (`bc0_eval`, `r1_eval`, `expert_eval`), `out/r1/round_summary.json` |
| 5 | `fp-r2c` | GPU | R1, `r2_training`, pool D0 + D1 | `out/r2/pool/`, `out/r2/collection/`, `out/r2/collection.done` |
| 6 | `fp-r2t` | GPU | mixture, R1 | `out/r2/training/lora`, `out/r2/training.done` |
| 7 | `fp-r2e` | GPU | R1, R2, development | `out/r2/round_summary.json`, `out/pipeline_summary.json` |

Every stage is output-guarded and the collector keeps a completed-root
ledger, so a preempted job resumes on requeue and a resubmitted chain skips
finished stages (`FROM=<stage> submit_pipeline.sh` starts later). The expert
aggregate itself is not resumable and restarts if stage 0 is requeued.
`PREVIOUS_PIPE` (empty to disable) links a finished D0 or BC0 from an
earlier run in place of regenerating it.

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
wsl -- scp dw_pipeline.bundle torch:/scratch/yx3882/research_full_pipeline_20260912/dw_pipeline.bundle
wsl -- ssh torch bash -s -- /scratch/yx3882/research_full_pipeline_20260912/dw_pipeline.bundle local/relaxed-current "$(git rev-parse HEAD)" \
  < research/hpc/full_pipeline_20260907/deploy_remote.sh
wsl -- ssh torch bash /scratch/yx3882/research_full_pipeline_20260912/submit_pipeline.sh
wsl -- ssh torch bash /scratch/yx3882/research_full_pipeline_20260912/status_pipeline.sh
```

`deploy_remote.sh BUNDLE BRANCH COMMIT [PIPE_DIR] [OVERRIDES]` stages the cell
into a fresh directory and optionally a file from `overrides/` as
`pipeline.overrides.env`, which `pipeline.env` sources last.

## Reading the results

`out/pipeline_summary.json` carries per-family truth-audited success for
BC0, R1, R2, and the expert on the same development roots, the same split
by stratum (`success_by_stratum`, `success_by_family_and_stratum`), the
overall rates, and a check that R1 scores identically as candidate (round
1) and as student (round 2). Each `out/<round>/round_summary.json` carries
the paired comparison, the collection metrics, the mixture report, and the
success basis per family (first-pass correction versus bounded handoff).
Measurement+HIF is scored as fault-identified rather than resolved, so the
ceiling on the 160 development roots is 152; on ambiguous roots the
expert's own row is the ceiling.

### Physical-ohm HIF sources (2026-09-20)

D0 collection and suite generation receive the same explicit train/validation HIF sources from `pipeline.env`. These use 100-200, 200-500 and 500-1000 ohm bands at 69 kV, with the `ieee14_nominal_69_13p8_18kv_v1` voltage profile and `ybus` injection convention. Historical sources remain available through their existing constants; choose `hif_resistance_search="legacy_pu"` for the old search box.
