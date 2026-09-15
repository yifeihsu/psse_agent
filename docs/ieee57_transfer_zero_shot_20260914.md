# IEEE 57 transfer: data generation and zero-shot test of the IEEE 14 checkpoint (2026-09-14)

Research-only results; nothing here is release evidence.

## Outcome

The IEEE 14 policy transfers to IEEE 57 without adaptation. The final adapter
of the 2026-09-12 node/breaker run (R2), evaluated unchanged on 68 freshly
generated IEEE 57 development roots under the same protocol, prompt, tools and
40-step budget, scores the same truth-audited task success as the rule-based
expert on the same roots.

| Policy on the 68 IEEE 57 development roots | Task success | Invalid actions | False commits | Loops | Mean steps |
| --- | --- | --- | --- | --- | --- |
| Frozen IEEE 14 R2 adapter (zero-shot) | 67/68 | 5 | 5 | 0 | 12.4 |
| Expert (ceiling) | 67/68 | 0 | 0 | 0 | 11.7 |

The one miss is shared and is the known rule-based ceiling: a misranked
parameter root on which both policies correct the higher-ranked neighbour
(branch row 57) instead of the true row 71. All ten adapter deviations sit on
five mixed roots where it attempted the branch correction before the meter
repair, was refused, rolled back and then recovered with the correct targets.

Cluster cell `research_full_pipeline_20260914_ieee57`, source commit e5429e7 on
`feature/ieee14-full-topology`, overrides
`research/hpc/full_pipeline_20260907/overrides/ieee57_transfer_20260914.env`,
jobs 17810194 (stage 0), 17810195 (round-1 collection), 17810196 (zero-shot).
Compact summary versioned as
[`research/ieee57/zero_shot_transfer_20260914.json`](../research/ieee57/zero_shot_transfer_20260914.json).

## What changed since the 2026-09-12 run

1. **IEEE 57 code merged.** The dw worktree branch
   `codex/ieee57-topology-review-20260912` (balanced corpus builder, logical
   topology testbed, three-phase testbed, audited pilot) was merged as 4c53d2f.
2. **`--system` switch** (74b34e4) on the round-0 aggregate, the research
   runner's scenario generator and the suite builder, with `SYSTEM`,
   `MEASUREMENT_CORPUS`, `BALANCED_ARTIFACT_DIR` and `ADMISSION_MODE` in the
   pipeline cell. A non-IEEE14 system runs the families its registry entry
   supports from a fresh balanced corpus; the IEEE 14 waveform corpora and the
   tracked parameter-case allowlist are refused for it.
3. **Unified WLS detector** (0443aef). Both systems now alarm on the chi-square
   test at alpha 0.01 *or* a maximum absolute normalized residual at 4.0, in
   scenario admission and in the environment. The threshold was chosen on
   fresh clean windows of both networks before any policy result was seen:

   | Residual threshold | Healthy IEEE 57 windows flagged (of 60) | Healthy IEEE 14 windows flagged (of 600) |
   | --- | --- | --- |
   | 3.0 | 38 | 178 |
   | 3.5 | 13 | 36 |
   | 4.0 | 3 | 12 |

   Every meter fault of ten sigma or more was caught at any of the three
   (25/25 on IEEE 57, 344/344 on IEEE 14), while the chi-square test alone
   missed 3 of the 25 IEEE 57 faults because one channel's error is diluted
   over 378 degrees of freedom. Under the combined rule the IEEE 57
   recoverable draw builds 15/15 requested single-meter roots instead of 3/15.
   Runs before 2026-09-14 used the chi-square-only detector; their scores are
   not comparable with this run without a rerun.
4. **Transfer-run machinery** (55617cb, e5429e7): a corpus stage inside stage 0
   (`scripts/build_balanced_corpus.py`), `FROZEN_STUDENT_ADAPTER` linked as the
   run's `bc0` so nothing is trained, `stage_zeroshot.sbatch` with the runner's
   `--eval-student-only` mode, and `CHAIN="d0 r1c zs"` submission.

The IEEE 14 checkpoint itself was not retrained: only the detector changed.

## Configuration

| Item | Value |
| --- | --- |
| System | case57 (57 buses, 80 branches, 491 channels, 113 states, 378 residual degrees of freedom) |
| Families | no_error, measurement, multi_measurement, parameter, measurement+parameter (the IEEE 57 registry's balanced families) |
| Corpus | fresh AC-OPF windows, load scale 0.80 to 1.00, sigma 0.001 pu voltage and 0.01 pu power, 300 clean, 900 single-meter, 900 parameter rows, three scans per parameter window, seed 20260910 |
| Detector | chi-square alpha 0.01 or normalized residual 4.0, in admission and environment |
| Admission | recoverable (teacher-solvable, 1.25 margin on either test); training draw at parameter-ranking threshold 1.2, development draw at 1.0 with rank allowance 2 |
| Meter errors | lifted to at least ten noise sigmas |
| D0 plan | 30 / 60 / 50 / 60 / 60 roots, three counterfactual branches per root, 40-step expert episodes |
| Suites | 52 round-1 training roots, 52 round-2 training roots, 68 development roots (candidate multiplier 3) |
| Student | `google/gemma-4-12B-it` 4-bit plus the 2026-09-12 run's R2 rank-16 LoRA, native prompt profile, no thinking |
| Collection | beta 0.25, 12-step episodes, expert labels every visited state |
| Evaluation | 40-step budget, production case loader, truth audit with the counterfactual completion audit for hand-offs |

## Data generation (stage 0, CPU, 1 h 33 min)

- **Corpus:** 2,100 of 2,100 rows admitted; four parameter attempts were
  physically rejected and retried. About five minutes.
- **Expert aggregate D0:** all 260 requested roots built; 3,190 episode rows
  plus 780 counterfactual recovery rows, none quarantined. Split by root into
  195 train (2,392 rows), 38 validation (468 rows) and 27 test (330 rows).
  State classes: 1,410 clean-successful, 1,083 accepted partial continuation,
  228 accepted final commit, 203 accepted partial commit, 230 terminal
  operator escalation, 30 terminal resolved, 6 rejected-candidate recovery.
- **Suites:** the training draw built 312 candidates for 104 slots and the
  development draw 204 for 68; every plan was filled. Rejections at
  admission were dominated by `anomaly_not_detectable` (314 training, 144
  development attempts), `parameter_context_target_not_dominant` (54, training
  only, the 1.2 threshold) and parameter-correction realizability (about 60).
  Development strata: 19 dominant, 12 ambiguous, 1 misranked, 36 not
  applicable.
- The eleven release failures recorded in the provenance are the expected
  research-mode ones: untracked fresh corpus, relaxed environment pins, and
  the teacher-realizability gates.

## Round-1 collection (GPU, 27 min)

The frozen adapter drove the 52 round-1 training roots with the expert
labelling every visited state.

| Metric | Value |
| --- | --- |
| Episodes | 52/52 complete |
| States visited / with an expert target | 496 / 332 |
| Learner actions executed | 370, no invalid action |
| Learner-expert tool disagreement on comparable states | 17 of 332 (5.1 percent) |
| Label yield | 0.861 |
| Mixture rows (D0 plus D1, 1:1) | 572; 286 safe learner-labelled rows, 46 quarantined |

This mixture is the round-1 training input should adaptation be run; nothing
was trained in this run.

## Zero-shot evaluation (GPU, 50 min)

Truth-audited task success per family and stratum, frozen adapter versus expert:

| Family | Adapter | Expert |
| --- | --- | --- |
| no_error | 8/8 | 8/8 |
| measurement | 16/16 | 16/16 |
| multi_measurement | 12/12 | 12/12 |
| parameter | 15/16 | 15/16 |
| measurement+parameter | 16/16 | 16/16 |

| Parameter stratum | Roots | Adapter | Expert |
| --- | --- | --- | --- |
| dominant (true line first, ratio at least 1.2) | 19 | 19 | 19 |
| ambiguous (true line first, ratio below 1.2) | 12 | 12 | 12 |
| misranked (neighbour outranks the true line) | 1 | 0 | 0 |
| not applicable | 36 | 36 | 36 |

| Overall | Adapter | Expert |
| --- | --- | --- |
| Truth-audited task success | 67/68 | 67/68 |
| Audited lifecycle-clean completion | 62/68 | 67/68 |
| Strict runtime resolution | 8/68 (the clean roots) | 8/68 |
| Post-correction hand-offs credited by the counterfactual audit | 54 | 59 |
| Healthy components preserved | 67/68 | 67/68 |
| False finalizations, false rollbacks, loops, evaluator errors | 0 | 0 |

**The five deviations.** On 5 of the 16 mixed roots the adapter's trace is
`run_wls`, three-phase and harmonic context requests (both unavailable on this
network), `get_parameter_context`, `correct_parameters`, `run_wls`,
`commit_state` (refused: the branch candidate was not verified with the meter
fault still present, counted as one invalid action and one false commit),
`rollback_state`, then the expert's order: meter context and correction,
commit, parameter context and correction, commit, and the hand-off. Those
episodes take 18 steps against the expert's 13; on the other 11 mixed roots
the adapter follows the expert's order exactly. The accepted targets on all
16 mixed roots match the truth, with healthy components preserved. This is
the same branch-first tendency the ambiguity design guards against on IEEE
14, and it is the only behaviour an adaptation round could still improve.

**The shared miss.** The single misranked parameter root: both policies test
and commit the higher-ranked neighbour (row 57), which fits the measurements,
and leave the true row 71 uncorrected. Both are scored as one healthy
component corrupted. This is the rule-based ceiling, not a transfer failure.

**Hand-offs, not autonomous resolutions.** Every fault episode, for both
policies, ends in `operator_escalation` after the corrections, credited as
task success by the counterfactual completion audit; only the eight clean
roots finalize. On this balanced-only network the harmonic and three-phase
acquisitions return unavailable, so the protocol hands off after the
corrections rather than finalizing, as the IEEE 57 pilots reported. The
67/68 must be read as correct corrections plus a clean hand-off, and not as
67 autonomous resolutions.

## Interpretation

- The learned diagnostic procedure is not tied to the 14-bus network: the
  frozen policy names correct meters (indices up to 473 of 491) and branches
  (rows up to 71 of 80) it has never seen, with no invalid action outside the
  five order errors and no loops.
- The gap to the expert is action discipline on mixed roots, not diagnosis.
  Adaptation on the collected mixture could remove it; it cannot raise the
  ceiling, which the expert sets at 67/68 here.
- The transfer review's remaining gaps stand: topology on IEEE 57 exists only
  as the logical-breaker adapter with its own protocol, and the three-phase
  and harmonic families have no IEEE 57 route, so this development set holds
  five families against the twelve of the IEEE 14 suite.
- Sixty-eight roots from one seed is a development reading. The IEEE 14
  baseline (147/160 for R2 and the expert on 2026-09-12) was measured under
  the chi-square-only detector and is not directly comparable.

## Reproduction

```bash
# stage the cell (from the repository root, over WSL ssh; Git Bash path conversion must be off)
MSYS_NO_PATHCONV=1 wsl -- ssh torch bash -s -- BUNDLE feature/ieee14-full-topology COMMIT \
  /scratch/yx3882/research_full_pipeline_20260914_ieee57 ieee57_transfer_20260914.env \
  < research/hpc/full_pipeline_20260907/deploy_remote.sh
# generate the data, collect round 1 with the frozen adapter, evaluate zero-shot
wsl -- ssh torch 'CHAIN="d0 r1c zs" bash /scratch/yx3882/research_full_pipeline_20260914_ieee57/submit_pipeline.sh'
# adaptation on the collected mixture, then the paired evaluation against the frozen student
wsl -- ssh torch 'FROM=r1t CHAIN="r1t r1e" bash /scratch/yx3882/research_full_pipeline_20260914_ieee57/submit_pipeline.sh'
```

Artifacts on the cluster under `out/`: `corpus/`, `d0/`, `suite/`,
`r1/collection/` (mixture, ledger, `zeroshot/{bc0_eval,expert_eval,comparison}.json`)
and `r1/zeroshot_summary.json`. The frozen adapter appears under the round-1
student label `bc0` in the summary.
