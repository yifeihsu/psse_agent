# Diagnostic-family research round (2026-09-03)

Research-only DAgger round on the explanation-only families that the
per-phase branch-current telemetry made identifiable: HIF, measurement+HIF,
three-phase unbalance, and the balanced telemetry control. It runs
`scripts/run_dagger_research.py --plan-preset diagnostic`, the entry point
wired for these families (branch-current corpora, honest unbalance sensor
signatures, research OpenDSS budget), and is not release evidence.

| Item | Value |
| --- | --- |
| Student, paired baseline, warm start | BC0 12B `checkpoint-192` on `google/gemma-4-12B-it@707f0a3b` |
| Training roots | 12 HIF, 6 measurement+HIF, 12 unbalance, 6 control |
| Development roots | 6 / 3 / 6 / 3, generated together and held out by physical root |
| Collection | beta 0.25, 12-step episodes, seed 20260903 |
| Mixture | D1 capped at 200 rows, exact 1:1 with D0 |
| Training | one pass, micro-batch 1 x 4 accumulation, lr 3e-5, final adapter kept |
| Evaluation | BC0 versus candidate on the same 18 development roots, 24-step budget |
| Environment | `gemma4_research_5104` (matches `psse_env/requirements-sft-research.txt`); the CPU test files run under the exposure-cell overlay interpreter because the research environment has no pytest |
| Scheduler | `a100\|h100\|h200\|rtx6000` union, preemption opt-in, requeue |

Protected roots: the 27-row trace validation, the published 15-root D1
development suite, the frozen BC0 evaluation suite, and every D0 root.

## Deploy

Local (Git Bash on the workstation, from the `local/relaxed-current`
worktree):

```bash
git bundle create dw_round.bundle local/relaxed-current
wsl -- scp dw_round.bundle torch:/scratch/yx3882/research_diag_round_20260903/dw_round.bundle
wsl -- ssh torch bash -s -- /scratch/yx3882/research_diag_round_20260903/dw_round.bundle local/relaxed-current "$(git rev-parse HEAD)" \
  < research/hpc/diagnostic_round_20260903/deploy_remote.sh
```

The remote step clones the bundle to `source/`, stages the scripts beside it,
syntax-checks them, and runs the dry-run prerequisites (input digests,
adapter, offline snapshot, environment pins).

## Submit and watch

```bash
wsl -- ssh torch bash /scratch/yx3882/research_diag_round_20260903/submit_diag.sh
wsl -- ssh torch bash /scratch/yx3882/research_diag_round_20260903/status_diag.sh
```

Stages are output-guarded and the collector keeps a completed-root ledger, so
a preempted job resumes on requeue. Receipts land in `out/`:
`prerequisites.json`, `collection.done`, `training.done`, and
`round_summary.json` (overall suite metrics for both adapters, their
difference, and per-family outcome counts on the development roots).

## Stale D0 HIF rows

The round-0 aggregate predates branch-current telemetry, so all 91 of its
HIF and measurement+HIF rows (49 + 42 of 1280) teach an operator handoff
after both estimators are exhausted, the opposite of what the diagnostic D1
rows teach for the same signature. The training stage therefore rebuilds the
1:1 mixture from the D0 pool with those two families removed
(`filter_mixture.py`, asserting exactly 91 dropped rows) and trains on
`round1.train.filtered.jsonl`. The collection stage's unfiltered
`round1.train.jsonl` is left in place as the recorded mixture identity. If
the training and evaluation jobs were already queued when the filter was
added, `amend_train_chain.sh COLLECT_JOB_ID` cancels them and resubmits the
staged scripts behind the running collection.

## First run (2026-09-04) and the repeat

The first chain (jobs 16920762, 16923642, 16923643) completed with the
stale-HIF filter in place. On the 18 development roots the candidate fixed
the HIF ladder (6 of 6, three steps per episode, versus 5 of 6 at 5.2 steps
for BC0) but regressed on unbalance (4 of 6 versus 6 of 6): after a failed
escalation on an explained unbalance root, the recovery expert's generic WLS
fallback had become a teacher target, the post-explanation WLS minted
residual signatures from the still-unbalanced operator vector, and the
classical route chased them into false commits. Measurement+HIF finalized
after the HIF explanation for both adapters; by decision that is the intended
terminal for that family, and the strict audit keeps counting the remaining
meter as unresolved, so read that family's row as "HIF identified".

The environment and expert fixes (waveform signatures block the
fundamental-frequency routes whether or not they are explained; the recovery
expert defers to the diagnostic ladder; the classical experts stand down) are
in `psse_env`. To repeat the round on the fixed source without touching the
first run's receipts, deploy into a fresh directory:

```bash
git bundle create dw_round.bundle local/relaxed-current
wsl -- scp dw_round.bundle torch:/scratch/yx3882/research_diag_round_20260904/dw_round.bundle
wsl -- ssh torch bash -s -- /scratch/yx3882/research_diag_round_20260904/dw_round.bundle local/relaxed-current "$(git rev-parse HEAD)" /scratch/yx3882/research_diag_round_20260904 \
  < research/hpc/diagnostic_round_20260903/deploy_remote.sh
wsl -- ssh torch bash /scratch/yx3882/research_diag_round_20260904/submit_diag.sh
```

The same seed reproduces the same 36 training and 18 development roots, so
the repeat is a paired comparison against the first run.

## Discovery mode

The first two runs seeded the unbalance sensor flag at reset. From commit
`3b4ea03`'s successor onward the research script defaults
`--unbalance-signature-mode discovered`: the operator starts from the
positive-sequence snapshot and the balanced model, the baseline WLS mints the
fundamental anomaly, and the expert screens the three-phase telemetry before
any correction route (see `psse_env/README.md`). HIF keeps its zero-sequence
relay flag (`--hif-signature-mode flagged`). Signatures do not enter the
physical root fingerprint, so a discovery round on the same seed is still a
paired comparison with the flagged runs; the modes are recorded in the run
config's `research_profile.scenario_sources.signature_modes`.

## Round 4: measurement requests and the HSE flow

From commit `48afc0f`'s successor the operator acquires additional
measurements explicitly. After the baseline WLS the residual breadth picks
the first request: a broad anomaly (spectral distortion elevates most of the
122 channels) asks for spectra through `get_harmonic_context`, a narrow one
asks for phase measurements through `get_three_phase_context`, and either
falls back to the other when it returns nothing. Answered-unavailable
requests stay answered for the whole episode. The resulting expert flows are
`run_wls -> get_three_phase_context -> run_three_phase_nlm_from_path ->
finalize_diagnosis` for unbalance and `run_wls -> get_harmonic_context ->
run_hse_from_path -> finalize_diagnosis` for harmonics, which the
`diagnostic` preset now includes (12 training and 6 development harmonic
roots, discovered). Deploy into a fresh round directory as above.

**Round 4 result** (`/scratch/yx3882/research_diag_round_20260905_requests/`,
collect 17015054 on `9ca833c`, train 17022616 and eval 17022617 on
`aad2b0c`; the first train attempt failed on the stale-registry gate that
`aad2b0c` removed). Collection: 48 of 48 roots after two preemptions, label
yield 0.77, 88 safe rows (harmonic 30, unbalance 24, HIF 23, control 6,
measurement+HIF 5), mixture 176 rows, 44 optimizer steps. On the 24
development roots the candidate scores 17 of 24 truth-audited (0.708)
against BC0's 7 of 24 (0.292): harmonic 6 of 6 in four steps
(`run_wls -> get_harmonic_context -> run_hse_from_path -> finalize`),
unbalance 6 of 6 in five steps (the candidate asks for spectra first, gets
nothing, then asks for phase measurements; the expert would have asked for
phase measurements first, so that step is a correction target for a later
round), control 3 of 3, measurement+HIF 0 of 3 by decision with the fault
identified 3 of 3, and HIF 2 of 6 with the branch identified 6 of 6. BC0
escalates on 11 of the 12 harmonic and unbalance roots after three steps.

Two caveats. First, **round 4 is not root-paired with rounds 1 to 3**: the
generator draws every family from one random stream in `SCENARIO_FAMILIES`
order, and `harmonic` precedes `hif` in that order, so adding harmonic to the
plan changed every later family's draws; none of the 24 development roots
appears in the earlier rounds. Round 4's roots are the reference for further
rounds with this family set. Second, the four HIF misses are the estimator,
not the policy: on every HIF root both adapters pick the true branch and the
multiscan estimate's alpha error is 0.055 to 0.10 against the audit's 0.05
tolerance (`ReleaseAuditTolerances.hif_alpha_abs`), while the estimator's own
validation gates are a median error of 0.05 and a 90th percentile of 0.10.
Rounds 1 to 3 drew six HIF roots whose errors all fell under 0.036. Read the
HIF row the way the measurement+HIF row is read: branch identified.

## What this round cannot claim

These families terminate through an accepted anomaly explanation or an
operator handoff, never a repair, so the round supervises the diagnostic
ladder rather than physical recovery. The training-loss monitor is the
classical-family validation set; the diagnostic outcome is the paired
closed-loop evaluation only.
