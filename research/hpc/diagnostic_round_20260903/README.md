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

## What this round cannot claim

These families terminate through an accepted anomaly explanation or an
operator handoff, never a repair, so the round supervises the diagnostic
ladder rather than physical recovery. The training-loss monitor is the
classical-family validation set; the diagnostic outcome is the paired
closed-loop evaluation only.
