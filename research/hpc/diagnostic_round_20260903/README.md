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

## What this round cannot claim

These families terminate through an accepted anomaly explanation or an
operator handoff, never a repair, so the round supervises the diagnostic
ladder rather than physical recovery. The training-loss monitor is the
classical-family validation set; the diagnostic outcome is the paired
closed-loop evaluation only.
