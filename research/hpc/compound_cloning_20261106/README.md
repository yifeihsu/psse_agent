# Cloning-pass cell (2026-11-06)

One supervised pass of the student (Gemma 4 12B + LoRA) over the selected chat rows, continuing R1
of the conventional cell, then the start adapter and the cloned adapter on the evaluation suites in
the environment the rows were made in. Spec:
`output/compound_rl_20261106/cloning_selection/cloning_pass_training_spec_20261106.md` (main
checkout). Work directory on torch: `/scratch/yx3882/compound_cloning_20261106` (`$WORK`). Nothing
here has been submitted; a run needs the user's go-ahead, and the options branch committed.

The first pass is IEEE 14 rows only, 8,866 rows and 2,217 optimizer steps. The cell holds no row
or step count: the stage reads `ceil(rows / 4)` from the row file at run time.

## Decision: which tool list the rows are rendered with

The row files carry the run-time tool list, but `research-train` does not render them as they are: it
replaces every row's tools with the whole sanitized registry before its row gate
(`psse_env/sft/research_cli.py:208-213`, `psse_env/sft/research_rows.py:58,66`), while the policy
renders `tool_schemas_for_observation(registry, observation)`
(`psse_env/dagger/research_policy_factory.py:686`). On the current view:

| | Tools | Tool JSON (characters) | `rollback_state` description |
| --- | --- | --- | --- |
| What stock `research-train` renders | 20 | 13,841 | "Roll back a verified rejected or inconclusive candidate case." |
| What the policy renders at run time | 13 | 6,973 | says an acceptable candidate may be declined |

So a stock pass trains the decline without the description the student sees when it acts, and with
seven tools it can never call. `CLONE_TOOLS=runtime` in `cell.env` (the default here) runs the same
trainer through `train_runtime_tools.py`, which gives each row the list the policy would render for
that row's observation (checked on all 15,152 rows of the current views: same 13 names as the row, the
row gate's message check passes; the policy's copy also has the sanitizer's filled-in schema descriptions,
which the rows' copy lacks). `CLONE_TOOLS=registry` runs the stock command as the spec wrote it.
Earlier cells trained under the stock rendering, R1 included. Choose before the long job, and do not
switch inside a pass.

## Files

| File | What it does |
| --- | --- |
| `cell.env` | Paths, settings, the job environment, the link guard and the checkpoint pruning; sourced by everything |
| `eval_suites.env` | The configurable evaluation list: name, suite file, seed, links. Default: IEEE 14 development suite and conventional test |
| `stage_clone_train.sbatch` | The pass, or with `CLONE_MODE=smoke` the same path on about 200 rows for 2 steps |
| `train_runtime_tools.py` | `research-train` with the run-time tool list (section above) |
| `stage_eval.sbatch` | One adapter on one suite through `evaluate_suite.py`, unchanged, sharded and resumable |
| `env_report.py` | Writes the report `evaluate_suite.py` reads for the suite's system, and refuses an environment that differs from the rows' |
| `episode_outcomes.py` | Per-episode records with `decline_accounting`, class and targets; the spec's measures overall, per class, breaker classes apart |
| `smoke_inputs.py` | Cuts the smoke rows and suite slices; times the row gate |
| `submit_cell.sh`, `status_cell.sh` | Submit chains; show queue, receipts, progress. Sent through ssh stdin |
| `deploy_from_windows.sh`, `remote_prepare.sh` | Deploy (refuses provisional rows), and its cluster side: adapter copy, checks, manifests |
| `test_cell.py` | What can be checked without the cluster: `python -m pytest research/hpc/compound_cloning_20261106/test_cell.py` |

## Stages

**Training** (`stage_clone_train.sbatch`, jobs `cc-train`, output `out/clone/`). The training command
of the pipeline's `stage_train.sbatch`: 4-bit base, `--batch-size 1 --gradient-accumulation-steps 4`,
lr 3e-5, one pass, final adapter kept, `--initial-adapter` R1. That stage cannot be driven by overrides:
it requires a round's `collection.done` and mixture, passes one variable to both `--save-steps` and
`--eval-steps`, and dates its run name. Receipt: `out/clone/training.done` (rows manifest, adapter tree
hashes, validation losses).

**Cadence.** Save every 16 steps, validate every 128.

- The trainer takes the two separately (`psse_env/sft/research_cli.py:818-819, 875-876`,
  `psse_env/sft/training.py:331-338`); it ties them only under `--select-best-eval-loss` (`:116-128`),
  which this pass does not use.
- A preemption slot is at least 30 minutes, but every restart first does the row gate over all rows
  (`research_cli.py:578`; 239 s measured, 72 s from the cache of `train_runtime_tools.py`), then loads the
  model and takes an optimizer smoke step (`:665-685`). The spec's 45 steps per slot leave about 4.5
  minutes for that. At 16 an incarnation needs about 9 minutes of training to reach a checkpoint at 34 s
  per step, at 32 about 18.
- Validation is only a monitor here: at 128 it costs 5 minutes per pass, and every validation point
  is a kept checkpoint.

**Resume.** `python -m psse_env.sft.research_checkpoint` picks the newest `checkpoint-N` whose
`trainer_state.json`, LoRA config and base model, adapter weights, `optimizer.pt`, `scheduler.pt` and
`rng_state.pth` all read back (`psse_env/sft/research_checkpoint.py:81-133`); `research-train` passes it to
`trainer.train` (`research_cli.py:707-710, 845-850`). As in `stage_train.sbatch:92-99`, "none complete"
means starting over.

**Chains.** `--requeue` restarts a job on preemption, not at its time limit. A pass longer than one
12-hour job therefore runs as a chain with `afterany` (`submit_cell.sh`): each link resumes, a link with
nothing left exits at once, and evaluations wait with `afterok` on the last link. A link that finds a
sibling running leaves (`another_link_running`). `TRAIN_LINKS` and `EVAL_LINKS` override the counts.

**Checkpoints on disk.** `research-train` has no save limit, so the stage removes old checkpoints when a
job starts and when the pass ends: it keeps the newest three and every multiple of 128
(`prune_checkpoints`). `CLONE_KEEP_EVERY=16` in `cell.env` keeps everything.

**Evaluation** (`stage_eval.sbatch`, jobs `cc-<suite>_<arm>`, output `out/eval/<suite>_<arm>/`). Arms:
`before` (start adapter, waits for nothing) and `after` (cloned adapter, waits for training). Greedy
decoding, 40 actions, no expert arm: teacher references are measured locally and stay out of the
cell. `evaluate_suite.py` is not changed. It reads
`--cell-dir/out/r1/collection/research_run_report.json`; `env_report.py` writes that file under
`out/env/<system>/`. Every evaluation names `lifecycle_choice=1, detailed_evidence=1,
pair_refit_adjacent=1, flow_at_noise_cue=1, route_comparison=0`, residual threshold 4.0, no triage
classifier, profile `scada_only`, because the cell's `PSSE_EXPERT_VARIANT=baseline_compare` (there only for
the check at `evaluate_suite.py:141-143`) supplies no cue. The suite's system is read from its roots: IEEE
14 takes the default progress basis and leaves `PSSE_MULTI_SCAN_RCOND` unset; IEEE 118 takes
`chi_square_excess` and `sparse_symmetric`. At every job start the environment is built from the report
and compared with these values; a difference stops the job.

Four workers per job share the GPU (one policy per job) and checkpoint every episode, so requeues and
later links resume. Results:

- `summary.json`, `episodes.jsonl`, `bc0_eval.json` (full report, `decline_accounting` on every episode);
- `outcomes_<label>.jsonl`: one compact record per episode with `scenario_id`, `class`, `group`
  (breaker or other), corrections with `target_is_true` and `lifecycle`, `decline_accounting`,
  `gate_decisions`. Class and truth join from the suite after the rollout; nothing reaches the model;
- `outcomes.json`: solved, wrong corrections kept, actions, true and wrong targets declined; overall,
  `by_group` (breaker classes apart) and `by_class`.

## Smoke of 2026-11-06

Commits b72639b and ee38110, on H200 nodes. Jobs 22086642, 22086644, 22086645, then 22092916 (training
again) and 22092917, 22092919 (the row gate over the full views on CPU, then its cache hit).

- **Row gate, all rows:** 8,866 + 192 rows, no failure, no prompt or target cut, 239 s on a CPU node. Prompt
  tokens in training: 5,708 at the median, 9,649 at the 95th percentile, 12,987 at most (limit 32,768).
  A cache hit took 72 s; on the smoke rows 33 s of the same step is loading the processor once. So the
  first smoke's projection of 27 minutes per restart was wrong: it scaled that load with the rows.
- **Training, 200 rows, 2 steps on the longest rows:** about 19 s per step, peak 43 GB, adapter reloaded and its
  canary parsed; validation 0.86 s per row, so about 3 minutes for all 192. Whole job 4.5 minutes.
- **Start adapter, new format, 8 cases per suite:** compound 1 solved of 8, no declines, 16.4 actions; conventional
  8 solved of 8, 10.8 actions, false commits on 3 of the 8. The sample is small; the `before` arm measures it.
- **Evaluation:** 8 compound cases in 8.75 minutes and 8 conventional in 6 minutes, startup included, 4 workers,
  at most 30 GB. GPU use in these short jobs says little about the long ones.

## Deploy, smoke, submit

From this worktree in Git Bash, SSH master up (`scripts/start_torch_ssh_master.ps1`), sources committed
on `claude/rl-lifecycle-evidence-20261106`. The deploy refuses while `PROVISIONAL.txt` is beside the rows.

```bash
S=/c/Users/Holiday/Documents/ChatGPT/PSSE_Agent/output/compound_rl_20261106/cloning_selection
TRAIN_VIEW=$S/cloning.train_view.jsonl TRAIN_VIEW_SHA256=<sha256 of the final file> \
VALIDATION_VIEW=$S/cloning.validation.jsonl VALIDATION_VIEW_SHA256=<sha256 of the final file> \
  bash research/hpc/compound_cloning_20261106/deploy_from_windows.sh "$(git rev-parse HEAD)"
```

`DEPLOY_ROWS=0` deploys source and suites without rows, enough for `smoke-eval` and `before` while the
selection is being re-made. Then:

```bash
C=research/hpc/compound_cloning_20261106
export MSYS_NO_PATHCONV=1
wsl -- ssh torch bash -s -- smoke < $C/submit_cell.sh     # training smoke, evaluation smoke per listed suite
wsl -- ssh torch bash -s < $C/status_cell.sh
wsl -- ssh torch bash -s -- before < $C/submit_cell.sh    # start adapter on every listed suite
wsl -- ssh torch bash -s -- pass < $C/submit_cell.sh      # the pass, then the cloned adapter on every listed suite
```

Read after the smoke: `out/smoke/gate_timing.json` (row gate failures stop here; token lengths),
`out/smoke/training.done`, the log's "a resume would start from" line,
`gpu-util-*.csv`, and `out/eval/smoke_*/outcomes.json`. To repeat it: remove `out/smoke` and
`out/eval/smoke_*`. Failed `reload` canaries after only two steps say the start adapter cannot yet answer
that row in this format; read `research_run.json` before concluding the plumbing is broken.

A suite name after the command selects suites, for example `-- before ieee14_dev`. Resubmitting redoes only
what is missing. Results come back through ssh, for example:

```bash
W=/scratch/yx3882/compound_cloning_20261106/out/eval/ieee14_dev_after
wsl -- ssh torch "cat $W/outcomes_after.jsonl" > ieee14_dev_after.jsonl
python -m research.compound_errors.commit_reader_loop compare --reference ieee14_dev_before.jsonl \
  --run after=ieee14_dev_after.jsonl --output before_after   # or a local teacher run as --reference
```

### Adding IEEE 118

Uncomment the two lines in `eval_suites.env` (the working copy is uploaded, no commit needed), then:

```bash
W=/c/Users/Holiday/Documents/ChatGPT/PSSE_Agent/.claude/worktrees/pensive-darwin-33a0d8/output
DEPLOY_ROWS=0 UPLOAD_SUITES="ieee118_draw1=$W/compound_errors_ieee118_20261028/suite.json ieee118_draw2=$W/compound_errors_ieee118_20261029/suite.json" \
  bash research/hpc/compound_cloning_20261106/deploy_from_windows.sh "$(git rev-parse HEAD)"
wsl -- ssh torch bash -s -- before ieee118_draw1 ieee118_draw2 < $C/submit_cell.sh
wsl -- ssh torch bash -s -- after ieee118_draw1 ieee118_draw2 < $C/submit_cell.sh
```

Any other suite joins the same way: a line in the list and a file. No stage changes.

## What is uploaded, and where it lands

| Local | On torch |
| --- | --- |
| `git bundle` of the branch from the torch clone's HEAD | fetched into `$WORK/source`, first cloned from `$SOURCE_CELL/source` |
| `TRAIN_VIEW`, `VALIDATION_VIEW` | `$WORK/data/rows/cloning.train_view.jsonl`, `cloning.validation.jsonl`, with `manifest.json` |
| `selection_report.json` beside the rows | `$WORK/data/rows/selection_report.json` |
| `output/compound_errors_20261027/suite.json` (this worktree) | `$WORK/data/suites/ieee14_dev.json` |
| IEEE 118 draws, when added | `$WORK/data/suites/ieee118_draw1.json`, `ieee118_draw2.json` |
| Working copy of `eval_suites.env` | `$WORK/data/eval_suites.env` |

Already on torch: R1 (`research_full_pipeline_20261022_conventional/out/r1/training/lora`, copied once to
`$WORK/data/start_adapter`) and the conventional test suite (`.../out/r1/collection/development_scenarios.json`,
read in place, the path `submit_extras_20261023.sh` calls `TEST`). Uploads are checksum-verified; a file
already there with the same hash is skipped. Replacing the rows while `out/clone/training` has checkpoints
is refused.

## Duration and GPU-hours

All extrapolated: nothing here has run.

**First pass (IEEE 14 rows, 2,217 steps; default evaluation list): about 50 to 85 GPU-hours.** Training
about 23 (up to about 32 with preemption), evaluation about 28 to 50 for both arms, smoke about 1. The `before`
arm (15 to 25 of those) needs no rows and can run while the selection is being re-made.

**Training.** Basis: 34 s per step on an A100 (spec) is 21 h for 2,217 steps, plus 17 validation passes of
5 minutes, 1.5 h.

| Pass | Rows | Steps | Steps at 34 s | Validation passes | GPU-hours, no preemption | Links |
| --- | --- | --- | --- | --- | --- | --- |
| IEEE 14 only (first pass) | 8,866 | 2,217 | 21 h | 17, 1.4 h | about 23 | 4 |
| Both systems, if a later selection has them | 14,960 | 3,740 | 35 h | 29, 2.4 h | about 38 | 5 |
| Any | R | ceil(R / 4) | 9.4 h per 1,000 steps | one per 128 steps | about 10 per 1,000 steps | from the row count |

Each restart adds the cached row gate (72 s measured), a model load and the lost work since the last
checkpoint: guess 10 to 20 minutes. The 885-step pass of 2026-10-21 was restarted nine
times. Under the run-time tool list the prompts are about 6,900 characters shorter than under the rendering
the 34 s was measured with, so steps may be faster.

**Evaluation.** Basis: the conventional cell's follow-ups ran about 33 student actions per minute per GPU (96 roots, two adapters at about 10 actions plus
the expert, 59 minutes); 25 to 30 per minute is assumed for these longer prompts, about 0.6 GPU-hour per
1,000 actions, plus up to about 15 minutes per job start (planning 1,275 roots took 2.5 minutes locally, then
four model loads in turn). An IEEE 118 action also costs about five times the CPU of an IEEE 14 one (teacher
runs: 2.6 s against 0.5 s), which may lower the rate there.

| Suite | Episodes per arm | Actions per episode (assumed) | GPU-hours per arm | Links |
| --- | --- | --- | --- | --- |
| `ieee14_dev` | 1,275 | 18 (the teacher's 13.5 plus a student's extra calls) to 30 | 13 to 23 | 3 |
| `conventional_test` | 192 | about 10 (R1) | 1 to 1.5 | 1 |
| `ieee118_draw1`, `ieee118_draw2`, each, if added | 625 | 15 (the teacher) to 30 | 6 to 11 | 2 |

Adding both IEEE 118 draws adds 24 to 44 GPU-hours.

**Smoke:** about 15 minutes for training and 20 to 25 per evaluation slice, about 1 GPU-hour in all.

**Disk:** about 0.8 GB per checkpoint (65.6M LoRA parameters, review note of
2026-10-31), so about 110 GB at 2,217 steps without pruning, about 20 GB after it, and up to about 65 GB more
between two starts of an unbroken 12-hour job. Evaluation reports are likely 1 to 2 GB per arm on `ieee14_dev`.

## Not verified without the cluster

Paths and environment:

- R1 at `.../research_full_pipeline_20261022_conventional/out/r1/training/lora`, its `adapter_config.json`
  naming `google/gemma-4-12B-it` (a different name would make `research_checkpoint` refuse every checkpoint and
  each restart begin again). `remote_prepare.sh` checks both.
- The conventional test suite's path and 192 roots (checked at deploy). A local copy of the suite plans
  correctly; the file on torch was not read.
- `$SOURCE_CELL/source` exists and its HEAD is a commit this repository has (checked at deploy).
- The conda environment path, its `trl` and `bitsandbytes`, the cached model snapshot (checked at deploy). The
  stages use that python directly, as the other cells do; no `module load` is involved, so no module name
  was checked.

Training:

- Measured by the smoke, see "Smoke of 2026-11-06": the row gate on these rows under the run-time list, its
  time on the full view, rendering under the run-time list, and the longest row on an H200. Not measured: the
  gate under the registry list.
- 34 s per step, about 45 steps per 30-minute slot and 5 minutes per validation pass on these rows, and on
  each GPU type.
- GPU memory at the longest row: the smoke's optimizer step uses it, on whichever GPU that job gets.
- `--save-steps` and `--eval-steps` apart under the cluster's Transformers (at least 5.10, TRL). Locally
  Transformers 4.49 ties them only under `load_best_model_at_end` and TRL is absent. The smoke runs save 1,
  validate 2.
- Checkpoint size and scratch quota.

Scheduling:

- QOS `gpu48`, the 16-GPU per-user limit and the 12-hour limit as they stand today; `squeue` on compute nodes
  (without it the link guard does nothing).
- `afterany` staying unreleased while a link is requeued after preemption. If it is released anyway the later
  starter of two links leaves, so nothing corrupts, but this has not been seen happening.
- Low-utilization cancellation: four workers should keep the mean above the cancel lines (H100/H200 60%);
  `gpu-util-eval-*.csv` will tell. A cancelled job resumes in its next link.

Evaluation:

- Four 4-bit copies on 80 GB with these prompts (about 15 GB each was measured on shorter ones).
- A worker crash on a particular root: the job fails after the other workers finish and the next link retries.
- The stages have run locally only with stand-ins for `sbatch`, the trainer and `evaluate_suite.py`'s rollout.
  `evaluate_suite.summarize` and `episode_outcomes.py` ran on real roots but synthetic episodes.
