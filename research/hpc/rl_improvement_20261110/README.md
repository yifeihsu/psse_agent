# RL improvement series, 2026-11-10

The run cell of the workplan in `docs/rl_improvement_workplan_20261110.md`. What was built and measured for it
is in `docs/rl_improvement_implementation_20261110.md`. Work directory on torch:
`/scratch/yx3882/rl_improvement_20261110`.

**On a GPU so far: the two smokes** (2026-11-11, both passed; section 7 of the implementation notes). Everything
else here is tested on a CPU only.

## The arms

Every run config is `series.json`'s base with its arm's overrides (`make_configs.py` writes `configs/`, and
`make_configs.py --check` fails when a file on disk is not what the series gives).

| Arm | Config | Observation | Objective |
| --- | --- | --- | --- |
| A | `arm_a_bounded.json` | bounded history (the last four events) | the current episode and lifecycle objective |
| B | `arm_b_complete.json` | complete observable history | the same as A |
| C | `arm_c_bounded_credit.json` | bounded history | decision and continuation credit |
| D | `arm_d_complete_credit.json` | complete observable history | the same as C |

- B differs from A in `observation_mode` alone, C from A in `decision_credit` alone, D from B in
  `decision_credit` and from C in `observation_mode` (`test_cell.py` holds this).
- All four start from the cloned adapter S0, keep it as the KL reference, draw the same roots with the same seeds,
  and take the same update.
- Arm A is the first arm (`compound_rl_20261107/rl_ieee14.json`) with the repaired update and nothing else
  changed, so its first rollout batch is the first arm's.

The update every arm takes (`base.update`):

| Setting | Value | Why |
| --- | --- | --- |
| `chunk_by` | `group` | a root's samples stay in one optimizer step, where their advantages sum to zero |
| `carry_optimizer` | `true` | AdamW's moments continue from one iteration's update to the next |
| `optimizer_steps`, `fresh_optimizer_steps` | 2, 1 | one step whenever the optimizer is new: two steps of a new AdamW follow the first chunk |
| `loss_normalization` | `dapo` | what a group's samples share cancels exactly (`trajectory` left 16 to 21% of it) |
| `first_step_baseline` | `own` | the first step clips against the row's own forward, not a baseline written on another GPU type |
| `max_row_grad_norm` | 1.0 | the approved row bound, the same in every arm |
| `step_diagnostics`, `diagnostic_components` | on, off | the step record without the extra backward passes (the calibration runs turn the split on) |

## The prompt limit

`max_input_tokens` and `max_length` are 49,152 tokens in every config (the earlier cells ran with 32,768).

- The model is not the limit: `google/gemma-4-12B-it` has 262,144 positions.
- The cards are. On the first arm the update peaked at 37.3 GB with rows of at most about 13,000 tokens, which is
  about 24 GB of weights and about 1 MB per prompt token. At 49,152 tokens that extrapolates to about 73 GB, inside
  an 80 GB card. This is an extrapolation; `TASK=capacity` below measures it.
- It covers what has been seen: the longest complete-history prompt of 140 teacher episodes is 33,737 tokens.
- A complete-history prompt over the limit is refused and counted, never cut. One that is under the limit and
  still does not fit the device is tried again twice and then refused the same way.
- To change it, edit `base.max_input_tokens` and `base.max_length` in `series.json` and run `make_configs.py`.

## Order of work

Nothing below is submitted by any script here except `submit_arm.sh` and the `sbatch` lines, which need a person.

1. **Deploy** (from the Windows checkout, sources committed):
   `bash research/hpc/rl_improvement_20261110/deploy_from_windows.sh <commit40>`
2. **Smokes** (8 roots, one iteration, no evaluation; the second one also branches four decisions):
   ```
   sbatch --export=ALL,CONFIG=smoke_complete.json,ITERATIONS=1,TARGET_ITERATIONS=1 --job-name=rli-smoke source/research/hpc/rl_improvement_20261110/rl.sbatch
   sbatch --export=ALL,CONFIG=smoke_complete_credit.json,ITERATIONS=1,TARGET_ITERATIONS=1 --job-name=rli-smoke-cd source/research/hpc/rl_improvement_20261110/rl.sbatch
   ```
   Read in each `update_report.json`: `integrity.ok`, `rows.target_renderer.refused == 0`,
   `rows.prompts_cut_at_generation == 0`, and in `rollout_summary.json` the `prompts` block (steps refused).
   On 2026-11-11 each took about 8 GPU-hours (5 and 6 preemptions): the update ran at 38 to 40 s per row, with the
   component split these two configs turn on.
3. **Prompt capacity** on each card type the jobs land on:
   `sbatch --export=ALL,CONFIG=arm_b_complete.json,TASK=capacity --job-name=rli-capacity source/.../rl.sbatch`
   writes `out/prompt_capacity/<card>-<job>.md`. If a card supports less than the limit, lower the limit. To see
   how far a larger card goes, add `CAPACITY_LENGTHS="32768 49152 65536 98304 131072"` to the export.
4. **Token audit on the learner's own states** (stage 1's gate; CPU, the processor's own counts):
   `python -m research.rl.history_audit trajectories --store out/smoke_complete/iteration_000/trajectories --suite <training suite> --tokenizer google/gemma-4-12B-it --revision 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 --limit 48896 --output-dir out/history_audit_smoke`
5. **Stage 0, calibration** on the fixed batch (the first arm's iteration 0, S0's samples):
   ```
   for run in calibration_1step calibration_2chunk calibration_4chunk; do
     python -m research.hpc.rl_improvement_20261110.calibration prepare --config configs/$run.json \
       --batch /scratch/yx3882/compound_rl_20261107/out/ieee14_v1/iteration_000/trajectories
   done
   bash source/research/hpc/rl_improvement_20261110/submit_arm.sh calibration_1step.json 2 1 rli-cal1   # and the other two
   python -m research.hpc.rl_improvement_20261110.calibration report --run out/calibration_1step --run out/calibration_2chunk --run out/calibration_4chunk --output out/calibration_report
   ```
   Start with the one-step and two-chunk runs; the four-chunk run is for weak decision movement only.
6. **Stage 2, fork probe**: `sbatch source/research/hpc/rl_improvement_20261110/fork_probe.sbatch`
   (learner rollouts on the probe's roots, the frozen fork sample, 48 states x 2 views, 384 suffix episodes, the
   report with the screening gate, which counts the learner's continuations alone and is "not evaluated" for a
   probe the rule expert continued). With the probe's forks in place, the decision panel of stage 0 is
   `sbatch --export=ALL,PANEL_ADAPTER=<an updated adapter>,PANEL_LABEL=cal1 fork_probe.sbatch`.
7. **Stage 3, arms A and B**: `submit_arm.sh arm_a_bounded.json 6 2 rli-a` and `submit_arm.sh arm_b_complete.json 8 2 rli-b`.
   After each evaluation: `research.rl.baseline_register add`, then `research.rl.scorecard`.
8. **Stage 4, arms C and D** only if the fork probe passes its gate (five distinct expert-failed roots solved by
   learner continuations, one or more in the validation split).

## Stopping and promotion

- Two iterations per arm (`TARGET_ITERATIONS=2`). A third is a new submission, made to resolve a concrete uncertain
  signal. After three completed iterations without a positive paired net of solved cases over the arm's own
  baseline, the arm is not extended. A better local decline rate does not reset the count.
- A run stops at once on a nonfinite update, wrong likelihood provenance, a cut history, duplicated or missing
  update work, or an invalid checkpoint recovery. Every arm has `stop_on_integrity_failure`: the update writes
  `iteration_NNN/integrity_failure.json`, the job ends with `rli-job-stopped`, and the phase is refused until the
  marker is removed by hand. A batch whose rows were all rejected stops the run the same way.
- An update's checkpoint is continued only under the settings and the order of rows it was written with. To change
  a setting in the middle of an update, remove that iteration's `update_checkpoint/` and let the update start again.
- A preemption is neither an iteration nor a failure.
- Promotion (`research.rl.scorecard`, `promotion_check`): the learner evaluated on exactly the matched reference's
  roots, positive paired net solves over that reference, an expert-failed root recovered, a nonnegative paired
  clean-success change, no more healthy-component corruption. These allocate compute on small samples. They are
  not tests of significance.

## Files

| File | What it is |
| --- | --- |
| `series.json` | the base config, the arms' overrides, the inputs with their digests, the protocol |
| `make_configs.py`, `configs/` | the generator and the run configs it writes |
| `reserve_roots.py`, `reserved_roots.json`, `probe_training_roots.json` | the 64 training roots no arm may draw (the probe's validation split and the decision panel) and the probe's 128 training-split roots |
| `rl.sbatch`, `submit_arm.sh` | one job of a run, and a run as a chain of 12-hour jobs |
| `fork_probe.sbatch` | the fork probe and the decision panel |
| `calibration.py` | start a run at the update of a fixed batch; compare the runs |
| `provenance.py` | what a job ran on (source commit, config and input digests), written per job |
| `baseline_register.json`, `seed_register.py`, `first_arm_read.json` | the register of finished evaluations, seeded from a read of the first arm on torch |
| `deploy_from_windows.sh` | source, configs and root lists to torch; submits nothing |
| `test_cell.py` | the configs are what the series gives, matched arms differ as it says, the tools work |

The digests in the configs (`series.planned_inputs`) and in each job's provenance record are a record of what
ran. A job on other inputs is reported, not refused.

## The first arm, for reference

The register starts with the first arm's four evaluations on the 300-case development subset (bf16, bounded
history, source `dcb6b38`): S0 182, S1 180, S2 181, S3 182 solved; rule expert 188; audit deciding 200. That arm's
updates were two steps of a new optimizer over halves of the batch. Its chain was cancelled on 2026-11-10 after
the third evaluation showed no gain; its run folder is kept.
