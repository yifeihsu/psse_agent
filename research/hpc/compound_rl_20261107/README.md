# Compound RL cell (2026-11-07)

Stage 3 of the RL plan (`docs/rl_beyond_teacher_plan_20261105.md`, sections 6, 9 and 11): on-policy
LoRA updates of the cloned student from its own sampled rollouts on the compound training draw,
under `scada_only` with the lifecycle choice and the detailed evidence on, scored by the offline
audit, with the lifecycle term from the audit's labels. The driver is `psse_env.rl.train`, the
October pilot's (`research/hpc/rl_pilot_20261014`), with the four changes section 9 of the plan
asked for, built on 2026-11-07:

| Change | Where | What it does |
| --- | --- | --- |
| Environment and teacher options | `RLConfig.environment_options`, `expert_options`; `configure_research_environment` | The named options of the research environment (`lifecycle_choice`, `detailed_evidence`, `pair_refit_adjacent`, `flow_at_noise_cue`, `route_comparison`, `revert_fruitless_cycles`, `target_first_partial`, `branch_first_partial`) and of the teacher, applied by every phase and shard before an environment is built, recorded in `rollout_summary.json` (`environment`). An option a config does not name is left to the factory, so the configs here name all five the cloning cell's evaluations named. |
| The lifecycle term | `psse_env/rl/lifecycle.py`, `psse_env/rl/targets.py`; `Trajectory.lifecycle`; `RewardWeights.lifecycle_credit` and `lifecycle_form`; `build_rows(lifecycle_credit=..., lifecycle_form=..., render_target=...)`; the update's lifecycle and supervised losses | Every commit or decline of an accepted candidate is read from the trace with the verifier's verdict and the correction it disposes of; the scenario's `audit.truth` says whether the target is true. In the `sampled` form a decision that agrees with the audit (commit a true target, decline a wrong one) gets `+credit` on its row, one that disagrees `-credit`, beside the group advantage, and in a group whose samples all scored the same the decision steps alone become rows. In the `supervised` form (the configs) each labelled decision gets a row whose completion is the audit's choice, rendered as the student was trained to emit it. Either way a decision is trained whether or not the outcome reached it, with one weight per decision, apart from the policy term. A config that sets the credit must open the lifecycle choice. |
| Compound-class strata | `root_sampling.key: compound_class` (`psse_env/rl/sampling.py`) | Fixed counts per `audit.compound.class` in every iteration; a root without a class is refused. |
| Evaluation at 16 and 20 actions | `RLConfig.evaluation_budgets`; `aggregate_evaluation` | `solved_within_actions`: the solved cases that used at most 16 and 20 actions under the 40-action run (the policy sees its remaining budget, so this is the plan's efficiency measure beside the solved count, not a run at that budget), with `mean_steps_solved` and the lifecycle counts (decisions, declines, true targets declined, wrong targets committed). |

Every greedy evaluation episode and every sampled trajectory now carries its decisions with the
audit's label, so declines, false declines and kept wrong corrections are read from the records
and not from the outcome field the cloning cell's `episode_outcomes.py` overstates.

A fifth change came out of the plan session's review of the four: **a baseline phase**
(`baseline_evaluation`). Every cloning-cell arm, the development `after` arm included, measures
the student on a 4-bit base (`evaluate_paired_adapters` loads 4-bit by default), while this cell
samples, updates and evaluates in bf16, and the driver only ever evaluated the adapter after an
update. The first iteration's score would have mixed RL's effect with the precision change. With
the flag, a run starts with a greedy evaluation of the starting adapter on the development subset
through the driver's own loader (`out/<run>/baseline/evaluation.json`, recorded in the manifest),
so S0 and every S1 are measured the same way; against the `after` arm on the same 300 cases it
also shows what the precision change alone moves. About 1 to 2 GPU-hours.

## Files

| File | Purpose |
| --- | --- |
| `rl_smoke.json` | The stage 3 smoke: one iteration of 8 roots from the training draw, 4 samples each, no evaluation, output `out/smoke_v1` |
| `rl_ieee14.json` | The IEEE 14 arm: the baseline phase, then 64 roots per iteration in the training draw's class mix (`root_sampling`), greedy evaluation of every iteration on the 300-case development subset, the supervised lifecycle term, output `out/ieee14_v1` |
| `rl_ieee14_sampled.json` | The comparison arm of decision 5: the same run with the per-decision (`sampled`) lifecycle term, output `out/ieee14_sampled_v1`; it differs from `rl_ieee14.json` in `reward.lifecycle_form` and `output_dir` alone (`test_cell.py`) |
| `rl.sbatch` | One job runs `ITERATIONS` iterations as separate rollout, update and evaluate processes (two rollout and evaluation workers on an 80 GB card, each holding a bf16 copy); requeues resume at the manifest's phase; `TARGET_ITERATIONS` ends a job once the manifest reaches it |
| `submit_arm.sh` | An arm as a chain of linked 12-hour jobs (`afterany`), each a link of `rl.sbatch` with the target iteration count; later links exit at once when the target is reached |
| `dev_subset.py` | The fixed stratified development subset (300 of the 1,275 cases, the suite's class mix, seed 20261107) |
| `deploy_from_windows.sh` | Clone the cloning cell's source as this cell's, upload the bundle, the training draw, the subset and the configs |
| `test_cell.py` | The configs load through the driver, name the plan's options, and their class mix matches the training draw |

Settings, from section 9 of the plan and the last working October configuration
(`rl_hard_screen_fix.json`): temperature 1.0 with no nucleus or top-k cut, group-relative
advantages in reward units, trajectory normalization, the clipped ratio 0.2 against the training
pass, on-policy tolerance 0.2, KL 0.04 to the starting adapter with one preservation row per group
(24 groups), learning rate 1e-5, two optimizer steps per iteration, per-row gradient bound 25,
complete groups, prompt digests, truncated prompts rejected, the base model in bf16. The reward
keeps the case terms, drops the tool-regret term (`tool_regret_cost` 0, no expert reference) and
adds the lifecycle credit of 0.25. Training keeps the 40-action budget.

The start adapter is the cloning pass's final adapter
(`/scratch/yx3882/compound_cloning_20261106/out/clone/training/lora`), the plan's S0, on the
strength of its conventional result (5.4 false first declines per 100, under the bar of about 6).
Its development `after` arm (jobs 22178117 to 22178119, submitted 2026-11-07) is S0's score on the
full 1,275-case set, on a 4-bit base; if that arm puts the false declines above the bar, a
correction pass follows and `initial_adapter` changes. `base_revision` is the registered revision
every cloning arm loaded and the only snapshot in torch's cache (the plan session's check of
2026-11-07).

**The form of the lifecycle term (plan decision 5).** `reward.lifecycle_form` chooses it. `sampled`
is the per-decision form on the tokens the policy sampled: with p the probability of the audit's
choice it moves the choice in proportion to p(1 − p), so it barely moves a choice the student seldom
makes, and two such choices hold the prize (declining a wrong accepted candidate; committing a
partial candidate after a decline, 0 of 37 so far). `supervised`, the configs' setting (the user's
decision of 2026-11-07), trains the audit's choice itself, in proportion to 1 − p: for every
labelled decision a row with the step's own prompt and, as its completion, the labelled tool call
rendered the way the cloning rows were (the assistant turn through the chat template, the
empty-thought normalization, the prompt as an exact prefix, a round trip through the policy's own
parser and schema check; `psse_env/rl/targets.py`). The call keeps the arguments the policy wrote
and swaps the tool name. Its loss is the mean negative log-likelihood of the target times the
credit, one weight per decision, outside the policy term, the KL term, the clip and the on-policy
check; the sampled decision rows carry no term in this form. `update_report.json` shows
`supervised_rows`, `supervised_loss` and, under `rows.target_renderer`, how many targets rendered
and how many were refused (a refused target is never trained). The `sampled` form stays as the
comparison arm.

## Running

Make the development subset in the checkout, commit, deploy, submit the smoke:

```bash
python -m research.hpc.compound_rl_20261107.dev_subset --suite output/compound_errors_20261027/suite.json --size 300 --seed 20261107 --output output/compound_rl_20261107/ieee14_dev300.json
```

```bash
bash research/hpc/compound_rl_20261107/deploy_from_windows.sh "$(git rev-parse HEAD)"
```

```bash
wsl -- ssh torch "cd /scratch/yx3882/compound_rl_20261107 && sbatch --export=ALL,CONFIG=rl_smoke.json,ITERATIONS=1 --job-name=crl-smoke source/research/hpc/compound_rl_20261107/rl.sbatch"
```

The smoke passes when, in `out/smoke_v1/iteration_000/`:

- `rollout_summary.json` records the five environment options (`environment.environment_options`),
  `store.groups_with_distinct_samples` is well above zero (the samples differ), and
  `store.lifecycle` shows labelled decisions (`decisions`, `agreeing`, `disagreeing`) and whether a
  decline was ever sampled (`declines`);
- `update_report.json` shows every sampled row with a sampling record and a prompt digest,
  `supervised_rows` equal to the labelled decisions of the paired trajectories with
  `rows.target_renderer.refused` at zero and a finite `supervised_loss`, a `clip_check` of zero at
  the first step, a sampler gap (`on_policy_check`) under 0.2 and a KL of zero at the first step.

Then the arm, as a chain of linked 12-hour jobs, because one iteration does not fit one job:

```bash
wsl -- ssh torch bash -s -- rl_ieee14.json 7 6 crl-ieee14 < research/hpc/compound_rl_20261107/submit_arm.sh
```

The first link runs the baseline phase before the first rollout; each later link resumes where the
one before it stopped (rollouts and evaluations per root, the update at its checkpoint), and links
after the target exit at once. The plan stops RL if after three iterations S1 does not differ from
S0 (the baseline) on the development subset.

**Pace, from the smoke of 2026-11-07 on an RTX PRO 6000:** about 5 minutes per root group per
worker and about 7.3 s per update row, the clip-baseline pass included. One 64-root iteration is
about 256 episodes (2.5 hours on two workers) plus an update of about 3,450 rows (7 hours) plus the
300-case evaluation: about 13 GPU-hours, three times what section 10 of the plan assumed, and the
baseline adds about 3 hours once. Six iterations are about 80 GPU-hours per arm; an H200 is faster.

**A reward fix from the smoke (2026-11-07).** The evaluation's healthy-component check does not
know its answer for an episode it stopped early (a repeated refused action), so such an episode paid
no penalty for a wrong correction it kept, while a clean hand-off that kept the same wrong edit paid
0.5; in one smoke group two refused-escalation loops at −0.61 outscored two clean hand-offs at −0.74.
Now, where the evaluation does not know, the audit of accepted targets decides
(`episode_reward(..., kept_wrong_targets=...)`, `healthy_component_source` in the breakdown). The
smoke's stored rewards predate the fix; both arms share the corrected reward.

Status and results:

```bash
wsl -- ssh torch "cat /scratch/yx3882/compound_rl_20261107/out/ieee14_v1/manifest.json"
```

Each iteration directory holds `roots.json`, `trajectories/groups/*.jsonl` (every sampled episode
with its reward, audit, recorded tokens, their sampling probabilities and its labelled decisions),
`rollout_summary.json`, `advantages.json`, `update_report.json`, the new `adapter/`,
`evaluation/episodes/*.json` and `evaluation.json` with the greedy development score, the solved
counts within 16 and 20 actions and the lifecycle counts.

Status on 2026-11-07: built and tested in the checkout (the driver's unit tests, and the
real-environment wiring check with the options on and the expert rolled out); deployed at a0a472b,
and the smoke (job 22193977) passed every criterion above: 42 supervised rows for 42 labelled
decisions with none refused, clip check and KL at zero at the first step, a sampler gap of 0.0004,
finite losses. The arm waits for the development `after` arm's read of the start adapter.
