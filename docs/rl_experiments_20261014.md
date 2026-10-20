# RL experiments: what was built and how to run them (2026-10-14, revised 2026-10-15)

Branch `codex/classifier-triage-20261004`. Implements the plan of
`docs/rl_hard_cases_plan_20261013.md` (steps 1 to 4) and the framework
decision of 2026-10-14 (an in-repo on-policy loop rather than TRL's or
verl's agent loops, which assume an append-only transcript the policy here
does not observe). Section 2 records the first cluster submission and what it
showed; section 3 the fixes it led to.

## 1. What exists now

| Piece | Where | Plan step |
| --- | --- | --- |
| Graded episode reward: basis credit (decisive 1.0, bounded handoff 0.6, unexplained handoff 0.5, budget exhaustion 0.4), per-call and unneeded-acquisition costs, tool regret against the expert's calls on the same root, penalties for false commits, false finalizations, a healthy edit, a loop and invalid actions; predeclared weights | `psse_env/rl/reward.py` | 2 |
| Group-relative advantage (GRPO or RLOO), zero-variance groups dropped and counted | `psse_env/rl/advantages.py` | 3 |
| Sampled decoding in the research policy with a per-step record of the model observation, the exact completion tokens and each token's sampling probability; sampling always passes `top_p` and `top_k` so the model's generation config cannot cut the distribution; greedy decoding unchanged by default | `psse_env/dagger/research_policy_factory.py` | 3 |
| Grouped rollouts: K seeded episodes per root through the closed-loop evaluator, paired with the trace and scored; a per-root JSONL store that resumes; per-group count of distinct trajectories and a store-wide summary | `psse_env/rl/rollout.py` | 3 |
| Exact on-policy training rows: prompt re-rendered by the policy's own renderer and checked against the recorded length, completion = the sampled tokens, assistant-only mask, zero side inputs | `psse_env/rl/rows.py` | 3 |
| The LoRA update: one on-policy gradient pass (ratio one) with log-probabilities at the sampling temperature, DAPO token normalization, optional k3 KL to the starting adapter loaded as a frozen second PEFT adapter, completion-only logits, checkpoints that resume a preempted update mid-step | `psse_env/rl/update.py` | 3 |
| The iteration driver: rollout, update, evaluate phases with a manifest; rollout and evaluation stored per root and sharded over worker processes; evaluation skipped when the adapter did not change; `policy_source: expert` for a CPU wiring check | `psse_env/rl/train.py` | 3, 4 |
| Hard-draw plumbing in the suite builder: training rank allowance, stratum plans per draw, effect band threaded into the research generator | `research/hpc/full_pipeline_20260907/build_suite.py`, `scripts/run_dagger_research.py` | 1 |
| Identifiability tagger: per root, which candidates the deployment verifier accepts (true and top-ranked wrong), tag and per-stratum summary | `research/rl/identifiability.py` | 1.3 |
| HPC cell: job script with per-phase processes, two workers on an 80 GB card, requeue resume; deploy script; pilot config; README | `research/hpc/rl_pilot_20261014/` | 4 |

Design points worth knowing before reading the code:

- The reward is computed only from the evaluator's episode record, after
  the episode ended; nothing truth-derived enters the policy's view.
- A trajectory whose generation records do not pair one-to-one with the
  evaluator's policy steps is kept for its reward but excluded from training
  (`pairing_ok`). The expert-sourced rollouts are such trajectories by
  construction.
- The update is on-policy with one optimizer pass per rollout batch, so no
  log-probabilities are stored and no clipping is needed; `optimizer_steps`
  only partitions the batch. The config refuses a nucleus or top-k cut,
  because samples from a cut distribution are not samples of the policy the
  update optimizes; exploration is set with the temperature, and the update
  takes log-probabilities at that same temperature.
- The KL reference, when `beta_kl > 0`, is the run's starting adapter (R2),
  loaded beside the trained one. It is never the bare base model.
- Every phase survives the cluster's preemption: rollouts and evaluation
  episodes are stored per root, and the update commits its adapter and
  optimizer after each optimizer step and its accumulated gradients every
  `update_checkpoint_rows` rows.

## 2. First cluster submission (2026-10-14) and what it showed

Job 19859785 from commit 3e78301, config `rl_pilot.json` as first written
(temperature 0.8, top-p 0.95, top-k unset, KL 0.02), output
`/scratch/yx3882/rl_pilot_20261014/out/pilot`. It ran twice on an A100, was
preempted at the 30-minute mark both times, and stored 25 of 61 root groups
before it was cancelled on 2026-10-15 (the groups stay on disk).

| Measurement on the 25 stored groups (four samples each) | Value |
| --- | --- |
| Groups whose four samples were token-identical | 23 |
| Groups with any reward spread | 2 |
| Episodes solved under the audit | 100 of 100 |

In both informative groups three samples followed the greedy trajectory and
one deviated into a worse one: an unbalance root took 11 calls instead of 4,
a parameter root fetched a meter context first and made an invalid action.
On the standard draw the round-2 student is close to deterministic, and its
rare deviations are mistakes, so a policy gradient there mostly sharpens the
greedy behaviour. That is the plan's prediction for the standard draw.

Three defects surfaced while reading the run, none of which had acted yet:

- **The sampling distribution was cut twice.** The pilot passed top-p 0.95,
  and the Gemma 4 generation config adds top-k 64 whenever top-k is unset.
- **The KL reference was the bare base model** (adapter disabled), which
  would have pulled the student back toward the untuned Gemma at every
  update. No update had run.
- **The update and evaluation phases restarted from scratch after a
  preemption**, so a two-hour evaluation could never finish under 30-minute
  preemption.

## 3. Fixes of 2026-10-15 and the relaunch

- Sampling passes `top_p` and `top_k` explicitly, `top_k: null` means no cut,
  and the config refuses either cut. The policy records each sampled token's
  probability, so a run can tell a peaked policy from a broken sampler.
- The update takes log-probabilities at the sampling temperature; the KL
  reference is the starting adapter as a second PEFT adapter, checked on a
  real tiny PEFT model (`psse_env/rl/test_peft_reference.py`); the pilot
  runs with `beta_kl: 0`.
- The update checkpoints itself (tested: an update interrupted mid-step and
  resumed reaches the same parameters as an uninterrupted one); evaluation
  is stored per root, sharded, and skipped when the update left the adapter
  unchanged.
- The tool-regret reference now covers the training roots: the expert solves
  all 122 round-2 training roots locally in 2.6 minutes, and its calls per
  root are merged with the development roots' into
  `output/rl_pilot_20261014/expert_reference.jsonl` (282 roots, mean 6.92
  calls).

The relaunched pilot (`rl_pilot.json`, output `out/pilot_v2`) samples at
temperature 1.0 with no cut, four samples per root over 61 roots per
iteration, two optimizer steps, no KL, and evaluates greedily on the 160
development roots when the adapter changed.

Deploy and submit (SSH master up):

```bash
bash research/hpc/rl_pilot_20261014/deploy_from_windows.sh "$(git rev-parse HEAD)"
```

### Relaunch results (job 19871504, 2026-10-15)

The relaunched pilot ran both iterations in 5 h 37 min on one A100 (gr103),
without a preemption or an error; the update checkpointed every 32 rows as
designed. The control is the frozen round-2 adapter scored by the same code
on the same 160 development roots (`rl_pilot_r2_baseline.json`, job
19889349), because the round-2 student's own 159 of 160 dates from the
2026-10-07 environment.

| Iteration | Groups with reward spread | Training rows (tokens) | Policy loss per step | Gradient norm before clipping |
| --- | --- | --- | --- | --- |
| 0 | 10 of 61 (16.4%) | 402 (8,673) | −0.021, −0.004 | 54, 37 |
| 1 | 6 of 61 (9.8%) | 201 (4,244) | −0.031, −0.010 | 58, 5,255 |

All 488 sampled episodes were solved; 95% of sampled steps had every token
at probability 0.99 or above. 16 of iteration 0's trajectories were left
unpaired by the invalid-call recording defect fixed in e622689 before
iteration 1.

| Greedy evaluation, 160 roots | Solved | Mean steps | Mean reward | False commits (episodes) | Invalid actions (episodes) | Loops |
| --- | --- | --- | --- | --- | --- | --- |
| Frozen round-2 adapter | 159 | 8.08 | 0.710 | 23 (20) | 39 (28) | 2 |
| After iteration 0 | 159 | 8.01 | 0.722 | 20 (18) | 35 (27) | 2 |
| After iteration 1 | 159 | 8.03 | 0.715 | 23 (20) | 33 (26) | 1 |

All three miss the same root, the misranked parameter root r0_4d7aa8a5a01f.

Reading: the machinery works end to end on the cluster, and on the
standard draw it changes nothing measurable, as the plan predicted. The
student already matches the expert's success, and in the few groups with a
reward spread the deviating sample is usually the worse one, so the update
mostly sharpens the greedy behaviour. Iteration 1's second optimizer step
had a gradient norm of 5,255 against 37 to 58 elsewhere; clipping bounded
its size, but one outlier row likely set its direction, and iteration 1
gave back iteration 0's small gain in false commits. Before a longer run:
more rows per optimizer step, or a per-row bound on the gradient, so one
row cannot steer a step. The hard draw (section 4) is where the next run
belongs.

## 4. The hard draw (built 2026-10-15)

Plan step 1. The draw keeps recoverable admission, so every root still
alarms and has a correction that verifies, and relaxes only the filters that
make roots teacher-solvable: parameter roots are admitted at dominance
threshold 1.0 with any rank of the true line, the 10-sigma meter floor is
off, and the synthesized harmonics use effect band 1.5 to 3. Physical
admission is not used: it also drops the alarm requirement, which would mix
sub-threshold faults (section 10 of the plan) into the draw. Strata are
labelled at the deployed context's threshold of 1.2
(`--stratum-dominance-threshold`), whatever the admission threshold.

```bash
python research/hpc/full_pipeline_20260907/build_suite.py --source-root . --evidence-profile classifier_gated_diagnostics --d0-raw <d0 roots> --protected-suite <classifier development.json> --protected-suite <r1_training.json> --protected-suite <r2_training.json> --round-train-plan '{"measurement": 32, "multi_measurement": 32, "harmonic": 24, "topology": 8, "measurement+topology": 8, "no_error": 8, "parameter": 52, "measurement+parameter": 44}' --development-plan '{"measurement": 24, "multi_measurement": 24, "harmonic": 16, "topology": 8, "measurement+topology": 8, "no_error": 8, "parameter": 44, "measurement+parameter": 32}' --rounds 1 --seed 20261015 --training-threshold 1.0 --development-threshold 1.0 --training-rank-allowance 99 --development-rank-allowance 99 --stratum-dominance-threshold 1.2 --stratum-candidates 600 --effect-band 1.5,3 --topology-effects dangling_line_terminal,bus_split --training-stratum-plan '{"parameter": {"misranked": 4, "ambiguous": 40, "dominant": 8}, "measurement+parameter": {"misranked": 4, "ambiguous": 32, "dominant": 8}}' --development-stratum-plan '{"parameter": {"misranked": 12, "ambiguous": 24, "dominant": 8}, "measurement+parameter": {"misranked": 8, "ambiguous": 16, "dominant": 8}}' --output-dir output/rl_hard_draw_20261015/suite
python -m research.rl.build_mimic_suite --per-variant 24 --seed 20261015 --output output/rl_hard_draw_20261015/mimic/development.json
python -m research.rl.build_mimic_suite --per-variant 24 --seed 20261016 --output output/rl_hard_draw_20261015/mimic/training.json
```

The build takes a few minutes locally. Inputs: the D0 root fingerprints of
the 2026-10-07 classifier cell and its three suites, all excluded; no root
of the hard draw overlaps them or each other. The merged suites are
`output/rl_hard_draw_20261015/hard_development.json` (209 roots) and
`hard_training.json` (256), uploaded with the expert reference to
`/scratch/yx3882/rl_pilot_20261014/data/hard_draw_20261015`.

**Pool limit.** Only 500 of the corpus's 620 parameter rows have their
true-network case files (the same in the local and the cluster checkout),
about 400 of them in the training partition, and misranked roots are about
3% of them. The development draw got 9 of the 12 misranked parameter roots
it asked for; more need a fresh IEEE 14 parameter corpus
(`psse_env/providers/balanced_corpus.py`, single-process, hours).

**Development draw, expert ceilings and identifiability** (baseline = the
current spectra-first expert under the classifier gate; ranked ledger = the
screen profile's ledger expert with the learned ranker):

| Stratum | Roots | Median effect ratio | Separable by verification | Baseline | Ranked ledger |
| --- | --- | --- | --- | --- | --- |
| Parameter, misranked | 9 | 15 | 1 | 1 | 5 |
| Meter plus parameter, misranked | 8 | 27 | 3 | 1 | 4 |
| Parameter, ambiguous | 24 | 107 | 15 | 23 | 24 |
| Meter plus parameter, ambiguous | 16 | 131 | 9 | 16 | 16 |
| Mimic, opposite-sign flow pair | 24 | 6.6 | 10 | 20 | 20 |
| Mimic, same-sign flow pair | 24 | 8.4 | 24 | 20 | 20 |
| Parameter and meter plus parameter, dominant | 16 | 6 to 46 | 14 | 16 | 16 |
| Single meter, no floor | 24 | 2.4 | 24 | 24 | 24 |
| Multi-meter, no floor | 24 | 9.2 | 24 | 24 | 24 |
| Harmonic, effect band 1.5 to 3 | 16 | 2.3 | n/a | 16 | 16 |
| Topology, meter plus topology, healthy | 24 | | | 24 | 24 |
| **All** | **209** | | 126 of 177 balanced | **185 (88.5%)** | **193 (92.3%)** |

The training draw (256 roots) reads the same way: baseline 242 (94.5%),
72% of its balanced roots separable. By the plan's predeclared rule (best
expert between 60 and 95%) the development draw is hard enough; without the
mimics it was not (ranked ledger 153 of 161, 95.0%).

What the draw shows:

- **Relaxing the teacher-specific filters alone barely hurts the expert.**
  Meters down to effect ratio 0.5, multi-meter roots without the floor and
  low-effect harmonics are all solved. The residual difficulty is in three
  places: misranked parameter roots, and both kinds of flow-meter mimic.
- **"Not separable by verification" is not "unsolvable".** On misranked
  roots the wrong neighbouring line passes verification, often as partial
  progress, yet the ranked-ledger expert solves 9 of 17 because the screen's
  refit ranking tests the true line first. A better ordering from
  observable evidence avoids the trap: this is the stratum where a learned
  ordering can add something.
- **Same-sign mimics fail on the protocol, not the diagnosis.** Both experts
  find exactly the right two meters and correct them one at a time; each
  correction is estimated while the other meter is still biased, so the
  final values miss the audit's tolerance. A joint correction of the pair
  would land.
- **Opposite-sign mimics fail on identifiability.** The pair reads as an
  injection error at the line's two buses; the experts correct healthy
  injection meters and verification accepts them (14 of 24 not separable).
- **The contract forbids comparing candidates.** The environment refuses to
  roll back an accepted candidate, so neither the expert nor a policy can
  try two lines and keep the better one. In 14 of the development draw's
  inseparable roots (19 in training) the true correction clears the alarm
  while the wrong one is accepted only as partial progress, so a comparison
  would separate them; allowing it is a contract decision.

Reproduce the tags, ceilings and report:

```bash
python -m research.rl.identifiability --suite output/rl_hard_draw_20261015/hard_development.json --output-dir output/rl_hard_draw_20261015/identifiability_hard_development
python -m research.classifier_triage.expert_e2e --suite output/rl_hard_draw_20261015/hard_development.json --profile classifier_gated_diagnostics --expert baseline --output-dir output/rl_hard_draw_20261015/e2e_hard_development_baseline
python -m research.rl.hard_draw_report --suite output/rl_hard_draw_20261015/hard_development.json --tags output/rl_hard_draw_20261015/merged/tags_development.jsonl --expert baseline=output/rl_hard_draw_20261015/merged/dev_baseline.jsonl --expert ledger_ranked=output/rl_hard_draw_20261015/merged/dev_ledger_ranked.jsonl --output output/rl_hard_draw_20261015/report_hard_development.json
```

**RL run on the hard draw (submitted 2026-10-15).** Job 19932222 from commit
a50b5ab, config `rl_hard.json` (output `out/hard_v1`, 64 of the 256 training
roots per iteration, four samples each, two iterations per job, target four
iterations, so one pass over the training draw). The update now bounds each
row's gradient (`max_row_grad_norm` 25, then five times the step's median
row norm after 16 rows; non-finite rows dropped; row norm quantiles and
clips reported per step), after the pilot's 5,255 spike. The control, job
19932223 (`rl_hard_r2_baseline.json`, output `out/hard_r2_baseline`), scores
the frozen round-2 adapter on the same 209 development roots with the same
code.

**Row bound corrected (2026-10-15).** The classifier arm's first update
showed per-row gradient norms with median 1e-4, 95th percentile 1.5 and
maximum 11.6: most rows sample tokens of probability close to 1, so five
times the median clipped 88 of 229 and 93 of 228 rows, the ones carrying the
signal, and the step norms fell to 0.92 and 0.23. The median-relative bound
was dropped (d3f250a); both arms keep the absolute cap of 25, which clipped
nothing in that update and still catches a spike like the pilot's. The
classifier arm's iteration-0 update was redone from the same stored
rollouts (job 19956235); the superseded adapter and partial evaluation are
kept in `iteration_000/superseded_20261015_median_bound`. The screen arm had
not reached its update and runs with the corrected config from the start.

**Screen-profile arm (submitted 2026-10-15).** Job 19952750, config
`rl_hard_screen.json` (output `out/hard_screen_v1`): the same draw, seed and
roots per iteration under `suspicion_gated_diagnostics`, where the policy sees
the balanced screen's report, including the refit ranking that lets the
ranked ledger expert solve 9 of the 17 misranked roots. It starts from the
2026-10-01 ranked cell's round-2 adapter, which was trained under that
profile; its regret reference is the ranked ledger expert's calls (243 of
the 256 training roots solved). Control: job 19952751
(`rl_hard_screen_baseline.json`), the frozen ranked-cell adapter on the same
209 development roots. The classifier arm's control finished at 182 of 209
(87.1%), stratum by stratum within three roots of the baseline expert.

## 5. Hard-draw results after two iterations, and the stratified, KL-anchored rerun (2026-10-16)

Greedy evaluation on the 209 hard development roots (experts: baseline 185
under the classifier gate, ranked ledger 193 under the screen profile):

| Run | Solved | False commits | Invalid actions | Loops | Mean steps |
| --- | --- | --- | --- | --- | --- |
| Classifier arm, frozen round-2 control | 182 | 14 | 51 | 7 | 10.11 |
| Classifier arm, iteration 0 | 183 | 13 | 37 | 7 | 10.01 |
| Classifier arm, iteration 1 | 183 | 15 | 35 | 6 | 10.02 |
| Screen arm, frozen ranked-cell control | 174 | 9 | 42 | 30 | 10.08 |
| Screen arm, iteration 0 | 173 | 11 | 51 | 19 | 10.20 |
| Screen arm, iteration 1 | 170 | 10 | 89 | 21 | 10.78 |

| Update | Informative groups | Rows | Policy loss per step | Step gradient norm | Largest row norm | Rows clipped |
| --- | --- | --- | --- | --- | --- | --- |
| Classifier, iteration 0 | 11 of 64 | 457 | −0.016, 0.004 | 22.4, 65.7 | 11.5, 80.3 | 0, 3 |
| Classifier, iteration 1 | 10 of 64 | 517 | 0.014, −0.012 | 14.7, 7.6 | 10.2, 3.0 | 0, 0 |
| Screen, iteration 0 | 16 of 64 | 960 | −0.069, −0.097 | 7.9, 3.1 | 3.7, 1.8 | 0, 0 |
| Screen, iteration 1 | 20 of 64 | 1,039 | −0.126, −0.091 | 3.5, 45.8 | 1.7, 22.8 | 0, 0 |

- **The classifier arm is level with its control.** The one lasting change
  is about 30% fewer invalid actions; 198 of 209 tool sequences are the
  same in both iterations. Misranked roots stay at 1 of 8 and 1 of 9.
- **The screen arm's second update broke its harmonic roots** (12 to 5 of
  16). On every lost root the adapter reads the measurement context after
  the three-phase estimate instead of the harmonic context, then cycles
  through corrections, estimator runs and rollbacks for 33 to 40 steps. It
  gained three mimic and multi-meter roots; misranked roots are unchanged.
- **Cause, from the iteration-1 batch.** 18 of its 20 informative groups were
  measurement-type roots, where the cost terms reward skipping the
  three-phase and harmonic acquisitions and going straight to measurement
  corrections, and one measurement root punished asking for the harmonic
  context after the three-phase estimate. One harmonic group held a 40-step
  looping sample (advantage −1.73, 1,014 tokens): it produced the 22.8 row
  and most of the 45.8 step norm, because the DAPO normalization weights a
  trajectory by its tokens. The harmonic decision was already near a tie
  (it flipped both ways in iteration 0), so a general push toward
  measurement corrections carried over.
- **Harmonic roots were not under-sampled.** Each iteration drew 64 disjoint
  training roots, 7 of them harmonic, above their proportional 6; a
  proportional stratified draw would not have changed their weight.

**The rerun** (`rl_hard_screen_strat_kl.json`, output
`out/hard_screen_strat_kl_v1`) keeps the screen arm's starting adapter,
suites, seed, reward and update settings and changes two things:

1. **Stratified roots** (`psse_env/rl/sampling.py`, config block
   `root_sampling`): every iteration draws fixed counts per hard-draw
   stratum, each stratum walked through its own seeded permutation, so a
   stratum's roots are used up before any repeats. The counts oversample the
   regression's stratum and the strata that carried signal in the first
   run, keep every misranked training root in every iteration, and keep a
   floor on the strata the policy solves four times out of four:

   | Stratum | Training roots | Proportional share | Count | Informative groups in the first run |
   | --- | --- | --- | --- | --- |
   | Harmonic | 24 | 6 | 12 | 4 of 14 |
   | Mimic, same-sign flow pair | 24 | 6 | 8 | 6 of 13 |
   | Mimic, opposite-sign flow pair | 24 | 6 | 6 | 3 of 13 |
   | Multi-meter | 32 | 8 | 8 | 8 of 14 |
   | Single meter | 32 | 8 | 6 | 5 of 18 |
   | Meter plus parameter, ambiguous | 32 | 8 | 4 | 4 of 17 |
   | Meter plus parameter, misranked | 4 | 1 | 4 | 1 of 3 |
   | Parameter, misranked | 4 | 1 | 4 | 1 of 1 |
   | Meter plus topology | 8 | 2 | 2 | 2 of 3 |
   | Topology | 8 | 2 | 2 | 2 of 4 |
   | Parameter, ambiguous | 40 | 10 | 3 | 0 of 17 |
   | Parameter, dominant | 8 | 2 | 1 | 0 of 3 |
   | Meter plus parameter, dominant | 8 | 2 | 2 | 0 of 5 |
   | Healthy | 8 | 2 | 2 | 0 of 3 |

   Over four iterations the draw covers 200 of the 256 training roots. The
   strata come from the audit and shape the training mix only; the policy
   never sees them. A run without `root_sampling` draws exactly as before
   (checked against both running arms' recorded roots).
2. **KL penalty to the starting adapter**, `beta_kl` 0.04 (the GRPO paper's
   value), k3 on the sampled tokens, the reference loaded as a second PEFT
   adapter. It resists large departures from the starting adapter; it
   cannot hold a near-tie decision by itself (a token moving from 0.55 to
   0.45 changes the k3 term very little), so the harmonic count is the main
   protection. The reference costs one extra forward pass per row, about a
   third more update time, so a two-iteration job may reach the 12-hour
   limit; the manifest resumes after a time-limit end.

The KL path had only run on tiny test models. `rl_hard_screen_strat_kl_smoke.json`
(8 stratified roots, at most 24 steps, no evaluation, output
`out/smoke_strat_kl`) runs one iteration on the real model: the first
optimizer step must report a KL of zero, because the trained adapter starts
equal to the reference and the pipeline trains its adapters without
dropout, and the second step a small positive one.

```bash
cd /scratch/yx3882/rl_pilot_20261014 && PATH=/opt/slurm/bin:$PATH sbatch --export=ALL,CONFIG=rl_hard_screen_strat_kl_smoke.json,ITERATIONS=1 --job-name=rl-smoke-strat-kl source/research/hpc/rl_pilot_20261014/rl_pilot.sbatch
cd /scratch/yx3882/rl_pilot_20261014 && PATH=/opt/slurm/bin:$PATH sbatch --export=ALL,CONFIG=rl_hard_screen_strat_kl.json,ITERATIONS=2 --job-name=rl-rl_hard_screen_strat_kl source/research/hpc/rl_pilot_20261014/rl_pilot.sbatch
```

The rerun's comparison is the screen arm's control (174 of 209) and the
screen arm itself, iteration by iteration and by stratum, above all the
harmonic and misranked roots.

### Results after four iterations (2026-10-17)

All three runs reached four iterations: the classifier arm (jobs 19956235
and 19996864), the screen arm (20013499) and the rerun (20028509; smoke
test 20015194 passed with KL 0 then 0.0023). The rerun's job was preempted
nine times and requeued each time without losing work; it was cancelled by
hand once its manifest reached iteration 4, because every requeue restarts
the job script's iteration loop and it had begun a fifth iteration
(`iteration_004` holds 30 partial groups). Greedy evaluation on the 209
hard development roots:

| Run | Control | Iteration 1 | 2 | 3 | 4 | Expert |
| --- | --- | --- | --- | --- | --- | --- |
| Classifier arm | 182 | 183 | 183 | 184 | 183 | 185 (baseline) |
| Screen arm | 174 | 173 | 170 | 170 | 177 | 193 (ranked ledger) |
| Rerun, stratified and KL-anchored | 174 | 173 | 176 | 167 | 171 | 193 (ranked ledger) |

No iteration of any run differs from its control at p < 0.05 (two-sided
sign test on the paired roots; the closest is the rerun's third iteration,
3 gains and 10 losses, p 0.09). The rerun's updates:

| Rerun iteration | Informative groups | Rows | Policy loss per step | Step gradient norm | Largest row norm | Rows clipped | KL per step |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 15 of 64 | 995 | −0.004, 0.022 | 3.7, 4.1 | 2.7, 97.5 | 2 | 0, 0.006 |
| 2 | 19 of 64 | 991 | −0.266, −0.016 | 4.0, 15.6 | 1.6, 14.8 | 0 | 0.012, 0.006 |
| 3 | 19 of 64 | 1,176 | −0.119, −0.096 | 4.7, 25.3 | 3.1, 412.4 | 1 | 0.041, 0.005 |
| 4 | 17 of 64 | 1,028 | −0.021, −0.090 | 17.5, 8.7 | 18.9, 6.6 | 0 | 0.041, 0.072 |

The strata that moved (every other stratum is identical in every column):

| Stratum | Roots | Control | Rerun 1 | 2 | 3 | 4 | Screen arm 4 | Ranked expert |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Harmonic | 16 | 12 | 16 | 16 | 8 | 11 | 12 | 16 |
| Mimic, opposite-sign | 24 | 19 | 16 | 17 | 19 | 18 | 20 | 20 |
| Mimic, same-sign | 24 | 14 | 13 | 13 | 13 | 13 | 15 | 20 |
| Multi-meter | 24 | 23 | 22 | 24 | 24 | 23 | 24 | 24 |
| Single meter | 24 | 24 | 24 | 24 | 22 | 24 | 24 | 24 |
| Meter plus parameter, misranked | 8 | 1 | 1 | 1 | 1 | 1 | 1 | 4 |
| Parameter, misranked | 9 | 1 | 1 | 1 | 1 | 1 | 1 | 5 |
| False commits | | 9 | 9 | 8 | 6 | 3 | 11 | |
| Invalid actions | | 42 | 47 | 37 | 57 | 42 | 56 | |
| Loop episodes | | 30 | 29 | 21 | 26 | 27 | 17 | |

What the three runs show:

- **The stratified mix did what it was built for, once.** With 12 harmonic
  roots per iteration, 6 of the first batch's 15 informative groups were
  harmonic, and one update took the harmonic roots from 12 to 16 of 16 with
  no invalid actions on them, against 11 of 16 for the screen arm's first
  iteration. The mix removed the sampling asymmetry of section 5.
- **The harmonic-versus-meter decision seesaws every update.** The first
  update that fixed harmonic broke five meter-type roots (the adapter asked
  for the harmonic or three-phase context on flow-meter mimics and repeated
  the request); the second won two back; the third dropped harmonic to 8 of
  16 with new failure modes (asking for every other context after the
  harmonic one instead of running the harmonic estimator, skipping the
  three-phase estimate); the fourth recovered to 11. The screen arm showed
  the same oscillation in the other order (12, 11, 5, 7, 12). In the screen
  evidence this choice is a near tie that each batch pushes one way.
- **The KL penalty did not hold it.** At `beta_kl` 0.04 the penalty is
  about 0.002 against policy losses near 0.1; the KL to the starting adapter
  grew from 0.006 to 0.07 across the four updates. As section 5 predicted,
  a brake of this size cannot pin a near-tie decision.
- **The misranked roots never moved**, in any run or iteration: 1 of 8 and
  1 of 9, the same as the baseline expert, against 4 of 8 and 5 of 9 for the
  ranked-ledger expert that reads the screen's refit ranking. The groups
  that fail these roots fail four times out of four, so they carry no
  gradient (section 4 of the plan, section 5 of the review). The flow-meter
  mimics stayed at or below the control throughout.
- **Efficiency moved more than success.** The classifier arm ended with
  invalid actions 29 against 51 and false commits 11 against 14; the rerun
  ended with false commits 3 against 9 and loop episodes 27 against 30, the
  screen arm with loops 17 against 30. Across the rerun's four evaluations
  161 roots were always solved, 30 never, and 18 flipped.

Next steps are the review's (section 8): advantages in reward units,
trajectory-level normalization, the clipped ratio for more than one
optimizer step, and a larger optimizer budget, with the stratified mix
kept; and for the misranked roots, rollouts that branch at the decision
state, since on-policy sampling of a peaked policy never produces the
reward spread those roots need. The pre-change arms (`rl_hard.json`,
`rl_hard_screen.json`) and the rerun are complete; nothing is queued.

## 6. What is not built

- The k-candidate handoff with length-discounted credit (plan step 2.3):
  the environment's bounded handoff still names at most two lines and the
  reward credits it at 0.6 whatever the list length.
- Compound strata (two parameter errors on adjacent lines, a meter and a
  parameter on the same line) and IEEE 118 roots are not in the hard draw;
  they need new generator overlays. The mimics come from the existing
  builder.

## 7. Verification

Run locally on 2026-10-15 (244 passed, 64 subtests); `psse_env/rl/test_sampling.py`
(2026-10-16) adds the stratified draw and pins the default draw to the
permutation the running arms use:

```bash
python -m pytest -q psse_env/rl research/rl psse_env/dagger/test_research_policy_factory.py psse_env/dagger/test_research_dagger_minimal.py psse_env/dagger/test_evaluator.py psse_env/sft/tests research/test_hpc_full_pipeline.py
```

`psse_env/rl/test_train_integration.py` rolls the expert out on two real
development roots through the driver's phases (skipped where the local
suite is absent). `psse_env/rl/test_rollout.py` drives the real evaluator
with a scripted environment; `psse_env/rl/test_update.py` checks the loss,
the temperature, the KL estimator and checkpoint resume on a tiny bigram
model; `psse_env/rl/test_peft_reference.py` checks the reference adapter on a
tiny GPT-2 with real PEFT adapters.

## 8. Review of the implementation (2026-10-16)

`docs/rl_implementation_review_20261016.md` reads the loop component by
component against the question of why the arms show no significant gain. In
short: the loop is a correct on-policy GRPO/RLOO implementation; the flat
results come from the absence of reward spread inside most groups, a budget of
four optimizer steps, the saturated metric and the group-standardized
advantage, not from the estimator, and switching to GAGPO would inherit all
four. The review added three instruments, all default-off or report-only:
`python -m psse_env.rl.diagnose --iteration <run>/iteration_NNN` (where the
signal of a stored iteration is and what it is made of), `on_policy_check` in
`update_report.json` (the gap between the recomputed and the recorded sampling
log-probabilities per optimizer step), and `update.clip_ratio` (the clipped
PPO surrogate against the recorded probabilities, for more than one optimizer
step per batch), and `update.loss_normalization: "trajectory"` (every
trajectory of an optimizer step weighs the same whatever its length, the
direct answer to the 1,014-token sample of section 5).

## 9. The estimator and data-contract rerun (prepared 2026-10-17)

A second, external review of the loop at `eeec327` (not in the repository;
its GAGPO sections are set aside here) agreed with section 8 on the
building blocks and the Dr. GRPO point, and added defects in the parts the
next rerun would rely on. Checked against the code:

| Claim | Finding |
| --- | --- |
| `clip_ratio` silently falls back to the plain loss on a row without a sampling record | Correct; it now refuses the batch (`policy_gradient_update`, `_policy_term`), and a clipped config builds rows with `require_sampling_records`. |
| Rows are shuffled individually, so under the `trajectory` normalization a trajectory split across optimizer steps does not get one weight | Correct; optimizer steps now hold whole trajectories, dealt round robin (`_chunks`). |
| Uninformative groups are dropped before the KL term, so the anchor never sees the states the policy already handles the same way every time | Correct; `rows.preservation` adds zero-advantage rows from uninformative groups, normalized in the KL term only (`_chunk_weights`). |
| The on-policy check reports but does not enforce; it averages over mostly near-certain tokens | Correct; `update.max_on_policy_log_ratio` refuses a first step beyond the tolerance before it is applied, and the check reports the decision tokens (recorded probability below 0.99) separately. |
| The row builder checks the re-rendered prompt's length, not its identity, and can truncate it again | Correct; the sampler now records a digest of the prompt token ids it decoded from (`psse_env/rl/prompt_digest.py`), `rows.require_prompt_digest` refuses a row without or with a different one, and `rows.reject_truncated_prompts` refuses update-side truncation. |
| The rollout may load the processor from the adapter directory while the update loads it from the base model | Not here: the adapters hold only the LoRA weights and config, so both load the base model's processor; the policy identity records `processor_source`. |
| Unpaired trajectories stay in the group baseline | `rows.require_complete_groups` skips such groups (none occurred after e622689). |
| The √3 bound on four-sample standardized advantages; RLOO at K = 4 is 4/3 of the centered GRPO advantage; zero KL at initialization is not a generation-parity test; the optimizer is re-created every iteration; expert scores are baselines, not ceilings | All correct, no code change. |

All defaults are unchanged: a config without a `rows` block and without the
new update fields trains exactly as before (`psse_env/rl/test_rows.py`,
`test_update.py`, `test_train.py`, `test_prompt_digest.py`).

**The rerun** (`rl_hard_screen_fix.json`, output `out/hard_screen_fix_v1`)
keeps the stratified mix, the KL penalty, four samples per root at
temperature 1.0, two optimizer steps at learning rate 1e-5 and the row
bound of 25, so that its control is the completed rerun of section 5
(same roots, same starting adapter, same budget). It changes:

| Setting | Section 5 rerun | This rerun | Why |
| --- | --- | --- | --- |
| `advantage_scale` | group | none | A one-call cost difference no longer weighs like a success flip (section 8, Dr. GRPO). |
| `update.loss_normalization` | dapo | trajectory | A 40-step looping sample weighs like a 6-step one; whole-trajectory optimizer steps make the weight exact. |
| `update.clip_ratio` | unset | 0.2 | The second optimizer step is a clipped PPO step against the recorded sampling probabilities instead of an uncorrected off-policy one. |
| `update.max_on_policy_log_ratio` | unset | 0.05 | The first step is refused unless the training pass reproduces the sampling pass to that tolerance. |
| `rows` | defaults | complete groups, prompt digest, no update-side truncation, sampling records required, preservation one trajectory per uninformative group, at most 24 groups | The KL anchor covers about 24 more states per iteration (about 220 rows, a fifth more update time); every trained row is exactly the sampled context. |

Not changed on purpose: the learning rate and the number of optimizer
steps. Section 8 recommends 3e-5 and four steps only after the scale and
normalization changes are seen to work, and the seesaw of section 5 shows
that two steps at 1e-5 already move the harmonic decision by whole
strata; raising the budget is the next factor, on its own.

Smoke test (`rl_hard_screen_fix_smoke.json`, output `out/smoke_fix`, 8
stratified roots, one iteration, no evaluation) and the launch:

```bash
cd /scratch/yx3882/rl_pilot_20261014 && PATH=/opt/slurm/bin:$PATH sbatch --export=ALL,CONFIG=rl_hard_screen_fix_smoke.json,ITERATIONS=1 --job-name=rl-smoke-fix source/research/hpc/rl_pilot_20261014/rl_pilot.sbatch
cd /scratch/yx3882/rl_pilot_20261014 && PATH=/opt/slurm/bin:$PATH sbatch --export=ALL,CONFIG=rl_hard_screen_fix.json,ITERATIONS=2 --job-name=rl-rl_hard_screen_fix source/research/hpc/rl_pilot_20261014/rl_pilot.sbatch
```

The smoke test passes when its update report shows every row with a
sampling record and a matching prompt digest (`rows_without_sampling_record_rejected`,
`prompt_digest_missing` and `prompt_digest_mismatches` all zero,
`prompt_truncated_rows_rejected` zero), preservation rows above zero, a
first-step `on_policy_check` mean absolute log-ratio under 0.05 with its
decision-token mean reported, a KL of zero at the first step and a
positive one at the second, and no row clipped by the ratio at the first
step (`tokens_beyond_threshold` zero).

The decision rule stays the one of section 5: paired gains and losses
against the control on the 209 roots, by stratum, above all harmonic
(does the seesaw stop?) and misranked (unchanged by every run so far).

### The smoke test measured the sampling-versus-training gap (2026-10-17)

Smoke job 20189365 (source 745fc7f) sampled the same 8 roots as the
section 5 smoke test (identical rollouts, so the digest recording changed
nothing in the sampler), built 188 rows (153 policy rows, 35 preservation
rows, no digest, truncation or record rejections) and then refused its
first optimizer step: over the step's rows the mean absolute log-ratio
between the recomputed and the recorded sampling log-probabilities was
0.053 against the limit of 0.05, and over the 179 decision tokens
(recorded probability below 0.99) it was 0.56, with single tokens off by
up to 31 nats. The decision tokens are the tool names: `get` at 0.0055 and
`harmonic` at 0.92 inside a `get_harmonic_context` call, `topology` at
0.017, `measurement` at 0.61, `three` at 0.32; the formatting tokens around
them are near-certain and agree to 1e-3. So the training pass reproduces
the sampling pass everywhere except at the few tokens that carry the
decision, where it is systematically sharper: a token the sampler drew at
5e-4 can come back at 1e-17.

Two passes differ: the sampler decodes with the KV cache in bf16 through
the inference loader, the update runs a full-sequence forward through the
4-bit training preparation (norms and embeddings in float32, gradient
checkpointing). Which of the two causes the gap at the decision tokens is
not yet attributed; a GPU diagnostic that scores the stored smoke rows
under both loaders, with and without the cache, is the next measurement.
Every earlier run trained on this gap without knowing it: the plain loss
takes the training pass's gradient on tokens the sampler drew from a
flatter distribution, which is an importance-sampling bias of the whole
method, not of the clip.

Two consequences for the rerun, deployed as the follow-up commit:

- `update.clip_reference: "training"`. The clipped ratio is taken against
  the training pass itself before the first optimizer step (one extra
  forward per row, persisted with the checkpoint as `old_logprobs.pt`), as
  verl and TRL do, so the first step has a ratio of exactly one and the
  second clips against the true pre-update policy. Against the sampler's
  record, a decision token off by 0.56 nats was clipped or scaled by a
  factor of 1.75 at the first step for a numerical reason, and a token off
  by 31 nats would have been scaled by a factor of 10^13 when its
  advantage was negative. The sampler's record now feeds the on-policy
  check only; the update report adds `clip_check`, the same statistics
  against the clip baseline, whose first step must read zero.
- `update.max_on_policy_log_ratio: 0.2`. The guard keeps its purpose, a
  broken template or tokenizer gives gaps of several nats, but no longer
  trips on the measured 0.05.

The rerun job 20189366 was sampling when this was found; its update phase
is a separate process that reads the source and the config when it starts,
so the redeploy reached it before its first update.

### Results after four iterations (job 20189366, completed 2026-10-19 06:15 EDT)

One Slurm job carried the run through 20 preemptions and about 37 hours
of wall time (10.5 hours of GPU time in its last, uninterrupted
incarnation on gr104); the baseline-pass checkpoint of commit 3279c2c
was needed on the way, and the sampler of every iteration carried the
prefill defect of section 10, as the control did. The four updates:

| Iteration | Informative groups | Rows (policy + preservation) | Policy loss (step 1, 2) | Step gradient norm | Largest row norm, clipped | KL to the start | Sampler gap, all / decision tokens (step 1; step 2) | Clip-baseline gap, beyond (step 1; step 2) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 15 of 64 (23%) | 1,232 = 995 + 237 | −0.039, 0.035 | 3.41, 5.24 | 2.28 / 2.12, none | 0.0, 0.072 | 0.053 / 1.49; 0.067 / 0.97 | 0.0015, 19 (cross-node); 0.0084, 122 |
| 2 | 17 (27%) | 1,178 = 985 + 193 | −0.0048, 0.0075 | 2.57, 1.94 | 1.86 / 1.82, none | 0.0072, 0.023 | 0.098 / 1.10; 0.113 / 0.97 | 0, 0; 0.0081, 90 |
| 3 | 20 (31%) | 1,311 = 1,088 + 223 | −0.065, 0.063 | 4.72, 4.18 | 3.31 / 2.48, none | 0.0073, 0.011 | 0.071 / 1.54; 0.088 / 1.23 | 0.0023, 44 (cross-node); 0.0069, 87 |
| 4 | 13 (20%) | 1,062 = 845 + 217 | 0.0047, −0.0038 | 26.1, 4.92 | 55.4 / 3.01, one | 0.0073, 0.0043 | 0.104 / 1.31; 0.084 / 1.37 | 0, 0; 0.0057, 78 |

Every row of every update carried a matching prompt digest and a
sampling record (no rejections), the preservation rows came from 24
uninformative groups each time, the clip baseline was exact wherever
the baseline and the step ran on one node, and the KL to the starting
adapter stayed below 0.08 throughout. The sampler gap of 0.05 to 0.10
with decision tokens at 1 to 1.5 nats is the prefill defect of section
10, not an estimator property.

Greedy evaluation on the 209 development roots, with the control (the
frozen starting adapter), the previous stratified rerun of section 5 and
the two experts:

| | it 1 | it 2 | it 3 | it 4 | Control | Previous rerun it 4 | Expert | Ranked expert |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Solved | 171 | 176 | 175 | **178** | 174 | 171 | 185 | 193 |
| harmonic (16) | 14 | 15 | 14 | **16** | 12 | 11 | | |
| mimic same sign (24) | 13 | 14 | 14 | **17** | 14 | 13 | | |
| mimic opposite sign (24) | 16 | 17 | 18 | 16 | 19 | 18 | | |
| multi-measurement (24) | 23 | 24 | 24 | 24 | 23 | 23 | | |
| measurement (24) | 23 | 24 | 23 | 23 | 24 | 24 | | |
| misranked, m+p (8) / p (9) | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 | | |
| other strata (80) | 80 | 80 | 80 | 80 | 80 | 80 | | |
| Invalid actions | 60 | 40 | 56 | 42 | 42 | 42 | | |
| False commits | 10 | 8 | 10 | 9 | 9 | 3 | | |
| Loop episodes | 33 | 24 | 26 | 27 | 30 | 27 | | |

The previous rerun's iterations scored 173, 176, 167, 171. The fourth
adapter is four above the control and seven above the previous rerun's
last, the first run of the series to end above its control, with the
whole gain on two strata: harmonic goes to 16 of 16 with no invalid
action (the control loops on four of them), and the same-sign mimics
gain three. The opposite-sign mimics lose three (the policy answers
them with the harmonic context and loops where the control corrects the
measurement), one measurement root is lost the same way, and the
misranked roots never move, as in every run. 168 roots are solved by
all four adapters and 26 by none; four control-solved roots are lost at
the end.

The run also shows what the estimator fixes bought and what they did
not. The reward-unit advantages and the trajectory normalization kept
the per-row norms in the single digits (one row clipped in four
updates, against two with a largest norm of 97 in the previous rerun's
first update), and the KL penalty held the policy within 0.08 nats of
its start; the harmonic path was learned in the first update and never
unlearned, where the previous rerun lost it in its third. What the
fixes did not change is the seesaw on the mimic strata: the
three-phase and harmonic requests, reinforced on the harmonic roots,
leak onto mimic prompts in one iteration and retreat in the next
(three-phase requests on 14, 9, 10 and 10 of the 48 mimic roots against
the control's 9), so the mimic score moves by ±3 between iterations
without a trend, and the run-to-run spread of the whole score is still
about ±3. The fourth evaluation is one draw of that spread, taken with
a sampler that is wrong on one prompt in eight; the comparison that
counts is the one under the repaired sampler, where the control itself
has to be re-measured first.

## 10. The bfloat16 loader (prepared 2026-10-18)

The third update of the fix rerun measured where the time goes. Its batch
is 1,311 rows (one per sampled step); each row's prompt is the re-rendered
state view, 5,672 tokens on average (median 5,839, p90 7,576, max 10,597,
none truncated), and its completion is the tool call, about 23 tokens, so
0.4% of the tokens pushed through the model carry loss. Every row is
processed five times: the clip baseline's forward, the KL reference's
forward under the starting adapter, the training forward, the gradient
checkpoint's recompute and the backward; about 37 million token passes
through the 12B model per update.

On an A100-SXM4-80GB the baseline's forward took 3.3 s per row (18 rows a
minute) and a training row 17 s (3.5 rows a minute), about 7.4 h of GPU
time for the update, delivered in 30-minute preemption slots of which
eight minutes load the model. A forward over 5.7k tokens of a 12B model
is about 1.6e14 FLOPs, 0.5 s at the card's 312 TFLOP/s, so the forward
runs at about 15% of peak and the training pass at about 12%. The card
holds 15 GB of its 80 GB. The stack, not the arithmetic, sets the pace:

- The base weights are 4-bit NF4 (bitsandbytes), dequantized inside every
  linear layer of every one of the five passes, the known two-to-three-fold
  penalty of QLoRA over bfloat16 LoRA on long sequences, paid here for
  memory the card does not need. The sampler loads the same quantization,
  and its training preparation (norms and embeddings in float32) is the
  loader difference behind the sampling-versus-training gap of section 9.
- Gradient checkpointing adds the recompute pass, a quarter of the
  training cost, again a memory trade.
- One row at a time: at 5.7k tokens a row saturates the card, so batching
  gains little, but attention is the default implementation.
- A100 rather than H100/H200 nodes (two to three times), and the
  preemption cadence.

A bfloat16 run of the same estimator at 40% of peak would take about 5 s
per training row, two hours per update on an A100, one on an H100; the
rest of the cost is the estimator itself, which back-propagates through a
5.7k-token prompt for 23 loss tokens. Running the prompt forward once
without gradient (a prefix cache) and training the completion positions
only would be about five times cheaper per row but drops the gradient
through the prompt-side adapter contributions, a different estimator and
a research decision rather than a tuning.

The prepared change is the loader only: `rl_hard_screen_bf16.json` is the
fix config with `load_in_4bit: false` and the output `out/hard_screen_bf16_v1`;
nothing else differs. The switch is already wired: the trainer's
`_load_research_base` loads `dtype=bfloat16` without a quantization config
and `attach_research_adapter` skips `prepare_model_for_kbit_training`, the
sampler's loader drops its `BitsAndBytesConfig` and keeps `dtype=bfloat16`,
gradient checkpointing stays on (24 GB of weights plus checkpointed
activations of a 10k-token row fit the 48 GB and 80 GB cards; two sampler
workers of 24 GB plus their caches fit an 80 GB card, and the job script's
70,000 MiB rule keeps one worker on smaller cards). The LoRA weights stay
float32 under PEFT's adapter autocast in both loaders. Expected: two to
three times the rows per minute in both passes, and with both loaders in
bfloat16 a smaller decision-token gap in the on-policy check; the smoke
config `rl_hard_screen_bf16_smoke.json` (8 roots, no evaluation, output
`out/smoke_bf16`) measures both before the full run. The fix rerun keeps
its 4-bit loader to the end: changing the loader under a run would change
the clip baseline and the KL reference mid-way.

Deploy and smoke, once the fix rerun is done or the user decides to run
both:

```bash
SUBMIT=0 bash research/hpc/rl_pilot_20261014/deploy_from_windows.sh "$(git rev-parse HEAD)" rl_hard_screen_bf16.json rl_hard_screen_bf16_smoke.json
```

```bash
wsl -- ssh torch "cd /scratch/yx3882/rl_pilot_20261014 && PATH=/opt/slurm/bin:\$PATH sbatch --export=ALL,CONFIG=rl_hard_screen_bf16_smoke.json,ITERATIONS=1 --job-name=rl-smoke-bf16 source/research/hpc/rl_pilot_20261014/rl_pilot.sbatch"
```

### The bfloat16 smoke test (job 20270602, 2026-10-18)

Run beside the fix rerun on gr103, an RTX PRO 6000 Blackwell (96 GB):
40.5 minutes in all, no Traceback, every criterion of section 9 met. Two
sampler workers took 57 GB; the update process 33 GB.

| | bf16 smoke (167 rows) | 4-bit fix smoke (188 rows) |
| --- | --- | --- |
| Rollout, 8 roots × 4 | 16.8 min, two workers | stored rollout reused |
| Rows | 132 policy + 35 preservation, zero rejections | 153 + 35, zero rejections |
| Losses, step gradient norms | (−0.0225, 0.0226), (2.09, 50.0) | (−0.0575, 0.0407), (1.22, 24.9) |
| Largest row norm, clipped | 2.13 / 90.1, 2 clipped at step 2 | 1.16 / 112.1, 1 clipped |
| KL to the starting adapter | 0.0, 1.32 | 0.0, 0.73 |
| Clip baseline (training pass) | step 1 exactly 0; step 2 0.0117, 13 of 1,849 tokens beyond, max 7.0 | step 1 exactly 0; step 2 0.0116, 28 of 2,978 beyond, max 7.4 |
| Sampler gap, all tokens | 0.048, 0.061 | 0.053, 0.092 |
| Sampler gap, decision tokens (p < 0.99) | **19** tokens, mean **4.52** nats, max 22.7; step 2: 18 tokens, mean 6.0 | **179** tokens, mean **0.56**, max 31.4 |
| Update time | 1,327 s = **7.9 s per row** | 2,005 s = 10.7 s per row |

Two results. The speed gain is real but smaller than the dequantization
argument predicted: 7.9 s per row against 10.7, and against the 4-bit
rerun's own pace on the same node class that evening (rows 160 to 320 of
its third update on gr106 in about 29 minutes, 5.5 to 6 rows a minute,
against the smoke's 8.9) about 1.5 times. The Blackwell cards are
themselves about 1.7 times the A100s (the 4-bit rerun did 3.5 rows a
minute there). So the five passes over a 5.7k-token prompt, the
checkpoint recompute and the default attention carry the cost, not the
4-bit weights.

The parity result changes the attribution of section 9. With both
loaders in bfloat16 the overall gap is unchanged, but the sampler is far
sharper: 19 of 1,778 tokens were drawn at a probability below 0.99
against 179 under 4-bit, so the quantization was flattening the sampling
distribution. On those 19 tokens the training pass disagrees with the
sampler's record by 4.5 nats on average, 22.7 at most, and 4 tokens fall
outside the clip band. The decision-token gap therefore does not come
from the 4-bit weights; it sits in the generation path (the KV-cache
decode) against the full forward, or in the numerics of small
probabilities. The clipped ratio against the training-pass baseline
makes the update indifferent to it, but the sampled tokens still come
from a distribution the update cannot reproduce, which is the
importance-sampling bias noted in section 9. The GPU parity diagnostic
(the stored smoke rows scored under both loaders with and without the
cache, on one card) is now the measurement that attributes it.

### The parity diagnostic (job 20276081, 2026-10-18)

`psse_env.rl.parity` rebuilt the rows of both smoke rollouts as the update
builds them and scored every completion token four ways on one card: the
sampler's record, `generate`'s cached decode path teacher-forced through
the recorded completion (the raw logits of each step), the full-sequence
forward of the sampling loader (the update's `completion_logprobs`) and
the full-sequence forward of the training loader (the clip baseline).
Mean and largest absolute gap in nats, and the number of tokens beyond
the clip band of ln 1.2:

| Pair | bf16 smoke (167 rows, 3,627 tokens) | 4-bit smoke (188 rows, 4,878 tokens) |
| --- | --- | --- |
| record vs decode | **0.0000**, max 0.000, 0 beyond | **0.0000**, max 0.000, 0 beyond |
| forward vs training | **0.0000**, max 0.000, 0 beyond | 0.0117, max 3.44, 53 beyond |
| decode vs forward | 0.0516, max **22.7**, 10 beyond | 0.0732, max **32.4**, 232 beyond |
| decode vs forward, tokens at p ≥ 0.99 | 0.0014, max 5.06 | 0.0003, max 0.67 |
| decode vs forward, tokens at 0.5 ≤ p < 0.99 | 4.04, max 22.7 (25 tokens) | 0.24, max 32.4 (387 tokens) |
| decode vs forward, tokens at 0.01 ≤ p < 0.5 | 6.75, max 21.7 (12 tokens) | 0.86, max 26.8 (233 tokens) |

Three facts follow. The forced decode reproduces the sampler's record to
the last digit, so the record is exactly what `generate`'s path computes
and sampling-time nondeterminism is nil. The two loaders agree exactly in
bfloat16 and differ only mildly under 4-bit (the training preparation's
float32 norms and embeddings), so the loader is not the gap. The whole
gap lies between the cached decode path and the full-sequence forward on
identical weights, and it is not spread over the tokens: in bfloat16 it
sits on the first completion token of 8 of 167 rows (five distinct
prompts, 4.6k to 7.7k tokens long, in the normal range), where the first
generated token, `<channel|>` in six of them, has a log-probability of
−0.03 to −3.2 under the decode path and −15 to −24 under the full
forward; the 4-bit run shows the same at position 0 of its worst rows
(`<channel|>`, `1`, `thought`, `Architect`, `.`, at −24 to −33 against
−0.4 to −7.5) with a little spill into the next positions. On the other
rows the two paths agree to a thousandth of a nat.

A causal model's logits at the last prompt position cannot depend on the
completion appended after it, and the prefill of `generate` over the
prompt alone is that same computation, so one of the two paths is not
computing the model's causal distribution for those prompts. Which one is
the question the probe of the next subsection answers, and the answer
decides the fix: if the full forward leaks the completion backwards (the
mask), the update has been training on corrupted log-probabilities and
the forward must be repaired; if the cached prefill is the deviating
path, the policy samples from a distribution that is not the model's and
the sampler must be repaired. Either way the sampled tokens of the hard
rows came from a distribution the update cannot reproduce, which is the
bias section 9 named.

### The probe: a kernel defect in the sampler's prefill (jobs 20279093, 20280069, 20281182)

`psse_env.rl.parity_probe` recomputed the first completion token's
log-probability of the eight worst bfloat16 rows and of two agreeing
controls under computations that differ in one thing each. The bad rows
all behave alike; one of them (r0_28ad…, step 16, first token
`<channel|>`, prompt 4,609 tokens):

| Computation | log-probability |
| --- | --- |
| the sampler's record; the prompt alone with or without the cache; `generate` for one token | −0.105 |
| the full sequence (the update's computation), with all logits, without side inputs, without an attention mask | −21.39 |
| the full sequence with the completion's attention mask zeroed; the prompt plus pad tokens, random tokens, or its first 1, 2, 4 or 8 completion tokens | −21.39 |
| the prompt alone under eager attention; the full sequence under eager attention | −21.90, −22.34 |
| the prompt alone under SDPA with the math backend forced; with the memory-efficient backend forced | −21.36, **−0.105** |
| the prompt with one left pad token under a zero attention mask, through the forward and through `generate` | −21.48, −21.30 |

So every computation agrees on about −21 except the SDPA forward of the
exact prompt with the memory-efficient backend, which is what `generate`
prefills with. Across both parity runs the affected rows are exactly the
rows whose prompt length is 1 modulo 8 (bfloat16: 8 of 34 such rows show
a gap above a nat, the other 26 have deterministic first tokens; 4-bit: 8
of 29), and no row with another residue shows any gap (mean largest gap
0.000 over 133 and 159 rows). The decode steps are unaffected: on the
rows with another residue, the gaps at later positions are zero at every
residue of the cache length. The defect is therefore the memory-efficient
SDPA kernel of PyTorch 2.10 corrupting the last query row of a
bfloat16 prefill under the materialized mask Gemma 4 builds whenever the
query length is 1 modulo 8 — the row `generate` draws the first token
from, on one prompt in eight. The training forward never uses the last
row of its sequence (it would predict the token after the completion),
which is why the update's log-probabilities and clip baseline were right
and why the full-sequence length showed no pattern.

Consequences. The sampler, not the update, was the deviating path: every
rollout, every greedy evaluation and the DAgger data collection drew the
first token of about an eighth of their steps from a corrupted
distribution (the following tokens are conditioned on the corrupted
hidden state of that position, mildly), and the scores of sections 5 and
9 measure the model together with this quirk. The "decision-token gap"
of section 9 was this defect, not the quantization or the loader, and
the importance-sampling bias it implied is an artefact of the sampler.

The cure is the one the probe validated: `generate_text` now puts one
pad token with a zero attention mask before any prompt whose length is 1
modulo 8 (`prefill_pad_tokens`, `left_pad_model_inputs` in
`psse_env/dagger/research_policy_factory.py`; `PSSE_SDPA_PREFILL_PAD=0`
switches it off for measurements), which moves the prefill to the next
residue and reproduces the math kernel; the record's prompt digest, the
prompt length and the completion slice refer to the unpadded prompt, so
the update's rows are unchanged. The action metrics carry
`prefill_pad_tokens`. The fix changes the policy's behaviour on the
affected prompts, so the control and the experts' student comparisons
must be re-measured under it before any run is compared with the
numbers above; the fix rerun keeps the old sampler to the end of its
four iterations.

### The controls under the repaired sampler (jobs 20330002 and 20330086, 2026-10-19)

Both frozen controls were re-measured with the prefill pad deployed
(source d2206f3), each as a run seeded at iteration 0, phase `evaluate`,
into a new directory beside the old result:

| Control | Old (defective prefill) | New (padded prefill) | Roots whose outcome changed | Roots whose tool sequence changed |
| --- | --- | --- | --- | --- |
| screen cell's round-2 adapter | 174 of 209; 42 invalid, 9 false commits, 30 loops | 174; 44 invalid, 9 false commits, 31 loops | **0** | 3 |
| classifier cell's round-2 adapter | 182; 51 invalid, 14 false commits, 7 loops | 182; 52 invalid, 14 false commits, 7 loops | **0** | 1 |

Every stratum count is identical to the old evaluation in both runs; the
three screen roots and the one classifier root whose sequences changed
took a different route to the same outcome (one extra invalid step or
context request). So the defect, which moves the first token's
distribution by twenty nats on prompts of length 1 modulo 8, almost
never changes the tool that greedy decoding ends up calling: on those
prompts the corrupted row mostly decides the channel preamble, not the
call. The historical greedy scores of sections 5 and 9 therefore stand.
Where the defect did matter is the sampled rollouts: the recorded
probabilities of one step in six to eight were those of the corrupted
row, which is the whole "decision-token gap" of the on-policy check, and
the policy gradient of every run was taken from samples drawn from that
slightly wrong distribution while the log-probabilities and the clip
baseline came from the correct training pass — a mild off-policy bias on
the affected steps, not a corruption of the rewards. A clean RL
comparison still needs fresh rollouts under the repaired sampler, but
the evaluations do not need redoing.

## 11. The decision-opportunity audit (Gate 2, 2026-10-19)

`research/rl/decision_audit.py` (plan: `docs/decision_audit_plan_20261019.md`)
rolled the observable ledger-ranked expert out on the 176 training roots of
the failure-prone strata under the suspicion-gated profile, found the
decision states of each trajectory (a rule-expert choice among at least two
admissible actions of its own observable menu), replayed the reference
actions on fresh episodes to take up to three alternatives at up to four
states per root with the expert finishing, and let the evaluator score every
branch as the runs are scored. At the first correction decision a
truth-guided probe applied every true target with the expert's lifecycle in
between. 176 roots, 2,205 branches, 53 minutes on four local cores, no
errors (`docs/decision_audit_20261019/train176_summary.md`).

| Stratum | Roots | Expert solves | rule_best | legal_alternative | evidence_limit | Truth probe solves |
| --- | --- | --- | --- | --- | --- | --- |
| parameter, ambiguous / dominant | 40 / 8 | 48 | 48 | 0 | 0 | 48 / 48 |
| measurement+parameter, ambiguous / dominant | 32 / 8 | 40 | 40 | 0 | 0 | 40 / 40 |
| misranked, parameter / m+p | 4 / 4 | 4 | 4 | **4** | 0 | 6 / 8 |
| mimic, opposite sign | 24 | 18 | 15 | **8** (5 failures recovered, 3 cheaper) | 1 | 21 / 24 |
| mimic, same sign | 24 | 22 | 22 | **1** | 1 | 22 / 24 |
| multi-measurement | 32 | 31 | 29 | **3** (1 recovered, 2 cheaper) | 0 | 31 / 32 |
| **all** | 176 | 163 | 158 | 16 | 2 | — |

No state fell in `interface_limit`: wherever the reference and every legal
alternative failed, the truth probe failed too, so the action interface
(singleton corrections, grouped meter corrections, the three contexts, the
acquisitions, the handoff) is not what is missing on this draw.

Of the expert's **13 failures, 11 have a legal alternative at a state the
expert visits that leads to a truth-audited success**, and 2 are evidence
limits (two mimics where no correction, true or not, passes the audit). The
alternatives are of two kinds:

- **Misranked parameter roots (4 of 4 failures).** At the correction
  decision the expert takes the parameter context's top line and the
  **runner-up line solves the root** in 6 to 10 steps; the expert's commit of
  the top line ends in a handoff. The dominance ratios are 1.04, 1.07, 1.18
  and 1.32, so "runner-up when the ratio is under the 1.2 threshold" is not
  the rule: one failure sits above it, and 71 ambiguous-ratio roots are
  solved by the top line. Whether the context's own scores separate the
  cases is the predictability question.
- **Mimics and multi-meter roots (7 failures).** At the first meter
  correction the expert takes the residual-ranked flow meter and the
  **correction of a different meter** (a third- or fourth-ranked channel)
  solves the root; on the three cheaper opposite-sign roots the same
  reordering saves six of eighteen steps.

On the 120 ambiguous and dominant parameter roots the top line is right
every time and no alternative beats the rule, which is the baseline the
predictability test must beat: a ranker that reorders the top two lines
helps on the 4 misranked failures and must not cost any of the 120.

**Decision.** Gate 2 passes for the ordering decision, narrowly: the
headroom on the training draw is 11 roots of 176, all of it in which
hypothesis to verify first, none of it in the interface and little in the
evidence. The next measurement is the predictability test of the plan
(phase 3): a ranker over each candidate's observable features (its abs-λ
score, ratio to the top, context rank, the dominance ratio and route
status; for meters the normalized residuals of the findings), trained with
physical parents held out, against the fixed rule — then the same audit
from the student's own recorded states (phase 2), since the RL runs' losses
are the student's deviations, not the expert's.
