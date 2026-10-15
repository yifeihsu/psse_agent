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

## 4. Building the hard draw

The suite builder reverses the admission filters when asked. Example for a
stratified development draw with the plan's parameter strata and an effect
band in place of the 10-sigma floor:

```bash
python research/hpc/full_pipeline_20260907/build_suite.py --source-root . --d0-raw <d0/aggregate.raw.jsonl> --protected-suite <classifier development.json> --round-train-plan "$ROUND_TRAIN_PLAN" --development-plan "$DEVELOPMENT_PLAN" --rounds 2 --seed 20261014 --admission-mode physical --development-rank-allowance 99 --training-rank-allowance 99 --effect-band 1.5,3 --development-stratum-plan '{"parameter": {"misranked": 24, "ambiguous": 24, "dominant": 8}}' --training-stratum-plan '{"parameter": {"misranked": 48, "ambiguous": 48, "dominant": 16}}' --output-dir output/rl_hard_draw_20261014/suite
```

Misranked roots are 8 of 250 in the population, so the candidate multiplier
or the requested count must be large enough; the manifest records
`stratum_shortfalls` when a stratum could not be filled. Then tag and score:

```bash
python -m research.rl.identifiability --suite output/rl_hard_draw_20261014/suite/development.json --output-dir output/rl_hard_draw_20261014/identifiability
python -m research.classifier_triage.expert_e2e --suite output/rl_hard_draw_20261014/suite/development.json --profile classifier_gated_diagnostics --output-dir output/rl_hard_draw_20261014/e2e_baseline
```

The tagger's `separable_fraction_of_balanced_roots` is the attainable
ceiling; the expert run is the floor the RL arm must clear.

## 5. What is not built

- The k-candidate handoff with length-discounted credit (plan step 2.3):
  the environment's bounded handoff still names at most two lines and the
  reward credits it at 0.6 whatever the list length.
- The mimic, compound and IEEE 118 strata of the hard draw are drawn with
  the existing builders; no new generator overlay was added.

## 6. Verification

Run locally on 2026-10-15 (244 passed, 64 subtests):

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
