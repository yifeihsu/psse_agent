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

The RL run on the hard draw is configured (`research/hpc/rl_pilot_20261014/rl_hard.json`,
output `out/hard_v1`, 64 roots per iteration) and not submitted.

## 5. What is not built

- The k-candidate handoff with length-discounted credit (plan step 2.3):
  the environment's bounded handoff still names at most two lines and the
  reward credits it at 0.6 whatever the list length.
- Compound strata (two parameter errors on adjacent lines, a meter and a
  parameter on the same line) and IEEE 118 roots are not in the hard draw;
  they need new generator overlays. The mimics come from the existing
  builder.

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
