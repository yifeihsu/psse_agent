# RL experiments: what was built and how to run them (2026-10-14)

Branch `codex/classifier-triage-20261004`. Implements the plan of
`docs/rl_hard_cases_plan_20261013.md` (steps 1 to 4) and the framework
decision of 2026-10-14 (an in-repo on-policy loop rather than TRL's or
verl's agent loops, which assume an append-only transcript the policy here
does not observe). Nothing has run on the cluster yet; the local checks are
listed in section 5.

## 1. What exists now

| Piece | Where | Plan step |
| --- | --- | --- |
| Graded episode reward: basis credit (decisive 1.0, bounded handoff 0.6, unexplained handoff 0.5, budget exhaustion 0.4), per-call and unneeded-acquisition costs, tool regret against the expert's calls on the same root, penalties for false commits, false finalizations, a healthy edit, a loop and invalid actions; predeclared weights | `psse_env/rl/reward.py` | 2 |
| Group-relative advantage (GRPO or RLOO), zero-variance groups dropped and counted | `psse_env/rl/advantages.py` | 3 |
| Sampled decoding in the research policy with a per-step record of the model observation and the exact completion tokens (`GenerationOptions`, `last_generation`, `render_prompt`, `prompt_token_ids`); greedy decoding unchanged by default | `psse_env/dagger/research_policy_factory.py` | 3 |
| Grouped rollouts: K seeded episodes per root through the closed-loop evaluator, paired with the trace and scored; a per-root JSONL store that resumes | `psse_env/rl/rollout.py` | 3 |
| Exact on-policy training rows: prompt re-rendered by the policy's own renderer and checked against the recorded length, completion = the sampled tokens, assistant-only mask, zero side inputs | `psse_env/rl/rows.py` | 3 |
| The LoRA update: one on-policy gradient pass (ratio one), DAPO token normalization, k3 KL to the reference adapter through `disable_adapter()`, completion-only logits via `logits_to_keep` | `psse_env/rl/update.py` | 3 |
| The iteration driver: rollout, update, evaluate phases with a manifest and atomic adapter writes; `policy_source: expert` for a CPU wiring check | `psse_env/rl/train.py` | 3, 4 |
| Hard-draw plumbing in the suite builder: training rank allowance, stratum plans per draw, effect band threaded into the research generator | `research/hpc/full_pipeline_20260907/build_suite.py`, `scripts/run_dagger_research.py` | 1 |
| Identifiability tagger: per root, which candidates the deployment verifier accepts (true and top-ranked wrong), tag and per-stratum summary | `research/rl/identifiability.py` | 1.3 |
| HPC cell: job script with per-phase processes and requeue resume, deploy script, pilot config, README | `research/hpc/rl_pilot_20261014/` | 4 |

Design points worth knowing before reading the code:

- The reward is computed only from the evaluator's episode record, after
  the episode ended; nothing truth-derived enters the policy's view.
- A trajectory whose generation records do not pair one-to-one with the
  evaluator's policy steps is kept for its reward but excluded from training
  (`pairing_ok`). The expert-sourced rollouts are such trajectories by
  construction.
- The update is on-policy with one optimizer pass per rollout batch, so no
  log-probabilities are stored and no clipping is needed; `optimizer_steps`
  only partitions the batch. The KL term uses the adapter-disabled base as
  the reference, so the reference is the R2 adapter's base behaviour, not
  R2 itself; set `beta_kl` to 0 to drop it.
- Rollouts and the update are separate model loads (the cached research
  bundle for sampling, the trainable 4-bit base plus adapter for the update),
  so the two phases can run as separate processes and the next iteration's
  rollouts load the saved adapter.

## 2. Running the pilot on the cluster

1. Build the expert reference for tool regret (already present locally as
   `output/classifier_triage_20261013/e2e_ieee14_spectra_first/episodes.jsonl`).
2. Deploy and submit (SSH master up):

```bash
bash research/hpc/rl_pilot_20261014/deploy_from_windows.sh "$(git rev-parse HEAD)"
```

3. Status:

```bash
wsl -- ssh torch "cat /scratch/yx3882/rl_pilot_20261014/out/pilot/manifest.json"
```

The pilot config rolls out the 122 round-2 roots of the classifier cell in
halves (61 per iteration, four samples each), updates with two optimizer
steps, and evaluates greedily on the 160 development roots every iteration.
Expected cost per iteration on one H100 or H200, from the measured rates:
about 1 h of rollouts and 2 to 3 h of training before zero-advantage
pruning. On the standard draw most groups will be uninformative (the
measurement the plan asks for first); the hard draw is what gives the loop
a signal.

## 3. Building the hard draw

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

## 4. What is not built

- The k-candidate handoff with length-discounted credit (plan step 2.3):
  the environment's bounded handoff still names at most two lines and the
  reward credits it at 0.6 whatever the list length.
- The mimic, compound and IEEE 118 strata of the hard draw are drawn with
  the existing builders; no new generator overlay was added.
- The cluster smoke test (sampling at temperature 0.8 through the real
  model, one iteration on three roots) has not run.

## 5. Verification

Unit tests, run locally on 2026-10-14:

```bash
python -m pytest -q psse_env/rl research/rl/test_hard_draw.py psse_env/dagger/test_research_policy_factory.py research/test_hpc_full_pipeline.py
```

`psse_env/rl/test_train_integration.py` rolls the expert out on two real
development roots through the driver's phases (skipped where the local
suite is absent). `psse_env/rl/test_rollout.py` drives the real evaluator
with a scripted environment; `psse_env/rl/test_update.py` checks the loss
on a tiny bigram model, including the fallback without `logits_to_keep`.
