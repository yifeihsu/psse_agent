# Small RL review experiments (prepared, not submitted)

Decision and evidence: `docs/rl_next_experiment_review_20261015.md`.
These configs do not modify or submit jobs 19932222/19932223. Use a committed,
isolated checkout; do not replace the source of an active job or resume its v2
update checkpoint with this v3 updater. The old four configs explicitly opt
into historical multi-step behavior; the new configs all take one update.

## First inspect existing work

On an already authorized NYU session, inspect `squeue -j 19932222,19932223`,
`sacct -j 19932222,19932223 --format=JobID,State,Elapsed,ExitCode`, and the
manifests/results under:

- `/scratch/yx3882/rl_pilot_20261014/out/hard_v1`
- `/scratch/yx3882/rl_pilot_20261014/out/hard_r2_baseline`

GitHub only records submission; absence from `squeue` does not establish
completion. Do not submit the existing deploy script as a status check.

Audit every completed iteration separately, for example:

```bash
python -m research.rl.signal_report --store /scratch/yx3882/rl_pilot_20261014/out/hard_v1/iteration_000/trajectories --suite /scratch/yx3882/rl_pilot_20261014/data/hard_draw_20261015/hard_training.json --output /scratch/yx3882/rl_pilot_20261014/out/hard_v1/iteration_000/signal_review.json
```

An incomplete iteration is useful for monitoring but is not a final rate.
`--greedy-evaluation` accepts `evaluation_episodes.jsonl` on the **same training
roots**; the 209-root development baseline cannot fill that role. Reports
distinguish missing comparisons from observed zero improvements. No new GPU
call is necessary to audit saved trajectories or compare reward weights.

## Freeze a training-only probe

```bash
python -m research.rl.prepare_probe --training-suite /scratch/yx3882/rl_pilot_20261014/data/hard_draw_20261015/hard_training.json --held-out-suite /scratch/yx3882/rl_pilot_20261014/data/hard_draw_20261015/hard_development.json --output /scratch/yx3882/rl_pilot_20261014/data/rl_review_20261015/training32.json
```

Add `--held-out-suite` for the old frozen evaluation suites when available.
The script refuses root/fingerprint/available-parent overlap, shortages and
overwriting an existing selection. It explicitly reports missing parent
lineage; this is not a guarantee of parent separation when source IDs are
absent. Reconcile that before a formal generalization claim.

The selection contains 8 misranked, 8 same-sign mimic, 8 opposite-sign mimic,
8 control roots. Selection uses audit strata offline; execution and policy
observations continue using the existing truth-separated contract.

## Probe before training

| Config | Purpose | Difference |
|---|---|---|
| `probe_t10.json` | Frozen-R2 rollout only | T=1.0; skip matching roots already collected |
| `probe_t13.json` | Frozen-R2 rollout only | T=1.3 |
| `probe_t16.json` | Conditional fallback probe | T=1.6 only if both lower settings lack useful signal |
| `train_group.json` | Single-update control | T=1.3, GRPO/group, original reward |
| `train_none.json` | Normalization ablation | T=1.3, GRPO/none, original reward |
| `train_cost.json` | Conditional reward ablation | GRPO/none, call .01, regret 0; safety penalties unchanged |

All use K=4, no top-p/top-k truncation, initial R2, LR=1e-5, beta_kl=0,
row cap=25, median factor=5, warmup=16, global cap=1. A configured logprob
check requires every recorded sampling probability and aborts before an
optimizer step if max absolute recomputation error exceeds 0.1. This is a
provisional smoke-test tolerance, not a measured Gemma numerical error bound.
`max_length=33024` accommodates the existing 32768-token rollout prompt plus
256 completion tokens without trimming the context during training. Keep
the existing `RESEARCH_MAX_INPUT_TOKENS=32768` and `RESEARCH_MAX_NEW_TOKENS=256`
environment settings consistent across the probe and both update arms.

Example phase command **inside an allocated GPU job**:

```bash
python -m psse_env.rl.train --config research/hpc/rl_review_20261015/probe_t13.json --phase rollout
```

Do not use `--iterations` for the probe: that would also update it. Analyze
its saved groups with `signal_report`. Read the predeclared continuation
criteria in the decision document before selecting a training arm. T=1.3 is
an untested candidate; if another temperature is selected, update all arm
configs consistently **before** creating their new output directories.

## Reuse the exact frozen samples for controlled updates

The probe's completed unsharded rollout writes `rollout_config.json`.
`fork_frozen_rollouts` checks source code revision, config, initial adapter,
root/sample counts, seed and pairing. It refuses an existing target output
directory or changed sampling/environment settings. It recomputes rewards
from full episode records (essential for the cost arm), keeps sampled tokens
and probabilities, and creates a manifest at the **update** phase.

```bash
python -m research.rl.fork_frozen_rollouts --source-config research/hpc/rl_review_20261015/probe_t13.json --target-config research/hpc/rl_review_20261015/train_group.json
python -m research.rl.fork_frozen_rollouts --source-config research/hpc/rl_review_20261015/probe_t13.json --target-config research/hpc/rl_review_20261015/train_none.json
```

Then run each arm's `--phase update` and `--phase evaluate` in its GPU
allocation. Both start from frozen R2; no sampling phase is repeated. Only
create `train_cost` if the saved-reward analysis justifies it. Source rollout
files and rewards remain unchanged. Never use this to keep updating a new
policy on stale samples; collect fresh trajectories after each new update.

The 209-root development evaluation is common across arms. Compare the
underlying physical/safety metrics and one common reward definition; each
arm's own mean reward is not a fair cross-reward score. Re-evaluate the
selected adapter on the original 160 roots for retention, and use an unseen
physical-parent-held-out test set for final claims.

## Local verification

CPU environment matches the repository's core pins: PyTorch 2.10.0+cpu,
Transformers 5.10.4, PEFT 0.19.1, NumPy 2.3.5, SciPy 1.16.3, PYPOWER 5.1.19.
The focused command is:

```bash
python -m pytest -q psse_env/rl research/rl psse_env/dagger/test_research_policy_factory.py
```

Final focused result: **91 passed, 1 skipped**. The full-suite integration
test skips because the private generated experiment suite is absent. Tiny
PEFT tests exercise actual adapter switching; they do not validate Gemma
4-bit numerics, A100 performance, or new RL outcomes.
