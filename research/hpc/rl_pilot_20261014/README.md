# RL pilot cell (2026-10-14)

On-policy LoRA updates of the 2026-10-07 classifier cell's round-2 student
from its own sampled rollouts, scored by the offline audit
(`docs/rl_experiments_20261014.md`, plan `docs/rl_hard_cases_plan_20261013.md`).

| File | Purpose |
| --- | --- |
| `rl_pilot.json` | The pilot config: suites and adapter of the classifier cell, the sampling, reward, advantage and update settings |
| `rl_pilot.sbatch` | One job runs `ITERATIONS` iterations as separate rollout, update and evaluate processes; the rollout and evaluation phases use two worker processes on an 80 GB card; requeues resume at the manifest's phase |
| `deploy_from_windows.sh` | Clone the classifier cell's source, upload the bundle, the configs and the expert reference, submit one job per config |
| `rl_hard.json`, `rl_hard_screen.json` | The two hard-draw arms (classifier gate, balanced screen), 64 roots per iteration from one seeded permutation |
| `rl_hard_r2_baseline.json`, `rl_hard_screen_baseline.json` | Their controls: the frozen starting adapters scored on the same 209 development roots |
| `rl_hard_screen_strat_kl.json` | The screen arm rerun with stratified roots (`root_sampling`, fixed counts per hard-draw stratum) and a KL penalty to the starting adapter (`beta_kl` 0.04); run note section 5 |
| `rl_hard_screen_strat_kl_smoke.json` | One short iteration of that rerun (8 roots, no evaluation) that exercises the KL update on the real model |
| `rl_hard_screen_fix.json` | The screen arm with the estimator and data-contract fixes of 2026-10-17: advantages in reward units, trajectory normalization with whole-trajectory optimizer steps, the clipped ratio against the recorded sampling probabilities, the on-policy tolerance, prompt digests, complete groups, preservation rows for the KL term; stratified mix and `beta_kl` 0.04 kept; run note section 9 |
| `rl_hard_screen_fix_smoke.json` | One short iteration of that config (8 roots, no evaluation); its update report must show every row with a sampling record and a prompt digest, preservation rows, a `clip_check` of zero at the first step (the clip baseline is the training pass), a sampler gap under 0.2 and a KL of zero at the first step. Its first run (job 20189365) measured the sampler gap at 0.053 overall and 0.56 on the decision tokens; run note section 9 |

Deploy and submit from the Windows checkout (needs the SSH master):

```bash
bash research/hpc/rl_pilot_20261014/deploy_from_windows.sh "$(git rev-parse HEAD)"
```

Status and results on the cluster:

```bash
wsl -- ssh torch "cat /scratch/yx3882/rl_pilot_20261014/out/pilot_v2/manifest.json"
```

The manifest is written when a phase completes; before the first phase
completes, the per-root files under the iteration directory show progress.
Each iteration directory holds `roots.json`, `trajectories/groups/*.jsonl`
(every sampled episode with its reward, audit, recorded tokens and their
sampling probabilities), `rollout_summary.json` (with the store-wide count
of groups whose samples differed), `advantages.json`, `update_report.json`,
the new `adapter/`, `update_checkpoint/` while an update is in flight,
`evaluation/episodes/*.json` and `evaluation.json` with the greedy
development score.

The expert reference (calls per root for the tool-regret term) covers the
development roots and the round-2 training roots:
`output/rl_pilot_20261014/expert_reference.jsonl`.

Where an iteration's learning signal is (groups with a reward spread, what the
spread is made of, where the samples part, how peaked the sampling was;
`docs/rl_implementation_review_20261016.md`):

```bash
wsl -- ssh torch "cd /scratch/yx3882/rl_pilot_20261014/source && PYTHONPATH=. python -m psse_env.rl.diagnose --iteration /scratch/yx3882/rl_pilot_20261014/out/hard_v1/iteration_000 --output /scratch/yx3882/rl_pilot_20261014/out/hard_v1/iteration_000/diagnosis.json"
```

`update_report.json` of every update since 2026-10-16 also carries
`on_policy_check`: per optimizer step, how far the recomputed log-probabilities
are from the ones the policy sampled with.

`out/pilot` holds the first submission's 25 groups (temperature 0.8,
top-p 0.95, before the fixes of 2026-10-15); keep it as the record.
