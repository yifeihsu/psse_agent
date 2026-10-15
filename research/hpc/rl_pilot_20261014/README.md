# RL pilot cell (2026-10-14)

On-policy LoRA updates of the 2026-10-07 classifier cell's round-2 student
from its own sampled rollouts, scored by the offline audit
(`docs/rl_experiments_20261014.md`, plan `docs/rl_hard_cases_plan_20261013.md`).

| File | Purpose |
| --- | --- |
| `rl_pilot.json` | The pilot config: suites and adapter of the classifier cell, the sampling, reward, advantage and update settings |
| `rl_pilot.sbatch` | One job runs `ITERATIONS` iterations as separate rollout, update and evaluate processes; requeues resume at the manifest's phase |
| `deploy_from_windows.sh` | Clone the classifier cell's source, upload the bundle, the configs and the expert reference, submit one job per config |

Deploy and submit from the Windows checkout (needs the SSH master):

```bash
bash research/hpc/rl_pilot_20261014/deploy_from_windows.sh "$(git rev-parse HEAD)"
```

Status and results on the cluster:

```bash
wsl -- ssh torch "cat /scratch/yx3882/rl_pilot_20261014/out/pilot/manifest.json"
```

Each iteration directory holds `roots.json`, `trajectories/groups/*.jsonl`
(every sampled episode with its reward, audit and recorded tokens),
`rollout_summary.json`, `advantages.json`, `update_report.json`, the new
`adapter/`, and `evaluation.json` with the greedy development score.

The expert reference (calls per root for the tool-regret term) is the
teacher's end-to-end run under the same profile,
`output/classifier_triage_20261013/e2e_ieee14_spectra_first/episodes.jsonl`.
