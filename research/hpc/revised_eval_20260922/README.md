# Revised Expert and R2 evaluation

These templates run revised code against the unchanged 160-root development
suite and finished R2 adapter from the 2026-09-21 physical pipeline. They do not
train, regenerate scenarios, or write to the old pipeline directory. The source
snapshot and its manifest are staged separately by the operator.

| Job template | Resources | Purpose |
| --- | --- | --- |
| `preflight.sbatch` | `cs`, 4 CPUs, 16 GB, 1 hour | Import the actual runtime and run focused checks |
| `expert.sbatch` | `cs`, 16 CPUs, 48 GB, 6 hours | Evaluate the observable expert on every frozen root |
| `r2.sbatch` | 1 GPU, 16 CPUs, 96 GB, 12 hours | Evaluate the frozen R2 adapter with the revised runtime |

The GPU job uses account `torch_pr_627_general`, the
`a100|h100|h200|rtx6000` constraint, and preemption/requeue support. It does not
select a GPU partition. Both evaluations should depend on the same successful
preflight, so they may then run independently.

`common.env` sources the old cell's `pipeline.env`, then resets source and output
paths to the separately exported `REVISED_EVAL_ROOT`. It preserves the original
Python/model settings and `pipeline_environment` generation budgets of 32,768
input tokens and 256 new tokens. The runner keeps its matched defaults: seed
20260912, 40 actions, HIF search 7×9×10, chi-square alpha .01 and normalized
residual threshold 4. Native numerical threads are capped at one per worker;
the inherited HIF worker count follows the allocated CPUs.

Fixed inputs:

- Scenarios: `/scratch/yx3882/research_full_pipeline_20260921_physical/out/r1/collection/development_scenarios.json`
- R2 adapter: `/scratch/yx3882/research_full_pipeline_20260921_physical/out/r2/training/lora`
- Old expert results: `/scratch/yx3882/research_full_pipeline_20260921_physical/out/r1/collection/evaluation/expert_eval.json`
- Old R2 results: `/scratch/yx3882/research_full_pipeline_20260921_physical/out/r2/collection/evaluation/r1_eval.json`

The last filename is intentional: the old paired runner named every candidate
`r1_eval.json`, including the round-2 candidate. Passing these full reports lets
the revised runner compare matching roots and seeds. Missing or incompatible
inputs fail rather than silently replacing the comparison with summary counts.

The same runner also evaluates the frozen BC0 and R1 adapters (`bc0.sbatch`,
`r1.sbatch`, same GPU resources as `r2.sbatch`). Their per-root baselines are
the round-1 paired evaluation, where the student was BC0 and the candidate R1:

- BC0 adapter: `/scratch/yx3882/research_full_pipeline_20260921_physical/out/bc0/lora`,
  baseline `.../out/r1/collection/evaluation/bc0_eval.json` (132/160)
- R1 adapter: `/scratch/yx3882/research_full_pipeline_20260921_physical/out/r1/training/lora`,
  baseline `.../out/r1/collection/evaluation/r1_eval.json` (140/160)

Outputs go to `out/bc0/` and `out/r1/` with `bc0_eval.json` and `r1_eval.json`.

Example staging/submission commands, for the operator to run after creating a
new cell and checking its source manifest:

```bash
export REVISED_EVAL_ROOT=/scratch/yx3882/NEW_UNIQUE_REVISED_EVAL_CELL
mkdir -p "$REVISED_EVAL_ROOT/logs"
TEMPLATES="$REVISED_EVAL_ROOT/source/research/hpc/revised_eval_20260922"
PREFLIGHT_ID=$(sbatch --parsable --chdir="$REVISED_EVAL_ROOT" "$TEMPLATES/preflight.sbatch")
sbatch --parsable --chdir="$REVISED_EVAL_ROOT" --dependency="afterok:$PREFLIGHT_ID" "$TEMPLATES/expert.sbatch"
sbatch --parsable --chdir="$REVISED_EVAL_ROOT" --dependency="afterok:$PREFLIGHT_ID" "$TEMPLATES/r2.sbatch"
```

The templates require an explicit new root and reject the old cell or any of its
descendants. Slurm logs are relative to `--chdir`, so create `logs/` before
submission. Each job copies the staged OpenDSS model into a unique `mktemp`
directory under `/dev/shm`; cleanup verifies the resolved path before removal.

Preflight always smoke-imports the **actual runtime `$PY`**, including the runner
and revised provider/controller/audit modules. It uses `$PY` for pytest when
available; otherwise it checks and uses the inherited `$TEST_PY`. A failed smoke
or test writes `out/preflight.json` with `passed=false` and exits nonzero. Logs
are retained under `out/preflight_logs/`. The actual runtime smoke also records
its PID/interpreter and verifies the staged import location, inherited model
descriptor, token limits, seed, episode horizon, detector threshold and HIF
grid settings. Tokenizer/model loading remains the evaluation runner's job.
Focused tests include the HIF policy-serialization checks; the unrelated
legacy synthetic production-pilot fixture is not included.

Evaluation outputs are separate: `out/expert/` and `out/r2/`. Each contains
`run_receipt.json`, `progress.jsonl`, per-root `episodes/*.json`, its
`expert_eval.json` or `r2_eval.json`, `comparison.json`, and `completed.json`.
Both commands pass `--resume`; the runner validates source, data, model and
configuration identity before reusing completed roots. A requeue therefore
resumes recorded work without interpreting a partial result as a completed run.
