#!/usr/bin/env bash
# Submit the three-stage diagnostic round as a dependency chain.
set -euo pipefail
export PATH=/opt/slurm/bin:$PATH
ROUND=/scratch/yx3882/research_diag_round_20260903
cd "$ROUND"
[[ -s out/prerequisites.dryrun.json ]] || { echo "run prerequisites.sh --output out/prerequisites.dryrun.json first" >&2; exit 2; }
if squeue -u "$USER" -h -o "%j" | grep -Eq '^diag(col|train|eval)$'; then
  echo "diagnostic round jobs are already queued or running; refusing a duplicate chain" >&2
  squeue -u "$USER" -o "%.12i %.10j %.8T %.10M %.24R" >&2
  exit 2
fi
job_id() { local id=${1%%;*}; [[ "$id" =~ ^[0-9]+$ ]] || { echo "sbatch returned no job id: $1" >&2; exit 2; }; printf '%s' "$id"; }
JC=$(job_id "$(sbatch --parsable diag_collect.sbatch)")
JT=$(job_id "$(sbatch --parsable --dependency=afterok:"$JC" diag_train.sbatch)")
JE=$(job_id "$(sbatch --parsable --dependency=afterok:"$JT" diag_eval.sbatch)")
echo "collect=$JC train=$JT eval=$JE submitted_at=$(date -u +%FT%TZ)" | tee -a submitted_jobs.txt
squeue -u "$USER" -o "%.12i %.10j %.8T %.10M %.24R"
