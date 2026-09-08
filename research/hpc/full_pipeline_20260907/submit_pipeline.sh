#!/usr/bin/env bash
# Submit the whole pipeline as one Slurm dependency chain:
#   d0 (CPU) -> bc0 -> r1 collect -> r1 train -> r1 eval
#             -> r2 collect -> r2 train -> r2 eval
# Every stage is output-guarded, so resubmitting after a failure resumes at
# the first stage without a receipt.  FROM=<stage> starts the chain at that
# stage (d0, bc0, r1c, r1t, r1e, r2c, r2t, r2e) when the earlier receipts
# already exist.
set -euo pipefail
export PATH=/opt/slurm/bin:$PATH
PIPE=/scratch/yx3882/research_full_pipeline_20260908
cd "$PIPE"
[[ -s out/prerequisites.dryrun.json ]] || { echo "run prerequisites.sh --output out/prerequisites.dryrun.json first" >&2; exit 2; }
if squeue -u "$USER" -h -o "%j" | grep -Eq '^fp-'; then
  echo "pipeline jobs are already queued or running; refusing a duplicate chain" >&2
  squeue -u "$USER" -o "%.12i %.10j %.8T %.10M %.24R" >&2
  exit 2
fi
FROM=${FROM:-d0}
ORDER=(d0 bc0 r1c r1t r1e r2c r2t r2e)
start=-1
for index in "${!ORDER[@]}"; do [[ "${ORDER[$index]}" == "$FROM" ]] && start=$index; done
[[ $start -ge 0 ]] || { echo "unknown FROM stage: $FROM" >&2; exit 2; }
job_id() { local id=${1%%;*}; [[ "$id" =~ ^[0-9]+$ ]] || { echo "sbatch returned no job id: $1" >&2; exit 2; }; printf '%s' "$id"; }
previous=""
submitted=()
for index in "${!ORDER[@]}"; do
  [[ $index -ge $start ]] || continue
  stage=${ORDER[$index]}
  dependency=()
  [[ -n "$previous" ]] && dependency=(--dependency=afterok:"$previous")
  case "$stage" in
    d0)  id=$(job_id "$(sbatch --parsable "${dependency[@]}" stage_d0.sbatch)") ;;
    bc0) id=$(job_id "$(sbatch --parsable "${dependency[@]}" stage_bc0.sbatch)") ;;
    r1c) id=$(job_id "$(sbatch --parsable "${dependency[@]}" -J fp-r1c --export=ALL,PIPELINE_ROUND=r1 stage_collect.sbatch)") ;;
    r1t) id=$(job_id "$(sbatch --parsable "${dependency[@]}" -J fp-r1t --export=ALL,PIPELINE_ROUND=r1 stage_train.sbatch)") ;;
    r1e) id=$(job_id "$(sbatch --parsable "${dependency[@]}" -J fp-r1e --export=ALL,PIPELINE_ROUND=r1 stage_eval.sbatch)") ;;
    r2c) id=$(job_id "$(sbatch --parsable "${dependency[@]}" -J fp-r2c --export=ALL,PIPELINE_ROUND=r2 stage_collect.sbatch)") ;;
    r2t) id=$(job_id "$(sbatch --parsable "${dependency[@]}" -J fp-r2t --export=ALL,PIPELINE_ROUND=r2 stage_train.sbatch)") ;;
    r2e) id=$(job_id "$(sbatch --parsable "${dependency[@]}" -J fp-r2e --export=ALL,PIPELINE_ROUND=r2 stage_eval.sbatch)") ;;
  esac
  submitted+=("$stage=$id")
  previous=$id
done
echo "${submitted[*]} submitted_at=$(date -u +%FT%TZ)" | tee -a submitted_jobs.txt
squeue -u "$USER" -o "%.12i %.10j %.8T %.10M %.24R"
