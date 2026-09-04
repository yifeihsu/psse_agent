#!/usr/bin/env bash
# Replace the queued training and evaluation jobs with the staged scripts while
# the collection job keeps running.  Slurm copies a batch script at submission,
# so an amended script only takes effect through a fresh submission.
#
#   amend_train_chain.sh COLLECT_JOB_ID
set -euo pipefail
export PATH=/opt/slurm/bin:$PATH
ROUND=/scratch/yx3882/research_diag_round_20260903
COLLECT_JOB=${1:?running collection job id}
[[ "$COLLECT_JOB" =~ ^[0-9]+$ ]] || { echo "collection job id must be numeric" >&2; exit 2; }
cd "$ROUND"
state=$(squeue -j "$COLLECT_JOB" -h -o "%T" 2>/dev/null || true)
if [[ "$state" != "RUNNING" && "$state" != "PENDING" ]]; then
  echo "collection job $COLLECT_JOB is not queued or running (state: ${state:-gone}); use submit_diag.sh instead" >&2
  exit 2
fi
for f in diag_train.sbatch diag_eval.sbatch; do bash -n "$f"; done
python3 -m py_compile filter_mixture.py summarize.py 2>/dev/null \
  || /scratch/yx3882/.conda/envs/gemma4_research_5104/bin/python -m py_compile filter_mixture.py summarize.py
pending=$(squeue -u "$USER" -h -o "%i %j" | awk '$2=="diagtrain"||$2=="diageval"{print $1}' | tr '\n' ' ')
for id in $pending; do scancel "$id"; done
if [[ -n "${pending// /}" ]]; then sleep 5; fi
job_id() { local id=${1%%;*}; [[ "$id" =~ ^[0-9]+$ ]] || { echo "sbatch returned no job id: $1" >&2; exit 2; }; printf '%s' "$id"; }
JT=$(job_id "$(sbatch --parsable --dependency=afterok:"$COLLECT_JOB" diag_train.sbatch)")
JE=$(job_id "$(sbatch --parsable --dependency=afterok:"$JT" diag_eval.sbatch)")
(sha256sum round.env ./*.sh ./*.sbatch ./*.py > scripts.sha256)
echo "amended: collect=$COLLECT_JOB train=$JT eval=$JE cancelled=[${pending% }] at $(date -u +%FT%TZ)" | tee -a submitted_jobs.txt
squeue -u "$USER" -o "%.12i %.10j %.8T %.10M %.24R"
