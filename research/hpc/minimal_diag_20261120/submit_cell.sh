#!/usr/bin/env bash
# Submit the stopping-state diagnostic's one job.  Run on torch, or through ssh stdin:
#
#   wsl -- ssh torch bash -s -- score < research/hpc/minimal_diag_20261120/submit_cell.sh
#
# One job, eight hours at most, preemptible and requeued; nothing here pins a card type or a partition.  It refuses
# to start before the supervised pass has its receipt and the inputs are uploaded, once the scores are written, and
# while a job of the same name is queued (ALLOW_DUPLICATE=1 lifts that).
set -euo pipefail
export PATH=/opt/slurm/bin:$PATH
WORK=/scratch/yx3882/minimal_diag_20261120
# shellcheck source=cell.env
source "$WORK/source/research/hpc/minimal_diag_20261120/cell.env"
COMMAND=${1:?score}
[[ -d "$CELL" ]] || { echo "no source at $CELL: deploy first" >&2; exit 2; }
mkdir -p "$LOGS" "$OUT" "$DATA"
not_queued() {
  [[ -n "${ALLOW_DUPLICATE:-}" ]] && return 0
  local queued
  queued=$(squeue -h -u "$USER" -n "$1" -o %i || true)
  [[ -z "$queued" ]] || { echo "jobs named $1 are already queued or running ($queued); ALLOW_DUPLICATE=1 lifts this" >&2; exit 2; }
}
record() { echo "$1 source=$(git -C "$SRC" rev-parse --short=7 HEAD) submitted_at=$(date -u +%FT%TZ)" | tee -a "$WORK/submitted_jobs.txt"; }
case "$COMMAND" in
  score)
    [[ -s "$PASS/training.done" ]] || { echo "no training receipt $PASS/training.done: the supervised pass has not finished" >&2; exit 2; }
    for inputs in "$INPUTS_DEV" "$INPUTS_TRAIN"; do
      [[ -s "$inputs" ]] || { echo "no scoring inputs at $inputs: deploy first" >&2; exit 2; }
    done
    [[ ! -s "$SCORES/summary.json" ]] || { echo "the scores are written already: $SCORES/summary.json" >&2; exit 2; }
    not_queued mdg-score
    id=$(sbatch --parsable --job-name=mdg-score "$CELL/stage_score.sbatch")
    record "mdg-score job=${id%%;*}"
    ;;
  *) echo "unknown command $COMMAND (score)" >&2; exit 2 ;;
esac
squeue -u "$USER" -o "%.12i %.28j %.9T %.10M %.26E %.20R"
