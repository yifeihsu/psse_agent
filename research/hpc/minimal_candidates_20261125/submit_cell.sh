#!/usr/bin/env bash
# Submit one arm of the candidate read-out.  Run on torch, or through ssh stdin:
#
#   wsl -- ssh torch bash -s -- score base < research/hpc/minimal_candidates_20261125/submit_cell.sh
#
# One job an arm, ten hours at most, preemptible and requeued; nothing here pins a card type or a partition.  It
# refuses to start before the arm's adapter and inputs are there, once the arm's scores are written, and while a
# job of the same name is queued (ALLOW_DUPLICATE=1 lifts that).
set -euo pipefail
export PATH=/opt/slurm/bin:$PATH
WORK=/scratch/yx3882/minimal_candidates_20261125
# shellcheck source=cell.env
source "$WORK/source/research/hpc/minimal_candidates_20261125/cell.env"
COMMAND=${1:?score <arm>}
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
    ARM=${2:?the arm to score (one of: $ARMS)}
    ADAPTER=$(arm_adapter "$ARM")
    INPUTS=$(arm_inputs "$ARM")
    SCORES=$(arm_scores "$ARM")
    [[ -s "$ADAPTER/adapter_config.json" ]] || { echo "no adapter at $ADAPTER" >&2; exit 2; }
    [[ -s "$INPUTS" ]] || { echo "no inputs at $INPUTS: deploy first" >&2; exit 2; }
    [[ ! -s "$SCORES/summary.json" ]] || { echo "the scores of $ARM are written already: $SCORES/summary.json" >&2; exit 2; }
    not_queued "mcs-$ARM"
    id=$(sbatch --parsable --job-name="mcs-$ARM" --export="ALL,ARM=$ARM" "$CELL/stage_candidates.sbatch")
    record "mcs-$ARM job=${id%%;*}"
    ;;
  *) echo "unknown command $COMMAND (score <arm>)" >&2; exit 2 ;;
esac
squeue -u "$USER" -o "%.12i %.28j %.9T %.10M %.26E %.20R"
