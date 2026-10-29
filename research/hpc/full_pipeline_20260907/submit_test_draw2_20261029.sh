#!/usr/bin/env bash
# Submit the conventional-only cell's second test draw and its evaluations
# (2026-10-29; note section 8):
#   fp-xdraw (CPU): the draw (stage_test_draw.sbatch), unless its receipt exists;
#   fp-xeval <prefix>_40            BC0, R1 and the teacher on the new draw at the cell's horizon (40)  (after the draw)
#   fp-xeval <prefix>_40_r2         R2 alone at the cell's horizon                                        (after the draw)
#   fp-xeval <prefix>_budget<b>     BC0, R1 and the teacher at <b> actions, b in BUDGETS (12 20)   (after <prefix>_40)
#   fp-xeval <prefix>_budget<b>_r2  R2 alone at <b> actions                                     (after <prefix>_40_r2)
# The prefix is TEST_DRAW_PREFIX; the suite is $OUT/extra/$TEST_DRAW_NAME/suite/development.json.
# Every evaluation spec names the cell's adapters, seed and teacher as in every
# evaluation of this cell (stage_eval_suite.sbatch).  Every job is
# output-guarded; ONLY="name ..." resubmits just those evaluations without
# dependencies (their inputs must exist).
set -euo pipefail
export PATH=/opt/slurm/bin:$PATH
PIPE=/scratch/yx3882/research_full_pipeline_20260912
cd "$PIPE"
# shellcheck source=pipeline.env
source "$PIPE/pipeline.env"
: "${TEST_DRAW_NAME:?}" "${TEST_DRAW_SEED:?}" "${TEST_DRAW_PREFIX:?}"
[[ -s "$OUT/r2/round_summary.json" ]] || { echo "the cell's chain has not finished" >&2; exit 2; }
EXTRA=$OUT/extra
DRAW=$EXTRA/$TEST_DRAW_NAME
SUITE=$DRAW/suite/development.json
SPECS=$EXTRA/specs
mkdir -p "$SPECS"
PREFIX=$TEST_DRAW_PREFIX
BUDGETS=${BUDGETS:-"12 20"}
FULL=${EVAL_MAX_STEPS:?}
job_id() { local id=${1%%;*}; [[ "$id" =~ ^[0-9]+$ ]] || { echo "sbatch returned no job id: $1" >&2; exit 2; }; printf '%s' "$id"; }
spec() {
  local name=$1 suite=$2 expert=$3 max_steps=$4
  shift 4
  printf 'EVAL_NAME=%s\nEVAL_SUITE=%s\nEVAL_POLICIES="%s"\nEVAL_EXPERT=%s\nEVAL_MAX_STEPS=%s\n' \
    "$name" "$suite" "$*" "$expert" "$max_steps" > "$SPECS/$name.env"
  printf '%s' "$SPECS/$name.env"
}
BC0=$BC0_DIR/lora
R1=$R1_DIR/training/lora
R2=$R2_DIR/training/lora
for adapter in "$BC0" "$R1" "$R2"; do
  [[ -s "$adapter/adapter_model.safetensors" ]] || { echo "adapter missing: $adapter" >&2; exit 2; }
done
declare -A SPEC_FILE=(
  ["${PREFIX}_${FULL}"]=$(spec "${PREFIX}_${FULL}" "$SUITE" 1 "$FULL" "bc0=$BC0" "r1=$R1")
  ["${PREFIX}_${FULL}_r2"]=$(spec "${PREFIX}_${FULL}_r2" "$SUITE" 0 "$FULL" "r2=$R2")
)
for budget in $BUDGETS; do
  [[ "$budget" =~ ^[0-9]+$ && "$budget" -ge 1 ]] || { echo "budget must be a positive integer: $budget" >&2; exit 2; }
  SPEC_FILE["${PREFIX}_budget$budget"]=$(spec "${PREFIX}_budget$budget" "$SUITE" 1 "$budget" "bc0=$BC0" "r1=$R1")
  SPEC_FILE["${PREFIX}_budget${budget}_r2"]=$(spec "${PREFIX}_budget${budget}_r2" "$SUITE" 0 "$budget" "r2=$R2")
done
submitted=()
LAST_ID=""
submit_eval() {
  local name=$1 dependency=$2
  local args=()
  [[ -n "$dependency" ]] && args=(--dependency=afterok:"$dependency")
  LAST_ID=$(job_id "$(sbatch --parsable "${args[@]}" --export=ALL,EVAL_SPEC="${SPEC_FILE[$name]}" stage_eval_suite.sbatch)")
  submitted+=("$name=$LAST_ID")
}
if [[ -n "${ONLY:-}" ]]; then
  for name in $ONLY; do
    [[ -n "${SPEC_FILE[$name]:-}" ]] || { echo "unknown evaluation: $name" >&2; exit 2; }
    submit_eval "$name" ""
  done
else
  draw=""
  if [[ ! -s "$DRAW/draw.done" ]]; then
    draw=$(job_id "$(sbatch --parsable stage_test_draw.sbatch)")
    submitted+=("draw=$draw")
  fi
  submit_eval "${PREFIX}_${FULL}" "$draw"
  full=$LAST_ID
  submit_eval "${PREFIX}_${FULL}_r2" "$draw"
  full_r2=$LAST_ID
  for budget in $BUDGETS; do
    submit_eval "${PREFIX}_budget$budget" "$full"
    submit_eval "${PREFIX}_budget${budget}_r2" "$full_r2"
  done
fi
echo "test-draw $TEST_DRAW_NAME seed=$TEST_DRAW_SEED ${submitted[*]} submitted_at=$(date -u +%FT%TZ)" | tee -a submitted_jobs.txt
squeue -u "$USER" -o "%.12i %.10j %.8T %.10M %.24R"
