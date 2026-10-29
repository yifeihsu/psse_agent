#!/usr/bin/env bash
# Submit the evaluations of the round-2 rerun's R2, alone (2026-10-29; note
# section 9): every R2 evaluation the cell recorded, under its recorded name,
# rows, horizon, seed and worker rule, once the recorded records are set aside
# under a _pool_20261023 name:
#   <prefix>_40_r2                     the second draw at the cell's horizon
#   test_budget12_r2, test_budget16_r2, test_budget20_r2
#                                      the first draw at 12, 16 and 20 actions
#   <prefix>_budget12_r2, <prefix>_budget20_r2
#                                      the second draw at 12 and 20 actions
#   validation_r2, hard_r2             the validation (96) and hard-draw (193) roots at the
#                                      cell's horizon; the recorded R2 ran there beside the
#                                      control as {suite}_r2_control
# The first draw at the cell's horizon is the round-2 evaluation itself (r2e).
# DEPEND=<job id> (the new r2t) makes every job wait for it (afterok); without
# DEPEND the R2 adapter must already exist.  ONLY="name ..." submits just those.
# A name whose evaluation directory, console receipt or spec still exists is
# refused: stage_eval_suite.sbatch skips an evaluation whose summary exists, so a
# record left in place would be read as the new R2's.
set -euo pipefail
export PATH=/opt/slurm/bin:$PATH
PIPE=/scratch/yx3882/research_full_pipeline_20260912
cd "$PIPE"
# shellcheck source=pipeline.env
source "$PIPE/pipeline.env"
: "${TEST_DRAW_NAME:?}" "${TEST_DRAW_PREFIX:?}"
EXTRA=$OUT/extra
EVALS=$EXTRA/evaluations
SPECS=$EXTRA/specs
R2=$R2_DIR/training/lora
TEST1=$R1_DIR/collection/development_scenarios.json
TEST2=$EXTRA/$TEST_DRAW_NAME/suite/development.json
VALIDATION=$EXTRA/suites/validation.json
HARD=$EXTRA/suites/hard.json
FULL=${EVAL_MAX_STEPS:?}
DEPEND=${DEPEND:-}
[[ -z "$DEPEND" || "$DEPEND" =~ ^[0-9]+$ ]] || { echo "DEPEND must be a Slurm job id" >&2; exit 2; }
if [[ -z "$DEPEND" && ! -s "$R2/adapter_model.safetensors" ]]; then
  echo "no DEPEND and the R2 adapter is missing: $R2" >&2; exit 2
fi
for suite in "$TEST1" "$TEST2" "$VALIDATION" "$HARD"; do
  [[ -s "$suite" ]] || { echo "suite missing: $suite" >&2; exit 2; }
done
# name, rows, horizon
PLAN=(
  "${TEST_DRAW_PREFIX}_${FULL}_r2 $TEST2 $FULL"
  "test_budget12_r2 $TEST1 12"
  "test_budget16_r2 $TEST1 16"
  "test_budget20_r2 $TEST1 20"
  "${TEST_DRAW_PREFIX}_budget12_r2 $TEST2 12"
  "${TEST_DRAW_PREFIX}_budget20_r2 $TEST2 20"
  "validation_r2 $VALIDATION $FULL"
  "hard_r2 $HARD $FULL"
)
ONLY=${ONLY:-}
selected=()
for entry in "${PLAN[@]}"; do
  read -r name suite horizon <<< "$entry"
  [[ -z "$ONLY" || " $ONLY " == *" $name "* ]] || continue
  for taken in "$EVALS/$name" "$EVALS/$name.console.json" "$SPECS/$name.env"; do
    [[ ! -e "$taken" ]] || { echo "$taken exists: set the recorded record aside first" >&2; exit 2; }
  done
  selected+=("$entry")
done
(( ${#selected[@]} )) || { echo "no evaluation selected" >&2; exit 2; }
mkdir -p "$SPECS"
job_id() { local id=${1%%;*}; [[ "$id" =~ ^[0-9]+$ ]] || { echo "sbatch returned no job id: $1" >&2; exit 2; }; printf '%s' "$id"; }
dependency=()
[[ -n "$DEPEND" ]] && dependency=(--dependency=afterok:"$DEPEND")
submitted=()
for entry in "${selected[@]}"; do
  read -r name suite horizon <<< "$entry"
  printf 'EVAL_NAME=%s\nEVAL_SUITE=%s\nEVAL_POLICIES="%s"\nEVAL_EXPERT=0\nEVAL_MAX_STEPS=%s\n' \
    "$name" "$suite" "r2=$R2" "$horizon" > "$SPECS/$name.env"
  id=$(job_id "$(sbatch --parsable "${dependency[@]}" --export=ALL,EVAL_SPEC="$SPECS/$name.env" stage_eval_suite.sbatch)")
  submitted+=("$name=$id")
done
echo "r2-rerun-evaluations ${submitted[*]} depend=${DEPEND:-none} submitted_at=$(date -u +%FT%TZ)" | tee -a submitted_jobs.txt
squeue -u "$USER" -o "%.12i %.10j %.8T %.10M %.24R"
