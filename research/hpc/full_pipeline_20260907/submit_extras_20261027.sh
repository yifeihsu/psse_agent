#!/usr/bin/env bash
# Submit the conventional-only cell's 2026-10-27 follow-ups (cell note, section 7),
# after the chain and the 2026-10-23 follow-ups:
#   fp-xeval test_budget<b>, one job per budget in BUDGETS (default "12 16 20"):
#     BC0, R1 and the teacher on the 192 test roots with the evaluation horizon
#     set to <b> actions.  EVAL_MAX_STEPS in the spec overrides the 40 of
#     pipeline.env (stage_eval_suite.sbatch sources the spec after it); the
#     suite rows, SEED_ROUND, the adapters, the teacher and the workers are
#     those of the round-1 evaluation.  Results: out/extra/evaluations/test_budget<b>.
#   ADAPTER="label=/path/to/lora" (optional): that adapter alone on the test,
#     validation and hard suites at the 40-action horizon (test_<label>,
#     validation_<label>, hard_<label>), to be paired with the existing records
#     by followup_tables.py --extra-policy <label>.
# Every job is output-guarded, so resubmitting after a failure redoes only what
# is missing.  ONLY="name ..." submits just those evaluations.
set -euo pipefail
export PATH=/opt/slurm/bin:$PATH
PIPE=/scratch/yx3882/research_full_pipeline_20260912
cd "$PIPE"
# shellcheck source=pipeline.env
source "$PIPE/pipeline.env"
[[ -s "$OUT/r1/round_summary.json" ]] || { echo "the cell's round-1 evaluation has not finished" >&2; exit 2; }
EXTRA=$OUT/extra
SPECS=$EXTRA/specs
mkdir -p "$SPECS"
BUDGETS=${BUDGETS:-"12 16 20"}
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
TEST=$R1_DIR/collection/development_scenarios.json
VALIDATION=$EXTRA/suites/validation.json
HARD=$EXTRA/suites/hard.json
declare -A SPEC_FILE=()
ORDER=""
for budget in $BUDGETS; do
  [[ "$budget" =~ ^[0-9]+$ && "$budget" -ge 1 ]] || { echo "budget must be a positive integer: $budget" >&2; exit 2; }
  SPEC_FILE[test_budget$budget]=$(spec "test_budget$budget" "$TEST" 1 "$budget" "bc0=$BC0" "r1=$R1")
  ORDER="$ORDER test_budget$budget"
done
if [[ -n "${ADAPTER:-}" ]]; then
  label=${ADAPTER%%=*}
  path=${ADAPTER#*=}
  [[ "$label" =~ ^[a-z][a-z0-9_]*$ && "$label" != bc0 && "$label" != r1 && "$label" != r2 && "$label" != control && "$label" != expert ]] \
    || { echo "ADAPTER label must be a fresh lower-case name: $label" >&2; exit 2; }
  [[ -s "$path/adapter_model.safetensors" ]] || { echo "adapter missing: $path" >&2; exit 2; }
  SPEC_FILE[test_$label]=$(spec "test_$label" "$TEST" 0 "$EVAL_MAX_STEPS" "$label=$path")
  SPEC_FILE[validation_$label]=$(spec "validation_$label" "$VALIDATION" 0 "$EVAL_MAX_STEPS" "$label=$path")
  SPEC_FILE[hard_$label]=$(spec "hard_$label" "$HARD" 0 "$EVAL_MAX_STEPS" "$label=$path")
  ORDER="$ORDER test_$label validation_$label hard_$label"
fi
[[ -z "${ONLY:-}" ]] || ORDER=$ONLY
submitted=()
for name in $ORDER; do
  [[ -n "${SPEC_FILE[$name]:-}" ]] || { echo "unknown evaluation: $name" >&2; exit 2; }
  id=$(job_id "$(sbatch --parsable --export=ALL,EVAL_SPEC="${SPEC_FILE[$name]}" stage_eval_suite.sbatch)")
  submitted+=("$name=$id")
done
echo "extras-20261027 ${submitted[*]} submitted_at=$(date -u +%FT%TZ)" | tee -a submitted_jobs.txt
squeue -u "$USER" -o "%.12i %.10j %.8T %.10M %.24R"
