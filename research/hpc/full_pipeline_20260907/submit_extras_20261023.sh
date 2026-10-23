#!/usr/bin/env bash
# Submit the conventional-only cell's follow-ups (2026-10-23), after its chain:
#   fp-xcpu (CPU): the extra suites and the fresh expert aggregate
#     -> fp-ctrain: the matched cloning control
#   fp-xeval (one per spec, results in out/extra/evaluations/<name>):
#     test_control           the control on the 192 test roots (R1's evaluation rows and seed)
#     hard_bc0_r1            BC0, R1 and the teacher on the hard-draw roots       (after fp-xcpu)
#     hard_r2_control        R2 and the control on the hard-draw roots            (after fp-ctrain)
#     validation_bc0_r1      BC0, R1 and the teacher on the validation roots      (after fp-xcpu)
#     validation_r2_control  R2 and the control on the validation roots           (after fp-ctrain)
# Every job is output-guarded, so resubmitting after a failure redoes only what
# is missing.  ONLY="name ..." submits just those evaluations (their inputs
# must exist).
set -euo pipefail
export PATH=/opt/slurm/bin:$PATH
PIPE=/scratch/yx3882/research_full_pipeline_20260912
cd "$PIPE"
# shellcheck source=pipeline.env
source "$PIPE/pipeline.env"
[[ -s "$OUT/r2/round_summary.json" ]] || { echo "the cell's chain has not finished" >&2; exit 2; }
EXTRA=$OUT/extra
SPECS=$EXTRA/specs
mkdir -p "$SPECS"
job_id() { local id=${1%%;*}; [[ "$id" =~ ^[0-9]+$ ]] || { echo "sbatch returned no job id: $1" >&2; exit 2; }; printf '%s' "$id"; }
spec() {
  local name=$1 suite=$2 expert=$3
  shift 3
  printf 'EVAL_NAME=%s\nEVAL_SUITE=%s\nEVAL_POLICIES="%s"\nEVAL_EXPERT=%s\n' "$name" "$suite" "$*" "$expert" > "$SPECS/$name.env"
  printf '%s' "$SPECS/$name.env"
}
BC0=$BC0_DIR/lora
R1=$R1_DIR/training/lora
R2=$R2_DIR/training/lora
CONTROL=$EXTRA/control/training/lora
TEST=$R1_DIR/collection/development_scenarios.json
VALIDATION=$EXTRA/suites/validation.json
HARD=$EXTRA/suites/hard.json
declare -A SPEC_FILE=(
  [test_control]=$(spec test_control "$TEST" 0 "control=$CONTROL")
  [hard_bc0_r1]=$(spec hard_bc0_r1 "$HARD" 1 "bc0=$BC0" "r1=$R1")
  [hard_r2_control]=$(spec hard_r2_control "$HARD" 0 "r2=$R2" "control=$CONTROL")
  [validation_bc0_r1]=$(spec validation_bc0_r1 "$VALIDATION" 1 "bc0=$BC0" "r1=$R1")
  [validation_r2_control]=$(spec validation_r2_control "$VALIDATION" 0 "r2=$R2" "control=$CONTROL")
)
submitted=()
if [[ -z "${ONLY:-}" ]]; then
  xcpu=$(job_id "$(sbatch --parsable stage_extra_cpu.sbatch)")
  ctrain=$(job_id "$(sbatch --parsable --dependency=afterok:"$xcpu" stage_control_train.sbatch)")
  submitted+=("xcpu=$xcpu" "ctrain=$ctrain")
  declare -A AFTER=([test_control]=$ctrain [hard_bc0_r1]=$xcpu [hard_r2_control]=$ctrain
                    [validation_bc0_r1]=$xcpu [validation_r2_control]=$ctrain)
  ORDER="hard_bc0_r1 validation_bc0_r1 test_control hard_r2_control validation_r2_control"
else
  declare -A AFTER=()
  ORDER=$ONLY
fi
for name in $ORDER; do
  [[ -n "${SPEC_FILE[$name]:-}" ]] || { echo "unknown evaluation: $name" >&2; exit 2; }
  dependency=()
  [[ -n "${AFTER[$name]:-}" ]] && dependency=(--dependency=afterok:"${AFTER[$name]}")
  id=$(job_id "$(sbatch --parsable "${dependency[@]}" --export=ALL,EVAL_SPEC="${SPEC_FILE[$name]}" stage_eval_suite.sbatch)")
  submitted+=("$name=$id")
done
echo "extras ${submitted[*]} submitted_at=$(date -u +%FT%TZ)" | tee -a submitted_jobs.txt
squeue -u "$USER" -o "%.12i %.10j %.8T %.10M %.24R"
