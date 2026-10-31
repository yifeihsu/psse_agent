#!/usr/bin/env bash
# Submit the prompted, untuned base model's evaluations (2026-10-31; note
# section 11): the cell's base model with no adapter
# (research_policy_factory.NO_ADAPTER), alone, through the students' harness,
# under the cell's seed, evaluator and worker rule:
#   test_40_<label>, <prefix>_40_<label>   the first and second test draw at the cell's horizon
#   test_budget12_<label>, test_budget16_<label>, test_budget20_<label>
#                                          the first draw at 12, 16 and 20 actions
#   <prefix>_budget12_<label>, <prefix>_budget20_<label>
#                                          the second draw at 12 and 20 actions
# DRYRUN=1 submits only validation_dryrun_<label>: the first validation root of
# each family, at the cell's horizon (a crash and parse check, not a result).
# ONLY="name ..." submits just those.  A name whose evaluation directory,
# console receipt or spec exists is refused (stage_eval_suite.sbatch skips an
# evaluation whose summary exists).
set -euo pipefail
export PATH=/opt/slurm/bin:$PATH
PIPE=/scratch/yx3882/research_full_pipeline_20260912
cd "$PIPE"
# shellcheck source=pipeline.env
source "$PIPE/pipeline.env"
: "${TEST_DRAW_NAME:?}" "${TEST_DRAW_PREFIX:?}"
LABEL=${LABEL:-base_prompted}
[[ "$LABEL" =~ ^[a-z0-9_]+$ ]] || { echo "LABEL must be lower-case letters, digits and underscores" >&2; exit 2; }
EXTRA=$OUT/extra
EVALS=$EXTRA/evaluations
SPECS=$EXTRA/specs
RECORDS=$PIPE/base_prompted_20261031
TEST1=$R1_DIR/collection/development_scenarios.json
TEST2=$EXTRA/$TEST_DRAW_NAME/suite/development.json
VALIDATION=$EXTRA/suites/validation.json
FULL=${EVAL_MAX_STEPS:?}
for suite in "$TEST1" "$TEST2" "$VALIDATION"; do
  [[ -s "$suite" ]] || { echo "suite missing: $suite" >&2; exit 2; }
done
mkdir -p "$RECORDS" "$SPECS"
if [[ "${DRYRUN:-0}" == 1 ]]; then
  DRY=$RECORDS/validation_dryrun.json
  python3 - "$VALIDATION" "$DRY" <<'PY'
import json
import sys

rows = json.load(open(sys.argv[1], encoding="utf-8"))
seen, subset = set(), []
for row in rows:
    family = (row.get("grouping") or {}).get("scenario_family")
    if family not in seen:
        seen.add(family)
        subset.append(row)
json.dump(subset, open(sys.argv[2], "w", encoding="utf-8"), indent=1)
print(f"dry-run suite: {len(subset)} roots, one per family: {sorted(seen)}")
PY
  PLAN=("validation_dryrun_${LABEL} $DRY $FULL")
else
  # name, rows, horizon
  PLAN=(
    "test_${FULL}_${LABEL} $TEST1 $FULL"
    "${TEST_DRAW_PREFIX}_${FULL}_${LABEL} $TEST2 $FULL"
    "test_budget12_${LABEL} $TEST1 12"
    "test_budget16_${LABEL} $TEST1 16"
    "test_budget20_${LABEL} $TEST1 20"
    "${TEST_DRAW_PREFIX}_budget12_${LABEL} $TEST2 12"
    "${TEST_DRAW_PREFIX}_budget20_${LABEL} $TEST2 20"
  )
fi
ONLY=${ONLY:-}
selected=()
for entry in "${PLAN[@]}"; do
  read -r name suite horizon <<< "$entry"
  [[ -z "$ONLY" || " $ONLY " == *" $name "* ]] || continue
  for taken in "$EVALS/$name" "$EVALS/$name.console.json" "$SPECS/$name.env"; do
    [[ ! -e "$taken" ]] || { echo "$taken exists" >&2; exit 2; }
  done
  selected+=("$entry")
done
(( ${#selected[@]} )) || { echo "no evaluation selected" >&2; exit 2; }
job_id() { local id=${1%%;*}; [[ "$id" =~ ^[0-9]+$ ]] || { echo "sbatch returned no job id: $1" >&2; exit 2; }; printf '%s' "$id"; }
submitted=()
for entry in "${selected[@]}"; do
  read -r name suite horizon <<< "$entry"
  printf 'EVAL_NAME=%s\nEVAL_SUITE=%s\nEVAL_POLICIES="%s"\nEVAL_EXPERT=0\nEVAL_MAX_STEPS=%s\n' \
    "$name" "$suite" "$LABEL=BASE_MODEL_NO_ADAPTER" "$horizon" > "$SPECS/$name.env"
  id=$(job_id "$(sbatch --parsable --export=ALL,EVAL_SPEC="$SPECS/$name.env" stage_eval_suite.sbatch)")
  submitted+=("$name=$id")
done
echo "base-prompted-evaluations ${submitted[*]} dryrun=${DRYRUN:-0} submitted_at=$(date -u +%FT%TZ)" | tee -a submitted_jobs.txt
squeue -u "$USER" -o "%.12i %.10j %.8T %.10M %.24R"
