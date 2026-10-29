#!/usr/bin/env bash
#SBATCH --job-name=fp-xdraw
#SBATCH --account=torch_pr_627_general
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/scratch/yx3882/research_full_pipeline_20260912/logs/xdraw-%j.out
# A second test draw for a finished cell (conventional-only cell, 2026-10-29;
# note section 8).  Steps, each output-guarded:
#  1. collect_reference_roots.py: every physical root of every earlier set
#     (TEST_DRAW_REFERENCES: patterns for every cell's expert aggregate and
#     suites, this cell's validation, hard and control sets, the hard draw)
#     into $DRAW/protected_roots.json;
#  2. build_suite.py exactly as stage 0 drew the first suite (the same round
#     and development plans, thresholds, rank allowance, admission, meter
#     floor, topology effects, system and corpora) with seed TEST_DRAW_SEED,
#     the cell's D0 as --d0-raw and the collected roots protected, into
#     $DRAW/suite.  If a family runs short of fresh candidates after the
#     exclusions (the builder's "Increase --candidate-multiplier"), the draw
#     is repeated once with TEST_DRAW_FALLBACK_MULTIPLIER: the candidates are
#     the same generator's identically distributed draws, so a larger pool
#     changes which are kept, not their distribution.  The round training
#     files the builder writes are a by-product of the identical procedure
#     and are not used;
#  3. check_suite_exclusion.py, the first draw's check: the new suites
#     against the cell's D0 and each other, and the new development set
#     against every readable reference file and the collected roots;
#  4. suite_stats.py on both draws, and the receipt $DRAW/draw.done.
set -euo pipefail
PIPE=/scratch/yx3882/research_full_pipeline_20260912
# shellcheck source=pipeline.env
source "$PIPE/pipeline.env"
pipeline_environment
: "${TEST_DRAW_SEED:?}" "${TEST_DRAW_NAME:?}" "${TEST_DRAW_REFERENCES:?}" "${TEST_DRAW_FALLBACK_MULTIPLIER:?}"
EXTRA=$OUT/extra
DRAW=$EXTRA/$TEST_DRAW_NAME
mkdir -p "$DRAW" "$LOGS"
echo "[xdraw:$TEST_DRAW_NAME] job ${SLURM_JOB_ID:-manual} restart ${SLURM_RESTART_COUNT:-0} node ${SLURMD_NODENAME:-?} seed $TEST_DRAW_SEED $(date -u +%FT%TZ)"
[[ -s "$OUT/d0.done" && -s "$OUT/suite.done" && -s "$EXTRA/suites/report.json" ]] || { echo "stage 0 or the 2026-10-23 suites are missing" >&2; exit 2; }
opendss_local_model
stage_cleanup() { rm -rf "$OPENDSS_LOCAL"; }
trap stage_cleanup EXIT

system_args=(--system "$SYSTEM" --admission-mode "$ADMISSION_MODE" --evidence-profile "$EVIDENCE_PROFILE" --hif-signature-mode "$HIF_SIGNATURE_MODE"
  --normalized-residual-threshold "$NORMALIZED_RESIDUAL_THRESHOLD" --progress-basis "$PROGRESS_BASIS")
if [[ -n "$MEASUREMENT_CORPUS" ]]; then system_args+=(--measurement-corpus "$MEASUREMENT_CORPUS"); fi
if [[ -n "$BALANCED_ARTIFACT_DIR" ]]; then system_args+=(--balanced-artifact-dir "$BALANCED_ARTIFACT_DIR"); fi
waveform_corpus_files
suite_waveform_args=()
if (( ${#HIF_CORPUS_FILES[@]} )); then suite_waveform_args=(--hif-sample-paths "${HIF_CORPUS_FILES[@]}"); fi
if [[ -n "$IMBALANCE_CORPUS" ]]; then suite_waveform_args+=(--imbalance-sample-path "$IMBALANCE_CORPUS"); fi

cd "$SRC"
# 1. The roots of every earlier set (patterns passed literally: the collector expands them).
if [[ ! -s "$DRAW/protected_roots.report.json" ]]; then
  set -f
  # shellcheck disable=SC2086
  "$PY" research/hpc/full_pipeline_20260907/collect_reference_roots.py --exclude-dir "$DRAW" \
    --output "$DRAW/protected_roots.json" $TEST_DRAW_REFERENCES
  set +f
fi

# 2. The draw: stage 0's builder invocation with the new seed and the collected roots protected.
if [[ ! -s "$DRAW/suite/manifest.json" ]]; then
  protected=(--protected-suite "$BC0_SUITE")
  if [[ -s "$TRACE_VALIDATION" ]]; then protected+=(--protected-suite "$TRACE_VALIDATION"); fi
  protected+=(--protected-suite "$DRAW/protected_roots.json")
  for multiplier in "$CANDIDATE_MULTIPLIER" "$TEST_DRAW_FALLBACK_MULTIPLIER"; do
    rm -rf "$DRAW/suite"
    set +e
    "$PY" "$PIPE/build_suite.py" \
      --source-root "$SRC" \
      --d0-raw "$D0_DIR/aggregate.raw.jsonl" \
      "${suite_waveform_args[@]}" \
      "${protected[@]}" \
      --round-train-plan "$ROUND_TRAIN_PLAN" \
      --development-plan "$DEVELOPMENT_PLAN" \
      --rounds "$DAGGER_ROUNDS" \
      --seed "$TEST_DRAW_SEED" \
      --candidate-multiplier "$multiplier" \
      --training-threshold "$SUITE_TRAINING_THRESHOLD" \
      --development-threshold "$SUITE_DEVELOPMENT_THRESHOLD" \
      --development-rank-allowance "$SUITE_DEVELOPMENT_RANK_ALLOWANCE" \
      --min-measurement-error-sigma "$MEASUREMENT_ERROR_MIN_SIGMA" \
      --topology-effects "$TOPOLOGY_EFFECTS" \
      "${system_args[@]}" \
      --output-dir "$DRAW/suite" \
      > "$DRAW/console.json" 2> "$DRAW/build_m$multiplier.stderr"
    status=$?
    set -e
    [[ "$status" == 0 ]] && break
    if ! grep -q "Increase --candidate-multiplier" "$DRAW/build_m$multiplier.stderr"; then
      tail -30 "$DRAW/build_m$multiplier.stderr" >&2
      exit 2
    fi
    echo "[xdraw:$TEST_DRAW_NAME] candidate multiplier $multiplier ran short of fresh roots: $(grep -o "Scenario family.*" "$DRAW/build_m$multiplier.stderr" | tail -1)"
  done
  [[ -s "$DRAW/suite/manifest.json" ]] || { echo "the draw failed at both candidate multipliers" >&2; exit 2; }
fi

# 3. The first draw's disjointness check, against every readable reference file and the collected roots.
references=()
while IFS= read -r path; do
  [[ -n "$path" ]] && references+=(--reference-suite "$path")
done < <("$PY" -c 'import json, sys; print("\n".join(json.load(open(sys.argv[1]))["checkable_paths"]))' "$DRAW/protected_roots.report.json")
references+=(--reference-suite "$DRAW/protected_roots.json")
set +e
"$PY" research/hpc/full_pipeline_20260907/check_suite_exclusion.py --d0-raw "$D0_DIR/aggregate.raw.jsonl" \
  --suite-dir "$DRAW/suite" "${references[@]}" > "$DRAW/disjointness_check.json"
check_status=$?
set -e
[[ "$check_status" == 0 ]] || { echo "the new draw shares a root with an earlier set (disjointness_check.json)" >&2; exit 2; }

# 4. Suite statistics of both draws and the receipt.
"$PY" research/hpc/full_pipeline_20260907/suite_stats.py --suite "draw1=$SUITE_DIR/development.json" \
  --suite "$TEST_DRAW_NAME=$DRAW/suite/development.json" --output "$DRAW/suite_stats.json" > /dev/null
"$PY" - "$DRAW" "$TEST_DRAW_SEED" "$EVIDENCE_PROFILE" "$EXPERT_VARIANT" <<'PY'
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

draw = Path(sys.argv[1])
rows = json.loads((draw / "suite" / "development.json").read_text(encoding="utf-8"))
manifest = json.loads((draw / "suite" / "manifest.json").read_text(encoding="utf-8"))
check = json.loads((draw / "disjointness_check.json").read_text(encoding="utf-8"))
references = json.loads((draw / "protected_roots.report.json").read_text(encoding="utf-8"))
if not check.get("passed"):
    raise SystemExit("disjointness check failed")
def root(row):
    return str((row.get("grouping") or {}).get("physical_root_fingerprint") or "")
payload = {
    "contract": "conventional_only_test_draw_done_v1",
    "name": draw.name,
    "seed": int(sys.argv[2]),
    "development_generator_seed": int(sys.argv[2]) + 1,
    "candidate_multiplier": manifest.get("candidate_multiplier"),
    "evidence_profile": sys.argv[3],
    "expert_variant": sys.argv[4],
    "development_rows": len(rows),
    "development_roots": len({root(r) for r in rows}),
    "families": dict(sorted(Counter(str((r.get("grouping") or {}).get("scenario_family")) for r in rows).items())),
    "strata": (manifest.get("files") or {}).get("development", {}).get("strata"),
    "development_sha256": hashlib.sha256((draw / "suite" / "development.json").read_bytes()).hexdigest(),
    "manifest_source_commit": manifest.get("source_commit"),
    "protected_roots": manifest.get("protected_roots"),
    "reference_roots": references.get("roots"),
    "reference_files_with_roots": references.get("files_with_roots"),
    "reference_files_checked": len(check.get("development_against_references") or {}),
    "disjointness_passed": bool(check.get("passed")),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
}
(draw / "draw.done").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({k: payload[k] for k in ("development_rows", "families", "strata", "candidate_multiplier",
                                          "reference_roots", "reference_files_checked")}))
PY
echo "fp-xdraw-$TEST_DRAW_NAME-complete"
