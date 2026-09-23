#!/usr/bin/env bash
#SBATCH --job-name=fp-d0
#SBATCH --account=torch_pr_627_general
#SBATCH --partition=cs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/scratch/yx3882/research_full_pipeline_20260912/logs/d0-%j.out
# Full pipeline, stage 0 (CPU only): the expert aggregate that trains BC0,
# then the DAgger suite drawn around it.
#
# The round-0 generator rolls the expert out on every D0 root (beta = 1),
# forces the recovery classes through counterfactual branches, audits every
# episode against truth, and splits the roots into train / validation / test.
# It is not resumable, so a requeue restarts it; the suite builder then draws
# the per-round training roots and the shared development roots excluding
# every D0 and protected root.
set -euo pipefail
PIPE=/scratch/yx3882/research_full_pipeline_20260912
# shellcheck source=pipeline.env
source "$PIPE/pipeline.env"
pipeline_environment
mkdir -p "$OUT" "$LOGS"
echo "[d0] job ${SLURM_JOB_ID:-manual} restart ${SLURM_RESTART_COUNT:-0} node ${SLURMD_NODENAME:-?} cpus ${SLURM_CPUS_PER_TASK:-?} $(date -u +%FT%TZ)"

if [[ ! -s "$OUT/prerequisites.json" ]]; then
  bash "$PIPE/prerequisites.sh" --with-tests --output "$OUT/prerequisites.json"
fi
opendss_local_model
stage_cleanup() { rm -rf "$OPENDSS_LOCAL"; }
trap stage_cleanup EXIT

# A non-IEEE14 system draws from a fresh balanced corpus; build it here when
# pipeline.env names none (output-guarded on its samples file).
if [[ "$SYSTEM" != case14 && -z "$MEASUREMENT_CORPUS" ]]; then
  CORPUS_DIR=${BALANCED_ARTIFACT_DIR:-$OUT/corpus}
  if [[ ! -s "$CORPUS_DIR/samples.jsonl" ]]; then
    rm -rf "$CORPUS_DIR"
    (cd "$SRC" && "$PY" scripts/build_balanced_corpus.py --system "$SYSTEM" \
      --output-dir "$CORPUS_DIR" --seed "$SEED_D0" --counts "$CORPUS_COUNTS" \
      --num-scans "$CORPUS_NUM_SCANS" > "$OUT/corpus_console.json")
  fi
  MEASUREMENT_CORPUS=$CORPUS_DIR/samples.jsonl
  BALANCED_ARTIFACT_DIR=$CORPUS_DIR
fi

# System selection and detector (pipeline.env): the fresh-corpus sources of
# a non-IEEE14 system, the shared normalized-residual threshold, and the
# IEEE-14 waveform corpora for case14 only.
system_args=(--system "$SYSTEM" --admission-mode "$ADMISSION_MODE" --evidence-profile "$EVIDENCE_PROFILE" --hif-signature-mode "$HIF_SIGNATURE_MODE"
  --normalized-residual-threshold "$NORMALIZED_RESIDUAL_THRESHOLD")
if [[ -n "$MEASUREMENT_CORPUS" ]]; then system_args+=(--measurement-corpus "$MEASUREMENT_CORPUS"); fi
if [[ -n "$BALANCED_ARTIFACT_DIR" ]]; then system_args+=(--balanced-artifact-dir "$BALANCED_ARTIFACT_DIR"); fi
waveform_args=()
if [[ "$SYSTEM" == case14 ]]; then
  waveform_args=(--hif-corpus "$HIF_CORPUS_TRAIN" --hif-corpus "$HIF_CORPUS_VALID" --imbalance-corpus "$IMBALANCE_CORPUS")
  waveform_args+=(--hif-corpus "$HIF_CORPUS_TRAIN_EXTRA" --hif-corpus "$HIF_CORPUS_VALID_EXTRA")
fi

cd "$SRC"
reuse_previous_stage d0 d0.done
if [[ ! -s "$OUT/d0.done" ]]; then
  rm -rf "$D0_DIR"
  mkdir -p "$D0_DIR"
  printf '%s\n' "$D0_PLAN" > "$OUT/d0_plan.json"
  "$PY" -m psse_env.examples.generate_round0_aggregate \
    --research \
    --output-dir "$D0_DIR" \
    --plan "$OUT/d0_plan.json" \
    --seed "$SEED_D0" \
    --protocol canonical \
    --max-steps "$D0_MAX_STEPS" \
    --counterfactuals-per-scenario "$D0_COUNTERFACTUALS" \
    --hif-alpha-grid "$HIF_ALPHA_GRID" \
    --hif-r-grid "$HIF_R_GRID" \
    --hif-max-scans "$HIF_MAX_SCANS" \
    --min-measurement-error-sigma "$MEASUREMENT_ERROR_MIN_SIGMA" \
    --topology-effects "$TOPOLOGY_EFFECTS" \
    "${system_args[@]}" \
    "${waveform_args[@]}" \
    > "$OUT/d0_console.json"
  "$PY" - "$D0_DIR" "$OUT/d0.done" "$EVIDENCE_PROFILE" <<'PY'
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

d0 = Path(sys.argv[1])
def rows(name):
    path = d0 / name
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
def root(row):
    for container in (row, row.get("metadata") or {}):
        if isinstance(container, dict) and container.get("physical_root_fingerprint"):
            return str(container["physical_root_fingerprint"])
    return ""
raw = rows("aggregate.raw.jsonl")
train = rows("aggregate.train_view.jsonl")
validation = rows("aggregate.validation.jsonl")
test = rows("aggregate.test.jsonl")
if not raw or not train or not validation:
    raise SystemExit("the expert aggregate did not produce raw, train, and validation rows")
train_roots = {root(r) for r in train}
validation_roots = {root(r) for r in validation}
if train_roots & validation_roots:
    raise SystemExit("train and validation roots overlap")
provenance = json.loads((d0 / "aggregate.generation_provenance.json").read_text(encoding="utf-8"))
payload = {
    "contract": "research_full_pipeline_d0_done_v1",
    "evidence_profile": sys.argv[3],
    "raw_rows": len(raw),
    "raw_roots": len({root(r) for r in raw}),
    "raw_families": dict(sorted(Counter(r.get("scenario_family") for r in raw).items())),
    "train_rows": len(train),
    "train_roots": len(train_roots),
    "train_families": dict(sorted(Counter((r.get("metadata") or {}).get("scenario_family") or r.get("scenario_family") for r in train).items())),
    "validation_rows": len(validation),
    "validation_roots": len(validation_roots),
    "test_rows": len(test),
    "release_eligible": provenance.get("release_eligible"),
    "release_failures": provenance.get("release_failures"),
    "train_view_sha256": hashlib.sha256((d0 / "aggregate.train_view.jsonl").read_bytes()).hexdigest(),
    "raw_sha256": hashlib.sha256((d0 / "aggregate.raw.jsonl").read_bytes()).hexdigest(),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
}
output = Path(sys.argv[2])
fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, output)
print(json.dumps({k: payload[k] for k in ("raw_rows", "raw_roots", "train_rows", "validation_rows", "test_rows", "release_eligible")}))
PY
fi

if [[ ! -s "$OUT/suite.done" ]]; then
  protected=(--protected-suite "$BC0_SUITE")
  if [[ -s "$TRACE_VALIDATION" ]]; then protected+=(--protected-suite "$TRACE_VALIDATION"); fi
  suite_waveform_args=()
  if [[ "$SYSTEM" == case14 ]]; then
    suite_waveform_args=(--hif-sample-paths "$HIF_CORPUS_TRAIN" "$HIF_CORPUS_VALID")
    suite_waveform_args+=("$HIF_CORPUS_TRAIN_EXTRA" "$HIF_CORPUS_VALID_EXTRA")
    suite_waveform_args+=(--imbalance-sample-path "$IMBALANCE_CORPUS")
  fi
  "$PY" "$PIPE/build_suite.py" \
    --source-root "$SRC" \
    --d0-raw "$D0_DIR/aggregate.raw.jsonl" \
    "${suite_waveform_args[@]}" \
    "${protected[@]}" \
    --round-train-plan "$ROUND_TRAIN_PLAN" \
    --development-plan "$DEVELOPMENT_PLAN" \
    --rounds "$DAGGER_ROUNDS" \
    --seed "$SEED_SUITE" \
    --candidate-multiplier "$CANDIDATE_MULTIPLIER" \
    --training-threshold "$SUITE_TRAINING_THRESHOLD" \
    --development-threshold "$SUITE_DEVELOPMENT_THRESHOLD" \
    --development-rank-allowance "$SUITE_DEVELOPMENT_RANK_ALLOWANCE" \
    --min-measurement-error-sigma "$MEASUREMENT_ERROR_MIN_SIGMA" \
    --topology-effects "$TOPOLOGY_EFFECTS" \
    "${system_args[@]}" \
    --output-dir "$SUITE_DIR" \
    > "$OUT/suite_console.json"
  cp "$SUITE_DIR/manifest.json" "$OUT/suite.done"
fi

# A frozen student from an earlier run stands in for BC0: link it as this
# run's adapter and write the receipt stage 1 checks, so no training runs.
if [[ -n "$FROZEN_STUDENT_ADAPTER" && ! -s "$OUT/bc0.done" ]]; then
  [[ "${FROZEN_STUDENT_EVIDENCE_PROFILE:-auxiliary_diagnostics}" == "$EVIDENCE_PROFILE" ]] || { echo "frozen student evidence profile differs; use a compatible checkpoint or a separate explicit historical evaluation" >&2; exit 2; }
  [[ -s "$FROZEN_STUDENT_ADAPTER/adapter_model.safetensors" ]] || { echo "frozen student adapter missing: $FROZEN_STUDENT_ADAPTER" >&2; exit 2; }
  mkdir -p "$BC0_DIR"
  ln -sfn "$FROZEN_STUDENT_ADAPTER" "$BC0_DIR/lora"
  "$PY" - "$BC0_DIR" "$OUT/bc0.done" "$FROZEN_STUDENT_ADAPTER" "$EVIDENCE_PROFILE" <<'PY'
import json
import os
import sys
import tempfile
from pathlib import Path
from psse_env.dagger.release_factories import checkpoint_tree_sha256

bc0 = Path(sys.argv[1])
frozen = Path(sys.argv[3]).resolve()
payload = {
    "contract": "research_full_pipeline_bc0_reused_v1",
    "evidence_profile": sys.argv[4],
    "adapter_path": str(bc0 / "lora"),
    "frozen_student_adapter": str(frozen),
    "adapter_tree_sha256": checkpoint_tree_sha256(bc0 / "lora"),
    "trained_in_this_run": False,
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
}
output = Path(sys.argv[2])
fd, temporary = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
with os.fdopen(fd, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
os.replace(temporary, output)
print(json.dumps({k: payload[k] for k in ("frozen_student_adapter", "adapter_tree_sha256")}))
PY
fi
echo fp-d0-complete
