#!/usr/bin/env bash
# Stage the full-pipeline cell on the cluster from a git bundle.
#
#   ssh torch bash -s -- BUNDLE BRANCH EXPECTED_COMMIT [PIPE_DIR] [OVERRIDES] < deploy_remote.sh
#
# Clones (or fetches into) PIPE_DIR/source, checks the commit, copies the
# stage scripts beside it with the cell's canonical directory rewritten to
# PIPE_DIR, stages an optional overrides file as pipeline.overrides.env,
# syntax-checks everything, and runs the dry-run prerequisites.
set -euo pipefail
BUNDLE=${1:?bundle path}
BRANCH=${2:?branch name}
EXPECTED=${3:?expected 40-hex commit}
SOURCE_PIPE=/scratch/yx3882/research_full_pipeline_20260912
PIPE=${4:-$SOURCE_PIPE}
OVERRIDES=${5:-}
CELL=research/hpc/full_pipeline_20260907
[[ -z "$OVERRIDES" || "$OVERRIDES" =~ ^[A-Za-z0-9_.-]+[.]env$ ]] || { echo "overrides must be a plain file name under overrides/" >&2; exit 2; }
[[ "$EXPECTED" =~ ^[0-9a-f]{40}$ ]] || { echo "expected commit must be 40 hex" >&2; exit 2; }
[[ "$PIPE" =~ ^/[A-Za-z0-9_./-]+$ ]] || { echo "pipeline directory must be an absolute plain path" >&2; exit 2; }
mkdir -p "$PIPE/logs" "$PIPE/out"
if [[ ! -d "$PIPE/source/.git" ]]; then
  git clone -q -b "$BRANCH" "$BUNDLE" "$PIPE/source"
else
  git -C "$PIPE/source" fetch -q "$BUNDLE" "$BRANCH"
  git -C "$PIPE/source" checkout -q --detach FETCH_HEAD
fi
HEAD=$(git -C "$PIPE/source" rev-parse HEAD)
[[ "$HEAD" == "$EXPECTED" ]] || { echo "deployed $HEAD differs from expected $EXPECTED" >&2; exit 2; }
printf '%s\n' "$HEAD" > "$PIPE/source_commit.txt"
for name in pipeline.env prerequisites.sh build_suite.py summarize.py submit_pipeline.sh \
  status_pipeline.sh stage_d0.sbatch stage_bc0.sbatch stage_collect.sbatch stage_train.sbatch stage_eval.sbatch \
  stage_zeroshot.sbatch; do
  sed "s#${SOURCE_PIPE}#${PIPE}#g" "$PIPE/source/$CELL/$name" > "$PIPE/$name"
done
if [[ -n "$OVERRIDES" ]]; then
  [[ -f "$PIPE/source/$CELL/overrides/$OVERRIDES" ]] || { echo "unknown overrides file: $OVERRIDES" >&2; exit 2; }
  # Copied verbatim: an overrides file names this run through $PIPE/$OUT and
  # other runs (a frozen adapter, a previous cell) by their absolute paths.
  cp "$PIPE/source/$CELL/overrides/$OVERRIDES" "$PIPE/pipeline.overrides.env"
else
  rm -f "$PIPE/pipeline.overrides.env"
fi
# Record the effective instrument capability after applying the staged overrides.
source "$PIPE/pipeline.env"
case "$EVIDENCE_PROFILE" in scada_only|auxiliary_diagnostics) ;; *) echo "unknown evidence profile" >&2; exit 2 ;; esac
case "$HIF_SIGNATURE_MODE" in discovered|flagged) ;; *) echo "unknown HIF signature mode" >&2; exit 2 ;; esac
printf '{"contract": "research_full_pipeline_deploy_v1", "pipeline_dir": "%s", "source_pipeline_dir": "%s", "source_commit": "%s", "branch": "%s", "overrides": "%s", "evidence_profile": "%s", "hif_signature_mode": "%s", "deployed_at_utc": "%s"}\n' \
  "$PIPE" "$SOURCE_PIPE" "$HEAD" "$BRANCH" "$OVERRIDES" "$EVIDENCE_PROFILE" "$HIF_SIGNATURE_MODE" "$(date -u +%FT%TZ)" > "$PIPE/deploy.json"
chmod +x "$PIPE"/*.sh
for f in "$PIPE"/*.sh "$PIPE"/*.sbatch "$PIPE"/pipeline.env "$PIPE"/pipeline.overrides.env; do
  [[ -f "$f" ]] || continue
  bash -n "$f"
done
python3 -m py_compile "$PIPE/summarize.py" "$PIPE/build_suite.py"
(cd "$PIPE" && sha256sum pipeline.env ./*.sh ./*.sbatch ./*.py > scripts.sha256)
bash "$PIPE/prerequisites.sh" --output "$PIPE/out/prerequisites.dryrun.json"
echo "deploy-complete $HEAD"
