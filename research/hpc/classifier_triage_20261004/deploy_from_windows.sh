#!/usr/bin/env bash
# Deploy the LLM leg of the triage benchmark on torch from the Windows checkout and submit its jobs.
#
#   bash research/hpc/classifier_triage_20261004/deploy_from_windows.sh <commit40> [variant ...]
#
# Needs a live WSL SSH master for the torch alias (scripts/start_torch_ssh_master.ps1).
# Steps: a local clone of the ranked cell's source as the work directory's source,
# the incremental git bundle uploaded through ssh stdin and checksum-verified, the
# built datasets (output/classifier_triage_20261004/llm/<variant>) uploaded as one
# compressed stream, then one job per variant.  SUBMIT=0 deploys without submitting.
#
# Remote commands that need a remote "$" (command substitution, PATH) go through a
# heredoc on stdin: wsl hands an inline command to its shell inside double quotes,
# so "$..." there would expand on the WSL side.
set -euo pipefail
COMMIT=${1:?40-hex commit to deploy}
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "commit must be 40 hex" >&2; exit 2; }
shift
VARIANTS=("$@")
[[ ${#VARIANTS[@]} -gt 0 ]] || VARIANTS=(prompt_top5 prompt_top10_signed)
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
BRANCH=${BRANCH:-codex/classifier-triage-20261004}
BASE=${BASE:-a76f704}                       # the commit the ranked cell's source is at
WORK=${WORK:-/scratch/yx3882/classifier_triage_20261004}
SOURCE_CELL=${SOURCE_CELL:-/scratch/yx3882/research_full_pipeline_20261001_ranked}
DATA_LOCAL=${DATA_LOCAL:-output/classifier_triage_20261004/llm}
SHORT=${COMMIT:0:7}
BUNDLE_LOCAL=output/deploy_triage_${SHORT}.bundle
export MSYS_NO_PATHCONV=1

cd "$REPO"
[[ "$(git rev-parse "$BRANCH")" == "$COMMIT" ]] || { echo "$BRANCH is not at $COMMIT" >&2; exit 2; }
for variant in "${VARIANTS[@]}"; do
  for name in train.jsonl validation.jsonl score.jsonl score_labels.json; do
    [[ -s "$DATA_LOCAL/$variant/$name" ]] || { echo "missing $DATA_LOCAL/$variant/$name" >&2; exit 2; }
  done
done
git bundle create "$BUNDLE_LOCAL" "$BASE..$BRANCH"
SHA_LOCAL=$(sha256sum "$BUNDLE_LOCAL" | cut -d' ' -f1)
echo "bundle $(stat -c %s "$BUNDLE_LOCAL") bytes sha256 $SHA_LOCAL"

wsl -- ssh -o NumberOfPasswordPrompts=0 torch "echo ssh_ok" || { echo "no SSH master: run scripts/start_torch_ssh_master.ps1 first" >&2; exit 2; }
wsl -- ssh torch "set -e; mkdir -p $WORK/logs $WORK/out $WORK/data; if [ ! -d $WORK/source/.git ]; then git clone -q $SOURCE_CELL/source $WORK/source; fi; echo source_ready"
wsl -- ssh torch "cat > $WORK/deploy_${SHORT}.bundle" < "$BUNDLE_LOCAL"
SHA_REMOTE=$(wsl -- ssh torch "sha256sum $WORK/deploy_${SHORT}.bundle | cut -d' ' -f1")
[[ "$SHA_LOCAL" == "$SHA_REMOTE" ]] || { echo "bundle upload corrupted" >&2; exit 2; }
wsl -- ssh torch bash -s <<EOF
set -e
git -C $WORK/source fetch -q $WORK/deploy_${SHORT}.bundle $BRANCH
git -C $WORK/source -c advice.detachedHead=false checkout -q --detach FETCH_HEAD
test "\$(git -C $WORK/source rev-parse HEAD)" = "$COMMIT"
echo source_at_$SHORT
EOF

tar czf - -C "$DATA_LOCAL" "${VARIANTS[@]}" | wsl -- ssh torch "tar xzf - -C $WORK/data"
for variant in "${VARIANTS[@]}"; do
  LINES_LOCAL=$(wc -l < "$DATA_LOCAL/$variant/train.jsonl")
  LINES_REMOTE=$(wsl -- ssh torch bash -s <<EOF
wc -l < $WORK/data/$variant/train.jsonl
EOF
)
  [[ "$LINES_LOCAL" -eq "$LINES_REMOTE" ]] || { echo "dataset upload of $variant is incomplete ($LINES_REMOTE of $LINES_LOCAL)" >&2; exit 2; }
done
echo "datasets uploaded: ${VARIANTS[*]}"

if [[ "${SUBMIT:-1}" == "1" ]]; then
  for variant in "${VARIANTS[@]}"; do
    wsl -- ssh torch bash -s <<EOF
export PATH=/opt/slurm/bin:\$PATH
cd $WORK
sbatch --parsable --export=ALL,VARIANT=$variant --job-name=triage-$variant source/research/hpc/classifier_triage_20261004/llm_triage.sbatch
EOF
  done
  wsl -- ssh torch bash -s <<EOF
export PATH=/opt/slurm/bin:\$PATH
squeue -u yx3882 -o '%.10i %.28j %.9T %.10M %.20R'
EOF
fi
echo "logs: $WORK/logs; results: $WORK/out/<variant>/scores.json"
