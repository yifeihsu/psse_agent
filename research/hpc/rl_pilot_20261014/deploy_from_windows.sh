#!/usr/bin/env bash
# Deploy the RL pilot on torch from the Windows checkout and submit its job.
#
#   bash research/hpc/rl_pilot_20261014/deploy_from_windows.sh <commit40> [config.json ...]
#
# Needs a live WSL SSH master for the torch alias (scripts/start_torch_ssh_master.ps1).
# Steps: a local clone of the classifier cell's source as the work directory's
# source, the incremental git bundle uploaded through ssh stdin and
# checksum-verified, the configs under research/hpc/rl_pilot_20261014/*.json and
# the expert reference (output/classifier_triage_20261013/e2e_ieee14_spectra_first/
# episodes.jsonl) uploaded, then one job per config.  SUBMIT=0 deploys without
# submitting; ITERATIONS sets the job's iteration count.
set -euo pipefail
COMMIT=${1:?40-hex commit to deploy}
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "commit must be 40 hex" >&2; exit 2; }
shift
CONFIGS=("$@")
[[ ${#CONFIGS[@]} -gt 0 ]] || CONFIGS=(rl_pilot.json)
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
BRANCH=${BRANCH:-codex/classifier-triage-20261004}
BASE=${BASE:-1003f7f}                       # the commit the classifier cell's source is at
WORK=${WORK:-/scratch/yx3882/rl_pilot_20261014}
SOURCE_CELL=${SOURCE_CELL:-/scratch/yx3882/research_full_pipeline_20261007_classifier}
REFERENCE_LOCAL=${REFERENCE_LOCAL:-output/classifier_triage_20261013/e2e_ieee14_spectra_first/episodes.jsonl}
SHORT=${COMMIT:0:7}
BUNDLE_LOCAL=output/deploy_rl_${SHORT}.bundle
export MSYS_NO_PATHCONV=1

cd "$REPO"
[[ "$(git rev-parse "$BRANCH")" == "$COMMIT" ]] || { echo "$BRANCH is not at $COMMIT" >&2; exit 2; }
for config in "${CONFIGS[@]}"; do
  [[ -s "research/hpc/rl_pilot_20261014/$config" ]] || { echo "no config $config" >&2; exit 2; }
done
[[ -s "$REFERENCE_LOCAL" ]] || { echo "missing expert reference $REFERENCE_LOCAL" >&2; exit 2; }
git bundle create "$BUNDLE_LOCAL" "$BASE..$BRANCH"
SHA_LOCAL=$(sha256sum "$BUNDLE_LOCAL" | cut -d' ' -f1)
echo "bundle $(stat -c %s "$BUNDLE_LOCAL") bytes sha256 $SHA_LOCAL"

wsl -- ssh -o NumberOfPasswordPrompts=0 torch "echo ssh_ok" || { echo "no SSH master: run scripts/start_torch_ssh_master.ps1 first" >&2; exit 2; }
wsl -- ssh torch "set -e; mkdir -p $WORK/logs $WORK/out $WORK/configs $WORK/data/expert_reference; if [ ! -d $WORK/source/.git ]; then git clone -q $SOURCE_CELL/source $WORK/source; fi; echo source_ready"
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
for config in "${CONFIGS[@]}"; do
  wsl -- ssh torch "cat > $WORK/configs/$config" < "research/hpc/rl_pilot_20261014/$config"
done
wsl -- ssh torch "cat > $WORK/data/expert_reference/episodes.jsonl" < "$REFERENCE_LOCAL"
echo "configs and expert reference uploaded"

if [[ "${SUBMIT:-1}" == "1" ]]; then
  for config in "${CONFIGS[@]}"; do
    wsl -- ssh torch bash -s <<EOF
export PATH=/opt/slurm/bin:\$PATH
cd $WORK
sbatch --parsable --export=ALL,CONFIG=$config,ITERATIONS=${ITERATIONS:-2} --job-name=rl-${config%.json} source/research/hpc/rl_pilot_20261014/rl_pilot.sbatch
EOF
  done
  wsl -- ssh torch bash -s <<EOF
export PATH=/opt/slurm/bin:\$PATH
squeue -u yx3882 -o '%.10i %.28j %.9T %.10M %.20R'
EOF
fi
echo "logs: $WORK/logs; run: $WORK/out/<config>/manifest.json and iteration_*/"
