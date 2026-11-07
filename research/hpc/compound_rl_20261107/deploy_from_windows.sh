#!/usr/bin/env bash
# Deploy the compound RL cell on torch from the Windows checkout.  Submits nothing (sbatch does).
#
#   bash research/hpc/compound_rl_20261107/deploy_from_windows.sh <commit40>
#
# Steps, as in the cloning cell's deploy: clone the cloning cell's source as this cell's (same
# objects, fast), upload the incremental git bundle through ssh stdin and verify its checksum (wsl
# scp cannot take a Windows path, and a pipe into wsl ssh hangs: files go in with <), then the
# suites and the configs the same way.  Needs a live WSL SSH master for the torch alias
# (scripts/start_torch_ssh_master.ps1) and the sources committed on BRANCH.
#
# TRAIN_SUITE names the compound training draw (default: the 2,550-case IEEE 14 draw of the main
# checkout, output/compound_rl_20261106/train_ieee14/suite.json), DEV_SUBSET the 300-case
# development subset dev_subset.py wrote (default: output/compound_rl_20261107/ieee14_dev300.json
# of this checkout; make it first).  A file already on torch with the same hash is not sent again.
set -euo pipefail
COMMIT=${1:?40-hex commit to deploy}
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "commit must be 40 hex" >&2; exit 2; }
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CELL_REL=research/hpc/compound_rl_20261107
WORK=/scratch/yx3882/compound_rl_20261107
BRANCH=${BRANCH:-claude/rl-lifecycle-evidence-20261106}
SOURCE_CELL=${SOURCE_CELL:-/scratch/yx3882/compound_cloning_20261106}
MAIN_CHECKOUT=${MAIN_CHECKOUT:-C:/Users/Holiday/Documents/ChatGPT/PSSE_Agent}
TRAIN_SUITE=${TRAIN_SUITE:-$MAIN_CHECKOUT/output/compound_rl_20261106/train_ieee14/suite.json}
DEV_SUBSET=${DEV_SUBSET:-$REPO/output/compound_rl_20261107/ieee14_dev300.json}
CONFIGS=${CONFIGS:-rl_smoke.json rl_ieee14.json rl_ieee14_sampled.json}
SHORT=${COMMIT:0:7}
BUNDLE_LOCAL=output/deploy_rl_${SHORT}.bundle
export MSYS_NO_PATHCONV=1

[[ -s "$TRAIN_SUITE" ]] || { echo "training suite missing: $TRAIN_SUITE" >&2; exit 2; }
[[ -s "$DEV_SUBSET" ]] || { echo "development subset missing: $DEV_SUBSET (run dev_subset.py first)" >&2; exit 2; }
for config in $CONFIGS; do
  [[ -s "$REPO/$CELL_REL/$config" ]] || { echo "config missing: $CELL_REL/$config" >&2; exit 2; }
  if grep -q $'\r' "$REPO/$CELL_REL/$config"; then echo "config has CR line endings: $config" >&2; exit 2; fi
done

cd "$REPO"
[[ "$(git rev-parse "$BRANCH")" == "$COMMIT" ]] || { echo "$BRANCH is not at $COMMIT" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no -- research psse_env scripts tools)" ]] \
  || { echo "commit the sources first (tracked modifications under research/, psse_env/, scripts/ or tools/)" >&2; exit 2; }
for path in "$CELL_REL/rl.sbatch" psse_env/rl/lifecycle.py; do
  git cat-file -e "$COMMIT:$path" 2>/dev/null || { echo "$path is not in $COMMIT: commit the cell and the driver first" >&2; exit 2; }
done

remote_sha() { wsl -- ssh torch "sha256sum $1 2>/dev/null | cut -d' ' -f1" || true; }

# upload_verified LOCAL REMOTE SHA256
upload_verified() {
  if [[ "$(remote_sha "$2")" == "$3" ]]; then
    echo "already on torch: $2"
    return 0
  fi
  wsl -- ssh torch "cat > $2.part" < "$1"
  [[ "$(remote_sha "$2.part")" == "$3" ]] || { echo "upload corrupted: $1" >&2; return 2; }
  wsl -- ssh torch "mv -f $2.part $2"
  echo "uploaded: $2"
}

wsl -- ssh -o NumberOfPasswordPrompts=0 torch "echo ssh_ok" || { echo "no SSH master: run scripts/start_torch_ssh_master.ps1 first" >&2; exit 2; }
wsl -- ssh torch "set -e; mkdir -p $WORK/logs $WORK/out $WORK/configs $WORK/data/suites; if [ ! -d $WORK/source/.git ]; then git clone -q $SOURCE_CELL/source $WORK/source; fi; echo source_ready"
REMOTE_HEAD=$(wsl -- ssh torch "git -C $WORK/source rev-parse HEAD")
if [[ "$REMOTE_HEAD" == "$COMMIT" ]]; then
  echo "source already at $SHORT"
else
  git cat-file -e "$REMOTE_HEAD^{commit}" 2>/dev/null || { echo "torch source is at $REMOTE_HEAD, which this repository does not have" >&2; exit 2; }
  mkdir -p "$(dirname "$BUNDLE_LOCAL")"
  git bundle create "$BUNDLE_LOCAL" "$REMOTE_HEAD..$BRANCH"
  SHA_LOCAL=$(sha256sum "$BUNDLE_LOCAL" | cut -d' ' -f1)
  echo "bundle $(stat -c %s "$BUNDLE_LOCAL") bytes sha256 $SHA_LOCAL"
  upload_verified "$BUNDLE_LOCAL" "$WORK/deploy_${SHORT}.bundle" "$SHA_LOCAL"
  wsl -- ssh torch bash -s <<EOF
set -e
git -C $WORK/source fetch -q $WORK/deploy_${SHORT}.bundle $BRANCH
git -C $WORK/source -c advice.detachedHead=false checkout -q --detach FETCH_HEAD
test "\$(git -C $WORK/source rev-parse HEAD)" = "$COMMIT"
echo source_at_$SHORT
EOF
fi

upload_verified "$TRAIN_SUITE" "$WORK/data/suites/ieee14_train.json" "$(sha256sum "$TRAIN_SUITE" | cut -d' ' -f1)"
upload_verified "$DEV_SUBSET" "$WORK/data/suites/ieee14_dev300.json" "$(sha256sum "$DEV_SUBSET" | cut -d' ' -f1)"
for config in $CONFIGS; do
  upload_verified "$REPO/$CELL_REL/$config" "$WORK/configs/$config" "$(sha256sum "$REPO/$CELL_REL/$config" | cut -d' ' -f1)"
done
wsl -- ssh torch "test -s /scratch/yx3882/compound_cloning_20261106/out/clone/training/lora/adapter_model.safetensors && echo start_adapter_present"
echo "deployed; nothing submitted.  Next: wsl -- ssh torch \"cd $WORK && sbatch --export=ALL,CONFIG=rl_smoke.json,ITERATIONS=1 --job-name=crl-smoke source/$CELL_REL/rl.sbatch\""
