#!/usr/bin/env bash
# Deploy the RL improvement series on torch from the Windows checkout.  Submits nothing (sbatch does).
#
#   bash research/hpc/rl_improvement_20261110/deploy_from_windows.sh <commit40>
#
# Steps, as in the cell of 2026-11-07: clone that cell's source as this cell's (same objects, fast), upload
# the incremental git bundle through ssh stdin and verify its checksum (wsl scp cannot take a Windows path,
# and a pipe into wsl ssh hangs: files go in with <), then the run configs, the reserved roots and the
# probe's training pool the same way.  Needs a live WSL SSH master for the torch alias
# (scripts/start_torch_ssh_master.ps1) and the sources committed on BRANCH.
#
# Nothing else is copied: the suites, the start adapter and the first arm's rollouts (the calibration batch)
# are read where the earlier cells left them.  Their digests on torch are printed beside the ones the series
# was planned on; a difference is reported here and in every job's provenance record, and stops nothing.
set -euo pipefail
COMMIT=${1:?40-hex commit to deploy}
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "commit must be 40 hex" >&2; exit 2; }
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CELL_REL=research/hpc/rl_improvement_20261110
WORK=/scratch/yx3882/rl_improvement_20261110
BRANCH=${BRANCH:-$(git -C "$REPO" rev-parse --abbrev-ref HEAD)}
SOURCE_CELL=${SOURCE_CELL:-/scratch/yx3882/compound_rl_20261107}
PYTHON=${PYTHON:-python}
SHORT=${COMMIT:0:7}
BUNDLE_LOCAL=output/deploy_rli_${SHORT}.bundle
export MSYS_NO_PATHCONV=1

cd "$REPO"
[[ "$(git rev-parse "$BRANCH")" == "$COMMIT" ]] || { echo "$BRANCH is not at $COMMIT" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no -- research psse_env scripts tools)" ]] \
  || { echo "commit the sources first (tracked modifications under research/, psse_env/, scripts/ or tools/)" >&2; exit 2; }
for path in "$CELL_REL/rl.sbatch" "$CELL_REL/fork_probe.sbatch" "$CELL_REL/series.json" psse_env/rl/forks.py psse_env/dagger/observation_modes.py; do
  git cat-file -e "$COMMIT:$path" 2>/dev/null || { echo "$path is not in $COMMIT: commit the cell and the driver first" >&2; exit 2; }
done
# The config files are what series.json gives, and none of the files that go up has CR line endings
# (-U: Git for Windows' grep strips the CR of CRLF lines before matching, so without it the check never fires).
PSSE_LOCAL_DIAGNOSTIC_BUILD=1 PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m research.hpc.rl_improvement_20261110.make_configs --check \
  || { echo "the configs are not what series.json gives: run make_configs.py and commit" >&2; exit 2; }
UPLOADS=("$CELL_REL"/configs/*.json "$CELL_REL/reserved_roots.json" "$CELL_REL/probe_training_roots.json")
for file in "${UPLOADS[@]}"; do
  [[ -s "$file" ]] || { echo "missing: $file" >&2; exit 2; }
  if grep -qU $'\r' "$file"; then echo "CR line endings in $file" >&2; exit 2; fi
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
wsl -- ssh torch "set -e; mkdir -p $WORK/logs $WORK/out $WORK/configs; if [ ! -d $WORK/source/.git ]; then git clone -q $SOURCE_CELL/source $WORK/source; fi; echo source_ready"
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

for file in "${UPLOADS[@]}"; do
  upload_verified "$REPO/$file" "$WORK/configs/$(basename "$file")" "$(sha256sum "$REPO/$file" | cut -d' ' -f1)"
done

# What the series reads in place, against what it was planned on.
planned() { "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["inputs"][sys.argv[2]][sys.argv[3]])' "$CELL_REL/series.json" "$1" "$2"; }
for name in training_suite development_suite expert_reference; do
  path=$(planned "$name" path)
  found=$(remote_sha "$path")
  if [[ "$found" == "$(planned "$name" sha256)" ]]; then echo "as planned: $name ($path)"; else echo "DIFFERS from the plan: $name ($path) is ${found:-missing}" >&2; fi
done
wsl -- ssh torch "test -s $(planned initial_adapter path)/adapter_model.safetensors && echo start_adapter_present"
wsl -- ssh torch "ls $(planned calibration_batch path)/groups | wc -l | sed 's/^/calibration batch groups: /'"
cat <<EOF
deployed; nothing submitted.  The order of work (README.md of the cell):
  1. smokes:      wsl -- ssh torch "cd $WORK && sbatch --export=ALL,CONFIG=smoke_complete.json,ITERATIONS=1,TARGET_ITERATIONS=1 --job-name=rli-smoke source/$CELL_REL/rl.sbatch"
  2. capacity:    wsl -- ssh torch "cd $WORK && sbatch --export=ALL,CONFIG=arm_b_complete.json,TASK=capacity --job-name=rli-capacity source/$CELL_REL/rl.sbatch"
  3. calibration: prepare the three update-only runs (calibration.py), then submit_arm.sh calibration_1step.json 2 1
  4. fork probe:  wsl -- ssh torch "cd $WORK && sbatch source/$CELL_REL/fork_probe.sbatch"
  5. arms A, B:   submit_arm.sh arm_a_bounded.json 6 2 rli-a ; submit_arm.sh arm_b_complete.json 8 2 rli-b
EOF
