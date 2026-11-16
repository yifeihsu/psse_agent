#!/usr/bin/env bash
# Deploy the cell of the minimal environment's RL run on torch from the Windows checkout.  Submits nothing.
#
#   bash research/hpc/minimal_rl_20261116/deploy_from_windows.sh <commit40>
#
# The cell gets a work directory and a source of its own: the supervised cell's source is cloned there (same
# objects, fast) and never moved, because that cell's jobs run from it.  Then the incremental git bundle goes in
# through ssh stdin with its checksum verified (wsl scp cannot take a Windows path, and a pipe into wsl ssh hangs:
# files go in with <).  Nothing else is uploaded: the suites, the reference and the supervised pass's adapter are
# read where they are.  Needs a live WSL SSH master for the torch alias (scripts/start_torch_ssh_master.ps1) and the
# sources committed on BRANCH.
set -euo pipefail
COMMIT=${1:?40-hex commit to deploy}
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "commit must be 40 hex" >&2; exit 2; }
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CELL_REL=research/hpc/minimal_rl_20261116
WORK=/scratch/yx3882/minimal_rl_20261116
BRANCH=${BRANCH:-$(git -C "$REPO" rev-parse --abbrev-ref HEAD)}
SOURCE_CELL=${SOURCE_CELL:-/scratch/yx3882/minimal_sft_20261115}
PASS=/scratch/yx3882/minimal_sft_20261115/out/pass_base
PYTHON=${PYTHON:-python}
SHORT=${COMMIT:0:7}
# Relative to the repo (the script works from it): with MSYS_NO_PATHCONV set, git.exe takes a POSIX /c/... path
# literally and cannot create the bundle.
BUNDLE_LOCAL=output/deploy_mrl_${SHORT}.bundle
export MSYS_NO_PATHCONV=1

cd "$REPO"
[[ "$(git rev-parse "$BRANCH")" == "$COMMIT" ]] || { echo "$BRANCH is not at $COMMIT" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no -- research psse_env scripts tools)" ]] \
  || { echo "commit the sources first (tracked modifications under research/, psse_env/, scripts/ or tools/)" >&2; exit 2; }
for path in "$CELL_REL/cell.env" "$CELL_REL/stage_rl.sbatch" "$CELL_REL/submit_cell.sh" "$CELL_REL/status_cell.sh" \
            "$CELL_REL/rl_ieee14.json" "$CELL_REL/rl_smoke.json" \
            research/hpc/minimal_sft_20261115/dev300_reference.json \
            research/minimal_env/rl_train.py research/minimal_env/evaluate.py \
            psse_env/minimal/env.py psse_env/minimal/observation.py psse_env/minimal/policy.py psse_env/minimal/rollout.py \
            psse_env/minimal/scoring.py psse_env/rl/update.py psse_env/rl/rows.py psse_env/rl/integrity.py; do
  git cat-file -e "$COMMIT:$path" 2>/dev/null || { echo "$path is not in $COMMIT: commit the cell first" >&2; exit 2; }
done
# What the stage and the configs ask of the driver must be in the deployed source.
git grep -q -e '"--check"' "$COMMIT" -- research/minimal_env/rl_train.py || { echo "$COMMIT's driver has no --check" >&2; exit 2; }
git grep -q "baseline_from" "$COMMIT" -- research/minimal_env/rl_train.py || { echo "$COMMIT's driver cannot take the supervised evaluation as its baseline" >&2; exit 2; }
git grep -q "development_limit" "$COMMIT" -- research/minimal_env/rl_train.py || { echo "$COMMIT's driver has no development_limit (the smoke's eight cases)" >&2; exit 2; }
# Both configs load under this checkout's driver.  Their paths are torch's; loading opens none of them.
"$PYTHON" -c 'import sys; from research.minimal_env.rl_train import MinimalRLConfig; [MinimalRLConfig.load(path) for path in sys.argv[1:]]; print("configs load")' \
  "$CELL_REL/rl_ieee14.json" "$CELL_REL/rl_smoke.json"

remote_sha() { wsl -- ssh torch "sha256sum $1 2>/dev/null | cut -d' ' -f1" || true; }

# upload_verified LOCAL REMOTE
upload_verified() {
  local sha
  sha=$(sha256sum "$1" | cut -d' ' -f1)
  if [[ "$(remote_sha "$2")" == "$sha" ]]; then
    echo "already on torch: $2"
    return 0
  fi
  wsl -- ssh torch "cat > $2.part" < "$1"
  [[ "$(remote_sha "$2.part")" == "$sha" ]] || { echo "upload corrupted: $1" >&2; return 2; }
  wsl -- ssh torch "mv -f $2.part $2"
  echo "uploaded: $2 ($(stat -c %s "$1") bytes)"
}

wsl -- ssh -o NumberOfPasswordPrompts=0 torch "echo ssh_ok" || { echo "no SSH master: run scripts/start_torch_ssh_master.ps1 first" >&2; exit 2; }
# A job of this cell reads the source at every phase: the source is not moved under one that is queued or running.
queued=$(wsl -- ssh torch "/opt/slurm/bin/squeue -h -u yx3882 -o %j" | grep -c '^mrl-' || true)
[[ "$queued" == 0 ]] || { echo "$queued job(s) of this cell are queued or running on torch: the source is not moved under them" >&2; exit 2; }
wsl -- ssh torch "set -e; mkdir -p $WORK/logs $WORK/out $WORK/data; if [ ! -d $WORK/source/.git ]; then git clone -q $SOURCE_CELL/source $WORK/source; fi; echo source_ready"
REMOTE_HEAD=$(wsl -- ssh torch "git -C $WORK/source rev-parse HEAD")
if [[ "$REMOTE_HEAD" == "$COMMIT" ]]; then
  echo "source already at $SHORT"
else
  git cat-file -e "$REMOTE_HEAD^{commit}" 2>/dev/null || { echo "torch source is at $REMOTE_HEAD, which this repository does not have" >&2; exit 2; }
  mkdir -p "$(dirname "$BUNDLE_LOCAL")"
  git bundle create "$BUNDLE_LOCAL" "$REMOTE_HEAD..$BRANCH"
  echo "bundle $(stat -c %s "$BUNDLE_LOCAL") bytes"
  upload_verified "$BUNDLE_LOCAL" "$WORK/deploy_${SHORT}.bundle"
  wsl -- ssh torch bash -s <<EOF
set -e
git -C $WORK/source fetch -q $WORK/deploy_${SHORT}.bundle $BRANCH
git -C $WORK/source -c advice.detachedHead=false checkout -q --detach FETCH_HEAD
test "\$(git -C $WORK/source rev-parse HEAD)" = "$COMMIT"
echo source_at_$SHORT
EOF
fi

# What the cell reads in place.
wsl -- ssh torch "test -x /scratch/yx3882/.conda/envs/gemma4_research_5104/bin/python && echo python_present || echo PYTHON_MISSING"
wsl -- ssh torch "test -s $WORK/source/research/hpc/minimal_sft_20261115/dev300_reference.json && echo reference_present || echo REFERENCE_MISSING"
wsl -- ssh torch "test -s $PASS/training.done && echo supervised_pass_finished || echo supervised_pass_not_finished"
wsl -- ssh torch "test -s $PASS/../eval_dev300_base/evaluation.json && echo supervised_evaluation_finished || echo supervised_evaluation_not_finished"
# The suites, as the RL series planned them: the training suite is the one the demonstrations were built from.  The
# series file is named relative to the repo: Windows Python cannot open a /c/... path.
planned() { "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["inputs"][sys.argv[2]][sys.argv[3]])' research/hpc/rl_improvement_20261110/series.json "$1" "$2"; }
for suite in training_suite development_suite; do
  path=$(planned "$suite" path)
  found=$(remote_sha "$path")
  if [[ "$found" == "$(planned "$suite" sha256)" ]]; then echo "as planned: $suite ($path)"; else echo "DIFFERS from the plan: $suite ($path) is ${found:-missing}" >&2; fi
done
cat <<EOF
deployed; nothing submitted.  The order of work (README.md of the cell), each step on the user's word:
  1. smoke:   wsl -- ssh torch bash -s -- smoke < $CELL_REL/submit_cell.sh
  2. status:  wsl -- ssh torch bash -s < $CELL_REL/status_cell.sh
  3. the run, once the supervised pass has finished and its evaluation is read:
              wsl -- ssh torch bash -s -- run ITERATIONS < $CELL_REL/submit_cell.sh
EOF
