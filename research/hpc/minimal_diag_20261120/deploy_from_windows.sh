#!/usr/bin/env bash
# Deploy the stopping-state diagnostic on torch from the Windows checkout.  Submits nothing.
#
#   bash research/hpc/minimal_diag_20261120/deploy_from_windows.sh <commit40>
#
# The cell gets a work directory and a source of its own: the RL cell's source is cloned there (same objects, fast)
# and never moved, because the focused run's jobs read it at every phase.  Then the incremental git bundle goes in
# through ssh stdin with its checksum verified (wsl scp cannot take a Windows path, and a pipe into wsl ssh hangs:
# files go in with <), and so do the two files of scoring inputs stopping_states.py wrote here.  The supervised
# pass's adapter is read where it is.  Needs a live WSL SSH master for the torch alias
# (scripts/start_torch_ssh_master.ps1) and the sources committed on BRANCH.
set -euo pipefail
COMMIT=${1:?40-hex commit to deploy}
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "commit must be 40 hex" >&2; exit 2; }
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CELL_REL=research/hpc/minimal_diag_20261120
WORK=/scratch/yx3882/minimal_diag_20261120
BRANCH=${BRANCH:-$(git -C "$REPO" rev-parse --abbrev-ref HEAD)}
SOURCE_CELL=${SOURCE_CELL:-/scratch/yx3882/minimal_rl_20261116}
PASS=/scratch/yx3882/minimal_sft_20261115/out/pass_base
# The scoring inputs, relative to the repo: research/minimal_env/stopping_states.py --output-dir.
INPUTS_LOCAL=${INPUTS_LOCAL:-output/minimal_env_20261115/stopping_states_20261120}
PYTHON=${PYTHON:-python}
SHORT=${COMMIT:0:7}
# Relative to the repo (the script works from it): with MSYS_NO_PATHCONV set, git.exe takes a POSIX /c/... path
# literally and cannot create the bundle.
BUNDLE_LOCAL=output/deploy_mdg_${SHORT}.bundle
export MSYS_NO_PATHCONV=1

cd "$REPO"
[[ "$(git rev-parse "$BRANCH")" == "$COMMIT" ]] || { echo "$BRANCH is not at $COMMIT" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no -- research psse_env scripts tools)" ]] \
  || { echo "commit the sources first (tracked modifications under research/, psse_env/, scripts/ or tools/)" >&2; exit 2; }
for path in "$CELL_REL/cell.env" "$CELL_REL/stage_score.sbatch" "$CELL_REL/submit_cell.sh" "$CELL_REL/status_cell.sh" \
            research/minimal_env/score_actions.py research/minimal_env/stopping_states.py research/minimal_env/rl_train.py \
            psse_env/minimal/observation.py psse_env/minimal/policy.py psse_env/rl/update.py psse_env/rl/rows.py psse_env/rl/targets.py; do
  git cat-file -e "$COMMIT:$path" 2>/dev/null || { echo "$path is not in $COMMIT: commit the cell first" >&2; exit 2; }
done
# The inputs are the ones this checkout's scorer reads, every state with its observation and its actions.
for split in dev train; do
  [[ -s "$INPUTS_LOCAL/scoring_inputs_$split.jsonl" ]] || { echo "no $INPUTS_LOCAL/scoring_inputs_$split.jsonl: run stopping_states first" >&2; exit 2; }
done
"$PYTHON" -c 'import sys; from research.minimal_env import score_actions; items = score_actions.read_inputs(sys.argv[1:]); print("scoring inputs:", len(items), "states,", sum(len(item["actions"]) for item in items), "actions")' \
  "$INPUTS_LOCAL/scoring_inputs_dev.jsonl" "$INPUTS_LOCAL/scoring_inputs_train.jsonl"

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
# The job reads the source and the inputs: neither is moved under a queued or running job of this cell.
queued=$(wsl -- ssh torch "/opt/slurm/bin/squeue -h -u yx3882 -o %j" | grep -c '^mdg-' || true)
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
for split in dev train; do
  upload_verified "$INPUTS_LOCAL/scoring_inputs_$split.jsonl" "$WORK/data/scoring_inputs_$split.jsonl"
done

# What the cell reads in place.
wsl -- ssh torch "test -x /scratch/yx3882/.conda/envs/gemma4_research_5104/bin/python && echo python_present || echo PYTHON_MISSING"
wsl -- ssh torch "test -s $PASS/training.done && echo supervised_pass_finished || echo supervised_pass_not_finished"
wsl -- ssh torch "test -s $PASS/training/lora/adapter_model.safetensors && echo adapter_present || echo ADAPTER_MISSING"
cat <<EOF
deployed; nothing submitted.  Each step on the user's word:
  1. the scores:  wsl -- ssh torch bash -s -- score < $CELL_REL/submit_cell.sh
  2. status:      wsl -- ssh torch bash -s < $CELL_REL/status_cell.sh
EOF
