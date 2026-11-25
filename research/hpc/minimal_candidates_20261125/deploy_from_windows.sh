#!/usr/bin/env bash
# Deploy the candidate read-out on torch from the Windows checkout.  Submits nothing.
#
#   bash research/hpc/minimal_candidates_20261125/deploy_from_windows.sh <commit40>
#
# The cell gets a work directory and a source of its own: the RL cell's source is cloned there (same objects, fast)
# and never moved.  The commit is BRANCH's tip or a commit below it.  Where the clone holds it already it is checked
# out; otherwise the incremental git bundle goes in through ssh stdin with its checksum verified (wsl scp cannot
# take a Windows path, and a pipe into wsl ssh hangs: files go in with <).  So do the inputs candidate_sets.py build
# wrote here: the arm base always, any other arm of cell.env whose file is there.  The adapters are read where they
# are.  Needs a live WSL SSH master for the torch alias (scripts/start_torch_ssh_master.ps1) and the commit on BRANCH.
set -euo pipefail
COMMIT=${1:?40-hex commit to deploy}
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "commit must be 40 hex" >&2; exit 2; }
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CELL_REL=research/hpc/minimal_candidates_20261125
WORK=/scratch/yx3882/minimal_candidates_20261125
BRANCH=${BRANCH:-$(git -C "$REPO" rev-parse --abbrev-ref HEAD)}
SOURCE_CELL=${SOURCE_CELL:-/scratch/yx3882/minimal_rl_20261122}
PASS=/scratch/yx3882/minimal_sft_20261115/out/pass_base
STEPS=/scratch/yx3882/minimal_rl_20261122/out/steps
# The inputs, relative to the repo: research/minimal_env/candidate_sets.py build --output.
INPUTS_LOCAL=${INPUTS_LOCAL:-output/minimal_candidates_20261125}
PYTHON=${PYTHON:-python}
SHORT=${COMMIT:0:7}
# Relative to the repo (the script works from it): with MSYS_NO_PATHCONV set, git.exe takes a POSIX /c/... path
# literally and cannot create the bundle.
BUNDLE_LOCAL=output/deploy_mcs_${SHORT}.bundle
export MSYS_NO_PATHCONV=1

cd "$REPO"
# The branch's tip or a commit below it: other sessions commit to this branch too, and the tip may hold work this
# cell was not tested with.
git merge-base --is-ancestor "$COMMIT" "$BRANCH" || { echo "$COMMIT is not on $BRANCH" >&2; exit 2; }
# What goes to torch is the commit, never the working tree: files that differ here are named, not sent.
differing=$(git diff --name-only "$COMMIT" -- research psse_env scripts tools)
if [[ -n "$differing" ]]; then
  echo "note: these tracked files differ from $SHORT in this checkout; the deploy sends the commit, not them:"
  echo "$differing" | sed 's/^/    /'
fi
for path in "$CELL_REL/cell.env" "$CELL_REL/stage_candidates.sbatch" "$CELL_REL/submit_cell.sh" "$CELL_REL/status_cell.sh" \
            research/minimal_env/candidate_sets.py research/minimal_env/score_actions.py research/minimal_env/repair_breakdown.py \
            research/minimal_env/rl_train.py psse_env/minimal/actions.py psse_env/minimal/observation.py psse_env/minimal/policy.py \
            psse_env/rl/update.py psse_env/rl/rows.py psse_env/rl/targets.py; do
  git cat-file -e "$COMMIT:$path" 2>/dev/null || { echo "$path is not in $COMMIT: commit the cell first" >&2; exit 2; }
done
# The inputs are the ones this checkout's scorer reads: every state with its observation and what the truth leaves.
[[ -s "$INPUTS_LOCAL/candidate_inputs_dev_base.jsonl" ]] \
  || { echo "no $INPUTS_LOCAL/candidate_inputs_dev_base.jsonl: run candidate_sets.py build first" >&2; exit 2; }
ARMS_LOCAL=()
for arm in base steps17; do
  [[ -s "$INPUTS_LOCAL/candidate_inputs_dev_$arm.jsonl" ]] || continue
  ARMS_LOCAL+=("$arm")
  "$PYTHON" -c 'import sys; from research.minimal_env import candidate_sets; items = candidate_sets.read_inputs(sys.argv[2:]); print("inputs of arm", sys.argv[1] + ":", len(items), "states,", sum(len(candidate_sets.candidates(item["observation"])) for item in items), "candidates")' \
    "$arm" "$INPUTS_LOCAL/candidate_inputs_dev_$arm.jsonl"
done

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
queued=$(wsl -- ssh torch "/opt/slurm/bin/squeue -h -u yx3882 -o %j" | grep -c '^mcs-' || true)
[[ "$queued" == 0 ]] || { echo "$queued job(s) of this cell are queued or running on torch: the source is not moved under them" >&2; exit 2; }
wsl -- ssh torch "set -e; mkdir -p $WORK/logs $WORK/out $WORK/data; if [ ! -d $WORK/source/.git ]; then git clone -q $SOURCE_CELL/source $WORK/source; fi; echo source_ready"
# The cell's source on torch, detached at the commit.
checkout_commit() {
  wsl -- ssh torch bash -s <<EOF
set -e
git -C $WORK/source -c advice.detachedHead=false checkout -q --detach $COMMIT
test "\$(git -C $WORK/source rev-parse HEAD)" = "$COMMIT"
echo source_at_$SHORT
EOF
}
REMOTE_HEAD=$(wsl -- ssh torch "git -C $WORK/source rev-parse HEAD")
if [[ "$REMOTE_HEAD" == "$COMMIT" ]]; then
  echo "source already at $SHORT"
elif [[ "$(wsl -- ssh torch "git -C $WORK/source cat-file -t $COMMIT 2>/dev/null || true")" == commit ]]; then
  # The clone holds the commit already (the source it came from has moved past it): nothing to send.
  echo "torch source is at ${REMOTE_HEAD:0:7} and holds $SHORT"
  checkout_commit
else
  git cat-file -e "$REMOTE_HEAD^{commit}" 2>/dev/null || { echo "torch source is at $REMOTE_HEAD, which this repository does not have" >&2; exit 2; }
  mkdir -p "$(dirname "$BUNDLE_LOCAL")"
  git bundle create "$BUNDLE_LOCAL" "$REMOTE_HEAD..$BRANCH"
  echo "bundle $(stat -c %s "$BUNDLE_LOCAL") bytes"
  upload_verified "$BUNDLE_LOCAL" "$WORK/deploy_${SHORT}.bundle"
  wsl -- ssh torch "git -C $WORK/source fetch -q $WORK/deploy_${SHORT}.bundle $BRANCH && echo bundle_fetched"
  checkout_commit
fi
for arm in "${ARMS_LOCAL[@]}"; do
  upload_verified "$INPUTS_LOCAL/candidate_inputs_dev_$arm.jsonl" "$WORK/data/candidate_inputs_dev_$arm.jsonl"
done

# What the cell reads in place.
wsl -- ssh torch "test -x /scratch/yx3882/.conda/envs/gemma4_research_5104/bin/python && echo python_present || echo PYTHON_MISSING"
wsl -- ssh torch "test -s $PASS/training/lora/adapter_config.json && echo adapter_base_present || echo ADAPTER_BASE_MISSING"
wsl -- ssh torch "test -s $STEPS/iteration_002/adapter/adapter_config.json && echo adapter_steps17_present || echo ADAPTER_STEPS17_MISSING"
cat <<EOF
deployed (arms with inputs: ${ARMS_LOCAL[*]}); nothing submitted.  Each step on the user's word:
  1. an arm:   wsl -- ssh torch bash -s -- score base < $CELL_REL/submit_cell.sh
  2. status:   wsl -- ssh torch bash -s < $CELL_REL/status_cell.sh
EOF
