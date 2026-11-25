#!/usr/bin/env bash
# Deploy the cell of the RL plan of 2026-11-22 on torch from the Windows checkout.  Submits nothing.
#
#   bash research/hpc/minimal_rl_20261122/deploy_from_windows.sh <commit40>
#
# The cell gets a work directory and a source of its own: the earlier RL cell's source is cloned there (same
# objects, fast) and never moved, because that cell's records were made from it.  Then the incremental git bundle
# goes in through ssh stdin with its checksum verified (wsl scp cannot take a Windows path, and a pipe into wsl ssh
# hangs: files go in with <).  Nothing else is uploaded: the suites, the supervised pass's adapter and its two
# evaluations are read where they are.  Needs a live WSL SSH master for the torch alias
# (scripts/start_torch_ssh_master.ps1) and the sources committed on BRANCH.
set -euo pipefail
COMMIT=${1:?40-hex commit to deploy}
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "commit must be 40 hex" >&2; exit 2; }
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CELL_REL=research/hpc/minimal_rl_20261122
WORK=/scratch/yx3882/minimal_rl_20261122
BRANCH=${BRANCH:-$(git -C "$REPO" rev-parse --abbrev-ref HEAD)}
SOURCE_CELL=${SOURCE_CELL:-/scratch/yx3882/minimal_rl_20261116}
PASS=/scratch/yx3882/minimal_sft_20261115/out/pass_base
DEV_PASS=/scratch/yx3882/minimal_sft_20261115/out/eval_dev300_base
TRAIN_PASS=/scratch/yx3882/minimal_rl_20261116/out/train_pass
PYTHON=${PYTHON:-python}
SHORT=${COMMIT:0:7}
CONFIGS=("$CELL_REL/rl_steps.json" "$CELL_REL/rl_steps_smoke.json" "$CELL_REL/rl_forks.json" "$CELL_REL/rl_forks_smoke.json")
# Relative to the repo (the script works from it): with MSYS_NO_PATHCONV set, git.exe takes a POSIX /c/... path
# literally and cannot create the bundle.
BUNDLE_LOCAL=output/deploy_mrp_${SHORT}.bundle
export MSYS_NO_PATHCONV=1

cd "$REPO"
[[ "$(git rev-parse "$BRANCH")" == "$COMMIT" ]] || { echo "$BRANCH is not at $COMMIT" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no -- research psse_env scripts tools)" ]] \
  || { echo "commit the sources first (tracked modifications under research/, psse_env/, scripts/ or tools/)" >&2; exit 2; }
for path in "$CELL_REL/cell.env" "$CELL_REL/stage_rl.sbatch" "$CELL_REL/stage_fork_probe.sbatch" "$CELL_REL/submit_cell.sh" \
            "$CELL_REL/status_cell.sh" "${CONFIGS[@]}" \
            research/hpc/minimal_rl_20261116/train_truth_sequences.json research/hpc/minimal_sft_20261115/dev300_reference.json \
            research/minimal_env/rl_train.py research/minimal_env/fork_probe.py research/minimal_env/evaluate.py \
            research/minimal_env/truth_sequences.py psse_env/minimal/forks.py psse_env/minimal/policy.py psse_env/minimal/env.py \
            psse_env/minimal/observation.py psse_env/minimal/rollout.py psse_env/minimal/scoring.py \
            psse_env/dagger/research_policy_factory.py psse_env/rl/update.py psse_env/rl/rows.py psse_env/rl/integrity.py; do
  git cat-file -e "$COMMIT:$path" 2>/dev/null || { echo "$path is not in $COMMIT: commit the cell first" >&2; exit 2; }
done
# What the stages and the configs ask of the code must be in the deployed source.
git grep -q "evaluate_every" "$COMMIT" -- research/minimal_env/rl_train.py || { echo "$COMMIT's driver has no evaluate_every (stage A's evaluation interval)" >&2; exit 2; }
git grep -q "fork_settings" "$COMMIT" -- research/minimal_env/rl_train.py || { echo "$COMMIT's driver has no forks (stage B)" >&2; exit 2; }
git grep -q "fresh_optimizer_steps" "$COMMIT" -- psse_env/rl/update.py || { echo "$COMMIT's update has no fresh_optimizer_steps" >&2; exit 2; }
git grep -q "forced_prefix" "$COMMIT" -- psse_env/dagger/research_policy_factory.py || { echo "$COMMIT's policy cannot start an answer with a forced prefix" >&2; exit 2; }
git grep -q "def act_forced" "$COMMIT" -- psse_env/minimal/policy.py || { echo "$COMMIT's minimal policy has no forced call" >&2; exit 2; }
git grep -q -e '"--unrepaired-in"' "$COMMIT" -- research/minimal_env/fork_probe.py || { echo "$COMMIT's probe has no --unrepaired-in" >&2; exit 2; }
git grep -q "stop_on_refused_call" "$COMMIT" -- psse_env/minimal/forks.py || { echo "$COMMIT's forks have no stop_on_refused_call" >&2; exit 2; }
git grep -q "merge_shared_steps" "$COMMIT" -- psse_env/rl/rows.py || { echo "$COMMIT's row builder has no merge_shared_steps" >&2; exit 2; }
git grep -q "supervised_normalization" "$COMMIT" -- psse_env/rl/update.py || { echo "$COMMIT's update has no supervised_normalization" >&2; exit 2; }
# Every config loads under this checkout's driver.  Their paths are torch's; loading opens none of them.
"$PYTHON" -c 'import sys; from research.minimal_env.rl_train import MinimalRLConfig; [MinimalRLConfig.load(path) for path in sys.argv[1:]]; print("configs load")' \
  "${CONFIGS[@]}"
# The file of truth-built sequences is the one this checkout's loader reads.
"$PYTHON" -c 'import sys; from research.minimal_env import truth_sequences; print("truth-built sequences:", len(truth_sequences.load(sys.argv[1])), "cases")' \
  research/hpc/minimal_rl_20261116/train_truth_sequences.json

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
queued=$(wsl -- ssh torch "/opt/slurm/bin/squeue -h -u yx3882 -o %j" | grep -c '^mrp-' || true)
[[ "$queued" == 0 ]] || { echo "$queued job(s) of this cell are queued or running on torch: the source is not moved under them" >&2; exit 2; }
wsl -- ssh torch "set -e; mkdir -p $WORK/logs $WORK/out; if [ ! -d $WORK/source/.git ]; then git clone -q $SOURCE_CELL/source $WORK/source; fi; echo source_ready"
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
wsl -- ssh torch "test -s $WORK/source/research/hpc/minimal_rl_20261116/train_truth_sequences.json && echo truth_sequences_present || echo TRUTH_SEQUENCES_MISSING"
wsl -- ssh torch "test -s $PASS/training.done && test -s $PASS/training/lora/adapter_model.safetensors && echo supervised_adapter_present || echo SUPERVISED_ADAPTER_MISSING"
wsl -- ssh torch "test -s $DEV_PASS/evaluation.json && echo supervised_development_evaluation_present || echo SUPERVISED_DEVELOPMENT_EVALUATION_MISSING"
wsl -- ssh torch "test -s $TRAIN_PASS/evaluation.json && echo supervised_training_pass_present || echo SUPERVISED_TRAINING_PASS_MISSING"
# The suites, as the RL series planned them.  The series file is named relative to the repo: Windows Python cannot
# open a /c/... path.
planned() { "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["inputs"][sys.argv[2]][sys.argv[3]])' research/hpc/rl_improvement_20261110/series.json "$1" "$2"; }
for suite in training_suite development_suite; do
  path=$(planned "$suite" path)
  found=$(remote_sha "$path")
  if [[ "$found" == "$(planned "$suite" sha256)" ]]; then echo "as planned: $suite ($path)"; else echo "DIFFERS from the plan: $suite ($path) is ${found:-missing}" >&2; fi
done
cat <<EOF
deployed; nothing submitted.  The order of work (README.md of the cell), each step on the user's word:
  1. stage A's smoke:  wsl -- ssh torch bash -s -- steps-smoke < $CELL_REL/submit_cell.sh
  2. the probe's smoke: wsl -- ssh torch bash -s -- probe-smoke < $CELL_REL/submit_cell.sh
  3. status:           wsl -- ssh torch bash -s < $CELL_REL/status_cell.sh
  4. stage A:          wsl -- ssh torch bash -s -- steps ITERATIONS < $CELL_REL/submit_cell.sh
  5. the probe:        wsl -- ssh torch bash -s -- probe dev < $CELL_REL/submit_cell.sh
                       wsl -- ssh torch bash -s -- probe train < $CELL_REL/submit_cell.sh
  6. stage B, once the probe is read:
                       wsl -- ssh torch bash -s -- forks-smoke < $CELL_REL/submit_cell.sh
                       wsl -- ssh torch bash -s -- forks ITERATIONS < $CELL_REL/submit_cell.sh
EOF
