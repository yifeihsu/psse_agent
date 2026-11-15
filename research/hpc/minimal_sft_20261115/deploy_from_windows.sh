#!/usr/bin/env bash
# Deploy the cell of the minimal environment's supervised pass on torch from the Windows checkout.  Submits nothing.
#
#   bash research/hpc/minimal_sft_20261115/deploy_from_windows.sh <commit40>
#
# The cell gets a work directory and a source of its own: an existing cell's source is cloned there (same objects,
# fast) and never moved.  Then, as in the other cells: the incremental git bundle through ssh stdin with its
# checksum verified (wsl scp cannot take a Windows path, and a pipe into wsl ssh hangs: files go in with <), the two
# row views with a manifest (rows_manifest.py) and the selection's report.  Needs a live WSL SSH master for the
# torch alias (scripts/start_torch_ssh_master.ps1) and the sources committed on BRANCH.
#
# SELECTION_DIR names the folder select_rows wrote.  output/ is not in git and these rows were built in the worktree
# this cell was written in, so the default is this checkout's output/minimal_env_20261115/selection_ieee14; from
# another checkout name the folder.
set -euo pipefail
COMMIT=${1:?40-hex commit to deploy}
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "commit must be 40 hex" >&2; exit 2; }
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CELL_REL=research/hpc/minimal_sft_20261115
WORK=/scratch/yx3882/minimal_sft_20261115
BRANCH=${BRANCH:-$(git -C "$REPO" rev-parse --abbrev-ref HEAD)}
SOURCE_CELL=${SOURCE_CELL:-/scratch/yx3882/clone_open_rules_20261112}
R1_ADAPTER=/scratch/yx3882/compound_cloning_20261106/data/start_adapter
SELECTION_DIR=${SELECTION_DIR:-$REPO/output/minimal_env_20261115/selection_ieee14}
PYTHON=${PYTHON:-python}
SHORT=${COMMIT:0:7}
# Relative to the repo (the script works from it): with MSYS_NO_PATHCONV set, git.exe takes a POSIX /c/... path
# literally and cannot create the bundle.
BUNDLE_LOCAL=output/deploy_msf_${SHORT}.bundle
export MSYS_NO_PATHCONV=1
# Windows Python reads C:\ paths; MSYS_NO_PATHCONV keeps Git Bash from converting /c/ ones on its own.
winpath() { cygpath -w "$1" 2>/dev/null || printf '%s' "$1"; }

cd "$REPO"
[[ "$(git rev-parse "$BRANCH")" == "$COMMIT" ]] || { echo "$BRANCH is not at $COMMIT" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no -- research psse_env scripts tools)" ]] \
  || { echo "commit the sources first (tracked modifications under research/, psse_env/, scripts/ or tools/)" >&2; exit 2; }
for path in "$CELL_REL/cell.env" "$CELL_REL/stage_row_gate.sbatch" "$CELL_REL/stage_train.sbatch" "$CELL_REL/stage_eval.sbatch" \
            "$CELL_REL/submit_cell.sh" "$CELL_REL/status_cell.sh" "$CELL_REL/dev300_reference.json" \
            research/minimal_env/train_sft.py research/minimal_env/row_gate.py research/minimal_env/evaluate.py \
            psse_env/minimal/env.py psse_env/minimal/observation.py psse_env/minimal/policy.py psse_env/minimal/scoring.py; do
  git cat-file -e "$COMMIT:$path" 2>/dev/null || { echo "$path is not in $COMMIT: commit the cell first" >&2; exit 2; }
done
# The trainer entry with the cached row gate and the evaluation that stores each case must be in the deployed source.
git grep -q MINIMAL_GATE_CACHE "$COMMIT" -- research/minimal_env/train_sft.py || { echo "$COMMIT's trainer entry has no row gate cache" >&2; exit 2; }
git grep -q "def evaluate_cases" "$COMMIT" -- research/minimal_env/evaluate.py || { echo "$COMMIT has no minimal evaluation" >&2; exit 2; }

[[ -d "$SELECTION_DIR" ]] || { echo "no selection at $SELECTION_DIR: name it with SELECTION_DIR (the rows were built in the worktree dreamy-bassi-27849a)" >&2; exit 2; }
TRAIN_LOCAL=$SELECTION_DIR/train.jsonl
VALIDATION_LOCAL=$SELECTION_DIR/validation.jsonl
REPORT_LOCAL=$SELECTION_DIR/selection_report.json
MANIFEST_LOCAL=$SELECTION_DIR/manifest.json
# The manifest refuses views that are not the ones the selection report describes, rows of another interface and
# any carriage return.
"$PYTHON" -m research.hpc.minimal_sft_20261115.rows_manifest --selection-dir "$(winpath "$SELECTION_DIR")" --commit "$COMMIT" \
  --output "$(winpath "$MANIFEST_LOCAL")"

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
# Replacing the rows under a pass that has checkpoints would resume it on other rows.  Two plain calls and a
# comparison here: wsl -- runs its arguments through a shell of its own, so a command substitution meant for torch
# inside an ssh command string is expanded on the WSL side, where the rows are not.
if [[ -n "$(wsl -- ssh torch "ls -d $WORK/out/pass_*/training/checkpoint-* 2>/dev/null | head -n 1" || true)" ]]; then
  [[ "$(remote_sha "$WORK/data/rows/train.jsonl")" == "$(sha256sum "$TRAIN_LOCAL" | cut -d' ' -f1)" ]] \
    || { echo "a pass on torch has checkpoints on other rows: remove its folder under $WORK/out first" >&2; exit 2; }
fi
wsl -- ssh torch "set -e; mkdir -p $WORK/logs $WORK/out $WORK/data/rows $WORK/data/suites; if [ ! -d $WORK/source/.git ]; then git clone -q $SOURCE_CELL/source $WORK/source; fi; echo source_ready"
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

upload_verified "$TRAIN_LOCAL" "$WORK/data/rows/train.jsonl"
upload_verified "$VALIDATION_LOCAL" "$WORK/data/rows/validation.jsonl"
upload_verified "$MANIFEST_LOCAL" "$WORK/data/rows/manifest.json"
upload_verified "$REPORT_LOCAL" "$WORK/data/rows/selection_report.json"

# What the cell reads in place.
wsl -- ssh torch "test -s $R1_ADAPTER/adapter_model.safetensors && echo r1_adapter_present || echo R1_ADAPTER_MISSING"
wsl -- ssh torch "test -x /scratch/yx3882/.conda/envs/gemma4_research_5104/bin/python && echo python_present || echo PYTHON_MISSING"
wsl -- ssh torch "test -s $WORK/source/$CELL_REL/dev300_reference.json && echo reference_present || echo REFERENCE_MISSING"
# The development cases, as the RL series planned them.  The series file is named relative to the repo: Windows
# Python cannot open a /c/... path.
planned() { "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["inputs"][sys.argv[2]][sys.argv[3]])' research/hpc/rl_improvement_20261110/series.json "$1" "$2"; }
path=$(planned development_suite path)
found=$(remote_sha "$path")
if [[ "$found" == "$(planned development_suite sha256)" ]]; then echo "as planned: development_suite ($path)"; else echo "DIFFERS from the plan: development_suite ($path) is ${found:-missing}" >&2; fi
cat <<EOF
deployed; nothing submitted.  The order of work (README.md of the cell), each step on the user's word:
  1. smoke:    wsl -- ssh torch bash -s -- smoke < $CELL_REL/submit_cell.sh
  2. status:   wsl -- ssh torch bash -s < $CELL_REL/status_cell.sh
  3. the row gate, the pass, its evaluation: wsl -- ssh torch bash -s -- all < $CELL_REL/submit_cell.sh
     (or one at a time: pass, then eval; a second word names the start, r1 or base)
EOF
