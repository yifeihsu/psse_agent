#!/usr/bin/env bash
# Deploy the open-rules clone's cell on torch from the Windows checkout.  Submits nothing.
#
#   bash research/hpc/clone_open_rules_20261112/deploy_from_windows.sh <commit40>
#
# The cell gets a work directory and a source of its own: the RL series' jobs read their source at every phase, so
# that source is cloned here (same objects, fast) and never moved.  Then, as in the other cells: the incremental git
# bundle through ssh stdin with its checksum verified (wsl scp cannot take a Windows path, and a pipe into wsl ssh
# hangs: files go in with <), the two row views with a manifest (rows_manifest.py), the selection's report and row
# checks, and the three baseline configs.  Needs a live WSL SSH master for the torch alias
# (scripts/start_torch_ssh_master.ps1) and the sources committed on BRANCH.
#
# SELECTION_DIR names the folder open_rules_selection.py wrote.  output/ is not in git, so the default is the main
# checkout's output/compound_rl_20261106/open_rules_selection_ieee14 whether this runs from it or from a worktree.
set -euo pipefail
COMMIT=${1:?40-hex commit to deploy}
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "commit must be 40 hex" >&2; exit 2; }
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
MAIN=$(cd "$REPO" && cd "$(git rev-parse --git-common-dir)/.." && pwd)
CELL_REL=research/hpc/clone_open_rules_20261112
WORK=/scratch/yx3882/clone_open_rules_20261112
BRANCH=${BRANCH:-$(git -C "$REPO" rev-parse --abbrev-ref HEAD)}
SOURCE_CELL=${SOURCE_CELL:-/scratch/yx3882/rl_improvement_20261110}
BOUNDED_WORK=/scratch/yx3882/compound_cloning_20261106
SELECTION_DIR=${SELECTION_DIR:-$MAIN/output/compound_rl_20261106/open_rules_selection_ieee14}
PYTHON=${PYTHON:-python}
SHORT=${COMMIT:0:7}
# Relative to the repo (the script works from it): with MSYS_NO_PATHCONV set, git.exe takes a POSIX /c/... path
# literally and cannot create the bundle (the first deploy of 68b1950 failed so).
BUNDLE_LOCAL=output/deploy_orc_${SHORT}.bundle
export MSYS_NO_PATHCONV=1
# Windows Python reads C:\ paths; MSYS_NO_PATHCONV keeps Git Bash from converting /c/ ones on its own.
winpath() { cygpath -w "$1" 2>/dev/null || printf '%s' "$1"; }

cd "$REPO"
[[ "$(git rev-parse "$BRANCH")" == "$COMMIT" ]] || { echo "$BRANCH is not at $COMMIT" >&2; exit 2; }
[[ -z "$(git status --porcelain --untracked-files=no -- research psse_env scripts tools)" ]] \
  || { echo "commit the sources first (tracked modifications under research/, psse_env/, scripts/ or tools/)" >&2; exit 2; }
for path in "$CELL_REL/cell.env" "$CELL_REL/stage_row_gate.sbatch" "$CELL_REL/stage_clone_train.sbatch" "$CELL_REL/stage_baseline.sbatch" \
            "$CELL_REL/configs/baseline_s0_open.json" "$CELL_REL/configs/baseline_clone_open.json" "$CELL_REL/configs/baseline_smoke_open.json" \
            research/hpc/compound_cloning_20261106/train_runtime_tools.py research/hpc/compound_cloning_20261106/smoke_inputs.py; do
  git cat-file -e "$COMMIT:$path" 2>/dev/null || { echo "$path is not in $COMMIT: commit the cell first" >&2; exit 2; }
done
# The environment option and the driver's reading of it must be in the deployed source.
git grep -q OPEN_RULES_CONTRACT "$COMMIT" -- psse_env/actions.py || { echo "$COMMIT has no open_rules option (psse_env/actions.py)" >&2; exit 2; }
git grep -q rejected_candidate_decisions "$COMMIT" -- psse_env/rl/train.py || { echo "$COMMIT's RL driver does not record the open rules" >&2; exit 2; }
PSSE_LOCAL_DIAGNOSTIC_BUILD=1 PYTHONDONTWRITEBYTECODE=1 "$PYTHON" -m research.hpc.clone_open_rules_20261112.baseline_config --check \
  || { echo "the baseline configs are not what the series gives: run baseline_config.py and commit" >&2; exit 2; }

TRAIN_LOCAL=$SELECTION_DIR/selection.train_view.jsonl
VALIDATION_LOCAL=$SELECTION_DIR/selection.validation.jsonl
REPORT_LOCAL=$SELECTION_DIR/selection_report.json
CHECKS_LOCAL=$SELECTION_DIR/checks/row_checks.json
MANIFEST_LOCAL=$SELECTION_DIR/manifest.json
for name in baseline_s0_open baseline_clone_open baseline_smoke_open; do
  # -U: Git for Windows' grep strips the CR of CRLF lines before matching, so without it the check never fires.
  if grep -qU $'\r' "$REPO/$CELL_REL/configs/$name.json"; then echo "CR line endings in $name.json" >&2; exit 2; fi
done
# The manifest refuses views that are not the ones the selection report and the row checks describe (and any CR).
"$PYTHON" -m research.hpc.clone_open_rules_20261112.rows_manifest --selection-dir "$(winpath "$SELECTION_DIR")" --commit "$COMMIT" \
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
# Replacing the rows under a pass that has checkpoints would resume it on other rows.
wsl -- ssh torch "if ls -d $WORK/out/clone/training/checkpoint-* >/dev/null 2>&1 && [ \"\$(sha256sum $WORK/data/rows/selection.train_view.jsonl 2>/dev/null | cut -d' ' -f1)\" != \"$(sha256sum "$TRAIN_LOCAL" | cut -d' ' -f1)\" ]; then echo rows_differ_under_checkpoints; fi" \
  | grep -q rows_differ_under_checkpoints && { echo "the pass on torch has checkpoints on other rows: remove $WORK/out/clone first" >&2; exit 2; }
wsl -- ssh torch "set -e; mkdir -p $WORK/logs $WORK/out $WORK/configs $WORK/data/rows $WORK/data/suites; if [ ! -d $WORK/source/.git ]; then git clone -q $SOURCE_CELL/source $WORK/source; fi; echo source_ready"
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

upload_verified "$TRAIN_LOCAL" "$WORK/data/rows/selection.train_view.jsonl"
upload_verified "$VALIDATION_LOCAL" "$WORK/data/rows/selection.validation.jsonl"
upload_verified "$MANIFEST_LOCAL" "$WORK/data/rows/manifest.json"
upload_verified "$REPORT_LOCAL" "$WORK/data/rows/selection_report.json"
upload_verified "$CHECKS_LOCAL" "$WORK/data/rows/row_checks.json"
for name in baseline_s0_open baseline_clone_open baseline_smoke_open; do
  upload_verified "$REPO/$CELL_REL/configs/$name.json" "$WORK/configs/$name.json"
done

# What the cell reads in place.
wsl -- ssh torch "test -s $BOUNDED_WORK/data/start_adapter/adapter_model.safetensors && echo start_adapter_present || echo START_ADAPTER_MISSING"
wsl -- ssh torch "test -s $BOUNDED_WORK/out/clone/training/lora/adapter_model.safetensors && echo s0_adapter_present || echo S0_ADAPTER_MISSING"
wsl -- ssh torch "test -x /scratch/yx3882/.conda/envs/gemma4_research_5104/bin/python && echo python_present || echo PYTHON_MISSING"
wsl -- ssh torch "test -s /scratch/yx3882/rl_improvement_20261110/out/arm_a_bounded/baseline/evaluation.json && echo arm_a_baseline_present || echo ARM_A_BASELINE_MISSING"
planned() { "$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["inputs"][sys.argv[2]][sys.argv[3]])' research/hpc/rl_improvement_20261110/series.json "$1" "$2"; }
path=$(planned development_suite path)
found=$(remote_sha "$path")
if [[ "$found" == "$(planned development_suite sha256)" ]]; then echo "as planned: development_suite ($path)"; else echo "DIFFERS from the plan: development_suite ($path) is ${found:-missing}" >&2; fi
cat <<EOF
deployed; nothing submitted.  The order of work (README.md of the cell), each step on the user's word:
  1. smoke:    wsl -- ssh torch bash -s -- smoke < $CELL_REL/submit_cell.sh
  2. status:   wsl -- ssh torch bash -s < $CELL_REL/status_cell.sh
  3. S0's baseline in the open environment:        wsl -- ssh torch bash -s -- s0 < $CELL_REL/submit_cell.sh
  4. the row gate, the pass, the clone's baseline: wsl -- ssh torch bash -s -- pass < $CELL_REL/submit_cell.sh
     then: wsl -- ssh torch bash -s -- clone < $CELL_REL/submit_cell.sh     (or 3 and 4 at once: all)
EOF
