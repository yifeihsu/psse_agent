#!/usr/bin/env bash
# Deploy the cloning cell on torch from the Windows checkout.  Submits nothing (submit_cell.sh does).
#
#   TRAIN_VIEW=.../cloning.train_view.jsonl TRAIN_VIEW_SHA256=<64 hex> \
#   VALIDATION_VIEW=.../cloning.validation.jsonl VALIDATION_VIEW_SHA256=<64 hex> \
#   bash research/hpc/compound_cloning_20261106/deploy_from_windows.sh <commit40>
#
# The row paths and hashes are inputs: the selection is re-made, and no hash lives in this cell.  The
# upload is refused while PROVISIONAL.txt is beside the rows.  DEPLOY_ROWS=0 deploys the source and
# the suites without rows, for the evaluation of the start adapter.
#
# UPLOAD_SUITES lists NAME=FILE words for $DATA/suites/NAME.json; NAME must be a line of the suite
# list.  Default: the IEEE 14 development suite of this worktree (the conventional test suite is
# already on torch).  To add the IEEE 118 draws, uncomment their lines in eval_suites.env and run
# this again with DEPLOY_ROWS=0 and UPLOAD_SUITES naming their files (README).  A file already on
# torch with the same hash is not sent again.  The working copy of the list is what gets uploaded,
# to $DATA/eval_suites.env; EVAL_SUITES_FILE names another list.
#
# Needs a live WSL SSH master for the torch alias (scripts/start_torch_ssh_master.ps1, interactive
# NYU SSO) and the sources committed on BRANCH.  Steps: clone a previous cell's source as this cell's
# (same objects, fast), upload the incremental git bundle through ssh stdin and verify its checksum
# (wsl scp cannot take a Windows path, and a pipe into wsl ssh hangs: files go in with <), upload the
# rows and suites the same way, then remote_prepare.sh: start adapter, checks, manifests.
set -euo pipefail
COMMIT=${1:?40-hex commit to deploy}
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "commit must be 40 hex" >&2; exit 2; }
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CELL_REL=research/hpc/compound_cloning_20261106
WORK=/scratch/yx3882/compound_cloning_20261106
BRANCH=${BRANCH:-claude/rl-lifecycle-evidence-20261106}
SOURCE_CELL=${SOURCE_CELL:-/scratch/yx3882/research_full_pipeline_20261022_conventional}
DEPLOY_ROWS=${DEPLOY_ROWS:-1}
EVAL_SUITES_FILE=${EVAL_SUITES_FILE:-$REPO/$CELL_REL/eval_suites.env}
UPLOAD_SUITES=${UPLOAD_SUITES-ieee14_dev=$REPO/output/compound_errors_20261027/suite.json}
SHORT=${COMMIT:0:7}
# Relative to the repo: with MSYS_NO_PATHCONV set, git.exe would take a POSIX /c/... path literally.
BUNDLE_LOCAL=output/deploy_cloning_${SHORT}.bundle
export MSYS_NO_PATHCONV=1

# Everything that can refuse locally, before any contact with the cluster.
TRAIN_SHA=
VALIDATION_SHA=
if [[ "$DEPLOY_ROWS" == 1 ]]; then
  : "${TRAIN_VIEW:?TRAIN_VIEW must name the training view (or DEPLOY_ROWS=0)}" "${VALIDATION_VIEW:?VALIDATION_VIEW must name the validation view}"
  for view in "$TRAIN_VIEW" "$VALIDATION_VIEW"; do
    marker=$(dirname "$view")/PROVISIONAL.txt
    if [[ -e "$marker" ]]; then
      {
        echo "REFUSED: $marker is still beside $view, so these rows are not final and nothing was uploaded."
        echo "Its text:"
        sed 's/^/  /' "$marker"
        echo "Run this again once the selection has been re-made and the marker removed, or use DEPLOY_ROWS=0 for the source and suites alone."
      } >&2
      exit 3
    fi
    [[ -s "$view" ]] || { echo "row file missing or empty: $view" >&2; exit 2; }
  done
  : "${TRAIN_VIEW_SHA256:?TRAIN_VIEW_SHA256 must give the hash of the training view}" "${VALIDATION_VIEW_SHA256:?VALIDATION_VIEW_SHA256 must give the hash of the validation view}"
  TRAIN_SHA=$(sha256sum "$TRAIN_VIEW" | cut -d' ' -f1)
  VALIDATION_SHA=$(sha256sum "$VALIDATION_VIEW" | cut -d' ' -f1)
  [[ "$TRAIN_SHA" == "$TRAIN_VIEW_SHA256" ]] || { echo "training view hash is $TRAIN_SHA, not the given $TRAIN_VIEW_SHA256" >&2; exit 2; }
  [[ "$VALIDATION_SHA" == "$VALIDATION_VIEW_SHA256" ]] || { echo "validation view hash is $VALIDATION_SHA, not the given $VALIDATION_VIEW_SHA256" >&2; exit 2; }
  rows=$(grep -c . "$TRAIN_VIEW")
  echo "training view: $rows rows, $(( (rows + 3) / 4 )) optimizer steps"
fi
[[ -s "$EVAL_SUITES_FILE" ]] || { echo "suite list missing: $EVAL_SUITES_FILE" >&2; exit 2; }
# -U: Git for Windows' grep strips the CR of CRLF lines before matching, so without it this check never fires.
if grep -qU $'\r' "$EVAL_SUITES_FILE"; then echo "suite list has CR line endings: $EVAL_SUITES_FILE" >&2; exit 2; fi
for item in $UPLOAD_SUITES; do
  name=${item%%=*}
  file=${item#*=}
  grep -q "^eval_suite $name " "$EVAL_SUITES_FILE" || { echo "suite $name is not a line of $EVAL_SUITES_FILE: add or uncomment it first" >&2; exit 2; }
  [[ -s "$file" ]] || { echo "suite file missing or empty: $file" >&2; exit 2; }
done

cd "$REPO"
[[ "$(git rev-parse "$BRANCH")" == "$COMMIT" ]] || { echo "$BRANCH is not at $COMMIT" >&2; exit 2; }
# The suite list is exempt: a change to it reaches the cluster as the uploaded file, not through git.
[[ -z "$(git status --porcelain --untracked-files=no -- research psse_env scripts tools ":(exclude)$CELL_REL/eval_suites.env")" ]] \
  || { echo "commit the sources first (tracked modifications under research/, psse_env/, scripts/ or tools/)" >&2; exit 2; }
for path in "$CELL_REL/cell.env" "$CELL_REL/stage_clone_train.sbatch" psse_env/providers/detailed_evidence.py; do
  git cat-file -e "$COMMIT:$path" 2>/dev/null || { echo "$path is not in $COMMIT: commit the cell and the options first" >&2; exit 2; }
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
wsl -- ssh torch "set -e; mkdir -p $WORK/logs $WORK/out $WORK/data/rows $WORK/data/suites; if [ ! -d $WORK/source/.git ]; then git clone -q $SOURCE_CELL/source $WORK/source; fi; echo source_ready"
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

if [[ "$DEPLOY_ROWS" == 1 ]]; then
  # Changing the rows under a running or partly done pass would break its resume.
  current=$(remote_sha "$WORK/data/rows/cloning.train_view.jsonl")
  if [[ -n "$current" && "$current" != "$TRAIN_SHA" ]]; then
    saved=$(wsl -- ssh torch "find $WORK/out/clone/training -maxdepth 1 -name 'checkpoint-*' 2>/dev/null | wc -l")
    if [[ "${saved:-0}" != 0 ]]; then
      echo "the pass under $WORK/out/clone has checkpoints from other rows: move that directory aside first" >&2
      exit 2
    fi
  fi
  upload_verified "$TRAIN_VIEW" "$WORK/data/rows/cloning.train_view.jsonl" "$TRAIN_SHA"
  upload_verified "$VALIDATION_VIEW" "$WORK/data/rows/cloning.validation.jsonl" "$VALIDATION_SHA"
  report=$(dirname "$TRAIN_VIEW")/selection_report.json
  if [[ -s "$report" ]]; then
    upload_verified "$report" "$WORK/data/rows/selection_report.json" "$(sha256sum "$report" | cut -d' ' -f1)"
  fi
fi
for item in $UPLOAD_SUITES; do
  upload_verified "${item#*=}" "$WORK/data/suites/${item%%=*}.json" "$(sha256sum "${item#*=}" | cut -d' ' -f1)"
done
upload_verified "$EVAL_SUITES_FILE" "$WORK/data/eval_suites.env" "$(sha256sum "$EVAL_SUITES_FILE" | cut -d' ' -f1)"

wsl -- ssh torch bash -s -- "$COMMIT" "$BRANCH" "$TRAIN_SHA" "$VALIDATION_SHA" < "$REPO/$CELL_REL/remote_prepare.sh"
echo "deployed; nothing submitted.  Next: MSYS_NO_PATHCONV=1 wsl -- ssh torch bash -s -- smoke < $CELL_REL/submit_cell.sh"
