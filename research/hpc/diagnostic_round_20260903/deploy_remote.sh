#!/usr/bin/env bash
# Cluster side of the deployment.  Run after the git bundle has been uploaded:
#
#   ssh torch bash -s -- BUNDLE BRANCH EXPECTED_COMMIT < deploy_remote.sh
#
# Clones the bundle into the round's source tree, stages the round scripts,
# syntax-checks them, records their digests, and runs the dry-run
# prerequisites (no tests, no GPU) so submission can refuse a broken setup.
set -euo pipefail
BUNDLE=${1:?bundle path}
BRANCH=${2:?branch name}
EXPECTED=${3:?expected 40-hex commit}
ROUND=/scratch/yx3882/research_diag_round_20260903
CELL=research/hpc/diagnostic_round_20260903
[[ "$EXPECTED" =~ ^[0-9a-f]{40}$ ]] || { echo "expected commit must be 40 hex" >&2; exit 2; }
mkdir -p "$ROUND/logs" "$ROUND/out"
if [[ ! -d "$ROUND/source/.git" ]]; then
  git clone -q -b "$BRANCH" "$BUNDLE" "$ROUND/source"
fi
HEAD=$(git -C "$ROUND/source" rev-parse HEAD)
[[ "$HEAD" == "$EXPECTED" ]] || { echo "deployed $HEAD differs from expected $EXPECTED" >&2; exit 2; }
printf '%s\n' "$HEAD" > "$ROUND/source_commit.txt"
for name in round.env prerequisites.sh summarize.py filter_mixture.py submit_diag.sh \
  status_diag.sh amend_train_chain.sh diag_collect.sbatch diag_train.sbatch diag_eval.sbatch; do
  cp "$ROUND/source/$CELL/$name" "$ROUND/$name"
done
chmod +x "$ROUND"/*.sh
for f in "$ROUND"/*.sh "$ROUND"/*.sbatch "$ROUND"/round.env; do
  bash -n "$f"
done
python3 -m py_compile "$ROUND/summarize.py" "$ROUND/filter_mixture.py"
(cd "$ROUND" && sha256sum round.env ./*.sh ./*.sbatch ./*.py > scripts.sha256)
bash "$ROUND/prerequisites.sh" --output "$ROUND/out/prerequisites.dryrun.json"
echo "deploy-complete $HEAD"
