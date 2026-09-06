#!/usr/bin/env bash
# Cluster side of the deployment.  Run after the git bundle has been uploaded:
#
#   ssh torch bash -s -- BUNDLE BRANCH EXPECTED_COMMIT [ROUND_DIR] [OVERRIDES] < deploy_remote.sh
#
# Clones the bundle into the round's source tree, stages the round scripts,
# syntax-checks them, records their digests, and runs the dry-run
# prerequisites (no tests, no GPU) so submission can refuse a broken setup.
#
# The committed scripts spell the original round directory literally (Slurm
# `--output` lines cannot take variables).  Passing ROUND_DIR stages the same
# scripts into a fresh directory with that path substituted, so a repeat of
# the round on a newer source commit never overwrites the first run's
# receipts; the substitution is recorded in deploy.json.  OVERRIDES names a
# file under the cell's overrides/ directory (for example
# scale_20260906.env); it is staged as round.overrides.env, which round.env
# sources last, and its name is recorded in deploy.json.
set -euo pipefail
BUNDLE=${1:?bundle path}
BRANCH=${2:?branch name}
EXPECTED=${3:?expected 40-hex commit}
SOURCE_ROUND=/scratch/yx3882/research_diag_round_20260903
ROUND=${4:-$SOURCE_ROUND}
OVERRIDES=${5:-}
CELL=research/hpc/diagnostic_round_20260903
[[ -z "$OVERRIDES" || "$OVERRIDES" =~ ^[A-Za-z0-9_.-]+[.]env$ ]] || { echo "overrides must be a plain file name under overrides/" >&2; exit 2; }
[[ "$EXPECTED" =~ ^[0-9a-f]{40}$ ]] || { echo "expected commit must be 40 hex" >&2; exit 2; }
[[ "$ROUND" =~ ^/[A-Za-z0-9_./-]+$ ]] || { echo "round directory must be an absolute plain path" >&2; exit 2; }
mkdir -p "$ROUND/logs" "$ROUND/out"
if [[ ! -d "$ROUND/source/.git" ]]; then
  git clone -q -b "$BRANCH" "$BUNDLE" "$ROUND/source"
fi
HEAD=$(git -C "$ROUND/source" rev-parse HEAD)
[[ "$HEAD" == "$EXPECTED" ]] || { echo "deployed $HEAD differs from expected $EXPECTED" >&2; exit 2; }
printf '%s\n' "$HEAD" > "$ROUND/source_commit.txt"
for name in round.env prerequisites.sh summarize.py filter_mixture.py submit_diag.sh \
  status_diag.sh amend_train_chain.sh diag_collect.sbatch diag_train.sbatch diag_eval.sbatch; do
  sed "s#${SOURCE_ROUND}#${ROUND}#g" "$ROUND/source/$CELL/$name" > "$ROUND/$name"
done
if [[ -n "$OVERRIDES" ]]; then
  [[ -f "$ROUND/source/$CELL/overrides/$OVERRIDES" ]] || { echo "unknown overrides file: $OVERRIDES" >&2; exit 2; }
  sed "s#${SOURCE_ROUND}#${ROUND}#g" "$ROUND/source/$CELL/overrides/$OVERRIDES" > "$ROUND/round.overrides.env"
else
  rm -f "$ROUND/round.overrides.env"
fi
printf '{"contract": "research_diagnostic_round_deploy_v1", "round_dir": "%s", "source_round_dir": "%s", "source_commit": "%s", "branch": "%s", "overrides": "%s", "deployed_at_utc": "%s"}\n' \
  "$ROUND" "$SOURCE_ROUND" "$HEAD" "$BRANCH" "$OVERRIDES" "$(date -u +%FT%TZ)" > "$ROUND/deploy.json"
chmod +x "$ROUND"/*.sh
for f in "$ROUND"/*.sh "$ROUND"/*.sbatch "$ROUND"/round.env "$ROUND"/round.overrides.env; do
  [[ -f "$f" ]] || continue
  bash -n "$f"
done
python3 -m py_compile "$ROUND/summarize.py" "$ROUND/filter_mixture.py"
(cd "$ROUND" && sha256sum round.env ./*.sh ./*.sbatch ./*.py > scripts.sha256)
bash "$ROUND/prerequisites.sh" --output "$ROUND/out/prerequisites.dryrun.json"
echo "deploy-complete $HEAD"
