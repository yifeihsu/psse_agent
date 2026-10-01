#!/usr/bin/env bash
# Deploy a cell of this pipeline on torch from the Windows checkout and submit its chain.
#
#   bash research/hpc/full_pipeline_20260907/deploy_cell_from_windows.sh <commit40>
#
# Defaults are the hypothesis-ranking cell (2026-10-01, ledger_ranked teacher);
# PIPE, OVERRIDES, SOURCE_CELL, BASE and BRANCH can be overridden in the
# environment.  Needs a live WSL SSH master for the torch alias
# (scripts/start_torch_ssh_master.ps1, interactive NYU SSO).  Steps: a local
# clone of a previous cell's source as the new cell's source (same objects,
# fast), the incremental git bundle from the last commit that clone knows
# uploaded through ssh stdin (wsl scp cannot take a Windows path) and
# checksum-verified, deploy_remote.sh with the overrides file (fetch, path
# rewrite, dry-run prerequisites), then submit_pipeline.sh.
set -euo pipefail
COMMIT=${1:?40-hex commit to deploy}
[[ "$COMMIT" =~ ^[0-9a-f]{40}$ ]] || { echo "commit must be 40 hex" >&2; exit 2; }
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
BRANCH=${BRANCH:-codex/cleanup-20260928}
BASE=${BASE:-b8d5117}                         # last commit the torch clones know (2026-09-27 deploy)
PIPE=${PIPE:-/scratch/yx3882/research_full_pipeline_20261001_ranked}
SOURCE_CELL=${SOURCE_CELL:-/scratch/yx3882/research_full_pipeline_20260924_wls_gated}
OVERRIDES=${OVERRIDES:-hypothesis_ranking_20261001.env}
SHORT=${COMMIT:0:7}
# Relative to the repo: with MSYS_NO_PATHCONV set, git.exe would take a POSIX /c/... path literally.
BUNDLE_LOCAL=${BUNDLE_LOCAL:-output/deploy_${SHORT}.bundle}
export MSYS_NO_PATHCONV=1

cd "$REPO"
[[ "$(git rev-parse "$BRANCH")" == "$COMMIT" ]] || { echo "$BRANCH is not at $COMMIT" >&2; exit 2; }
mkdir -p "$(dirname "$BUNDLE_LOCAL")"
git bundle create "$BUNDLE_LOCAL" "$BASE..$BRANCH"
SHA_LOCAL=$(sha256sum "$BUNDLE_LOCAL" | cut -d' ' -f1)
echo "bundle $(stat -c %s "$BUNDLE_LOCAL") bytes sha256 $SHA_LOCAL"

# Plain ssh through the ControlMaster socket; BatchMode=yes bypasses it on this setup.
wsl -- ssh -o NumberOfPasswordPrompts=0 torch "echo ssh_ok" || { echo "no SSH master: run scripts/start_torch_ssh_master.ps1 first" >&2; exit 2; }
wsl -- ssh torch "set -e; mkdir -p $PIPE/logs $PIPE/out; if [ ! -d $PIPE/source/.git ]; then git clone -q $SOURCE_CELL/source $PIPE/source; fi; echo source_ready"
wsl -- ssh torch "cat > $PIPE/deploy_${SHORT}.bundle" < "$BUNDLE_LOCAL"
SHA_REMOTE=$(wsl -- ssh torch "sha256sum $PIPE/deploy_${SHORT}.bundle | cut -d' ' -f1")
[[ "$SHA_LOCAL" == "$SHA_REMOTE" ]] || { echo "bundle upload corrupted" >&2; exit 2; }
wsl -- ssh torch bash -s -- "$PIPE/deploy_${SHORT}.bundle" "$BRANCH" "$COMMIT" "$PIPE" "$OVERRIDES" \
  < "$REPO/research/hpc/full_pipeline_20260907/deploy_remote.sh"
wsl -- ssh torch "cat $PIPE/deploy.json"
echo "deployed; submitting the chain"
wsl -- ssh torch "cd $PIPE && bash submit_pipeline.sh"
echo "status: MSYS_NO_PATHCONV=1 wsl -- ssh torch bash $PIPE/status_pipeline.sh"
