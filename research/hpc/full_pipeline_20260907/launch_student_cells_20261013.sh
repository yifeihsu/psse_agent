#!/usr/bin/env bash
# Deploy and submit the two IEEE 118 student cells (2026-10-13) from Windows.
#
#   bash research/hpc/full_pipeline_20260907/launch_student_cells_20261013.sh [commit40]
#
# Cell A (overrides/ieee118_student_20261013.env): the frozen IEEE 14 student's
# first DAgger round on IEEE 118 under the spectra-first teacher, chain
# d0 -> r1c -> r1t -> r1e; the suite, balanced corpus and waveform corpora come
# from the 2026-10-11 transfer cell so the development roots are its 124.
# Cell B (overrides/ieee14_reverse_20261013.env): cell A's adapted adapter and
# the expert zero-shot on the 2026-10-07 classifier cell's 160 IEEE 14
# development roots, chain d0 -> r1c -> zs, queued behind cell A's training job.
# Needs a live WSL SSH master for the torch alias (scripts/start_torch_ssh_master.ps1).
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
CELL=$REPO/research/hpc/full_pipeline_20260907
# git.exe takes the POSIX form of the repository path literally once MSYS path conversion is off: work from inside it.
cd "$REPO"
BRANCH=${BRANCH:-codex/classifier-triage-20261004}
COMMIT=${1:-$(git rev-parse "$BRANCH")}
BASE=${BASE:-bf22a8d}                        # the last commit the 2026-10-11 cell's clone knows
OLD118=/scratch/yx3882/research_full_pipeline_20261011_ieee118
CLS14=/scratch/yx3882/research_full_pipeline_20261007_classifier
A=/scratch/yx3882/research_full_pipeline_20261013_ieee118_student
B=/scratch/yx3882/research_full_pipeline_20261013_ieee14_reverse
export MSYS_NO_PATHCONV=1

[[ -z "$(git status --porcelain --untracked-files=no -- research psse_env scripts)" ]] \
  || { echo "commit the pipeline sources first (tracked modifications under research/, psse_env/ or scripts/)" >&2; exit 2; }

echo "== cell A: $A from $COMMIT"
PIPE=$A SOURCE_CELL=$OLD118 BASE=$BASE BRANCH=$BRANCH OVERRIDES=ieee118_student_20261013.env SUBMIT=0 \
REMOTE_PREPARE="set -e
mkdir -p $A/out
ln -sfn $OLD118/corpora $A/corpora
[ -e $A/out/corpus ] || cp -r $OLD118/out/corpus $A/out/corpus
[ -e $A/out/suite ] || cp -r $OLD118/out/suite $A/out/suite
[ -s $A/out/suite.done ] || cp $OLD118/out/suite.done $A/out/suite.done
ls $A/corpora/ && echo prepared_A" \
  bash "$CELL/deploy_cell_from_windows.sh" "$COMMIT"
wsl -- ssh torch "cd $A && CHAIN='d0 r1c r1t r1e' bash submit_pipeline.sh"
R1T=$(wsl -- ssh torch "tail -1 $A/submitted_jobs.txt" | tr ' ' '\n' | sed -n 's/^r1t=//p' | tr -d '\r')
[[ "$R1T" =~ ^[0-9]+$ ]] || { echo "could not read cell A's r1t job id from submitted_jobs.txt" >&2; exit 2; }
echo "cell A submitted; r1t job $R1T"

echo "== cell B: $B (waits for job $R1T)"
PIPE=$B SOURCE_CELL=$OLD118 BASE=$BASE BRANCH=$BRANCH OVERRIDES=ieee14_reverse_20261013.env SUBMIT=0 \
REMOTE_PREPARE="set -e
mkdir -p $B/out
[ -e $B/out/suite ] || cp -r $CLS14/out/suite $B/out/suite
[ -s $B/out/suite.done ] || cp $CLS14/out/suite.done $B/out/suite.done
echo prepared_B" \
  bash "$CELL/deploy_cell_from_windows.sh" "$COMMIT"
wsl -- ssh torch "cd $B && DEPEND=$R1T ALLOW_PARALLEL_CHAIN=1 CHAIN='d0 r1c zs' bash submit_pipeline.sh"
echo "both cells submitted"
echo "status A: MSYS_NO_PATHCONV=1 wsl -- ssh torch bash $A/status_pipeline.sh"
echo "status B: MSYS_NO_PATHCONV=1 wsl -- ssh torch bash $B/status_pipeline.sh"
echo "after cell A's stage 0: wsl -- ssh torch \"cd $A && \$(sed -n 's/^PY=//p' pipeline.env) source/research/hpc/full_pipeline_20260907/check_suite_exclusion.py --d0-raw out/d0/aggregate.raw.jsonl --suite-dir out/suite --reference-d0-raw $OLD118/out/d0/aggregate.raw.jsonl\""
