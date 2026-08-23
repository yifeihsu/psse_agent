#!/bin/bash
#SBATCH --job-name=stress-physical-dev
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail

: "${SFT_PYTHON:?export SFT_PYTHON as the validated Python interpreter}"

repo_path="${1:?first argument must be the repository path}"
holdout_path="${2:?second argument must be the development holdout path}"

cd "$repo_path"
export PYTHONPATH="$repo_path${PYTHONPATH:+:$PYTHONPATH}"
"$SFT_PYTHON" scripts/validate_dagger1_recovery_stress_physical.py \
    --development-holdout "$holdout_path"
