#!/bin/bash
#SBATCH --job-name=gpt_oss_dump
#SBATCH --output=/scratch/yx3882/psse_agent/logs/dump_%j.log
#SBATCH --error=/scratch/yx3882/psse_agent/logs/dump_%j.err
#SBATCH --chdir=/scratch/yx3882/psse_agent
#SBATCH --account=torch_pr_627_general
#SBATCH --comment=preemption=yes;requeue=true
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100"

set -euo pipefail

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

ENV_PREFIX=/scratch/yx3882/.conda/envs/unsloth_sft
PYTHON=$ENV_PREFIX/bin/python

REPO_ROOT=/scratch/yx3882/psse_agent
cd "$REPO_ROOT"

echo "===== Dump diagnostics ====="
$PYTHON tmp_dump_prompt.py
echo "============================"
