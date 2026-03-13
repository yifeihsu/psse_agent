#!/bin/bash
#SBATCH --job-name=gpt_oss_sft
#SBATCH --output=/scratch/yx3882/psse_agent/logs/sft_%j.log
#SBATCH --error=/scratch/yx3882/psse_agent/logs/sft_%j.err
#SBATCH --chdir=/scratch/yx3882/psse_agent
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100"

set -euo pipefail

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

ENV_PREFIX=/scratch/yx3882/.conda/envs/unsloth_sft
PYTHON=$ENV_PREFIX/bin/python

mkdir -p /scratch/yx3882/psse_agent/logs
mkdir -p /scratch/yx3882/.cache/huggingface
mkdir -p /scratch/yx3882/.cache/torch

export HF_HOME=/scratch/yx3882/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/yx3882/.cache/huggingface
export HF_DATASETS_CACHE=/scratch/yx3882/.cache/huggingface/datasets
export TORCH_HOME=/scratch/yx3882/.cache/torch

echo "===== Job diagnostics ====="
echo "hostname: $(hostname)"
echo "pwd: $(pwd)"
echo "which python: $(which python)"
echo "env python: $PYTHON"
$PYTHON -V
$PYTHON -c "import sys; print('sys.executable =', sys.executable)"
$PYTHON -m pip show datasets || true
$PYTHON -c "import datasets; print('datasets version =', datasets.__version__)"
nvidia-smi
echo "==========================="

$PYTHON gpt_oss_power_sft_revised.py \
    --train-file out_traces_balanced/split_train.jsonl \
    --model-name unsloth/gpt-oss-20b \
    --output-dir outputs/gpt_oss_sft_power_agent \
    --max-seq-length 16384 \
    --dataset-num-proc 16 \
    --load-in-4bit \
    --lora-r 64 \
    --lora-alpha 64 \
    --per-device-train-batch-size 4 \
    --gradient-accumulation-steps 4 \
    --learning-rate 2e-4 \
    --logging-steps 5 \
    --num-train-epochs 1