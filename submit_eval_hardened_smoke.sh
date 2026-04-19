#!/bin/bash
#SBATCH --job-name=psse_eval_smoke_hardened
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --comment=preemption=no;requeue=false
#SBATCH --output=outputs/gpt_oss_sft_power_agent/eval_smoke_hardened_%j.log

set -euo pipefail

cd /scratch/yx3882/psse_agent

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python eval_sft_agent_hardened.py \
  --adapter harshith0214/psse-agent-gpt-oss-20b \
  --test-file out_traces_balanced/sft_traces.test.jsonl \
  --max-samples 20 \
  --max-turns 4 \
  --max-new-tokens 2048 \
  --max-seq-length 8192 \
  --output outputs/gpt_oss_sft_power_agent/eval_smoke_hardened.jsonl \
  --verbose
