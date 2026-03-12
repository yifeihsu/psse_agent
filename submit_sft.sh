#!/bin/bash
#SBATCH --job-name=gpt_oss_sft
#SBATCH --output=sft_%j.log
#SBATCH --error=sft_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1             # Requesting exactly 1 H100 GPU
#SBATCH --cpus-per-task=16            # H100s process data so fast, feed it more CPUs
#SBATCH --mem=128G                    # Increase system RAM to keep up with the GPU
#SBATCH --time=24:00:00               
#SBATCH --partition=gpu               # Ensure this matches your HPC's H100 partition name

# Load Anaconda and wake up the environment
module purge
module load anaconda3/2025.06
source activate unsloth_sft

# Ensure huggingface cache goes to scratch if needed (uncomment and edit if your home drive fills up)
# export HF_HOME=/scratch/$USER/huggingface

# Increase batch size and sequence length to utilize the H100's massive 80GB VRAM!
srun python gpt_oss_power_sft_revised.py \
    --train-file "out_traces_balanced/split_train.jsonl" \
    --valid-file "out_traces_balanced/split_valid.jsonl" \
    --model-name "unsloth/gpt-oss-20b" \
    --output-dir "outputs/gpt_oss_sft_power_agent" \
    --max-seq-length 16384 \
    --dataset-num-proc 16 \
    --load-in-4bit \
    --lora-r 64 \
    --lora-alpha 64 \
    --per-device-train-batch-size 4 \
    --gradient-accumulation-steps 4 \
    --learning-rate 2e-4 \
    --logging-steps 5 \
    --eval-steps 50 \
    --num-train-epochs 3
