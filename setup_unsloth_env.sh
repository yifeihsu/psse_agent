#!/bin/bash
# One-time setup: creates the conda env with unsloth + all SFT dependencies.
# Run this ONCE on a login node (no GPU needed):
#   bash setup_unsloth_env.sh

set -euo pipefail

ENV_PREFIX=/scratch/yx3882/.conda/envs/unsloth_sft

echo "============================================"
echo " Setting up Unsloth SFT environment"
echo " Target: $ENV_PREFIX"
echo "============================================"

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

# Create env with Python 3.11
if [ -d "$ENV_PREFIX" ]; then
    echo "[info] env already exists at $ENV_PREFIX — skipping creation"
else
    conda create -y -p "$ENV_PREFIX" python=3.11
fi

conda activate "$ENV_PREFIX"

# Core torch stack (CUDA 12.1)
pip install --upgrade pip

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# HF ecosystem (install before unsloth so its deps are already satisfied)
pip install "transformers>=4.40.0"
pip install "datasets>=2.18.0"
pip install "accelerate>=0.29.0"
pip install "trl>=0.8.6"
pip install "peft>=0.10.0"
pip install "bitsandbytes>=0.43.0"
pip install sentencepiece tokenizers

# Unsloth — uses Triton kernels, flash-attn not required
pip install unsloth
pip install unsloth_zoo

# Logging
pip install wandb

echo ""
echo "============================================"
echo " Done! Verify with:"
echo "   conda activate $ENV_PREFIX"
echo "   python -c \"import unsloth; print('unsloth OK')\""
echo "============================================"
