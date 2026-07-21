#!/bin/bash
# One-time setup for the current reviewed Gemma 4 SFT release environment. The
# filename and environment path are retained for compatibility with the H200
# pilot, but the dependency pins have been reviewed and changed since that run.
# Run this ONCE on a login node (no GPU needed):
#   bash setup_unsloth_env.sh

set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENV_PREFIX=/scratch/yx3882/.conda/envs/unsloth_sft

echo "============================================"
echo " Setting up Unsloth SFT environment"
echo " Target: $ENV_PREFIX"
echo "============================================"

module purge
module load anaconda3/2025.06
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh

# Build the current reviewed Python/CUDA release stack.
if [ -d "$ENV_PREFIX" ]; then
    echo "[info] env already exists at $ENV_PREFIX — skipping creation"
else
    conda create -y -p "$ENV_PREFIX" python=3.12
fi

conda activate "$ENV_PREFIX"

# Core torch stack (CUDA 12.8)
pip install --upgrade pip
pip install "torch==2.10.0" "torchvision==0.25.0" \
    --index-url https://download.pytorch.org/whl/cu128

# Install the reviewed release pins without the unrelated unsloth or wandb
# packages; torchvision and Pillow remain required by Gemma4Processor.
pip install --upgrade -r "$REPO_ROOT/psse_env/requirements-sft.txt"
python -m pip check
python - <<'PY'
import sys

import torch

if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"expected Python 3.12, found {sys.version.split()[0]}")
if torch.__version__ != "2.10.0+cu128":
    raise SystemExit(
        f"expected torch 2.10.0+cu128, found {torch.__version__}; "
        "recreate the environment instead of reusing an incompatible prefix"
    )
print("verified Python 3.12 and torch 2.10.0+cu128")
PY

echo ""
echo "============================================"
echo " Done! Verify with:"
echo "   conda activate $ENV_PREFIX"
echo "   python -c \"import torch, accelerate; print(torch.__version__, accelerate.__version__)\""
echo "============================================"
