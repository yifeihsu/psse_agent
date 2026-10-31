#!/bin/bash
#SBATCH --job-name=fp-r2wmodes
#SBATCH --account=torch_pr_627_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/yx3882/research_full_pipeline_20261022_conventional/logs/r2wmodes-%j.out
set -euo pipefail
P=/scratch/yx3882/research_full_pipeline_20261022_conventional
W=$P/r2warm_20261030
echo "[r2wmodes] job $SLURM_JOB_ID node $(hostname) $(date -u +%FT%TZ)"
cd "$W"
/scratch/yx3882/.conda/envs/gemma4_research_5104/bin/python -I "$W/mode_counts.py" "$W" "$P"
echo "fp-r2wmodes-complete $(date -u +%FT%TZ)"
