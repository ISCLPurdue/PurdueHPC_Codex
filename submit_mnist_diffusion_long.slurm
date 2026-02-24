#!/bin/bash
#SBATCH -A rmaulik
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:1
#SBATCH -t 02:00:00
#SBATCH -J mnist_ddpm_long
#SBATCH -o slurm-%j.out

set -euo pipefail

cd /scratch/gautschi/rmaulik/codex_test
source .venv/bin/activate

python mnist_diffusion.py \
  --epochs 40 \
  --batch-size 128 \
  --num-workers 8 \
  --num-samples 64 \
  --run-tag long_${SLURM_JOB_ID} \
  --outdir outputs
