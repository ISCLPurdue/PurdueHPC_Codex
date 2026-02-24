#!/bin/bash
#SBATCH -A rmaulik
#SBATCH -p ai
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=14
#SBATCH --gres=gpu:1
#SBATCH -t 00:30:00
#SBATCH -J mnist_ddpm
#SBATCH -o slurm-%j.out

set -euo pipefail

cd /scratch/gautschi/rmaulik/codex_test
source .venv/bin/activate

python mnist_diffusion.py \
  --epochs 5 \
  --batch-size 128 \
  --num-workers 8 \
  --outdir outputs
