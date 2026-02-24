#!/bin/bash
#SBATCH -A rmaulik
#SBATCH -p a100-40gb
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=240G
#SBATCH -t 00:45:00
#SBATCH -J mnist_ddpm_gilbreth
#SBATCH -o slurm-%j.out

set -euo pipefail

cd /scratch/gilbreth/rmaulik/codex_test
source .venv/bin/activate

python mnist_diffusion.py \
  --epochs 5 \
  --batch-size 128 \
  --num-workers 8 \
  --run-tag gilbreth_${SLURM_JOB_ID} \
  --outdir outputs
