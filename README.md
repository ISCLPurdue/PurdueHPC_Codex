# PurdueHPC_Codex
Codex interface with Purdue HPCs

## MNIST diffusion experiment on Gautschi

Scratch working directory:

`/scratch/gautschi/rmaulik/codex_test`

Repo-managed scripts (mirror these into scratch):

- `mnist_diffusion.py`
- `submit_mnist_diffusion.slurm`
- `requirements.txt`

### One-time environment setup (on Gautschi)

```bash
mkdir -p /scratch/gautschi/rmaulik/codex_test
cd /scratch/gautschi/rmaulik/codex_test
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Run interactively

```bash
cd /scratch/gautschi/rmaulik/codex_test
source .venv/bin/activate
python mnist_diffusion.py --epochs 5 --batch-size 128 --num-workers 8 --outdir outputs
```

### Submit with Slurm

```bash
cd /scratch/gautschi/rmaulik/codex_test
sbatch submit_mnist_diffusion.slurm
squeue -u rmaulik
```

Notes:
- The script targets partition `ai` and requests 1 GPU with 14 CPUs (required ratio on Gautschi AI partition).
- Generated images and checkpoints are written to `outputs/`.
