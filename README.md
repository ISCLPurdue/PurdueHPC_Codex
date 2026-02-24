# PurdueHPC_Codex
Codex interface with Purdue RCAC HPC clusters.

## MNIST diffusion experiment on Gilbreth

Scratch working directory:

`/scratch/gilbreth/rmaulik/codex_test`

Repo-managed scripts to mirror into scratch:

- `mnist_diffusion.py`
- `submit_mnist_diffusion_gilbreth.slurm`
- `submit_mnist_diffusion_gilbreth_long.slurm`
- `requirements.txt`
- `serve_dashboard.sh`

### One-time environment setup (on Gilbreth)

```bash
mkdir -p /scratch/gilbreth/rmaulik/codex_test
cd /scratch/gilbreth/rmaulik/codex_test
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Run interactively

```bash
cd /scratch/gilbreth/rmaulik/codex_test
source .venv/bin/activate
python mnist_diffusion.py --epochs 5 --batch-size 128 --num-workers 8 --outdir outputs
```

### Submit with Slurm (Gilbreth)

```bash
cd /scratch/gilbreth/rmaulik/codex_test
sbatch submit_mnist_diffusion_gilbreth.slurm
squeue -u rmaulik
```

Notes:
- The script targets partition `a100-40gb` and requests 1 GPU with 8 CPUs.
- Gilbreth requires explicit memory requests; scripts set `--mem=240G`.
- Generated images and checkpoints are written to `outputs/<run_tag>/`.

Validated run on Gilbreth (`a100-40gb`):
- Job ID: `10330760`
- State: `COMPLETED` (exit `0:0`)
- Elapsed (Slurm): `00:03:41`
- Run tag: `gilbreth_10330760`
- Final epoch mean loss: `0.048376138451129896`
- Best epoch mean loss: `0.048376138451129896`
- Device: `cuda`
- Artifacts: `outputs/gilbreth_10330760/` and live `outputs/dashboard.html`

## Longer run + dashboard

Submit a longer run:

```bash
cd /scratch/gilbreth/rmaulik/codex_test
sbatch submit_mnist_diffusion_gilbreth_long.slurm
```

This writes a run folder under `outputs/<run_tag>/` with:

- `dashboard.html`
- `mnist_samples.png`
- `loss_curve_step.png`
- `loss_curve_epoch.png`
- `architecture_schematic.png`
- `loss_history.csv`
- `metrics.json`

Serve the latest run on a cluster login node:

```bash
cd /scratch/gilbreth/rmaulik/codex_test
./serve_dashboard.sh /scratch/gilbreth/rmaulik/codex_test 8080
```

From your local machine, open an SSH tunnel and browse:

```bash
ssh -N -L 8080:localhost:8080 rmaulik@gilbreth.rcac.purdue.edu
```

Then open:

`http://localhost:8080/dashboard.html`

Notes:
- `outputs/dashboard.html` serves a root dashboard that follows `outputs/LATEST_RUN.txt` and `outputs/current`.
- The run dashboard refreshes periodically and supports click-to-zoom controls (`+`, `-`, `reset`) on images.
- Refresh polling stops automatically when run status becomes `completed`.
- Latest validated run dashboard: `outputs/gilbreth_10330760/dashboard.html`

## Troubleshooting on Gilbreth

- If `sbatch` reports `Job size specification needs to be provided`, add/set `--mem` (for example `--mem=240G` per GPU).
- If jobs stay pending with reason `AssocGrpGRES`, your account has reached current GPU concurrency limits; wait for running jobs to finish or submit to a less contended GPU partition if your allocation allows it.
