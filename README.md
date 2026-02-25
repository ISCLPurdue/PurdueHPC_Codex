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
- The script now runs posterior sampling after training for partial observations (default: 70% observed pixels).

## Longer run + dashboard

Submit a longer run:

```bash
cd /scratch/gautschi/rmaulik/codex_test
sbatch submit_mnist_diffusion_long.slurm
```

This writes a run folder under `outputs/<run_tag>/` with:

- `dashboard.html`
- `mnist_samples.png`
- `loss_curve_step.png`
- `loss_curve_epoch.png`
- `architecture_schematic.png`
- `posterior_conditioning_overview.png`
- `posterior_samples.png`
- `loss_history.csv`
- `metrics.json`

Serve the latest run on Gautschi:

```bash
cd /scratch/gautschi/rmaulik/codex_test
./serve_dashboard.sh
```

From your local machine, open an SSH tunnel and browse:

```bash
ssh -N -L 8080:localhost:8080 rmaulik@gautschi.rcac.purdue.edu
```

Then open:

`http://localhost:8080/dashboard.html`

Notes:
- `http://localhost:8080/dashboard.html` now serves a live root dashboard that always follows `outputs/LATEST_RUN.txt` and `outputs/current`.
- The run dashboard refreshes periodically and supports click-to-zoom controls (`+`, `-`, `reset`) on images.
- Refresh polling stops automatically when the run status becomes `completed`.

## Posterior sampling options

Posterior sampling is likelihood-guided and configured via:

- `--posterior-digit` (default `7`)
- `--posterior-observed-fraction` (default `0.7`)
- `--posterior-guidance-scale` (default `1.5`)
- `--posterior-guidance-min-frac` (default `0.25`, low-noise-end guidance floor as a fraction of full scale)
- `--posterior-guidance-power` (default `1.5`, annealing exponent for timestep-dependent guidance)
- `--posterior-likelihood-sigma` (default `0.1`)
- `--posterior-noise-aware-coeff` (default `0.05`, adds timestep noise term to effective likelihood variance)
- `--posterior-disable-hard-consistency` (if set, disables projection/data-consistency on observed pixels)
- `--num-posterior-samples` (default `8`)

Current posterior sampler improvements:
- Noise-aware likelihood variance scheduling with timestep-dependent effective sigma.
- Guidance annealing across denoising steps.
- Hard data-consistency projection on observed pixels during reverse sampling.

### Dashboard preview

<img src="dashboard_screenshot_v2.png" alt="MNIST diffusion dashboard preview" width="900" />
