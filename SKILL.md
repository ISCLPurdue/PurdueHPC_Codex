# Purdue HPC (Gautschi + Gilbreth) Codex Skill

Use this skill when Codex needs to run experiments, prepare jobs, and manage files on Purdue RCAC clusters (`gautschi`, `gilbreth`) while keeping reproducible scripts in this repository.

## Goal

- Keep development scripts versioned in this repo.
- Run compute workloads from cluster scratch/depot (not `$HOME`).
- Submit and monitor Slurm jobs in a repeatable way.

## Prerequisites

- You can authenticate with Purdue credentials + BoilerKey/2FA.
- You have an active allocation/account on the target cluster.
- You have this repo checked out locally:
  - `/Users/rmaulik/Desktop/Codex_Stuff/PurdueHPC_Codex`

## Cluster login patterns

Gautschi:

```bash
ssh <username>@gautschi.rcac.purdue.edu
```

Gilbreth:

```bash
ssh <username>@gilbreth.rcac.purdue.edu
```

After login, validate account/queue context:

```bash
slist
myquota
```

## Storage policy

Do not run experiments from `$HOME`.

- Use scratch for active jobs and temporary outputs.
- Use depot for larger persistent group/project data.

Example on Gautschi:

```bash
echo "$SCRATCH"
mkdir -p "$SCRATCH/codex_test"
```

## Repo-to-cluster workflow

1. Create or update scripts in this repo.
2. Mirror those scripts to cluster scratch path for execution.
3. Run interactively for quick checks or submit with Slurm.
4. Capture important outcomes (commands, job ids, output summaries) back in repo docs.

## Python environment pattern (cluster side)

```bash
cd /scratch/<cluster>/<username>/codex_test
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Slurm workflow

Submit:

```bash
sbatch submit_mnist_diffusion.slurm
```

Monitor running/pending jobs:

```bash
squeue -u <username>
```

Check completed job state:

```bash
sacct -j <jobid> --format=JobID,JobName,Partition,State,ExitCode,Elapsed,Start,End -P
```

Inspect output:

```bash
tail -n 100 slurm-<jobid>.out
```

## Gautschi-specific Slurm notes

- Partition must be explicitly set (for example `#SBATCH -p ai`).
- AI partition enforces CPU/GPU proportional requests; use 14 CPUs per 1 GPU.
- If submission fails, read `sbatch` error text and adjust directives first.

## Current validated experiment in this repo

- `mnist_diffusion.py`
- `submit_mnist_diffusion.slurm`
- `requirements.txt`

Validated run on Gautschi:

- Job ID: `8109439`
- State: `COMPLETED`
- Exit code: `0:0`
- Output artifact: `outputs/mnist_samples.png`

## Safety rules for Codex actions

- Never store passwords, BoilerKey responses, or secrets in repo files.
- Do not commit full raw logs unless explicitly requested.
- Prefer adding concise output tails/summaries to `README.md`.
- Keep scratch paths user-specific and explicit in scripts.

