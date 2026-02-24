# Purdue HPC (Gilbreth) Codex Skill

Use this skill when Codex needs to run experiments, prepare jobs, and manage files on Purdue RCAC Gilbreth while keeping reproducible scripts in this repository.

## Goal

- Keep development scripts versioned in this repo.
- Run compute workloads from cluster scratch/depot (not `$HOME`).
- Submit and monitor Slurm jobs in a repeatable way.

## Prerequisites

- You can authenticate with Purdue credentials + BoilerKey/2FA.
- You have an active allocation/account on the target cluster.
- You have this repo checked out locally:
  - `/Users/rmaulik/Desktop/Codex_Stuff/PurdueHPC_Codex`

## Cluster login pattern

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

Example:

```bash
echo "$CLUSTER_SCRATCH"
mkdir -p "$CLUSTER_SCRATCH/codex_test"
```

## Repo-to-cluster workflow

1. Create or update scripts in this repo.
2. Mirror those scripts to cluster scratch path for execution.
3. Run interactively for quick checks or submit with Slurm.
4. Capture important outcomes (commands, job ids, output summaries) back in repo docs.

## Python environment pattern (cluster side)

```bash
cd "$CLUSTER_SCRATCH/codex_test"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Slurm workflow

Submit:

```bash
sbatch submit_mnist_diffusion_gilbreth.slurm
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

## Gilbreth-specific Slurm notes

Validated on Gilbreth with:

- Partition: `a100-40gb`
- GPU request: `--gres=gpu:1`
- CPU request: `--cpus-per-task=8`
- Memory request: `--mem=240G` (required by Gilbreth for submission)

Useful checks:

```bash
sinfo -s
sfeatures
```

If jobs remain pending with reason `AssocGrpGRES`, wait for active account GPU jobs to clear or submit to a different partition permitted by your allocation.

## Current validated experiment scripts in this repo

- `mnist_diffusion.py`
- `submit_mnist_diffusion_gilbreth.slurm`
- `submit_mnist_diffusion_gilbreth_long.slurm`
- `requirements.txt`

## Safety rules for Codex actions

- Never store passwords, BoilerKey responses, or secrets in repo files.
- Do not commit full raw logs unless explicitly requested.
- Prefer adding concise output tails/summaries to `README.md`.
- Keep scratch paths user-specific and explicit in scripts.
