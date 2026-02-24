#!/bin/bash
set -euo pipefail

SCRATCH_BASE="${CLUSTER_SCRATCH:-/scratch/gautschi/${USER}}"
WORKDIR="${1:-${SCRATCH_BASE}/codex_test}"
PORT="${2:-8080}"

cd "${WORKDIR}/outputs"
echo "Serving live dashboard root from: $(pwd)"
echo "This always follows the latest run via outputs/current and outputs/LATEST_RUN.txt"
echo "Open locally with SSH tunnel: ssh -N -L ${PORT}:localhost:${PORT} <username>@<cluster>.rcac.purdue.edu"
echo "Then browse: http://localhost:${PORT}/dashboard.html"
python -m http.server "$PORT"
