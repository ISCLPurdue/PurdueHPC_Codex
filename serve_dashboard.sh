#!/bin/bash
set -euo pipefail

cd /scratch/gautschi/rmaulik/codex_test/outputs
RUN_TAG="${1:-$(cat LATEST_RUN.txt)}"
cd "$RUN_TAG"

PORT="${2:-8080}"
echo "Serving dashboard for run: $RUN_TAG"
echo "Open locally with SSH tunnel: ssh -N -L ${PORT}:localhost:${PORT} <username>@gautschi.rcac.purdue.edu"
echo "Then browse: http://localhost:${PORT}/dashboard.html"
python -m http.server "$PORT"
