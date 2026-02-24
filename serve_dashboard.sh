#!/bin/bash
set -euo pipefail

cd /scratch/gautschi/rmaulik/codex_test/outputs
PORT="${2:-8080}"
echo "Serving live dashboard root from: $(pwd)"
echo "This always follows the latest run via outputs/current and outputs/LATEST_RUN.txt"
echo "Open locally with SSH tunnel: ssh -N -L ${PORT}:localhost:${PORT} <username>@gautschi.rcac.purdue.edu"
echo "Then browse: http://localhost:${PORT}/dashboard.html"
python -m http.server "$PORT"
