#!/usr/bin/env bash
# Cloud Run entrypoint that replicates the LOCAL run_demo.sh two-server setup in a
# single container:
#   1. the SFG imaging backend (uvicorn) on 127.0.0.1:8091 — deterministic MRI-QC
#      checks + volume serving over the bundled real ADNI/IXI/fixture cohort;
#   2. the NeuroAD demo server on $PORT (Cloud Run injects it, default 8080), which
#      proxies /api/sfg/* to (1) via the SFG_BACKEND default of 127.0.0.1:8091.
# Only $PORT is exposed by Cloud Run; :8091 stays container-internal.
set -euo pipefail

SFG_PORT="${SFG_PORT:-8091}"
DEMO_PORT="${PORT:-8080}"

echo "[start] launching Silent-Failure Guard backend on :${SFG_PORT}"
# Auto-restart the SFG process if it ever dies, so a single crash doesn't leave the
# brain-viz permanently on the cached fallback for the life of the container.
(
  cd /srv/mri_visualizations/backend
  while true; do
    PYTHONPATH=. python -m uvicorn sfg.server:app --host 127.0.0.1 --port "${SFG_PORT}" \
      || echo "[start] SFG backend exited ($?); restarting in 2s"
    sleep 2
  done
) &

# Give SFG a moment to generate fixtures + register checks before the demo starts
# accepting traffic, so the first brain-viz open is live rather than cached.
sleep 4

echo "[start] launching NeuroAD demo on :${DEMO_PORT} (SFG proxied to :${SFG_PORT})"
export SFG_BACKEND="http://127.0.0.1:${SFG_PORT}"
export PYTHONPATH=/srv/src
exec python -m app.server
