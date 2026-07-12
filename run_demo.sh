#!/usr/bin/env bash
# Launch the NeuroAD demo WITH the live Silent-Failure Guard MRI-QC backend.
#
#   1. the SFG FastAPI backend (mri_visualizations/backend) on :8091 — runs the
#      deterministic imaging checks over the real IXI cohort and serves volumes;
#   2. the NeuroAD demo server on :8080, which proxies /api/sfg/* to (1).
#
# First run only: create the venv + fetch a small real IXI subset:
#   python3.12 -m venv mri_visualizations/.venv-sfg
#   mri_visualizations/.venv-sfg/bin/pip install fastapi "uvicorn[standard]" \
#       nibabel numpy scipy scikit-image "pydantic>=2" SimpleITK
#   mri_visualizations/.venv-sfg/bin/python mri_visualizations/scripts/fetch_ixi_stream.py 2
#
# (Optional) skull-strip's SynthStrip reference needs weights + torch — without
# them the check still fires on the classical stripper, just no Dice comparison.
set -euo pipefail
cd "$(dirname "$0")"

SFG_PORT="${SFG_PORT:-8091}"
DEMO_PORT="${PORT:-8080}"

echo "Starting Silent-Failure Guard backend on :$SFG_PORT ..."
( cd mri_visualizations/backend && PYTHONPATH=. ../.venv-sfg/bin/python -m uvicorn sfg.server:app \
    --host 127.0.0.1 --port "$SFG_PORT" ) &
SFG_PID=$!

cleanup() { echo; echo "shutting down"; kill "$SFG_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

sleep 4  # let fixture generation + check registration finish
echo "Starting NeuroAD demo on http://localhost:$DEMO_PORT  (flags proxied to :$SFG_PORT)"
SFG_BACKEND="http://127.0.0.1:$SFG_PORT" PORT="$DEMO_PORT" PYTHONPATH=src ./.venv/bin/python -m app.server
