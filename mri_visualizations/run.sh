#!/usr/bin/env bash
# Start the Silent-Failure Guard: FastAPI backend + Vite frontend.
# Open http://localhost:5173 once both are up.
set -euo pipefail
cd "$(dirname "$0")"

ENV="${SFG_ENV:-sfg}"

cleanup() { kill 0 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "[sfg] starting backend on :8000 ..."
( cd backend && micromamba run -n "$ENV" uvicorn sfg.server:app --host 127.0.0.1 --port 8000 ) &

echo "[sfg] starting frontend on :5173 ..."
( cd frontend && npm run dev ) &

wait
