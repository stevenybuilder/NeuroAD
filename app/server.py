"""
server — the interactive live backend for the NeuroAD Discovery Engine.

A dependency-free stdlib HTTP server (no FastAPI/Flask) so it runs from a clean
clone and deploys in the same container. It exposes the L5 entry point over HTTP:
a researcher POSTs a plain-language hypothesis + a registered dataset and gets
back the refereed, honesty-stamped ExperimentCard — INCLUDING the live
molecule/wet-lab translation lead (PI4AD -> AlphaFold -> repurposing).

Routes
    GET  /api/health              -> {status, claude_live, datasets}
    GET  /api/datasets            -> {datasets: [...]}  (loaders.AVAILABLE)
    POST /api/investigate         -> ExperimentCard.to_dict()
         body: {"hypothesis": str, "dataset": str, "seed"?: int, "api"?: bool}
    POST /api/orchestrate         -> Claude-as-orchestrator tool-runner result
         body: {"goal": str, "api"?: bool}   (live iff ANTHROPIC_API_KEY set,
         else a scripted deterministic pipeline over the same tools)
    GET  / and static assets      -> app/index.html, app/demo_data.json

Honesty / cost contract:
  * The referee + translation run OFFLINE and deterministic by default (no cost,
    no network beyond the optional AlphaFold/PI4AD live fetch inside translation).
  * ``api=true`` enables the live Claude adjudicator ONLY when ANTHROPIC_API_KEY
    is set; otherwise it is silently ignored and the deterministic templates run.
    /api/health reports ``claude_live`` so the UI can show the true state.

Run:  PYTHONPATH=src ./.venv/bin/python -m app.server   (PORT env, default 8080)
"""
from __future__ import annotations

import json
import logging
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

_APP_DIR = Path(__file__).resolve().parent
_ROOT = _APP_DIR.parent
# Make the src/ layout importable when run as `python -m app.server` from root.
sys.path.insert(0, str(_ROOT / "src"))

_log = logging.getLogger("neuroad.server")

_MAX_HYPOTHESIS_CHARS = 2000
_STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/demo_data.json": ("demo_data.json", "application/json"),
}


def _claude_live() -> bool:
    """True only if a real key is present — the honest live-adjudicator signal."""
    return bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())


def _available_datasets() -> list[str]:
    from neuroad.data import loaders
    return list(loaders.AVAILABLE)


class Handler(BaseHTTPRequestHandler):
    server_version = "NeuroAD/1.0"

    # -- helpers ----------------------------------------------------------
    def _send_json(self, obj: dict, status: int = 200) -> None:
        body = json.dumps(obj, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send_static(self, name: str, ctype: str) -> None:
        path = _APP_DIR / name
        if not path.exists():
            self._send_json({"error": f"{name} not found"}, 404)
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if name == "demo_data.json":
            self.send_header("Cache-Control", "no-store")
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:  # quieter default logging
        _log.info("%s - %s", self.address_string(), fmt % args)

    # -- routes -----------------------------------------------------------
    def do_OPTIONS(self) -> None:  # CORS preflight
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route == "/api/health":
            self._send_json({
                "status": "ok",
                "claude_live": _claude_live(),
                "datasets": _available_datasets(),
            })
            return
        if route == "/api/datasets":
            self._send_json({"datasets": _available_datasets()})
            return
        if route in _STATIC:
            self._send_static(*_STATIC[route])
            return
        self._send_json({"error": f"not found: {route}"}, 404)

    def do_POST(self) -> None:
        route = urlparse(self.path).path
        if route not in ("/api/investigate", "/api/orchestrate"):
            self._send_json({"error": f"not found: {route}"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
        except Exception as exc:  # noqa: BLE001
            self._send_json({"error": f"bad JSON body: {exc}"}, 400)
            return

        if route == "/api/orchestrate":
            self._handle_orchestrate(payload)
            return

        hypothesis = str(payload.get("hypothesis", "")).strip()
        dataset = str(payload.get("dataset", "")).strip()
        if not hypothesis:
            self._send_json({"error": "hypothesis is required"}, 400)
            return
        if len(hypothesis) > _MAX_HYPOTHESIS_CHARS:
            self._send_json({"error": "hypothesis too long"}, 400)
            return
        if not dataset:
            self._send_json(
                {"error": "dataset is required", "datasets": _available_datasets()}, 400)
            return

        # Live Claude only when a key is actually present — never fake it.
        want_api = bool(payload.get("api", False)) and _claude_live()
        try:
            seed = int(payload.get("seed", 0))
        except (TypeError, ValueError):
            seed = 0

        # The dataset vocabulary is broader than the AVAILABLE catalogue (e.g.
        # 'adni:3t', 'adni:combat', 'oasis:neurojepa'); let the loader be the
        # source of truth — an unknown name raises ValueError -> 400, everything
        # else is a genuine 500.
        try:
            from neuroad.harness import orchestrator
            xcard = orchestrator.investigate(
                hypothesis, dataset, api=want_api, seed=seed)
            result = xcard.to_dict()
            result["_meta"] = {
                "claude_live": want_api,
                "dataset": dataset,
                "hypothesis": hypothesis,
            }
            self._send_json(result)
        except ValueError as exc:  # unknown dataset name from loaders.load
            self._send_json(
                {"error": str(exc), "dataset": dataset,
                 "datasets": _available_datasets()}, 400)
        except Exception as exc:  # noqa: BLE001
            _log.exception("investigate failed")
            self._send_json(
                {"error": f"investigate failed: {exc}", "dataset": dataset}, 500)

    def _handle_orchestrate(self, payload: dict) -> None:
        """Claude-as-orchestrator: sequence the engine's tools toward a goal.

        Runs the live tool-runner iff a key is present (or api=true is forced and
        a key exists); otherwise a scripted deterministic pipeline over the same
        tools. The response's ``path`` says which drove it — never faked."""
        goal = str(payload.get("goal", "")).strip()
        if not goal:
            self._send_json({"error": "goal is required"}, 400)
            return
        if len(goal) > _MAX_HYPOTHESIS_CHARS:
            self._send_json({"error": "goal too long"}, 400)
            return
        # api: None -> auto (live iff key present); an explicit true only takes
        # effect when a key is actually configured.
        raw_api = payload.get("api", None)
        api = None if raw_api is None else (bool(raw_api) and _claude_live())
        try:
            from neuroad.harness import agent
            result = agent.orchestrate(goal, api=api)
            self._send_json(result)
        except Exception as exc:  # noqa: BLE001
            _log.exception("orchestrate failed")
            self._send_json({"error": f"orchestrate failed: {exc}"}, 500)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    port = int(os.environ.get("PORT", "8080"))
    host = os.environ.get("HOST", "0.0.0.0")
    httpd = ThreadingHTTPServer((host, port), Handler)
    live = "LIVE (key present)" if _claude_live() else "offline templates"
    print(f"NeuroAD backend on http://{host}:{port}  |  Claude: {live}")
    print("  GET  /api/health   GET /api/datasets   POST /api/investigate")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
