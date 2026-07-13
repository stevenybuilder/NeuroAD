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
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

_APP_DIR = Path(__file__).resolve().parent
_ROOT = _APP_DIR.parent
# Make the src/ layout importable when run as `python -m app.server` from root.
sys.path.insert(0, str(_ROOT / "src"))

_log = logging.getLogger("neuroad.server")

_MAX_HYPOTHESIS_CHARS = 2000

# --- "Ask Claude" rail: repo-grounded Q&A (NOT the deterministic referee) -----
# A real LLM answers questions about the current investigation, grounded ONLY in
# the case + hypothesis registry we pass it. Live iff ANTHROPIC_API_KEY is set.
_ASK_MODEL = os.environ.get("ASK_MODEL", "claude-opus-4-7")
_ASK_SYSTEM = (
    "You are Claude, the research partner inside NeuroAD — an Alzheimer's "
    "structural-MRI discovery referee. NeuroAD points a linear probe at frozen "
    "Neuro-JEPA brain-MRI embeddings, runs a five-test confound gauntlet "
    "(site/scanner, age/sex, brain-age, biomarker anchor, replication), and "
    "promotes only the signals that survive. Answer the scientist's question about "
    "THIS investigation, grounded ONLY in the CONTEXT below (the live case, its "
    "ranked targets, cohort, gauntlet tests, and the hypothesis registry).\n\n"
    "VOICE — write like a sharp, generous colleague at the bench: frank, warm, "
    "plain-spoken, and precise. Treat the reader as an expert adult. Lead with a "
    "direct answer to what was asked, then support it. No cold formality, no "
    "throat-clearing, no filler.\n\n"
    "HONESTY — be diplomatically honest, never dishonestly diplomatic. Give a "
    "committed assessment; don't retreat into vague, noncommittal mush to avoid "
    "taking a position. Assert only what the context supports, and never create a "
    "false impression through selective emphasis or by implying more than the data "
    "shows.\n\n"
    "CALIBRATION — carry uncertainty proportional to the evidence. When the context "
    "doesn't answer the question, say so plainly instead of inventing an answer. "
    "Use hedges ('suggests', 'may', 'is consistent with') only where they reflect "
    "real uncertainty — never to dress an inconclusive or unproven claim as if it "
    "were established.\n\n"
    "EVIDENCE — cite concrete numbers and fields from the context (effect sizes, n, "
    "p-values, priority scores, leakage margins, gauntlet verdicts); don't state a "
    "number you can't trace to the context. Sharply distinguish what SURVIVED the "
    "gauntlet (a finding validated on this data) from ranked candidate targets, "
    "which are testable HYPOTHESES for the bench, not validated drugs. "
    "Frozen-embedding scanner/site leakage is published prior art — name it as "
    "such, don't claim it as a discovery.\n\n"
    "Keep it tight: usually 2-5 sentences; expand only when the question truly "
    "needs it."
)


def _build_ask_context(payload: dict) -> str:
    """Assemble a compact, grounded knowledge context from the passed case + the
    committed hypothesis registry. Bounded so the prompt stays cheap."""
    blocks: list[str] = []
    # Canonical project fact base — the single source of truth for data scale,
    # voxels, GB, the honest headline metrics (and the confusable variants NOT to
    # swap in), the stack, discovery framing, and common Q&A. Injected FIRST and
    # marked authoritative so Ask-Claude cites these verbatim instead of
    # improvising numbers that contradict the rest of the demo.
    try:
        kb = json.loads((_APP_DIR / "knowledge_base.json").read_text())
        kb_facts = {k: kb[k] for k in (
            "project", "data_scale", "voxels_and_training", "canonical_metrics",
            "the_stack", "discovery_framing", "how_our_data_compares",
            "deployment_provenance", "common_qa") if k in kb}
        blocks.append(
            "PROJECT KNOWLEDGE BASE (CANONICAL — authoritative; cite these facts and "
            "numbers verbatim, and NEVER state a figure that contradicts "
            "canonical_metrics; prefer the honest confound-matched number over the "
            "raw one):\n" + json.dumps(kb_facts, default=str)[:9000])
    except Exception:  # noqa: BLE001
        pass
    hyp = str(payload.get("hypothesis", "")).strip()
    if hyp:
        blocks.append("HYPOTHESIS UNDER INVESTIGATION:\n" + hyp[:_MAX_HYPOTHESIS_CHARS])
    case = payload.get("case")
    if isinstance(case, dict) and case:
        keep = {k: case[k] for k in (
            "verdict", "score", "naive_effect", "leakage_margin", "tests",
            "translation", "cohort", "biology_hypothesis", "next_experiment",
            "caveats") if k in case}
        blocks.append("CURRENT INVESTIGATION CASE (real referee output):\n"
                      + json.dumps(keep, default=str)[:6000])
    try:
        reg = json.loads((_APP_DIR / "hypothesis_registry.json").read_text())
        blocks.append("HYPOTHESIS REGISTRY (real hypothesis -> real cohort -> cited "
                      "verdict):\n" + json.dumps(reg.get("hypotheses", []), default=str)[:6000])
    except Exception:  # noqa: BLE001
        pass
    return "\n\n".join(blocks) if blocks else "(no structured context provided)"

# The live Silent-Failure Guard (MRI QC) backend. The frontend talks to it via a
# same-origin /api/sfg/* proxy so the demo stays one origin and self-contained.
# Point SFG_BACKEND at a deployed instance to use that instead of a local run.
_SFG_BASE = os.environ.get("SFG_BACKEND", "http://127.0.0.1:8091").rstrip("/")
# neuroad.html is the served demo surface; "/" serves it. zui.html and
# index.html remain reachable at their own paths. Routing only — no other
# server logic changes.
_STATIC = {
    "/": ("neuroad.html", "text/html; charset=utf-8"),
    "/neuroad.html": ("neuroad.html", "text/html; charset=utf-8"),
    "/zui.html": ("zui.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    # claude_science.html is the "Start" / Claude entry surface. Additive routing
    # only — "/" (neuroad.html) and all existing routes are unchanged.
    "/start": ("claude_science.html", "text/html; charset=utf-8"),
    "/claude": ("claude_science.html", "text/html; charset=utf-8"),
    "/claude_science.html": ("claude_science.html", "text/html; charset=utf-8"),
    "/demo_data.json": ("demo_data.json", "application/json"),
    # Demo-drivable hypothesis -> real-cohort -> cited-verdict registry, fetched by
    # the Claude Science (/start) prefill so the happy-path maps to real data.
    "/hypothesis_registry.json": ("hypothesis_registry.json", "application/json"),
    # Frontend assets for the live 3D brain viewer (NiiVue + a bundled real
    # MNI152 T1). Routing only — not part of the science pipeline.
    "/vendor/niivue.umd.js": ("vendor/niivue.umd.js", "text/javascript; charset=utf-8"),
    "/scans/mni152.nii.gz": ("scans/mni152.nii.gz", "application/gzip"),
    # 3Dmol.js viewer for the in-demo protein-structure payoff on the ranked-targets
    # card. Routing only — not part of the science pipeline. Per-gene AlphaFold CIFs
    # are served by the /structures/<GENE>.cif prefix handler in do_GET.
    "/vendor/3Dmol-min.js": ("vendor/3Dmol-min.js", "text/javascript; charset=utf-8"),
}

# The root page is env-controlled so the SAME backend serves NeuroAD at "/"
# (localhost default) or Claude Science at "/" (the Cloud Run demo entry, matching
# the static deploy) without a code change. Only known pages are honored.
_ROOT_PAGE = os.environ.get("ROOT_PAGE", "neuroad.html")
if _ROOT_PAGE in ("neuroad.html", "claude_science.html", "zui.html", "index.html"):
    _STATIC["/"] = (_ROOT_PAGE, "text/html; charset=utf-8")


def _claude_live() -> bool:
    """True only if a real key is present — the honest live-adjudicator signal."""
    return bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())


def _available_datasets() -> list[str]:
    from neuroad.data import loaders
    return list(loaders.AVAILABLE)


def _enrich_case(xcard, hypothesis: str, dataset: str, seed: int) -> dict:
    """Build the rich ``case`` shape the tree / story UI renders from a refereed
    ExperimentCard, reusing the already-tested build_demo_data transformers (DRY).

    Returns a dict with the real ``tests[]`` (leakage/anchor/replication), a top
    level ``leakage_margin``/``score``/``verdict``, ``cohort`` summary,
    ``courtroom``/``narration``/``translation`` (when the survivor carries them),
    an ``investigate`` plan-out block, and a normalized decision ``tree``. Purely
    additive — the caller attaches it as ``result['case']`` and every existing
    top-level key of the plain card is preserved. Raises on failure (the caller
    degrades to the plain card rather than 500-ing)."""
    from neuroad.data import loaders
    from neuroad import contract
    import app.build_demo_data as B

    df = loaders.load(dataset, seed=seed)
    card = xcard.card
    claim = card.claim
    badge = loaders.honest_substrate(dataset)
    promoted = bool(card.to_dict().get("promoted"))
    scaffold = {
        "id": getattr(claim, "claim_id", "case") or "case",
        "label": "Case",
        "kind": "SURVIVOR" if promoted else "KILL",
        "substrate_badge": badge,
        "claim": {
            "claim_id": getattr(claim, "claim_id", "") or "",
            "claim_text": getattr(claim, "claim_text", "") or "",
            "target": getattr(claim, "target", "") or "",
            "group_a": getattr(claim, "group_a", "") or "",
            "group_b": getattr(claim, "group_b", "") or "",
            "substrate": getattr(claim, "substrate", "") or badge,
            "head": getattr(claim, "head", "linear probe") or "linear probe",
        },
        "naive_effect": {},
        "leakage_margin": {},
        "score": 0,
        "verdict": "",
        "promoted": False,
        "tests": [B._test(k, "not_available", 0.5, l, "", {})
                  for (k, l, q, w, s) in B.GAUNTLET_META],
        "confound_leaderboard": [],
        "double_dissociation": {},
        "caveats": [],
        "scatter": {"n": 90, "n_scanners": 2, "seed": seed,
                    "outcome_gap": 2.0, "scanner_gap": 1.0, "converter_frac": 0.35},
    }
    case = B._real_case(scaffold, card, df)  # real tests/leakage/score/courtroom/…
    case["investigate"] = B._investigate_block(hypothesis, dataset, seed, case)
    cohort = contract.cohort_summary(df)
    # Short cohort stamp ("REAL ADNI" / "REAL OASIS"), matching the demo_data
    # payload so the frontend badge chip renders identically on the live path;
    # the long honest substrate description rides in ``substrate_line``.
    _fam = (dataset.split(":")[0].strip().upper() or "COHORT")
    cohort["badge"] = f"REAL {_fam}"
    cohort["substrate_line"] = badge
    case["cohort"] = cohort
    case["tree"] = B._derive_tree(case)  # UI ignores it, but honest / audit-complete
    return case


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

    def _proxy_sfg(self, method: str, body: bytes | None = None) -> None:
        """Forward /api/sfg/<rest> -> <SFG_BASE>/api/<rest>, streaming the response
        through verbatim (JSON for flags, gzip/obj bytes for volumes/overlays)."""
        parsed = urlparse(self.path)
        rest = parsed.path[len("/api/sfg/"):]
        target = f"{_SFG_BASE}/api/{rest}"
        if parsed.query:
            target += "?" + parsed.query
        headers = {}
        ctype = self.headers.get("Content-Type")
        if ctype:
            headers["Content-Type"] = ctype
        req = urllib.request.Request(target, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                payload = resp.read()
                self.send_response(resp.status)
                self.send_header(
                    "Content-Type", resp.headers.get("Content-Type", "application/octet-stream"))
                self.send_header("Content-Length", str(len(payload)))
                self._cors()
                self.end_headers()
                self.wfile.write(payload)
        except urllib.error.HTTPError as exc:  # backend answered with an error
            payload = exc.read()
            self.send_response(exc.code)
            self.send_header("Content-Type", exc.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(payload)))
            self._cors()
            self.end_headers()
            self.wfile.write(payload)
        except (urllib.error.URLError, ConnectionError, OSError) as exc:
            # Backend not running — the frontend degrades to an offline notice.
            self._send_json({"error": "sfg backend unreachable", "detail": str(exc),
                             "sfg_base": _SFG_BASE}, 503)

    # -- routes -----------------------------------------------------------
    def do_OPTIONS(self) -> None:  # CORS preflight
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route.startswith("/api/sfg/"):
            self._proxy_sfg("GET")
            return
        # Serve bundled scan/overlay volumes from app/scans/*.nii.gz (frontend
        # assets for the 3D viewer). Basename-only, so no path traversal.
        if route.startswith("/scans/") and route.endswith(".nii.gz"):
            name = route[len("/scans/"):]
            if "/" not in name and ".." not in name:
                self._send_static(f"scans/{name}", "application/gzip")
                return
        # Serve vendored AlphaFold structures from app/structures/<GENE>.cif for the
        # in-demo 3D protein viewer. Basename-only, so no path traversal.
        if route.startswith("/structures/") and route.endswith(".cif"):
            name = route[len("/structures/"):]
            if "/" not in name and ".." not in name:
                self._send_static(f"structures/{name}", "chemical/x-cif")
                return
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
        if route.startswith("/api/sfg/"):
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b""
            self._proxy_sfg("POST", raw)
            return
        if route not in ("/api/investigate", "/api/orchestrate", "/api/ask"):
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

        if route == "/api/ask":
            self._handle_ask(payload)
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
            # Additive: attach the rich `case` shape the tree / story UI renders
            # (tests[], cohort, leakage_margin, courtroom, narration, translation,
            # tree). Every existing top-level key of the plain card is preserved;
            # a failure here degrades to the plain card (never a 500 regression).
            try:
                result["case"] = _enrich_case(xcard, hypothesis, dataset, seed)
            except Exception:  # noqa: BLE001
                _log.exception("investigate case enrichment failed; plain card")
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

    def _handle_ask(self, payload: dict) -> None:
        """Ask Claude rail: a real LLM answers a question about the current
        investigation, grounded in the passed case + registry. Live iff a key is
        set; otherwise returns an honest offline notice (never a faked answer)."""
        question = str(payload.get("question", "")).strip()
        if not question:
            self._send_json({"error": "question is required"}, 400)
            return
        if len(question) > _MAX_HYPOTHESIS_CHARS:
            self._send_json({"error": "question too long"}, 400)
            return
        if not _claude_live():
            self._send_json({
                "answer": "Ask Claude is offline — no ANTHROPIC_API_KEY is set on "
                          "the server, so I can't reach the model. Set the key and "
                          "restart to get answers grounded in this investigation.",
                "live": False, "model": None})
            return
        context = _build_ask_context(payload)
        try:
            import anthropic  # lazy import: offline path needs no dependency
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model=_ASK_MODEL,
                max_tokens=1024,
                system=_ASK_SYSTEM,
                messages=[{"role": "user",
                           "content": f"CONTEXT:\n{context}\n\n---\n\nQUESTION: {question}"}],
            )
            answer = "".join(
                b.text for b in msg.content if getattr(b, "type", None) == "text").strip()
            self._send_json({"answer": answer or "(no answer returned)",
                             "live": True, "model": _ASK_MODEL})
        except Exception as exc:  # noqa: BLE001
            _log.exception("ask failed")
            self._send_json({"error": f"ask failed: {exc}", "live": False}, 500)


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
