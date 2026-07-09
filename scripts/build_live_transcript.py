#!/usr/bin/env python3
"""build_live_transcript.py — reports/live_transcript.json from the pipeline.

Captures the three-voice courtroom (prosecution / defense / judge) plus the
reviewer self-critique and narration on a REFUSED (KILL) case — the beat where
"Claude as adjudicator" is most interesting is a refusal, not a rubber-stamp.

Honesty contract: the ``live`` / ``last_call_live`` flags are taken verbatim
from ``neuroad.claude._client.model_badge()``. With no ANTHROPIC_API_KEY the
deterministic offline template produces the text and the file is stamped
live=false / is_placeholder=true. With a key present and calls actually reaching
the API, last_call_live becomes true and the file is a genuine live transcript.
It NEVER writes live=true over template content.

Usage:
    PYTHONPATH=src ./.venv/bin/python scripts/build_live_transcript.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from neuroad import pipeline
from neuroad.claude import _client, courtroom
from neuroad.contract import Claim
from neuroad.data import loaders


def main() -> int:
    # The KILL preset — a refused case, so the courtroom shows Claude declining
    # to promote and the reviewer arguing against the (already negative) verdict.
    df = loaders.load("synthetic:KILL", seed=6)
    claim = Claim(
        claim_id="SYN-B",
        claim_text=("A structural embedding signature separates MCI converters "
                    "from non-converters."),
        target="conversion", group_a="MCI converters",
        group_b="MCI non-converters")
    claim.substrate = "frozen Neuro-JEPA structural embeddings"

    card = pipeline.run_referee(df, claim)

    adj = getattr(card, "adjudication", None)
    if not isinstance(adj, dict):
        adj = courtroom.adjudicate(card.claim, getattr(card, "tests_evidence", []))
    adj = adj or {}
    reviewer = getattr(card, "reviewer", {}) or {}
    badge = _client.model_badge()
    live = bool(badge.get("last_call_live"))

    payload = {
        "_README": (
            "Frozen Claude courtroom + self-critique transcript for the demo, on a "
            "REFUSED (KILL) case. The live / last_call_live flags below are taken "
            "verbatim from the Claude client badge: with no ANTHROPIC_API_KEY the "
            "deterministic OFFLINE TEMPLATE produced this text (live=false). Rerun "
            "with a key present to capture a genuine live-Claude transcript — this "
            "generator NEVER stamps live=true over template output."),
        "captured_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "live": live,
        "last_call_live": live,
        "is_placeholder": not live,
        "case": "REFUSED (synthetic KILL)",
        "model_badge": badge,
        "claim": {
            "claim_id": card.claim.claim_id,
            "claim_text": card.claim.claim_text,
            "substrate": card.claim.substrate,
            "target": card.claim.target,
        },
        "verdict": card.verdict.value if hasattr(card.verdict, "value") else card.verdict,
        "score": card.score,
        "promoted": bool(card.promoted),
        "courtroom": {k: adj.get(k, "") for k in
                      ("prosecution", "defense", "judge_reasoning")},
        "reviewer_self_critique": {
            "critique": reviewer.get("critique", []),
            "revised_caveats": reviewer.get("revised_caveats", []),
        },
        "narration": getattr(card, "narration", "") or "",
    }

    out = ROOT / "reports" / "live_transcript.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"[build_live_transcript] wrote {out} "
          f"(live={live}, verdict={payload['verdict']}, promoted={payload['promoted']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
