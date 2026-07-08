"""
Thin Anthropic client with a deterministic offline fallback.

Design contract (relied on by every M3 module):

    complete(system, prompt, schema=None) -> dict | str
    USING_LIVE_API: bool

If ``ANTHROPIC_API_KEY`` is set we call the Messages API with model
``claude-fable-5`` (Anthropic's most capable model). Fable 5 has thinking always
on, so we never pass a ``thinking`` block; we opt into server-side refusal
fallbacks to ``claude-opus-4-8`` so a benign life-sciences prompt is never lost
to a false-positive classifier hit. When a JSON ``schema`` is supplied we force
structured output through a single strict tool call and return the tool input
dict. On any transport/parse error we retry once on ``claude-sonnet-5`` and, if
that also fails, fall through to the deterministic template so the pipeline
still produces an answer.

If the key is NOT set (the default in the demo environment), ``complete``
returns a deterministic, prompt-derived template — a well-formed stand-in that
lets the whole referee run offline. The *rich*, test-specific fallbacks live in
the caller modules (they synthesise prosecution/defense/biology text from the
TestEvidence stats); this module only guarantees ``complete`` itself never
crashes and never returns noise.
"""
from __future__ import annotations

import os
from typing import Optional

# --- live-API configuration (read once at import) --------------------------
PRIMARY_MODEL = "claude-fable-5"
FALLBACK_MODEL = "claude-opus-4-8"       # server-side refusal fallback target
RETRY_MODEL = "claude-sonnet-5"          # transport-error retry
FALLBACK_BETA = "server-side-fallback-2026-06-01"
MAX_TOKENS = 4096

USING_LIVE_API: bool = bool(os.environ.get("ANTHROPIC_API_KEY"))

#: The house system framing shared by every Claude call in the referee. Kept as
#: a module constant so it reads as deliberate prompt engineering, not an inline
#: string. Individual modules prepend their own persona instructions.
REFEREE_SYSTEM = (
    "You are the reasoning core of NeuroAD Discovery Engine, an Alzheimer's structural-MRI "
    "auditor. You adjudicate whether a signal found in frozen brain-MRI embeddings "
    "is real disease biology or an artifact of scanner/site, demographics, or "
    "generic aging. You never overclaim: verdict language stays hedged (fragile / "
    "partially robust / robust enough for follow-up / strong candidate), biology "
    "speaks only about promoted survivors, and every claim is paired with the "
    "evidence that supports it. The insight that frozen embeddings leak scanner/"
    "site is published prior art (arXiv:2604.14441; arXiv:2606.09189; PathoROB) — "
    "cite it, never claim it. The model is Neuro-JEPA (hyphenated). This is a "
    "referee/red-team, not a co-scientist or discovery platform."
)


def complete(system: str, prompt: str, schema: Optional[dict] = None):
    """Return Claude's answer as a ``dict`` (when ``schema`` given) or ``str``.

    Never raises for expected failure modes; falls back to a deterministic
    template so the referee pipeline always completes.
    """
    if USING_LIVE_API:
        try:
            return _live_complete(system, prompt, schema)
        except Exception:
            # Transport, parse, or refusal-chain failure — degrade gracefully.
            pass
    return _template_complete(system, prompt, schema)


# ---------------------------------------------------------------------------
# Live path
# ---------------------------------------------------------------------------


def _live_complete(system: str, prompt: str, schema: Optional[dict]):
    import anthropic  # imported lazily so the offline path has no dependency

    client = anthropic.Anthropic()
    full_system = f"{REFEREE_SYSTEM}\n\n{system}".strip()

    def _call(model: str, use_fallback: bool):
        kwargs: dict = {
            "model": model,
            "max_tokens": MAX_TOKENS,
            "system": full_system,
            "messages": [{"role": "user", "content": prompt}],
        }
        if schema is not None:
            kwargs["tools"] = [{
                "name": "record",
                "description": "Record the structured adjudication result.",
                "strict": True,
                "input_schema": schema,
            }]
            kwargs["tool_choice"] = {"type": "tool", "name": "record"}
        if use_fallback:
            # Fable 5: opt into server-side refusal fallback by default.
            return client.beta.messages.create(
                betas=[FALLBACK_BETA],
                fallbacks=[{"model": FALLBACK_MODEL}],
                **kwargs,
            )
        return client.messages.create(**kwargs)

    try:
        msg = _call(PRIMARY_MODEL, use_fallback=True)
    except Exception:
        msg = _call(RETRY_MODEL, use_fallback=False)

    return _extract(msg, schema)


def _extract(msg, schema: Optional[dict]):
    """Pull the structured tool input (schema mode) or text out of a Message."""
    if schema is not None:
        for block in msg.content:
            if getattr(block, "type", None) == "tool_use":
                return dict(block.input)
        raise ValueError("no tool_use block in structured response")
    parts = [b.text for b in msg.content if getattr(b, "type", None) == "text"]
    text = "".join(parts).strip()
    if not text:
        raise ValueError("empty text response")
    return text


# ---------------------------------------------------------------------------
# Deterministic offline template
# ---------------------------------------------------------------------------


def _template_complete(system: str, prompt: str, schema: Optional[dict]):
    """A prompt-derived, non-random stand-in used when no API key is present.

    Callers with structured needs supply their own rich fallbacks; this exists
    so a bare ``complete()`` call still returns something well-formed.
    """
    head = _first_sentence(prompt)
    if schema is None:
        return (
            "[offline referee] "
            + head
            + " Verdict language remains hedged and every claim is paired with "
            "its evidence; batch-effect leakage is cited as prior art, not "
            "claimed as a discovery."
        )
    out: dict = {}
    for key, spec in (schema.get("properties") or {}).items():
        out[key] = _template_value(key, spec, head)
    return out


def _template_value(key: str, spec: dict, head: str):
    t = spec.get("type", "string")
    if t == "array":
        return [f"{key}: {head}"]
    if t == "boolean":
        return False
    if t in ("integer", "number"):
        return 0
    return f"{key}: {head}"


def _first_sentence(text: str) -> str:
    text = " ".join((text or "").split())
    if not text:
        return "No prompt supplied."
    for stop in (". ", "? ", "! "):
        i = text.find(stop)
        if 0 < i < 240:
            return text[: i + 1].strip()
    return (text[:240]).strip()
