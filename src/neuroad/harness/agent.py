"""
agent — Claude as the ORCHESTRATOR (plan §4.2), not the determinator.

This is the "Claude harness / agentic orchestrator" the plan puts at the center:
Claude is handed the engine's capabilities AS TOOLS and decides which to call, in
what order, reasoning over what they return — "treating encoders as tools,
reasoning over fragments, and producing coherent hypotheses/outputs" (plan L125).

The hard line this module draws: **Claude orchestrates, the tools decide.** Every
fact — a kill/promote verdict, a gene priority, a pLDDT, a compound — is produced
by the deterministic engine (the gauntlet, PI4AD, AlphaFold, the repurposing
snapshot) and returned by a tool. Claude sequences and explains them; it is
instructed never to invent or override a verdict/score/priority. So the
anti-overclaim, reproducible-rigor identity of the engine is untouched — the LLM
adds orchestration and connective reasoning, not adjudication.

``orchestrate(goal)`` runs live (Anthropic tool-runner) when ANTHROPIC_API_KEY is
set and ``anthropic`` is installed; otherwise it runs a SCRIPTED deterministic
pipeline over the same tools and labels itself ``path="scripted_offline"`` so the
result is honest about which drove it. Either way the tools are identical, so the
science is byte-identical whether Claude or the script sequenced the calls.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Optional

_log = logging.getLogger("neuroad.harness.agent")


# ===========================================================================
# TOOLS — thin, JSON-returning wrappers over the deterministic engine.
# Each returns a compact JSON string (the tool-result content Claude reads).
# ===========================================================================

def list_datasets() -> str:
    """List the registered cohorts the engine can be pointed at.

    Returns a JSON array of dataset names usable as the ``dataset`` argument to
    ``referee_hypothesis`` / ``describe_cohort`` (e.g. "adni:3t", "oasis",
    "synthetic:SURVIVOR").
    """
    from ..data import loaders
    return json.dumps({"datasets": list(loaders.AVAILABLE)})


def describe_cohort(dataset: str) -> str:
    """Summarize a registered cohort without exposing subject-level data.

    Args:
        dataset: A registered dataset name (see ``list_datasets``).

    Returns a JSON object with n_subjects, embedding_dim, dx/label coverage,
    biomarker coverage, sites/scanners — enough to reason about what a contrast
    on this cohort can and cannot support.
    """
    from ..data import loaders
    from ..contract import cohort_summary
    try:
        df = loaders.load(dataset)
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    return json.dumps(cohort_summary(df), default=str)


def referee_hypothesis(hypothesis: str, dataset: str) -> str:
    """Run the deterministic referee on a hypothesis — the KILL/PROMOTE decision.

    This is the authoritative verdict tool. It runs the five-test gauntlet
    (age/sex, site/scanner leakage, brain-age, biomarker anchor, replication),
    scores robustness, applies the honesty caps, and routes the survivor to a
    biomarker-dominant MECHANISM. The verdict is deterministic — you MUST report
    it as returned and never override or inflate it.

    Args:
        hypothesis: The researcher's plain-language hypothesis.
        dataset: A registered dataset name (see ``list_datasets``).

    Returns a JSON object: verdict, robustness_score, promoted, substrate,
    mechanism, dominant_biomarker, per-test results, and the naive effect. If the
    hypothesis names an unsupported target it comes back promoted=false with an
    "unsupported" note — report that honestly rather than answering a different
    question.
    """
    from . import orchestrator
    try:
        xcard = orchestrator.investigate(hypothesis, dataset, api=False)
    except ValueError as exc:
        return json.dumps({"error": str(exc)})
    d = xcard.to_dict()
    t = d.get("translation", {}) or {}
    return json.dumps({
        "verdict": d.get("verdict"),
        "robustness_score": d.get("robustness_score"),
        "promoted": d.get("promoted"),
        "substrate": d.get("substrate"),
        "novelty_class": d.get("novelty_class"),
        "mechanism": t.get("mechanism"),
        "dominant_biomarker": t.get("dominant_biomarker"),
        "tests": d.get("robustness"),
        "naive_effect": d.get("naive_effect"),
        "caveats": d.get("caveats", []),
    }, default=str)


def prioritize_targets(mechanism: str) -> str:
    """Rank candidate genes for a mechanism via PI4AD (Priority Index for AD).

    Args:
        mechanism: One of "amyloid_cascade", "glial", "vascular" (the mechanism
            a promoted survivor routed to, from ``referee_hypothesis``).

    Returns a JSON object with the PI4AD-ranked gene list (gene, priority_score
    0-10, rank, source provenance). Live portal when reachable, else the bundled
    snapshot — the ``source`` field says which.
    """
    from . import translation
    ranked = translation._rank_targets(mechanism, prefer_offline=False)
    return json.dumps({"mechanism": mechanism, "ranked_targets": ranked}, default=str)


def protein_structure(gene: str) -> str:
    """Fetch a gene's predicted structure + confidence from AlphaFold.

    Args:
        gene: An AD target gene symbol (e.g. "APP", "MAPT", "TREM2").

    Returns a JSON object: uniprot, mean_plddt (0-100), model/cif URLs, and a
    ``source`` of "live" (EBI AlphaFold DB) or "offline_snapshot".
    """
    from ..integrations.alphafold import AlphaFoldClient
    try:
        s = AlphaFoldClient(prefer_offline=False).fetch_structure(gene)
        return json.dumps(s.to_dict(), default=str)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"gene_symbol": gene, "error": str(exc)})


def repurposing_candidates(gene: str) -> str:
    """Propose drug-repurposing candidates for a target gene (GNN/LLM layer).

    Args:
        gene: A prioritized target gene symbol.

    Returns a JSON object with ranked candidate compounds (compound,
    mechanism_note, evidence_strength as a curated 0-1 prior — NOT a clinical
    claim, source provenance). Curated snapshot by default; describe evidence as
    a hypothesis basis, never as proven benefit.
    """
    from ..integrations.gnn_llm import RepurposingEngine
    cands = RepurposingEngine().rank_compounds(gene, top_n=5)
    return json.dumps({"gene": gene, "candidates": [c.to_dict() for c in cands]},
                      default=str)


#: The orchestration tool surface. Plain functions (offline-safe); wrapped with
#: anthropic.beta_tool only inside the live path so importing this module never
#: needs the SDK.
_TOOL_FNS = [
    list_datasets,
    describe_cohort,
    referee_hypothesis,
    prioritize_targets,
    protein_structure,
    repurposing_candidates,
]

ORCHESTRATOR_SYSTEM = (
    "You are the ORCHESTRATOR of the NeuroAD Discovery Engine — an Alzheimer's "
    "imaging-to-molecule discovery instrument. You do NOT judge evidence yourself: "
    "every verdict, score, gene priority, protein structure, and compound comes "
    "from a TOOL, and you must report those values exactly as returned — never "
    "invent, inflate, or override them. Your job is to SEQUENCE the tools to serve "
    "the researcher's goal and reason over what they return.\n\n"
    "Typical flow: describe_cohort to check feasibility -> referee_hypothesis for "
    "the kill/promote verdict -> ONLY IF it is promoted, follow the returned "
    "mechanism to prioritize_targets -> protein_structure on the top gene -> "
    "repurposing_candidates on that gene -> finish with a falsifiable wet-lab "
    "experiment (named organoid model, readout, and an explicit kill criterion). "
    "If referee_hypothesis returns promoted=false, STOP the molecular chain and "
    "report the kill honestly — a killed imaging signal must never reach target "
    "discovery. If it returns an unsupported-target note, say so plainly. Keep the "
    "anti-overclaim contract: hedge, cite the tool that produced each number, and "
    "call batch-effect leakage prior art, not a discovery."
)


# ===========================================================================
# ENTRY POINT
# ===========================================================================

def _key_present() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())


def _orchestrator_model() -> str:
    """The model that drives the tool-runner — the same one the narration layer
    uses (claude-sonnet-5 by default; override via _client.PRIMARY_MODEL)."""
    try:
        from ..claude import _client
        return _client.PRIMARY_MODEL
    except Exception:  # noqa: BLE001
        return "claude-sonnet-5"


def orchestrate(goal: str, *, api: Optional[bool] = None,
                max_tokens: int = 4096) -> dict:
    """Drive an end-to-end investigation from a plain-language ``goal``.

    ``api`` forces the live tool-runner (True) or the scripted path (False);
    when None it uses the live path iff a key is present. Returns a JSON-safe
    dict: ``path`` ("live_tool_runner" | "scripted_offline"), ``final`` (the
    orchestrator's answer), ``tool_calls`` (ordered [{tool, input, result}]),
    and ``model``.
    """
    want_live = _key_present() if api is None else bool(api)
    if want_live:
        try:
            return _orchestrate_live(goal, max_tokens=max_tokens)
        except Exception as exc:  # noqa: BLE001
            _log.warning("live orchestration failed, scripting offline: %r", exc)
    return _orchestrate_scripted(goal)


def _orchestrate_live(goal: str, *, max_tokens: int) -> dict:
    """Anthropic tool-runner: Claude calls the tools until it is done."""
    import anthropic

    client = anthropic.Anthropic()
    model = _orchestrator_model()
    tools = [anthropic.beta_tool(fn) for fn in _TOOL_FNS]
    runner = client.beta.messages.tool_runner(
        model=model,
        max_tokens=max_tokens,
        system=ORCHESTRATOR_SYSTEM,
        tools=tools,
        messages=[{"role": "user", "content": goal}],
    )

    tool_calls: list[dict] = []
    final_text = ""
    for message in runner:
        for block in getattr(message, "content", []) or []:
            btype = getattr(block, "type", None)
            if btype == "tool_use":
                tool_calls.append({"tool": block.name, "input": dict(block.input)})
            elif btype == "text":
                final_text = block.text  # last text block wins = final answer
    # Pair each tool call with the result the runner fed back, best-effort.
    return {
        "path": "live_tool_runner",
        "model": model,
        "goal": goal,
        "final": final_text.strip(),
        "tool_calls": tool_calls,
    }


def _orchestrate_scripted(goal: str) -> dict:
    """Deterministic stand-in: run the same tools in the canonical order, no LLM.

    Uses the offline claim parser to turn ``goal`` into a (hypothesis, dataset)
    and then walks describe -> referee -> (if promoted) prioritize -> structure ->
    repurpose. Honestly labeled ``scripted_offline`` — no Claude drove this.
    """
    from ..data import loaders
    from ..claude import claim_parser

    dataset = _guess_dataset(goal)
    try:
        claim = claim_parser._fallback(goal, None)
        hypothesis = getattr(claim, "claim_text", goal) or goal
    except Exception:  # noqa: BLE001
        hypothesis = goal

    calls: list[dict] = []

    def _record(tool, fn, *args):
        out = fn(*args)
        calls.append({"tool": tool, "input": list(args), "result": json.loads(out)})
        return calls[-1]["result"]

    _record("describe_cohort", describe_cohort, dataset)
    ref = _record("referee_hypothesis", referee_hypothesis, hypothesis, dataset)

    promoted = bool(ref.get("promoted"))
    mechanism = ref.get("mechanism")
    if promoted and mechanism:
        ranked = _record("prioritize_targets", prioritize_targets, mechanism)
        scored = [r for r in ranked.get("ranked_targets", [])
                  if r.get("priority_score") is not None]
        if scored:
            top = scored[0]["gene"]
            _record("protein_structure", protein_structure, top)
            _record("repurposing_candidates", repurposing_candidates, top)

    verdict = ref.get("verdict")
    if promoted:
        final = (f"[scripted] Promoted survivor ({verdict}, "
                 f"score {ref.get('robustness_score')}) on {dataset}; routed to "
                 f"{mechanism}. Molecular follow-up sequenced through PI4AD -> "
                 f"AlphaFold -> repurposing above.")
    else:
        final = (f"[scripted] {verdict!s} on {dataset} — not promoted "
                 f"({'; '.join(ref.get('caveats', [])) or 'failed the gauntlet'}). "
                 "Molecular chain intentionally NOT run on a non-survivor.")
    return {
        "path": "scripted_offline",
        "model": "none (deterministic script)",
        "goal": goal,
        "hypothesis": hypothesis,
        "dataset": dataset,
        "final": final,
        "tool_calls": calls,
    }


def _guess_dataset(goal: str) -> str:
    """Pick a registered dataset from free text; default to the ADNI 3T survivor."""
    low = f" {goal.lower()} "
    if "kill" in low:
        return "synthetic:KILL"
    if "survivor" in low or "synthetic" in low:
        return "synthetic:SURVIVOR"
    for name in ("adni:combat", "adni:3t", "adni", "oasis:oasis2",
                 "oasis:oasis1", "oasis", "openbhb"):
        if name in low:
            return name
    return "adni:3t"
