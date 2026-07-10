"""Regression tests for the translation loop + the two honesty fixes.

Covers, all offline / deterministic (no network, no creds):
  * item 1 — honest per-feeder substrate labels (no feeder mislabeled Neuro-JEPA);
  * item 2 — out-of-scope hypotheses are REFUSED, not silently coerced to
             conversion, while in-scope contrasts still run;
  * item 3 — a promoted survivor carries a provenance-labeled translation lead
             (PI4AD -> AlphaFold -> repurposing), and the chain is offline-safe.
"""
from __future__ import annotations

from neuroad.data.loaders import honest_substrate
from neuroad.harness import orchestrator, translation


# --------------------------------------------------------------------------
# Item 1 — honest substrate labels
# --------------------------------------------------------------------------

def test_honest_substrate_never_mislabels_feeders():
    assert "FreeSurfer" in honest_substrate("adni")
    assert "FreeSurfer" in honest_substrate("adni:3t")
    assert "FreeSurfer" in honest_substrate("adni:combat")
    assert "synthetic" in honest_substrate("synthetic:SURVIVOR").lower()
    assert "OASIS" in honest_substrate("oasis")
    assert "OpenBHB" in honest_substrate("openbhb")
    # Only the actual Neuro-JEPA feeders may claim the foundation model.
    assert "Neuro-JEPA" in honest_substrate("oasis:neurojepa")
    assert "Neuro-JEPA" in honest_substrate("openbhb:neurojepa")
    # ADNI / synthetic / tabular feeders must NOT claim Neuro-JEPA.
    for ds in ("adni", "synthetic:SURVIVOR", "oasis", "openbhb"):
        assert "Neuro-JEPA" not in honest_substrate(ds)


def test_investigate_stamps_honest_substrate():
    x = orchestrator.investigate(
        "Does the embedding predict MCI->AD conversion?", "synthetic:SURVIVOR")
    assert "Neuro-JEPA" not in x.to_dict()["substrate"]
    assert "synthetic" in x.to_dict()["substrate"].lower()


# --------------------------------------------------------------------------
# Item 2 — out-of-scope refusal (no silent coercion to `conversion`)
# --------------------------------------------------------------------------

def test_out_of_scope_targets_are_refused():
    for h in ("What is the tau-PET SUVR trajectory?",
              "APOE4 genotype vs CSF abeta42",
              "amyloid-positive vs amyloid-negative status"):
        x = orchestrator.investigate(h, "synthetic:SURVIVOR")
        d = x.to_dict()
        assert d["promoted"] is False
        assert d["novelty_class"] == "unsupported"
        assert any("unsupported" in c.lower() for c in d.get("caveats", []))


def test_in_scope_contrasts_still_run():
    d = orchestrator.investigate(
        "Does the embedding predict AD vs CN diagnosis?", "synthetic:SURVIVOR"
    ).to_dict()
    assert d["novelty_class"] != "unsupported"
    assert d["claim_id"] != "claim-unsupported"


def test_out_of_scope_reason_helper_is_conservative():
    # Clear out-of-scope target -> a reason.
    assert orchestrator._out_of_scope_reason("tau-PET SUVR trajectory")
    # In-scope language present -> no refusal even if a stray word matches.
    assert not orchestrator._out_of_scope_reason(
        "Does the embedding predict MCI to AD conversion?")
    assert not orchestrator._out_of_scope_reason("AD vs CN diagnosis")


# --------------------------------------------------------------------------
# Item 3 — translation loop (promoted survivor -> molecule/wet-lab lead)
# --------------------------------------------------------------------------

def test_translate_is_offline_and_provenance_labeled():
    lead = translation.translate("amyloid_cascade", prefer_offline=True)
    assert lead["status"] == "translated"
    assert lead["top_target"]  # a gene was resolved
    # PI4AD ranked the candidates; the top one has a numeric priority.
    top = lead["ranked_targets"][0]
    assert top["gene"] == lead["top_target"]
    assert top["priority_score"] is not None
    # Structure + repurposing legs are present and OFFLINE-labeled (never "live"
    # on the default offline path — the anti-overclaim contract).
    assert lead["structure"].get("source") in ("offline_snapshot",)
    assert lead["provenance"]["alphafold"] == "offline_snapshot"
    assert lead["provenance"]["pi4ad"] in ("offline_snapshot", "candidate_only")
    assert lead["wet_lab_experiment"]  # a falsifiable readout was proposed


def test_translate_never_raises_on_unknown_mechanism():
    lead = translation.translate("not_a_real_mechanism", prefer_offline=True)
    # Unknown mechanism falls back to amyloid_cascade rather than crashing.
    assert lead["mechanism"] == "amyloid_cascade"


def test_promoted_survivor_gets_a_translation_lead():
    x = orchestrator.investigate(
        "Does the embedding predict MCI->AD conversion?", "synthetic:SURVIVOR")
    d = x.to_dict()
    if d["promoted"]:
        t = d.get("translation", {})
        assert t.get("status") == "translated"
        assert t.get("top_target")
        assert t.get("structure", {}).get("gene_symbol") == t.get("top_target")


def test_mechanism_gene_map_only_names_known_ad_targets():
    from neuroad.integrations.alphafold import _load_snapshot
    snap = _load_snapshot()
    known = {v["gene_symbol"] for v in snap.values()
             if isinstance(v, dict) and v.get("gene_symbol")}
    assert known, "AlphaFold snapshot exposed no gene symbols"
    # Every mechanism gene must be a recognized AD target in the AlphaFold
    # snapshot — no invented symbols leaking into the translation layer.
    for genes in translation.MECHANISM_GENES.values():
        for g in genes:
            assert g in known, f"{g} not in AlphaFold AD target map"
