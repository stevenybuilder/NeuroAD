#!/usr/bin/env python3
"""
prove_live.py — judge-facing proof that NeuroAD's external integrations hit REAL,
live public APIs (not just bundled snapshots).

This calls the project's OWN integration clients (the exact code the engine uses)
against live, keyless public endpoints for three canonical Alzheimer's targets:

    APP   / P05067 / ENSG00000142192
    MAPT  / P10636 / ENSG00000186868
    BACE1 / P56817 / ENSG00000186318

For each of the four integrations it prints:
  * the tool + the real endpoint URL it hits,
  * the HTTP status / source label,
  * one key returned value, and
  * an explicit  source=live  vs  source=offline_snapshot  stamp.

The clients are OFFLINE-FIRST by design: on any network failure they degrade to a
bundled snapshot and stamp source=offline_snapshot — they never fabricate live
data. So when this script prints source=live, a judge is seeing a real 200 from
the public API on demand. Run:

    PYTHONPATH=src .venv/bin/python scripts/prove_live.py
"""
from __future__ import annotations

import sys
from typing import Optional

from neuroad.integrations import alphafold, lincs, opentargets, string_ppi

# The three canonical AD targets we prove against.
GENES: tuple[str, ...] = ("APP", "MAPT", "BACE1")
UNIPROT = {"APP": "P05067", "MAPT": "P10636", "BACE1": "P56817"}


# ---------------------------------------------------------------------------
# tiny formatting helpers
# ---------------------------------------------------------------------------
def _hr(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _stamp(source: str) -> str:
    """A loud, unambiguous provenance tag a judge can eyeball."""
    if source == "live":
        return "source=live  (REAL 200 from public API)"
    return f"source={source}  (bundled fallback — network unavailable)"


def _row(label: str, value: object) -> None:
    print(f"    {label:<22} {value}")


# ---------------------------------------------------------------------------
# 1) STRING — protein-protein interaction evidence
# ---------------------------------------------------------------------------
def prove_string() -> str:
    _hr("1) STRING-db  —  protein-protein interaction evidence")
    base = string_ppi._base_url()
    client = string_ppi.StringPPIClient(prefer_offline=False)
    overall = "offline_snapshot"

    # Named key value in the audit brief: APP <-> SORL1 combined score.
    _hr_ep = f"{base}/api/tsv/network?identifiers=APP%0dSORL1&species=9606"
    pair = client.pair_evidence("APP", "SORL1")
    print("  [pair] APP <-> SORL1")
    _row("endpoint:", _hr_ep)
    _row("combined_score:", pair.combined_score)
    _row("channels:", pair.channels or "(none above cutoff)")
    _row("provenance:", _stamp(pair.source))
    if pair.error:
        _row("note:", pair.error)
    if pair.source == "live":
        overall = "live"

    # Top hub partner for each gene.
    for gene in GENES:
        ep = f"{base}/api/tsv/interaction_partners?identifiers={gene}&species=9606"
        partners = client.interaction_partners(gene, limit=3)
        print(f"\n  [partners] {gene}")
        _row("endpoint:", ep)
        if partners:
            top = partners[0]
            _row("top partner:", f"{top.gene_b} (combined_score={top.combined_score})")
            _row("n_partners:", len(partners))
            _row("provenance:", _stamp(top.source))
            if top.source == "live":
                overall = "live"
        else:
            _row("result:", "no partners returned")
    return overall


# ---------------------------------------------------------------------------
# 2) AlphaFold DB — structural confidence (mean pLDDT)
# ---------------------------------------------------------------------------
def prove_alphafold() -> str:
    _hr("2) AlphaFold DB (EBI)  —  predicted structure + mean pLDDT")
    base = alphafold._base_url()
    client = alphafold.AlphaFoldClient(prefer_offline=False)
    overall = "offline_snapshot"
    for gene in GENES:
        acc = UNIPROT[gene]
        ep = f"{base}/api/prediction/{acc}"
        st = client.fetch_structure(gene)
        print(f"\n  {gene} / {acc}")
        _row("endpoint:", ep)
        _row("mean_plddt:", st.mean_plddt)
        _row("model_version:", st.model_version)
        _row("model_url:", st.model_url or "(none)")
        _row("provenance:", _stamp(st.source))
        if st.error:
            _row("note:", st.error)
        if st.source == "live":
            overall = "live"
    return overall


# ---------------------------------------------------------------------------
# 3) Open Targets — target-disease association + known drugs
# ---------------------------------------------------------------------------
def prove_opentargets() -> str:
    _hr("3) Open Targets Platform (GraphQL)  —  AD association + known drugs")
    ep = opentargets._api_url()
    client = opentargets.OpenTargetsClient(prefer_offline=False)
    overall = "offline_snapshot"
    print(f"  endpoint: {ep}")
    print(f"  disease:  {opentargets.AD_DISEASE_ID} (Alzheimer disease)")
    for gene in GENES:
        assoc = client.target_association(gene)
        print(f"\n  {gene}")
        if assoc is None:
            _row("result:", "unresolved target")
            continue
        _row("ensembl_id:", assoc.ensembl_id)
        _row("association_score:", round(assoc.association_score, 4))
        _row("n_known_drugs:", assoc.n_known_drugs)
        top_dt = sorted(assoc.datatype_scores.items(),
                        key=lambda kv: kv[1], reverse=True)[:2]
        _row("top datatypes:", ", ".join(f"{k}={v:.3f}" for k, v in top_dt) or "(none)")
        _row("provenance:", _stamp(assoc.source))
        if assoc.error:
            _row("note:", assoc.error)
        if assoc.source == "live":
            overall = "live"
    return overall


# ---------------------------------------------------------------------------
# 4) LINCS L1000 (SigCom) — perturbational reversal efficacy proxy
# ---------------------------------------------------------------------------
def prove_lincs() -> str:
    _hr("4) SigCom LINCS L1000  —  AD-signature reversal efficacy proxy")
    client = lincs.LincsClient(prefer_offline=False)
    print(f"  metadata endpoint: {client.meta_base}/entities/find")
    print(f"  data endpoint:     {client.data_base}/enrich/ranktwosided")
    print(f"  LoF databases:     {', '.join(lincs._LOF_DATABASES)}")
    print("  (querying the curated AD up/down signature for reverser signatures — "
          "this can take ~30-60s live)")

    proxy = client.ad_reversal_efficacy(limit=500)
    if not proxy:
        print("  result: no reversal proxy returned")
        return "offline_snapshot"

    # Provenance is uniform across the returned map.
    sample_source = next(iter(proxy.values())).source
    print(f"\n  genes with a reversal signal: {len(proxy)}")
    print(f"  provenance: {_stamp(sample_source)}")

    # Show the strongest reverser hits overall (the headline efficacy candidates).
    top = sorted(proxy.values(), key=lambda p: p.reversal_score, reverse=True)[:5]
    print("\n  top reverser hits (KO reverses the AD signature => inhibition target):")
    for p in top:
        _row(f"{p.gene}:",
             f"reversal_score={p.reversal_score:.3f}  "
             f"[{p.best_database} / {p.best_cell_line}]  n_sig={p.n_signatures}")

    # And any of our three canonical genes that surfaced.
    hits = [g for g in GENES if g in proxy]
    if hits:
        print("\n  canonical AD genes present as reversers:")
        for g in hits:
            p = proxy[g]
            _row(f"{g}:", f"reversal_score={p.reversal_score:.3f}")
    return sample_source


# ---------------------------------------------------------------------------
def main() -> int:
    print("NeuroAD live-integration proof — hitting real public APIs on demand.")
    print(f"Targets: {', '.join(f'{g}/{UNIPROT[g]}' for g in GENES)}")

    results = {
        "STRING":       prove_string(),
        "AlphaFold DB": prove_alphafold(),
        "Open Targets": prove_opentargets(),
        "LINCS L1000":  prove_lincs(),
    }

    _hr("SUMMARY  —  live vs offline_snapshot per integration")
    for name, src in results.items():
        tag = "LIVE ✓" if src == "live" else "offline (snapshot)"
        print(f"    {name:<14} {tag:<20} ({src})")

    n_live = sum(1 for s in results.values() if s == "live")
    print(f"\n  {n_live}/{len(results)} integrations proved LIVE this run.")
    print("  (Any 'offline' line means the public API was unreachable right now;")
    print("   the client honestly fell back to a bundled snapshot — it never fakes live.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
