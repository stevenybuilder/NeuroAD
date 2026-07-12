#!/usr/bin/env python
"""Regenerate the bundled STRING v12.0 PPI subgraph for L5 network propagation.

Fetches ONCE from the free, no-credential STRING v12.0 API over the NeuroAD gene
universe (74 PI4AD snapshot genes UNION 50 Open Targets AD targets UNION
translation.MECHANISM_GENES = 115 symbols) and writes the provenance-headed CSV
consumed by ``neuroad.integrations.pi4ad.fetch_string_subgraph``.

STRING is CC BY 4.0, so bundling this subset is license-clean. Run from the repo
root:  PYTHONPATH=src python scripts/build_string_subgraph.py
This is the ONLY code path that touches the network; the engine itself never does
(offline-first). Re-running reproduces the bundled snapshot byte-for-byte.
"""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "src" / "neuroad" / "integrations" / "data"
OT_SNAPSHOT = DATA / "opentargets_snapshot.json"
PI4AD_SNAPSHOT = DATA / "pi4ad_priority_snapshot.csv"
OUT = DATA / "string_ppi_subgraph.csv"

STRING_API = "https://string-db.org/api/tsv/network"
# translation.MECHANISM_GENES flattened (kept in sync deliberately).
MECHANISM_GENES = ["APP", "MAPT", "PSEN1", "BACE1", "APOE", "ESR1",
                   "TREM2", "CLU", "MAPK1", "HRAS", "BIN1"]


def gene_universe() -> list[str]:
    ot = json.loads(OT_SNAPSHOT.read_text())
    genes = {t["gene"] for t in ot["associated_targets"]}
    genes |= set(ot["engine_genes"].keys())
    with open(PI4AD_SNAPSHOT, newline="") as fh:
        genes |= {r["gene"] for r in csv.DictReader(fh)}
    genes |= set(MECHANISM_GENES)
    return sorted(g for g in genes if g)


def main() -> None:
    universe = gene_universe()
    resp = requests.post(
        STRING_API,
        data={
            "identifiers": "\r".join(universe),
            "species": 9606,
            "caller_identity": "neuroad_discovery_engine",
        },
        timeout=60,
    )
    resp.raise_for_status()
    rows = list(csv.DictReader(io.StringIO(resp.text), delimiter="\t"))

    dedup: dict[tuple[str, str], int] = {}
    for r in rows:
        a, b = r["preferredName_A"], r["preferredName_B"]
        if not a or not b or a == b:
            continue
        score = int(round(float(r["score"]) * 1000))
        key = (a, b) if a <= b else (b, a)
        if key not in dedup or score > dedup[key]:
            dedup[key] = score

    with open(OUT, "w", newline="") as fh:
        fh.write("# STRING v12.0 protein-protein interaction subgraph over the "
                 "NeuroAD gene universe\n")
        fh.write("# (74 PI4AD snapshot genes UNION 50 Open Targets AD targets "
                 "UNION MECHANISM_GENES = 115 symbols).\n")
        fh.write("# Source: STRING v12.0 https://string-db.org/api/tsv/network "
                 "(Homo sapiens, taxon 9606),\n")
        fh.write("# fetched 2026-07-11, no credentials. Licensed CC BY 4.0 "
                 "(bundling a subset is license-clean).\n")
        fh.write("# combined_score is STRING's integrated confidence on the "
                 "0-1000 scale (network default cutoff 400).\n")
        fh.write("# This is the offline-first, deterministic PPI network for "
                 "in-repo RWR/heat-diffusion propagation.\n")
        w = csv.writer(fh)
        w.writerow(["gene_a", "gene_b", "combined_score", "string_version"])
        for (a, b), s in sorted(dedup.items()):
            w.writerow([a, b, s, "v12.0"])
    print(f"wrote {len(dedup)} edges over {len(universe)} genes -> {OUT}")


if __name__ == "__main__":
    main()
