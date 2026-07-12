#!/usr/bin/env python3
"""
Reproducible generator for the OASIS-1 Neuro-JEPA AD-signal report.

Replaces the ad-hoc, hand-written `reports/oasis_neurojepa_ad.json` with a
deterministic recomputation straight from the (git-ignored) local embedding
table. Run it after `scripts/neurojepa_embed_colab.py` grows the table from 61
toward ~234 subjects, so the headline AD-vs-CN number is regenerated at the new
n with a bootstrap 95% CI and a permutation-null p — not a frozen literal.

Metric: the referee's own probe (`neuroad.probe.auc_ci_perm`) on the raw 768-d
frozen embeddings. With D=768 >> n the probe's automatic PCA-10 whitening
front-end engages INSIDE each CV fold, so the "pca10" AUC is leakage-free
(PCA fit on training rows only) rather than the in-sample number the old JSON
carried.

Contrasts (dx derived from CDR, the OASIS-1 convention):
  * clean clinical AD  : CDR>=1 (AD)  vs CDR=0 (CN)
  * any impairment     : CDR>=0.5 (AD+MCI) vs CDR=0 (CN)

Compliance: reads only the derived embedding vectors (never the gated weights);
writes only AUC-grade numbers to reports/ (repo-safe). The embedding table stays
git-ignored.

Usage:
    PYTHONPATH=src ./.venv/bin/python scripts/run_oasis_neurojepa.py
    PYTHONPATH=src ./.venv/bin/python scripts/run_oasis_neurojepa.py --emb data/real/oasis1_neurojepa_embeddings.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from neuroad import probe

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EMB = _ROOT / "data" / "real" / "oasis1_neurojepa_embeddings.csv"
DEFAULT_OUT = _ROOT / "reports" / "oasis_neurojepa_ad.json"


def _dx_from_cdr(cdr: float) -> str:
    if pd.isna(cdr):
        return "NA"
    if cdr == 0:
        return "CN"
    if cdr >= 1:
        return "AD"
    return "MCI"  # CDR 0.5 = very mild / questionable


def _contrast(df: pd.DataFrame, emb_cols: list[str], positive_mask, n_boot: int, n_perm: int) -> dict:
    """AUC + CI + perm-p for (positive vs CN), raw 768-d -> internal auto-PCA-10 CV."""
    cn = df["dx"].eq("CN")
    sub = df[cn | positive_mask]
    y = positive_mask[cn | positive_mask].astype(int).to_numpy()
    X = sub[emb_cols].to_numpy(dtype=float)
    res = probe.auc_ci_perm(X, y, groups=None, n_boot=n_boot, n_perm=n_perm)
    return {
        "auc": res["auc"],
        "ci": None if res["ci_lo"] is None else [res["ci_lo"], res["ci_hi"]],
        "p_perm": res["p_perm"],
        "ci_excludes_chance": res["ci_excludes_chance"],
        "n": int(len(y)),
        "n_positive": int(y.sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default=str(DEFAULT_EMB))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-perm", type=int, default=1000)
    args = ap.parse_args()

    emb_path = Path(args.emb)
    if not emb_path.exists():
        raise SystemExit(
            f"{emb_path} not found — embedding table is git-ignored by design.\n"
            "Regenerate it on a GPU runtime with your own HF_TOKEN "
            "(scripts/neurojepa_embed_colab.py; see docs/COLAB_RUNBOOK.md).")

    raw = pd.read_csv(emb_path)
    emb_cols = [c for c in raw.columns if c.startswith("emb_")]
    if not emb_cols or "cdr" not in raw.columns:
        raise SystemExit(f"{emb_path} missing emb_* columns or a 'cdr' column.")

    df = raw.copy()
    df["dx"] = df["cdr"].astype(float).map(_dx_from_cdr)
    counts = df["dx"].value_counts().to_dict()

    ad_pos = df["dx"].eq("AD")                      # CDR>=1
    imp_pos = df["dx"].isin(["AD", "MCI"])          # CDR>=0.5

    ad = _contrast(df, emb_cols, ad_pos, args.n_boot, args.n_perm)
    imp = _contrast(df, emb_cols, imp_pos, args.n_boot, args.n_perm)

    report = {
        "dataset": "oasis_neurojepa",
        "n_subjects": int(len(df)),
        "embedding_dim": len(emb_cols),
        "dx": {k: int(counts.get(k, 0)) for k in ("CN", "MCI", "AD")},
        # Back-compat tuple [auc, p_perm, n, n_positive] — build_demo_data reads [0].
        "ad_vs_cn_clean_CDRge1_pca10": [ad["auc"], ad["p_perm"], ad["n"], ad["n_positive"]],
        "impaired_vs_cn_CDRge0p5_pca10": [imp["auc"], imp["p_perm"], imp["n"], imp["n_positive"]],
        # Richer, structured stats (CI band + permutation null).
        "ad_vs_cn_clean": ad,
        "impaired_vs_cn": imp,
        "method": ("neuroad.probe.auc_ci_perm on raw 768-d frozen Neuro-JEPA embeddings; "
                   "automatic PCA-10 whitening fit inside each CV fold (leakage-free); "
                   f"bootstrap {args.n_boot} / permutation {args.n_perm}."),
        "interpretation": (
            f"Frozen Neuro-JEPA embeddings separate clinical AD (CDR>=1) from CN at "
            f"AUC {ad['auc']} (n={ad['n']}, {ad['n_positive']} AD); any-impairment "
            f"(CDR>=0.5) vs CN at AUC {imp['auc']}."),
        "caveats": ("Single-cohort / single-scanner (OASIS-1, 1.5T Siemens): scanner-leakage "
                    "and cross-cohort replication tests are NA here. CDR>=1 n is modest -> "
                    "read the CI band, not the point estimate."),
        "compliance": "Frozen inference only; weights never stored; embedding table git-ignored.",
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"[oasis-nj] n={len(df)} dx={report['dx']}")
    print(f"[oasis-nj] AD vs CN (CDR>=1):   AUC {ad['auc']}  CI {ad['ci']}  p_perm {ad['p_perm']}  (n={ad['n']}, {ad['n_positive']} AD)")
    print(f"[oasis-nj] impaired vs CN (>=0.5): AUC {imp['auc']}  CI {imp['ci']}  p_perm {imp['p_perm']}  (n={imp['n']})")
    print(f"[oasis-nj] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
