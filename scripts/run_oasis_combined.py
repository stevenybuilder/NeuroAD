#!/usr/bin/env python3
"""
Combined OASIS-1 + OASIS-2 Neuro-JEPA analysis: replication + honest pooling.

Two cohorts, same frozen encoder, same 1.5T Siemens acquisition, but different
subjects and different preprocessing (OASIS-1 ships brain-masked T88 volumes;
OASIS-2 ships raw mpr-1, optionally skull-stripped here). This script reports:

  1. PER-COHORT AD-vs-CN and impaired-vs-CN AUCs — the honest replication test:
     does the OASIS-1 finding independently reproduce in OASIS-2?
  2. A COHORT-LEAKAGE check: can the frozen embedding predict which cohort a
     subject is from? A high AUC means the preprocessing/site difference is a
     batch confound (the same failure mode the referee gates against on scanner),
     so a POOLED disease number would be partly reading cohort, not disease.
  3. A POOLED AD-vs-CN AUC — reported ONLY with the cohort-leakage caveat, and
     with site-disjoint CV (train on one cohort, test on the other) as the
     strict cross-cohort generalization estimate.

Metric: neuroad.probe.auc_ci_perm (leakage-free CV, auto PCA-10) on raw 768-d.
Compliance: reads only derived embeddings; writes only numbers to reports/.

Usage:
    PYTHONPATH=src ./.venv/bin/python scripts/run_oasis_combined.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from neuroad import probe

_ROOT = Path(__file__).resolve().parents[1]
OASIS1 = _ROOT / "data" / "real" / "oasis1_neurojepa_embeddings.csv"
OASIS2 = _ROOT / "data" / "real" / "oasis2_neurojepa_embeddings.csv"
OUT = _ROOT / "reports" / "oasis_neurojepa_combined.json"


def _dx(cdr: float) -> str:
    cdr = float(cdr)
    return "CN" if cdr == 0 else ("AD" if cdr >= 1 else "MCI")


def _load(path: Path, cohort: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["dx"] = df["cdr"].map(_dx)
    df["cohort"] = cohort
    return df


def _contrast(df: pd.DataFrame, emb: list[str], pos_mask, groups=None, n_boot=1000, n_perm=1000) -> dict:
    cn = df["dx"].eq("CN")
    keep = cn | pos_mask
    sub = df[keep]
    y = pos_mask[keep].astype(int).to_numpy()
    X = sub[emb].to_numpy(float)
    g = None if groups is None else groups[keep.to_numpy()]
    res = probe.auc_ci_perm(X, y, groups=g, n_boot=n_boot, n_perm=n_perm)
    return {"auc": res["auc"], "ci": None if res["ci_lo"] is None else [res["ci_lo"], res["ci_hi"]],
            "p_perm": res["p_perm"], "n": int(len(y)), "n_positive": int(y.sum())}


def _report_block(df, emb, **kw) -> dict:
    return {
        "n": int(len(df)),
        "dx": {k: int((df["dx"] == k).sum()) for k in ("CN", "MCI", "AD")},
        "ad_vs_cn": _contrast(df, emb, df["dx"].eq("AD"), **kw),
        "impaired_vs_cn": _contrast(df, emb, df["dx"].isin(["AD", "MCI"]), **kw),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oasis1", default=str(OASIS1))
    ap.add_argument("--oasis2", default=str(OASIS2))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-perm", type=int, default=1000)
    args = ap.parse_args()

    o1 = _load(Path(args.oasis1), "OASIS-1")
    if not Path(args.oasis2).exists():
        raise SystemExit(f"{args.oasis2} not found — run the OASIS-2 embed first.")
    o2 = _load(Path(args.oasis2), "OASIS-2")

    emb = [c for c in o1.columns if c.startswith("emb_")]
    o2 = o2[[c for c in o2.columns if c in o1.columns]]  # align
    pooled = pd.concat([o1, o2], ignore_index=True)
    boot, perm = args.n_boot, args.n_perm

    # (1) per-cohort replication
    rep = {
        "OASIS-1": _report_block(o1, emb, n_boot=boot, n_perm=perm),
        "OASIS-2": _report_block(o2, emb, n_boot=boot, n_perm=perm),
    }

    # (2) cohort-leakage: can the embedding tell the two cohorts apart?
    yb = (pooled["cohort"] == "OASIS-2").astype(int).to_numpy()
    leak = probe.auc_ci_perm(pooled[emb].to_numpy(float), yb, groups=None, n_boot=boot, n_perm=perm)
    cohort_leakage = {"auc": leak["auc"], "ci": None if leak["ci_lo"] is None else [leak["ci_lo"], leak["ci_hi"]],
                      "p_perm": leak["p_perm"],
                      "interpretation": ("HIGH -> the two cohorts are distinguishable from the embedding "
                                         "(preprocessing/site batch effect); a pooled disease AUC is partly "
                                         "reading cohort. LOW (~0.5) -> safe to pool.")}

    # (3a) naive pooled AD-vs-CN, (3b) strict cross-cohort (site-disjoint CV: train one cohort, test the other)
    pooled_naive = _report_block(pooled, emb, n_boot=boot, n_perm=perm)
    site_codes = pooled["cohort"].astype("category").cat.codes.to_numpy()
    pooled_xcohort = {
        "ad_vs_cn": _contrast(pooled, emb, pooled["dx"].eq("AD"), groups=site_codes, n_boot=boot, n_perm=perm),
        "impaired_vs_cn": _contrast(pooled, emb, pooled["dx"].isin(["AD", "MCI"]), groups=site_codes,
                                    n_boot=boot, n_perm=perm),
        "note": "site-disjoint CV: probe trained on one cohort, scored on the other — the strict "
                "generalization estimate. Meaningful only if cohort_leakage is low.",
    }

    report = {
        "cohorts": rep,
        "cohort_leakage": cohort_leakage,
        "pooled_naive": pooled_naive,
        "pooled_cross_cohort": pooled_xcohort,
        "n_total": int(len(pooled)),
        "method": ("neuroad.probe.auc_ci_perm on raw 768-d frozen Neuro-JEPA embeddings; auto PCA-10 "
                   f"inside CV; bootstrap {boot} / permutation {perm}."),
        "caveat": ("OASIS-1 = brain-masked T88; OASIS-2 = raw mpr-1 (+/- skull-strip). Read per-cohort "
                   "replication as the primary result; pool only if cohort_leakage AUC ~0.5."),
        "compliance": "Frozen inference only; weights never stored; embedding tables git-ignored.",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")

    print(f"[combined] n_total={len(pooled)}  (OASIS-1={len(o1)}, OASIS-2={len(o2)})")
    for c, b in rep.items():
        print(f"[{c}] AD-vs-CN AUC {b['ad_vs_cn']['auc']} CI {b['ad_vs_cn']['ci']} "
              f"(n={b['ad_vs_cn']['n']}, {b['ad_vs_cn']['n_positive']} AD) | "
              f"impaired AUC {b['impaired_vs_cn']['auc']} CI {b['impaired_vs_cn']['ci']}")
    print(f"[cohort-leakage] AUC {cohort_leakage['auc']} CI {cohort_leakage['ci']}  "
          f"({'CONFOUND — do not pool naively' if cohort_leakage['auc'] and cohort_leakage['auc']>0.7 else 'low — pooling defensible'})")
    print(f"[pooled naive]  AD-vs-CN AUC {pooled_naive['ad_vs_cn']['auc']} CI {pooled_naive['ad_vs_cn']['ci']} "
          f"(n={pooled_naive['ad_vs_cn']['n']}, {pooled_naive['ad_vs_cn']['n_positive']} AD)")
    print(f"[cross-cohort]  AD-vs-CN AUC {pooled_xcohort['ad_vs_cn']['auc']} CI {pooled_xcohort['ad_vs_cn']['ci']}")
    print(f"[combined] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
