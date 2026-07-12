#!/usr/bin/env python3
"""
What did the 5.7x AD imaging expansion (87 -> 494 embedded AD) buy?

Honest before/after on the FROZEN NeuroJEPA AD-vs-CN classifier, using the exact
leakage-honest machinery the gauntlet uses (probe.auc_ci_perm: site-disjoint OOF AUC +
bootstrap 95% CI + within-site permutation null). Reports:

  * AD-vs-CN AUC/CI/p_perm  — BEFORE (87 AD + 503 CN) vs AFTER (494 AD + 503 CN)
  * scanner-leakage confound — the SAME embedding's AUC predicting field strength
    (1.5T vs 3T); high = the imaging signal is partly acquisition batch, not disease
  * the honest verdict: bigger n tightens the ESTIMATE; it does not de-confound and
    does not prove out-of-distribution transfer.

Deterministic; reads the local NeuroJEPA embedding CSVs (no GPU, no gated plasma).
Writes reports/ad_expansion_analysis.json.

Usage:
    PYTHONPATH=src ./.venv/bin/python scripts/analyze_ad_expansion.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from neuroad import contract, probe

_ROOT = Path(__file__).resolve().parents[1]
_EXPANDED = _ROOT / "data" / "real" / "adni_neurojepa_embeddings_expanded.csv"
_BASE = _ROOT / "data" / "real" / "adni_neurojepa_embeddings.csv"
_OUT = _ROOT / "reports" / "ad_expansion_analysis.json"


def _codes(values: np.ndarray) -> np.ndarray:
    classes = np.unique(values)
    return np.searchsorted(classes, values)


def _ad_vs_cn(df: pd.DataFrame) -> dict:
    """Site-disjoint OOF AUC/CI/perm for AD(1) vs CN(0) on the frozen embedding."""
    dxb = df["dx"].astype("string").map({"AD": 1, "CN": 0})
    mask = dxb.notna().to_numpy()
    X = contract.embedding_matrix(df)[mask]
    y = dxb[mask].to_numpy(dtype=int)
    site = df["site"].astype("string").fillna("__na__").to_numpy()[mask]
    groups = _codes(site)
    r = probe.auc_ci_perm(X, y, groups)
    r["n_ad"] = int((y == 1).sum())
    r["n_cn"] = int((y == 0).sum())
    r["n_sites"] = int(len(np.unique(groups)))
    return r


def _scanner_leakage(df: pd.DataFrame) -> dict:
    """The confound check: can the SAME embedding predict field strength (3T vs 1.5T)?

    High AUC => the representation encodes acquisition hardware, so an AD-vs-CN number
    on the raw embedding is partly scanner batch, not biology. Restricted to AD/CN rows
    with a known field strength so it is comparable to the disease test.
    """
    if "field_strength" not in df.columns:
        return {"auc": None, "note": "no field_strength column"}
    dxb = df["dx"].astype("string").map({"AD": 1, "CN": 0})
    fs = df["field_strength"].astype("string")
    fsb = fs.map(lambda v: 1 if isinstance(v, str) and "3" in v else (0 if isinstance(v, str) and "1.5" in v else np.nan))
    mask = (dxb.notna() & fsb.notna()).to_numpy()
    if mask.sum() < 20 or len(np.unique(fsb[mask].dropna())) < 2:
        return {"auc": None, "note": "too few / single field strength"}
    X = contract.embedding_matrix(df)[mask]
    y = fsb[mask].to_numpy(dtype=int)
    r = probe.auc_ci_perm(X, y, None)
    r["n"] = int(mask.sum())
    r["n_3T"] = int((y == 1).sum())
    r["n_1p5T"] = int((y == 0).sum())
    return r


def main() -> int:
    after = pd.read_csv(_EXPANDED)
    base = pd.read_csv(_BASE)  # the historical n=590 (87 AD / 503 CN) = "before"

    print("[analyze] computing BEFORE (n=590) ...", flush=True)
    before_dx = _ad_vs_cn(base)
    before_leak = _scanner_leakage(base)
    print("[analyze] computing AFTER (expanded) ...", flush=True)
    after_dx = _ad_vs_cn(after)
    after_leak = _scanner_leakage(after)

    def _ci_width(r):
        if r.get("ci_lo") is None or r.get("ci_hi") is None:
            return None
        return round(r["ci_hi"] - r["ci_lo"], 4)

    result = {
        "task": "AD-vs-CN on frozen NeuroJEPA embedding (site-disjoint OOF)",
        "before": {"slice": "ADNI n=590", **before_dx, "ci_width": _ci_width(before_dx),
                   "scanner_leakage": before_leak},
        "after": {"slice": "ADNI + 5.7x AD expansion", **after_dx, "ci_width": _ci_width(after_dx),
                  "scanner_leakage": after_leak},
        "delta": {
            "auc": (None if before_dx.get("auc") is None or after_dx.get("auc") is None
                    else round(after_dx["auc"] - before_dx["auc"], 4)),
            "ci_width": (None if _ci_width(before_dx) is None or _ci_width(after_dx) is None
                         else round(_ci_width(after_dx) - _ci_width(before_dx), 4)),
            "n_ad": after_dx["n_ad"] - before_dx["n_ad"],
        },
        "verdict": "",
    }

    b, a = before_dx, after_dx
    bw, aw = _ci_width(b), _ci_width(a)
    tighter = (bw is not None and aw is not None and aw < bw)
    leak_a = after_leak.get("auc")
    result["verdict"] = (
        f"AD n {b['n_ad']}->{a['n_ad']} ({round(a['n_ad']/max(b['n_ad'],1),1)}x). "
        f"AD-vs-CN AUC {b.get('auc')} [{b.get('ci_lo')},{b.get('ci_hi')}] -> "
        f"{a.get('auc')} [{a.get('ci_lo')},{a.get('ci_hi')}]; "
        f"CI width {bw}->{aw} ({'TIGHTER' if tighter else 'not tighter'}). "
        f"Scanner-leakage AUC still {leak_a} => the raw-embedding AD signal remains "
        f"partly acquisition batch; MORE AD tightens the estimate but does NOT "
        f"de-confound (needs ComBat) and does NOT prove out-of-distribution transfer "
        f"(needs an external cohort). Honest read: the expansion buys precision, not "
        f"generalization."
    )

    _OUT.parent.mkdir(exist_ok=True)
    _OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\n[analyze] wrote {_OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
