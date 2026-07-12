#!/usr/bin/env python3
"""
Harmonized OASIS-1 + OASIS-2 analysis: remove the cohort batch effect with ComBat,
then re-measure cohort-leakage and the AD signal.

The combined analysis showed cohort-leakage AUC ~1.0 (the two cohorts are perfectly
separable from the raw embeddings — a residual batch effect from intensity/skull-strip/
registration differences that survives skull-stripping). ComBat is the neuroimaging-
standard fix: it removes per-feature location/scale batch effects while PRESERVING the
biological covariates (age, sex), and is deliberately LABEL-BLIND (dx is excluded), so
measuring AD-vs-CN afterward is not circular.

Pass criteria for honest pooling:
  * cohort-leakage AUC drops toward ~0.5 (batch removed), AND
  * AD-vs-CN AUC survives (disease signal was separable from batch, not entangled).

Usage:
    PYTHONPATH=src ./.venv/bin/python scripts/run_oasis_harmonized.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from neuroad import contract, probe
from neuroad.data import harmonize

_ROOT = Path(__file__).resolve().parents[1]
OASIS1 = _ROOT / "data" / "real" / "oasis1_neurojepa_embeddings.csv"
OASIS2 = _ROOT / "data" / "real" / "oasis2_neurojepa_embeddings.csv"
OUT = _ROOT / "reports" / "oasis_neurojepa_harmonized.json"


def _dx(c: float) -> str:
    c = float(c)
    return "CN" if c == 0 else ("AD" if c >= 1 else "MCI")


def _build(path: Path, cohort: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    emb = [c for c in raw.columns if c.startswith("emb_")]
    f = raw[emb].copy()
    f["subject_id"] = raw["participant_id"].astype(str)
    f["age"] = raw["age"].astype(float)
    f["sex"] = pd.Categorical(raw["sex"].map({"F": "F", "M": "M", "female": "F", "male": "M"}),
                              categories=contract.SEX_LEVELS)
    f["dx"] = raw["cdr"].map(_dx)
    f["site"] = cohort
    return f


def _auc(df, emb, y, groups=None, nb=1000, npm=1000):
    r = probe.auc_ci_perm(df[emb].to_numpy(float), y, groups=groups, n_boot=nb, n_perm=npm)
    return {"auc": r["auc"], "ci": None if r["ci_lo"] is None else [r["ci_lo"], r["ci_hi"]],
            "p_perm": r["p_perm"], "n": int(len(y)), "n_pos": int(y.sum())}


def _suite(df, emb, tag):
    cnad = df[df["dx"].isin(["AD", "CN"])]
    y_ad = (cnad["dx"] == "AD").astype(int).to_numpy()
    imp = df.copy(); y_imp = df["dx"].isin(["AD", "MCI"]).astype(int).to_numpy()
    yb = (df["site"] == df["site"].unique()[-1]).astype(int).to_numpy()
    leak = _auc(df, emb, yb)
    print(f"[{tag}] cohort-leakage AUC {leak['auc']} CI {leak['ci']}")
    ad = _auc(cnad, emb, y_ad)
    print(f"[{tag}] pooled AD-vs-CN AUC {ad['auc']} CI {ad['ci']} (n={ad['n']}, {ad['n_pos']} AD)")
    return {"cohort_leakage": leak, "pooled_ad_vs_cn": ad,
            "pooled_impaired_vs_cn": _auc(imp, emb, y_imp)}


def main() -> int:
    if not OASIS2.exists():
        raise SystemExit("OASIS-2 embeddings missing — run the skull-stripped OASIS-2 embed first.")
    o1, o2 = _build(OASIS1, "OASIS-1"), _build(OASIS2, "OASIS-2")
    emb = [c for c in o1.columns if c.startswith("emb_")]
    o2 = o2[o1.columns]
    combined = pd.concat([o1, o2], ignore_index=True)
    print(f"[harmonize] n_total={len(combined)} (OASIS-1={len(o1)}, OASIS-2={len(o2)})\n")

    print("=== BEFORE harmonization (raw embeddings) ===")
    before = _suite(combined, emb, "raw")

    # ComBat: remove the cohort ('site') batch effect, preserve age+sex, label-blind on dx.
    harmonized = harmonize.harmonize(combined, batch="site", covariates=("age", "sex"))
    print("\n=== AFTER ComBat (batch=cohort, preserve age+sex, dx label-blind) ===")
    after = _suite(harmonized, emb, "combat")

    report = {
        "n_total": int(len(combined)),
        "cohorts": {"OASIS-1": int(len(o1)), "OASIS-2": int(len(o2))},
        "before_harmonization": before,
        "after_combat": after,
        "method": ("ComBat (neuroad.data.harmonize) batch=cohort, covariates=(age,sex), dx-blind; "
                   "AUCs via probe.auc_ci_perm leakage-free CV with auto PCA-10."),
        "pass_criteria": ("cohort_leakage AUC -> ~0.5 (batch removed) AND pooled AD-vs-CN survives "
                          "-> honest pooling to n_total."),
        "verdict": None,
    }
    lk = after["cohort_leakage"]["auc"]; ad = after["pooled_ad_vs_cn"]["auc"]
    report["verdict"] = (
        "PASS — batch removed and disease survives; pooled AD-vs-CN is honest."
        if (lk is not None and lk < 0.7 and ad is not None and ad >= 0.75)
        else "PARTIAL — see numbers; batch not fully removed or disease attenuated.")
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n[verdict] {report['verdict']}")
    print(f"[harmonized] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
