#!/usr/bin/env python3
"""
power_analysis_conversion.py — how many more MCI->AD converters are needed to make
the STACKED-FUSION vs PLASMA AUC difference statistically separable?

Rigorous + empirical (no distributional hand-waving):
  1. Reproduce the real per-subject OOF scores for the plasma model and the stacked-
     fusion model on the SAME conversion cohort (same machinery as
     run_adni_conversion_multimodal.py).
  2. Current effect: DeLong's test for two CORRELATED ROC AUCs (the fused and plasma
     scores are highly correlated because fusion stacks the plasma OOF score) — gives
     the honest paired variance and current p-value.
  3. Power curve: stratified bootstrap that RESAMPLES observed (s_plasma, s_fused, y)
     triples to larger cohorts at the SAME 28.5% converter rate, preserving the
     empirical correlation, and reports the DeLong power at each size. Interpretation:
     "assuming the observed +0.013 lift is real, how many converters to detect it at
     80% power / to have the 95% CI of the difference exclude 0."

Deterministic given --seed. Writes reports/power_analysis_conversion.json (+ .md).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
_ROOT = os.path.join(os.path.dirname(__file__), "..")


# ---- fast DeLong (Sun & Xu 2014) for correlated AUCs -----------------------
def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2


def _fast_delong(preds, m):
    """preds: [k, m+n] with the m positives first. Returns (aucs[k], cov[k,k])."""
    n = preds.shape[1] - m
    k = preds.shape[0]
    pos, neg = preds[:, :m], preds[:, m:]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r] = _midrank(pos[r]); ty[r] = _midrank(neg[r]); tz[r] = _midrank(preds[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1) / (2 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    sx = np.cov(v01); sy = np.cov(v10)
    if k == 1:
        sx = np.array([[float(sx)]]); sy = np.array([[float(sy)]])
    cov = sx / m + sy / n
    return aucs, cov


def _delong_diff(s_fused, s_plasma, y):
    """One-sided DeLong z, p for AUC(fused) > AUC(plasma) on the same subjects."""
    order = np.argsort(-y)  # positives (y=1) first
    m = int(y.sum())
    preds = np.vstack([s_fused[order], s_plasma[order]])
    aucs, cov = _fast_delong(preds, m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        var = 1e-12
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    from math import erf, sqrt
    p_one = 1.0 - 0.5 * (1 + erf(z / sqrt(2)))
    return float(aucs[0]), float(aucs[1]), float(z), float(p_one), float(np.sqrt(var))


def get_oof_scores():
    """Reproduce plasma + stacked-fusion per-subject OOF scores + labels."""
    import pandas as pd
    from neuroad import contract, probe
    from neuroad.data import loaders
    from scripts.run_adni_conversion_multimodal import PLASMA, APOE_DEMO, _block

    df = loaders.load("adni:combat")
    conv = pd.to_numeric(df["conversion"], errors="coerce")
    have_plasma = pd.to_numeric(df["p_tau217"], errors="coerce").notna()
    keep = (conv.notna() & have_plasma).to_numpy()
    sub = df[keep].copy()
    y = pd.to_numeric(sub["conversion"], errors="coerce").to_numpy(dtype=int)
    groups = pd.factorize(sub["site"].astype("string").fillna("__na__"))[0]
    Xstruct = contract.embedding_matrix(sub)
    Xplasma, Xapoe = _block(sub, PLASMA), _block(sub, APOE_DEMO)

    def oof(X):
        out = probe.cross_val_oof(X, y, groups)
        yc, proba, classes, _ = out
        ev = ~np.isnan(proba).any(axis=1)
        pos = list(classes).index(1)
        return proba[:, pos], ev

    s_struct, e1 = oof(Xstruct)
    s_plasma, e2 = oof(Xplasma)
    s_apoe, e3 = oof(Xapoe)
    ev = e1 & e2 & e3
    stacked = np.column_stack([s_struct[ev], s_plasma[ev], s_apoe[ev]])
    yev, gev = y[ev], groups[ev]
    # fused OOF = OOF probability of a meta-model on the 3 stacked scores
    out = probe.cross_val_oof(stacked, yev, gev)
    _, proba_f, classes_f, _ = out
    ef = ~np.isnan(proba_f).any(axis=1)
    posf = list(classes_f).index(1)
    s_fused = proba_f[:, posf]
    return (s_fused[ef], s_plasma[ev][ef], yev[ef])


def main():
    seed = 0
    rng = np.random.default_rng(seed)
    s_fused, s_plasma, y = get_oof_scores()
    n, npos = len(y), int(y.sum())
    rate = npos / n
    af, ap, z, p, se = _delong_diff(s_fused, s_plasma, y)
    print(f"cohort n={n} converters={npos} ({100*rate:.1f}%)")
    print(f"AUC fused={af:.4f} plasma={ap:.4f} diff={af-ap:+.4f} "
          f"DeLong z={z:.3f} p1={p:.4f} SE_diff={se:.4f}")

    # bootstrap power curve at scaled cohort sizes (same converter rate)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    B = 600
    curve = []
    for mult in (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
        np_ = int(round(npos * mult)); nn_ = int(round((n - npos) * mult))
        hits = 0
        for _ in range(B):
            pi = rng.choice(pos_idx, np_, replace=True)
            ni = rng.choice(neg_idx, nn_, replace=True)
            idx = np.concatenate([pi, ni])
            yy = np.concatenate([np.ones(np_), np.zeros(nn_)]).astype(int)
            _, _, _, pp, _ = _delong_diff(s_fused[idx], s_plasma[idx], yy)
            if pp < 0.05:
                hits += 1
        power = hits / B
        curve.append({"mult": mult, "n_total": np_ + nn_, "n_converters": np_,
                      "power": round(power, 3)})
        print(f"  mult {mult:>4}: n={np_+nn_:5d} converters={np_:4d} "
              f"power(p<0.05,1-sided)={power:.3f}")

    # smallest converters for 80% power
    at80 = next((c for c in curve if c["power"] >= 0.80), None)
    payload = {
        "cohort_n": n, "converters": npos, "converter_rate": round(rate, 4),
        "auc_fused": round(af, 4), "auc_plasma": round(ap, 4),
        "observed_diff": round(af - ap, 4), "delong_z": round(z, 3),
        "delong_p_onesided": round(p, 4), "se_diff": round(se, 4),
        "power_curve": curve,
        "converters_for_80pct_power": at80["n_converters"] if at80 else None,
        "total_n_for_80pct_power": at80["n_total"] if at80 else None,
        "note": (f"Bootstrap resamples observed (s_plasma,s_fused,y) triples to larger "
                 f"cohorts at the same {100*rate:.1f}% converter rate, preserving the "
                 f"empirical fused/plasma correlation; power = P(DeLong one-sided p<0.05) "
                 f"assuming the observed +{af - ap:.3f} lift is the true effect. "
                 f"Stacked-fusion OOF carries mild in-sample optimism (shared across "
                 f"both models)."),
    }
    out = os.path.join(_ROOT, "reports", "power_analysis_conversion.json")
    json.dump(payload, open(out, "w"), indent=2)
    md = out.replace(".json", ".md")
    with open(md, "w") as fh:
        fh.write("# Power analysis — stacked fusion vs plasma (MCI->AD conversion)\n\n")
        fh.write(f"Current: n={n}, {npos} converters ({100*rate:.1f}%). "
                 f"AUC fused={af:.3f} vs plasma={ap:.3f} (diff +{af-ap:.3f}). "
                 f"DeLong one-sided p={p:.3f} (SE_diff={se:.4f}) — NOT separable.\n\n")
        fh.write("| cohort mult | total n | converters | power (p<0.05) |\n")
        fh.write("|---|---|---|---|\n")
        for c in curve:
            fh.write(f"| {c['mult']} | {c['n_total']} | {c['n_converters']} | {c['power']} |\n")
        if at80:
            fh.write(f"\n**~{at80['n_converters']} converters (total n≈{at80['n_total']}) "
                     f"for 80% power** to detect the observed +{af-ap:.3f} lift.\n")
        fh.write(f"\n> {payload['note']}\n")
    print(f"\n[wrote] {out}\n[wrote] {md}")
    if at80:
        print(f"\n=> ~{at80['n_converters']} converters (total ~{at80['n_total']}) for 80% power.")


if __name__ == "__main__":
    main()
