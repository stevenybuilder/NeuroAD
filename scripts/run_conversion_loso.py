#!/usr/bin/env python3
"""
MCI->AD conversion (pMCI vs sMCI) through the ``adni:conversion`` feeder, under a
LEAVE-ONE-SITE-OUT (site-disjoint) split — the honest single-cohort generalization
test for the prognostic arm.

Why this and not a cross-DATASET leave-one-cohort-out: only ADNI carries a
longitudinal converter label today, so a true LOCO (>=2 conversion-labeled cohorts
in the shared frozen 768-d space) is not yet runnable. The multi-site structure of
the 334-subject cohort (58 sites) makes leave-one-site-out the honest analog: no
subject's acquisition site appears in both train and test, so a site/scanner batch
effect cannot inflate the number. When a second conversion cohort (OASIS-2 cdr
trajectory / AIBL / NACC) is embedded into the same space, swap ``groups=site`` for
``groups=cohort`` for the real cross-dataset LOCO.

Three feature blocks, all measured with the referee's own leakage-free probe
(``probe.auc_ci_perm``: grouped CV + auto PCA-10 + bootstrap 95% CI + permutation p),
on the SAME row set so the contrast is clean:

  * neurojepa — the frozen Neuro-JEPA 768-d MRI embedding (imaging)
  * plasma    — p_tau217, gfap, nfl (the blood workhorse)
  * fused     — neurojepa + plasma concatenated

The headline question the conversion arm was built to answer: does the structural
MRI embedding ADD prognostic signal over plasma for who converts? The cross-sectional
AD-vs-CN cohort says imaging adds little on top of plasma p-tau217 (~0.93); this is
where imaging is supposed to earn its keep. The cohort is SMALL (58 converters) —
genuinely underpowered — so an at-chance or not-CI-separable result is reported as
such, never dressed up.

Compliance: reads only the derived embedding table (git-ignored) + the gated
conversion label; writes only numbers to reports/.

Usage:
    PYTHONPATH=src ./.venv/bin/python scripts/run_conversion_loso.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from neuroad import contract, probe
from neuroad.data import adni_conversion_jepa, loaders

_ROOT = Path(__file__).resolve().parents[1]
_OUT_JSON = _ROOT / "reports" / "adni_conversion_loso.json"
_OUT_MD = _ROOT / "reports" / "ADNI_CONVERSION_LOSO.md"

#: The frozen Neuro-JEPA foundation space is 768-d. Guard so a mis-placed 323-d
#: FreeSurfer contract can never silently score the wrong feature space.
NEUROJEPA_DIM = 768
_PLASMA = ["p_tau217", "gfap", "nfl"]
_NB, _NPM = 1000, 1000


def _block(df, cols, y, groups) -> dict:
    X = df[cols].to_numpy(dtype=float)
    r = probe.auc_ci_perm(X, y, groups=groups, n_boot=_NB, n_perm=_NPM)
    return {"auc": r["auc"],
            "ci": None if r["ci_lo"] is None else [r["ci_lo"], r["ci_hi"]],
            "p_perm": r["p_perm"], "n": int(len(y)),
            "n_converters": int(np.asarray(y).sum()), "d": len(cols)}


def _fmt(d: dict) -> str:
    return f"AUC {d['auc']} [{d['ci'][0]}, {d['ci'][1]}], p_perm={d['p_perm']}" \
        if d.get("ci") else f"AUC {d.get('auc')} (CI n/a)"


def main() -> int:
    df = loaders.load("adni:conversion")

    emb = contract.embedding_columns(df)
    if len(emb) != NEUROJEPA_DIM:
        raise SystemExit(
            f"adni:conversion has {len(emb)} emb_* columns, expected {NEUROJEPA_DIM} "
            "(the frozen Neuro-JEPA imaging space). This looks like the 323-d "
            "FreeSurfer contract mis-placed at the embedding path — refusing to "
            "score the wrong feature space.")

    # Same complete-case row set for every block: conversion label + full plasma.
    keep = df["conversion"].notna()
    for c in _PLASMA:
        keep &= df[c].notna()
    sub = df[keep].reset_index(drop=True)
    y = sub["conversion"].astype(int).to_numpy()
    groups = sub["site"].astype("category").cat.codes.to_numpy()
    n_sites = int(sub["site"].nunique())

    print(f"[conversion-loso] n={len(sub)} ({int(y.sum())} pMCI / {int((y == 0).sum())} "
          f"sMCI), {n_sites} sites, leave-one-site-out (site-disjoint grouped CV), "
          f"NeuroJEPA D={len(emb)}", flush=True)

    neuro = _block(sub, emb, y, groups)
    plasma = _block(sub, _PLASMA, y, groups)
    fused = _block(sub, emb + _PLASMA, y, groups)
    print(f"[neurojepa] {_fmt(neuro)}", flush=True)
    print(f"[plasma]    {_fmt(plasma)}", flush=True)
    print(f"[fused]     {_fmt(fused)}", flush=True)

    delta = round(float(fused["auc"]) - float(plasma["auc"]), 4)
    # CI-separability: does the fused CI clear the plasma point estimate?
    fused_beats_plasma = bool(
        fused.get("ci") and fused["ci"][0] > float(plasma["auc"]))
    if fused_beats_plasma:
        verdict = (f"Fused imaging+plasma ({fused['auc']}) CI-clears plasma alone "
                   f"({plasma['auc']}): the NeuroJEPA embedding adds prognostic "
                   "signal for conversion under a site-disjoint split.")
    elif delta > 0:
        verdict = (f"Fused ({fused['auc']}) edges plasma ({plasma['auc']}) by "
                   f"{delta:+} AUC but the CIs overlap: no CI-supported imaging gain "
                   "at this sample size (underpowered; 58 converters).")
    else:
        verdict = (f"Plasma alone ({plasma['auc']}) is not beaten by adding imaging "
                   f"(naive-concat fused {fused['auc']}, {delta:+}): imaging does not "
                   "add conversion signal over plasma on this single cohort. (The "
                   "concat block PCA-10s 768 imaging + 3 plasma dims together, so it "
                   "under-weights plasma and understates fusion — the attention-"
                   "weighted fusion in scripts/run_conversion_fusion.py reaches ~0.82 "
                   "but still shows no CI-supported gain over plasma. Both fusion "
                   "methods agree: plasma dominates conversion at this sample size.)")

    report = {
        "cohort": "ADNI MCI-conversion (adni:conversion feeder), 334 baseline-MCI",
        "target": "conversion (pMCI=1 / sMCI=0)",
        "split": "leave-one-site-out (site-disjoint grouped CV; no site in both "
                 "train and test)",
        "n": len(sub), "n_sites": n_sites,
        "substrate": loaders.honest_substrate("adni:conversion"),
        "blocks": {"neurojepa": neuro, "plasma": plasma, "fused": fused},
        "fused_minus_plasma_auc": delta,
        "fused_ci_clears_plasma": fused_beats_plasma,
        "verdict": verdict,
        "honesty": (
            "Single-cohort, underpowered (58 converters). A TRUE cross-dataset "
            "leave-one-cohort-out awaits a second conversion-labeled cohort "
            "(OASIS-2 cdr trajectory / AIBL / NACC) embedded in the same frozen "
            "768-d space; site-disjoint LOSO is the honest single-cohort analog. "
            "The 'fused' block is a NAIVE concat (768 imaging + 3 plasma) auto-PCA-10'd "
            "together, which under-weights the low-dim plasma signal and understates "
            "fusion; the attention-weighted late fusion in run_conversion_fusion.py is "
            "the proper multimodal number (~0.82) and still shows no CI-supported "
            "imaging gain over plasma. AUCs via probe.auc_ci_perm (grouped CV, auto "
            "PCA-10, bootstrap CI, permutation null). Frozen inference only; weights "
            "never stored."),
    }
    _OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    _OUT_JSON.write_text(json.dumps(report, indent=2) + "\n")

    md = [
        "# MCI->AD Conversion — Leave-One-Site-Out (adni:conversion feeder)",
        "",
        f"**Cohort:** {report['n']} baseline-MCI subjects "
        f"({neuro['n_converters']} pMCI / {report['n'] - neuro['n_converters']} sMCI), "
        f"{n_sites} sites. **Target:** conversion (pMCI vs sMCI). **Split:** "
        "leave-one-site-out (site-disjoint) — no acquisition site in both train and test.",
        f"**Substrate:** {report['substrate']}.",
        "",
        "| Block | Features | AUC (site-disjoint) |",
        "|---|---|---|",
        f"| Neuro-JEPA imaging | 768-d frozen MRI embedding | {_fmt(neuro)} |",
        f"| Plasma | p-tau217, GFAP, NfL | {_fmt(plasma)} |",
        f"| Fused | Neuro-JEPA + plasma | {_fmt(fused)} |",
        "",
        f"**Fused − plasma:** {delta:+} AUC "
        f"({'CI-clears plasma' if fused_beats_plasma else 'CIs overlap'}).",
        "",
        f"**Verdict:** {verdict}",
        "",
        "## Honesty",
        "",
        report["honesty"],
    ]
    _OUT_MD.write_text("\n".join(md) + "\n")

    print(f"[conversion-loso] verdict: {verdict}", flush=True)
    print(f"[conversion-loso] wrote {_OUT_JSON}", flush=True)
    print(f"[conversion-loso] wrote {_OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
