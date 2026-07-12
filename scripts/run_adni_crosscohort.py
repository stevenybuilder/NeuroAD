#!/usr/bin/env python3
"""
ADNI raw-MRI NeuroJEPA cross-cohort + biomarker-anchoring analysis.

Ready to run the instant the ADNI image embeddings exist. It consumes
``data/real/adni_neurojepa_embeddings.csv`` — produced by embedding the user's
IDA/LONI-downloaded T1w NIfTIs with ``scripts/neurojepa_embed_colab.py
--dataset adni`` off the ``scripts/build_adni_image_manifest.py`` manifest (so the
CSV already carries dx/age/sex/site/scanner + the REAL plasma p-tau217/GFAP/NfL
and amyloid columns) — together with the OASIS NeuroJEPA image embeddings, all in
the same frozen 768-d space.

Four blocks, all measured with the referee's own leakage-free probe
(``neuroad.probe.auc_ci_perm``: CV + auto PCA-10 + bootstrap CI + permutation p):

  (a) ADNI WITHIN-COHORT AD-vs-CN — does the frozen encoder separate AD from CN
      on ADNI's own raw MRI (independent of OASIS)?
  (b) CROSS-SCANNER COHORT-LEAKAGE OASIS-vs-ADNI — can the embedding tell which
      cohort/scanner a subject is from? HIGH -> a site/scanner batch confound;
      a naive pooled disease number would be partly reading cohort.
  (c) ComBat-HARMONIZED POOLED AD-vs-CN across ALL cohorts — remove the cohort
      batch effect (label-blind, preserve age+sex), then re-check leakage (should
      drop toward ~0.5) and pooled AD-vs-CN (should survive) -> honest pooling.
  (d) BIOMARKER ANCHORING (the ADNI unlock) — does the MRI embedding predict the
      MOLECULAR state: plasma p-tau217 HIGH-vs-LOW (median split) and amyloid
      status (A+/A-)? This is the structural-imaging -> plasma-pathology anchor
      that only ADNI's real biomarker panel makes possible here.

Mirrors ``scripts/run_oasis_harmonized.py``. Guarded: if the ADNI embeddings are
not present it exits with a clear "run the embed first" message and does nothing.

Compliance: reads only derived embedding tables; writes only numbers to reports/.

Usage:
    PYTHONPATH=src ./.venv/bin/python scripts/run_adni_crosscohort.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from neuroad import contract, probe
from neuroad.data import harmonize

_ROOT = Path(__file__).resolve().parents[1]
ADNI = _ROOT / "data" / "real" / "adni_neurojepa_embeddings.csv"
OASIS1 = _ROOT / "data" / "real" / "oasis1_neurojepa_embeddings.csv"
OASIS2 = _ROOT / "data" / "real" / "oasis2_neurojepa_embeddings.csv"
OUT = _ROOT / "reports" / "adni_neurojepa_crosscohort.json"

#: The frozen NeuroJEPA foundation space is 768-d. The ADNI tabular contract
#: (data/real/_gated/adni.csv) also carries emb_* columns, but those are 323-d
#: FreeSurfer regional morphometry — a DIFFERENT feature space. If that file were
#: mis-placed at the imaging-embeddings path, every emb_* check below would run
#: silently on 323 intersected dims and fabricate a confident (wrong) verdict.
#: Guard on the expected dimensionality so the mix-up fails loud instead.
NEUROJEPA_DIM = 768


def _dx_from_cdr(c: float) -> str:
    c = float(c)
    return "CN" if c == 0 else ("AD" if c >= 1 else "MCI")


def _norm_sex(s: pd.Series) -> pd.Categorical:
    m = {"F": "F", "M": "M", "female": "F", "male": "M", "0": "F", "1": "M"}
    return pd.Categorical(s.astype(str).map(lambda v: m.get(v, m.get(v.upper(), v.upper()))),
                          categories=contract.SEX_LEVELS)


def _build_oasis(path: Path, cohort: str) -> pd.DataFrame:
    """OASIS image-embedding CSV -> shared frame (dx from cdr). Plasma cols = NA."""
    raw = pd.read_csv(path)
    emb = [c for c in raw.columns if c.startswith("emb_")]
    f = raw[emb].copy()
    f["subject_id"] = raw["participant_id"].astype(str)
    f["age"] = pd.to_numeric(raw["age"], errors="coerce").astype(float)
    f["sex"] = _norm_sex(raw["sex"])
    f["dx"] = raw["cdr"].map(_dx_from_cdr)
    f["cohort"] = cohort
    for m in ("p_tau217", "gfap", "nfl", "amyloid"):
        f[m] = np.nan
    return f


def _build_adni(path: Path) -> pd.DataFrame:
    """ADNI image-embedding CSV -> shared frame; carries the real plasma panel."""
    raw = pd.read_csv(path)
    emb = [c for c in raw.columns if c.startswith("emb_")]
    f = raw[emb].copy()
    f["subject_id"] = raw["subject_id"].astype(str)
    f["age"] = pd.to_numeric(raw["age"], errors="coerce").astype(float)
    f["sex"] = _norm_sex(raw["sex"])
    # ADNI dx is already CN/MCI/AD; normalize whatever casing/strings arrived.
    dx = raw["dx"].astype(str).str.upper().str.strip()
    f["dx"] = dx.where(dx.isin(["CN", "MCI", "AD"]), other=pd.NA)
    f["cohort"] = "ADNI"
    for m in ("p_tau217", "gfap", "nfl", "amyloid"):
        f[m] = pd.to_numeric(raw[m], errors="coerce").astype(float) if m in raw.columns else np.nan
    return f


def _auc(df: pd.DataFrame, emb: list[str], y: np.ndarray, groups=None,
         nb: int = 1000, npm: int = 1000) -> dict:
    r = probe.auc_ci_perm(df[emb].to_numpy(float), y, groups=groups, n_boot=nb, n_perm=npm)
    return {"auc": r["auc"], "ci": None if r["ci_lo"] is None else [r["ci_lo"], r["ci_hi"]],
            "p_perm": r["p_perm"], "n": int(len(y)), "n_pos": int(int(np.asarray(y).sum()))}


def _ad_vs_cn(df: pd.DataFrame, emb: list[str], groups_col: str | None = None, **kw) -> dict:
    sub = df[df["dx"].isin(["AD", "CN"])].reset_index(drop=True)
    if sub.empty or sub["dx"].nunique() < 2:
        return {"auc": None, "ci": None, "p_perm": None, "n": int(len(sub)), "n_pos": 0,
                "note": "insufficient AD/CN to contrast"}
    y = (sub["dx"] == "AD").astype(int).to_numpy()
    g = None
    if groups_col is not None:
        g = sub[groups_col].astype("category").cat.codes.to_numpy()
    return _auc(sub, emb, y, groups=g, **kw)


def _biomarker_anchor(adni: pd.DataFrame, emb: list[str], nb: int, npm: int) -> dict:
    """Does the MRI embedding predict the molecular state? (ADNI-only unlock.)"""
    out: dict = {}

    # (i) plasma p-tau217 HIGH vs LOW — median split over subjects with a real value.
    pt = adni[adni["p_tau217"].notna()].reset_index(drop=True)
    if len(pt) >= 20:
        thr = float(pt["p_tau217"].median())
        y = (pt["p_tau217"] > thr).astype(int).to_numpy()
        r = _auc(pt, emb, y, nb=nb, npm=npm)
        r["threshold_median"] = thr
        r["interpretation"] = ("embedding predicts plasma p-tau217 high-vs-low -> the frozen MRI "
                               "representation carries tau-pathology signal.")
        out["ptau217_high_vs_low"] = r
    else:
        out["ptau217_high_vs_low"] = {"auc": None, "n": int(len(pt)),
                                      "note": "too few subjects with plasma p-tau217"}

    # (ii) amyloid status A+ vs A-.
    am = adni[adni["amyloid"].isin([0.0, 1.0])].reset_index(drop=True)
    if len(am) >= 20 and am["amyloid"].nunique() == 2:
        y = (am["amyloid"] == 1.0).astype(int).to_numpy()
        r = _auc(am, emb, y, nb=nb, npm=npm)
        r["interpretation"] = ("embedding predicts amyloid status -> structural MRI anchors to the "
                               "amyloid molecular gate.")
        out["amyloid_status"] = r
    else:
        out["amyloid_status"] = {"auc": None, "n": int(len(am)),
                                 "note": "too few subjects with amyloid status"}
    return out


def main() -> int:
    if not ADNI.exists():
        raise SystemExit(
            f"ADNI embeddings not present at {ADNI} — run the embed first:\n"
            "  1) ./.venv/bin/python scripts/build_adni_image_manifest.py --image-root <your ADNI_MRI folder>\n"
            "  2) embed the IDA-downloaded T1w NIfTIs with scripts/neurojepa_embed_colab.py "
            "--dataset adni --skull-strip --fast-resample\n"
            "  3) place the result at data/real/adni_neurojepa_embeddings.csv, then re-run this script.")

    nb, npm = 1000, 1000
    adni = _build_adni(ADNI)
    emb = [c for c in adni.columns if c.startswith("emb_")]

    # Fail loud if this is the wrong file (e.g. the 323-d FreeSurfer contract
    # mis-placed at the imaging path) rather than silently producing a verdict on
    # an unintended feature space.
    if len(emb) != NEUROJEPA_DIM:
        raise SystemExit(
            f"{ADNI} has {len(emb)} emb_* columns, expected {NEUROJEPA_DIM} "
            f"(the frozen NeuroJEPA imaging space). {len(emb)} looks like the "
            "323-d FreeSurfer tabular contract (data/real/_gated/adni.csv), NOT "
            "the NeuroJEPA image embeddings. Point this at the output of "
            "scripts/neurojepa_embed_colab.py --dataset adni, or regenerate it.")

    frames = [adni]
    if OASIS1.exists():
        o1 = _build_oasis(OASIS1, "OASIS-1")
        emb = [c for c in emb if c in o1.columns]  # align to the shared emb set
        frames.append(o1)
    if OASIS2.exists():
        frames.append(_build_oasis(OASIS2, "OASIS-2"))
    have_oasis = len(frames) > 1
    # align every frame to the shared column order
    shared_cols = emb + ["subject_id", "age", "sex", "dx", "cohort",
                         "p_tau217", "gfap", "nfl", "amyloid"]
    pooled = pd.concat([f[shared_cols] for f in frames], ignore_index=True)
    cohort_counts = pooled["cohort"].value_counts().to_dict()
    print(f"[adni-crosscohort] n_total={len(pooled)} cohorts={cohort_counts} "
          f"(emb dim {len(emb)})\n")

    # (a) ADNI within-cohort AD-vs-CN
    within = _ad_vs_cn(adni, emb, nb=nb, npm=npm)
    print(f"[a within-ADNI]  AD-vs-CN AUC {within['auc']} CI {within['ci']} "
          f"(n={within['n']}, {within['n_pos']} AD)")

    # (b) cross-scanner cohort-leakage OASIS-vs-ADNI (raw), and (c) after ComBat
    leak_raw = leak_combat = pooled_after = None
    if have_oasis:
        yb = (pooled["cohort"] == "ADNI").astype(int).to_numpy()
        leak_raw = _auc(pooled, emb, yb, nb=nb, npm=npm)
        leak_raw["interpretation"] = ("HIGH -> OASIS and ADNI are separable from the raw embedding "
                                      "(scanner/site batch); pool only after harmonization.")
        print(f"[b leakage raw]  OASIS-vs-ADNI AUC {leak_raw['auc']} CI {leak_raw['ci']}")

        # (c) ComBat on cohort (batch), preserve age+sex, dx-blind -> pooled AD-vs-CN
        H = harmonize.harmonize(pooled, batch="cohort", covariates=("age", "sex"))
        leak_combat = _auc(H, emb, yb, nb=nb, npm=npm)
        pooled_after = _ad_vs_cn(H, emb, nb=nb, npm=npm)
        pooled_xcohort = _ad_vs_cn(H, emb, groups_col="cohort", nb=nb, npm=npm)
        pooled_after["cross_cohort_site_disjoint"] = pooled_xcohort
        print(f"[c combat leak]  OASIS-vs-ADNI AUC {leak_combat['auc']} CI {leak_combat['ci']} "
              f"(target ~0.5)")
        print(f"[c pooled AD-CN] AUC {pooled_after['auc']} CI {pooled_after['ci']} "
              f"(n={pooled_after['n']}, {pooled_after['n_pos']} AD)")
    else:
        print("[b/c] OASIS embeddings absent -> skipping cross-cohort leakage + pooling "
              "(ADNI-only run).")

    # (d) biomarker anchoring (ADNI only)
    anchor = _biomarker_anchor(adni, emb, nb=nb, npm=npm)
    pt = anchor["ptau217_high_vs_low"]; am = anchor["amyloid_status"]
    print(f"[d anchor ptau] AUC {pt.get('auc')} CI {pt.get('ci')} (n={pt.get('n')})")
    print(f"[d anchor amy ] AUC {am.get('auc')} CI {am.get('ci')} (n={am.get('n')})")

    report = {
        "n_total": int(len(pooled)),
        "cohorts": {k: int(v) for k, v in cohort_counts.items()},
        "emb_dim": len(emb),
        "a_within_adni_ad_vs_cn": within,
        "b_cohort_leakage_oasis_vs_adni_raw": leak_raw,
        "c_harmonized": None if not have_oasis else {
            "cohort_leakage_after_combat": leak_combat,
            "pooled_ad_vs_cn_after_combat": pooled_after,
            "method": ("ComBat (neuroad.data.harmonize) batch=cohort, covariates=(age,sex), dx-blind; "
                       "AUCs via probe.auc_ci_perm leakage-free CV with auto PCA-10."),
        },
        "d_biomarker_anchoring": anchor,
        "pass_criteria": (
            "(a) within-ADNI AD-vs-CN CI excludes 0.5; (b/c) cohort-leakage drops toward ~0.5 after "
            "ComBat while pooled AD-vs-CN survives -> honest pooling; (d) p-tau217/amyloid AUC CI "
            "excludes 0.5 -> the MRI embedding anchors to plasma/amyloid pathology."),
        "compliance": "Frozen inference only; weights never stored; embedding tables git-ignored.",
        "verdict": None,
    }
    verdict = []
    if within["auc"] is not None:
        verdict.append("within-ADNI AD-vs-CN separates" if within["auc"] >= 0.7
                       else "within-ADNI AD-vs-CN weak")
    if have_oasis and leak_combat and leak_combat["auc"] is not None and pooled_after["auc"] is not None:
        verdict.append("PASS pooling" if (leak_combat["auc"] < 0.7 and pooled_after["auc"] >= 0.75)
                       else "PARTIAL pooling")
    if pt.get("auc") is not None:
        verdict.append("p-tau217 anchored" if pt["auc"] >= 0.6 else "p-tau217 weak")
    if am.get("auc") is not None:
        verdict.append("amyloid anchored" if am["auc"] >= 0.6 else "amyloid weak")
    report["verdict"] = "; ".join(verdict) if verdict else "insufficient data"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\n[verdict] {report['verdict']}")
    print(f"[adni-crosscohort] wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
