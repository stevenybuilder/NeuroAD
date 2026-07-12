#!/usr/bin/env python3
"""
Reproducible generator for the Layer-2 ATTENTIVE MLP PROBE report.

Persists the head-to-head result that ``attentive_probe.evaluate`` was written to
produce but that lived only in unit tests until now: the nonlinear MLP head vs the
linear probe on the frozen Neuro-JEPA embedding, for AD-vs-CN — plus the
leave-one-group-out ``feature_grounding`` attribution ("what drives the signal").

WHY THIS MATCHES THE PAPER (honestly). NeuroVFM (Nat Med 2026, "Health system
learning enables generalist neuroimaging models") freezes a Vol-JEPA encoder
trained on 5.24M clinical volumes and trains ATTENTIVE MLP PROBES for downstream
diagnosis: a learned attention pooling over per-patch tokens feeding an MLP class
head, encoder frozen. Two honest differences from our substrate, stated up front:

  * SCALE. Their encoder saw 5.24M volumes; ours is a frozen NeuroJEPA embedding
    over cohorts of hundreds. We therefore REPORT whether the nonlinear head helps
    at our n rather than assuming it — the module's verdict says so plainly.
  * TOKENS. Their attention pools over per-patch tokens (enabling spatial attention
    MAPS). Our embedding is a single POOLED 768-d vector per subject, so we cannot
    reproduce literal spatial maps. We substitute leave-one-group-out attribution
    over the embedding block + NAMED clinical features (p_tau217, gfap, nfl, apoe4,
    age) — interpretable grounding, honestly labelled as attribution not attention.

Everything runs through the SAME leakage-honest machinery as the linear probe
(StandardScaler + in-fold PCA, site-disjoint CV, bootstrap CI, permutation null,
repeated-CV ensembling), so linear-vs-MLP is apples-to-apples.

Compliance: reads only the derived (git-ignored) embedding tables, never the gated
weights; writes only AUC-grade numbers to reports/ (repo-safe).

Usage:
    PYTHONPATH=src ./.venv/bin/python scripts/run_attentive_probe.py
    PYTHONPATH=src ./.venv/bin/python scripts/run_attentive_probe.py --cohort oasis1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from neuroad import attentive_probe as ap

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = _ROOT / "reports" / "attentive_probe_ad.json"

#: Cohort embedding tables (git-ignored; regenerate on a GPU runtime).
COHORTS = {
    "adni": _ROOT / "data" / "real" / "adni_neurojepa_embeddings.csv",
    "oasis1": _ROOT / "data" / "real" / "oasis1_neurojepa_embeddings.csv",
    "oasis2": _ROOT / "data" / "real" / "oasis2_neurojepa_embeddings.csv",
}


def _dx_from_cdr(cdr: float) -> str:
    if pd.isna(cdr):
        return "NA"
    if cdr == 0:
        return "CN"
    if cdr >= 1:
        return "AD"
    return "MCI"  # CDR 0.5 = very mild / questionable


def _load(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "dx" not in df.columns and "cdr" in df.columns:
        df["dx"] = df["cdr"].astype(float).map(_dx_from_cdr)
    if "site" not in df.columns:
        df["site"] = "__na__"
    return df


def _run_cohort(name: str, df: pd.DataFrame, *, n_repeats: int,
                n_boot: int, n_perm: int) -> dict:
    emb_dim = sum(c.startswith("emb_") for c in df.columns)
    dx_counts = (df["dx"].value_counts().to_dict()
                 if "dx" in df.columns else {})
    evalr = ap.evaluate(df, "dx_binary", n_repeats=n_repeats,
                        n_boot=n_boot, n_perm=n_perm)
    grounding = ap.feature_grounding(df, "dx_binary", n_repeats=n_repeats)
    return {
        "cohort": name,
        "n_subjects": int(len(df)),
        "embedding_dim": emb_dim,
        "dx": {k: int(dx_counts.get(k, 0)) for k in ("CN", "MCI", "AD")},
        "evaluate": evalr,
        "feature_grounding": grounding,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cohort", choices=[*COHORTS, "all"], default="all")
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--n-repeats", type=int, default=5)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--n-perm", type=int, default=1000)
    args = p.parse_args()

    wanted = list(COHORTS) if args.cohort == "all" else [args.cohort]
    results = []
    missing = []
    for name in wanted:
        df = _load(COHORTS[name])
        if df is None:
            missing.append(name)
            continue
        results.append(_run_cohort(name, df, n_repeats=args.n_repeats,
                                   n_boot=args.n_boot, n_perm=args.n_perm))

    if not results:
        raise SystemExit(
            "no embedding tables found (git-ignored by design).\n"
            "Regenerate on a GPU runtime with your own HF_TOKEN "
            "(scripts/neurojepa_embed_colab.py; see docs/COLAB_RUNBOOK.md).")

    report = {
        "layer": "2 — attentive MLP probe (NeuroVFM-style, frozen embedding)",
        "target": "dx_binary (AD vs CN)",
        "method": (
            "attentive_probe.evaluate: small regularized MLPClassifier head "
            "(hidden=(32,), alpha=1.0) vs linear probe, both through "
            "probe.auc_ci_perm (StandardScaler + in-fold auto-PCA, site-disjoint "
            "CV, bootstrap CI, permutation null, repeated-CV ensembling). Grounding "
            "= leave-one-group-out AUC attribution over embedding + named markers."),
        "paper_reference": (
            "NeuroVFM, Nat Med 2026 (s41591-026-04497-1): frozen Vol-JEPA encoder "
            "(5.24M volumes) + attentive MLP probe for downstream diagnosis. We "
            "adapt honestly to a pooled 768-d embedding at n in the hundreds: report "
            "(not assume) nonlinear lift; LOGO attribution substitutes for spatial "
            "attention maps our pooled embedding cannot provide."),
        "n_repeats": args.n_repeats,
        "n_boot": args.n_boot,
        "n_perm": args.n_perm,
        "cohorts": results,
        "missing_cohorts": missing,
        "compliance": ("Frozen inference only; weights never stored; embedding "
                       "tables git-ignored. Only AUC-grade numbers written here."),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    print(f"[attentive-probe] wrote {out}")
    for r in results:
        e = r["evaluate"]
        lin, mlp = e["linear"], e["mlp"]
        print(f"[{r['cohort']}] n={e['n']} dx={r['dx']}")
        print(f"   linear AUC {lin['auc']}  CI [{lin['ci_lo']},{lin['ci_hi']}]"
              f"  p={lin['p_perm']}")
        print(f"   MLP    AUC {mlp['auc']}  CI [{mlp['ci_lo']},{mlp['ci_hi']}]"
              f"  p={mlp['p_perm']}  (delta={e['delta_auc_mlp_minus_linear']:+})")
        print(f"   verdict: {e['verdict']}")
        g = r["feature_grounding"]
        if g:
            drivers = ", ".join(f"{a['group']}:{a['loo_auc_drop']:+}"
                                for a in g["attribution"])
            print(f"   grounding: full_auc={g['full_auc']} top={g['top_driver']}"
                  f" | {drivers}")
    if missing:
        print(f"[attentive-probe] skipped (no embedding table): {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
