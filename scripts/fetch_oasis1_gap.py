#!/usr/bin/env python3
"""
Build the OASIS-1 embedding GAP manifest — the 61->~235 power injection.

We already hold Neuro-JEPA embeddings for 61 OASIS-1 subjects; the cross-sectional
release has 235 CDR-labeled subjects. This script diffs the two, derives the
label/covariate columns, and emits a manifest CSV ready for `neurojepa_embed.py`
(one row per not-yet-embedded labeled subject). It does NOT download volumes or
touch a GPU — it just tells you exactly which subjects to fetch and embed.

Usage:
    PYTHONPATH=src ./.venv/bin/python scripts/fetch_oasis1_gap.py \
        --out data/real/_manifests/oasis1_gap_manifest.csv

Then (Lane B, on Colab — see docs/COLAB_RUNBOOK.md):
    1. download each subject's T88 masked_gfc volume,
    2. set the manifest's image_path column to the local .img path,
    3. run neurojepa_embed.py --manifest ... on a T4,
    4. concat the result onto data/real/oasis1_neurojepa_embeddings.csv.
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

EMB = "data/real/oasis1_neurojepa_embeddings.csv"
XSEC = "data/real/oasis_cross-sectional.csv"


def _dx_from_cdr(cdr: float) -> str:
    """OASIS-1 convention: CDR 0 = CN, 0.5 = MCI (very mild), >=1 = AD/dementia."""
    if pd.isna(cdr):
        return "NA"
    if cdr == 0:
        return "CN"
    if cdr == 0.5:
        return "MCI"
    return "AD"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default=EMB)
    ap.add_argument("--xsec", default=XSEC)
    ap.add_argument("--out", default="data/real/_manifests/oasis1_gap_manifest.csv")
    args = ap.parse_args()

    emb = pd.read_csv(args.emb)
    xs = pd.read_csv(args.xsec)

    embedded = set(emb["participant_id"].astype(str))
    labeled = xs[xs["CDR"].notna()].copy()
    labeled["participant_id"] = labeled["ID"].astype(str)

    gap = labeled[~labeled["participant_id"].isin(embedded)].copy()
    gap["dx"] = gap["CDR"].map(_dx_from_cdr)
    gap["age"] = gap["Age"]
    gap["sex"] = gap["M/F"]
    gap["cdr"] = gap["CDR"]
    gap["site"] = "OASIS-1"          # single-site cohort (stated in every claim)
    gap["scanner"] = "1.5T-Siemens"  # OASIS-1 acquisition (constant -> no leakage test here)
    # image_path is filled AFTER download; default to the canonical OASIS-1 layout.
    gap["image_path"] = gap["participant_id"].map(
        lambda s: f"OASIS1_RAW/{s}/PROCESSED/MPRAGE/T88_111/"
                  f"{s}_mpr_n4_anon_111_t88_masked_gfc.img"
    )

    cols = ["participant_id", "image_path", "dx", "cdr", "age", "sex", "site", "scanner"]
    out = gap[cols].sort_values("participant_id").reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False)

    print(f"[gap] embedded already: {len(embedded)}")
    print(f"[gap] CDR-labeled total: {len(labeled)}")
    print(f"[gap] NEW subjects to embed: {len(out)}  (target total {len(embedded)+len(out)})")
    print(f"[gap] dx mix of the gap: {out['dx'].value_counts().to_dict()}")
    print(f"[gap] wrote manifest -> {args.out}")
    print("[gap] next: download volumes, point image_path at local .img, run neurojepa_embed.py (see docs/COLAB_RUNBOOK.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
