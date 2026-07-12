#!/usr/bin/env python3
"""
Attention-weighted late fusion for MCI->AD CONVERSION on the 334-subject fusion
cohort, wiring the frozen NeuroJEPA 768-d embedding in through the module's seam.

The 334 subjects are the MCI_conversion_T1_fusion cohort we just embedded on Colab
(scripts/run_conversion_embed_colab.py -> adni_conversion_neurojepa_embeddings.csv).
They are all baseline MCI with a converter/stable outcome, so the prognostic target
is ``conversion`` (converter=1 / stable=0), NOT cross-sectional dx (AD vs CN).

``integrations.fusion.attention_fusion`` gates on ``dx_binary`` (AD->1, CN->0). To
run the SAME validated machinery on the conversion phenotype without touching that
guarded module, we encode the outcome in the ``dx`` column (converter->"AD",
stable->"CN"); ``dx_binary`` then IS the conversion label. The output is re-stamped
``target="conversion"`` so the artifact never mislabels itself as a dx-AD/CN run.

Three modalities, gated leakage-free (site-disjoint OOF, same CV as the gauntlet):
  * imaging   — the contract's FreeSurfer structural embedding (contract.embedding_matrix)
  * plasma    — p_tau217, gfap, nfl, apoe4, age, sex
  * neurojepa — the 768-d frozen NeuroJEPA imaging embedding (the seam), aligned by
                subject_id and gated in like the others

We report the fusion WITH the NeuroJEPA modality and, for a clean before/after, the
2-modality baseline WITHOUT it (imaging_embedding=None) — so the leave-one-out
attribution and the baseline delta both answer: does NeuroJEPA imaging ADD prognostic
signal over FreeSurfer + plasma for MCI->AD conversion?

Deterministic; reads the GATED contract (data/real/_gated/adni.csv) — local use only.

Usage:
    PYTHONPATH=src ./.venv/bin/python scripts/run_conversion_fusion.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from neuroad import contract
from neuroad.integrations import fusion

_ROOT = Path(__file__).resolve().parents[1]
_CONTRACT = _ROOT / "data" / "real" / "_gated" / "adni.csv"
_NEURO = _ROOT / "data" / "real" / "adni_conversion_neurojepa_embeddings.csv"
_OUT_JSON = _ROOT / "reports" / "adni_conversion_fusion.json"
_OUT_MD = _ROOT / "reports" / "ADNI_CONVERSION_FUSION.md"


def _build_slice() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (contract_df_for_the_334, neurojepa_frame) ready for attention_fusion.

    ``dx`` is overwritten with the conversion outcome (converter->AD, stable->CN) so
    the fusion head's dx_binary equals the conversion label; the contract's FreeSurfer
    embedding columns are left intact (they become the imaging modality).
    """
    c = pd.read_csv(_CONTRACT, low_memory=False)
    neu = pd.read_csv(_NEURO)
    ids = set(neu["subject_id"].astype(str))
    c["__sid"] = c["subject_id"].astype(str)
    sub = c[c["__sid"].isin(ids)].copy()

    conv = pd.to_numeric(sub["conversion"], errors="coerce")
    # converter -> "AD" (positive), stable -> "CN" (negative); unlabeled -> drop.
    sub = sub[conv.notna()].copy()
    conv = conv[conv.notna()]
    sub["dx"] = conv.map({1.0: "AD", 0.0: "CN"}).to_numpy()
    sub = sub.drop(columns="__sid")

    # NeuroJEPA seam frame: subject_id + emb_* only (contract.embedding_columns finds them).
    emb_cols = [col for col in neu.columns if col.startswith("emb_")]
    neu_frame = neu[["subject_id"] + emb_cols].copy()
    return sub, neu_frame


def _restamp(result_dict: dict, note: str) -> dict:
    """Re-label the artifact as the conversion target (never a dx-AD/CN run)."""
    result_dict["target"] = "conversion"
    result_dict["label_encoding"] = "MCI->AD conversion: converter=1 (dx=AD), stable=0 (dx=CN)"
    result_dict["modality_legend"] = {
        "imaging": "contract FreeSurfer structural embedding",
        "plasma": "p_tau217, gfap, nfl, apoe4, age, sex",
        "neurojepa": "frozen NeuroJEPA 768-d MRI embedding (this run's Colab output)",
    }
    result_dict["cohort_note"] = note
    return result_dict


def _fmt_view(d: dict) -> str:
    if not d:
        return "n/a"
    return (f"AUC {d.get('auc')} "
            f"[{d.get('ci_lo')}, {d.get('ci_hi')}], p_perm={d.get('p_perm')}")


def _report_md(with_n: dict, without: dict, n_conv: int, n_stable: int) -> str:
    gates = with_n.get("gates", {})
    gate_str = ", ".join(f"{k}={v}" for k, v in gates.items())
    lines = [
        "# MCI->AD Conversion — Attention-Weighted Late Fusion (+ NeuroJEPA seam)",
        "",
        f"**Cohort:** {with_n.get('n')} baseline-MCI subjects with a conversion "
        f"outcome and a complete fusion block ({n_conv} converters / {n_stable} "
        f"stable), {with_n.get('n_sites')} sites. Target = **conversion** "
        "(converter=1 / stable=0). Real ADNI; leakage-free site-disjoint OOF.",
        "",
        "## Modalities (gated leave-one-out)",
        "",
        "| Modality | Standalone AUC | Gate weight |",
        "|---|---|---|",
    ]
    mods = with_n.get("modalities", {})
    for row in with_n.get("attribution", []):
        m = row["modality"]
        lines.append(f"| {m} | {_fmt_view(mods.get(m, {}))} | {row.get('gate')} |")
    lines += [
        "",
        f"**Attention gate:** {gate_str}",
        f"**Top modality (largest leave-one-out drop):** {with_n.get('top_modality')}",
        "",
        "## Fusion result",
        "",
        f"- **3-modality (imaging + plasma + NeuroJEPA):** {_fmt_view(with_n.get('fused', {}))}",
        f"- **2-modality baseline (imaging + plasma, NO NeuroJEPA):** {_fmt_view(without.get('fused', {}))}",
        "",
        "### Does NeuroJEPA add prognostic signal?",
        "",
    ]
    # NeuroJEPA leave-one-out attribution (fused AUC drop when NeuroJEPA is removed).
    nj = next((r for r in with_n.get("attribution", [])
               if r["modality"] == "neurojepa"), None)
    if nj is not None:
        lines.append(
            f"- Leave-one-out: removing NeuroJEPA changes fused AUC by "
            f"**{nj.get('attribution_delta')}** (fused {with_n.get('fused', {}).get('auc')} "
            f"-> {nj.get('loo_fused_auc')} without it).")
    wa = with_n.get("fused", {}).get("auc")
    ba = without.get("fused", {}).get("auc")
    if wa is not None and ba is not None:
        lines.append(f"- Baseline delta: 3-modality {wa} vs 2-modality {ba} = "
                     f"**{round(float(wa) - float(ba), 4):+}** AUC.")
    lines += [
        "",
        f"**Verdict:** {with_n.get('verdict')}",
        "",
        "## Calibration (out-of-fold fused P(convert))",
        "",
        f"Brier={with_n.get('calibration', {}).get('brier')}, "
        f"ECE={with_n.get('calibration', {}).get('ece')}, "
        f"MCE={with_n.get('calibration', {}).get('mce')}",
        "",
        "## Honesty",
        "",
        with_n.get("disclaimer", ""),
        "",
        "Conversion prediction is genuinely harder than cross-sectional diagnosis; a "
        "cross-validated, permutation-significant AUC above chance is a REAL prognostic "
        "signal, and an at-chance result is reported as such.",
    ]
    return "\n".join(lines)


def main() -> int:
    sub, neu_frame = _build_slice()
    n_conv = int((sub["dx"] == "AD").sum())
    n_stable = int((sub["dx"] == "CN").sum())
    print(f"[fusion] conversion slice: {len(sub)} MCI subjects "
          f"({n_conv} converters / {n_stable} stable), "
          f"FreeSurfer D={len(contract.embedding_columns(sub))}, "
          f"NeuroJEPA D={sum(c.startswith('emb_') for c in neu_frame.columns)}", flush=True)

    with_nj = fusion.attention_fusion(sub, imaging_embedding=neu_frame)
    without_nj = fusion.attention_fusion(sub, imaging_embedding=None)

    wd = _restamp(with_nj.to_dict(),
                  "3-modality run: FreeSurfer + plasma + NeuroJEPA (seam wired).")
    bd = _restamp(without_nj.to_dict(),
                  "2-modality baseline: FreeSurfer + plasma (NeuroJEPA seam open).")

    _OUT_JSON.parent.mkdir(exist_ok=True)
    _OUT_JSON.write_text(json.dumps(
        {"with_neurojepa": wd, "baseline_without_neurojepa": bd}, indent=2))
    _OUT_MD.write_text(_report_md(wd, bd, n_conv, n_stable))

    print(f"[fusion] neurojepa_wired={with_nj.neurojepa_wired} "
          f"seam_open={with_nj.seam_open}", flush=True)
    print(f"[fusion] gates: {wd.get('gates')}", flush=True)
    print(f"[fusion] 3-modality fused: {_fmt_view(wd.get('fused', {}))}", flush=True)
    print(f"[fusion] 2-modality fused: {_fmt_view(bd.get('fused', {}))}", flush=True)
    print(f"[fusion] verdict: {wd.get('verdict')}", flush=True)
    print(f"[fusion] wrote {_OUT_JSON}", flush=True)
    print(f"[fusion] wrote {_OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
