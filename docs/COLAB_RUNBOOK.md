# Colab Runbook — OASIS-1 embedding (Lane B, ~4 CU on T4)

Grows the labeled cohort **61 → 235** by embedding the 174 not-yet-embedded
CDR-labeled OASIS-1 subjects. Frozen Neuro-JEPA inference only. **~1 CU expected;
budget 4 CU with debug. Use a T4 — L4/A100/H100 are wasteful for frozen inference.**

## 0. Prereqs (once)
- `HF_TOKEN` for gated `NYUMedML/Neuro-JEPA` weights — **env var only, never commit it.**
- `colab` CLI already authenticated (`colab quota` works → ~1799 CU, Pro+).

## 1. Build the gap manifest (local, no GPU)
```bash
PYTHONPATH=src ./.venv/bin/python scripts/fetch_oasis1_gap.py
# -> data/real/_manifests/oasis1_gap_manifest.csv  (174 rows: participant_id,image_path,dx,cdr,age,sex,site,scanner)
```

## 2. Get the raw volumes (OASIS-1, open access)
OASIS-1 is openly downloadable (no DUA). For each `participant_id` (e.g. `OAS1_0001_MR1`)
fetch the **T88, skull-stripped, gain-field-corrected** volume — already registered +
masked, so **no FreeSurfer / no MNI reg needed**:
```
<subject>/PROCESSED/MPRAGE/T88_111/<subject>_mpr_n4_anon_111_t88_masked_gfc.img (+ .hdr)
```
Mirror them under a local `OASIS1_RAW/` matching the manifest's `image_path` column
(edit the column if your layout differs). `.img/.hdr` (Analyze) is read natively by nibabel.
> Tip: download on a **CPU** Colab runtime (or locally) *before* attaching the GPU — don't burn GPU minutes on I/O.

## 3. Embed on a T4
```bash
colab start --gpu t4                       # note the session id; T4 only
colab upload OASIS1_RAW               OASIS1_RAW
colab upload data/real/_manifests/oasis1_gap_manifest.csv  manifest.csv
colab upload scripts/neurojepa_embed.py    neurojepa_embed.py
colab exec -c "import os; os.environ['HF_TOKEN']='hf_xxx'"   # or set via runtime secret
colab exec neurojepa_embed.py --timeout 30m -- \
    --manifest manifest.csv --out oasis1_gap_embeddings.csv \
    --image-col image_path --id-col participant_id
colab download oasis1_gap_embeddings.csv data/real/oasis1_gap_embeddings.csv
colab stop                                  # release the instant the CSV lands
```
Sanity: `oasis1_gap_embeddings.csv` should be 174 rows × (768 emb + metadata).

## 4. Concatenate → 235
```bash
PYTHONPATH=src ./.venv/bin/python - <<'PY'
import pandas as pd
a=pd.read_csv("data/real/oasis1_neurojepa_embeddings.csv")   # 61
b=pd.read_csv("data/real/oasis1_gap_embeddings.csv")          # 174
out=pd.concat([a,b],ignore_index=True).drop_duplicates("participant_id")
out.to_csv("data/real/oasis1_neurojepa_embeddings.csv",index=False)
from neuroad import contract; contract.validate_table(out)   # must pass
print("total:",len(out),"dx:",out["dx"].value_counts().to_dict())
PY
```
When this file grows to ~235, the harness's real-data Act-II beat has the power to
surface a survivor (see `docs/STAGE2_BLUEPRINT.md` §Power). Until then, the n=61 file
runs the honest method-as-result fallback.

## Guardrails
- **Never commit** `HF_TOKEN`, the raw volumes, or the encoder weights (CC-BY-NC-ND).
  Only the derived `*_embeddings.csv` (AUC-grade numbers) is repo-safe.
- `colab stop` immediately after download — no warm idle instances.
- If the download stalls, **the demo still ships on n=61** — don't let the stretch block the build.
