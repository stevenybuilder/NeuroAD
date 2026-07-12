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

## 2. Raw volumes — fetched ON the runtime (no local download)
`scripts/neurojepa_embed_colab.py` is self-contained: it streams the 12 open-access
OASIS-1 disc tarballs (`download.nrg.wustl.edu`, no login/DUA) **on the GPU runtime**,
extracting ONLY the `*_t88_masked_gfc.img/.hdr` volumes (already T88-registered +
skull-stripped → no FreeSurfer / no MNI reg). ~18 GB streams through; peak disk is
tiny (~2 MB/subject kept, the rest discarded in-flight). Raw MRI never leaves the
runtime and never touches git. **You do not download or upload OASIS1_RAW yourself.**

## 3. Embed on a T4 (one self-contained job)
Two CLI gotchas this flow works around, both real:
- `colab exec <file.py>` runs the file as a notebook cell and **does NOT forward argv**
  — so we invoke via an inline wrapper that sets `sys.argv` then `runpy.run_path`s the
  uploaded script (and prints real tracebacks; Colab's IPython formatter is buggy).
- `colab exec -c` env vars **don't persist** across exec calls — so the token rides in
  as an uploaded file the script reads (`--token-file`, git-ignored, wiped on `stop`).

```bash
colab start --gpu t4                        # note the session id; T4 only
SID=<session-id>
# stage inputs (token file lives OUTSIDE the repo — never commit it):
grep '^HF_TOKEN=' .env | cut -d= -f2- > /tmp/hf_token.txt
colab upload --session $SID /tmp/hf_token.txt hf_token.txt
colab upload --session $SID data/real/_manifests/oasis1_gap_manifest.csv manifest.csv
colab upload --session $SID scripts/neurojepa_embed_colab.py neurojepa_embed_colab.py
# smoke first (disc3 has the earliest gap subject; --limit counts SUCCESSFUL embeds):
colab exec --session $SID --timeout 15m -c "import sys,runpy; sys.argv=['x','--manifest','manifest.csv','--out','smoke.csv','--discs','3','--limit','2']; runpy.run_path('neurojepa_embed_colab.py',run_name='__main__')"
# full run (all 12 discs, ~174 subjects; deps already installed by the smoke):
colab exec --session $SID --timeout 40m -c "import sys,runpy; sys.argv=['x','--manifest','manifest.csv','--out','oasis1_gap_embeddings.csv','--discs','1-12','--skip-install']; runpy.run_path('neurojepa_embed_colab.py',run_name='__main__')"
colab download --session $SID oasis1_gap_embeddings.csv data/real/oasis1_gap_embeddings.csv
colab stop --session $SID                    # release the instant the CSV lands
```
The script also streams a durable gzip+base64 copy of the CSV to stdout between
`===CSVGZ_START===`/`===CSVGZ_END===`, so a websocket drop can't lose the result —
reconstruct locally with `gzip.decompress(base64.b64decode(blob))` if `download` fails.
Sanity: `oasis1_gap_embeddings.csv` should be ~174 rows × (768 emb + metadata).

## 4. Concatenate → ~235 and regenerate the AD-signal report
```bash
PYTHONPATH=src ./.venv/bin/python - <<'PY'
import pandas as pd
a=pd.read_csv("data/real/oasis1_neurojepa_embeddings.csv")   # 61
b=pd.read_csv("data/real/oasis1_gap_embeddings.csv")          # ~174
out=pd.concat([a,b],ignore_index=True).drop_duplicates("participant_id")
out.to_csv("data/real/oasis1_neurojepa_embeddings.csv",index=False)
print("total:",len(out),"dx:",out["dx"].value_counts().to_dict())
PY
# regenerate the headline number at the new n (bootstrap CI + permutation p):
PYTHONPATH=src ./.venv/bin/python scripts/run_oasis_neurojepa.py
```
`run_oasis_neurojepa.py` recomputes AD-vs-CN with the referee's own leakage-free
probe (auto PCA-10 inside CV) → `reports/oasis_neurojepa_ad.json`. When the table
reaches ~235, the CDR≥1 contrast has the n to carry a tight CI (see
`docs/STAGE2_BLUEPRINT.md` §Power). Until then, the n=61 file is the honest fallback.

## Guardrails
- **Never commit** `HF_TOKEN`, the token file, the raw volumes, or the encoder weights
  (CC-BY-NC-ND). `.gitignore` + `.githooks/pre-commit` block all of them; only the
  derived `*_embeddings.csv` (AUC-grade numbers) is repo-safe — and even that is
  git-ignored here, kept local, with just the numbers published to `reports/`.
- `colab stop` immediately after download — no warm idle instances.
- If the fetch stalls, **the demo still ships on n=61** — don't let the stretch block the build.
