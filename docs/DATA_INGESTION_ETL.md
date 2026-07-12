# Data Ingestion & ETL — the efficient, reproducible playbook

_How we ingest imaging cohorts (ADNI, OASIS, and future cohorts) into NeuroJEPA
embeddings without the slow local↔cloud round-trips. Distilled from the 590-subject
ADNI embed, the MCI-conversion pull, and the Boltz run. Secrets live in `.env`
(gitignored) — see §7; none are in this file._

---

## 0. The one golden rule

**Never process imaging on the local Mac.** The laptop has ~16–22 GB free; a single
ADNI DICOM zip unzips to ~16 GB, and `dcm2niix` + skull-strip + embed need a GPU and
~100 GB scratch. **All heavy ETL runs on Colab.** Only the tiny embeddings CSV (~9 MB)
ever comes back. Everything else in this doc follows from that.

## 1. The efficient data flow (memorize this shape)

```
  LONI/IDA  ──download──►  local ~/Downloads (zips)     [browser step is unavoidable:
 (browser,                        │                      IDA needs interactive auth]
  gated)                          │  gcloud storage cp  (bulk, ONCE)
                                  ▼
                        GCS bucket (durable, in-region)
                                  │  Colab pulls in-region (fast, resumable)
                                  ▼
                 Colab GPU runtime  (~100 GB scratch + GPU)
                 unzip → dcm2niix → skull-strip → NeuroJEPA embed
                                  │
                 stream gzip+base64 to stdout  +  push CSV to GCS
                                  ▼
                 tiny embeddings CSV → local → fuse / analyze
```

The two transfers that matter: (a) one bulk `gcloud storage cp` of the raw zips
local→GCS, and (b) Colab pulling them GCS→runtime **in-region** (much faster than
pushing 4–8 GB up through the `colab exec` channel, which is slow and drops websockets).

## 2. Anti-patterns — the "past mistakes" that made it slow

| ❌ Don't | ✅ Do |
|---|---|
| Unzip DICOM locally | Unzip on Colab (Mac can't even fit it) |
| `dcm2niix` on the CPU Mac | Convert on the Colab GPU runtime |
| Download → convert → **re-upload NIfTIs** | Upload raw zips once; convert on Colab |
| Push 4–8 GB up via `colab upload` | Put zips on GCS; Colab pulls in-region |
| Per-file transfers | Bulk zip → GCS → bulk pull |
| Re-run the whole embed on any crash | Resumable: GCS checkpoints + stdout stream |
| Ad-hoc token pasting each run | Secrets in `.env` (gitignored), sourced once |

## 3. Repeatable recipe — add an ADNI imaging cohort

**Step 1 — define the cohort & map to Image IDs (local, no GPU).**
```bash
# cohort subject_ids come from the contract (subject_id == ADNI RID)
# adni_image_manifest.csv maps subject_id(RID) ↔ IMAGEUID for every ADNI scan.
# Emit the IMAGEUID worklist (comma-separated for IDA):
PYTHONPATH=src ./.venv/bin/python - <<'PY'
import pandas as pd
from neuroad.data import loaders
df = loaders.load('adni:combat')
# ... filter to your cohort (e.g. conversion+plasma) -> subject_ids ...
im = pd.read_csv('data/real/_manifests/adni_image_manifest.csv', dtype=str)
ids = im[im.subject_id.isin(cohort_ids)].IMAGEUID.dropna().unique()
print(",".join(sorted(ids, key=int)))
PY
```

**Step 2 — IDA download (browser).** Search & Download → **Advanced Image Search** →
paste the Image IDs into the **Image ID** field (comma-separated, `123,456` form) →
`SEARCH` → **Select All** → **Add To Collection** (name it) → **Data Collections** tab →
select the collection → **1-Click Download**. Choose **As Archived (DICOM)** to match the
590's preprocessing exactly (dcm2niix → deepbet → resample); this matters because we
statistically fuse the cohorts. You get 2 zips + an `*_IDA_Metadata.zip`. No DUA
re-prompt once you've accepted it. (Notes: ~200-ish scans/GB; automatable via the
Chrome extension — Advanced Image Search has a real `Image ID` field, Simple Search does
not.)

**Step 3 — stage to GCS (bulk, once).**
```bash
set -a; source .env; set +a
gcloud storage cp ~/Downloads/<collection>*.zip           $GCS_BUCKET/<cohort>/raw/
gcloud storage cp ~/Downloads/*_IDA_Metadata.zip          $GCS_BUCKET/<cohort>/raw/
```

**Step 4 — build the crosswalk (local, no GPU).** The Colab driver needs
`crosswalk.csv` with columns **RID, PTID, IMAGEUID** (it converts each `I<IMAGEUID>`
anchor series → `ADNI_MRI/<RID>/T1.nii.gz`). Build it from two sources:
- **PTID ↔ IMAGEUID** from the metadata zip paths: `ADNI/<PTID>/<desc>/<date>/I<IMAGEUID>/…`
- **RID ↔ IMAGEUID** from `adni_image_manifest.csv` (subject_id == RID).

```bash
# see the exact join in git history; output:
#   data/real/_manifests/adni_<cohort>_crosswalk.csv   (RID,PTID,IMAGEUID)
gcloud storage cp data/real/_manifests/adni_<cohort>_crosswalk.csv $GCS_BUCKET/<cohort>/inputs/crosswalk.csv
gcloud storage cp data/real/_manifests/adni_image_manifest.csv     $GCS_BUCKET/<cohort>/inputs/manifest_full.csv
gcloud storage cp scripts/neurojepa_embed_colab.py                 $GCS_BUCKET/<cohort>/inputs/
gcloud storage cp scripts/adni_colab_dicom_to_embed.py             $GCS_BUCKET/<cohort>/inputs/
```

**Step 5 — one Colab GPU job does the rest.** Check the GPU slot is free first
(`colab status` → "No active runtime"; only ONE runtime at a time). Then start a runtime,
have it pull inputs+zips from GCS (auth via the ADC — §7), and run the one-shot driver:
```bash
colab start --gpu a100                 # note the session id; A100/L4 (T4 can OOM)
colab exec --session <id> --timeout 3h scripts/adni_colab_dicom_to_embed.py
#   → apt dcm2niix → unzip → convert exact I<IMAGEUID> → skull-strip → embed
#   → streams a durable gzip+base64 embeddings CSV to stdout (+ push to GCS)
colab stop --session <id>
```
Rebuild the CSV locally from the streamed blob (or `colab download` / pull from GCS) →
`data/real/adni_<cohort>_neurojepa_embeddings.csv`.

**Step 6 — analyze/fuse (local, no GPU).** The embeddings are keyed by `subject_id`
(== RID), so they join the contract directly, e.g.
`attention_fusion(df, imaging_embedding=<frame>)`.

## 4. Colab gotchas (hard-won — read before any GPU job)

- **One runtime at a time.** Starting a second **reclaims** the first — `colab status`
  before starting; skip if a job is running elsewhere.
- **`colab exec script.py -- args` does NOT forward argv** (the script runs in a Jupyter
  kernel that only sees its own `-f kernel.json`). **Bake defaults into the script**
  (`HERO_*`, cohort constants); don't rely on CLI args. Also: don't edit a script right
  after launching a background `colab exec` — it reads the file async → race.
- **STREAM stdout, never capture.** A long job with silent/captured stdout starves the
  exec websocket and it drops (`failed to read frame header: EOF`). Stream, and
  `emit_durable()` (gzip+base64 to stdout) after every unit so a drop still yields
  completed work.
- **Detached children:** launch with `start_new_session=True`, else the child gets
  SIGINT when the `colab exec` cell completes.
- **Runtimes get reclaimed at ~10–22 min** even while busy (free tier; Pro+ A100 is
  steadier but budget for it). → the resumable pattern below is mandatory for long jobs.
- **Keep Colab's `numpy` 2.0.2** — do NOT pin `numpy<2` (trips pip's resolver on Colab).
- **`deepbet --no-deps` misses** `fill_voids` / `connected-components-3d` / `fastremap` —
  install them explicitly or skull-strip fails mid-run.
- **Boltz:** pass `--no_kernels` (`cuequivariance_torch` isn't on Colab's build; the
  pure-PyTorch path is correct, ~2–3× slower).
- **venv quirk:** use `./.venv/bin/python -m pip` (stale shebang after the project rename).

## 5. The resumable pattern (for jobs that outlive a runtime)

Used for the 590 embed. Each unit (NIfTI / partial CSV) is checkpointed to GCS the moment
it's produced; a local **auto-restart loop** re-launches `colab exec` on every death, and
a ~45 s keepalive poll holds the session. Cumulative progress converges over cycles even
though any single runtime dies early. The one-shot `adni_colab_dicom_to_embed.py` wraps the
whole chain in a single exec so a preemption can't strand half-converted data between steps.

## 6. The ID / crosswalk model (why joins "just work")

- **`subject_id == ADNI RID` everywhere** — contract, manifest, and embeddings all key on
  the RID. (That's why the AD/CN 590 (RIDs 6001, 6005…) and the MCI-conversion cohort
  (RIDs 6002, 6033…) are disjoint: adjacent enrollment, genuinely different subjects.)
- **PTID format:** `<site>_S_<RID>` — e.g. `000_S_0000` → RID `0000`.
- **Metadata zip paths** give PTID↔IMAGEUID: `ADNI/<PTID>/<desc>/<date>/I<IMAGEUID>/…`.
- **Image manifest** gives RID↔IMAGEUID (+ the plasma/biomarker panel).
- The **crosswalk (RID, PTID, IMAGEUID)** ties them together and drives the *exact*
  anchor-series conversion, so we never fabricate or mis-pair a scan.

## 7. Secrets & GCS (reproducible, never committed)

All secrets are in **`.env` (gitignored)** — load once with `set -a; source .env; set +a`:
- `HF_TOKEN` — gated NeuroJEPA weights (CC-BY-NC-ND; **frozen inference only**, never
  fine-tune or redistribute; see `docs/HF_ACCESS.md`).
- `GCS_BUCKET` — `gs://neuroad-adni-project-flash-490419` (us-west1).
- `GOOGLE_APPLICATION_CREDENTIALS` — ADC path; **upload to the runtime as
  `/content/adc.json`** so Colab can pull/push GCS (service-account keys are
  org-policy-blocked).
- `RCLONE_GDRIVE_TOKEN` — optional output backup; **GCS already covers durability**, so
  rclone/Drive is usually unnecessary.

**Never commit:** weights (`*.safetensors`/`*.pt`/`*.ckpt`), tokens (`hf_token.txt`,
`*token*.txt`, `.env`), or embedding tables (`*embeddings*.csv`) — all gitignored. Result
*numbers* in `reports/` are fine to publish.

**GCS layout:**
```
gs://neuroad-adni-project-flash-490419/
  adni/nifti/<rid>/T1.nii.gz            # 590 AD/CN NIfTI (durable)
  adni/adni_neurojepa_embeddings.csv    # final 590 embeddings
  <cohort>/raw/*.zip                    # IDA download (zips + metadata)
  <cohort>/inputs/{crosswalk,manifest_full}.csv + embed scripts
```

## 8. When NOT to do all this

The referee runs fully on open data (`openbhb`, `oasis`) and `synthetic` cohorts with **no
weights and no GPU**. NeuroJEPA imaging embeddings are an *optional* substrate upgrade.
Only run this pipeline when a claim genuinely needs cross-cohort imaging — and remember the
power analysis: imaging adds ~+0.012 AUC over plasma p-tau217 and needs ~650 converters to
prove, so "more scans" only helps the specific arms that are actually thin.
