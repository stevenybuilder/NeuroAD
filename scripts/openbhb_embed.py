#!/usr/bin/env python3
"""
Produce REAL frozen Neuro-JEPA embeddings for a multi-site OpenBHB subset.

Runs on a GPU runtime (built for Colab T4 via `colab exec`). OpenBHB quasi-raw
volumes are already MNI152-registered + skull-stripped, so they are valid
Neuro-JEPA input with no registration step. Emits
``data/real/openbhb_neurojepa_embeddings.csv`` (768-d per subject + metadata),
which the ``openbhb:neurojepa`` feeder consumes.

Compliance (do not remove):
  * Frozen inference only — NO fine-tuning (not a derivative of the gated
    CC-BY-NC-ND Neuro-JEPA weights).
  * Auth via env only: set HF_TOKEN with YOUR OWN gated grant. Never hardcode or
    commit a token. The weights are fetched to the runtime's HF cache and never
    written into this repo; only the derived embedding table is saved (and it is
    git-ignored — do not redistribute it).

Usage on Colab (weights fetched ephemerally on the GPU box, not from your laptop):
    export HF_TOKEN=hf_your_own_gated_token
    colab start --gpu t4
    colab exec --session <id> scripts/openbhb_embed.py
    colab download --session <id> /content/openbhb_neurojepa_embeddings.csv \
        data/real/openbhb_neurojepa_embeddings.csv
    colab stop
"""
import base64
import gzip
import os
import subprocess
import sys
import time

t0 = time.time()

TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
if not TOKEN:
    sys.exit("Set HF_TOKEN (your own gated Neuro-JEPA access) in the environment.")

REPO = "benoit-dufumier/openBHB"
TOP_K_SITES, N_PER_SITE, MAX_TOTAL, SEED = 6, 16, 96, 0
OUT = "/content/openbhb_neurojepa_embeddings.csv" if os.path.isdir("/content") else "openbhb_neurojepa_embeddings.csv"


def sh(c):
    print(f"$ {c}", flush=True)
    return subprocess.run(c, shell=True).returncode


# --- environment (idempotent): clone Neuro-JEPA + install deps minus torch -----
if not os.path.exists("/content/Neuro-JEPA"):
    sh("git clone -q https://github.com/NYUMedML/Neuro-JEPA /content/Neuro-JEPA")
sys.path.insert(0, "/content/Neuro-JEPA/src")
sh("grep -vE '^torch==' /content/Neuro-JEPA/requirements.txt > /content/reqs.txt")
sh("pip -q install -r /content/reqs.txt 2>&1 | tail -2")
# torchmetrics incidentally imports transformers (bert score), which forces a newer
# huggingface_hub than the repo pins. Neuro-JEPA never uses transformers -> drop it.
sh("pip -q uninstall -y transformers 2>&1 | tail -1")

import numpy as np
import pandas as pd
import torch
import nibabel as nib
from huggingface_hub import hf_hub_download
from monai import data as mdata, transforms
from monai.transforms import Lambdad

print(f"torch {torch.__version__} cuda={torch.cuda.is_available()}", flush=True)

# --- pick a balanced multi-site subset ----------------------------------------
pt = hf_hub_download(REPO, "participants.tsv", repo_type="dataset", token=TOKEN)
df = pd.read_csv(pt, sep="\t")
tr = df[df["split"] == "train"].copy()
top_sites = tr["site"].value_counts().head(TOP_K_SITES).index.tolist()
sel = (pd.concat([tr[tr["site"] == s].sample(min(N_PER_SITE, (tr["site"] == s).sum()), random_state=SEED)
                  for s in top_sites])
       .sample(frac=1, random_state=SEED).head(MAX_TOTAL).reset_index(drop=True))
print(f"selected {len(sel)} subjects across {sel['site'].nunique()} sites", flush=True)

# --- frozen backbone -----------------------------------------------------------
from neurojepa.utils.init_utils import load_backbone_from_hf
device = "cuda" if torch.cuda.is_available() else "cpu"
backbone = load_backbone_from_hf("NYUMedML/Neuro-JEPA", device=device, token=TOKEN)
backbone.eval()


def remove_nan(img):
    img[np.isnan(img)] = 0.0
    return img


trans = transforms.Compose([
    transforms.LoadImaged(keys=["image"], image_only=False),
    transforms.EnsureChannelFirstd(keys=["image"]),
    Lambdad(("image",), remove_nan),
    transforms.Orientationd(keys=["image"], axcodes="RAS"),
    transforms.Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=[5]),
    transforms.CropForegroundd(keys=["image"], source_key="image", select_fn=lambda x: x > 0.0, margin=4, allow_smaller=True),
    transforms.ResizeWithPadOrCropd(keys=["image"], spatial_size=[180, 216, 180], mode="edge"),
    transforms.Resized(keys=["image"], spatial_size=[100, 120, 100]),
    transforms.CenterSpatialCropd(keys=["image"], roi_size=[96, 108, 96], allow_missing_keys=True),
    transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0, b_max=1, clip=True),
    transforms.CastToTyped(keys=["image"], dtype=np.float32, allow_missing_keys=True),
])
torch.backends.cudnn.enabled = False


def embed(nii_path):
    ds = mdata.Dataset(data=[{"image": nii_path}], transform=trans)
    x = next(iter(mdata.DataLoader(ds, batch_size=1, num_workers=1)))["image"].to(device)
    with torch.no_grad(), torch.autocast(device if device == "cuda" else "cpu"):
        out = backbone(x)
    f = (out[0] if isinstance(out, (tuple, list)) else out).float()
    if f.dim() == 3:
        f = f.mean(dim=1)
    return f[0].cpu().numpy()


rows = []
for i, r in sel.iterrows():
    pid = str(r["participant_id"])
    fp = f"train/derivatives/sub-{pid}/ses-1/sub-{pid}_preproc-quasiraw_T1w.npy"
    try:
        npy = hf_hub_download(REPO, fp, repo_type="dataset", token=TOKEN)
        arr = np.load(npy).squeeze().astype(np.float32)
        nii = f"/content/{pid}.nii.gz" if os.path.isdir("/content") else f"{pid}.nii.gz"
        nib.save(nib.Nifti1Image(arr, np.eye(4)), nii)
        emb = embed(nii)
        os.remove(npy)
        os.remove(nii)
        row = {"participant_id": pid, "site": r["site"], "age": r["age"], "sex": r["sex"],
               "field_strength": r.get("magnetic_field_strength")}
        row.update({f"emb_{j}": float(v) for j, v in enumerate(emb)})
        rows.append(row)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(sel)}", flush=True)
    except Exception as e:
        print(f"  skip {pid}: {type(e).__name__} {str(e)[:80]}", flush=True)

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
print(f"WROTE {len(out)} x 768-d embeddings across {out['site'].nunique()} sites -> {OUT}", flush=True)

# Stream the result durably (survives a post-run runtime drop): gzip+base64 to stdout.
blob = base64.b64encode(gzip.compress(open(OUT, "rb").read())).decode()
print("===CSVGZ_START===", flush=True)
print(blob, flush=True)
print("===CSVGZ_END===", flush=True)
print(f"== DONE in {time.time() - t0:.0f}s ==", flush=True)
