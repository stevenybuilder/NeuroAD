#!/usr/bin/env python3
"""
ONE-SHOT Colab driver: raw ADNI DICOM zips -> NeuroJEPA 768-d embeddings, entirely
on the Colab runtime (T4). Runs the WHOLE chain in a single `colab exec` so a
free-tier preemption can't strand half-converted data between steps.

Why on Colab: the local Mac has ~16GB free — too little to even unzip the ~16GB of
DICOM, let alone convert + embed. The Colab runtime has ~100GB scratch + a GPU, and
the embedding step needs the GPU anyway. Only the tiny embeddings CSV comes back
(streamed durably to stdout).

Chain:
  1. apt-install dcm2niix on the runtime.
  2. Unzip the uploaded IDA DICOM zip(s) (skips the *_Metadata.zip).
  3. For each subject in the PTID<->RID<->IMAGEUID crosswalk, convert the EXACT
     I<IMAGEUID> anchor series -> /content/ADNI_MRI/<RID>/T1.nii.gz.
  4. Build a manifest (relative image_path + biomarker panel) for what converted.
  5. Hand off to scripts/neurojepa_embed_colab.py --dataset adni --skull-strip
     --fast-resample, inheriting stdout so its durable gzip+base64 checkpoints flow
     straight back into your local colab-exec log.

Upload these to the runtime first (see the RUNBOOK printed at the end of this file):
  - both dataset zips (adni t1 mprage n=590*.zip)
  - data/real/_manifests/adni_ptid_rid_crosswalk.csv   -> /content/crosswalk.csv
  - data/real/_manifests/adni_image_manifest.csv       -> /content/manifest_full.csv
  - scripts/neurojepa_embed_colab.py                   -> /content/neurojepa_embed_colab.py
  - hf_token.txt  (raw 'hf_...')                        -> /content/hf_token.txt

Then:  colab exec --session <id> --timeout 3h scripts/adni_colab_dicom_to_embed.py
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
import tempfile

import pandas as pd

CONTENT = os.environ.get("ADNI_CONTENT", "/content")
DICOM = f"{CONTENT}/ADNI_dicom"
OUT_MRI = f"{CONTENT}/ADNI_MRI"
XWALK = f"{CONTENT}/crosswalk.csv"
MANIFEST_FULL = f"{CONTENT}/manifest_full.csv"
MANIFEST_READY = f"{CONTENT}/adni_manifest_ready.csv"
EMBED = f"{CONTENT}/neurojepa_embed_colab.py"
OUT_CSV = f"{CONTENT}/adni_neurojepa_embeddings.csv"


def sh(cmd, **kw):
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, **kw)


def install_dcm2niix():
    if shutil.which("dcm2niix"):
        return
    sh("apt-get -qq update", check=False)
    sh("apt-get -qq install -y dcm2niix", check=False)
    if shutil.which("dcm2niix") is None:
        # pip fallback ships a manylinux binary
        sh(f"{sys.executable} -m pip install -q dcm2niix", check=False)


def unzip_all():
    os.makedirs(DICOM, exist_ok=True)
    zips = [z for z in glob.glob(f"{CONTENT}/*.zip") if "Metadata" not in os.path.basename(z)]
    if not zips:
        sys.exit(f"No dataset zips found in {CONTENT} (upload the IDA zips first).")
    for z in zips:
        print(f"[unzip] {os.path.basename(z)}", flush=True)
        sh(f"unzip -q -o '{z}' -d '{DICOM}'", check=True)


def _find_series(imageuid, ptid):
    if pd.notna(imageuid):
        for h in glob.glob(f"{DICOM}/**/I{int(imageuid)}", recursive=True):
            if os.path.isdir(h) and glob.glob(f"{h}/*.dcm"):
                return h
    # fallback: any MPRAGE/IR-FSPGR series under this PTID
    for desc in glob.glob(f"{DICOM}/**/{ptid}/*", recursive=True):
        u = os.path.basename(desc).upper()
        if os.path.isdir(desc) and ("MPRAGE" in u or "FSPGR" in u):
            leaves = [d for d, _, fs in os.walk(desc) if any(f.endswith(".dcm") for f in fs)]
            if leaves:
                return leaves[0]
    return None


def convert():
    xw = pd.read_csv(XWALK)
    conv = miss = exact = fallback = 0
    for _, r in xw.iterrows():
        rid = int(r["RID"])
        dest = f"{OUT_MRI}/{rid}"
        os.makedirs(dest, exist_ok=True)
        if os.path.exists(f"{dest}/T1.nii.gz"):
            conv += 1
            continue
        series = _find_series(r.get("IMAGEUID"), str(r["PTID"]))
        if series is None:
            miss += 1
            continue
        is_exact = pd.notna(r.get("IMAGEUID")) and os.path.basename(series) == f"I{int(r['IMAGEUID'])}"
        exact += int(is_exact); fallback += int(not is_exact)
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["dcm2niix", "-z", "y", "-f", "t1", "-o", tmp, series],
                           capture_output=True, text=True)
            niis = sorted(glob.glob(f"{tmp}/*.nii.gz"), key=os.path.getsize, reverse=True)
            if niis:
                shutil.copy2(niis[0], f"{dest}/T1.nii.gz")
                conv += 1
            else:
                miss += 1
    print(f"[convert] converted={conv} missing={miss} "
          f"(exact-IMAGEUID={exact} fell-back={fallback})", flush=True)
    return conv


def build_manifest():
    xw = pd.read_csv(XWALK)
    rids = {str(int(x)) for x in xw["RID"].dropna()}
    m = pd.read_csv(MANIFEST_FULL)
    m = m[m["subject_id"].astype(str).isin(rids)].copy()
    m["image_path"] = m["subject_id"].apply(lambda s: f"ADNI_MRI/{int(s)}/T1.nii.gz")
    m = m[m["subject_id"].apply(lambda s: os.path.exists(f"{OUT_MRI}/{int(s)}/T1.nii.gz"))]
    m.to_csv(MANIFEST_READY, index=False)
    print(f"[manifest] ready rows: {len(m)} -> {MANIFEST_READY}", flush=True)
    return len(m)


def embed():
    os.chdir(CONTENT)
    cmd = [sys.executable, EMBED,
           "--dataset", "adni", "--manifest", MANIFEST_READY,
           "--id-col", "subject_id", "--image-col", "image_path",
           "--skull-strip", "--fast-resample",
           "--out", OUT_CSV, "--checkpoint-every", "25", "--resume"]
    print("[embed] " + " ".join(cmd), flush=True)
    # inherit stdout/stderr so the embed script's durable ===CSVGZ=== markers land
    # in the local colab-exec log even if the websocket drops at the very end.
    subprocess.run(cmd, check=True)


GDRIVE_FOLDER = "1Qd754tBNX-CfkjYG_fztdjVszIbdM8Jh"


def push_result():
    """Copy the embeddings CSV to Drive so the result survives even if the runtime
    is reclaimed right after — durable off-runtime backstop."""
    if os.path.exists(OUT_CSV) and shutil.which("rclone") and os.path.exists("/content/rclone.conf"):
        subprocess.run(
            f'rclone --config /content/rclone.conf copy "{OUT_CSV}" gdrive: '
            f'--drive-root-folder-id {GDRIVE_FOLDER}', shell=True)
        print(f"[push] {os.path.basename(OUT_CSV)} copied to Drive", flush=True)


def main():
    install_dcm2niix()
    if shutil.which("dcm2niix") is None:
        sys.exit("dcm2niix unavailable after apt + pip fallback.")
    unzip_all()
    if convert() == 0:
        sys.exit("No volumes converted — check the zip layout / crosswalk.")
    if build_manifest() == 0:
        sys.exit("Manifest empty — no converted NIfTI matched the crosswalk RIDs.")
    embed()
    push_result()
    print("[done] embeddings streamed above and pushed to Drive.", flush=True)


if __name__ == "__main__":
    main()
