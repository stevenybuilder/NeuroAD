#!/usr/bin/env python3
"""
train_neurojepa_decoder.py — train L2's NeuroJEPA-conditioned 3D U-Net decoder,
with GCS checkpoint/resume so it SURVIVES runtime death.

Why resumable: the Colab CLI runtimes here are reclaimed at ~10-22 min even while
busy, and training is ~1-2 h. This loop checkpoints model+optimizer+epoch to GCS
every ``--ckpt-every-sec`` seconds and, on start, RESUMES from the latest GCS
checkpoint. So the same command re-run after a death continues instead of
restarting — turning an unstable ~20-min runtime into usable training time (at the
cost of ~15-25 resume cycles for a full run). On a stable GPU (GCE quota bump /
Databricks) it just runs straight through.

Data contract (one dir of per-subject .npz, produced by the prep step): each
``<sid>.npz`` holds ``mri`` [1,D,H,W] float32 (skull-stripped, resampled to
--size^3, intensity-normalized), ``jepa`` [768] float32 (the L1 embedding we
already extract), ``label`` [D,H,W] int64 (FastSurfer aseg remapped to LABELS).
The prep step (scripts, TODO) writes these to gs://<bucket>/decoder_data/.

This file is the harness; it needs a GPU + the prepped data to actually run.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/content/decoder_data",
                    help="local dir of <sid>.npz (pulled from GCS decoder_data/)")
    ap.add_argument("--gcs-ckpt", default="decoder_ckpt/latest.pt",
                    help="GCS object path for the resumable checkpoint")
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ckpt-every-sec", type=int, default=180)
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()

    import numpy as np
    import torch
    from torch.utils.data import Dataset, DataLoader
    from neuroad.integrations.neurojepa_decoder import (
        NeuroJEPADecoder, dice_ce_loss, N_CLASSES)
    from neuroad.integrations import gcs_store as gcs

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train] device={dev} data={args.data_dir}", flush=True)

    class NPZSet(Dataset):
        def __init__(self, files):
            self.files = files
        def __len__(self):
            return len(self.files)
        def __getitem__(self, i):
            d = np.load(self.files[i])
            return (torch.from_numpy(d["mri"]).float(),
                    torch.from_numpy(d["jepa"]).float(),
                    torch.from_numpy(d["label"]).long())

    files = sorted(os.path.join(args.data_dir, f)
                   for f in os.listdir(args.data_dir) if f.endswith(".npz"))
    if not files:
        sys.exit(f"[train] no .npz in {args.data_dir} — run the data-prep step first.")
    nval = max(1, int(len(files) * args.val_frac))
    tr, va = files[nval:], files[:nval]
    tl = DataLoader(NPZSet(tr), batch_size=args.batch, shuffle=True, num_workers=2)
    vl = DataLoader(NPZSet(va), batch_size=1, num_workers=1)

    model = NeuroJEPADecoder().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_epoch = 0

    # ---- resume from GCS if a checkpoint exists ----
    local_ckpt = "/tmp/_resume.pt"
    if gcs.try_download(args.gcs_ckpt, local_ckpt):
        ck = torch.load(local_ckpt, map_location=dev)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_epoch = ck["epoch"] + 1
        print(f"[train] RESUMED from {gcs.uri(args.gcs_ckpt)} @ epoch {start_epoch}",
              flush=True)
    else:
        print("[train] no checkpoint — fresh start", flush=True)

    def save_ckpt(epoch):
        buf = io.BytesIO()
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": epoch}, buf)
        open(local_ckpt, "wb").write(buf.getvalue())
        gcs.upload(local_ckpt, args.gcs_ckpt)
        print(f"[ckpt] epoch {epoch} -> {gcs.uri(args.gcs_ckpt)}", flush=True)

    last_ckpt = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train(); tot = 0.0
        for mri, jepa, label in tl:
            mri, jepa, label = mri.to(dev), jepa.to(dev), label.to(dev)
            loss = dice_ce_loss(model(mri, jepa), label)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
            if time.time() - last_ckpt > args.ckpt_every_sec:
                save_ckpt(epoch); last_ckpt = time.time()
        print(f"[epoch {epoch}] train_loss={tot/len(tl):.4f}", flush=True)
        save_ckpt(epoch); last_ckpt = time.time()

    print("[train] done.", flush=True)


if __name__ == "__main__":
    main()
