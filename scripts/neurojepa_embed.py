#!/usr/bin/env python3
"""
Frozen Neuro-JEPA embedding extractor (encoder-agnostic representation path).

Runs the released Neuro-JEPA ViT-MoE backbone FROZEN (no fine-tuning, no gradient)
over 3D T1w MRI volumes and emits a per-subject 768-d embedding table that plugs
directly into the NeuroAD referee's embedding contract (emb_0..emb_767 + metadata).

Design constraints (license + hackathon compliance):
  * Non-derivative: inference only. We never train, fine-tune, or adapt the weights.
  * The weights are gated CC-BY-NC-ND. This script fetches an EPHEMERAL copy from
    HuggingFace at runtime via HF_TOKEN and never writes them to the repo. Only the
    small derived embedding vectors are persisted. Do not commit weights or a large
    embedding dump.

Intended host: a GPU runtime (e.g. Colab T4 via `colab exec`). The local CPU/Py3.14
box cannot run torch; that is fine — this is deliberately off the referee's critical
path, and the referee also runs on weight-free structural features.

Usage (on a GPU runtime):
    export HF_TOKEN=hf_xxx
    python neurojepa_embed.py --manifest subjects.csv --out embeddings.csv
        --image-col image_path --id-col subject_id

`manifest.csv` needs an image-path column pointing at .nii/.nii.gz volumes plus any
metadata columns (age, sex, dx, scanner, ...) which are passed straight through so
the referee can point its single reused probe at outcome / scanner / biomarker.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd


def build_transform(roi_size=(96, 108, 96)):
    """Exact preprocessing pipeline from the Neuro-JEPA feature-extraction example."""
    from monai import transforms
    from monai.transforms import Lambdad

    def remove_nan(img):
        img[np.isnan(img)] = 0.0
        return img

    return transforms.Compose([
        transforms.LoadImaged(keys=["image"], image_only=False),
        transforms.EnsureChannelFirstd(keys=["image"]),
        Lambdad(("image",), remove_nan),
        transforms.Orientationd(keys=["image"], axcodes="RAS"),
        transforms.Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=[5]),
        transforms.CropForegroundd(keys=["image"], source_key="image",
                                   select_fn=lambda x: x > 0.0, margin=4, allow_smaller=True),
        transforms.ResizeWithPadOrCropd(keys=["image"], spatial_size=[180, 216, 180], mode="edge"),
        transforms.Resized(keys=["image"], spatial_size=[100, 120, 100]),
        transforms.CenterSpatialCropd(keys=["image"], roi_size=list(roi_size), allow_missing_keys=True),
        transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5,
                                                   b_min=0, b_max=1, clip=True),
        transforms.CastToTyped(keys=["image"], dtype=np.float32, allow_missing_keys=True),
    ])


def load_backbone(device, token):
    from neurojepa.utils.init_utils import load_backbone_from_hf
    backbone = load_backbone_from_hf("NYUMedML/Neuro-JEPA", device=device, token=token)
    backbone.eval()
    return backbone


def embed_volume(backbone, x, device):
    """x: (1,1,96,108,96) tensor -> (768,) pooled embedding."""
    import torch
    torch.backends.cudnn.enabled = False
    with torch.no_grad(), torch.autocast(device if device == "cuda" else "cpu"):
        out = backbone(x.to(device))
    feats = out[0] if isinstance(out, (tuple, list)) else out
    feats = feats.float()
    if feats.dim() == 3:            # (B, tokens, dim) -> mean-pool tokens
        feats = feats.mean(dim=1)
    return feats[0].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="CSV with an image-path column + metadata")
    ap.add_argument("--out", default="embeddings.csv")
    ap.add_argument("--image-col", default="image_path")
    ap.add_argument("--id-col", default="subject_id")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        sys.exit("Set HF_TOKEN (gated Neuro-JEPA access) in the environment.")

    import torch
    from monai import data
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = pd.read_csv(args.manifest)
    if args.image_col not in df.columns:
        sys.exit(f"manifest missing image column '{args.image_col}'")

    backbone = load_backbone(device, token)
    trans = build_transform()

    rows = []
    meta_cols = [c for c in df.columns if c != args.image_col]
    for i, r in df.iterrows():
        ds = data.Dataset(data=[{"image": r[args.image_col]}], transform=trans)
        loader = data.DataLoader(ds, batch_size=1, num_workers=1)
        x = next(iter(loader))["image"]
        emb = embed_volume(backbone, x, device)
        row = {c: r[c] for c in meta_cols}
        row.update({f"emb_{j}": float(v) for j, v in enumerate(emb)})
        rows.append(row)
        print(f"[{i + 1}/{len(df)}] {r.get(args.id_col, i)} -> emb dim {emb.shape[0]}", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}: {len(out)} subjects x {emb.shape[0]}-d embeddings", flush=True)


if __name__ == "__main__":
    main()
