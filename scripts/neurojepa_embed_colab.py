#!/usr/bin/env python3
"""
Self-contained Colab T4 job: OASIS-1 61 -> ~234 NeuroJEPA embedding.

Runs ENTIRELY on the ephemeral GPU runtime, so a websocket drop or a preempted
runtime never leaves a half-done mess in the repo:

  1. streams the 12 open-access OASIS-1 cross-sectional disc tarballs, extracting
     ONLY the T88 skull-stripped, gain-field-corrected `*_masked_gfc.img/.hdr`
     volumes (raw MRI stays on the runtime, never comes home, never hits git),
  2. fetches the gated Neuro-JEPA ViT-MoE weights EPHEMERALLY via HF_TOKEN
     (frozen inference only; weights never written to the repo -> CC-BY-NC-ND safe),
  3. runs FROZEN inference -> one 768-d embedding per subject,
  4. writes `--out` incrementally AND streams a durable gzip+base64 copy of it to
     stdout between ===CSVGZ_START===/===CSVGZ_END=== markers, so the result lands
     in the local exec log the instant it is computed.

Only the small derived embedding table is meant to leave the runtime. Do NOT
download or commit the weights or the raw volumes (see gated-weights-compliance).

Why OASIS-1 `*_masked_gfc`: it ships already T88-registered + skull-stripped, which
is exactly what the Neuro-JEPA MONAI transform expects -> no FreeSurfer, no MNI reg.

Usage (see docs/COLAB_RUNBOOK.md):
    colab start --gpu t4                                   # note the session id
    colab upload data/real/_manifests/oasis1_gap_manifest.csv manifest.csv
    # HF_TOKEN must be set in the runtime env (never commit it):
    colab exec --session <id> -c "import os; assert os.environ.get('HF_TOKEN')"
    # smoke test (disc1 only, 2 subjects) then the full run:
    colab exec --session <id> --timeout 15m scripts/neurojepa_embed_colab.py -- --manifest manifest.csv --discs 1 --limit 2
    colab exec --session <id> --timeout 40m scripts/neurojepa_embed_colab.py -- --manifest manifest.csv
    # rebuild the CSV locally from the durable blob in the exec log, or:
    colab download --session <id> oasis1_gap_embeddings.csv data/real/oasis1_gap_embeddings.csv
    colab stop --session <id>

ADNI (--dataset adni): raw ADNI MRI is DUA-gated, so there is NO tarball fetch —
the user downloads T1w NIfTIs from ida.loni.usc.edu with their own LONI account
(Download -> Image Collections -> Advanced Search -> MRI/MPRAGE -> DICOM ->
dcm2niix), stages them at the paths in `scripts/build_adni_image_manifest.py`'s
manifest, and this script skull-strips (deepbet) + fast-resamples them exactly
like OASIS-2 raw, then embeds. The output CSV carries the manifest's biomarker
columns through, so it lands already joined to plasma p-tau217/GFAP/NfL + amyloid:
    ./.venv/bin/python scripts/build_adni_image_manifest.py \
        --image-root /data/ADNI_MRI          # after the IDA download
    python scripts/neurojepa_embed_colab.py -- \
        --dataset adni --skull-strip --fast-resample \
        --manifest data/real/_manifests/adni_image_manifest.csv \
        --id-col subject_id --out adni_neurojepa_embeddings.csv
"""
import argparse
import base64
import gzip
import os
import subprocess
import sys

OASIS_DISC_URL = "https://download.nrg.wustl.edu/data/oasis_cross-sectional_disc{d}.tar.gz"
OASIS2_PART_URL = "https://download.nrg.wustl.edu/data/OAS2_RAW_PART{p}.tar.gz"
HF_REPO = "NYUMedML/Neuro-JEPA"
ROI = (96, 108, 96)


def _resolve_token(token_file: str):
    """HF token from env first, else a plain-text file (raw token or HF_TOKEN=... line).

    The file lives only on the ephemeral runtime (wiped on `colab stop`); it is never
    written to the repo and the loader passes the token straight to hf_hub_download.
    """
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if tok:
        return tok.strip()
    if token_file and os.path.exists(token_file):
        for line in open(token_file):
            line = line.strip()
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
            if line.startswith("hf_"):
                return line
    return None


def sh(cmd: str, check: bool = True):
    print(f"    $ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check)


def install_deps():
    """Keep Colab's CUDA torch; install only the frozen-inference dependency set.

    The repo pins torch==2.10.0 + torchmetrics, which would clobber Colab's working
    CUDA build and drag in the torchmetrics->bert-score->transformers huggingface_hub
    conflict. So: install curated deps, break the transformers chain, then install the
    package itself with --no-deps.
    """
    print("[deps] installing (keeping Colab torch) ...", flush=True)
    sh("pip install -q monai==1.5.1 nibabel einops timm safetensors omegaconf "
       "pyyaml termcolor tqdm 'scikit-image>=0.22' torchmetrics 'huggingface_hub>=0.30'")
    # torchmetrics lazily imports transformers (bert-score); we never use it -> remove
    # it so it can't demand a newer huggingface_hub than we install.
    sh("pip uninstall -y -q transformers", check=False)
    sh("pip install -q --no-deps 'git+https://github.com/NYUMedML/Neuro-JEPA.git'")
    print("[deps] done.", flush=True)


def fetch_volumes(discs, raw_dir):
    """Stream each disc tarball through tar, extracting ONLY masked_gfc .img/.hdr.

    tar reads from the pipe and writes just the matching members; the ~1.4 GB/disc
    of everything else is discarded in-flight, so peak disk stays tiny (~2 MB/subject).
    --strip-components=1 drops the leading `discN/` so paths match the manifest's
    `OASIS1_RAW/<pid>/PROCESSED/MPRAGE/T88_111/...` layout.
    """
    os.makedirs(raw_dir, exist_ok=True)
    procs = []
    for d in discs:
        url = OASIS_DISC_URL.format(d=d)
        cmd = (
            f"curl -sfL --retry 3 --retry-delay 2 '{url}' | "
            f"tar -xz --strip-components=1 -C '{raw_dir}' --wildcards "
            f"'*t88_masked_gfc.img' '*t88_masked_gfc.hdr'"
        )
        print(f"[fetch] disc{d} -> streaming, extracting masked_gfc only", flush=True)
        procs.append((d, subprocess.Popen(cmd, shell=True,
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)))
    for d, p in procs:
        rc = p.wait()
        print(f"[fetch] disc{d} done (rc={rc})", flush=True)
    got = sum(1 for root, _, files in os.walk(raw_dir) for f in files if f.endswith("_masked_gfc.img"))
    print(f"[fetch] extracted {got} masked_gfc volumes into {raw_dir}", flush=True)


def fetch_oasis2(raw_dir="OAS2_RAW"):
    """Stream the 2 OASIS-2 raw parts, extracting only each session's mpr-1 volume.

    OASIS-2 ships RAW (non-skull-stripped) Analyze volumes as `OAS2_RAW_PART{1,2}/
    <session>/RAW/mpr-1.nifti.img(.hdr)`. --strip-components=1 drops the part prefix so
    paths match the manifest's `OAS2_RAW/<session>/RAW/mpr-1.nifti.img`. Optional
    skull-stripping (see --skull-strip) is applied per-volume at embed time.
    """
    os.makedirs(raw_dir, exist_ok=True)
    procs = []
    for p_ in (1, 2):
        url = OASIS2_PART_URL.format(p=p_)
        cmd = (
            f"curl -sfL --retry 3 --retry-delay 2 '{url}' | "
            f"tar -xz --strip-components=1 -C '{raw_dir}' --wildcards "
            f"'*/RAW/mpr-1.nifti.img' '*/RAW/mpr-1.nifti.hdr'"
        )
        print(f"[fetch] OAS2 part{p_} -> streaming, extracting mpr-1 only", flush=True)
        procs.append((p_, subprocess.Popen(cmd, shell=True,
                                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)))
    for p_, proc in procs:
        print(f"[fetch] OAS2 part{p_} done (rc={proc.wait()})", flush=True)
    got = sum(1 for root, _, files in os.walk(raw_dir) for f in files if f.endswith("mpr-1.nifti.img"))
    print(f"[fetch] extracted {got} OASIS-2 mpr-1 volumes into {raw_dir}", flush=True)


def skull_strip_batch(paths, out_dir="BRAIN", dataset="oasis2"):
    """GPU brain-extract each volume with deepbet; return {orig_path: brain_path}.

    Matches OASIS-1's brain-masked preprocessing so the two cohorts share a
    representation space (the cohort-leakage check must drop toward ~0.5 for a
    valid pooled/replication analysis). deepbet reads via nibabel (handles the
    Analyze/NIfTI .img pair) and writes a skull-stripped .nii.gz.

    The brain file is named by a per-subject id derived from the input path.
    OASIS-2 volumes are 3 levels deep (``OAS2_RAW/<session>/RAW/mpr-1.nifti.img``)
    so the session id is the parent-of-parent dir. ADNI's user-supplied layout is
    ``<root>/<subject>/T1.nii.gz`` (2 levels), so there the id is the IMMEDIATE
    parent dir — taking parent-of-parent would collide every subject onto the
    shared root name.
    """
    from deepbet import run_bet
    os.makedirs(out_dir, exist_ok=True)
    inputs, brains, mapping = [], [], {}
    for ip in paths:
        if not os.path.exists(ip):
            continue
        # session/subject id from the path (see docstring for the per-dataset depth).
        if dataset == "adni":
            sid = os.path.basename(os.path.dirname(ip))              # <subject>
        else:
            sid = os.path.basename(os.path.dirname(os.path.dirname(ip)))  # OAS2 <session>
        bp = os.path.join(out_dir, f"{sid}_brain.nii.gz")
        mapping[ip] = bp
        if not os.path.exists(bp):        # idempotent: skip already-extracted
            inputs.append(ip); brains.append(bp)
    print(f"[skull-strip] deepbet on {len(inputs)} new volumes "
          f"({len(mapping)-len(inputs)} already done) -> {out_dir} ...", flush=True)
    if inputs:
        run_bet(inputs, brains, no_gpu=False)
    ok = sum(1 for b in brains if os.path.exists(b))
    print(f"[skull-strip] wrote {ok}/{len(inputs)} brain volumes", flush=True)
    return mapping


def build_transform(resample_mode=5):
    """Static preprocessing from the Neuro-JEPA feature-extraction example.

    resample_mode: Spacingd interpolation order (5 = published spline; 1 = trilinear,
    ~10x faster on large raw volumes).
    """
    import numpy as np
    from monai import transforms
    from monai.transforms import Lambdad

    def remove_nan(img):
        img[np.isnan(img)] = 0.0
        return img

    return transforms.Compose([
        # Force the nibabel reader: OASIS-1 ships Analyze .img/.hdr pairs, and
        # MONAI's suffix-based auto-selection won't claim ".img" in monai 1.5.1
        # ("cannot find a suitable reader"). nibabel reads the pair natively.
        transforms.LoadImaged(keys=["image"], image_only=False, reader="NibabelReader"),
        transforms.EnsureChannelFirstd(keys=["image"]),
        Lambdad(("image",), remove_nan),
        transforms.Orientationd(keys=["image"], axcodes="RAS"),
        # Spacing resample dominates runtime on large raw (non-masked) volumes.
        # Order-5 spline (mode=[5]) matches the published pipeline but is ~10x slower
        # than trilinear; for big OASIS-2 raw heads use resample_mode=1 so the job
        # finishes inside one ephemeral runtime (negligible effect on the pooled 96^3
        # embedding, and cohort-leakage guards any residual difference).
        transforms.Spacingd(keys=["image"], pixdim=(1.0, 1.0, 1.0), mode=[resample_mode]),
        transforms.CropForegroundd(keys=["image"], source_key="image",
                                   select_fn=lambda x: x > 0.0, margin=4, allow_smaller=True),
        transforms.ResizeWithPadOrCropd(keys=["image"], spatial_size=[180, 216, 180], mode="edge"),
        transforms.Resized(keys=["image"], spatial_size=[100, 120, 100]),
        transforms.CenterSpatialCropd(keys=["image"], roi_size=list(ROI), allow_missing_keys=True),
        transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5,
                                                   b_min=0, b_max=1, clip=True),
        transforms.CastToTyped(keys=["image"], dtype=np.float32, allow_missing_keys=True),
    ])


def embed_volume(backbone, x, device):
    """x: (1,1,96,108,96) -> (768,) token-mean-pooled embedding. Frozen, no grad."""
    import torch
    torch.backends.cudnn.enabled = False
    with torch.no_grad(), torch.autocast(device if device == "cuda" else "cpu"):
        out = backbone(x.to(device))
    feats = out[0] if isinstance(out, (tuple, list)) else out
    feats = feats.float()
    if feats.dim() == 3:            # (B, tokens, dim) -> mean-pool tokens
        feats = feats.mean(dim=1)
    return feats[0].cpu().numpy()


def emit_durable(path):
    """Stream a gzip+base64 copy of the result to stdout so a runtime drop can't lose it."""
    blob = base64.b64encode(gzip.compress(open(path, "rb").read())).decode()
    print("===CSVGZ_START===", flush=True)
    print(blob, flush=True)
    print("===CSVGZ_END===", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="oasis1_gap_embeddings.csv")
    ap.add_argument("--image-col", default="image_path")
    ap.add_argument("--id-col", default="participant_id")
    ap.add_argument("--raw-dir", default="OASIS1_RAW")
    ap.add_argument("--dataset", choices=["oasis1", "oasis2", "adni"], default="oasis1",
                    help="oasis1 = disc tarballs, skull-stripped masked_gfc; "
                         "oasis2 = OAS2_RAW parts, raw mpr-1 (optionally --skull-strip); "
                         "adni = USER-SUPPLIED local T1w NIfTIs from an IDA/LONI download "
                         "(DUA-gated, no tarball fetch) — use --skull-strip --fast-resample, "
                         "same preprocessing as oasis2 raw")
    ap.add_argument("--skull-strip", action="store_true",
                    help="brain-extract each volume before embedding (deepbet) — for OASIS-2 raw, "
                         "to match OASIS-1's brain-masked preprocessing")
    ap.add_argument("--fast-resample", action="store_true",
                    help="use trilinear (order-1) Spacing resample instead of order-5 spline — "
                         "~10x faster on large raw volumes so the job fits one runtime")
    ap.add_argument("--discs", default="1-12", help="e.g. '1-12' or '1,3,5' (oasis1 only)")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N SUCCESSFULLY embedded subjects (0 = all) — for smoke tests; "
                         "counts embeds, not manifest rows, so it works with a single --discs")
    ap.add_argument("--token-file", default="hf_token.txt",
                    help="fallback file holding the HF token (raw 'hf_...' or 'HF_TOKEN=hf_...'); "
                         "uploaded to the ephemeral runtime, never committed")
    ap.add_argument("--skip-install", action="store_true")
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="skip subjects already present in --out (re-upload a partial CSV to resume)")
    ap.add_argument("--checkpoint-every", type=int, default=25,
                    help="emit a durable gzip+base64 copy of --out to stdout every N embeds (0=off)")
    args = ap.parse_args()

    token = _resolve_token(args.token_file)
    if not token:
        sys.exit("No HF token: set HF_TOKEN in the runtime env or upload a token file "
                 f"to '{args.token_file}' (gated Neuro-JEPA access required).")

    # parse discs spec
    discs = []
    for part in args.discs.split(","):
        if "-" in part:
            a, b = part.split("-")
            discs.extend(range(int(a), int(b) + 1))
        elif part.strip():
            discs.append(int(part))

    if not args.skip_install:
        install_deps()
        if args.skull_strip:
            # --no-deps keeps Colab's CUDA torch, but deepbet's runtime deps
            # fill_voids / connected-components-3d / fastremap are NOT in Colab's
            # stack — install them explicitly or run_bet dies with
            # `ModuleNotFoundError: No module named 'fill_voids'` mid-run.
            sh("pip install -q --no-deps deepbet", check=False)
            sh("pip install -q fill_voids connected-components-3d fastremap",
               check=False)
    if args.dataset == "adni":
        # ADNI raw MRI is DUA-gated and cannot be streamed like the open OASIS
        # tarballs: the user downloads T1w NIfTIs from ida.loni.usc.edu with their
        # own LONI credentials (Download -> Image Collections -> Advanced Search ->
        # MRI/MPRAGE -> DICOM -> dcm2niix), stages them locally/on the runtime, and
        # the manifest's image_path column points straight at those files. So there
        # is NOTHING to fetch here — the volumes are already present. Skull-strip +
        # fast-resample (matching oasis2 raw) are applied below via --skull-strip.
        print("[fetch] dataset=adni -> no tarball fetch; expecting USER-SUPPLIED local "
              "T1w NIfTIs at the manifest's image paths (IDA/LONI download).", flush=True)
        if not args.skull_strip:
            print("[warn] adni volumes are raw T1w heads — pass --skull-strip --fast-resample "
                  "to match OASIS's brain-masked preprocessing (shared embedding space).", flush=True)
    elif not args.skip_fetch:
        if args.dataset == "oasis2":
            fetch_oasis2(args.raw_dir)
        else:
            fetch_volumes(discs, args.raw_dir)

    import numpy as np
    import pandas as pd
    import torch
    from neurojepa.utils.init_utils import load_backbone_from_hf

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[gpu] device={device} "
          f"{'('+torch.cuda.get_device_name(0)+')' if device=='cuda' else ''}", flush=True)

    df = pd.read_csv(args.manifest)
    if args.image_col not in df.columns:
        sys.exit(f"manifest missing image column '{args.image_col}'")

    # Brain-extract up front (batch) and repoint the manifest at the brain volumes,
    # so the embed loop + transform run unchanged on skull-stripped input. Runs
    # whenever --skull-strip is set (independent of --skip-fetch, so a re-run can
    # reuse already-extracted raw volumes). Skips volumes already brain-extracted.
    if args.skull_strip:
        mapping = skull_strip_batch(list(df[args.image_col]), dataset=args.dataset)
        df[args.image_col] = df[args.image_col].map(lambda p: mapping.get(p, p))

    print(f"[load] fetching frozen backbone from HF ({HF_REPO}) ...", flush=True)
    backbone = load_backbone_from_hf(HF_REPO, device=device, token=token)
    backbone.eval()
    trans = build_transform(resample_mode=1 if args.fast_resample else 5)

    # Resume: skip subjects already embedded in a re-uploaded partial CSV.
    already = set()
    if args.resume and os.path.exists(args.out):
        try:
            prev = pd.read_csv(args.out)
            already = set(prev[args.id_col].astype(str))
            print(f"[resume] {len(already)} subjects already in {args.out} -- skipping those.", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[resume] could not read {args.out} ({exc}); starting fresh.", flush=True)

    meta_cols = [c for c in df.columns if c != args.image_col]
    rows, missing, failed = [], [], []
    if already:  # seed with prior rows so the output stays complete
        rows = pd.read_csv(args.out).to_dict("records")
    emb_dim = None

    for i, r in df.iterrows():
        pid = r.get(args.id_col, i)
        if str(pid) in already:
            continue
        img_path = r[args.image_col]
        if not os.path.exists(img_path):
            print(f"[{i+1}/{len(df)}] {pid} MISSING {img_path} -- skipping", flush=True)
            missing.append(pid)
            continue
        try:
            # Apply the transform directly (no DataLoader worker process): faster,
            # and a bad volume raises a clean error instead of a wrapped multiprocess
            # traceback. MONAI returns [C,H,W,D]; add the batch dim -> [1,C,H,W,D].
            import torch as _torch
            sample = trans({"image": img_path})
            x = _torch.as_tensor(np.asarray(sample["image"])).unsqueeze(0).float()
            emb = embed_volume(backbone, x, device)
        except Exception as exc:  # noqa: BLE001 -- one bad volume shouldn't kill the run
            print(f"[{i+1}/{len(df)}] {pid} FAILED: {exc!r} -- skipping", flush=True)
            failed.append(pid)
            continue
        emb_dim = emb.shape[0]
        row = {c: r[c] for c in meta_cols}
        row.update({f"emb_{j}": float(v) for j, v in enumerate(emb)})
        rows.append(row)
        # incremental durable write: rewrite the CSV after every subject, and emit a
        # durable gzip+base64 checkpoint every N so a websocket drop / runtime release
        # mid-run still leaves a recoverable partial in the local exec log.
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"[{i+1}/{len(df)}] {pid} -> emb dim {emb_dim}", flush=True)
        if args.checkpoint_every and len(rows) % args.checkpoint_every == 0:
            print(f"[checkpoint] {len(rows)} embedded so far", flush=True)
            emit_durable(args.out)
        if args.limit and len(rows) >= args.limit:
            print(f"[limit] reached {args.limit} embeddings -- stopping (smoke test).", flush=True)
            break

    if not rows:
        sys.exit("No embeddings produced -- check volume paths / extraction.")

    print(f"\n[done] wrote {args.out}: {len(rows)} subjects x {emb_dim}-d", flush=True)
    if missing:
        print(f"[warn] {len(missing)} volumes missing: {missing[:10]}{'...' if len(missing)>10 else ''}", flush=True)
    if failed:
        print(f"[warn] {len(failed)} volumes failed to embed: {failed}", flush=True)
    emit_durable(args.out)


if __name__ == "__main__":
    main()
