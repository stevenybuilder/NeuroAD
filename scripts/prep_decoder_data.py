#!/usr/bin/env python3
"""
prep_decoder_data.py — build L2's NeuroJEPA-decoder training set on a Colab GPU.

This is the "prep step (scripts, TODO)" referenced by
``scripts/train_neurojepa_decoder.py``: it materializes the per-subject ``.npz``
triples the trainer consumes and uploads them to GCS so they SURVIVE runtime death
and can be pulled straight into the training runtime.

For each subject in the ADNI image manifest (``subject_id, image_path, ...``) it:

  1. loads the T1w NIfTI and skull-strips it (deepbet, GPU) — matching the
     brain-masked preprocessing the NeuroJEPA embeddings already use,
  2. gets the subject's 768-d NeuroJEPA embedding — either read from an existing
     embeddings CSV (``--embeddings-csv``, the honest fast path: reuse the exact
     vector we already extracted) or recomputed with the frozen backbone by reusing
     ``scripts/neurojepa_embed_colab.py`` (``build_transform`` + ``embed_volume``),
  3. runs FastSurfer ``--seg_only`` on the T1w -> an aseg segmentation VOLUME,
     resamples it into the skull-stripped brain's voxel grid (nearest-neighbour, so
     MRI and label stay voxel-aligned) and remaps the FreeSurfer aseg label ids into
     the compact :data:`neurojepa_decoder.LABELS` set,
  4. resamples/normalizes BOTH the brain and the remapped label to a ``--size``^3
     cube (MONAI; trilinear for the image, nearest for the label so no class is
     invented at a boundary),
  5. writes ``<sid>.npz`` with keys ``mri`` [1,D,H,W] float32, ``jepa`` [768]
     float32, ``label`` [D,H,W] int64 (the exact contract the decoder/trainer pin),
     and uploads it to ``gs://<bucket>/decoder_data/<sid>.npz`` via
     ``neuroad.integrations.gcs_store``.

RESUMABLE: subjects whose ``.npz`` already exists on GCS are skipped (a single
``list_prefix`` up front, so a re-run after a runtime death continues instead of
restarting). The uploaded ``.npz`` itself is the durable checkpoint — there is no
partial-CSV to rebuild.

HONEST: a subject is SKIPPED (never fabricated) if its T1w is missing, its
embedding is absent from the CSV / fails to compute, or FastSurfer fails to
segment it. Labels are always real FastSurfer voxels, only coarsened — a class is
a real voxel of that structure, never invented.

Compliance: raw MRI, deepbet output, FastSurfer + NeuroJEPA weights all live ONLY
on the ephemeral runtime and are never committed; only the derived ``.npz`` triples
leave the runtime (to the project's private GCS bucket). This runs ONLY on a Colab
GPU — it is not locally executable (heavy imports are lazy; ``--help`` works
offline).

Usage (see docs/COLAB_RUNBOOK.md; parallels the embed / FastSurfer runbooks):
    colab start --gpu t4                                   # note the session id
    colab upload data/real/_manifests/adni_image_manifest.csv manifest.csv
    # fast path — reuse the embeddings we already extracted:
    colab upload data/real/adni_neurojepa_embeddings.csv emb.csv
    # HF_TOKEN only needed if RECOMPUTING embeddings (no --embeddings-csv):
    colab exec --session <id> --timeout 20m scripts/prep_decoder_data.py -- \
        --manifest manifest.csv --embeddings-csv emb.csv --limit 2   # smoke test
    colab exec --session <id> --timeout 120m scripts/prep_decoder_data.py -- \
        --manifest manifest.csv --embeddings-csv emb.csv
    # then train straight off GCS:
    colab exec --session <id> scripts/train_neurojepa_decoder.py -- --size 96
"""
from __future__ import annotations

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# FreeSurfer aseg id -> compact decoder LABELS remap (pure; matches
# neurojepa_decoder.LABELS order exactly: background/left_hippocampus/
# right_hippocampus/ventricle/cortex/other_brain -> 0..5). Kept as module-level
# constants so the remap is inspectable/testable without importing torch.
# ---------------------------------------------------------------------------
LBL_BACKGROUND = 0
LBL_LEFT_HIPPOCAMPUS = 1
LBL_RIGHT_HIPPOCAMPUS = 2
LBL_VENTRICLE = 3
LBL_CORTEX = 4
LBL_OTHER_BRAIN = 5

# FreeSurfer/FastSurfer aseg StructId constants.
FS_LEFT_HIPPOCAMPUS = 17
FS_RIGHT_HIPPOCAMPUS = 53
# lateral + inf-lat + 3rd + 4th + 5th ventricles (mirrors structural_segmenter's
# _VENTRICLE_STRUCTS; choroid plexus is deliberately NOT ventricle).
FS_VENTRICLES = (4, 43, 5, 44, 14, 15, 72)
# aseg cortical-ribbon ids (Left/Right-Cerebral-Cortex). In an aparc+aseg volume
# the cortex is instead parcellated into 1000-1035 (L) / 2000-2035 (R); both
# conventions are folded into CORTEX below so whichever seg file exists works.
FS_CORTEX = (3, 42)


def remap_aseg_to_labels(seg):
    """FreeSurfer aseg id volume -> compact :data:`LABELS` id volume (int64).

    Pure + deterministic. ``seg`` is an int array of FreeSurfer StructIds. Returns
    an int64 array of the same shape holding only 0..5. Assignment order is
    specific->generic (hippocampus, ventricle, cortex, then any remaining nonzero
    brain voxel -> other_brain); background (0) stays 0. Never invents a voxel — it
    only coarsens real FastSurfer labels.
    """
    import numpy as np

    seg = np.asarray(seg)
    out = np.zeros(seg.shape, dtype=np.int64)
    out[seg == FS_LEFT_HIPPOCAMPUS] = LBL_LEFT_HIPPOCAMPUS
    out[seg == FS_RIGHT_HIPPOCAMPUS] = LBL_RIGHT_HIPPOCAMPUS
    out[np.isin(seg, np.asarray(FS_VENTRICLES))] = LBL_VENTRICLE
    # cortex: aseg ribbon ids OR any aparc cortical parcel (1000-2999).
    cortex = np.isin(seg, np.asarray(FS_CORTEX)) | ((seg >= 1000) & (seg < 3000))
    out[cortex] = LBL_CORTEX
    # remaining brain tissue (any nonzero not already assigned) -> other_brain.
    other = (seg != 0) & (out == 0)
    out[other] = LBL_OTHER_BRAIN
    return out


# ---------------------------------------------------------------------------
# Colab-runtime setup (heavy; only invoked inside main, never at import)
# ---------------------------------------------------------------------------


def sh(cmd: str, check: bool = True):
    import subprocess
    print(f"    $ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check)


def install_deps(*, need_embed_backbone: bool):
    """Install MONAI + deepbet + nibabel + FastSurfer (and NeuroJEPA if recomputing).

    Keeps Colab's working CUDA torch (``--no-deps`` on torch-adjacent installs) so we
    don't clobber the GPU build. All weights/code land on the ephemeral runtime only.
    """
    print("[deps] installing decoder-prep deps (keeping Colab torch) ...", flush=True)
    # Install resiliently: keep Colab's numpy 2.x (do NOT pin numpy<2 — it forces a
    # full re-resolve that trips pip's version parser on Colab's preinstalled stack),
    # and never let one flaky package abort the whole prep (check=False, split installs).
    for pkg in ("monai==1.5.1", "nibabel", "scikit-image>=0.22", "pyyaml", "h5py",
                "yacs", "simpleitk", "huggingface_hub>=0.30"):
        sh(f"pip install -q --no-deps '{pkg}'", check=False)
    # deepbet's other deps (numpy/nibabel/scipy/requests/tqdm) are already present,
    # BUT run_bet also needs fill_voids/connected-components-3d/fastremap which the
    # --no-deps install skips — install them explicitly (numpy-safe compiled wheels).
    sh("pip install -q --no-deps deepbet fill-voids connected-components-3d fastremap",
       check=False)
    # FastSurfer (Apache-2.0); its network weights download on first run.
    if not os.path.exists("FastSurfer"):
        sh("git clone --depth 1 https://github.com/Deep-MI/FastSurfer.git", check=False)
    sh("pip install -q --no-deps -r FastSurfer/requirements.txt", check=False)
    os.environ.setdefault("FASTSURFER_HOME", os.path.abspath("FastSurfer"))
    print(f"[deps] FASTSURFER_HOME={os.environ['FASTSURFER_HOME']}", flush=True)
    if need_embed_backbone:
        # Only needed when recomputing embeddings (no --embeddings-csv): pull the
        # frozen Neuro-JEPA backbone + its deps the same way the embed job does.
        import neurojepa_embed_colab as nje  # noqa: F401 (sibling script)
        nje.install_deps()


def skull_strip_batch(pairs, out_dir="DECODER_BRAIN"):
    """deepbet brain-extract each (sid, t1_path); return {sid: brain_path}.

    Idempotent: a brain already written is reused. Missing/failed inputs are simply
    absent from the returned mapping (honest degrade — never fabricates a brain).
    """
    from deepbet import run_bet
    os.makedirs(out_dir, exist_ok=True)
    inputs, brains, mapping = [], [], {}
    for sid, ip in pairs:
        if not ip or not os.path.exists(ip):
            continue
        bp = os.path.join(out_dir, f"{sid}_brain.nii.gz")
        mapping[sid] = bp
        if not os.path.exists(bp):
            inputs.append(ip)
            brains.append(bp)
    print(f"[skull-strip] deepbet on {len(inputs)} new volumes "
          f"({len(mapping) - len(inputs)} already done) -> {out_dir} ...", flush=True)
    if inputs:
        try:
            run_bet(inputs, brains, no_gpu=False)
        except Exception as exc:  # noqa: BLE001 -- one bad batch shouldn't kill prep
            print(f"[skull-strip] run_bet error: {exc!r}", flush=True)
    # drop any sid whose brain didn't materialize (honest).
    return {sid: bp for sid, bp in mapping.items() if os.path.exists(bp)}


def run_fastsurfer_seg_volume(runner_sh, nifti, sid, sd, timeout):
    """Run FastSurfer ``--seg_only`` for one T1w; return the aseg seg VOLUME path.

    Unlike ``fastsurfer_volumes_colab`` (which wants aseg.stats), the decoder needs
    the segmentation VOLUME to build a per-voxel label. Returns the first seg-volume
    that exists (aparc+aseg deep seg preferred — it carries cortical parcels — then
    plain aseg), or ``None`` on non-zero exit / timeout / missing output.
    """
    import subprocess
    cmd = [
        runner_sh, "--t1", str(nifti), "--sid", str(sid), "--sd", str(sd),
        "--seg_only", "--no_cereb", "--no_biasfield",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        print(f"    [fastsurfer] {sid} error: {exc!r}", flush=True)
        return None
    if proc.returncode != 0:
        print(f"    [fastsurfer] {sid} rc={proc.returncode}: "
              f"{proc.stderr.strip()[-300:]}", flush=True)
        return None
    for name in ("aparc.DKTatlas+aseg.deep.mgz", "aparc.DKTatlas+aseg.mgz",
                 "aseg.auto_noCCseg.mgz", "aseg.mgz"):
        cand = os.path.join(sd, str(sid), "mri", name)
        if os.path.exists(cand):
            return cand
    return None


def build_label_cube(seg_path, brain_img, size):
    """aseg seg volume -> label cube [size,size,size] int64, voxel-aligned to brain.

    Resamples the FastSurfer seg into the skull-stripped brain's voxel grid
    (nearest-neighbour, so labels are never blended), remaps to compact LABELS, then
    resizes to the ``size``^3 cube with nearest interpolation (same as the MRI's
    grid, so image and label correspond voxel-for-voxel).
    """
    import nibabel as nib
    import numpy as np
    from nibabel.processing import resample_from_to
    from monai.transforms import Resize

    seg_img = nib.load(seg_path)
    # into the brain's grid (shape+affine) with nearest-neighbour (order=0).
    seg_on_brain = resample_from_to(seg_img, (brain_img.shape, brain_img.affine),
                                    order=0)
    seg_arr = np.asanyarray(seg_on_brain.dataobj)
    seg_arr = np.rint(seg_arr).astype(np.int64)     # order-0 keeps ints; guard dtype
    labels = remap_aseg_to_labels(seg_arr)          # [x,y,z] int64, 0..5
    lab_t = labels.astype(np.float32)[None]         # [1,x,y,z]
    lab_t = Resize(spatial_size=[size, size, size], mode="nearest")(lab_t)
    return np.rint(np.asarray(lab_t)[0]).astype(np.int64)   # [size,size,size]


def build_mri_cube(brain_img, size):
    """Skull-stripped brain -> normalized MRI cube [1,size,size,size] float32.

    Percentile intensity scaling to [0,1] then trilinear resize to the cube. Uses
    the SAME grid the label cube is built on, so the two align.
    """
    import numpy as np
    from monai.transforms import Resize, ScaleIntensityRangePercentiles

    arr = np.asanyarray(brain_img.dataobj).astype(np.float32)[None]   # [1,x,y,z]
    arr = ScaleIntensityRangePercentiles(lower=0.5, upper=99.5, b_min=0.0,
                                         b_max=1.0, clip=True)(arr)
    arr = Resize(spatial_size=[size, size, size], mode="trilinear",
                 align_corners=False)(arr)
    return np.asarray(arr).astype(np.float32)        # [1,size,size,size]


def load_embeddings_csv(path, id_col):
    """Read an embeddings CSV -> {sid: np.float32[768]}. Skips rows with any NaN emb.

    The 768 columns are ``emb_0..emb_767`` (the convention every embed CSV uses).
    Returns ({}, None) with a clear message if the file lacks those columns — so the
    caller degrades honestly rather than fabricating vectors.
    """
    import numpy as np
    import pandas as pd

    df = pd.read_csv(path)
    if id_col not in df.columns:
        print(f"[emb] embeddings CSV missing id column '{id_col}' -- ignoring CSV.",
              flush=True)
        return {}, None
    emb_cols = [f"emb_{j}" for j in range(768)]
    missing = [c for c in emb_cols if c not in df.columns]
    if missing:
        print(f"[emb] embeddings CSV missing {len(missing)} emb_* columns "
              f"(first: {missing[0]}) -- ignoring CSV.", flush=True)
        return {}, None
    out = {}
    for _, r in df.iterrows():
        vec = r[emb_cols].to_numpy(dtype=np.float32)
        if not np.isfinite(vec).all():
            continue                                 # honest: drop NaN/inf rows
        out[str(r[id_col])] = vec
    print(f"[emb] loaded {len(out)} embeddings from {path}", flush=True)
    return out, emb_cols


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True,
                    help="ADNI image manifest CSV (subject_id, image_path, ...)")
    ap.add_argument("--image-col", default="image_path")
    ap.add_argument("--id-col", default="subject_id")
    ap.add_argument("--size", type=int, default=96,
                    help="edge of the D=H=W cube the decoder trains on (default 96)")
    ap.add_argument("--embeddings-csv", default=None,
                    help="reuse already-extracted 768-d embeddings (emb_0..emb_767) "
                         "instead of recomputing them on the backbone")
    ap.add_argument("--emb-id-col", default=None,
                    help="id column in --embeddings-csv (defaults to --id-col)")
    ap.add_argument("--gcs-prefix", default="decoder_data",
                    help="GCS object prefix; each subject -> <prefix>/<sid>.npz")
    ap.add_argument("--bucket", default=None,
                    help="override GCS bucket (else env NEUROAD_GCS_BUCKET / default)")
    ap.add_argument("--sd", default="FASTSURFER_OUT",
                    help="FastSurfer subjects dir (per-subject seg volumes land here)")
    ap.add_argument("--work-dir", default="DECODER_NPZ",
                    help="local scratch dir for the .npz before upload")
    ap.add_argument("--timeout", type=int, default=1800,
                    help="per-subject FastSurfer timeout (seconds)")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N SUCCESSFULLY built subjects (0 = all)")
    ap.add_argument("--no-skull-strip", action="store_true",
                    help="feed the manifest volume straight to the MRI cube (assume "
                         "it is already brain-extracted); FastSurfer still runs on it")
    ap.add_argument("--keep-local", action="store_true",
                    help="keep each .npz on local disk after upload (default: delete)")
    ap.add_argument("--token-file", default="hf_token.txt",
                    help="HF token fallback file when recomputing embeddings")
    ap.add_argument("--skip-install", action="store_true")
    args = ap.parse_args()

    if args.bucket:
        os.environ["NEUROAD_GCS_BUCKET"] = args.bucket
    emb_id_col = args.emb_id_col or args.id_col
    use_csv = args.embeddings_csv is not None

    # repo importable so we reuse gcs_store + (optionally) the embed helpers.
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "..", "src"))
    sys.path.insert(0, here)   # sibling scripts (neurojepa_embed_colab)

    if not args.skip_install:
        install_deps(need_embed_backbone=not use_csv)

    import numpy as np
    import pandas as pd
    import nibabel as nib
    try:
        from neuroad.integrations import gcs_store as gcs
    except ModuleNotFoundError:
        import gcs_store as gcs  # flat-staged on the Colab runtime

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gpu = torch.cuda.get_device_name(0) if device == "cuda" else ""
    except Exception:  # noqa: BLE001
        device, gpu = "cpu", ""
    print(f"[gpu] device={device} {('('+gpu+')') if gpu else ''}", flush=True)
    if device != "cuda":
        print("[warn] no CUDA GPU — deepbet/FastSurfer will be very slow on CPU.",
              flush=True)

    df = pd.read_csv(args.manifest)
    if args.image_col not in df.columns:
        sys.exit(f"manifest missing image column '{args.image_col}'")
    if args.id_col not in df.columns:
        sys.exit(f"manifest missing id column '{args.id_col}'")

    # ---- embeddings source (CSV reuse vs recompute) ----
    emb_map = {}
    backbone = trans = None
    if use_csv:
        emb_map, ok = load_embeddings_csv(args.embeddings_csv, emb_id_col)
        if ok is None:
            sys.exit("embeddings CSV unusable (see message above); pass a valid CSV "
                     "or drop --embeddings-csv to recompute.")
    else:
        token = None
        import neurojepa_embed_colab as nje
        token = nje._resolve_token(args.token_file)
        if not token:
            sys.exit("recomputing embeddings needs an HF token: set HF_TOKEN or "
                     f"upload it to '{args.token_file}' (or pass --embeddings-csv).")
        from neurojepa.utils.init_utils import load_backbone_from_hf
        print(f"[load] fetching frozen backbone from HF ({nje.HF_REPO}) ...", flush=True)
        backbone = load_backbone_from_hf(nje.HF_REPO, device=device, token=token)
        backbone.eval()
        trans = nje.build_transform(resample_mode=1)   # fast trilinear resample

    # ---- resumable skip-set: what already lives on GCS ----
    prefix = args.gcs_prefix.rstrip("/")
    existing = set()
    for name in gcs.list_prefix(prefix + "/"):
        base = os.path.basename(name)
        if base.endswith(".npz"):
            existing.add(base[:-4])
    print(f"[resume] {len(existing)} subjects already on {gcs.uri(prefix + '/')}",
          flush=True)

    os.makedirs(args.work_dir, exist_ok=True)

    # ---- select pending subjects (not on GCS, T1 present) ----
    pending = []          # (sid, t1_path)
    missing = []
    for i, r in df.iterrows():
        sid = str(r[args.id_col])
        if sid in existing:
            continue
        t1 = r[args.image_col]
        if not isinstance(t1, str) or not os.path.exists(t1):
            print(f"[{sid}] MISSING T1 {t1!r} -- skipping", flush=True)
            missing.append(sid)
            continue
        pending.append((sid, t1))
    print(f"[plan] {len(pending)} subjects to build "
          f"({len(existing)} done, {len(missing)} missing T1)", flush=True)
    if not pending:
        print("[done] nothing to do.", flush=True)
        return

    # ---- batch skull-strip the pending T1s (fast; one deepbet model load) ----
    if args.no_skull_strip:
        brain_map = {sid: t1 for sid, t1 in pending}
    else:
        brain_map = skull_strip_batch(pending)

    runner_sh = os.path.join(os.environ.get("FASTSURFER_HOME", "FastSurfer"),
                             "run_fastsurfer.sh")
    if not os.path.exists(runner_sh):
        sys.exit(f"run_fastsurfer.sh not found at {runner_sh} — install FastSurfer "
                 "(drop --skip-install).")

    built, no_emb, seg_fail, err = 0, [], [], []
    for idx, (sid, t1) in enumerate(pending):
        tag = f"[{idx + 1}/{len(pending)}] {sid}"
        brain_path = brain_map.get(sid)
        if brain_path is None or not os.path.exists(brain_path):
            print(f"{tag} no brain volume (skull-strip failed) -- skipping", flush=True)
            err.append(sid)
            continue

        # 1) embedding (honest: skip if absent / non-finite)
        if use_csv:
            jepa = emb_map.get(sid)
            if jepa is None:
                print(f"{tag} no embedding in CSV -- skipping", flush=True)
                no_emb.append(sid)
                continue
        else:
            try:
                sample = trans({"image": brain_path})
                x = torch.as_tensor(np.asarray(sample["image"])).unsqueeze(0).float()
                jepa = nje.embed_volume(backbone, x, device)
            except Exception as exc:  # noqa: BLE001
                print(f"{tag} embed FAILED: {exc!r} -- skipping", flush=True)
                no_emb.append(sid)
                continue
        jepa = np.asarray(jepa, dtype=np.float32).reshape(-1)
        if jepa.shape[0] != 768 or not np.isfinite(jepa).all():
            print(f"{tag} bad embedding (dim={jepa.shape[0]}) -- skipping", flush=True)
            no_emb.append(sid)
            continue

        # 2) FastSurfer seg volume -> label cube (honest: skip on failure)
        seg_path = run_fastsurfer_seg_volume(runner_sh, t1, sid, args.sd, args.timeout)
        if seg_path is None:
            print(f"{tag} FastSurfer produced no seg volume -- skipping", flush=True)
            seg_fail.append(sid)
            continue

        try:
            brain_img = nib.load(brain_path)
            mri = build_mri_cube(brain_img, args.size)          # [1,S,S,S] f32
            label = build_label_cube(seg_path, brain_img, args.size)  # [S,S,S] i64
        except Exception as exc:  # noqa: BLE001
            print(f"{tag} cube build FAILED: {exc!r} -- skipping", flush=True)
            err.append(sid)
            continue

        # 3) write + upload the .npz (the durable, resumable checkpoint)
        local = os.path.join(args.work_dir, f"{sid}.npz")
        try:
            np.savez_compressed(local, mri=mri.astype(np.float32),
                                jepa=jepa.astype(np.float32),
                                label=label.astype(np.int64))
            gcs.upload(local, f"{prefix}/{sid}.npz")
        except Exception as exc:  # noqa: BLE001
            print(f"{tag} upload FAILED: {exc!r} -- skipping", flush=True)
            err.append(sid)
            continue
        finally:
            if not args.keep_local and os.path.exists(local):
                try:
                    os.remove(local)
                except OSError:
                    pass

        built += 1
        present = int((label > 0).sum())
        print(f"{tag} -> {gcs.uri(prefix + '/' + sid + '.npz')} "
              f"(mri{tuple(mri.shape)} label{tuple(label.shape)} "
              f"fg_voxels={present})", flush=True)
        if args.limit and built >= args.limit:
            print(f"[limit] reached {args.limit} built -- stopping.", flush=True)
            break

    print(f"\n[done] built {built} .npz -> {gcs.uri(prefix + '/')}", flush=True)
    if missing:
        print(f"[warn] {len(missing)} missing T1: {missing[:10]}"
              f"{'...' if len(missing) > 10 else ''}", flush=True)
    if no_emb:
        print(f"[warn] {len(no_emb)} skipped (no/bad embedding): {no_emb[:10]}"
              f"{'...' if len(no_emb) > 10 else ''}", flush=True)
    if seg_fail:
        print(f"[warn] {len(seg_fail)} skipped (FastSurfer failed): {seg_fail[:10]}"
              f"{'...' if len(seg_fail) > 10 else ''}", flush=True)
    if err:
        print(f"[warn] {len(err)} skipped (build/upload error): {err[:10]}"
              f"{'...' if len(err) > 10 else ''}", flush=True)


if __name__ == "__main__":
    main()
