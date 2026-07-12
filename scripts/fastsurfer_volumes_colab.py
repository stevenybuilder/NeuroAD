#!/usr/bin/env python3
"""
Self-contained Colab GPU job: OASIS-1 T1w -> FastSurfer aseg volumes CSV.

Mirrors ``scripts/neurojepa_embed_colab.py`` but the model is FastSurfer (Apache-2.0,
~1 min/volume on a GPU) instead of Neuro-JEPA, and the derived table is structural
VOLUMES (hippocampal / ventricle / cortex / whole-brain / intracranial in mm^3),
not embeddings. It runs ENTIRELY on the ephemeral GPU runtime so a websocket drop or
a preempted runtime never leaves a half-done mess in the repo:

  1. streams the 12 open-access OASIS-1 cross-sectional disc tarballs, extracting
     ONLY the T88 skull-stripped, gain-field-corrected ``*_masked_gfc.img/.hdr``
     volumes (raw MRI stays on the runtime, never comes home, never hits git),
  2. installs FastSurfer + its Apache-2.0 weights on the runtime (weights NEVER
     written to the repo),
  3. runs FastSurfer ``--seg_only`` per subject -> an ``aseg.stats`` per subject,
  4. parses each aseg.stats via ``neuroad.integrations.structural_segmenter.
     parse_aseg_stats`` -> one normalized volume row per subject,
  5. writes ``--out`` (default ``data/real/oasis_volumes.csv``) incrementally AND
     streams a durable gzip+base64 copy of it to stdout between
     ===CSVGZ_START===/===CSVGZ_END=== markers, so the result lands in the local
     exec log the instant it is computed.

Only the small derived volume table is meant to leave the runtime. Do NOT download
or commit the FastSurfer weights or the raw volumes (they are git-ignored).

Usage (see docs/COLAB_RUNBOOK.md; parallels the Neuro-JEPA runbook):
    colab start --gpu t4                                   # note the session id
    colab upload data/real/_manifests/oasis1_gap_manifest.csv manifest.csv
    # smoke test (disc1 only, 2 subjects) then the full run:
    colab exec --session <id> --timeout 20m scripts/fastsurfer_volumes_colab.py -- --manifest manifest.csv --discs 1 --limit 2
    colab exec --session <id> --timeout 90m scripts/fastsurfer_volumes_colab.py -- --manifest manifest.csv
    colab download --session <id> oasis_volumes.csv data/real/oasis_volumes.csv
    colab stop --session <id>

The output CSV joins onto the OASIS contract table by ``subject_id`` (``OAS1_...``);
``src/neuroad/data/real.py`` additively picks it up when present and works unchanged
when it is absent.
"""
import argparse
import base64
import gzip
import os
import subprocess
import sys

OASIS_DISC_URL = "https://download.nrg.wustl.edu/data/oasis_cross-sectional_disc{d}.tar.gz"
# The volume keys the parser exposes, in a stable CSV column order.
VOLUME_COLS = [
    "hippocampal_volume", "ventricle_volume", "whole_brain_volume",
    "cortex_volume", "intracranial_volume",
]


def sh(cmd: str, check: bool = True):
    print(f"    $ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check)


def install_deps():
    """Install FastSurfer (Apache-2.0) on the runtime, keeping Colab's CUDA torch.

    FastSurfer ships as a git repo with a pip requirements set; its network weights
    download on first run. We keep Colab's working CUDA torch (``--no-deps`` on the
    torch-adjacent bits) so we don't clobber the GPU build. Weights land under
    ``$FASTSURFER_HOME`` on the ephemeral runtime and are NEVER committed.
    """
    print("[deps] installing FastSurfer (keeping Colab torch) ...", flush=True)
    if not os.path.exists("FastSurfer"):
        sh("git clone --depth 1 https://github.com/Deep-MI/FastSurfer.git")
    # FastSurfer's python deps (nibabel/scipy/etc.); torch stays Colab's CUDA build.
    sh("pip install -q nibabel 'scikit-image>=0.22' 'numpy<2' pyyaml h5py "
       "yacs simpleitk", check=False)
    sh("pip install -q --no-deps -r FastSurfer/requirements.txt", check=False)
    os.environ.setdefault("FASTSURFER_HOME", os.path.abspath("FastSurfer"))
    print(f"[deps] FASTSURFER_HOME={os.environ['FASTSURFER_HOME']}", flush=True)


def fetch_volumes(discs, raw_dir):
    """Stream each disc tarball through tar, extracting ONLY masked_gfc .img/.hdr.

    Identical streaming strategy to neurojepa_embed_colab.fetch_volumes: the ~1.4 GB
    of everything else per disc is discarded in-flight, so peak disk stays tiny.
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
                                          stdout=subprocess.DEVNULL,
                                          stderr=subprocess.DEVNULL)))
    for d, p in procs:
        rc = p.wait()
        print(f"[fetch] disc{d} done (rc={rc})", flush=True)
    got = sum(1 for _, _, files in os.walk(raw_dir)
              for f in files if f.endswith("_masked_gfc.img"))
    print(f"[fetch] extracted {got} masked_gfc volumes into {raw_dir}", flush=True)


def run_fastsurfer_seg(runner_sh, nifti, sid, sd, timeout):
    """Run FastSurfer --seg_only for one volume; return the aseg.stats path or None.

    Segmentation-only (no surface recon) is the ~1 min/GPU path and yields the
    aseg volumes we need. Returns None on non-zero exit / timeout / missing output
    (honest degrade — a failed subject is skipped, never fabricated).
    """
    cmd = [
        runner_sh, "--t1", nifti, "--sid", sid, "--sd", sd,
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
    for name in ("aseg+DKT.stats", "aseg.stats"):
        cand = os.path.join(sd, sid, "stats", name)
        if os.path.exists(cand):
            return cand
    return None


def emit_durable(path):
    """Stream a gzip+base64 copy of the result to stdout so a runtime drop can't lose it."""
    blob = base64.b64encode(gzip.compress(open(path, "rb").read())).decode()
    print("===CSVGZ_START===", flush=True)
    print(blob, flush=True)
    print("===CSVGZ_END===", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True,
                    help="CSV with an image-path column + a subject-id column")
    ap.add_argument("--out", default="oasis_volumes.csv")
    ap.add_argument("--image-col", default="image_path")
    ap.add_argument("--id-col", default="participant_id")
    ap.add_argument("--raw-dir", default="OASIS1_RAW")
    ap.add_argument("--sd", default="FASTSURFER_OUT",
                    help="FastSurfer subjects dir (per-subject aseg.stats land here)")
    ap.add_argument("--discs", default="1-12", help="e.g. '1-12' or '1,3,5'")
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after N SUCCESSFULLY segmented subjects (0 = all)")
    ap.add_argument("--timeout", type=int, default=1800,
                    help="per-subject FastSurfer timeout (seconds)")
    ap.add_argument("--skip-install", action="store_true")
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="skip subjects already present in --out")
    ap.add_argument("--checkpoint-every", type=int, default=10,
                    help="emit a durable gzip+base64 copy of --out every N (0=off)")
    args = ap.parse_args()

    # repo importable so we can reuse the SAME parser the tests pin.
    sys.path.insert(0, os.path.join(os.getcwd(), "src"))

    discs = []
    for part in args.discs.split(","):
        if "-" in part:
            a, b = part.split("-")
            discs.extend(range(int(a), int(b) + 1))
        elif part.strip():
            discs.append(int(part))

    if not args.skip_install:
        install_deps()
    if not args.skip_fetch:
        fetch_volumes(discs, args.raw_dir)

    import pandas as pd
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gpu = torch.cuda.get_device_name(0) if device == "cuda" else ""
    except Exception:  # noqa: BLE001
        device, gpu = "cpu", ""
    print(f"[gpu] device={device} {('('+gpu+')') if gpu else ''}", flush=True)
    if device != "cuda":
        print("[warn] no CUDA GPU — FastSurfer --seg_only will be very slow on CPU.",
              flush=True)

    from neuroad.integrations.structural_segmenter import parse_aseg_stats

    runner_sh = os.path.join(os.environ.get("FASTSURFER_HOME", "FastSurfer"),
                             "run_fastsurfer.sh")
    if not os.path.exists(runner_sh):
        sys.exit(f"run_fastsurfer.sh not found at {runner_sh} — install FastSurfer "
                 "(drop --skip-install).")

    df = pd.read_csv(args.manifest)
    if args.image_col not in df.columns:
        sys.exit(f"manifest missing image column '{args.image_col}'")

    already = set()
    rows = []
    if args.resume and os.path.exists(args.out):
        try:
            prev = pd.read_csv(args.out)
            already = set(prev["subject_id"].astype(str))
            rows = prev.to_dict("records")
            print(f"[resume] {len(already)} subjects already in {args.out}.", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[resume] could not read {args.out} ({exc}); starting fresh.",
                  flush=True)

    meta_cols = [c for c in df.columns if c != args.image_col]
    missing, failed = [], []

    for i, r in df.iterrows():
        pid = str(r.get(args.id_col, i))
        if pid in already:
            continue
        img = r[args.image_col]
        if not os.path.exists(img):
            print(f"[{i+1}/{len(df)}] {pid} MISSING {img} -- skipping", flush=True)
            missing.append(pid)
            continue
        aseg = run_fastsurfer_seg(runner_sh, img, pid, args.sd, args.timeout)
        if aseg is None:
            failed.append(pid)
            continue
        vols = parse_aseg_stats(aseg)
        # carry manifest metadata through; stamp subject_id in the contract's
        # OAS1_ convention if the manifest id isn't already prefixed.
        sid = pid if pid.startswith(("OAS1_", "OAS2_")) else f"OAS1_{pid}"
        row = {c: r[c] for c in meta_cols}
        row["subject_id"] = sid
        for k in VOLUME_COLS:
            row[k] = vols.get(k)
        row["source"] = vols.get("source")
        rows.append(row)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print(f"[{i+1}/{len(df)}] {sid} -> hippo={vols.get('hippocampal_volume')} "
              f"vent={vols.get('ventricle_volume')}", flush=True)
        if args.checkpoint_every and len(rows) % args.checkpoint_every == 0:
            emit_durable(args.out)
        if args.limit and len(rows) >= args.limit:
            print(f"[limit] reached {args.limit} -- stopping (smoke test).", flush=True)
            break

    if not rows:
        sys.exit("No volumes produced -- check volume paths / FastSurfer install.")

    print(f"\n[done] wrote {args.out}: {len(rows)} subjects", flush=True)
    if missing:
        print(f"[warn] {len(missing)} volumes missing: {missing[:10]}"
              f"{'...' if len(missing) > 10 else ''}", flush=True)
    if failed:
        print(f"[warn] {len(failed)} volumes failed to segment: {failed}", flush=True)
    emit_durable(args.out)


if __name__ == "__main__":
    main()
