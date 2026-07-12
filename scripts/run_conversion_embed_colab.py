#!/usr/bin/env python3
"""
Resumable, GCS-checkpointed Colab driver for the MCI-conversion cohort embed.

This ONE file plays two roles, auto-detected at runtime (`/content` exists only on
a Colab runtime):

  * LOOP role (run on the Mac): a `colab start` -> `colab exec` -> resume loop. It
    refreshes the GCS `inputs/` scripts, then repeatedly provisions a GPU runtime and
    execs THIS script on it. A free-tier preemption or websocket drop just ends one
    `colab exec`; the loop starts a fresh runtime and re-execs, which RESUMES from the
    GCS checkpoints. It exits when the final embeddings CSV is present on GCS.

  * DRIVER role (run on Colab, shipped by `colab exec`): the actual work, made
    crash-safe by checkpointing EVERY UNIT to GCS in-region:
      1. auth GCS via /content/adc.json,
      2. pull inputs (crosswalk, manifest, embed scripts) + any prior checkpoints,
      3. pull already-converted NIfTIs from GCS -> skip re-unzip/convert of done RIDs
         (only pull the 7 GB raw zips + unzip if subjects still need converting),
      4. convert each remaining subject's anchor series -> T1.nii.gz and PUSH IT to
         GCS immediately (per-unit NIfTI checkpoint),
      5. seed the embeddings CSV from GCS and run the frozen NeuroJEPA embed with
         --resume; a background thread heart-beats (keepalive) AND uploads the partial
         CSV to GCS on a cadence (per-unit embedding checkpoint),
      6. push the final embeddings CSV to GCS so ANY future session can fuse with NO
         GPU:  gs://.../adni_conversion/adni_conversion_neurojepa_embeddings.csv

`colab exec` does NOT forward argv, so all paths/params are baked in (edit here).

Runbook:
  # secrets live only on the runtime, never on GCS; the LOOP uploads them each session
  ./.venv/bin/python scripts/run_conversion_embed_colab.py      # runs the loop locally
  # or drive a single session by hand (driver auto-detected on the runtime):
  colab exec --session <id> --timeout 3h scripts/run_conversion_embed_colab.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time

# --- GCS layout (shared by both roles) -------------------------------------
BUCKET_NAME = "neuroad-adni-project-flash-490419"
PROJECT = "project-flash-490419"


def _resolve_prefix():
    """Cohort GCS prefix, resolved for BOTH roles.

    `colab exec` runs the driver with a FRESH env on the runtime, so the loop's
    CONV_PREFIX does NOT cross over — the loop therefore also uploads it as
    `conv_prefix.txt`, which the driver reads here. Precedence: env (loop/local) ->
    uploaded file (driver on Colab) -> default.
    """
    p = os.environ.get("CONV_PREFIX")
    if p:
        return p.strip()
    for f in ("/content/conv_prefix.txt", "conv_prefix.txt"):
        if os.path.exists(f):
            return open(f).read().strip()
    return "adni_conversion"


# Cohort GCS prefix — set CONV_PREFIX (loop) to reuse this whole driver+loop for
# another cohort (e.g. adni_ad_expansion); the loop ships it to the runtime too.
PREFIX = _resolve_prefix()

# Per-unit checkpoint blobs. NIfTIs and the partial embeddings CSV live under
# checkpoints/ so a fresh runtime can pull them and skip finished work; the final
# CSV is the durable, GPU-free deliverable other sessions consume.
NIFTI_PREFIX = f"{PREFIX}/checkpoints/nifti"                 # /<rid>/T1.nii.gz
PARTIAL_CSV_BLOB = f"{PREFIX}/checkpoints/embeddings_partial.csv"
FINAL_BLOB = f"{PREFIX}/adni_conversion_neurojepa_embeddings.csv"

CONTENT = os.environ.get("ADNI_CONTENT", "/content")
OUT_CSV = f"{CONTENT}/adni_neurojepa_embeddings.csv"

# Cadence for the keepalive/uploader thread (seconds).
HEARTBEAT_EVERY = 45
UPLOAD_EVERY = 90


# ===========================================================================
# DRIVER role  (runs ON the Colab runtime)
# ===========================================================================

def _bucket():
    try:
        from google.cloud import storage
    except Exception:
        subprocess.run("pip install -q google-cloud-storage", shell=True, check=True)
        from google.cloud import storage
    return storage.Client(project=PROJECT).bucket(BUCKET_NAME)


def _pull_inputs(b):
    """Pull the small text inputs + both helper scripts (in-region, fast)."""
    print("[gcs] pulling inputs ...", flush=True)
    for name in ("inputs/crosswalk.csv", "inputs/manifest_full.csv",
                 "inputs/neurojepa_embed_colab.py",
                 "inputs/adni_colab_dicom_to_embed.py"):
        dst = f"{CONTENT}/{os.path.basename(name)}"
        b.blob(f"{PREFIX}/{name}").download_to_filename(dst)
        print(f"    {name} -> {dst}", flush=True)


def _pull_nifti_checkpoints(b, chain):
    """Download every already-converted T1.nii.gz from GCS into OUT_MRI.

    Returns the set of RID strings that are now present locally, so convert() can
    skip them and a fully-converted cohort skips the 7 GB unzip entirely.
    """
    os.makedirs(chain.OUT_MRI, exist_ok=True)
    done = set()
    for blob in b.list_blobs(prefix=f"{NIFTI_PREFIX}/"):
        if not blob.name.endswith("/T1.nii.gz"):
            continue
        rid = blob.name.split("/")[-2]
        dest_dir = f"{chain.OUT_MRI}/{rid}"
        os.makedirs(dest_dir, exist_ok=True)
        dest = f"{dest_dir}/T1.nii.gz"
        if not os.path.exists(dest):
            blob.download_to_filename(dest)
        done.add(str(rid))
    print(f"[ckpt] pulled {len(done)} converted NIfTIs from GCS", flush=True)
    return done


def _pull_raw_zips(b):
    """Pull the raw DICOM zips (7 GB, in-region) — only when conversion is needed."""
    for name in ("raw/part1.zip", "raw/part2.zip"):
        dst = f"{CONTENT}/{os.path.basename(name)}"
        print(f"[gcs] pulling {name} -> {dst} (large, in-region) ...", flush=True)
        b.blob(f"{PREFIX}/{name}").download_to_filename(dst)


def _convert_with_checkpoints(b, chain, done):
    """Convert every not-yet-done subject and push each T1.nii.gz to GCS at once.

    Reuses the anchor-series finder from adni_colab_dicom_to_embed so the exact
    I<IMAGEUID> series logic (and MPRAGE/FSPGR fallback) stays in one place; the
    only addition here is the per-unit GCS upload so a preemption mid-convert never
    loses a finished volume.
    """
    import glob
    import tempfile

    import pandas as pd

    xw = pd.read_csv(chain.XWALK)
    conv = miss = pushed = 0
    for _, r in xw.iterrows():
        rid = str(int(r["RID"]))
        dest_dir = f"{chain.OUT_MRI}/{rid}"
        dest = f"{dest_dir}/T1.nii.gz"
        if rid in done or os.path.exists(dest):
            conv += 1
            continue
        os.makedirs(dest_dir, exist_ok=True)
        series = chain._find_series(r.get("IMAGEUID"), str(r["PTID"]))
        if series is None:
            miss += 1
            continue
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["dcm2niix", "-z", "y", "-f", "t1", "-o", tmp, series],
                           capture_output=True, text=True)
            niis = sorted(glob.glob(f"{tmp}/*.nii.gz"), key=os.path.getsize, reverse=True)
            if not niis:
                miss += 1
                continue
            shutil.copy2(niis[0], dest)
        # per-unit checkpoint: push this volume before touching the next subject.
        b.blob(f"{NIFTI_PREFIX}/{rid}/T1.nii.gz").upload_from_filename(dest)
        done.add(rid)
        conv += 1
        pushed += 1
        if pushed % 25 == 0:
            print(f"[convert] {conv} converted ({pushed} new pushed to GCS)", flush=True)
    print(f"[convert] converted={conv} missing={miss} "
          f"(new NIfTIs checkpointed to GCS this run={pushed})", flush=True)
    return conv


class _Keepalive:
    """Background heartbeat + partial-CSV uploader.

    Two jobs in one thread: (a) print a heartbeat so the exec websocket never looks
    idle during long CPU steps (unzip / convert / skull-strip), and (b) upload the
    in-progress embeddings CSV to GCS on a cadence, so even a hard runtime kill leaves
    a recoverable partial off-runtime. Uploads a temp copy to avoid shipping a
    half-written file while the embed process rewrites OUT_CSV.
    """

    def __init__(self, b):
        self._b = b
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._max_rows = 0   # monotonic high-water mark; never regress the checkpoint

    def start(self):
        self._t.start()

    def stop(self):
        self._stop.set()
        self._t.join(timeout=UPLOAD_EVERY + 10)
        self._upload_partial()  # one final flush

    def _upload_partial(self):
        if not os.path.exists(OUT_CSV):
            return
        try:
            tmp = OUT_CSV + ".ckpt"
            shutil.copy2(OUT_CSV, tmp)
            # NEVER regress the checkpoint: pandas rewrites OUT_CSV truncate-then-write
            # after every subject, so a copy can catch it mid-write (0 rows). Only push
            # if the snapshot has >= the most rows we've ever checkpointed.
            rows = _csv_rows(tmp)
            if rows <= 0 or rows < self._max_rows:
                os.remove(tmp)
                return
            self._b.blob(PARTIAL_CSV_BLOB).upload_from_filename(tmp)
            self._max_rows = rows
            os.remove(tmp)
        except Exception as exc:  # noqa: BLE001 -- checkpointing must never crash the run
            print(f"[ckpt] partial-CSV upload skipped: {exc!r}", flush=True)

    def _run(self):
        last_upload = 0.0
        while not self._stop.is_set():
            self._stop.wait(HEARTBEAT_EVERY)
            n = _csv_rows(OUT_CSV)
            print(f"[keepalive] alive; {n} subjects embedded so far", flush=True)
            if time.time() - last_upload >= UPLOAD_EVERY:
                self._upload_partial()
                last_upload = time.time()


def _csv_rows(path):
    if not os.path.exists(path):
        return 0
    try:
        with open(path) as fh:
            return max(0, sum(1 for _ in fh) - 1)
    except Exception:  # noqa: BLE001
        return 0


def run_driver():
    """The on-runtime work: pull -> resume-aware convert -> embed -> push."""
    adc = os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", f"{CONTENT}/adc.json")
    if not os.path.exists(adc):
        sys.exit(f"Missing ADC at {adc} (the loop uploads it as adc.json each session).")

    b = _bucket()
    _pull_inputs(b)

    # Import the shared conversion/embed helpers we just pulled to /content.
    sys.path.insert(0, CONTENT)
    import adni_colab_dicom_to_embed as chain  # noqa: E402

    keep = _Keepalive(b)
    keep.start()
    try:
        # 1) Resume converts from GCS; only unzip + convert what's still missing.
        done = _pull_nifti_checkpoints(b, chain)
        import pandas as pd
        need = {str(int(x)) for x in pd.read_csv(chain.XWALK)["RID"].dropna()}
        remaining = need - done
        if remaining:
            print(f"[convert] {len(remaining)} subjects still need conversion "
                  f"({len(done)} already checkpointed).", flush=True)
            chain.install_dcm2niix()
            if shutil.which("dcm2niix") is None:
                sys.exit("dcm2niix unavailable after apt + pip fallback.")
            _pull_raw_zips(b)
            chain.unzip_all()
            if _convert_with_checkpoints(b, chain, done) == 0:
                sys.exit("No volumes converted — check zip layout / crosswalk.")
        else:
            print(f"[convert] all {len(done)} subjects already converted — "
                  "skipping the 7 GB unzip.", flush=True)

        # 2) Build the ready manifest from whatever converted.
        if chain.build_manifest() == 0:
            sys.exit("Manifest empty — no converted NIfTI matched the crosswalk RIDs.")

        # 3) Seed the embeddings CSV from GCS so the embed's --resume skips done rows.
        pb = b.blob(PARTIAL_CSV_BLOB)
        if pb.exists():
            pb.download_to_filename(OUT_CSV)
            print(f"[ckpt] seeded {OUT_CSV} from GCS partial "
                  f"({_csv_rows(OUT_CSV)} rows) — embed will resume.", flush=True)

        # 4) Frozen NeuroJEPA embed (inherits stdout for durable ===CSVGZ=== markers;
        #    writes OUT_CSV incrementally; --resume skips seeded rows). The keepalive
        #    thread uploads OUT_CSV to GCS on its cadence throughout.
        chain.embed()
    finally:
        keep.stop()

    # 5) Durable deliverable: final CSV to both the partial slot and the final slot.
    if os.path.exists(OUT_CSV):
        b.blob(PARTIAL_CSV_BLOB).upload_from_filename(OUT_CSV)
        b.blob(FINAL_BLOB).upload_from_filename(OUT_CSV)
        print(f"[push] embeddings CSV ({_csv_rows(OUT_CSV)} subjects) -> "
              f"gs://{BUCKET_NAME}/{FINAL_BLOB}", flush=True)
    else:
        print(f"[push] WARNING: {OUT_CSV} missing — nothing pushed", flush=True)
    print("[done] conversion embeddings on GCS (future sessions: gcloud storage cp "
          "the CSV — no GPU needed).", flush=True)


# ===========================================================================
# LOOP role  (runs on the Mac; drives Colab sessions with resume)
# ===========================================================================

# NOTE: `__file__` is UNDEFINED when `colab exec` runs this file in a Jupyter
# kernel, and this module executes top-to-bottom on the runtime (driver role) too —
# so guard it, or the driver crashes here before main() ever dispatches.
try:
    _REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    _REPO = os.getcwd()
_SELF_REMOTE = "scripts/run_conversion_embed_colab.py"
MAX_ATTEMPTS = int(os.environ.get("CONV_LOOP_ATTEMPTS", "12"))
EXEC_TIMEOUT = os.environ.get("CONV_EXEC_TIMEOUT", "3h")
GPU = os.environ.get("CONV_GPU", "t4")


def _gcloud_env():
    env = dict(os.environ)
    env.setdefault("GOOGLE_APPLICATION_CREDENTIALS",
                   os.path.expanduser("~/.config/gcloud/application_default_credentials.json"))
    return env


def _final_exists() -> bool:
    """True once the final embeddings CSV is on GCS (loop's exit condition)."""
    r = subprocess.run(
        ["gcloud", "storage", "ls", f"gs://{BUCKET_NAME}/{FINAL_BLOB}"],
        capture_output=True, text=True, env=_gcloud_env())
    return r.returncode == 0


def _progress_marker() -> int:
    """Total NIfTI + partial-CSV checkpoints on GCS — the loop's progress signal.

    Used to distinguish a genuine preemption (progress advanced, keep resuming) from
    an instant crash (nothing moved) so a code bug can't burn every attempt.
    """
    r = subprocess.run(
        ["gcloud", "storage", "ls", "-r",
         f"gs://{BUCKET_NAME}/{PREFIX}/checkpoints/"],
        capture_output=True, text=True, env=_gcloud_env())
    return len([ln for ln in r.stdout.splitlines() if ln.strip().endswith((".nii.gz", ".csv"))])


def _refresh_inputs():
    """Re-upload the current local scripts to GCS inputs/ so the runtime pulls latest."""
    env = _gcloud_env()
    pairs = [
        (f"{_REPO}/scripts/neurojepa_embed_colab.py",
         f"gs://{BUCKET_NAME}/{PREFIX}/inputs/neurojepa_embed_colab.py"),
        (f"{_REPO}/scripts/adni_colab_dicom_to_embed.py",
         f"gs://{BUCKET_NAME}/{PREFIX}/inputs/adni_colab_dicom_to_embed.py"),
    ]
    for local, remote in pairs:
        subprocess.run(["gcloud", "storage", "cp", local, remote],
                       check=False, env=env)
    print("[loop] refreshed GCS inputs/ scripts", flush=True)


def _start_session() -> str | None:
    """Provision a GPU runtime; return its session id (parsed defensively)."""
    import json
    import re
    r = subprocess.run(["colab", "start", "--gpu", GPU, "--json"],
                       capture_output=True, text=True, env=_gcloud_env())
    out = (r.stdout or "") + (r.stderr or "")
    try:
        data = json.loads(r.stdout)
        for k in ("sessionId", "session_id", "session", "id"):
            if isinstance(data, dict) and data.get(k):
                return str(data[k])
    except Exception:  # noqa: BLE001 -- fall back to a regex over plain output
        pass
    m = re.search(r"\b([0-9a-f]{8,}(?:-[0-9a-f]{4,}){0,4})\b", out)
    if m:
        return m.group(1)
    print(f"[loop] could not parse session id from:\n{out}", flush=True)
    return None


def _upload_secrets(session: str):
    env = _gcloud_env()
    adc = env["GOOGLE_APPLICATION_CREDENTIALS"]
    subprocess.run(["colab", "upload", "--session", session, adc, "adc.json"],
                   check=False, env=env)
    tok = f"{_REPO}/hf_token.txt"
    if os.path.exists(tok):
        subprocess.run(["colab", "upload", "--session", session, tok, "hf_token.txt"],
                       check=False, env=env)
    # Ship the cohort prefix to the runtime — `colab exec` won't forward CONV_PREFIX,
    # so the driver reads it from this uploaded file (see _resolve_prefix).
    pf = os.path.join(_REPO, ".conv_prefix.txt")
    with open(pf, "w") as fh:
        fh.write(PREFIX)
    subprocess.run(["colab", "upload", "--session", session, pf, "conv_prefix.txt"],
                   check=False, env=env)


def run_loop():
    """colab start -> upload secrets -> exec (driver resumes from GCS) -> repeat."""
    if _final_exists():
        print(f"[loop] final CSV already on GCS ({FINAL_BLOB}); nothing to do.", flush=True)
        return
    _refresh_inputs()
    provided = os.environ.get("CONV_SESSION")  # reuse an already-started runtime once
    fast_fails = 0                              # consecutive no-progress crashes
    for attempt in range(1, MAX_ATTEMPTS + 1):
        print(f"\n[loop] === attempt {attempt}/{MAX_ATTEMPTS} ===", flush=True)
        if attempt == 1 and provided:
            session = provided
            print(f"[loop] reusing provided session {session}", flush=True)
        else:
            session = _start_session()
        if not session:
            print("[loop] no session; retrying in 30s ...", flush=True)
            time.sleep(30)
            continue
        print(f"[loop] session {session}: uploading secrets + exec driver", flush=True)
        _upload_secrets(session)
        before = _progress_marker()
        t0 = time.time()
        # Stream the driver's stdout straight through (durable markers land locally).
        subprocess.run(
            ["colab", "exec", "--session", session, "--timeout", EXEC_TIMEOUT,
             _SELF_REMOTE],
            check=False, env=_gcloud_env())
        subprocess.run(["colab", "stop", "--session", session],
                       check=False, env=_gcloud_env())
        if _final_exists():
            print(f"[loop] final CSV present on GCS after attempt {attempt}. Done.",
                  flush=True)
            return
        # Crash-loop guard: an exec that returns fast AND advanced no checkpoints is a
        # bug, not a preemption — bail after 2 in a row instead of burning runtimes.
        elapsed = time.time() - t0
        if elapsed < 120 and _progress_marker() <= before:
            fast_fails += 1
            print(f"[loop] exec returned in {elapsed:.0f}s with no new checkpoints "
                  f"(fast-fail {fast_fails}/2) — likely a crash, not a preemption.",
                  flush=True)
            if fast_fails >= 2:
                print("[loop] ABORTING: two consecutive no-progress crashes. "
                      "Fix the driver, then relaunch.", flush=True)
                return
        else:
            fast_fails = 0
            print("[loop] final CSV not yet on GCS — resuming in a fresh session ...",
                  flush=True)
        time.sleep(10)
    print(f"[loop] exhausted {MAX_ATTEMPTS} attempts without a final CSV; "
          "inspect the logs / GCS checkpoints.", flush=True)


# ===========================================================================
# Role dispatch: /content exists only on a Colab runtime.
# ===========================================================================

def main():
    role = os.environ.get("CONV_ROLE")
    if role == "driver" or (role is None and os.path.isdir("/content")):
        run_driver()
    else:
        run_loop()


if __name__ == "__main__":
    main()
