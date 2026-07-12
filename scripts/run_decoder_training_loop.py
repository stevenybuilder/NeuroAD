#!/usr/bin/env python3
"""
run_decoder_training_loop.py — LOCAL, unattended orchestration driver that trains the
NeuroJEPA decoder to completion across Colab's ~10-22 min runtime reclaims, using ONLY
Colab units (no paid GCE).

Why this exists
---------------
``scripts/train_neurojepa_decoder.py`` already survives a runtime death: it checkpoints
model+optimizer+epoch to GCS every ``--ckpt-every-sec`` and RESUMES from the latest GCS
checkpoint on start. What was missing is the *outer* loop — something that keeps
re-provisioning a fresh Colab GPU whenever the last one is reclaimed and re-launches the
(resumable) trainer, so a full ~1-2 h run completes with nobody watching. This script is
that loop. It runs on the Mac and drives the official ``colab`` CLI:

  1. ``colab start --gpu <t4|l4|a100>``           -> capture the session id
  2. stage the trainer + its two runtime modules (neurojepa_decoder.py, gcs_store.py) +
     the ADC creds (~/.config/gcloud/application_default_credentials.json -> /content/
     adc.json) to the runtime, arrange them into an importable ``neuroad.integrations``
     package under /content/src, and pull decoder_data/*.npz from GCS onto the runtime.
  3. launch train_neurojepa_decoder.py DETACHED (subprocess.Popen, start_new_session=True)
     writing to /content/train.log, with GOOGLE_APPLICATION_CREDENTIALS set — so the
     trainer outlives the ``colab exec`` cell that launched it.
  4. keep-alive: every ``--poll-sec`` (default 45 s) exec a tiny poll on the runtime that
     tails /content/train.log and reports whether the trainer PID is still alive. The poll
     both watches progress ([epoch]/[ckpt]) and keeps the runtime from idling out.
  5. when the runtime dies / is reclaimed (the poll exec fails / loses its sentinel),
     detect it and GOTO (1). The trainer resumes from the GCS checkpoint automatically, so
     nothing is lost — an unstable ~20-min runtime becomes usable training time across
     ~15-25 resume cycles.

Stop conditions: /content/train.log shows the anchored terminal marker ``[train] done.``
(success), OR ``--max-wall-sec`` elapses (honest give-up: the run is simply not finished).

Honesty / robustness
--------------------
  * This driver NEVER fabricates progress — it prints only what the runtime's train.log
    actually contains. If GCS holds no decoder_data/*.npz the runtime says so and the
    driver aborts with a clear message rather than looping forever on empty data.
  * Runtime-death detection is by the ABSENCE of an explicit ``===POLL_OK===`` sentinel
    from a successful poll exec, NOT by grepping the log for scary words — in particular
    it never greps a bare ``FATAL`` (which would match innocuous ``FATAL=False`` status
    fields). "Training finished" is detected only by the anchored token ``[train] done.``.
  * A transient poll error (websocket EOF mid-cell) is retried once before the runtime is
    declared dead, so we don't throw away a live GPU on a flaky frame.
  * Every colab CLI call is wrapped with a Python-level timeout (macOS has no ``timeout``
    binary) and can never raise out of the loop.
  * On any exit path (success, timeout, Ctrl-C, unexpected error) the active runtime is
    released with ``colab stop`` so Colab units stop burning.

This is an orchestration script (it shells out to ``colab`` and needs a network + the
CLI to do anything). It imports nothing from the repo and is safe to syntax-check offline.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

# ---------------------------------------------------------------------------
# Local repo paths we stage to the runtime.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
TRAIN_SCRIPT = os.path.join(_REPO, "scripts", "train_neurojepa_decoder.py")
DECODER_MOD = os.path.join(_REPO, "src", "neuroad", "integrations", "neurojepa_decoder.py")
GCS_MOD = os.path.join(_REPO, "src", "neuroad", "integrations", "gcs_store.py")
DEFAULT_ADC = os.path.expanduser(
    "~/.config/gcloud/application_default_credentials.json")

# Sentinels the remote helper scripts emit (kept in sync with the sources below).
POLL_OK = "===POLL_OK==="
LOGTAIL_START = "===LOGTAIL_START==="
LOGTAIL_END = "===LOGTAIL_END==="
SETUP_OK = "===SETUP_OK==="
SETUP_NODATA = "===SETUP_NODATA==="
LAUNCH_OK = "===LAUNCH_OK==="
DONE_MARKER = "[train] done."   # anchored terminal marker printed by the trainer

# ---------------------------------------------------------------------------
# Remote helper scripts (executed ON the runtime via `colab exec <file>`). They read
# their config from /content/loop_config.json (uploaded each fresh runtime) so we never
# have to string-interpolate/quote code. They import the SAME modules we staged, so the
# runtime uses the exact code in this repo — no drift.
# ---------------------------------------------------------------------------

_REMOTE_SETUP = r'''
# setup.py (remote) — arrange staged files into an importable package tree, install the
# GCS client, and pull decoder_data/*.npz from GCS. Prints ===SETUP_OK=== <n_npz> on
# success, or ===SETUP_NODATA=== if the bucket holds no .npz (honest abort upstream).
import json, os, shutil, subprocess, sys

cfg = json.load(open("/content/loop_config.json"))
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/content/adc.json"
if cfg.get("bucket"):
    os.environ["NEUROAD_GCS_BUCKET"] = cfg["bucket"]

# Build /content/src/neuroad/integrations/{__init__, neurojepa_decoder, gcs_store}.
pkg = "/content/src/neuroad/integrations"
os.makedirs(pkg, exist_ok=True)
open("/content/src/neuroad/__init__.py", "a").close()
open(os.path.join(pkg, "__init__.py"), "a").close()
for name in ("neurojepa_decoder.py", "gcs_store.py"):
    src = os.path.join("/content", name)
    if os.path.exists(src):
        shutil.copy(src, os.path.join(pkg, name))
sys.path.insert(0, "/content/src")

# GCS python client — Colab usually lacks it. Keep Colab's CUDA torch untouched.
try:
    import google.cloud.storage  # noqa: F401
except Exception:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "google-cloud-storage"], check=False)

from neuroad.integrations import gcs_store as gcs  # noqa: E402

prefix = cfg.get("data_prefix", "decoder_data/")
if not prefix.endswith("/"):
    prefix += "/"
data_dir = "/content/decoder_data"
os.makedirs(data_dir, exist_ok=True)
names = [n for n in gcs.list_prefix(prefix) if n.endswith(".npz")]
print("[setup] bucket=%s prefix=%s objects=%d" % (gcs.bucket_name(), prefix, len(names)),
      flush=True)

pulled, skipped, failed = 0, 0, 0
for n in names:
    dst = os.path.join(data_dir, os.path.basename(n))
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        skipped += 1
        continue
    tmp = dst + ".tmp"
    try:
        gcs.download(n, tmp)           # download to .tmp then atomically rename
        os.replace(tmp, dst)
        pulled += 1
    except Exception as exc:
        failed += 1
        try:
            os.remove(tmp)
        except OSError:
            pass
        print("[setup] FAILED %s: %r" % (n, exc), flush=True)

have = len([f for f in os.listdir(data_dir) if f.endswith(".npz")])
print("[setup] pulled=%d skipped=%d failed=%d have=%d" % (pulled, skipped, failed, have),
      flush=True)
if have == 0:
    print("%s" % "===SETUP_NODATA===", flush=True)
else:
    print("%s %d" % ("===SETUP_OK===", have), flush=True)
'''

_REMOTE_LAUNCH = r'''
# launch.py (remote) — start the resumable trainer DETACHED so it survives the exec cell.
# start_new_session=True puts it in its own session/process group; when this cell's
# websocket closes the trainer keeps running until the runtime itself dies.
import json, os, subprocess, sys

cfg = json.load(open("/content/loop_config.json"))
env = dict(os.environ)
env["GOOGLE_APPLICATION_CREDENTIALS"] = "/content/adc.json"
env["PYTHONPATH"] = "/content/src" + (
    os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
if cfg.get("bucket"):
    env["NEUROAD_GCS_BUCKET"] = cfg["bucket"]

cmd = [sys.executable, "/content/train_neurojepa_decoder.py",
       "--data-dir", "/content/decoder_data",
       "--gcs-ckpt", str(cfg["gcs_ckpt"]),
       "--size", str(cfg["size"]),
       "--batch", str(cfg["batch"]),
       "--epochs", str(cfg["epochs"]),
       "--lr", str(cfg["lr"]),
       "--ckpt-every-sec", str(cfg["ckpt_every_sec"]),
       "--val-frac", str(cfg["val_frac"])]

log = open("/content/train.log", "ab")
proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                        start_new_session=True, cwd="/content", env=env)
with open("/content/train.pid", "w") as f:
    f.write(str(proc.pid))
print("%s pid=%d" % ("===LAUNCH_OK===", proc.pid), flush=True)
'''

_REMOTE_POLL = r'''
# poll.py (remote) — tail /content/train.log and report whether the trainer PID is alive.
# A successful run always ends with ===POLL_OK===; its ABSENCE (exec failed / runtime
# reclaimed) is how the driver detects a dead runtime. We never grep the log for status
# keywords here — the driver keys off the anchored [train] done. marker only.
import os

LOG = "/content/train.log"
PIDF = "/content/train.pid"

tail = ""
if os.path.exists(LOG):
    with open(LOG, "rb") as f:
        tail = f.read()[-6000:].decode("utf-8", "replace")

alive = False
if os.path.exists(PIDF):
    try:
        pid = int(open(PIDF).read().strip())
        os.kill(pid, 0)
        alive = True
    except Exception:
        alive = False

print("===LOGTAIL_START===")
print(tail)
print("===LOGTAIL_END===")
print("PID_ALIVE=%s" % alive)
print("===POLL_OK===")
'''


# ---------------------------------------------------------------------------
# colab CLI plumbing — every call timeout-guarded and non-raising.
# ---------------------------------------------------------------------------
def _run(cmd, timeout):
    """Run a command; return (rc, stdout, stderr). Never raises; timeout -> rc 124."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        err = exc.stderr or ""
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        if isinstance(err, bytes):
            err = err.decode("utf-8", "replace")
        return 124, out, err + "\n[timeout]"
    except Exception as exc:  # noqa: BLE001 — CLI missing, etc.
        return 127, "", repr(exc)


def colab_start(colab, gpu, timeout):
    """`colab start --gpu <gpu>` -> session id, parsed from the 'Session: <id>' line."""
    rc, out, err = _run([colab, "start", "--gpu", gpu], timeout)
    blob = out + "\n" + err
    m = re.search(r"Session:\s*(\S+)", blob)
    if rc == 0 and m:
        return m.group(1)
    # Fallback: some builds print 'Runtime started: T4 (<id>)'.
    m2 = re.search(r"Runtime started:.*\((\S+)\)", blob)
    if rc == 0 and m2:
        return m2.group(1)
    print("[loop] colab start failed (rc=%d):\n%s" % (rc, blob.strip()[-500:]),
          flush=True)
    return None


def colab_upload(colab, sid, local, remote, timeout):
    rc, out, err = _run(
        [colab, "upload", "--session", sid, local, remote], timeout)
    if rc != 0:
        print("[loop] upload %s -> %s failed (rc=%d): %s"
              % (local, remote, rc, (out + err).strip()[-300:]), flush=True)
    return rc == 0


def colab_exec_file(colab, sid, local_py, timeout):
    """Run a LOCAL .py on the runtime; return (rc, combined_output)."""
    rc, out, err = _run(
        [colab, "exec", "--session", sid, local_py], timeout)
    return rc, out + ("\n" + err if err else "")


def colab_stop(colab, sid, timeout=120):
    """Best-effort release. Session-less stop releases the active runtime."""
    cmd = [colab, "stop"]
    if sid:
        cmd += ["--session", sid]
    _run(cmd, timeout)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _between(text, start, end):
    i = text.find(start)
    if i < 0:
        return ""
    i += len(start)
    j = text.find(end, i)
    return text[i:j] if j >= 0 else text[i:]


def _write_temp(name, source, workdir):
    path = os.path.join(workdir, name)
    with open(path, "w") as f:
        f.write(source)
    return path


def _print_new_tail(prev, tail):
    """Print only the log lines we haven't shown yet (keeps the console readable)."""
    for line in tail.splitlines():
        if line and line not in prev:
            print("    | " + line, flush=True)
    return set(tail.splitlines())


# ---------------------------------------------------------------------------
# main loop
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default="t4", choices=["t4", "l4", "a100"])
    ap.add_argument("--colab-bin", default="colab", help="path to the colab CLI")
    ap.add_argument("--max-wall-sec", type=int, default=8 * 3600,
                    help="overall wall-clock budget across all runtimes (default 8h)")
    ap.add_argument("--poll-sec", type=int, default=45,
                    help="keep-alive / log-tail poll interval (default 45s)")
    ap.add_argument("--start-timeout", type=int, default=600,
                    help="seconds to wait for `colab start` (default 600)")
    ap.add_argument("--setup-timeout", type=int, default=2400,
                    help="seconds for stage+GCS data pull per fresh runtime (default 40m)")
    ap.add_argument("--exec-timeout", type=int, default=180,
                    help="seconds for a launch/poll exec (default 180)")
    ap.add_argument("--max-relaunch", type=int, default=3,
                    help="consecutive in-runtime trainer relaunches before we recycle "
                         "the whole runtime (default 3)")
    # ADC + GCS
    ap.add_argument("--adc", default=DEFAULT_ADC,
                    help="local ADC creds json to stage to /content/adc.json")
    ap.add_argument("--bucket", default=None,
                    help="override GCS bucket (else the module default) on the runtime")
    ap.add_argument("--data-prefix", default="decoder_data/",
                    help="GCS prefix holding the per-subject <sid>.npz")
    # trainer passthrough (mirror train_neurojepa_decoder.py defaults)
    ap.add_argument("--gcs-ckpt", default="decoder_ckpt/latest.pt")
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ckpt-every-sec", type=int, default=180)
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()

    colab = args.colab_bin

    # Preflight: the staged files must exist locally (fail fast, honestly).
    missing = [p for p in (TRAIN_SCRIPT, DECODER_MOD, GCS_MOD, args.adc)
               if not os.path.exists(p)]
    if missing:
        sys.exit("[loop] missing local file(s) required to stage:\n  "
                 + "\n  ".join(missing))

    workdir = tempfile.mkdtemp(prefix="decoder_loop_")
    cfg = {
        "gcs_ckpt": args.gcs_ckpt, "data_prefix": args.data_prefix,
        "bucket": args.bucket, "size": args.size, "batch": args.batch,
        "epochs": args.epochs, "lr": args.lr,
        "ckpt_every_sec": args.ckpt_every_sec, "val_frac": args.val_frac,
    }
    cfg_path = os.path.join(workdir, "loop_config.json")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)
    setup_py = _write_temp("setup.py", _REMOTE_SETUP, workdir)
    launch_py = _write_temp("launch.py", _REMOTE_LAUNCH, workdir)
    poll_py = _write_temp("poll.py", _REMOTE_POLL, workdir)

    # local -> remote staging list (remote paths land under /content).
    uploads = [
        (TRAIN_SCRIPT, "train_neurojepa_decoder.py"),
        (DECODER_MOD, "neurojepa_decoder.py"),
        (GCS_MOD, "gcs_store.py"),
        (args.adc, "adc.json"),
        (cfg_path, "loop_config.json"),
    ]

    deadline = time.time() + args.max_wall_sec
    cycle = 0
    print("[loop] starting; gpu=%s max_wall=%ds poll=%ds ckpt=gs://.../%s"
          % (args.gpu, args.max_wall_sec, args.poll_sec, args.gcs_ckpt), flush=True)

    sid = None
    exit_code = 1
    try:
        while time.time() < deadline:
            cycle += 1
            remaining = int(deadline - time.time())
            print("\n[loop] === runtime cycle %d (%ds budget left) ==="
                  % (cycle, remaining), flush=True)

            # (1) provision a fresh GPU
            sid = colab_start(colab, args.gpu, args.start_timeout)
            if not sid:
                print("[loop] could not start a runtime; retrying in 30s", flush=True)
                time.sleep(30)
                continue
            print("[loop] session=%s" % sid, flush=True)

            # (2) stage files
            ok = all(colab_upload(colab, sid, lp, rp, args.exec_timeout)
                     for lp, rp in uploads)
            if not ok:
                print("[loop] staging failed — recycling runtime", flush=True)
                colab_stop(colab, sid)
                sid = None
                continue

            # (2b) arrange package + pull decoder_data/*.npz from GCS
            rc, out = colab_exec_file(colab, sid, setup_py, args.setup_timeout)
            if SETUP_NODATA in out:
                print("[loop] GCS holds no %s*.npz — nothing to train on. Aborting "
                      "(run the data-prep step first)." % args.data_prefix, flush=True)
                colab_stop(colab, sid)
                sid = None
                exit_code = 2
                break
            if rc != 0 or SETUP_OK not in out:
                print("[loop] setup/data-pull did not complete (rc=%d) — likely a "
                      "reclaim mid-pull; recycling.\n%s"
                      % (rc, out.strip()[-600:]), flush=True)
                colab_stop(colab, sid)
                sid = None
                continue
            print("[loop] setup ok: %s" % _between(out, SETUP_OK, "\n").strip(),
                  flush=True)

            # (3) launch the detached, resumable trainer
            rc, out = colab_exec_file(colab, sid, launch_py, args.exec_timeout)
            if rc != 0 or LAUNCH_OK not in out:
                print("[loop] trainer launch failed (rc=%d) — recycling.\n%s"
                      % (rc, out.strip()[-600:]), flush=True)
                colab_stop(colab, sid)
                sid = None
                continue
            print("[loop] launched trainer: %s"
                  % _between(out, LAUNCH_OK, "\n").strip(), flush=True)

            # (4) keep-alive poll loop
            seen_lines = set()
            relaunches = 0
            poll_fail_streak = 0
            runtime_alive = True
            while runtime_alive and time.time() < deadline:
                time.sleep(args.poll_sec)
                rc, out = colab_exec_file(colab, sid, poll_py, args.exec_timeout)
                if rc != 0 or POLL_OK not in out:
                    poll_fail_streak += 1
                    if poll_fail_streak >= 2:
                        print("[loop] runtime %s appears dead/reclaimed (2 failed "
                              "polls) — recycling." % sid, flush=True)
                        runtime_alive = False
                        break
                    print("[loop] transient poll error (rc=%d) — retrying once" % rc,
                          flush=True)
                    continue
                poll_fail_streak = 0

                tail = _between(out, LOGTAIL_START, LOGTAIL_END)
                seen_lines = _print_new_tail(seen_lines, tail)

                if DONE_MARKER in tail:
                    print("\n[loop] '%s' — TRAINING COMPLETE." % DONE_MARKER, flush=True)
                    colab_stop(colab, sid)
                    sid = None
                    exit_code = 0
                    return exit_code
                if "PID_ALIVE=False" in out:
                    # runtime alive but the trainer process is gone and not done —
                    # relaunch it (it resumes from the GCS checkpoint).
                    relaunches += 1
                    if relaunches > args.max_relaunch:
                        print("[loop] trainer died %d times on this runtime — recycling "
                              "the runtime." % relaunches, flush=True)
                        runtime_alive = False
                        break
                    print("[loop] trainer PID gone (not done) — relaunch %d/%d"
                          % (relaunches, args.max_relaunch), flush=True)
                    rc, out = colab_exec_file(colab, sid, launch_py, args.exec_timeout)
                    if rc != 0 or LAUNCH_OK not in out:
                        print("[loop] relaunch failed — recycling runtime", flush=True)
                        runtime_alive = False
                        break
                else:
                    relaunches = 0  # healthy training tick

            # cycle ended — release this runtime before starting a new one
            if sid:
                colab_stop(colab, sid)
                sid = None

        else:
            print("\n[loop] max wall-clock (%ds) reached without '%s' — stopping. "
                  "The GCS checkpoint is intact; re-run to continue."
                  % (args.max_wall_sec, DONE_MARKER), flush=True)

    except KeyboardInterrupt:
        print("\n[loop] interrupted — releasing runtime.", flush=True)
    finally:
        if sid:
            colab_stop(colab, sid)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
