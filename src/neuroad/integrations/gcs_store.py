"""
gcs_store — durable object storage on Google Cloud Storage for the pipeline.

Replaces the Drive+rclone workaround (interactive OAuth, whole-file download) with
headless, service-account/ADC auth and streaming access. This is the durable home
for the artifacts that MUST survive a runtime death (the Colab CLI runtimes here
were reclaimed at ~10-22 min, wiping /content):

  * converted T1w NIfTI (so a re-run skips the ~15-min unzip+convert),
  * partial embedding / volume CSVs checkpointed mid-job (resume, not restart),
  * decoder training checkpoints.

Default bucket: env ``NEUROAD_GCS_BUCKET`` else ``neuroad-adni-project-flash-490419``.
Auth: Application Default Credentials (``gcloud auth application-default login`` or a
service-account key in ``GOOGLE_APPLICATION_CREDENTIALS``). Every function degrades
honestly — a missing bucket/credential raises a clear error rather than silently
succeeding, but ``exists`` returns False and ``try_download`` returns None so callers
can branch without try/except.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_DEFAULT_BUCKET = "neuroad-adni-project-flash-490419"


def bucket_name() -> str:
    return os.environ.get("NEUROAD_GCS_BUCKET", _DEFAULT_BUCKET)


def _client():
    from google.cloud import storage  # lazy: keep import-time light / optional dep
    return storage.Client()


def _blob(path: str):
    return _client().bucket(bucket_name()).blob(path.lstrip("/"))


def upload(local_path, remote_path: str) -> str:
    """Upload a local file to ``gs://<bucket>/<remote_path>``; return the gs:// URI."""
    _blob(remote_path).upload_from_filename(str(local_path))
    return f"gs://{bucket_name()}/{remote_path.lstrip('/')}"


def download(remote_path: str, local_path) -> str:
    """Download an object to ``local_path`` (parents created). Returns local path."""
    p = Path(local_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    _blob(remote_path).download_to_filename(str(p))
    return str(p)


def exists(remote_path: str) -> bool:
    """True iff the object exists. Never raises — a bad credential/bucket → False."""
    try:
        return _blob(remote_path).exists()
    except Exception:
        return False


def try_download(remote_path: str, local_path) -> Optional[str]:
    """Download if present, else return None (for resume-if-checkpoint-exists)."""
    if not exists(remote_path):
        return None
    try:
        return download(remote_path, local_path)
    except Exception:
        return None


def list_prefix(prefix: str) -> list[str]:
    """List object names under a prefix. Returns [] on failure."""
    try:
        return [b.name for b in _client().list_blobs(bucket_name(), prefix=prefix)]
    except Exception:
        return []


def uri(remote_path: str) -> str:
    return f"gs://{bucket_name()}/{remote_path.lstrip('/')}"


__all__ = ["bucket_name", "upload", "download", "exists", "try_download",
           "list_prefix", "uri"]
