"""
structural_segmenter — FastSurfer structural-volume extractor for the pipeline.

Two layers, only one of which needs a GPU:

  1. ``parse_aseg_stats(path)`` — a PURE, offline, deterministic parser that turns a
     FreeSurfer/FastSurfer ``aseg.stats`` file into a normalized volume dict keyed by
     the fusion/contract schema names (``hippocampal_volume``, ``ventricle_volume``,
     ``whole_brain_volume``, ``cortex_volume``, ``intracranial_volume``). This has NO
     model dependency and is fully testable NOW against a bundled fixture.

  2. ``segment_volume(nifti)`` — the GPU path. It LAZY-imports torch and shells out to
     FastSurfer's ``run_fastsurfer.sh --seg_only``; it HONESTLY DEGRADES to ``None``
     (never fabricates volumes) whenever torch, a CUDA GPU, or the FastSurfer install
     is absent, or the segmentation itself fails. On success it parses the produced
     ``aseg.stats`` via layer 1 and returns the same normalized dict.

Why FastSurfer: Apache-2.0, ~1 min/volume on a GPU, and its aseg output is
FreeSurfer-``aseg.stats``-compatible — so the parser here is the ONLY schema glue
needed to feed structural volumes (hippocampal atrophy, ventricular enlargement,
cortical volume) into the same contract table the referee already consumes.

Compliance: FastSurfer weights are NEVER committed (they live only on the GPU
runtime); only the small derived volume table leaves the runtime. Every output is
provenance-stamped so a parsed number can never be mistaken for a fabricated one.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Provenance stamps + schema mapping
# ---------------------------------------------------------------------------

#: Provenance stamp for a dict parsed straight from an aseg.stats file (no model
#: was run in *this* process — the stats already existed).
SOURCE_ASEG = "fastsurfer_aseg_stats"
#: Provenance stamp for a dict produced by actually running FastSurfer here.
SOURCE_SEGMENT = "fastsurfer_segment"

#: Normalized volume keys we expose to the fusion/contract schema (mm^3).
VOLUME_KEYS = (
    "hippocampal_volume",   # bilateral hippocampus (Left+Right-Hippocampus)
    "ventricle_volume",     # lateral + inf-lat + 3rd + 4th ventricles
    "whole_brain_volume",   # BrainSegVolNotVent (== ADNI "WholeBrain")
    "cortex_volume",        # CortexVol (total cortical gray matter)
    "intracranial_volume",  # eTIV (estimated total intracranial volume; normalizer)
)

#: aseg.stats table StructNames summed into each subcortical volume key. Names are
#: matched case-insensitively so a FastSurfer/FreeSurfer minor-version rename of
#: casing does not silently drop a structure.
_HIPPOCAMPUS_STRUCTS = ("left-hippocampus", "right-hippocampus")
_VENTRICLE_STRUCTS = (
    "left-lateral-ventricle", "right-lateral-ventricle",
    "left-inf-lat-vent", "right-inf-lat-vent",
    "3rd-ventricle", "4th-ventricle", "5th-ventricle",
)

#: ``# Measure`` short-name -> the whole-brain/cortex/eTIV volume key it feeds.
#: The short name is the SECOND comma-field of a ``# Measure`` header line, e.g.
#: ``# Measure BrainSegNotVent, BrainSegVolNotVent, ..., 1130000.0, mm^3``.
_MEASURE_MAP = {
    "brainsegvolnotvent": "whole_brain_volume",
    "cortexvol": "cortex_volume",
    "etiv": "intracranial_volume",
}


# ---------------------------------------------------------------------------
# Layer 1 — pure aseg.stats parser (offline, deterministic, no model)
# ---------------------------------------------------------------------------


def _parse_measure_line(line: str) -> Optional[tuple[str, float]]:
    """Parse a ``# Measure key, shortname, desc, value, unit`` line.

    Returns ``(shortname_lower, value)`` or ``None`` if the line is not a
    well-formed Measure header. The value is the second-to-last comma field
    (the numeric volume); the unit is the last field.
    """
    body = line.lstrip("#").strip()
    if not body.lower().startswith("measure"):
        return None
    # drop the leading "Measure" token, then split the remaining CSV.
    rest = body[len("Measure"):].strip()
    parts = [p.strip() for p in rest.split(",")]
    if len(parts) < 4:
        return None
    shortname = parts[1].lower()
    # value is the last numeric-parseable field (unit like "mm^3" is non-numeric).
    for field in reversed(parts):
        try:
            return shortname, float(field)
        except ValueError:
            continue
    return None


def _parse_table_row(line: str) -> Optional[tuple[str, float]]:
    """Parse a data row -> ``(structname_lower, volume_mm3)``.

    aseg.stats columns are: Index SegId NVoxels Volume_mm3 StructName ...
    So the volume is column index 3 and the structure name is column index 4.
    Returns ``None`` for comment/blank/short lines.
    """
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    parts = s.split()
    if len(parts) < 5:
        return None
    try:
        vol = float(parts[3])
    except ValueError:
        return None
    return parts[4].lower(), vol


def parse_aseg_stats(path) -> dict:
    """Parse a FreeSurfer/FastSurfer ``aseg.stats`` into a normalized volume dict.

    The returned dict maps each of :data:`VOLUME_KEYS` to a float (mm^3) or
    ``None`` when that structure/measure was absent from the file, plus provenance:
    ``source`` (:data:`SOURCE_ASEG`), ``aseg_path``, and ``subject_id`` when it can
    be recovered from a ``# subjectname`` header.

    Pure and offline — no torch, no model, no network. This is the schema glue that
    lets FastSurfer volumes join the contract table.

    Raises ``FileNotFoundError`` if ``path`` does not exist (a missing file is a
    caller error, distinct from the honest ``None`` degrade of the GPU path).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"aseg.stats not found: {p}")

    measures: dict[str, float] = {}
    struct_vols: dict[str, float] = {}
    subject_id: Optional[str] = None

    with open(p, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.lstrip().startswith("#"):
                body = line.lstrip("#").strip()
                low = body.lower()
                if low.startswith("subjectname") or low.startswith("subject "):
                    toks = body.split()
                    if len(toks) >= 2:
                        subject_id = toks[1]
                    continue
                m = _parse_measure_line(line)
                if m is not None:
                    measures[m[0]] = m[1]
                continue
            row = _parse_table_row(line)
            if row is not None:
                struct_vols[row[0]] = row[1]

    def _sum(names: tuple[str, ...]) -> Optional[float]:
        present = [struct_vols[n] for n in names if n in struct_vols]
        return round(float(sum(present)), 4) if present else None

    out: dict = {k: None for k in VOLUME_KEYS}
    out["hippocampal_volume"] = _sum(_HIPPOCAMPUS_STRUCTS)
    out["ventricle_volume"] = _sum(_VENTRICLE_STRUCTS)
    for short, key in _MEASURE_MAP.items():
        if short in measures:
            out[key] = round(float(measures[short]), 4)

    out["source"] = SOURCE_ASEG
    out["aseg_path"] = str(p)
    if subject_id is not None:
        out["subject_id"] = subject_id
    return out


# ---------------------------------------------------------------------------
# Layer 2 — GPU FastSurfer path (lazy, honest degrade to None)
# ---------------------------------------------------------------------------


def _torch_gpu_available() -> tuple[bool, str]:
    """(has_gpu, reason). Lazy-imports torch so module import stays torch-free.

    Returns ``(False, reason)`` when torch is not importable or no CUDA device is
    present — the honest-degrade signal for :func:`segment_volume`.
    """
    try:
        import torch  # noqa: F401  (lazy: heavy dep must not import at module load)
    except Exception as exc:  # noqa: BLE001
        return False, f"torch not importable ({exc.__class__.__name__})"
    try:
        if not torch.cuda.is_available():
            return False, "no CUDA GPU available"
    except Exception as exc:  # noqa: BLE001
        return False, f"cuda probe failed ({exc.__class__.__name__})"
    return True, ""


def _locate_fastsurfer() -> Optional[str]:
    """Path to ``run_fastsurfer.sh`` from ``$FASTSURFER_HOME`` or ``$PATH``, else None.

    The FastSurfer install (code + Apache-2.0 weights) lives ONLY on the GPU
    runtime and is never committed; its absence is a normal, honest degrade.
    """
    home = os.environ.get("FASTSURFER_HOME")
    if home:
        cand = Path(home) / "run_fastsurfer.sh"
        if cand.exists():
            return str(cand)
    return shutil.which("run_fastsurfer.sh")


def _run_fastsurfer(runner_sh: str, nifti: Path, sid: str, out_dir: Path,
                    timeout: int) -> Optional[Path]:
    """Shell out to FastSurfer ``--seg_only`` and return the produced aseg.stats path.

    Segmentation-only (no surface reconstruction) is the fast ~1 min/GPU path and is
    all we need for the aseg volumes. Returns the aseg.stats Path on success, else
    ``None`` (non-zero exit, timeout, or missing output). Never raises to the caller.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        runner_sh,
        "--t1", str(nifti),
        "--sid", sid,
        "--sd", str(out_dir),
        "--seg_only",
        "--no_cereb", "--no_biasfield",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except Exception:  # noqa: BLE001  (timeout / OSError -> honest degrade)
        return None
    if proc.returncode != 0:
        return None
    aseg = out_dir / sid / "stats" / "aseg+DKT.stats"
    if not aseg.exists():
        aseg = out_dir / sid / "stats" / "aseg.stats"
    return aseg if aseg.exists() else None


def segment_volume(nifti, *, subject_id: Optional[str] = None,
                   output_dir: Optional[str] = None, timeout: int = 1800,
                   _runner=None) -> Optional[dict]:
    """Run FastSurfer on a T1w NIfTI -> normalized volume dict, or ``None``.

    HONEST DEGRADE: returns ``None`` (never fabricates volumes) when the input file
    is missing, torch is not importable, no CUDA GPU is present, FastSurfer is not
    installed, or the segmentation fails. On success, returns the same normalized
    dict as :func:`parse_aseg_stats`, re-stamped ``source=`` :data:`SOURCE_SEGMENT`.

    ``_runner`` is a seam for tests: a callable ``(runner_sh, nifti, sid, out_dir,
    timeout) -> aseg_path|None``. It is NEVER used to synthesize volumes — it only
    points at an aseg.stats file that :func:`parse_aseg_stats` then reads.

    Lazy by construction: torch is imported only inside :func:`_torch_gpu_available`,
    so importing this module (and running the whole test suite) needs no heavy deps.
    """
    p = Path(nifti)
    if not p.exists():
        return None

    ok, _reason = _torch_gpu_available()
    if not ok:
        return None

    runner_sh = _locate_fastsurfer()
    if runner_sh is None:
        return None

    sid = subject_id or p.stem.split(".")[0]
    out_dir = Path(output_dir) if output_dir else p.parent / "fastsurfer_out"

    run = _runner or _run_fastsurfer
    try:
        aseg_path = run(runner_sh, p, sid, out_dir, timeout)
    except Exception:  # noqa: BLE001  (a broken runner must not fabricate anything)
        return None
    if not aseg_path or not Path(aseg_path).exists():
        return None

    vols = parse_aseg_stats(aseg_path)
    vols["source"] = SOURCE_SEGMENT   # re-stamp: FastSurfer actually ran here
    vols["subject_id"] = sid
    return vols
