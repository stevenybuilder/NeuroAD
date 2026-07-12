"""
boltz — REAL open molecular targeting for prioritized AD targets (Boltz-2).

This is the shipping, REAL-predictor answer to the L6 "molecular targeting"
complex step. Unlike ``string_ppi`` (protein-protein INTERACTION EVIDENCE, the
lightweight literature/DB stand-in) and unlike AlphaFold3 (whose weights are
gated behind a non-commercial license and stay intentionally UNWIRED, see
``alphafold``), Boltz-2 is an **open, MIT-licensed, AlphaFold3-class** structure
predictor that ALSO predicts protein-ligand binding affinity. It is NOT a
stand-in: given a target protein pair (or a target + a small-molecule ligand) it
folds the complex de-novo and returns real predicted-confidence scalars
(ipTM / pTM / PAE / a complex confidence score) and, for a target+ligand, a
predicted binding-affinity value + a binary binding probability.

WHY IT IS NOT WIRED TO RUN LOCALLY BY DEFAULT: Boltz-2 needs a CUDA GPU and a
~model download; this repo's host has neither. HONESTY CONTRACT (paramount): this
module NEVER fabricates coordinates, confidence, or affinity. It resolves along
exactly two honest paths:

  1. PRECOMPUTED SNAPSHOT — if a committed JSON of REAL Boltz-2 results exists
     (produced by ``scripts/boltz_fold_colab.py`` on a Colab GPU and committed as
     confidence+affinity SCALARS only; coordinates are NEVER committed), return it
     stamped ``source="precomputed_snapshot"`` with full run provenance.
  2. DEFERRED — otherwise return a clearly-labeled, non-fabricated 'deferred'
     result (``source="deferred"``, ``status="deferred"``) whose note says
     "boltz not installed — GPU run required". No numbers are invented.

An optional local-run path (``allow_local_run=True``) LAZY-imports ``boltz`` and
best-effort invokes its CLI; on ANY failure (no ``boltz``, no GPU, bad output) it
degrades to the deferred result — it never raises and never fabricates. The heavy
inference is meant to live in ``scripts/boltz_fold_colab.py`` (Colab GPU), exactly
like the Neuro-JEPA embedding job.

Every returned record is provenance-stamped and honesty-labeled (``BOLTZ_LABEL``).
A ``deferred`` result is NEVER dressed up as a real prediction.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Canonical AD target -> UniProt accession, mirrored from the AlphaFold adapter so
# the Colab folding job can fetch each target's sequence keylessly from UniProt.
# (Imported lazily-tolerant: fall back to a local copy if the import is unavailable
# during isolated testing.)
try:  # pragma: no cover - trivial import shim
    from .alphafold import AD_PROTEIN_MAP  # type: ignore
except Exception:  # pragma: no cover
    AD_PROTEIN_MAP = {
        "APP": "P05067", "MAPT": "P10636", "TAU": "P10636", "APOE": "P02649",
        "PSEN1": "P49768", "PSEN2": "P49810", "BACE1": "P56817",
        "TREM2": "Q9NZC2", "HRAS": "P01112", "MAPK1": "P28482",
        "ESR1": "P03372", "CLU": "P10909", "BIN1": "O00499",
    }

#: The AD targets the engine ranks (same canon as string_ppi.AD_TARGETS).
AD_TARGETS: tuple[str, ...] = (
    "APP", "MAPT", "APOE", "PSEN1", "PSEN2", "BACE1",
    "TREM2", "HRAS", "MAPK1", "ESR1", "CLU", "BIN1",
)

_SNAPSHOT_PATH = Path(__file__).with_name("data") / "boltz_snapshot.json"

#: One-line honesty banner attached to every returned record.
BOLTZ_LABEL = (
    "Boltz-2 open (MIT) AlphaFold3-class complex + binding-affinity prediction "
    "— a REAL predictor, not a stand-in; requires a GPU run"
)

#: The honest note stamped on a deferred (not-run) result.
DEFERRED_NOTE = (
    "boltz not installed — GPU run required. Run scripts/boltz_fold_colab.py on a "
    "Colab GPU and commit its confidence+affinity scalars to data/boltz_snapshot.json"
)


def _load_snapshot(path: Optional[Path] = None) -> dict:
    """Load the committed PRECOMPUTED Boltz-2 results snapshot (may be empty)."""
    p = path or _SNAPSHOT_PATH
    try:
        with open(p, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        if not isinstance(raw, dict):
            return {}
        return raw
    except Exception:
        return {}


def _complex_key(a: str, b: str) -> str:
    """Order-independent key for a protein-protein complex."""
    return "|".join(sorted((a.upper(), b.upper())))


def _ligand_key(gene: str, ligand_id: str) -> str:
    """Key for a target + small-molecule ligand pair."""
    return f"{gene.upper()}::{ligand_id.strip()}"


# ---------------------------------------------------------------------------
# Structured return
# ---------------------------------------------------------------------------


@dataclass
class BoltzTargeting:
    """One Boltz-2 targeting result (a protein complex OR a target+ligand pair).

    Confidence scalars are on Boltz-2's native scales: ``iptm``/``ptm`` in [0, 1]
    (interface / global predicted TM-score), ``pae`` a mean predicted aligned
    error in Ångström (lower = better), ``confidence_score`` Boltz's aggregate
    confidence in [0, 1]. For a ``target_ligand`` result ``binding_affinity`` is
    Boltz-2's predicted affinity value (a pIC50-like log-scaled scalar) and
    ``binding_probability`` its binary binder probability in [0, 1].

    ``source`` is the provenance stamp: "precomputed_snapshot" (REAL GPU-run
    result read from the committed snapshot), "boltz_live" (a real local GPU run),
    or "deferred" (no result — GPU run required; NO numbers fabricated).
    ``status`` is "predicted" or "deferred". A deferred record carries None for
    every scalar — it is NEVER dressed up as a prediction.
    """
    gene_a: str
    gene_b: str = ""                       # "" for a target+ligand result
    kind: str = "complex"                  # "complex" | "target_ligand"
    ligand_id: str = ""                    # ligand name/id (target_ligand only)
    ligand_smiles: str = ""                # ligand SMILES (target_ligand only)
    iptm: Optional[float] = None
    ptm: Optional[float] = None
    pae: Optional[float] = None
    confidence_score: Optional[float] = None
    binding_affinity: Optional[float] = None       # Boltz-2 predicted affinity
    binding_probability: Optional[float] = None    # binary binder probability
    source: str = "deferred"               # "precomputed_snapshot"|"boltz_live"|"deferred"
    status: str = "deferred"               # "predicted" | "deferred"
    note: str = BOLTZ_LABEL
    provenance: dict = field(default_factory=dict)  # model/license/run env/captured
    error: str = ""                        # non-fatal note (why deferred, etc.)

    def to_dict(self) -> dict:
        return {
            "gene_a": self.gene_a,
            "gene_b": self.gene_b,
            "kind": self.kind,
            "ligand_id": self.ligand_id,
            "ligand_smiles": self.ligand_smiles,
            "iptm": self.iptm,
            "ptm": self.ptm,
            "pae": self.pae,
            "confidence_score": self.confidence_score,
            "binding_affinity": self.binding_affinity,
            "binding_probability": self.binding_probability,
            "source": self.source,
            "status": self.status,
            "note": self.note,
            "provenance": dict(self.provenance),
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------


class BoltzClient:
    """Offline-first adapter over Boltz-2, snapshot-or-deferred by default.

    Default (``prefer_offline=True``): NEVER touches a GPU or the network — it
    reads the committed PRECOMPUTED snapshot and, when a complex/ligand has no
    committed result, returns an honest ``deferred`` object. Set
    ``allow_local_run=True`` (and ``prefer_offline=False``) to best-effort
    LAZY-import ``boltz`` and run it locally; ANY failure degrades to deferred.

    Pass ``snapshot`` (a dict) or ``snapshot_path`` to inject precomputed results
    (used by tests and by the referee once a real GPU run has been committed).
    """

    def __init__(self, *, prefer_offline: bool = True,
                 allow_local_run: bool = False,
                 snapshot: Optional[dict] = None,
                 snapshot_path: Optional[str] = None,
                 work_dir: Optional[str] = None) -> None:
        self.prefer_offline = prefer_offline
        self.allow_local_run = allow_local_run
        self.work_dir = Path(work_dir) if work_dir else None
        if snapshot is not None:
            self._snapshot = snapshot
        else:
            self._snapshot = _load_snapshot(
                Path(snapshot_path) if snapshot_path else None)
        self._prov = self._snapshot.get("_provenance") if isinstance(
            self._snapshot, dict) else None

    # -- resolution --------------------------------------------------------

    def resolve_symbol(self, gene: str) -> Optional[str]:
        """Normalize a gene symbol (upper-cased, alnum-ish). None if malformed."""
        if not gene:
            return None
        g = gene.strip().upper()
        if not g:
            return None
        if not all(c.isalnum() or c in "-_." for c in g):
            return None
        return g

    # -- public API: protein-protein complex -------------------------------

    def predict_complex(self, gene_a: str, gene_b: str) -> BoltzTargeting:
        """Boltz-2 complex confidence (ipTM/pTM/PAE/score) for a target pair.

        Snapshot-first: returns a committed REAL result if present, else an honest
        deferred object. Never raises, never fabricates."""
        a = self.resolve_symbol(gene_a)
        b = self.resolve_symbol(gene_b)
        if a is None or b is None:
            return self._deferred(
                gene_a or "", gene_b or "", kind="complex",
                error=f"could not resolve pair ('{gene_a}', '{gene_b}')")
        if a == b:
            return self._deferred(a, b, kind="complex",
                                  error="self-pair is not a meaningful complex")

        snap = self._snapshot_complex(a, b)
        if snap is not None:
            return snap
        if not self.prefer_offline and self.allow_local_run:
            live = self._run_local_complex(a, b)
            if live is not None:
                return live
        return self._deferred(a, b, kind="complex")

    # -- public API: target + ligand affinity ------------------------------

    def predict_affinity(self, gene: str, ligand_smiles: str = "", *,
                         ligand_id: str = "") -> BoltzTargeting:
        """Boltz-2 predicted binding affinity for a target + small-molecule ligand.

        Snapshot-first (keyed by target::ligand_id), else an honest deferred
        object. Never raises, never fabricates an affinity."""
        g = self.resolve_symbol(gene)
        if g is None:
            return self._deferred(
                gene or "", "", kind="target_ligand", ligand_id=ligand_id,
                ligand_smiles=ligand_smiles,
                error=f"could not resolve target '{gene}'")
        lid = ligand_id.strip() or (ligand_smiles.strip() and "ligand")
        snap = self._snapshot_affinity(g, ligand_id or ligand_smiles)
        if snap is not None:
            return snap
        if not self.prefer_offline and self.allow_local_run:
            live = self._run_local_affinity(g, ligand_smiles, ligand_id)
            if live is not None:
                return live
        return self._deferred(g, "", kind="target_ligand", ligand_id=lid or "",
                              ligand_smiles=ligand_smiles)

    # -- snapshot lookups --------------------------------------------------

    def _snapshot_complex(self, a: str, b: str) -> Optional[BoltzTargeting]:
        rec = (self._snapshot.get("complexes") or {}).get(_complex_key(a, b))
        if not isinstance(rec, dict):
            return None
        return self._from_record(a, b, rec, kind="complex")

    def _snapshot_affinity(self, gene: str, ligand: str) -> Optional[BoltzTargeting]:
        rec = (self._snapshot.get("affinities") or {}).get(
            _ligand_key(gene, ligand or ""))
        if not isinstance(rec, dict):
            return None
        t = self._from_record(gene, "", rec, kind="target_ligand")
        t.ligand_id = rec.get("ligand_id", ligand) or ""
        t.ligand_smiles = rec.get("ligand_smiles", "") or ""
        return t

    def _from_record(self, a: str, b: str, rec: dict, *, kind: str) -> BoltzTargeting:
        """Build a provenance-stamped precomputed result from a snapshot record."""
        prov = dict(self._prov or {})
        prov.update(rec.get("provenance") or {})
        return BoltzTargeting(
            gene_a=a, gene_b=b, kind=kind,
            iptm=_num(rec.get("iptm")),
            ptm=_num(rec.get("ptm")),
            pae=_num(rec.get("pae")),
            confidence_score=_num(rec.get("confidence_score")),
            binding_affinity=_num(rec.get("binding_affinity")),
            binding_probability=_num(rec.get("binding_probability")),
            source="precomputed_snapshot",
            status="predicted",
            note=BOLTZ_LABEL,
            provenance=prov,
        )

    # -- honest deferral ---------------------------------------------------

    def _deferred(self, a: str, b: str, *, kind: str, ligand_id: str = "",
                  ligand_smiles: str = "", error: str = "") -> BoltzTargeting:
        """A clearly-labeled, non-fabricated 'not run' result. All scalars None."""
        note = DEFERRED_NOTE if not error else f"{error}; {DEFERRED_NOTE}"
        return BoltzTargeting(
            gene_a=a, gene_b=b, kind=kind, ligand_id=ligand_id,
            ligand_smiles=ligand_smiles,
            source="deferred", status="deferred", note=note, error=error,
            provenance={"model": "Boltz-2", "license": "MIT",
                        "run": "scripts/boltz_fold_colab.py"},
        )

    # -- optional local GPU run (LAZY, fully wrapped) ----------------------

    def _run_local_complex(self, a: str, b: str) -> Optional[BoltzTargeting]:
        """Best-effort local Boltz-2 complex run; None on ANY failure (never fake)."""
        seqs = _sequences_for([a, b])
        if seqs is None:
            return None
        out = _run_boltz_cli(entities=[("protein", s) for s in seqs],
                             work_dir=self.work_dir)
        if out is None:
            return None
        conf, _aff = out
        if conf is None:
            return None
        return BoltzTargeting(
            gene_a=a, gene_b=b, kind="complex",
            iptm=_num(conf.get("iptm")), ptm=_num(conf.get("ptm")),
            pae=_num(conf.get("pae")),
            confidence_score=_num(conf.get("confidence_score")),
            source="boltz_live", status="predicted", note=BOLTZ_LABEL,
            provenance={"model": "Boltz-2", "license": "MIT", "run": "local_gpu"},
        )

    def _run_local_affinity(self, gene: str, smiles: str,
                            ligand_id: str) -> Optional[BoltzTargeting]:
        seqs = _sequences_for([gene])
        if seqs is None or not smiles:
            return None
        out = _run_boltz_cli(
            entities=[("protein", seqs[0]), ("ligand", smiles)],
            work_dir=self.work_dir, affinity=True)
        if out is None:
            return None
        conf, aff = out
        return BoltzTargeting(
            gene_a=gene, kind="target_ligand", ligand_id=ligand_id,
            ligand_smiles=smiles,
            iptm=_num((conf or {}).get("iptm")),
            ptm=_num((conf or {}).get("ptm")),
            confidence_score=_num((conf or {}).get("confidence_score")),
            binding_affinity=_num((aff or {}).get("affinity_pred_value")),
            binding_probability=_num((aff or {}).get("affinity_probability_binary")),
            source="boltz_live", status="predicted", note=BOLTZ_LABEL,
            provenance={"model": "Boltz-2", "license": "MIT", "run": "local_gpu"},
        )


# ---------------------------------------------------------------------------
# Helpers (pure / lazy)
# ---------------------------------------------------------------------------


def _num(value: object) -> Optional[float]:
    """Coerce to float or None — never a fabricated 0.0 for a missing value."""
    if value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _sequences_for(genes: list[str]) -> Optional[list[str]]:
    """Best-effort keyless UniProt FASTA fetch for the given AD gene symbols.

    Returns None on ANY failure (offline, unknown symbol) so the caller degrades
    to deferred. Never fabricates a sequence."""
    accs = [AD_PROTEIN_MAP.get(g.upper()) for g in genes]
    if any(a is None for a in accs):
        return None
    seqs: list[str] = []
    try:
        import requests  # lazy: only touched on the explicit local-run path
        for acc in accs:
            resp = requests.get(
                f"https://rest.uniprot.org/uniprotkb/{acc}.fasta", timeout=15)
            if resp.status_code != 200:
                return None
            body = "".join(ln for ln in resp.text.splitlines()
                           if not ln.startswith(">"))
            if not body:
                return None
            seqs.append(body)
    except Exception:
        return None
    return seqs


def _run_boltz_cli(entities, *, work_dir: Optional[Path] = None,
                   affinity: bool = False):
    """LAZY-import boltz and best-effort run its CLI; return (conf, aff) or None.

    Fully wrapped: a missing ``boltz`` package, a missing GPU, a subprocess error,
    or unparsable output all yield None (caller -> honest deferred). This never
    fabricates coordinates, confidence, or affinity. Heavy inference is intended
    for scripts/boltz_fold_colab.py (Colab GPU) — this local path exists only so
    that a host that DOES have boltz+GPU works without code changes."""
    import importlib.util
    if importlib.util.find_spec("boltz") is None:
        return None
    import subprocess
    import tempfile
    try:
        base = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="boltz_"))
        base.mkdir(parents=True, exist_ok=True)
        yaml_path = base / "input.yaml"
        yaml_path.write_text(_build_boltz_yaml(entities, affinity=affinity),
                             encoding="utf-8")
        out_dir = base / "out"
        cmd = ["boltz", "predict", str(yaml_path), "--out_dir", str(out_dir)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if proc.returncode != 0:
            return None
        return _parse_boltz_outputs(out_dir)
    except Exception:
        return None


def _build_boltz_yaml(entities, *, affinity: bool = False) -> str:
    """Render a Boltz-2 input YAML (stdlib only — no yaml dependency).

    ``entities`` is a list of ("protein"|"ligand", sequence-or-smiles). Chains are
    lettered A, B, C…; when ``affinity`` is set the last ligand chain is named as
    the affinity binder. Pure/stringy so it is unit-testable offline."""
    import string
    lines = ["version: 1", "sequences:"]
    letters = iter(string.ascii_uppercase)
    ligand_chain = ""
    for kind, seq in entities:
        cid = next(letters)
        if kind == "protein":
            lines += [f"  - protein:", f"      id: {cid}", f"      sequence: {seq}"]
        else:  # ligand
            ligand_chain = cid
            lines += [f"  - ligand:", f"      id: {cid}", f"      smiles: '{seq}'"]
    if affinity and ligand_chain:
        lines += ["properties:", "  - affinity:", f"      binder: {ligand_chain}"]
    return "\n".join(lines) + "\n"


def _parse_boltz_outputs(out_dir):
    """Parse a Boltz-2 output tree into (confidence_dict, affinity_dict|None).

    Scans for the ``confidence_*.json`` (ipTM/pTM/confidence_score) and, if
    present, ``affinity_*.json`` (affinity_pred_value / affinity_probability_binary)
    that Boltz writes per prediction. Returns (None, None) if no confidence file is
    found. Pure filesystem+JSON — unit-testable with a fabricated output tree, and
    it only ever surfaces values that the files actually contain."""
    out = Path(out_dir)
    conf_files = sorted(out.rglob("confidence_*.json"))
    aff_files = sorted(out.rglob("affinity_*.json"))
    conf = None
    if conf_files:
        try:
            conf = json.loads(conf_files[0].read_text(encoding="utf-8"))
        except Exception:
            conf = None
    aff = None
    if aff_files:
        try:
            aff = json.loads(aff_files[0].read_text(encoding="utf-8"))
        except Exception:
            aff = None
    if conf is None and aff is None:
        return None
    return conf, aff


# ---------------------------------------------------------------------------
# Module-level convenience (thin wrappers for harness callers)
# ---------------------------------------------------------------------------


def boltz_targeting(gene_a: str, gene_b: str, *,
                    prefer_offline: bool = True) -> dict:
    """One protein pair's Boltz-2 complex targeting result (snapshot-or-deferred).

    Offline-safe, provenance-stamped, honesty-labeled. Returns
    ``BoltzTargeting.to_dict``. A deferred result carries None scalars — a REAL
    Boltz-2 prediction requires a GPU run (scripts/boltz_fold_colab.py)."""
    return BoltzClient(prefer_offline=prefer_offline).predict_complex(
        gene_a, gene_b).to_dict()


def ligand_affinity(gene: str, ligand_smiles: str = "", *, ligand_id: str = "",
                    prefer_offline: bool = True) -> dict:
    """One target+ligand Boltz-2 affinity result (snapshot-or-deferred)."""
    return BoltzClient(prefer_offline=prefer_offline).predict_affinity(
        gene, ligand_smiles, ligand_id=ligand_id).to_dict()


def has_precomputed_results(snapshot_path: Optional[str] = None) -> bool:
    """True iff the committed snapshot holds at least one REAL Boltz-2 result."""
    snap = _load_snapshot(Path(snapshot_path) if snapshot_path else None)
    return bool((snap.get("complexes") or {})) or bool((snap.get("affinities") or {}))


__all__ = [
    "AD_TARGETS",
    "AD_PROTEIN_MAP",
    "BOLTZ_LABEL",
    "DEFERRED_NOTE",
    "BoltzClient",
    "BoltzTargeting",
    "boltz_targeting",
    "ligand_affinity",
    "has_precomputed_results",
]
