"""
targeting — L6 evidence-FUSION druggability ranking over committed predictions.

This module answers the L6 "molecular targeting" question — *which AD targets are
most druggable, and why* — by FUSING evidence that other adapters have ALREADY
committed, into one transparent, per-component, provenance-stamped score. It does
NO folding, NO GPU work, NO network I/O, and trains NO model. It is a
deterministic weighted-mean RANKING over predictions produced elsewhere:

  * AlphaFold structure confidence  (``integrations.alphafold``, mean pLDDT 0-100),
  * Boltz-2 complex confidence      (``integrations.boltz`` committed snapshot),
  * Boltz-2 ligand binding          (``integrations.boltz`` committed affinities),
  * PI4AD target priority           (``integrations.pi4ad`` 0-10 priority).

HONESTY CONTRACT (paramount, mirrors the sibling adapters):
  * Every component carries ``source=`` and ``model=`` provenance stamps.
  * A component whose upstream evidence is missing is ABSENT (``present=False``,
    ``value_raw=None``, ``value_norm=None``) — never a fabricated ``0.0``. Its
    weight is DROPPED and the remaining weights are renormalized to sum to 1, so a
    1-component score is not silently diluted by the absent ones.
  * A target with ZERO present components gets ``composite_score=None`` + an honest
    "no evidence" note and SINKS to the bottom of the ranking (never a fake 0.0
    that would out-rank a genuinely-but-weakly evidenced target).
  * Absent values are ``None`` (never ``NaN``) so nothing leaks as a number.

This module MUST NOT overclaim: it is not new folding, not a trained druggability
classifier, not a validated affinity model. No p-values, CIs, or held-out AUCs are
produced or implied (see ``TARGETING_LABEL``).

Ligand caveat (documented, not hidden): the two committed Boltz-2 affinities target
ABL1 and RXRA, which are NOT in ``boltz.AD_TARGETS`` / ``AD_PROTEIN_MAP``. Under the
current snapshot the ligand-binding fusion component therefore attaches to NO AD
target — the repurposing-ligand ranking is reported as its own independent table
rather than force-joined onto an AD target. Boltz-2's ``binding_affinity`` is a
log-scaled pIC50-like scalar whose sign/scale can DISAGREE with the well-defined
[0, 1] ``binding_probability`` (the committed sample does disagree), so the ligand
summary LEADS on ``binding_probability`` and surfaces the raw affinity with a
caveat rather than treating it as monotone.

Offline + deterministic: imports and runs with numpy/pandas only. No torch, no GPU,
no network on any path (the AlphaFold and Boltz clients are constructed
``prefer_offline=True`` internally to honor the contract).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from . import boltz as _boltz
from .alphafold import AlphaFoldClient
from .boltz import AD_TARGETS, BoltzClient, has_precomputed_results
from .pi4ad import gene_priority

# ---------------------------------------------------------------------------
# Documented, overridable component weights (renormalized over present subset).
# ---------------------------------------------------------------------------
#: Default fusion weights. Structure + complex confidence lead (they are the two
#: structural-biology signals); ligand binding + PI4AD priority contribute the
#: remainder. Renormalized to sum to 1 over ONLY the components present for a
#: given target — an absent component's weight is dropped, never spent on a zero.
DEFAULT_WEIGHTS: dict[str, float] = {
    "struct_plddt": 0.30,
    "complex_confidence": 0.30,
    "ligand_binding": 0.20,
    "pi4ad_priority": 0.20,
}

#: One-line honesty banner attached to every fused row's provenance.
TARGETING_LABEL = (
    "deterministic evidence-fusion RANKING over committed AlphaFold/Boltz-2/PI4AD "
    "predictions — NOT new folding, NOT a trained druggability model"
)

# Provenance stamps (source, model) per component — kept explicit so every value
# can be traced to the adapter that produced it.
_STAMP = {
    "struct_plddt": ("offline_snapshot", "AlphaFold-DB"),
    "complex_confidence": ("precomputed_snapshot", "Boltz-2"),
    "ligand_binding": ("precomputed_snapshot", "Boltz-2"),
    "pi4ad_priority": ("offline_snapshot", "PI4AD"),
}


def _num(value: object) -> Optional[float]:
    """Coerce to float or None — never a fabricated 0.0 for a missing value."""
    if value is None:
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _clamp01(x: Optional[float]) -> Optional[float]:
    """Clamp a value into [0, 1]; passes None through unchanged."""
    if x is None:
        return None
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


# ---------------------------------------------------------------------------
# Structured returns
# ---------------------------------------------------------------------------


@dataclass
class TargetingComponent:
    """One fused evidence component for a target.

    ``value_raw`` is the component's native-scale value (mean pLDDT 0-100, a Boltz
    confidence/probability in [0, 1], a PI4AD priority 0-10); ``value_norm`` is that
    value mapped into [0, 1] for fusion (or None when the component is absent).
    ``present`` is False iff the upstream evidence is missing — in which case
    ``value_raw``/``value_norm`` are None and the component contributes NOTHING (its
    weight is dropped from renormalization). ``source``/``model`` provenance-stamp
    the producing adapter.
    """
    name: str
    value_raw: Optional[float]
    value_norm: Optional[float]
    weight: float
    present: bool
    source: str
    model: str
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value_raw": self.value_raw,
            "value_norm": self.value_norm,
            "weight": self.weight,
            "present": self.present,
            "source": self.source,
            "model": self.model,
            "note": self.note,
        }


@dataclass
class TargetDruggability:
    """One target's explainable, fused druggability row.

    ``composite_score`` is a weighted mean over ONLY the present components, weights
    renormalized to sum to 1 over that subset (so it stays in [0, 1]); it is None
    when zero components are present. ``effective_weights`` are those renormalized
    weights (per present component name). ``components`` holds all four component
    objects (present or absent) for full transparency.
    """
    gene: str
    composite_score: Optional[float]
    components: list[TargetingComponent]
    n_components: int
    components_present: list[str]
    effective_weights: dict[str, float]
    note: str
    source: str = "targeting_fusion"
    provenance: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "gene": self.gene,
            "composite_score": self.composite_score,
            "components": [c.to_dict() for c in self.components],
            "n_components": self.n_components,
            "components_present": list(self.components_present),
            "effective_weights": dict(self.effective_weights),
            "note": self.note,
            "source": self.source,
            "provenance": dict(self.provenance),
        }


@dataclass
class LigandDruggability:
    """One repurposing compound's committed Boltz-2 affinity, ranked by binding.

    Ranked by ``binding_probability`` (Boltz-2's well-defined [0, 1] binder
    probability). ``binding_affinity`` is Boltz-2's raw log-scaled pIC50-like
    scalar (lower may mean stronger binding) surfaced alongside — it can DISAGREE
    with the probability, so it is NOT treated as a monotone score.
    ``complex_confidence`` / ``iptm`` are the pose confidence for the docked
    target+ligand.
    """
    gene: str
    ligand_id: str
    ligand_smiles: str
    binding_probability: Optional[float]
    binding_affinity: Optional[float]
    complex_confidence: Optional[float]
    iptm: Optional[float]
    rank: Optional[int]
    source: str
    model: str = "Boltz-2"
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "gene": self.gene,
            "ligand_id": self.ligand_id,
            "ligand_smiles": self.ligand_smiles,
            "binding_probability": self.binding_probability,
            "binding_affinity": self.binding_affinity,
            "complex_confidence": self.complex_confidence,
            "iptm": self.iptm,
            "rank": self.rank,
            "source": self.source,
            "model": self.model,
            "note": self.note,
        }


# ---------------------------------------------------------------------------
# The engine
# ---------------------------------------------------------------------------

_LIGAND_CAVEAT = (
    "Boltz-2 binding_affinity is a log-scaled pIC50-like scalar (lower may mean "
    "stronger binding) and can DISAGREE with binding_probability; this ranking "
    "leads on binding_probability, the well-defined [0,1] signal."
)


class TargetingEngine:
    """Fuse committed AlphaFold/Boltz-2/PI4AD evidence into a druggability ranking.

    Offline + deterministic by construction: the AlphaFold and Boltz clients are
    built ``prefer_offline=True`` so NO network/GPU is touched on any path. Clients
    and the Boltz snapshot are injectable for tests. Never raises, never fabricates
    a value — missing evidence yields an explicitly-absent component.
    """

    def __init__(self, *, prefer_offline: bool = True,
                 weights: Optional[dict] = None,
                 boltz_snapshot: Optional[dict] = None,
                 boltz_snapshot_path: Optional[str] = None,
                 alphafold_client=None,
                 boltz_client=None,
                 pi4ad_prefer_offline: bool = True) -> None:
        # prefer_offline is accepted for API symmetry; the clients are FORCED
        # offline regardless to honor the no-network contract.
        self.prefer_offline = prefer_offline
        self.weights = dict(weights) if weights else dict(DEFAULT_WEIGHTS)
        self.pi4ad_prefer_offline = pi4ad_prefer_offline

        if boltz_client is not None:
            self.boltz = boltz_client
        else:
            self.boltz = BoltzClient(
                prefer_offline=True,
                snapshot=boltz_snapshot,
                snapshot_path=boltz_snapshot_path,
            )
        # The snapshot dict backing the Boltz client — used to enumerate the
        # committed complexes/affinities a target participates in.
        snap = getattr(self.boltz, "_snapshot", None)
        self._snapshot = snap if isinstance(snap, dict) else {}

        self.alphafold = (alphafold_client if alphafold_client is not None
                          else AlphaFoldClient(prefer_offline=True))

    # -- per-component evidence -------------------------------------------

    def _component(self, name: str, value_raw: Optional[float],
                   value_norm: Optional[float], note: str = "") -> TargetingComponent:
        source, model = _STAMP[name]
        present = value_norm is not None
        return TargetingComponent(
            name=name,
            value_raw=value_raw if present else None,
            value_norm=_clamp01(value_norm) if present else None,
            weight=float(self.weights.get(name, 0.0)),
            present=present,
            source=source,
            model=model,
            note=note,
        )

    def _struct_component(self, gene: str) -> TargetingComponent:
        """AlphaFold structure confidence: mean pLDDT (0-100) -> /100."""
        try:
            struct = self.alphafold.fetch_structure(gene)
            plddt = _num(getattr(struct, "mean_plddt", None))
        except Exception:
            plddt = None
        if plddt is None:
            return self._component("struct_plddt", None, None,
                                   note="no offline AlphaFold pLDDT for this target")
        return self._component("struct_plddt", plddt, plddt / 100.0,
                               note="mean pLDDT/100")

    def _complex_component(self, gene: str) -> TargetingComponent:
        """Boltz-2 complex confidence: max confidence_score over committed complexes
        this target participates in (deterministic max aggregation)."""
        g = gene.strip().upper()
        best_val: Optional[float] = None
        best_key = ""
        complexes = self._snapshot.get("complexes") or {}
        if isinstance(complexes, dict):
            for key, rec in complexes.items():
                if not isinstance(rec, dict):
                    continue
                members = {str(rec.get("gene_a", "")).upper(),
                           str(rec.get("gene_b", "")).upper()}
                # Also honor the order-independent "A|B" key form.
                members |= {p.upper() for p in str(key).split("|")}
                if g not in members:
                    continue
                cs = _num(rec.get("confidence_score"))
                if cs is None:
                    continue
                if best_val is None or cs > best_val:
                    best_val, best_key = cs, str(key)
        if best_val is None:
            return self._component(
                "complex_confidence", None, None,
                note="no committed Boltz-2 complex for this target")
        return self._component(
            "complex_confidence", best_val, best_val,
            note=f"max confidence_score over committed complexes ({best_key})")

    def _ligand_component(self, gene: str) -> TargetingComponent:
        """Boltz-2 ligand binding: max binding_probability over committed affinities
        keyed to THIS gene. Under the current snapshot (ABL1/RXRA) this attaches to
        no AD target — reported as absent, never force-joined."""
        g = gene.strip().upper()
        best_val: Optional[float] = None
        best_lig = ""
        affinities = self._snapshot.get("affinities") or {}
        if isinstance(affinities, dict):
            for key, rec in affinities.items():
                if not isinstance(rec, dict):
                    continue
                target = str(rec.get("gene_a", "")).upper()
                if not target:
                    target = str(key).split("::", 1)[0].upper()
                if target != g:
                    continue
                bp = _num(rec.get("binding_probability"))
                if bp is None:
                    continue
                if best_val is None or bp > best_val:
                    best_val = bp
                    best_lig = str(rec.get("ligand_id", "")) or str(key)
        if best_val is None:
            return self._component(
                "ligand_binding", None, None,
                note="no committed Boltz-2 ligand affinity keyed to this target")
        return self._component(
            "ligand_binding", best_val, best_val,
            note=f"binding_probability for {best_lig}")

    def _pi4ad_component(self, gene: str) -> TargetingComponent:
        """PI4AD priority (0-10) -> /10."""
        try:
            gp = gene_priority(gene, prefer_offline=self.pi4ad_prefer_offline)
        except Exception:
            gp = None
        score = _num(getattr(gp, "priority_score", None)) if gp is not None else None
        if score is None:
            return self._component("pi4ad_priority", None, None,
                                   note="not in PI4AD priority snapshot")
        rank = getattr(gp, "rank", None)
        return self._component("pi4ad_priority", score, score / 10.0,
                               note=f"PI4AD priority/10 (rank {rank})")

    # -- fusion -----------------------------------------------------------

    def score_target(self, gene: str) -> TargetDruggability:
        """Fuse all available evidence for one gene. Never raises, never fabricates."""
        g = (gene or "").strip()
        components = [
            self._struct_component(g),
            self._complex_component(g),
            self._ligand_component(g),
            self._pi4ad_component(g),
        ]
        present = [c for c in components if c.present]
        present_names = [c.name for c in present]

        total_w = sum(c.weight for c in present)
        effective: dict[str, float] = {}
        composite: Optional[float] = None
        if present and total_w > 0:
            effective = {c.name: c.weight / total_w for c in present}
            composite = sum(effective[c.name] * c.value_norm for c in present)
            composite = float(_clamp01(composite))
        elif present:
            # Present components but all weights zero: fall back to a plain mean so
            # nothing is silently dropped; weights split uniformly.
            effective = {c.name: 1.0 / len(present) for c in present}
            composite = float(_clamp01(
                sum(c.value_norm for c in present) / len(present)))

        note = self._note(g, present_names)
        provenance = {
            "label": TARGETING_LABEL,
            "weights_requested": dict(self.weights),
            "boltz_has_precomputed": has_precomputed_results(),
            "n_components": len(present),
        }
        return TargetDruggability(
            gene=g.upper() if g else g,
            composite_score=composite,
            components=components,
            n_components=len(present),
            components_present=present_names,
            effective_weights=effective,
            note=note,
            source="targeting_fusion",
            provenance=provenance,
        )

    def _note(self, gene: str, present_names: list[str]) -> str:
        if not present_names:
            return "no committed structural/priority evidence for this target"
        thin = []
        if "complex_confidence" not in present_names and "ligand_binding" not in present_names:
            thin.append("no committed Boltz-2 structural/binding evidence")
        if "struct_plddt" not in present_names:
            thin.append("no AlphaFold pLDDT")
        if len(present_names) == 1:
            thin.append(f"scored on a single component ({present_names[0]})")
        base = f"fused {len(present_names)} component(s): {', '.join(present_names)}"
        if thin:
            return base + " — thin evidence: " + "; ".join(thin)
        return base

    def rank_targets(self, genes: Optional[list[str]] = None, *,
                     top_n: Optional[int] = None) -> list[TargetDruggability]:
        """Rank targets by composite score (desc); None composites sink to the bottom
        in a stable, gene-alphabetical order. Defaults to ``boltz.AD_TARGETS``."""
        universe = list(genes) if genes is not None else list(AD_TARGETS)
        rows = [self.score_target(g) for g in universe]
        rows.sort(key=lambda r: (
            r.composite_score is None,
            -(r.composite_score if r.composite_score is not None else 0.0),
            r.gene,
        ))
        if top_n is not None and top_n >= 0:
            rows = rows[:top_n]
        return rows

    def ligand_druggability(self) -> list[LigandDruggability]:
        """Rank the committed repurposing compounds by Boltz-2 binding_probability.

        Reads the committed affinities from the snapshot; returns [] when none are
        committed. Ranked by binding_probability desc (affinity tie-break); rank
        stamped. Honestly notes that its targets sit outside AD_TARGETS."""
        affinities = self._snapshot.get("affinities") or {}
        if not isinstance(affinities, dict) or not affinities:
            return []
        rows: list[LigandDruggability] = []
        ad = {t.upper() for t in AD_TARGETS}
        for key, rec in affinities.items():
            if not isinstance(rec, dict):
                continue
            gene = str(rec.get("gene_a", "")) or str(key).split("::", 1)[0]
            outside = gene.upper() not in ad
            note = _LIGAND_CAVEAT
            if outside:
                note += (f" NOTE: {gene} is OUTSIDE boltz.AD_TARGETS, so this "
                         "compound contributes no per-AD-target ligand component.")
            rows.append(LigandDruggability(
                gene=gene,
                ligand_id=str(rec.get("ligand_id", "")) or str(key),
                ligand_smiles=str(rec.get("ligand_smiles", "")),
                binding_probability=_num(rec.get("binding_probability")),
                binding_affinity=_num(rec.get("binding_affinity")),
                complex_confidence=_num(rec.get("confidence_score")),
                iptm=_num(rec.get("iptm")),
                rank=None,
                source="precomputed_snapshot",
                model="Boltz-2",
                note=note,
            ))

        def _sort_key(r: LigandDruggability):
            bp = r.binding_probability
            aff = r.binding_affinity
            return (
                bp is None,
                -(bp if bp is not None else 0.0),
                # more-negative affinity first as a stable tie-break; None last
                aff if aff is not None else float("inf"),
                r.ligand_id,
            )

        rows.sort(key=_sort_key)
        for i, r in enumerate(rows, start=1):
            r.rank = i
        return rows

    # -- flat table -------------------------------------------------------

    def to_frame(self, genes: Optional[list[str]] = None):
        """Flat explainable table: one row per target, one column per component's
        normalized value + present flag, plus composite_score / n_components / note."""
        import pandas as pd

        rows = self.rank_targets(genes)
        records = []
        comp_names = list(DEFAULT_WEIGHTS.keys())
        for r in rows:
            by_name = {c.name: c for c in r.components}
            rec: dict = {"gene": r.gene}
            for name in comp_names:
                c = by_name.get(name)
                rec[f"{name}_norm"] = (c.value_norm if c is not None else None)
                rec[f"{name}_present"] = bool(c.present) if c is not None else False
            rec["composite_score"] = r.composite_score
            rec["n_components"] = r.n_components
            rec["note"] = r.note
            records.append(rec)
        # object dtype keeps None as None (not NaN) for the nullable numeric cells.
        return pd.DataFrame(records, dtype=object)


# ---------------------------------------------------------------------------
# Module-level conveniences
# ---------------------------------------------------------------------------


def druggability_ranking(genes: Optional[list[str]] = None, *,
                         prefer_offline: bool = True,
                         top_n: Optional[int] = None) -> list[dict]:
    """Ranked fused druggability rows (as dicts) over the AD target universe."""
    eng = TargetingEngine(prefer_offline=prefer_offline)
    return [r.to_dict() for r in eng.rank_targets(genes, top_n=top_n)]


def target_druggability(gene: str, *, prefer_offline: bool = True) -> dict:
    """One target's fused druggability row (as a dict)."""
    return TargetingEngine(prefer_offline=prefer_offline).score_target(gene).to_dict()


def ligand_druggability_summary(*, prefer_offline: bool = True) -> list[dict]:
    """Committed-affinity ligand ranking (as dicts), ranked by binding_probability."""
    eng = TargetingEngine(prefer_offline=prefer_offline)
    return [r.to_dict() for r in eng.ligand_druggability()]


__all__ = [
    "TargetingComponent",
    "TargetDruggability",
    "LigandDruggability",
    "TargetingEngine",
    "DEFAULT_WEIGHTS",
    "TARGETING_LABEL",
    "druggability_ranking",
    "target_druggability",
    "ligand_druggability_summary",
]
