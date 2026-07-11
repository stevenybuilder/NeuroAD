"""
discovery_loop — active-learning experiment selection + wet-lab feedback.

The referee is one-shot: hypothesis -> verdict. This closes the loop the plan
asks for (§4 "user-in-the-loop iteration or active learning"): turn a promoted
survivor into a SEQUENCE of wet-lab experiments, chosen to learn the most per
experiment, and fold each result back in.

Model (Bayesian, deliberately simple and legible):
  * Each candidate target carries a Beta(alpha, beta) belief that it is a real,
    perturbable driver of the routed mechanism. P(hit) ~ Beta(alpha, beta).
  * The PRIOR is a composite of the engine's own evidence — the imaging-routed
    mechanism fit, PI4AD priority (0-10), structural confidence (AlphaFold
    pLDDT), and Open Targets association (0-1) when available — converted to
    pseudo-counts alpha0/beta0. So the imaging + molecule evidence seeds belief;
    it does not decide anything.
  * A wet-lab RESULT (hit / miss, optionally weighted by effect size) is a
    Bernoulli update: alpha += w on a hit, beta += w on a miss. Beliefs sharpen
    with evidence.

Acquisition (which experiment next):
  * "ucb"          mean + kappa*std       — exploit the best lead, explore the uncertain
  * "uncertainty"  std (posterior var)    — pure information gain, most-uncertain first
  * "greedy"       mean                   — follow the current best only
  optionally divided by experiment COST (cheaper model systems win ties).

State persists to reports/discovery_loop_state.json so the loop survives across
sessions — the feedback is longitudinal, not within one call. No network, no
Claude: the orchestrator (harness/agent.py) can DRIVE this, but the math is
deterministic and reproducible.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

_log = logging.getLogger("neuroad.harness.discovery_loop")

_STATE_PATH = Path(__file__).resolve().parents[3] / "reports" / "discovery_loop_state.json"

#: Pseudo-count strength of the composite prior. Small so a few real experiments
#: move belief materially (weak prior = data speaks quickly).
_PRIOR_STRENGTH = 4.0

#: Relative cost of a wet-lab model system (for info-gain-per-cost selection).
_MODEL_COST = {
    "iPSC 2D neuron": 1.0,
    "iPSC-microglia co-culture": 1.5,
    "3D cerebral organoid": 3.0,
    "BBB-on-chip": 2.5,
}

#: Mechanism -> (model system, perturbation, readout) for the experiment spec.
_MECHANISM_EXPERIMENT = {
    "amyloid_cascade": (
        "3D cerebral organoid", "CRISPRi knockdown",
        "Aβ42/40 ratio + p-tau217 (MSD) at 6 weeks"),
    "glial": (
        "iPSC-microglia co-culture", "CRISPRi knockdown",
        "secreted GFAP + IL-6/TNF-α + synaptophysin puncta at 6 weeks"),
    "vascular": (
        "BBB-on-chip", "CRISPRi knockdown",
        "trans-endothelial resistance + NfL leakage at 6 weeks"),
}


@dataclass
class TargetBelief:
    """A Beta belief that a target is a real, perturbable driver."""
    gene: str
    mechanism: str
    alpha: float
    beta: float
    prior_score: float                 # the composite prior (0-1) that seeded it
    evidence: dict = field(default_factory=dict)   # per-source prior contributions
    n_experiments: int = 0
    hits: float = 0.0

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta
        s = a + b
        return (a * b) / (s * s * (s + 1.0))

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["posterior_mean"] = round(self.mean, 4)
        d["posterior_std"] = round(self.std, 4)
        return d


@dataclass
class ExperimentSpec:
    """A concrete, falsifiable wet-lab experiment proposed for a target."""
    experiment_id: str
    gene: str
    mechanism: str
    model_system: str
    perturbation: str
    readout: str
    rationale: str
    acquisition_score: float
    kill_criterion: str

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Composite prior — fuse the engine's own evidence into a 0-1 promise score.
# ---------------------------------------------------------------------------

def _composite_prior(gene: str, mechanism: str, *,
                     use_opentargets: bool = True) -> tuple[float, dict]:
    """Fuse PI4AD priority, structural confidence, (Open Targets) into a prior.

    Returns (score in [0,1], per-source evidence dict). Each source is optional;
    the score is the mean of whatever is available, so the loop still runs if a
    source is offline/unbuilt."""
    ev: dict = {}

    # PI4AD priority (0-10 -> 0-1)
    try:
        from ..integrations.pi4ad import PI4AD
        gp = PI4AD(prefer_offline=True).priority(gene)
        if gp is not None and gp.priority_score is not None:
            ev["pi4ad"] = round(float(gp.priority_score) / 10.0, 4)
    except Exception as exc:  # noqa: BLE001
        _log.debug("pi4ad prior unavailable for %s: %r", gene, exc)

    # AlphaFold structural confidence (pLDDT/100) — a "well-characterized target"
    # proxy, offline snapshot is fine.
    try:
        from ..integrations.alphafold import AlphaFoldClient
        s = AlphaFoldClient(prefer_offline=True).fetch_structure(gene)
        if s.mean_plddt is not None:
            ev["structure"] = round(float(s.mean_plddt) / 100.0, 4)
    except Exception as exc:  # noqa: BLE001
        _log.debug("alphafold prior unavailable for %s: %r", gene, exc)

    # Open Targets AD association (0-1) — used if the adapter is present.
    if use_opentargets:
        try:
            from ..integrations.opentargets import OpenTargetsClient
            ta = OpenTargetsClient(prefer_offline=True).target_association(gene)
            if ta is not None and ta.association_score is not None:
                ev["opentargets"] = round(float(ta.association_score), 4)
        except Exception as exc:  # noqa: BLE001
            _log.debug("opentargets prior unavailable for %s: %r", gene, exc)

    vals = [v for v in ev.values() if v is not None]
    score = float(sum(vals) / len(vals)) if vals else 0.5   # uninformative default
    return max(0.02, min(0.98, score)), ev


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

class DiscoveryLoop:
    """Stateful active-learning loop over candidate targets for a mechanism."""

    def __init__(self, *, state_path: Optional[Path] = None):
        self.state_path = Path(state_path) if state_path else _STATE_PATH
        self.beliefs: dict[str, TargetBelief] = {}
        self.experiments: list[dict] = []   # ledger: proposed + completed
        self._counter = 0

    # -- seeding ----------------------------------------------------------
    def seed_mechanism(self, mechanism: str, genes: Optional[list[str]] = None,
                       *, use_opentargets: bool = True) -> list[TargetBelief]:
        """Seed Beta priors for a mechanism's candidate genes from the composite
        evidence. Idempotent: re-seeding a gene keeps any accumulated results."""
        from . import translation
        if genes is None:
            genes = translation.MECHANISM_GENES.get(
                mechanism, translation.MECHANISM_GENES["amyloid_cascade"])
        seeded = []
        for gene in genes:
            if gene in self.beliefs:
                seeded.append(self.beliefs[gene])
                continue
            score, ev = _composite_prior(gene, mechanism,
                                         use_opentargets=use_opentargets)
            a0 = max(0.5, score * _PRIOR_STRENGTH)
            b0 = max(0.5, (1.0 - score) * _PRIOR_STRENGTH)
            tb = TargetBelief(gene=gene, mechanism=mechanism, alpha=a0, beta=b0,
                              prior_score=round(score, 4), evidence=ev)
            self.beliefs[gene] = tb
            seeded.append(tb)
        return seeded

    # -- acquisition ------------------------------------------------------
    def _acquisition(self, tb: TargetBelief, strategy: str, kappa: float,
                     per_cost: bool) -> float:
        if strategy == "uncertainty":
            a = tb.std
        elif strategy == "greedy":
            a = tb.mean
        else:  # ucb (default)
            a = tb.mean + kappa * tb.std
        if per_cost:
            model = _MECHANISM_EXPERIMENT.get(tb.mechanism, (None,))[0]
            a = a / _MODEL_COST.get(model, 1.0)
        return a

    def propose_next_experiment(self, *, strategy: str = "ucb", kappa: float = 1.0,
                                per_cost: bool = False) -> Optional[ExperimentSpec]:
        """Select the most valuable next experiment under the acquisition rule.

        Targets already tested twice+ are de-prioritized implicitly (their variance
        has shrunk). Returns None if nothing is seeded."""
        if not self.beliefs:
            return None
        best = max(self.beliefs.values(),
                   key=lambda tb: self._acquisition(tb, strategy, kappa, per_cost))
        self._counter += 1
        eid = f"exp-{self._counter:03d}-{best.gene}"
        model, perturb, readout = _MECHANISM_EXPERIMENT.get(
            best.mechanism, ("3D cerebral organoid", "CRISPRi knockdown",
                             "pathology markers at 6 weeks"))
        acq = round(self._acquisition(best, strategy, kappa, per_cost), 4)
        spec = ExperimentSpec(
            experiment_id=eid, gene=best.gene, mechanism=best.mechanism,
            model_system=model, perturbation=perturb, readout=readout,
            acquisition_score=acq,
            rationale=(f"{strategy} pick: posterior mean {best.mean:.2f} "
                       f"(±{best.std:.2f}) from prior {best.prior_score:.2f} "
                       f"[{', '.join(f'{k}={v}' for k, v in best.evidence.items()) or 'no prior evidence'}] "
                       f"after {best.n_experiments} experiment(s)."),
            kill_criterion=(f"If {perturb} of {best.gene} does not move the readout "
                            "beyond vehicle ±2SD, downweight this target."),
        )
        self.experiments.append({"status": "proposed", **spec.to_dict()})
        return spec

    # -- feedback ---------------------------------------------------------
    def record_result(self, experiment_id: str, *, hit: bool,
                      effect_size: Optional[float] = None,
                      notes: str = "") -> TargetBelief:
        """Fold a wet-lab result back in as a Bernoulli update on the target's Beta.

        ``effect_size`` (|standardized effect|) optionally weights the update so a
        strong hit/miss moves belief more than a marginal one (weight in [0.5, 2]).
        """
        rec = next((e for e in self.experiments
                    if e.get("experiment_id") == experiment_id), None)
        gene = rec["gene"] if rec else experiment_id.rsplit("-", 1)[-1]
        tb = self.beliefs.get(gene)
        if tb is None:
            raise KeyError(f"no seeded target for experiment {experiment_id!r}")
        w = 1.0
        if effect_size is not None:
            w = max(0.5, min(2.0, abs(float(effect_size))))
        if hit:
            tb.alpha += w
            tb.hits += w
        else:
            tb.beta += w
        tb.n_experiments += 1
        if rec is not None:
            rec.update({"status": "completed", "hit": bool(hit),
                        "effect_size": effect_size, "notes": notes})
        else:
            self.experiments.append({
                "status": "completed", "experiment_id": experiment_id,
                "gene": gene, "hit": bool(hit), "effect_size": effect_size,
                "notes": notes})
        return tb

    # -- views ------------------------------------------------------------
    def ranking(self) -> list[dict]:
        """Current posterior ranking of targets (best-supported first)."""
        return [tb.to_dict() for tb in sorted(
            self.beliefs.values(), key=lambda tb: tb.mean, reverse=True)]

    def summary(self) -> dict:
        completed = [e for e in self.experiments if e.get("status") == "completed"]
        return {
            "n_targets": len(self.beliefs),
            "n_experiments_proposed": len(self.experiments),
            "n_experiments_completed": len(completed),
            "ranking": self.ranking(),
            "top_target": self.ranking()[0]["gene"] if self.beliefs else None,
        }

    # -- persistence ------------------------------------------------------
    def save(self) -> Path:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "beliefs": {g: tb.to_dict() for g, tb in self.beliefs.items()},
            "experiments": self.experiments,
            "counter": self._counter,
        }
        self.state_path.write_text(json.dumps(payload, indent=2, default=str))
        return self.state_path

    @classmethod
    def load(cls, state_path: Optional[Path] = None) -> "DiscoveryLoop":
        loop = cls(state_path=state_path)
        p = loop.state_path
        if p.exists():
            try:
                data = json.loads(p.read_text())
                for g, d in (data.get("beliefs") or {}).items():
                    loop.beliefs[g] = TargetBelief(
                        gene=d["gene"], mechanism=d["mechanism"],
                        alpha=float(d["alpha"]), beta=float(d["beta"]),
                        prior_score=float(d.get("prior_score", 0.5)),
                        evidence=dict(d.get("evidence") or {}),
                        n_experiments=int(d.get("n_experiments", 0)),
                        hits=float(d.get("hits", 0.0)))
                loop.experiments = list(data.get("experiments") or [])
                loop._counter = int(data.get("counter", 0))
            except Exception as exc:  # noqa: BLE001
                _log.warning("could not load loop state: %r", exc)
        return loop
