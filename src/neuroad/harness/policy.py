"""
policy — the L3 policy LOADER (Stage-2 blueprint §4).

The `policy/` directory holds the engine's declarative domain knowledge as
dual-consumption docs (YAML front-matter / `meta:` for machines, Markdown /
structured YAML body for Claude). This module is the single reader that turns
those docs into three things deterministic code and Claude prompts need:

    policy.table(name)      -> dict    machine-readable data for deterministic code
    policy.thresholds(name) -> dict    named float thresholds (replace magic numbers)
    policy.brief(name)      -> str     Markdown prose composed into Claude prompts

THE FROZEN-CONSTANT GUARANTEE
-----------------------------
Every accessor has a HARDCODED FALLBACK to today's live constants:

    * calibration.CAL                 (retained bands, calibration ranges)
    * contract.VERDICT_BANDS / GAUNTLET / RESULT_CREDIT / PROMOTION_FLOOR
    * gauntlet thresholds             (_ANCHOR_CI_PASS/_WEAK, _ANCHOR_MIN_N,
                                       replication PASS/WEAK — transcribed as
                                       literals here so importing this loader
                                       never drags in scipy/sklearn)
    * bridge._MECHANISMS              (biomarker routing / mechanisms)

The policy docs are *transcriptions* of those constants, so a value loaded from
`policy/` is expected to EQUAL its fallback. If `policy/` is missing or a file is
malformed, the loader silently returns the fallback and the demo runs
byte-identically to the frozen path. No network, no exceptions escape.
"""
from __future__ import annotations

import re
from pathlib import Path

try:                                    # PyYAML is a normal dep; guard anyway.
    import yaml
except Exception:                       # pragma: no cover - yaml is installed
    yaml = None

from ..calibration import CAL
from ..contract import (
    GAUNTLET,
    PROMOTION_FLOOR,
    RESULT_CREDIT,
    VERDICT_BANDS,
    Verdict,
)

# Repo layout: <root>/src/neuroad/harness/policy.py -> parents[3] == <root>.
POLICY_DIR: Path = Path(__file__).resolve().parents[3] / "policy"

#: policy-slug -> filename in policy/.
POLICY_FILES: dict[str, str] = {
    "confound_priors": "confound_priors.yaml",
    "biomarker_routing": "biomarker_routing.yaml",
    "hypothesis_schema": "hypothesis_schema.yaml",
    "atn_framework": "atn_framework.md",
    "verdict_rubric": "verdict_rubric.md",
    "novelty_rubric": "novelty_rubric.md",
    "policy_layer_readme": "README.md",
}

# gauntlet.py module-private thresholds, transcribed as literals (mirrors
# gauntlet._ANCHOR_CI_PASS / _ANCHOR_CI_WEAK / _ANCHOR_MIN_N and the replication
# PASS/WEAK cutoffs in gauntlet.test_replication). Kept as literals so this
# loader stays import-light (no scipy/sklearn pulled via gauntlet).
_ANCHOR_CI_PASS = 0.12
_ANCHOR_CI_WEAK = 0.0
_ANCHOR_MIN_N = 20
_REPLICATION_PASS = 0.65
_REPLICATION_WEAK = 0.58

# Parsed-file cache keyed by (policy dir, name) so monkeypatching POLICY_DIR in
# tests transparently re-reads. Parse failures are never cached.
_RAW_CACHE: dict[tuple[str, str], object] = {}


def reload() -> None:
    """Drop the parsed-file cache (call after editing policy/ at runtime)."""
    _RAW_CACHE.clear()


# ---------------------------------------------------------------------------
# Small parsing helpers
# ---------------------------------------------------------------------------
def _num(x) -> float:
    """Coerce to float, rejecting bool/None/non-numeric (so a bad value in a
    loaded doc falls through to the hardcoded fallback rather than poisoning a
    threshold)."""
    if x is None or isinstance(x, bool):
        raise TypeError(f"not a numeric threshold: {x!r}")
    return float(x)


def _split_front_matter(text: str) -> tuple[dict, str]:
    """Return (front_matter_dict, markdown_body) for a `---`-fenced .md file.

    A file with no front matter yields ({}, whole_text)."""
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", text, re.S)
    if not m:
        return {}, text
    meta = yaml.safe_load(m.group(1)) if yaml is not None else {}
    return (meta or {}), m.group(2)


def _raw(name: str) -> object:
    """Read + parse a policy file. Returns the parsed data (dict for tables /
    front-matter). Raises on any missing-file / malformed / no-yaml condition —
    callers convert that into a fallback."""
    if yaml is None:
        raise RuntimeError("PyYAML unavailable")
    if name not in POLICY_FILES:
        raise KeyError(name)
    key = (str(POLICY_DIR), name)
    if key in _RAW_CACHE:
        return _RAW_CACHE[key]
    path = POLICY_DIR / POLICY_FILES[name]
    text = path.read_text(encoding="utf-8")          # raises if missing
    if POLICY_FILES[name].endswith(".md"):
        data, _body = _split_front_matter(text)
    else:
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"policy file {name} did not parse to a mapping")
    _RAW_CACHE[key] = data
    return data


def _raw_body(name: str) -> str:
    """Return the Markdown body of a .md brief (front matter stripped)."""
    if yaml is None:
        raise RuntimeError("PyYAML unavailable")
    fn = POLICY_FILES.get(name)
    if not fn:
        raise KeyError(name)
    text = (POLICY_DIR / fn).read_text(encoding="utf-8")
    if fn.endswith(".md"):
        _meta, body = _split_front_matter(text)
        return body
    return text                                      # .yaml has no md body


# ---------------------------------------------------------------------------
# Public accessor 1: table(name) -> dict (for deterministic code)
# ---------------------------------------------------------------------------
def table(name: str) -> dict:
    """Machine-readable policy data for `name`, keyed by its policy slug.

    Returns the parsed doc (YAML body, or front-matter for a .md brief). Falls
    back to a hardcoded transcription of today's live constants if the file is
    absent or malformed. Always returns a dict; never raises for a known slug.
    """
    if name not in POLICY_FILES:
        raise KeyError(f"unknown policy: {name!r}")
    try:
        data = _raw(name)
        if isinstance(data, dict) and data:
            return data
    except Exception:
        pass
    return _fallback_table(name)


# ---------------------------------------------------------------------------
# Public accessor 2: thresholds(name) -> dict[str, float]
# ---------------------------------------------------------------------------
#: The named threshold groups this loader exposes (documents the vocabulary).
THRESHOLD_GROUPS = (
    "retained", "anchor", "replication", "verdict", "result_credit",
    "gauntlet_weights",
)


def thresholds(name: str) -> dict:
    """Named float thresholds that replace magic numbers in deterministic code.

    Starts from the hardcoded fallback (guaranteed identical to today's
    constants) and overlays any well-formed numeric values read from `policy/`,
    so a partially-malformed doc degrades key-by-key rather than all-or-nothing.
    """
    fb = _fallback_thresholds(name)
    out = dict(fb)
    try:
        loaded = _extract_thresholds(name)
    except Exception:
        loaded = {}
    for k, v in (loaded or {}).items():
        if k not in fb:                              # only keys we know about
            continue
        try:
            out[k] = _num(v)
        except Exception:
            pass                                     # keep the fallback for this key
    return out


def _extract_thresholds(name: str) -> dict:
    """Pull the raw (un-coerced) threshold values for `name` out of the policy
    docs. May raise / return partial dicts; thresholds() sanitizes."""
    if name == "retained":
        rb = table("confound_priors").get("retained_bands", {})
        return {"survivor_retained": rb.get("survivor_retained"),
                "kill_retained": rb.get("kill_retained")}
    if name == "anchor":
        g = table("atn_framework").get("anchor_gate", {})
        return {"ci_pass": g.get("ci_lower_pass"),
                "ci_weak": g.get("ci_lower_weak"),
                "min_n": g.get("min_complete_case_n")}
    if name == "replication":
        return {}                                    # no policy source; fallback only
    if name == "verdict":
        t = table("verdict_rubric")
        d: dict = {}
        for row in t.get("verdict_bands", []) or []:
            key = str(row.get("key", "")).lower()
            if key:
                d[key] = row.get("min_score")
        d["promotion_floor"] = t.get("promotion_floor_min_score")
        return d
    if name == "result_credit":
        return dict(table("verdict_rubric").get("score", {}).get("result_credit", {}))
    if name == "gauntlet_weights":
        return dict(table("verdict_rubric").get("score", {}).get("dimension_weights", {}))
    raise KeyError(f"unknown threshold group: {name!r}")


# ---------------------------------------------------------------------------
# Public accessor 3: brief(name) -> Markdown str (for Claude system prompts)
# ---------------------------------------------------------------------------
def brief(name: str) -> str:
    """The Markdown brief for `name`, ready to compose into a Claude system
    prompt. Returns the doc body when available, else a hardcoded fallback that
    still carries the load-bearing thresholds/rules. Never raises for a known
    slug."""
    if name not in POLICY_FILES:
        raise KeyError(f"unknown policy: {name!r}")
    try:
        body = _raw_body(name)
        if body and body.strip():
            return body.strip()
    except Exception:
        pass
    return _fallback_brief(name)


# ===========================================================================
# HARDCODED FALLBACKS — transcriptions of today's live constants.
# ===========================================================================
def _promotion_floor_score() -> int:
    """Score band lower-bound of contract.PROMOTION_FLOOR (== 40 today)."""
    for lo, v in VERDICT_BANDS:
        if v == PROMOTION_FLOOR:
            return int(lo)
    return 40


def _credit_fallback() -> dict:
    return {k.value: float(v) for k, v in RESULT_CREDIT.items()}


def _fallback_thresholds(name: str) -> dict:
    if name == "retained":
        # gauntlet: _SURVIVOR_RETAINED = CAL["survivor_retained"][0];
        #           _KILL_RETAINED     = CAL["kill_retained"][1].
        return {"survivor_retained": float(CAL["survivor_retained"][0]),
                "kill_retained": float(CAL["kill_retained"][1])}
    if name == "anchor":
        return {"ci_pass": float(_ANCHOR_CI_PASS),
                "ci_weak": float(_ANCHOR_CI_WEAK),
                "min_n": float(_ANCHOR_MIN_N)}
    if name == "replication":
        return {"pass": float(_REPLICATION_PASS),
                "weak": float(_REPLICATION_WEAK)}
    if name == "verdict":
        d = {v.name.lower(): float(lo) for lo, v in VERDICT_BANDS}
        d["promotion_floor"] = float(_promotion_floor_score())
        return d
    if name == "result_credit":
        return _credit_fallback()
    if name == "gauntlet_weights":
        return {d.key: float(d.weight) for d in GAUNTLET}
    raise KeyError(f"unknown threshold group: {name!r}")


def _bridge_mechanisms() -> dict:
    """Transcribe bridge._MECHANISMS into the biomarker_routing table shape.
    Imported lazily so the loader doesn't pull the claude package at import."""
    try:
        from ..claude.bridge import _MECHANISMS
    except Exception:
        return {}
    out = {}
    for key, m in _MECHANISMS.items():
        row = {"label": m.get("label"), "markers": m.get("markers"),
               "cohort": m.get("cohort"), "n": m.get("n"),
               "expected_direction": m.get("direction"),
               "kill_criterion": m.get("kill")}
        if "fact" in m:
            row["fact"] = m["fact"]
        out[key] = row
    return out


def _fallback_table(name: str) -> dict:
    credit = _credit_fallback()
    if name == "confound_priors":
        return {
            "retained_bands": {
                "survivor_retained": float(CAL["survivor_retained"][0]),
                "kill_retained": float(CAL["kill_retained"][1]),
                "result_credit": credit,
            },
            # weights/star mirror contract.GAUNTLET (age_sex split into age+sex).
            "confounds": {
                "scanner_site": {"label": "Site / scanner leakage",
                                 "gauntlet_test": "test_site_scanner",
                                 "weight": 25, "star": True},
                "age": {"label": "Age adjustment (component of age_sex)",
                        "gauntlet_test": "test_age_sex", "weight": 15, "star": False},
                "sex": {"label": "Sex adjustment (component of age_sex)",
                        "gauntlet_test": "test_age_sex", "weight": 15, "star": False},
                "brain_age": {"label": "Brain-age control",
                              "gauntlet_test": "test_brain_age",
                              "weight": 25, "star": True},
            },
        }
    if name == "biomarker_routing":
        return {
            "routing": {
                "default_mechanism": "amyloid_cascade",
                "marker_to_mechanism": {"p_tau217": "amyloid_cascade",
                                        "gfap": "glial", "nfl": "vascular"},
            },
            "mechanisms": _bridge_mechanisms(),
            "calibration_anchors": {
                "conversion_auc_target": float(CAL["conversion_auc"][2]),
                "ptau217_r_target": float(CAL["ptau217_r"][2]),
                "gfap_r_target": float(CAL["gfap_r"][2]),
            },
            "gate": {"fires_only_if": "card.promoted == True"},
        }
    if name == "verdict_rubric":
        return {
            "verdict_bands": [{"min_score": int(lo), "verdict": v.value,
                               "key": v.name} for lo, v in VERDICT_BANDS],
            "promotion_floor": PROMOTION_FLOOR.name,
            "promotion_floor_min_score": _promotion_floor_score(),
            "score": {
                "range": [0, 100],
                "dimension_weights": {d.key: int(d.weight) for d in GAUNTLET},
                "result_credit": credit,
            },
        }
    if name == "atn_framework":
        return {
            "anchor_gate": {
                "primary_marker": "p_tau217",
                "secondary_marker": "gfap",
                "min_complete_case_n": int(_ANCHOR_MIN_N),
                "ci_lower_pass": float(_ANCHOR_CI_PASS),
                "ci_lower_weak": float(_ANCHOR_CI_WEAK),
            },
            "anchors": {
                "A": {"axis": "amyloid", "columns": ["amyloid"],
                      "anchor_eligible": False, "routes_to": "amyloid_cascade"},
                "T": {"axis": "tau", "columns": ["p_tau217"],
                      "anchor_eligible": True, "routes_to": "amyloid_cascade",
                      "fact_key": "ptau217"},
                "N": {"axis": "neurodegeneration", "columns": ["nfl"],
                      "anchor_eligible": False, "routes_to": "vascular"},
                "I": {"axis": "inflammation", "columns": ["gfap"],
                      "anchor_eligible": True, "routes_to": "glial",
                      "fact_key": "gfap"},
            },
        }
    if name == "hypothesis_schema":
        return {
            "claim": {"required_fields": ["claim_text", "target", "group_a",
                                          "group_b", "covariates"]},
            "label_targets": {
                "conversion": {"default_groups": ["MCI converters",
                                                  "MCI non-converters"]},
                "dx_binary": {"default_groups": ["AD", "CN"]},
                "site": {"default_groups": ["site A", "site B"]},
                "scanner": {"default_groups": ["scanner A", "scanner B"]},
            },
            "fallback_target": "conversion",
            "discovery_mode": {"values": ["novel_pattern", "named_contrast"]},
        }
    if name == "novelty_rubric":
        return {
            "novelty_class": {"values": ["known", "adjacent", "novel"]},
            "honesty_rung": {
                "ordered": True,
                "rungs": [
                    {"id": 1, "key": "raw_pattern", "label": "raw pattern"},
                    {"id": 2, "key": "stable_cluster", "label": "stable cluster"},
                    {"id": 3, "key": "confound_survivor",
                     "label": "confound-survivor"},
                    {"id": 4, "key": "severity_anchored",
                     "label": "severity-anchored candidate"},
                    {"id": 5, "key": "externally_replicated",
                     "label": "externally-replicated"},
                ],
            },
        }
    if name == "policy_layer_readme":
        return {"policy": "policy_layer_readme", "layer": "L3", "kind": "brief"}
    raise KeyError(f"unknown policy: {name!r}")


def _fallback_brief(name: str) -> str:
    """Compact, faithful hardcoded briefs carrying the load-bearing thresholds/
    rules. Used only when the policy/ doc is absent or malformed; the offline
    demo never consumes briefs, so this keeps behavior byte-identical."""
    sr = float(CAL["survivor_retained"][0])
    kr = float(CAL["kill_retained"][1])
    floor = _promotion_floor_score()
    briefs = {
        "verdict_rubric": (
            "# Verdict Rubric (fallback)\n\n"
            "Turn a 0-100 robustness score into one verdict (inclusive lower "
            "bounds, top-down): 85+ strong candidate; 70+ robust enough for "
            "follow-up; 40+ partially robust; else fragile. "
            f"Promotion floor = partially robust (score >= {floor}); only "
            "promoted claims reach the biology/Bridge step. Never upgrade the "
            "noun (no 'discovery'/'proof'); hedge below 70; state NA tests as a "
            "completeness caveat; a FAILED site/scanner star caps the language "
            "to 'likely an acquisition artifact'."
        ),
        "atn_framework": (
            "# AT(N)(+I) anchor brief (fallback)\n\n"
            "A structural (N) finding must be anchored to a MOLECULAR axis before "
            "promotion: T (plasma p-tau217) is the PRIMARY anchor, (+I) (GFAP) the "
            "SECONDARY; A (amyloid) enriches routing but is not the gate; NfL routes "
            "vascular/axonal. Gate on the 95% CI lower bound of Pearson r between the "
            "out-of-fold probe score and the marker, on complete cases only: "
            f"PASSED if ci_lower >= {_ANCHOR_CI_PASS}; WEAKENED if "
            f"{_ANCHOR_CI_WEAK} < ci_lower < {_ANCHOR_CI_PASS}; FAILED if "
            f"ci_lower <= {_ANCHOR_CI_WEAK}; NA below n={_ANCHOR_MIN_N} coverage "
            "(route to ADNI/EPAD). A modest correlation (p-tau217 r~0.3-0.55) is "
            "the correct expectation, not a failure."
        ),
        "novelty_rubric": (
            "# Novelty + honesty-ladder brief (fallback)\n\n"
            "Two independent stamps ride every card. novelty_class in "
            "{known, adjacent, novel}: how new the idea is. honesty_rung is the "
            "cumulative 5-rung ladder for how far it is defended: 1 raw pattern; "
            "2 stable cluster; 3 confound-survivor (survives the STAR tests); "
            f"4 severity-anchored candidate (clears the anchor gate and is promoted, "
            f"score >= {floor}); 5 externally-replicated (held-out AUC >= "
            f"{_REPLICATION_PASS}). Walk upward, stop at the highest rung whose "
            "evidence is present. Rung >= 4 is required before the word 'candidate'; "
            "a novel idea at a low rung is a hunch, never a discovery."
        ),
        "confound_priors": (
            "# Confound priors (fallback)\n\n"
            "Shared retained-fraction bands for the age/sex and brain-age controls "
            f"(fraction of naive effect surviving): PASSED if retained >= {sr}; "
            f"WEAKENED if {kr} <= retained < {sr}; FAILED if retained < {kr}. "
            "The site/scanner STAR test is a margin+CI rule instead: FAILED if "
            "margin <= 0 (scanner predicts as well as outcome); PASSED if margin > 0 "
            "AND the 95% CI excludes zero; WEAKENED otherwise. Scanner/site leakage "
            "in frozen embeddings is published prior art we measure, not claim."
        ),
        "biomarker_routing": (
            "# Biomarker routing (fallback)\n\n"
            "Only promoted survivors route. Pick the mechanism by the plasma marker "
            "with the largest standardized mean difference across the disease pole; "
            "amyloid prevalence adds to the p-tau217/tau pole. p_tau217 -> "
            "amyloid-cascade (tau-driven); gfap -> neuroinflammatory/glial; nfl -> "
            "vascular/axonal; default amyloid-cascade when nothing separates. Emit "
            "one mechanism, one falsifiable next experiment (named cohort, target N, "
            "expected direction, explicit kill criterion)."
        ),
        "hypothesis_schema": (
            "# Hypothesis parse contract (fallback)\n\n"
            "A free-text hunch becomes a structured Claim with required fields "
            "claim_text, target, group_a, group_b, covariates (default [age, sex]). "
            "target must be one of conversion | dx_binary | site | scanner; "
            "fall back to conversion on an invalid/absent target. Every promoted "
            "claim advances with ONE pre-registered kill criterion (metric, "
            "threshold, cohort, N, direction) fixed BEFORE the experiment is run."
        ),
        "policy_layer_readme": (
            "# L3 policy layer (fallback)\n\n"
            "Declarative domain knowledge read by BOTH deterministic code (YAML "
            "tables/thresholds) and Claude prompts (Markdown briefs). Every "
            "threshold here is a transcription of a live constant in src/neuroad; "
            "if a policy value disagrees with its mirrored constant, the code is "
            "authoritative and the policy file has a bug."
        ),
    }
    return briefs.get(name, f"(no brief available for policy '{name}')")
