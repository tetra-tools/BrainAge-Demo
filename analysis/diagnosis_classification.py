"""Diagnosis classification of the thesis (proposed classification, replacing the notebook groups).

The correspondence table is ``analysis/diagnosis_terms.csv`` (one row per term, case-insensitive),
read by this module and by ``demographics_tables.R``. Each term has a type:

  CN                 cognitively normal / healthy
  GROUP              named group (AD, MCI, Vascular, Depression, Sleep, Movement_disorder ...)
  OTHER_DISEASE      Other_neuro_disease / Other_psych_disease
  SYMPTOM            Other_neuro_symptom / Other_psych_symptom, only when the cell has no disease
                     of the same domain (R5)
  MOVEMENT_SYNDROME  Movement_disorder unless a disease that explains it is in the cell (R8)
  PPA                MCI when no neurological disease is in the cell (R7)
  DEMENTIA           the cause written in the cell, else Dementia_unspecified (R6)
  OTHER_COGNITIVE    Other_neuro_disease when no other neurological disease is in the cell
  ATTRIBUTION        attributed cause (alcohol, medication ...): Other_neuro_disease when nothing
                     else is in the cell (R4), otherwise a descriptor
  DESCRIPTOR         no group (subtype, cognitive domain, variant, certainty, treatment ...)
  MISSING            no information

A term is classified by its meaning, not by the column it is written in (R2): a depression written
in the neurological column counts in the psychiatric domain. Cells are split on , ; | (R1), as in
the notebook. Terms absent from the table are reported as unmapped and set no group.

Groups of fewer than 30 scans in a split are merged into Other_neuro_disease / Other_psych_disease
for models (R10); ``rare_groups`` lists them and ``merge_rare`` applies the merge. Counts here are
never merged. The reference category is the healthy scans (healthy == 1, R11); patients without any
group are Unclassified_patient. ``single_label`` gives one label per scan for analyses that need one
(a scan in two groups or more is Mixed).

A verbatim copy of this file and of diagnosis_terms.csv is in the training repository
(RS-JYC-HK-T1wMNIBrainAgeRegression, camera_ready/common/) for the autoencoder analyses: edit both.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

import pandas as pd

TERMS_FILE = Path(__file__).resolve().parent / "diagnosis_terms.csv"
NEURO, PSYCH = "neurological", "psychiatric"
COLUMN_DOMAIN = {"neuro": NEURO, "psych": PSYCH}
TYPES = {"CN", "GROUP", "OTHER_DISEASE", "SYMPTOM", "MOVEMENT_SYNDROME", "PPA", "DEMENTIA",
         "OTHER_COGNITIVE", "ATTRIBUTION", "DESCRIPTOR", "MISSING"}
CAUSES_OF_DEMENTIA = {"AD", "Vascular", "FTD", "Atypical_parkinsonism", "Parkinson", "NPH", "TBI"}
EXPLAINS_PARKINSONISM = {"Parkinson", "Mutation_PD", "Prodromal_PD", "Atypical_parkinsonism"}
OTHER_DISEASE = {NEURO: "Other_neuro_disease", PSYCH: "Other_psych_disease"}
ALWAYS_KEPT = {"Other_neuro_disease", "Other_psych_disease", "Other_neuro_symptom", "Other_psych_symptom"}
UNCLASSIFIED = "Unclassified_patient"
REFERENCE = "CN"        # healthy == 1 scans without any group: the reference category (R11)
MIXED = "Mixed"         # single-label analyses only: a scan in two groups or more
# when a term of each type sets its group (the column "condition" of the correspondence table)
TYPE_CONDITION = {
    "CN": "no group (cognitively normal / healthy)",
    "GROUP": "always",
    "OTHER_DISEASE": "always",
    "SYMPTOM": "only if no disease of the same domain is in the cell",
    "MOVEMENT_SYNDROME": "unless Parkinson, Mutation_PD, Prodromal_PD or Atypical_parkinsonism is in the cell",
    "PPA": "MCI only if no neurological disease is in the cell",
    "DEMENTIA": "the cause written in the cell; with only an attributed cause or Other Cognitive Disorder: "
                "Other_neuro_disease; with no cause: Dementia_unspecified",
    "OTHER_COGNITIVE": "only if no other neurological disease is in the cell",
    "ATTRIBUTION": "only if no other neurological disease is in the cell (otherwise a descriptor)",
    "DESCRIPTOR": "never",
    "MISSING": "never",
    "UNMAPPED": "never: term not in analysis/diagnosis_terms.csv",
}
MIN_SCANS = 30


def load_terms(path: Path = TERMS_FILE) -> pd.DataFrame:
    terms = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    terms["key"] = terms["term"].str.strip().str.lower()
    bad = sorted(set(terms["type"]) - TYPES)
    if bad:
        raise ValueError(f"{path.name}: unknown type(s) {bad}")
    dup = terms["key"][terms["key"].duplicated()].tolist()
    if dup:
        raise ValueError(f"{path.name}: terms listed twice: {dup}")
    return terms


_TERMS = load_terms()
_LOOKUP = {r.key: (r.type, r.group, r.domain) for r in _TERMS.itertuples()}


def split_terms(dx: Optional[str]) -> List[str]:
    """Terms of one cell: split on , ; | and stripped (as the notebook); empty for a missing value."""
    if dx is None or (not isinstance(dx, str) and pd.isna(dx)):
        return []
    s = re.sub(r"[;|]", ",", str(dx))
    return [t.strip() for t in s.split(",") if t.strip()]


def term_info(term: str) -> Tuple[str, str, str]:
    """(type, group, domain) of a term; type 'UNMAPPED' if the table does not list it."""
    return _LOOKUP.get(term.strip().lower(), ("UNMAPPED", "", ""))


def classify_cell(dx: Optional[str]) -> Tuple[Dict[str, Set[str]], bool, List[str]]:
    """Groups of one diagnosis cell by domain, whether it says CN, and its unmapped terms."""
    flags: Dict[str, Set[str]] = {NEURO: set(), PSYCH: set()}
    symptoms: Dict[str, Set[str]] = {NEURO: set(), PSYCH: set()}
    seen = set()
    cn = False
    unmapped = []
    for term in split_terms(dx):
        typ, group, domain = term_info(term)
        seen.add(typ)
        if typ == "UNMAPPED":
            unmapped.append(term)
        elif typ == "CN":
            cn = True
        elif typ in ("GROUP", "OTHER_DISEASE"):
            flags[domain].add(group)
        elif typ == "SYMPTOM":
            symptoms[domain].add(group)
    neuro = flags[NEURO]
    if "DEMENTIA" in seen:                                             # R6
        if not neuro & CAUSES_OF_DEMENTIA:
            neuro.add(OTHER_DISEASE[NEURO] if seen & {"ATTRIBUTION", "OTHER_COGNITIVE"} else "Dementia_unspecified")
    if seen & {"OTHER_COGNITIVE", "ATTRIBUTION"} and not neuro:        # R4, Other Cognitive Disorder
        neuro.add(OTHER_DISEASE[NEURO])
    if "MOVEMENT_SYNDROME" in seen and not neuro & EXPLAINS_PARKINSONISM:   # R8
        neuro.add("Movement_disorder")
    if "PPA" in seen and not neuro:                                    # R7
        neuro.add("MCI")
    for domain in (NEURO, PSYCH):                                      # R5
        if not flags[domain]:
            flags[domain] |= symptoms[domain]
    return flags, cn, unmapped


def scan_groups(neuro_dx: Optional[str], psych_dx: Optional[str]) -> FrozenSet[str]:
    """Groups of one scan: the union of what both diagnosis cells say."""
    groups: Set[str] = set()
    for dx in (neuro_dx, psych_dx):
        flags, _, _ = classify_cell(dx)
        groups |= flags[NEURO] | flags[PSYCH]
    return frozenset(groups)


def group_domain(group: str) -> str:
    """Domain of a group, from the table."""
    if group == UNCLASSIFIED:
        return "both"
    d = _TERMS.loc[_TERMS["group"] == group, "domain"]
    d = d[d != ""]
    return d.iloc[0] if len(d) else ""


def all_groups() -> List[str]:
    """Every group the table can produce, neurological first, in table order."""
    g = _TERMS[(_TERMS["group"] != "") & (_TERMS["type"] != "CN")]
    order = list(dict.fromkeys(g.loc[g["domain"] == NEURO, "group"])) + \
        list(dict.fromkeys(g.loc[g["domain"] == PSYCH, "group"]))
    return order


def rare_groups(counts: Dict[str, int], min_scans: int = MIN_SCANS) -> Dict[str, str]:
    """Named groups with fewer than ``min_scans`` scans (and at least one) -> the Other group they
    are merged into for models (R10). Other_* groups are always kept."""
    return {g: OTHER_DISEASE[group_domain(g)] for g, n in counts.items()
            if 0 < n < min_scans and g not in ALWAYS_KEPT and group_domain(g) in OTHER_DISEASE}


def group_counts(groups: Iterable[FrozenSet[str]]) -> Dict[str, int]:
    """Scans per group, from the group sets of the scans (a scan counts once in each of its groups)."""
    counts: Dict[str, int] = {}
    for scan in groups:
        for g in scan:
            counts[g] = counts.get(g, 0) + 1
    return counts


def merge_rare(groups: FrozenSet[str], mapping: Dict[str, str]) -> FrozenSet[str]:
    """Groups of one scan after the R10 merge (``mapping`` as returned by ``rare_groups``)."""
    return frozenset(mapping.get(g, g) for g in groups)


def single_label(groups: FrozenSet[str], healthy: bool) -> str:
    """One label per scan: its group, Mixed if it has several, CN for a healthy scan without group,
    Unclassified_patient for a patient without group."""
    if len(groups) == 1:
        return next(iter(groups))
    if groups:
        return MIXED
    return REFERENCE if healthy else UNCLASSIFIED


def group_flags(groups: Iterable[FrozenSet[str]], names: Optional[List[str]] = None) -> pd.DataFrame:
    """One boolean column ``group_<name>`` per group (every group of the table by default), for
    multi-label analyses where each group is compared with the rest."""
    groups = list(groups)
    names = all_groups() if names is None else list(names)
    return pd.DataFrame({f"group_{g}": [g in scan for scan in groups] for g in names})
