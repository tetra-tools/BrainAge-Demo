"""Diagnosis cleaning and grouping rules of the Figure 13 forest-plot notebook.

The constants and the private helpers below are copied verbatim from
``multilabel_diagnostic_classification_v2-1.ipynb`` (cells 1 and 5). Do not edit them here
without editing the notebook: the inventory must describe the grouping actually used.

Rules, per diagnosis string:
* the string is split into tokens on ``,``, ``;`` and ``|``;
* excluded tokens (``*_EXCLUDE_TOKENS``) are ignored; if every token is excluded (or the
  value is missing), no diagnosis group is set;
* a group is set when one remaining token matches its regex exactly (case-insensitive);
* neurological only: ``AD`` is detected on the whole raw string (``detect_ad``), and
  ``Mutation_PD`` / ``Prodromal_PD`` by substring on the remaining tokens;
* ``CN`` (cognitively normal / healthy) is matched on all tokens and is not a diagnosis group
  (it is the reference category of the OLS models).
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------- copied from the notebook
NEURO_EXCLUDE_TOKENS = [
    "Undetermined", "NA", "Other", "Other Cognitive Disorder", "Unknown",
    "Alcohol-related", "Medical Illness-related", "CNS Neoplasm", "Epilepsy",
    "Substance-related", "Posterior Cortical Atrophy", "Essential Tremor"
]
PSYCH_EXCLUDE_TOKENS = [
    "Undetermined", "NA", "Other", "Unknown", "surgery"
]
NEURO_MUTATION_TOKENS = ["LRRK2", "GBA", "SNCA", "PRKN", "SWEDD"]
NEURO_PRODROMAL_TOKENS = ["Prodromal Synucleinopathy"]
NEURO_PATTERNS_EXACT = {
    "CN": r"^(?:Cognitively Normal)$",
    "Impaired_not_MCI": r"^(?:Impaired not MCI)$",
    "MCI": r"^(?:MCI|Primary Progressive Aphasia)$",
    "Vascular": r"^(?:Vascular Brain Injury|PostStroke|Vascular Dementia|Stroke)$",
    "TBI": r"^(?:Traumatic Brain Injury)$",
    "FTD": r"^(?:FrontoTemporal Dementia)$",
    "PSP": r"^(?:Progressive Supranuclear Palsy)$",
    "LBD": r"^(?:Dementia with Lewy Bodies)$",
    "CBS": r"^(?:CorticoBasal Syndrome)$",
    "NPH": r"^(?:Normal Pressure Hydrocephalus)$",
    "Parkinson": r"^(?:Parkinson's Disease|Parkinsons Disease)$",
    "SCC": r"^(?:Subjective Cognitive Complaint|Subjective Cognitive Concern)$",
}
PSYCH_PATTERNS_EXACT = {
    "CN": r"^(?:Healthy)$",
    "Bipolar_and_Dysthymia": r"^(?:Bipolar Disorder|Dysthymia)$",
    "Schizophrenia": r"^(?:Schizophrenia)$",
    "PTSD": r"^(?:Post-Traumatic Stress Disorder|PostTraumatic Stress Disorder)$",
    "MDD": r"^(?:Major Depressive Disorder)$",
    "Anxiety": r"^(?:Anxiety Disorder|Generalized Anxiety Disorder|Panic Disorder|GAD)$",
    "OCD": r"^(?:ObsessiveCompulsive Disorder|Obsessive Compulsive Disorder|OCD)$",
    "Pain": r"^(?:Pain|Osteoarthritis|Fibromyalgia|Primary Tremor|surgery)$",
    "Developmental": r"^(?:Attention Deficit Hyperactivity Disorder|Autistic Spectrum Disorders|Developmental Neuropsychiatric Disorder|Autism Spectrum Disorder|Autism|ASD|ADHD)$",
    "Addiction": r"^(?:Alcohol Use Disorder|Substance Use Disorder|Cannabis)$",
}

_neuro_exclude_lower = {t.lower() for t in NEURO_EXCLUDE_TOKENS}
_psych_exclude_lower = {t.lower() for t in PSYCH_EXCLUDE_TOKENS}


def _norm(s: Optional[str]) -> str:
    if s is None:
        return ''
    s = str(s)
    s = re.sub(r'[;|]', ',', s)
    s = re.sub(r'\s*,\s*', ',', s.strip())
    return s


def _tokens(s: str) -> List[str]:
    s = _norm(s)
    return [t for t in s.split(',') if t]


def _match_token(token: str, pattern: str) -> bool:
    return bool(re.search(pattern, token, flags=re.I))


def detect_ad(dx_str: str) -> bool:
    if not dx_str or pd.isna(dx_str):
        return False
    low = str(dx_str).lower()
    if 'alzheimer' in low:
        return True
    if 'dementia' in low:
        non_ad_terms = [
            'vascular', 'frontotemporal', 'lewy', 'parkinson',
            'corticobasal', 'progressive supranuclear', 'hydrocephalus'
        ]
        if not any(t in low for t in non_ad_terms):
            return True
    return False


def _filtered_tokens(dx: Optional[str], domain: str) -> List[str]:
    if not dx or pd.isna(dx):
        return []
    toks = _tokens(dx)
    if domain == 'neuro':
        return [t for t in toks if t.lower() not in _neuro_exclude_lower]
    if domain == 'psych':
        return [t for t in toks if t.lower() not in _psych_exclude_lower]
    return toks


def _all_tokens_excluded(dx: Optional[str], domain: str) -> bool:
    if not dx or pd.isna(dx):
        return True
    toks = _tokens(dx)
    if not toks:
        return True
    if domain == 'neuro':
        return all(t.lower() in _neuro_exclude_lower for t in toks)
    if domain == 'psych':
        return all(t.lower() in _psych_exclude_lower for t in toks)
    return False


def detect_mutation_pd(dx: Optional[str]) -> bool:
    toks = _filtered_tokens(dx, 'neuro')
    if not toks:
        return False
    return any(any(mut.lower() in t.lower() for mut in NEURO_MUTATION_TOKENS) for t in toks)


def detect_prodromal_pd(dx: Optional[str]) -> bool:
    toks = _filtered_tokens(dx, 'neuro')
    if not toks:
        return False
    return any(any(prod.lower() in t.lower() for prod in NEURO_PRODROMAL_TOKENS) for t in toks)
# ---------------------------------------------------------------- end of copied code

DOMAINS = {"neuro": "neurological", "psych": "psychiatric"}


def _patterns(domain: str) -> dict:
    return NEURO_PATTERNS_EXACT if domain == "neuro" else PSYCH_PATTERNS_EXACT


def string_groups(dx: Optional[str], domain: str) -> Tuple[List[str], bool]:
    """Diagnosis groups set for one raw string, and whether it is flagged CN.

    Same result as ``build_multilabel_frame`` for one cell: groups are the ``*_flag_*``
    columns that are True (CN excluded), after the "all tokens excluded" reset.
    """
    groups: List[str] = []
    if not _all_tokens_excluded(dx, domain):
        kept = _filtered_tokens(dx, domain)
        for cond, pat in _patterns(domain).items():
            if cond != "CN" and any(_match_token(tok, pat) for tok in kept):
                groups.append(cond)
        if domain == "neuro":
            if detect_ad(dx):
                groups.append("AD")
            if detect_mutation_pd(dx):
                groups.append("Mutation_PD")
            if detect_prodromal_pd(dx):
                groups.append("Prodromal_PD")
    # notebook: `any(_match_token(tok, CN) for tok in _tokens(dx)) if dx else False`
    is_cn = any(_match_token(tok, _patterns(domain)["CN"]) for tok in _tokens(dx)) if dx else False
    return groups, is_cn


def token_rule(token: str, domain: str) -> Tuple[str, str]:
    """(rule, groups) applied to a single token, for the token table of the inventory."""
    excluded = _neuro_exclude_lower if domain == "neuro" else _psych_exclude_lower
    if token.lower() in excluded:
        return "excluded token (ignored)", ""
    groups = [cond for cond, pat in _patterns(domain).items() if _match_token(token, pat)]
    if domain == "neuro":
        if detect_ad(token):
            groups.append("AD (detect_ad)")
        if any(m.lower() in token.lower() for m in NEURO_MUTATION_TOKENS):
            groups.append("Mutation_PD")
        if any(p.lower() in token.lower() for p in NEURO_PRODROMAL_TOKENS):
            groups.append("Prodromal_PD")
    if not groups:
        return "NOT MAPPED (ignored)", ""
    return ("matched by several rules" if len(groups) > 1 else "matched"), " + ".join(groups)


def split_tokens(dx: Optional[str]) -> List[str]:
    """Tokens of a raw value, as the notebook splits them (empty for missing values)."""
    if dx is None or (not isinstance(dx, str) and pd.isna(dx)):
        return []
    return _tokens(dx)
